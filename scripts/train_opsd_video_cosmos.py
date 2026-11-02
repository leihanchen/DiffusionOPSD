#!/usr/bin/env python3
"""OPSD post-training of Cosmos3-Nano using native UniPC trajectories."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

from diffusionopsd.stat_tracking import PerPromptStatTracker
from diffusionopsd.video.cosmos_quality import IsolatedVideoJudge
from diffusionopsd.video.cosmos_rollout import CosmosRollout
from diffusionopsd.video.cosmos_training import (
    PromptCursor, fit_queries, load_training_state, save_training_state, validate_cuda_devices,
)
from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
from diffusionopsd.video.gates import effective_weight
from diffusionopsd.video.geo_reward import GeoReward
from diffusionopsd.video.quality_judge import gap_to_probability, quality_gap
from diffusionopsd.video.training_eval import evaluation_due, save_video
from diffusionopsd.video.training_logging import VideoTrainingLogger, evaluation_metrics
from diffusionopsd.video.wan_policy import ema_adapter_


def emit(event, **values):
    print(json.dumps({"event": event, "time": time.time(), **values}), flush=True)


def read_config(path):
    spec = importlib.util.spec_from_file_location("cosmos_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def structured_prompt(prompt):
    """Keep source captions deterministic; no external upsampling service."""
    if prompt.lstrip().startswith("{"):
        json.loads(prompt)
        return prompt
    return json.dumps({"description": prompt})


def validate_config(cfg):
    if cfg.num_frames < 5 or (cfg.num_frames - 1) % 4 or cfg.height % 32 or cfg.width % 32:
        raise ValueError("Cosmos video requires 4k+1 frames >=5 and dimensions divisible by 32")
    if min(cfg.height, cfg.width, cfg.updates, cfg.clips_per_prompt, cfg.calibration_prompts) <= 0:
        raise ValueError("Geometry and training counts must be positive")
    if cfg.reward_frames < 2 or cfg.reward_frames > cfg.num_frames:
        raise ValueError("Reward needs 2..num_frames frames")
    if cfg.checkpoint_every < 1 or cfg.eval_every < 1 or cfg.eval_count < 1:
        raise ValueError("Checkpoint/evaluation intervals and counts must be positive")
    if not 0 < cfg.query_sigma < 1 or cfg.rho <= 0 or cfg.n_ascent < 1:
        raise ValueError("Invalid OPSD query or target settings")
    if not 0 <= cfg.ema_decay < 1 or cfg.learning_rate <= 0 or cfg.lora_rank < 1:
        raise ValueError("Invalid optimizer/EMA settings")


def preflight(cfg):
    from diffusers import Cosmos3OmniPipeline

    model = Path(os.path.expandvars(cfg.model))
    revision_path = model / "opsd_revision.txt"
    if not revision_path.is_file() or revision_path.read_text().strip() != cfg.model_revision:
        raise ValueError("Model revision is unverified; run scripts/setup_cosmos_nano.sh")
    for item in ("model_index.json", "transformer/config.json", "vae/config.json", "scheduler/scheduler_config.json",
                 "transformer/diffusion_pytorch_model.safetensors.index.json", "text_tokenizer/tokenizer.json"):
        if not (model / item).is_file():
            raise FileNotFoundError(model / item)
    index = json.loads((model / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        if not (model / "transformer" / shard).is_file():
            raise FileNotFoundError(model / "transformer" / shard)
    if not (model / "vae/diffusion_pytorch_model.safetensors").is_file():
        raise FileNotFoundError(model / "vae/diffusion_pytorch_model.safetensors")
    root = Path(os.environ["VIDEO_REWARD_CKPT_PATH"])
    for item in ("depth-anything-3-large-v1.1", "WAFT", "waft_tar_c_t.pth", "dinov2-base", "VideoReward", "VideoAlign"):
        if not (root / item).exists():
            raise FileNotFoundError(root / item)
    scheduler = json.loads((model / "scheduler/scheduler_config.json").read_text())
    if scheduler["prediction_type"] != "flow_prediction":
        raise ValueError("Expected flow_prediction checkpoint")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "peft")}
    emit("preflight", versions=versions, model=str(model), pipeline=Cosmos3OmniPipeline.__name__)
    return model, versions


def cpu_adapters(policy):
    return {name: {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(policy, adapter_name=name).items()}
            for name in ("default", "old")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/cosmos_nano.py")
    parser.add_argument("--run-dir", default="logs/cosmos-nano")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-updates-this-job", type=int, default=0)
    args = parser.parse_args()
    if args.max_updates_this_job < 0:
        parser.error("max-updates-this-job must be nonnegative")
    cfg = read_config(args.config)
    validate_config(cfg)
    model_path, versions = preflight(cfg)
    if args.preflight:
        return
    validate_cuda_devices(
        [cfg.policy_device, cfg.vae_device, cfg.reward_device, cfg.judge_device],
        visible_count=torch.cuda.device_count(),
    )
    from diffusers import Cosmos3OmniPipeline, UniPCMultistepScheduler

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "training_state.pt"
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError(f"Use --resume for existing checkpoint {checkpoint_path}")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    train_prompts = [p for p in Path(cfg.train_prompts).read_text().splitlines() if p.strip()]
    eval_prompts = [p for p in Path(cfg.eval_prompts).read_text().splitlines() if p.strip()][:cfg.eval_count]
    if not train_prompts or not eval_prompts:
        raise ValueError("Training and held-out prompt files must not be empty")
    signature = cfg.to_dict()
    signature["train_prompt_sha256"] = hashlib.sha256(Path(cfg.train_prompts).read_bytes()).hexdigest()
    signature["eval_prompt_sha256"] = hashlib.sha256(Path(cfg.eval_prompts).read_bytes()).hexdigest()
    (run_dir / "manifest.json").write_text(json.dumps({"config": signature, "versions": versions}, indent=2))
    resume_update, logging_group = 0, None
    if args.resume:
        metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        resume_update = metadata["update"]
        logging_group = metadata["extra"].get("logging_experiment_group")
        del metadata
    with VideoTrainingLogger(
        run_dir, {**signature, "training_versions": versions}, "cosmos-nano",
        start_update=resume_update, resume_checkpoint=checkpoint_path if args.resume else None,
        experiment_group=logging_group,
    ) as logger:
        emit("load_pipeline")
        pipe = Cosmos3OmniPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, local_files_only=True,
            sound_tokenizer=None, safety_checker=None, enable_safety_checker=False,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=cfg.flow_shift, use_karras_sigmas=False
        )
        pipe.transformer.to(cfg.policy_device).eval().requires_grad_(False)
        pipe.vae.to(cfg.vae_device).eval().requires_grad_(False)
        # Keep native full-frame decoding. Overlapping tile graphs plus save_on_cpu
        # retain ~233 GiB per target decode at this resolution, versus ~98 GiB untiled.
        policy = pipe.transformer
        lora = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_rank * 2, target_modules=list(cfg.lora_targets))
        policy.add_adapter(lora, adapter_name="default")
        policy.add_adapter(lora, adapter_name="old")
        set_peft_model_state_dict(policy, get_peft_model_state_dict(policy), adapter_name="old")
        policy.set_adapter("default")
        policy.enable_gradient_checkpointing()
        trainable = [p for p in policy.parameters() if p.requires_grad]
        for p in trainable:
            p.data = p.data.float()
        optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
        roll = CosmosRollout(pipe, cfg.num_steps, cfg.guidance_scale, offload_vae_activations=cfg.offload_saved_tensors)
        emit("load_geometry_reward", trainable_parameters=sum(p.numel() for p in trainable))
        reward = GeoReward(
            load_depth_anything3(cfg.reward_device), load_waft(cfg.reward_device), load_dinov2(cfg.reward_device)
        ).eval().requires_grad_(False)
        tracker = PerPromptStatTracker(False)
        references, thresholds, completed = {}, {}, 0
        cursor_state = None
        if args.resume:
            state = load_training_state(checkpoint_path, config=signature)
            for name, values in state["adapters"].items():
                set_peft_model_state_dict(policy, values, adapter_name=name)
            optimizer.load_state_dict(state["optimizer"])
            completed = state["update"]
            references, thresholds = state["extra"]["references"], state["extra"]["thresholds"]
            tracker.__dict__.update(state["extra"]["tracker"])
            cursor_state = state["extra"].get("prompt_cursor")
        prompt_cursor = (
            PromptCursor(**cursor_state) if cursor_state is not None
            else PromptCursor(num_prompts=len(train_prompts), cursor=completed)
        )
        if prompt_cursor.num_prompts != len(train_prompts):
            raise ValueError("Checkpoint prompt cursor does not match the training prompt set")
        logger.log_event(completed, "resume_position", {"prompt_cursor": prompt_cursor.cursor})
        emit("training_position", completed_updates=completed, prompt_cursor=prompt_cursor.cursor)

        trace_stages = os.environ.get("COSMOS_TRACE_STAGES", "0") == "1"

        def traced(stage, function, *values, **keywords):
            if not trace_stages:
                return function(*values, **keywords)
            emit("stage_enter", stage=stage)
            started = time.monotonic()
            result = function(*values, **keywords)
            # Attribute asynchronous CUDA failures to the component that issued them.
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
            emit("stage_exit", stage=stage, seconds=time.monotonic() - started,
                 host_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
            return result

        def decode(latents):
            return traced("vae_decode", roll.decode01, latents)

        def geometry(clip):
            indices = torch.linspace(0, clip.shape[1] - 1, cfg.reward_frames, device=clip.device).round().long()
            clip = clip[:, indices].to(cfg.reward_device)
            b, t, c, h, w = clip.shape
            reduced = F.interpolate(clip.reshape(b*t, c, h, w), (cfg.reward_height, cfg.reward_width), mode="bilinear",
                                    align_corners=False).reshape(b, t, c, cfg.reward_height, cfg.reward_width)
            return traced("geometry_reward", reward, reduced)

        def local_reward(y):
            return geometry(decode(y)).geo.to(y.device)

        def fitting_progress(stage, sample_index):
            logger.log_event(completed, "fitting_stage", {
                "stage": stage, "sample_index": sample_index,
                "host_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                **{f"gpu/{i}/allocated_bytes": torch.cuda.memory_allocated(i)
                   for i in range(torch.cuda.device_count())},
                **{f"gpu/{i}/reserved_bytes": torch.cuda.memory_reserved(i)
                   for i in range(torch.cuda.device_count())},
            })
            emit(
                "fitting_stage", stage=stage, sample_index=sample_index,
                host_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                gpu_allocated_gib=[torch.cuda.memory_allocated(i) / 1024**3 for i in range(torch.cuda.device_count())],
                gpu_reserved_gib=[torch.cuda.memory_reserved(i) / 1024**3 for i in range(torch.cuda.device_count())],
            )

        def generate(prompt, seed):
            return traced(
                "native_generation", roll.rollout, structured_prompt(prompt),
                generator=torch.Generator(cfg.policy_device).manual_seed(seed),
                sigma_star=cfg.query_sigma, num_frames=cfg.num_frames, height=cfg.height, width=cfg.width, fps=cfg.fps,
            )

        judge = IsolatedVideoJudge(ROOT, run_dir, cfg.judge_device)
        try:
            def quality_score(clip, prompt):
                return traced("quality_reward", judge.score, clip, prompt)

            def reference(prompt, seed=None):
                seed = cfg.seed if seed is None else seed
                key = hashlib.sha256(f"{seed}:{prompt}".encode()).hexdigest()
                if key not in references:
                    policy.disable_adapters()
                    try:
                        with torch.no_grad():
                            rec = generate(prompt, seed)
                            clip = decode(rec.x0)
                            scored = geometry(clip)
                            references[key] = {
                                "quality": quality_score(clip[0], prompt), "geo": float(scored.geo),
                                "identity": float(scored.s_id), "motion": float(scored.motion),
                            }
                    finally:
                        policy.enable_adapters()
                return references[key]

            policy.set_adapter("old")
            if not thresholds:
                emit("calibration")
                baseline = [reference(p) for p in train_prompts[:cfg.calibration_prompts]]
                thresholds = {
                    "identity": cfg.tau_id if cfg.tau_id >= 0 else float(np.quantile([s["identity"] for s in baseline], .1)),
                    "motion": cfg.tau_motion if cfg.tau_motion >= 0 else float(np.quantile([s["motion"] for s in baseline], .1)),
                }
                logger.log_event(completed, "calibration", thresholds)
                emit("calibrated", **thresholds)

            def evaluate(update):
                policy.set_adapter("default")
                rows, base_rows, videos = [], [], []
                started = time.monotonic()
                with torch.no_grad():
                    for i, prompt in enumerate(eval_prompts):
                        seed = cfg.seed + 1000 + i
                        base = reference(prompt, seed)
                        rec = generate(prompt, seed)
                        clip = decode(rec.x0)
                        score = geometry(clip)
                        quality = quality_score(clip[0], prompt)
                        emit("evaluation", update=update, prompt=prompt, seed=seed,
                             geo=float(score.geo), base_geo=base["geo"],
                             identity=float(score.s_id), motion=float(score.motion), quality=quality)
                        if cfg.save_videos:
                            video_path = run_dir / f"eval_{update}_{i}.mp4"
                            save_video(clip[0], video_path, cfg.fps)
                            videos.append({"path": str(video_path), "prompt": prompt, "seed": seed, "index": i})
                        rows.append({"geo": float(score.geo), "identity": float(score.s_id),
                                     "motion": float(score.motion),
                                     **{f"quality_{key}": value for key, value in quality.items()}})
                        base_rows.append({"geo": base["geo"], "identity": base["identity"], "motion": base["motion"],
                                          **{f"quality_{key}": value for key, value in base["quality"].items()}})
                logger.log_eval(update, {**evaluation_metrics(rows, base_rows),
                                         "eval/duration_seconds": time.monotonic() - started}, videos)

            if completed == 0:
                evaluate(0)
            stop_at = min(cfg.updates, completed + args.max_updates_this_job) if args.max_updates_this_job else cfg.updates
            while completed < stop_at:
                prompt_index, sampling_attempt = prompt_cursor.take()
                prompt = train_prompts[prompt_index]
                base = reference(prompt)
                policy.set_adapter("old")
                records, rewards, motions, quality_probs = [], [], [], []
                emit("sampling", update=completed+1, prompt=prompt, sampling_attempt=sampling_attempt)
                with torch.no_grad():
                    for k in range(cfg.clips_per_prompt):
                        rec = generate(prompt, cfg.seed + (sampling_attempt+1)*100 + k)
                        clip = decode(rec.x0)
                        scored = geometry(clip)
                        quality = quality_score(clip[0], prompt)
                        p_quality = gap_to_probability(quality_gap(quality, base["quality"]))
                        identity = float(geometry(decode(rec.y0)).s_id)
                        if identity >= thresholds["identity"]:
                            records.append(rec)
                            rewards.append(float(scored.geo))
                            motions.append(float(scored.motion))
                            quality_probs.append(p_quality)
                if not records:
                    logger.log_event(completed, "sampling_skipped", {"reason": "identity_gate",
                                     "sampling_attempt": sampling_attempt, "prompt": prompt})
                    emit("sampling_skipped", reason="identity_gate", completed_updates=completed,
                         prompt=prompt, sampling_attempt=sampling_attempt,
                         consecutive_empty=prompt_cursor.consecutive_empty + 1)
                if not prompt_cursor.record_batch(len(records)):
                    continue
                advantages = tracker.update([prompt]*len(records), np.array(rewards))
                weights = effective_weight(
                    torch.tensor(advantages), cfg.loss_scale, torch.tensor(motions), thresholds["motion"],
                    torch.tensor(quality_probs), cfg.tau_quality,
                ).tolist()
                policy.set_adapter("default")
                policy.eval()  # LoRA still trains; keep inference/dropout behavior consistent.
                before = {k: v.detach().clone() for k, v in get_peft_model_state_dict(policy).items()}
                context = torch.autograd.graph.save_on_cpu() if cfg.offload_saved_tensors else nullcontext()
                logger.log_event(completed, "sampling", {"generated": cfg.clips_per_prompt,
                                 "retained": len(records), "sampling_attempt": sampling_attempt})
                emit("fitting", update=completed+1, retained=len(records))
                with context, torch.autocast("cuda", dtype=torch.bfloat16):
                    metrics = fit_queries(roll, records, weights, local_reward, optimizer, cfg, progress=fitting_progress)
                changed = any(not torch.equal(before[k], v) for k, v in get_peft_model_state_dict(policy).items())
                if not changed:
                    raise RuntimeError("Optimizer did not change any policy adapter weights")
                ema_adapter_(policy, decay=cfg.ema_decay)
                completed += 1
                logger.log_train(completed, {**metrics, "endpoint_reward": float(np.mean(rewards)),
                                 "mean_weight": float(np.mean(weights)), "policy_changed": changed,
                                 "learning_rate": optimizer.param_groups[0]["lr"],
                                 "n_rollouts": cfg.clips_per_prompt,
                                 "frac_motion_masked": float(np.mean(np.array(motions) < thresholds["motion"])),
                                 "frac_quality_masked": float(np.mean(np.array(quality_probs) < cfg.tau_quality)),
                                 "loss_scale": cfg.loss_scale})
                emit("update", update=completed, endpoint_reward=float(np.mean(rewards)), mean_weight=float(np.mean(weights)),
                     policy_changed=changed, **metrics)
                with open(run_dir / "metrics.jsonl", "a") as stream:
                    stream.write(json.dumps({"update": completed, "endpoint_reward": float(np.mean(rewards)), **metrics})+"\n")
                del records
                if evaluation_due(completed, stop_at, cfg.eval_every):
                    evaluate(completed)
                if evaluation_due(completed, stop_at, cfg.checkpoint_every):
                    save_training_state(
                        checkpoint_path, adapters=cpu_adapters(policy), optimizer=optimizer.state_dict(), update=completed,
                        config=signature, extra={"references": references, "thresholds": thresholds,
                                                 "tracker": tracker.__dict__, "prompt_cursor": prompt_cursor.state_dict(),
                                                     "logging_experiment_group": logger.experiment_group},
                    )
                    logger.log_event(completed, "checkpoint", {"path": str(checkpoint_path)})
                    emit("checkpoint", update=completed, path=str(checkpoint_path))
            status = "complete" if completed >= cfg.updates else "continue"
            (run_dir / f"job_{os.environ.get('SLURM_JOB_ID', 'local')}.json").write_text(
                json.dumps({"status": status, "completed_updates": completed, "checkpoint": str(checkpoint_path)})
            )
            emit("finished", status=status, completed_updates=completed)
            logger.log_event(completed, "job_result", {"status": status})
        finally:
            judge.close()


if __name__ == "__main__":
    main()
