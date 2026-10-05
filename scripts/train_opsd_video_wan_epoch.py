#!/usr/bin/env python3
# ruff: noqa: E402
"""Four-worker, one-pass Wan2.2 OPSD trainer on four three-A100 nodes."""
from __future__ import annotations

import hashlib
import json
import os
import random
import resource
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
import torch.distributed as dist
import wandb
from absl import app, flags
from diffusers import WanPipeline
from ml_collections import config_flags
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from diffusionopsd.metrics import install_wandb_jsonl_tee
from diffusionopsd.stat_tracking import PerPromptStatTracker
from diffusionopsd.video.branch_loss import branch_loss
from diffusionopsd.video.distributed_epoch import (
    atomic_checkpoint, deterministic_seed, load_checkpoint, prompt_schedule,
    should_start_update, split_update, synchronize_gradients,
)
from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
from diffusionopsd.video.gates import effective_weight, identity_keep, percentile_threshold
from diffusionopsd.video.geo_reward import GeoReward
from diffusionopsd.video.opa_video import opa_tr_step_nd
from diffusionopsd.video.quality_judge import load_video_reward
from diffusionopsd.video.training_eval import EvaluationMonitor, evaluation_due
from diffusionopsd.video.wan_clean_output import WanRollout, clean_output
from diffusionopsd.video.wan_geometry import latent_shape_from_pipe, require_expand_timesteps
from diffusionopsd.video.wan_policy import attach_lora, ema_adapter_

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/wan22_ti2v_epoch.py")
flags.DEFINE_float("job_budget_hours", 20.0, "Stop starting updates within the 24-hour allocation")


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _prompt_seed(prompt):
    return int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)


def _broadcast_adapter(policy):
    # Initialize every worker from rank 0's exact default and old LoRA weights.
    for name, parameter in policy.named_parameters():
        if "lora_" in name:
            dist.broadcast(parameter.data, src=0)


def _log(record, videos=()):
    payload = dict(record)
    if videos:
        payload["eval/videos"] = [wandb.Video(v["path"], format="mp4",
                                        caption=f"{v['prompt']} | seed={v['seed']}") for v in videos]
    wandb.log(payload, step=record["optimizer_updates"])
    print(json.dumps(record, allow_nan=False), flush=True)


def _load_adapter(policy, checkpoint):
    set_peft_model_state_dict(policy, checkpoint["policy_default"], adapter_name="default")
    set_peft_model_state_dict(policy, checkpoint["policy_old"], adapter_name="old")
    policy.set_adapter("default")


def _cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _calibrate(pipe, roll, reward, cfg, prompts, run_dir):
    refs = {}
    s_ids, motions = [], []
    for index in random.Random(0).sample(range(len(prompts)), cfg.gates.calib_prompts):
        prompt = prompts[index]
        pe, ne = pipe.encode_prompt(prompt, negative_prompt="", do_classifier_free_guidance=True,
                                    device="cuda:0")[:2]
        generator = torch.Generator("cuda:0").manual_seed(_prompt_seed(prompt))
        noise = torch.randn(latent_shape_from_pipe(pipe, cfg.video.num_frames, cfg.video.height, cfg.video.width),
                            device="cuda:0", dtype=torch.float32, generator=generator)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rec = roll.rollout(pe, ne, noise, cfg.opa.query_sigma)
            clip = roll.decode01(rec.x0)
            scored = reward(clip.to(cfg.reward_device))
        path = run_dir / "calibration" / f"reference_{index:03d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clip[0].cpu(), path)
        refs[index] = str(path)
        s_ids.append(scored.s_id.cpu())
        motions.append(scored.motion.cpu())
        del pe, ne, noise, rec, clip, scored
    tau_id = cfg.gates.tau_id if cfg.gates.tau_id >= 0 else percentile_threshold(torch.cat(s_ids), 0.10)
    tau_motion = cfg.gates.tau_motion if cfg.gates.tau_motion >= 0 else percentile_threshold(torch.cat(motions), 0.10)
    return refs, float(tau_id), float(tau_motion)


def _sample_local(pipe, roll, reward, judge, cfg, prompt, prompt_index, nonce, reference_files):
    pe, ne = pipe.encode_prompt(prompt, negative_prompt="", do_classifier_free_guidance=True,
                                device="cuda:0")[:2]
    if prompt_index in reference_files:
        reference = torch.load(reference_files[prompt_index], map_location="cpu", weights_only=True)
    else:
        generator = torch.Generator("cuda:0").manual_seed(_prompt_seed(prompt))
        noise = torch.randn(latent_shape_from_pipe(pipe, cfg.video.num_frames, cfg.video.height, cfg.video.width),
                            device="cuda:0", dtype=torch.float32, generator=generator)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rec_ref = roll.rollout(pe, ne, noise, cfg.opa.query_sigma)
            reference = roll.decode01(rec_ref.x0)[0].cpu()
        del noise, rec_ref
    tuples = []
    for repetition in range(cfg.sample.num_image_per_prompt):
        generator = torch.Generator("cuda:0").manual_seed(deterministic_seed(nonce, prompt_index, repetition))
        latents = torch.randn(latent_shape_from_pipe(pipe, cfg.video.num_frames, cfg.video.height, cfg.video.width),
                              device="cuda:0", dtype=torch.float32, generator=generator)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rec = roll.rollout(pe, ne, latents, cfg.opa.query_sigma)
            clip = roll.decode01(rec.x0)
            scored = reward(clip.to(cfg.reward_device))
            p_q = judge.p_win(clip[0].to(cfg.judge.device), reference.to(cfg.judge.device), prompt)
        tuples.append({"prompt": prompt, "pe": pe, "ne": ne, "rec": rec,
                       "r": float(scored.geo), "m": float(scored.motion), "p_q": float(p_q)})
        del latents, clip, scored
    return tuples


def _targets_and_gradients(tuples, weights, roll, reward, policy, opt, cfg, tau_id):
    def r_geo(latents):
        return reward(roll.decode01(latents).to(cfg.reward_device)).geo.to(latents.device)

    kept = []
    for sample, weight in zip(tuples, weights):
        rec = sample["rec"]
        sigma = torch.tensor([rec.sigma_q], device="cuda:0")
        y0 = clean_output(rec.z_q, rec.v_old_q, sigma).float()
        with torch.no_grad():
            s_id = reward(roll.decode01(y0).to(cfg.reward_device)).s_id
        if not bool(identity_keep(s_id, tau_id).all()):
            continue
        yg = y0.clone().requires_grad_(True)
        (g0,) = torch.autograd.grad(r_geo(yg).sum(), yg)
        sample["y0"] = y0
        sample["y_plus"] = opa_tr_step_nd(y0, r_geo, cfg.opa.rho, cfg.opa.n_ascent,
                                           cfg.opa.eta, +1.0, first_grad=g0)
        sample["y_minus"] = opa_tr_step_nd(y0, r_geo, cfg.opa.rho, cfg.opa.n_ascent,
                                            cfg.opa.eta, -1.0, first_grad=g0)
        sample["weight"] = weight
        kept.append(sample)
    policy.set_adapter("default")
    policy.train()
    opt.zero_grad(set_to_none=True)
    local_loss = 0.0
    for sample in kept:
        rec = sample["rec"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            velocity = roll.velocity(rec.z_q, torch.tensor(rec.sigma_q, device="cuda:0"),
                                     sample["pe"], sample["ne"], transformer=policy)
        y_theta = clean_output(rec.z_q, velocity, torch.tensor([rec.sigma_q], device="cuda:0")).float()
        loss = branch_loss(y_theta, sample["y0"], sample["y_plus"], sample["y_minus"],
                           torch.tensor([sample["weight"]], device="cuda:0"), cfg.beta).mean()
        local_loss += float(loss.detach())
        (loss * cfg.train.adv_clip_max).backward()
    return len(kept), local_loss


def main(_):
    cfg = FLAGS.config
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    if world != 4 or cfg.sample.num_batches_per_epoch != world or cfg.sample.num_image_per_prompt != 4:
        raise ValueError("This preset requires four workers, one prompt and four rollouts per worker")
    if cfg.num_epochs != 50 or cfg.save_freq != 50 or cfg.eval_freq != 25 or not cfg.use_lora:
        raise ValueError("Expected 50-update Wan2.2 LoRA preset with evaluation every 25")
    torch.cuda.set_device(0)
    # Rank 0 performs calibration and video evaluation while peers wait.
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(hours=6))
    started = time.monotonic()
    run_dir = Path(cfg.logdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_bytes = Path(cfg.prompt_fn_kwargs["path"]).read_bytes()
    prompts = [line.strip() for line in prompt_bytes.decode().splitlines() if line.strip()]
    prompt_hash = _hash(prompt_bytes)
    config_hash = _hash(json.dumps(cfg.to_dict(), sort_keys=True).encode())
    if len(prompts) != 200:
        raise ValueError(f"Expected 200 training prompts, got {len(prompts)}")
    evaluator = EvaluationMonitor(cfg) if rank == 0 else None
    state_path = run_dir / "training_state.pt"
    state = load_checkpoint(state_path, prompt_hash, config_hash) if state_path.exists() else None
    if rank == 0:
        wandb.init(project="diffusionopsd", group=run_dir.name,
                   name=f"{run_dir.name}-job-{os.environ['SLURM_JOB_ID']}", dir=str(run_dir),
                   mode="offline", config=cfg.to_dict())
        install_wandb_jsonl_tee(wandb, run_dir / "metrics.jsonl", durable=True, strict=True)
    model_path = os.path.expandvars(cfg.pretrained.model)
    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda:0")
    pipe.vae.requires_grad_(False).to(cfg.vae_device)
    pipe.text_encoder.requires_grad_(False)
    require_expand_timesteps(pipe, model_path)
    pipe.transformer.requires_grad_(False)
    pipe.transformer.enable_gradient_checkpointing()
    policy = attach_lora(pipe.transformer, cfg.train.lora_path)
    pipe.transformer = policy
    if state is None:
        _broadcast_adapter(policy)
    else:
        _load_adapter(policy, state)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.train.learning_rate, weight_decay=cfg.train.adam_weight_decay)
    if state is not None:
        opt.load_state_dict(state["optimizer"])
    roll = WanRollout(pipe, cfg.sample.num_steps, cfg.sample.guidance_scale,
                      offload_vae_activations=cfg.offload_vae_activations)
    reward = GeoReward(load_depth_anything3(cfg.reward_device), load_waft(cfg.reward_device),
                       load_dinov2(cfg.reward_device)).requires_grad_(False)
    judge = load_video_reward(cfg.judge.device, num_frames=cfg.judge.num_frames)
    tracker = PerPromptStatTracker(cfg.sample.global_std) if rank == 0 else None
    if state is None:
        if rank == 0:
            nonce = int.from_bytes(os.urandom(8), "big")
            order = prompt_schedule(prompts, nonce, world)
            policy.set_adapter("old")
            refs, tau_id, tau_motion = _calibrate(pipe, roll, reward, cfg, prompts, run_dir)
            policy.set_adapter("default")
            with torch.no_grad():
                metrics, videos = evaluator.evaluate(pipe, policy, roll, reward, judge, 0)
            _log({"optimizer_updates": 0, **metrics}, videos)
            evaluator_state = evaluator.baseline
            payload = (nonce, order, refs, tau_id, tau_motion, evaluator_state)
        else:
            payload = None
        values = [payload]
        dist.broadcast_object_list(values, src=0)
        nonce, order, refs, tau_id, tau_motion, evaluator_state = values[0]
        completed, max_update_seconds = 0, None
    else:
        nonce, order, refs = state["nonce"], state["order"], state["reference_files"]
        tau_id, tau_motion = state["tau_id"], state["tau_motion"]
        completed, max_update_seconds = state["next_update"], state["max_update_seconds"]
        if rank == 0:
            tracker.stats = state["tracker_stats"]
            tracker.history_prompts = {hash(prompt) for prompt in tracker.stats}
            evaluator.baseline = state["eval_baseline"]
        _restore_rng(state["rng_states"][rank])
    if state is None:
        rng_states = [None] * world
        dist.all_gather_object(rng_states, _rng_state())
    if rank == 0 and state is None:
        _save_state(state_path, policy, opt, tracker, evaluator, prompt_hash, config_hash,
                    nonce, order, refs, tau_id, tau_motion, completed, max_update_seconds, rng_states)
    dist.barrier()
    del state
    while completed < cfg.num_epochs:
        if rank == 0:
            can_start = should_start_update(completed, cfg.num_epochs, time.monotonic() - started,
                                            max_update_seconds, FLAGS.job_budget_hours * 3600)
        else:
            can_start = None
        decision = [can_start]
        dist.broadcast_object_list(decision, src=0)
        if not decision[0]:
            break
        update_start = time.monotonic()
        for device in range(3):
            torch.cuda.reset_peak_memory_stats(device)
        index = split_update(order, completed, rank, world)[0]
        prompt = prompts[index]
        policy.set_adapter("old")
        tuples = _sample_local(pipe, roll, reward, judge, cfg, prompt, index, nonce, refs)
        # VideoReward is not used by the differentiable geometry target. Move its
        # frozen weights off the scorer GPU before WAFT recomputes in backward.
        judge.to("cpu")
        with torch.cuda.device(cfg.judge.device):
            torch.cuda.empty_cache()
        gathered = [None] * world
        dist.all_gather_object(gathered, [(t["prompt"], t["r"], t["m"], t["p_q"]) for t in tuples])
        sample_seconds = time.monotonic() - update_start
        if rank == 0:
            flat = [entry for group in gathered for entry in group]
            advantages = tracker.update([entry[0] for entry in flat], np.array([entry[1] for entry in flat]))
            weights = [float(effective_weight(torch.tensor([float(a)]), cfg.train.adv_clip_max,
                                              torch.tensor([entry[2]]), tau_motion,
                                              torch.tensor([entry[3]]), cfg.gates.tau_q))
                       for entry, a in zip(flat, advantages)]
            weight_groups = [weights[i * 4:(i + 1) * 4] for i in range(world)]
        else:
            weight_groups = None
        weight_box = [weight_groups]
        dist.broadcast_object_list(weight_box, src=0)
        target_start = time.monotonic()
        local_kept, local_loss = _targets_and_gradients(tuples, weight_box[0][rank], roll, reward,
                                                         policy, opt, cfg, tau_id)
        judge.to(cfg.judge.device)
        target_seconds = time.monotonic() - target_start
        optimizer_start = time.monotonic()
        counts = torch.tensor([local_kept, local_loss, sum(t["r"] for t in tuples),
                               sum(weight_box[0][rank]), sum(t["m"] < tau_motion for t in tuples),
                               sum(t["p_q"] < cfg.gates.tau_q for t in tuples)],
                              device="cuda:0", dtype=torch.float64)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        total_kept = int(counts[0].item())
        if total_kept == 0:
            raise RuntimeError("No retained rollout in global update")
        synchronize_gradients(trainable, total_kept)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.train.max_grad_norm)
        if not torch.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError(f"Invalid gradient norm at update {completed + 1}: {grad_norm}")
        opt.step()
        ema_adapter_(policy, src="default", dst="old", decay=0.99)
        opt.zero_grad(set_to_none=True)
        tuples.clear()
        for device in range(3):
            torch.cuda.synchronize(device)
        optimizer_seconds = time.monotonic() - optimizer_start
        local_duration = time.monotonic() - update_start
        timings = torch.tensor([local_duration], device="cuda:0")
        dist.all_reduce(timings, op=dist.ReduceOp.MAX)
        phase_times = torch.tensor([sample_seconds, target_seconds, optimizer_seconds], device="cuda:0")
        dist.all_reduce(phase_times, op=dist.ReduceOp.MAX)
        completed += 1
        max_update_seconds = max(max_update_seconds or 0, float(timings.item()))
        peaks = {f"gpu/{rank}/{device}/peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                 for device in range(3)}
        peaks[f"worker/{rank}/host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        peak_list = [None] * world
        dist.all_gather_object(peak_list, peaks)
        if rank == 0:
            record = {"optimizer_updates": completed, "train/rl_distillation_loss": counts[1].item() / total_kept,
                      "train/optimization_loss": counts[1].item() * cfg.train.adv_clip_max / total_kept,
                      "grad_norm": float(grad_norm), "n_rollouts": world * cfg.sample.num_image_per_prompt,
                      "n_kept": total_kept, "mean_r": counts[2].item() / 16,
                      "mean_w": counts[3].item() / 16,
                      "frac_motion_masked": counts[4].item() / 16,
                      "frac_quality_masked": counts[5].item() / 16,
                      "tau_id": tau_id, "tau_motion": tau_motion,
                      "train/update_seconds": float(timings.item()),
                      "train/sampling_seconds": float(phase_times[0]),
                      "train/target_seconds": float(phase_times[1]),
                      "train/optimizer_seconds": float(phase_times[2]),
                      **{key: value for group in peak_list for key, value in group.items()}}
            videos = []
            if evaluation_due(completed, cfg.num_epochs, cfg.eval_freq):
                metrics, videos = evaluator.evaluate(pipe, policy, roll, reward, judge, completed)
                record.update(metrics)
            _log(record, videos)
            if completed % cfg.save_freq == 0:
                torch.save(get_peft_model_state_dict(policy, adapter_name="default"),
                           run_dir / f"policy_{completed}.pt")
        rng_states = [None] * world
        dist.all_gather_object(rng_states, _rng_state())
        if rank == 0:
            _save_state(state_path, policy, opt, tracker, evaluator, prompt_hash, config_hash,
                        nonce, order, refs, tau_id, tau_motion, completed, max_update_seconds, rng_states)
        dist.barrier()
    if rank == 0:
        result = {"status": "complete" if completed == cfg.num_epochs else "continue",
                  "completed_updates": completed, "total_updates": cfg.num_epochs,
                  "run_dir": str(run_dir), "job_id": os.environ["SLURM_JOB_ID"]}
        (run_dir / f"job_{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(result, indent=2) + "\n")
        wandb.finish()
        print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def _save_state(path, policy, opt, tracker, evaluator, prompt_hash, config_hash,
                nonce, order, refs, tau_id, tau_motion, completed, max_update_seconds, rng_states):
    state = {"version": 1, "next_update": completed, "prompt_hash": prompt_hash,
             "config_hash": config_hash, "nonce": nonce, "order": order,
             "reference_files": refs, "tau_id": tau_id, "tau_motion": tau_motion,
             "policy_default": _cpu_state(get_peft_model_state_dict(policy, adapter_name="default")),
             "policy_old": _cpu_state(get_peft_model_state_dict(policy, adapter_name="old")),
             "optimizer": _cpu_state(opt.state_dict()), "tracker_stats": tracker.stats,
             "eval_baseline": evaluator.baseline, "max_update_seconds": max_update_seconds,
             "rng_states": rng_states}
    atomic_checkpoint(path, state)


if __name__ == "__main__":
    app.run(main)
