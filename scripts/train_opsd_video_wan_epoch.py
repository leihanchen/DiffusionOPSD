#!/usr/bin/env python3
# ruff: noqa: E402
"""Fixed-global-batch Wan2.2 OPSD trainer with scalable worker placement."""

from __future__ import annotations

import hashlib
import faulthandler
import json
import os
import random
import resource
import socket
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
from diffusionopsd.video.nccl_network import configure_socket_interface
from diffusionopsd.video.branch_loss import branch_loss
from diffusionopsd.video.distributed_epoch import (
    active_cuda_devices,
    adapter_fingerprint,
    canonical_samples,
    configure_isolated_worker_nccl,
    prepare_resume,
    route_sample_weights,
    topology_rng_seed,
    training_config_hash,
    validate_worker_devices,
    worker_sample_assignments,
    atomic_checkpoint,
    deterministic_seed,
    prompt_schedule,
    should_start_update,
    synchronize_gradients,
)
from scripts.wan_paths import portable_path, portable_reference_files, resolve_reference_files
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
flags.DEFINE_float("distributed_timeout_minutes", 180.0, "Timeout for a distributed collective")
flags.DEFINE_integer("max_updates_this_job", 0, "Stop after this many new updates; zero uses the time budget")
flags.DEFINE_integer("monitor_eval_freq", 0, "Evaluation interval override; zero retains the checkpoint config")
flags.DEFINE_bool("evaluate_on_resume", False, "Evaluate the loaded policy before further training")
flags.DEFINE_integer("expected_resume_update", -1, "Require this checkpoint update; -1 permits any start")
flags.DEFINE_string("resume_from", "", "Explicit trusted local training checkpoint; write to a new run directory")
flags.DEFINE_bool("allow_topology_change", False, "Permit validated resume with deterministic worker RNG reseeding")
flags.DEFINE_bool("scalable_two_gpu", False, "Require isolated two-GPU workers with 4/8/16 processes")
flags.DEFINE_bool("single_node_smoke", False, "Explicit two-worker, one-node, one-update test with the full batch")


def _trace(stage, **details):
    print(
        json.dumps(
            {
                "event": "progress",
                "stage": stage,
                "rank": os.environ.get("SLURM_PROCID"),
                "host": socket.gethostname(),
                "time": time.time(),
                "cuda_device": torch.cuda.current_device() if torch.cuda.is_initialized() else None,
                **details,
            }
        ),
        flush=True,
    )


def _barrier(stage):
    _trace(f"{stage}:enter")
    dist.barrier(device_ids=[0])
    _trace(f"{stage}:exit")


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
        payload["eval/videos"] = [
            wandb.Video(v["path"], format="mp4", caption=f"{v['prompt']} | seed={v['seed']}") for v in videos
        ]
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
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


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
        pe, ne = pipe.encode_prompt(prompt, negative_prompt="", do_classifier_free_guidance=True, device="cuda:0")[:2]
        generator = torch.Generator("cuda:0").manual_seed(_prompt_seed(prompt))
        noise = torch.randn(
            latent_shape_from_pipe(pipe, cfg.video.num_frames, cfg.video.height, cfg.video.width),
            device="cuda:0",
            dtype=torch.float32,
            generator=generator,
        )
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


def _sample_local(pipe, roll, reward, judge, cfg, prompt, prompt_index, nonce, reference_files, rollout_indices):
    pe, ne = pipe.encode_prompt(prompt, negative_prompt="", do_classifier_free_guidance=True, device="cuda:0")[:2]
    if prompt_index in reference_files:
        reference = torch.load(reference_files[prompt_index], map_location="cpu", weights_only=True)
    else:
        generator = torch.Generator("cuda:0").manual_seed(_prompt_seed(prompt))
        noise = torch.randn(
            latent_shape_from_pipe(pipe, cfg.video.num_frames, cfg.video.height, cfg.video.width),
            device="cuda:0",
            dtype=torch.float32,
            generator=generator,
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rec_ref = roll.rollout(pe, ne, noise, cfg.opa.query_sigma)
            reference = roll.decode01(rec_ref.x0)[0].cpu()
        del noise, rec_ref
    tuples = []
    for repetition in rollout_indices:
        generator = torch.Generator("cuda:0").manual_seed(deterministic_seed(nonce, prompt_index, repetition))
        latents = torch.randn(
            latent_shape_from_pipe(pipe, cfg.video.num_frames, cfg.video.height, cfg.video.width),
            device="cuda:0",
            dtype=torch.float32,
            generator=generator,
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rec = roll.rollout(pe, ne, latents, cfg.opa.query_sigma)
            clip = roll.decode01(rec.x0)
            scored = reward(clip.to(cfg.reward_device))
            p_q = judge.p_win(clip[0].to(cfg.judge.device), reference.to(cfg.judge.device), prompt)
        tuples.append(
            {
                "prompt": prompt,
                "prompt_index": prompt_index,
                "rollout_index": repetition,
                "pe": pe,
                "ne": ne,
                "rec": rec,
                "r": float(scored.geo),
                "m": float(scored.motion),
                "p_q": float(p_q),
            }
        )
        del latents, clip, scored
    return tuples


def _targets_and_gradients(tuples, weights, roll, reward, policy, opt, cfg, tau_id):
    def r_geo(latents):
        return reward(roll.decode01(latents).to(cfg.reward_device)).geo.to(latents.device)

    kept = []
    for sample_index, (sample, weight) in enumerate(zip(tuples, weights)):
        stage = f"target_sample_{sample_index}"
        _trace(f"{stage}:identity:enter")
        rec = sample["rec"]
        sigma = torch.tensor([rec.sigma_q], device="cuda:0")
        y0 = clean_output(rec.z_q, rec.v_old_q, sigma).float()
        with torch.no_grad():
            s_id = reward(roll.decode01(y0).to(cfg.reward_device)).s_id
        keep = bool(identity_keep(s_id, tau_id).all())
        _trace(f"{stage}:identity:exit", kept=keep, identity_score=s_id.detach().cpu().tolist())
        if not keep:
            continue
        _trace(f"{stage}:first_gradient:enter")
        yg = y0.clone().requires_grad_(True)
        (g0,) = torch.autograd.grad(r_geo(yg).sum(), yg)
        _trace(f"{stage}:first_gradient:exit")
        sample["y0"] = y0
        _trace(f"{stage}:positive_target:enter")
        sample["y_plus"] = opa_tr_step_nd(y0, r_geo, cfg.opa.rho, cfg.opa.n_ascent, cfg.opa.eta, +1.0, first_grad=g0)
        _trace(f"{stage}:positive_target:exit")
        _trace(f"{stage}:negative_target:enter")
        sample["y_minus"] = opa_tr_step_nd(y0, r_geo, cfg.opa.rho, cfg.opa.n_ascent, cfg.opa.eta, -1.0, first_grad=g0)
        _trace(f"{stage}:negative_target:exit")
        sample["weight"] = weight
        kept.append(sample)
    policy.set_adapter("default")
    policy.train()
    opt.zero_grad(set_to_none=True)
    local_loss = 0.0
    _trace("policy_gradients:enter", retained_rollouts=len(kept))
    for sample_index, sample in enumerate(kept):
        _trace(f"policy_sample_{sample_index}:enter")
        rec = sample["rec"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            velocity = roll.velocity(
                rec.z_q, torch.tensor(rec.sigma_q, device="cuda:0"), sample["pe"], sample["ne"], transformer=policy
            )
        y_theta = clean_output(rec.z_q, velocity, torch.tensor([rec.sigma_q], device="cuda:0")).float()
        loss = branch_loss(
            y_theta,
            sample["y0"],
            sample["y_plus"],
            sample["y_minus"],
            torch.tensor([sample["weight"]], device="cuda:0"),
            cfg.beta,
        ).mean()
        local_loss += float(loss.detach())
        (loss * cfg.train.adv_clip_max).backward()
        _trace(f"policy_sample_{sample_index}:exit", loss=float(loss.detach()))
    _trace("policy_gradients:exit", retained_rollouts=len(kept))
    return len(kept), local_loss


def main(_):
    cfg = FLAGS.config
    if FLAGS.max_updates_this_job < 0 or FLAGS.distributed_timeout_minutes <= 0:
        raise ValueError("Update limit must be nonnegative and distributed timeout must be positive")
    if FLAGS.monitor_eval_freq < 0 or FLAGS.expected_resume_update < -1:
        raise ValueError("Monitoring frequency must be nonnegative; expected resume update must be >= -1")
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    allowed_worlds = (2,) if FLAGS.single_node_smoke else (4, 8, 16)
    if FLAGS.single_node_smoke and (not FLAGS.scalable_two_gpu or FLAGS.max_updates_this_job != 1):
        raise ValueError("Single-node smoke requires the scaled launcher and exactly one new update")
    if world not in allowed_worlds or cfg.sample.num_batches_per_epoch != 4 or cfg.sample.num_image_per_prompt != 4:
        raise ValueError("Require 4/8/16 workers and a fixed four-prompt, four-rollout-per-prompt batch")
    if not FLAGS.scalable_two_gpu and world != 4:
        raise ValueError("Additional workers require the scalable two-GPU launcher")
    if cfg.num_epochs != 50 or cfg.save_freq != 50 or cfg.eval_freq != 25 or not cfg.use_lora:
        raise ValueError("Expected 50-update Wan2.2 LoRA preset with evaluation every 25")
    devices = active_cuda_devices(cfg, torch.cuda.device_count())
    if FLAGS.scalable_two_gpu and (torch.cuda.device_count() != 2 or devices != (0, 1)):
        raise ValueError("Scalable workers must see exactly two CUDA devices and use cuda:0/cuda:1")
    if FLAGS.scalable_two_gpu:
        # Slurm snapshots batch scripts at submission; queued jobs use this live entry point.
        _trace("nccl_environment", **configure_isolated_worker_nccl())
        _trace("nccl_network", **configure_socket_interface(os.environ["MASTER_ADDR"]))
    torch.cuda.set_device(0)
    faulthandler.enable()
    # CPython's timer dumps thread frames without the GIL and can itself segfault
    # mid-dump (python/cpython#116008). Keep fatal-error handling and stage traces.
    placement = {
        "rank": rank,
        "local_rank": int(os.environ.get("SLURM_LOCALID", "0")),
        "host": socket.gethostname(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_uuids": [str(getattr(torch.cuda.get_device_properties(d), "uuid", "")) for d in devices],
    }
    # Emit local evidence before NCCL: a failed first collective cannot gather it.
    _trace(
        "worker_placement:local",
        **placement,
        cuda_device_count=torch.cuda.device_count(),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        nccl_version=torch.cuda.nccl.version(),
    )
    _trace("process_group:enter")
    # Rank 0 performs calibration and video evaluation while peers wait.
    # Slurm isolates each worker's devices; global rank is never a local CUDA index.
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=FLAGS.distributed_timeout_minutes),
        device_id=torch.device("cuda:0"),
    )
    _trace("process_group:exit")
    placements = [None] * world
    dist.all_gather_object(placements, placement)
    if FLAGS.scalable_two_gpu:
        validate_worker_devices(placements, world)
    hosts = {entry["host"] for entry in placements}
    per_host = {host: sum(entry["host"] == host for entry in placements) for host in hosts}
    if len(set(per_host.values())) != 1:
        raise ValueError("Require equal numbers of workers on each node")
    topology = {"world_size": world, "workers_per_node": next(iter(per_host.values())), "nodes": len(hosts)}
    if FLAGS.single_node_smoke:
        if topology["nodes"] != 1 or topology["workers_per_node"] != 2:
            raise ValueError("Single-node smoke must run two workers on exactly one node")
        topology["single_node_smoke"] = True
    _trace("worker_placement", **placement, topology=topology)
    _trace(
        "model_placement",
        policy_device="cuda:0",
        vae_device=cfg.vae_device,
        reward_device=cfg.reward_device,
        judge_device=cfg.judge.device,
        active_devices=devices,
    )
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
    _trace("resume_checkpoint:enter")
    source_path = Path(FLAGS.resume_from).expanduser().resolve() if FLAGS.resume_from else state_path
    if FLAGS.resume_from:
        if not source_path.is_file():
            raise ValueError(f"Missing trusted resume checkpoint: {source_path}")
        if source_path.parent.resolve() == run_dir.resolve() or state_path.exists():
            raise ValueError("Explicit checkpoint migration requires a new destination run directory")
    state = torch.load(source_path, map_location="cpu", weights_only=False) if source_path.exists() else None
    topology_changed = False
    if state is not None:
        state, topology_changed = prepare_resume(
            state,
            cfg.to_dict(),
            prompt_hash,
            topology,
            FLAGS.allow_topology_change,
            str(source_path.parent) if FLAGS.resume_from else None,
        )
        reference_root = source_path.resolve().parents[2] if source_path.parent.parent.name == "logs" else None
        state["reference_files"] = resolve_reference_files(state["reference_files"], ROOT, source_root=reference_root)
    migrations = state.get("topology_migrations", []) if state else []
    if topology_changed:
        migrations[-1]["source_checkpoint"] = portable_path(source_path, ROOT)
    _trace("checkpoint_topology", changed=topology_changed, migrations=migrations)
    if FLAGS.expected_resume_update >= 0 and (state is None or state["next_update"] != FLAGS.expected_resume_update):
        raise ValueError(
            f"Expected checkpoint at update {FLAGS.expected_resume_update}, "
            f"got {state['next_update'] if state else 'no checkpoint'}"
        )
    _trace("resume_checkpoint:exit")
    if rank == 0:
        wandb.init(
            project="diffusionopsd",
            group=run_dir.name,
            name=f"{run_dir.name}-job-{os.environ['SLURM_JOB_ID']}",
            dir=str(run_dir),
            mode="offline",
            config={
                **cfg.to_dict(),
                "execution": {
                    **topology,
                    "placements": placements,
                    "topology_migrations": migrations,
                    "resume_checkpoint": str(source_path) if state is not None else None,
                },
                "monitoring": {
                    "eval_freq": FLAGS.monitor_eval_freq or cfg.eval_freq,
                    "evaluate_on_resume": FLAGS.evaluate_on_resume,
                    "expected_resume_update": FLAGS.expected_resume_update,
                    "max_updates_this_job": FLAGS.max_updates_this_job,
                },
            },
        )
        install_wandb_jsonl_tee(wandb, run_dir / "metrics.jsonl", durable=True, strict=True)
    model_path = os.path.expandvars(cfg.pretrained.model)
    _trace("wan_model:enter")
    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda:0")
    _trace("wan_model:exit")
    pipe.vae.requires_grad_(False).to(cfg.vae_device)
    pipe.text_encoder.requires_grad_(False)
    require_expand_timesteps(pipe, model_path)
    pipe.transformer.requires_grad_(False)
    pipe.transformer.enable_gradient_checkpointing()
    policy = attach_lora(pipe.transformer, cfg.train.lora_path)
    pipe.transformer = policy
    _trace("adapter_state:enter")
    if state is None:
        _broadcast_adapter(policy)
    else:
        _load_adapter(policy, state)
    _trace("adapter_state:exit")
    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.train.learning_rate, weight_decay=cfg.train.adam_weight_decay)
    if state is not None:
        opt.load_state_dict(state["optimizer"])
    roll = WanRollout(
        pipe, cfg.sample.num_steps, cfg.sample.guidance_scale, offload_vae_activations=cfg.offload_vae_activations
    )
    _trace("geometry_models:enter")
    reward = GeoReward(
        load_depth_anything3(cfg.reward_device), load_waft(cfg.reward_device), load_dinov2(cfg.reward_device)
    ).requires_grad_(False)
    _trace("geometry_models:exit")
    _trace("quality_model:enter")
    judge = load_video_reward(cfg.judge.device, num_frames=cfg.judge.num_frames)
    _trace("quality_model:exit")
    tracker = PerPromptStatTracker(cfg.sample.global_std) if rank == 0 else None
    if state is None:
        if rank == 0:
            nonce = int.from_bytes(os.urandom(8), "big")
            order = prompt_schedule(prompts, nonce, 4)
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
        _trace("rng_restore:enter")
        if topology_changed:
            seed = topology_rng_seed(nonce, completed, rank)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            _trace("rng_reseed", seed=seed)
        else:
            _restore_rng(state["rng_states"][rank])
        _trace("rng_restore:exit")
        if rank == 0 and FLAGS.evaluate_on_resume:
            _trace("resume_evaluation:enter", optimizer_updates=completed)
            metrics, videos = evaluator.evaluate(pipe, policy, roll, reward, judge, completed)
            _log({"optimizer_updates": completed, **metrics}, videos)
            _trace("resume_evaluation:exit", optimizer_updates=completed)
    if state is None:
        rng_states = [None] * world
        dist.all_gather_object(rng_states, _rng_state())
    if rank == 0 and state is None:
        _save_state(
            state_path,
            policy,
            opt,
            tracker,
            evaluator,
            prompt_hash,
            config_hash,
            nonce,
            order,
            refs,
            tau_id,
            tau_motion,
            completed,
            max_update_seconds,
            rng_states,
            cfg.to_dict(),
            topology,
            migrations,
        )
    _barrier("startup_barrier")
    del state
    initial_completed = completed
    while completed < cfg.num_epochs:
        if rank == 0:
            can_start = should_start_update(
                completed, cfg.num_epochs, time.monotonic() - started, max_update_seconds, FLAGS.job_budget_hours * 3600
            )
            if FLAGS.max_updates_this_job and completed - initial_completed >= FLAGS.max_updates_this_job:
                can_start = False
        else:
            can_start = None
        decision = [can_start]
        _trace("update_decision:enter")
        dist.broadcast_object_list(decision, src=0)
        _trace("update_decision:exit")
        if not decision[0]:
            break
        update_start = time.monotonic()
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        assignments = worker_sample_assignments(
            order, completed, rank, world, single_node_smoke=FLAGS.single_node_smoke
        )
        policy.set_adapter("old")
        _trace(f"update_{completed + 1}:sampling:enter")
        tuples = []
        for index, rollout_indices in assignments:
            tuples.extend(
                _sample_local(pipe, roll, reward, judge, cfg, prompts[index], index, nonce, refs, rollout_indices)
            )
        local_rollouts = len(tuples)
        _trace(f"update_{completed + 1}:sampling:exit")
        # VideoReward is not used by the differentiable geometry target. Move its
        # frozen weights off the scorer GPU before WAFT recomputes in backward.
        judge.to("cpu")
        with torch.cuda.device(cfg.judge.device):
            torch.cuda.empty_cache()
        gathered = [None] * world
        dist.all_gather_object(
            gathered,
            [{key: t[key] for key in ("prompt", "prompt_index", "rollout_index", "r", "m", "p_q")} for t in tuples],
        )
        flat = canonical_samples(gathered, order, completed, world, single_node_smoke=FLAGS.single_node_smoke)
        global_rollouts = len(flat)
        sample_seconds = time.monotonic() - update_start
        if rank == 0:
            advantages = tracker.update([entry["prompt"] for entry in flat], np.array([entry["r"] for entry in flat]))
            weights = [
                float(
                    effective_weight(
                        torch.tensor([float(a)]),
                        cfg.train.adv_clip_max,
                        torch.tensor([entry["m"]]),
                        tau_motion,
                        torch.tensor([entry["p_q"]]),
                        cfg.gates.tau_q,
                    )
                )
                for entry, a in zip(flat, advantages)
            ]
            weight_groups = route_sample_weights(flat, weights, gathered)
        else:
            weight_groups = None
        weight_box = [weight_groups]
        dist.broadcast_object_list(weight_box, src=0)
        target_start = time.monotonic()
        _trace(f"update_{completed + 1}:target_gradients:enter")
        local_kept, local_loss = _targets_and_gradients(
            tuples, weight_box[0][rank], roll, reward, policy, opt, cfg, tau_id
        )
        _trace(f"update_{completed + 1}:target_gradients:exit")
        judge.to(cfg.judge.device)
        target_seconds = time.monotonic() - target_start
        optimizer_start = time.monotonic()
        counts = torch.tensor(
            [
                local_kept,
                local_loss,
                sum(t["r"] for t in tuples),
                sum(weight_box[0][rank]),
                sum(t["m"] < tau_motion for t in tuples),
                sum(t["p_q"] < cfg.gates.tau_q for t in tuples),
            ],
            device="cuda:0",
            dtype=torch.float64,
        )
        _trace("update_counts:enter", retained_rollouts=local_kept)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        _trace("update_counts:exit")
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
        _trace("adapter_consistency:enter")
        adapter_hashes = {
            name: adapter_fingerprint(get_peft_model_state_dict(policy, adapter_name=name))
            for name in ("default", "old")
        }
        worker_hashes = [None] * world
        dist.all_gather_object(worker_hashes, adapter_hashes)
        if any(hashes != adapter_hashes for hashes in worker_hashes):
            raise RuntimeError("Worker adapter states differ after the global optimizer update")
        _trace("adapter_consistency:exit", adapter_hashes=adapter_hashes)
        for device in devices:
            torch.cuda.synchronize(device)
        optimizer_seconds = time.monotonic() - optimizer_start
        local_duration = time.monotonic() - update_start
        timings = torch.tensor([local_duration], device="cuda:0")
        dist.all_reduce(timings, op=dist.ReduceOp.MAX)
        phase_times = torch.tensor([sample_seconds, target_seconds, optimizer_seconds], device="cuda:0")
        dist.all_reduce(phase_times, op=dist.ReduceOp.MAX)
        completed += 1
        max_update_seconds = max(max_update_seconds or 0, float(timings.item()))
        peaks = {
            f"gpu/{rank}/{device}/peak_{kind}_bytes": measure(device)
            for device in devices
            for kind, measure in (
                ("allocated", torch.cuda.max_memory_allocated),
                ("reserved", torch.cuda.max_memory_reserved),
            )
        }
        peaks[f"worker/{rank}/host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        peak_list = [None] * world
        dist.all_gather_object(peak_list, peaks)
        if rank == 0:
            record = {
                "optimizer_updates": completed,
                "train/rl_distillation_loss": counts[1].item() / total_kept,
                "train/optimization_loss": counts[1].item() * cfg.train.adv_clip_max / total_kept,
                "grad_norm": float(grad_norm),
                "n_rollouts": global_rollouts,
                "execution/world_size": world,
                "execution/adapters_consistent": True,
                "execution/default_adapter_sha256": adapter_hashes["default"],
                "execution/old_adapter_sha256": adapter_hashes["old"],
                "execution/workers_per_node": topology["workers_per_node"],
                "execution/prompts_per_update": 4,
                "execution/rollouts_per_worker": local_rollouts,
                "execution/single_node_smoke": FLAGS.single_node_smoke,
                "n_kept": total_kept,
                "mean_r": counts[2].item() / global_rollouts,
                "mean_w": counts[3].item() / global_rollouts,
                "frac_motion_masked": counts[4].item() / global_rollouts,
                "frac_quality_masked": counts[5].item() / global_rollouts,
                "tau_id": tau_id,
                "tau_motion": tau_motion,
                "train/update_seconds": float(timings.item()),
                "train/sampling_seconds": float(phase_times[0]),
                "train/target_seconds": float(phase_times[1]),
                "train/optimizer_seconds": float(phase_times[2]),
                **{key: value for group in peak_list for key, value in group.items()},
            }
            videos = []
            if evaluation_due(completed, cfg.num_epochs, cfg.eval_freq, override_freq=FLAGS.monitor_eval_freq):
                metrics, videos = evaluator.evaluate(pipe, policy, roll, reward, judge, completed)
                record.update(metrics)
            _log(record, videos)
            if completed % cfg.save_freq == 0:
                torch.save(
                    get_peft_model_state_dict(policy, adapter_name="default"), run_dir / f"policy_{completed}.pt"
                )
        rng_states = [None] * world
        dist.all_gather_object(rng_states, _rng_state())
        if rank == 0:
            _save_state(
                state_path,
                policy,
                opt,
                tracker,
                evaluator,
                prompt_hash,
                config_hash,
                nonce,
                order,
                refs,
                tau_id,
                tau_motion,
                completed,
                max_update_seconds,
                rng_states,
                cfg.to_dict(),
                topology,
                migrations,
            )
        _barrier(f"update_{completed}:checkpoint_barrier")
    if rank == 0:
        result = {
            "status": "complete" if completed == cfg.num_epochs else "continue",
            "completed_updates": completed,
            "total_updates": cfg.num_epochs,
            "run_dir": str(run_dir),
            "job_id": os.environ["SLURM_JOB_ID"],
            "single_node_smoke": FLAGS.single_node_smoke,
            "topology": topology,
        }
        (run_dir / f"job_{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(result, indent=2) + "\n")
        wandb.finish()
        print(json.dumps(result), flush=True)
    _barrier("shutdown_barrier")
    dist.destroy_process_group()


def _save_state(
    path,
    policy,
    opt,
    tracker,
    evaluator,
    prompt_hash,
    config_hash,
    nonce,
    order,
    refs,
    tau_id,
    tau_motion,
    completed,
    max_update_seconds,
    rng_states,
    training_config,
    topology,
    migrations,
):
    state = {
        "version": 2,
        "training_config": {**training_config, "logdir": portable_path(training_config["logdir"], ROOT)},
        "training_config_hash": training_config_hash(training_config),
        "topology": topology,
        "topology_migrations": migrations,
        "next_update": completed,
        "prompt_hash": prompt_hash,
        "config_hash": config_hash,
        "nonce": nonce,
        "order": order,
        "reference_files": portable_reference_files(refs, ROOT),
        "reference_path_base": "repository",
        "tau_id": tau_id,
        "tau_motion": tau_motion,
        "policy_default": _cpu_state(get_peft_model_state_dict(policy, adapter_name="default")),
        "policy_old": _cpu_state(get_peft_model_state_dict(policy, adapter_name="old")),
        "optimizer": _cpu_state(opt.state_dict()),
        "tracker_stats": tracker.stats,
        "eval_baseline": evaluator.baseline,
        "max_update_seconds": max_update_seconds,
        "rng_states": rng_states,
    }
    state["adapter_fingerprints"] = {
        "default": adapter_fingerprint(state["policy_default"]),
        "old": adapter_fingerprint(state["policy_old"]),
    }
    atomic_checkpoint(path, state)


if __name__ == "__main__":
    app.run(main)
