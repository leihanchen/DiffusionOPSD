#!/usr/bin/env python3
"""Compare pretrained/trained Wan with custom Euler and native UniPC inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

from diffusionopsd.video.inference_investigation import (
    file_hash,
    prepare_output,
    run_investigation,
    validate_adapter,
    write_json,
)


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser.add_argument("--config", type=Path, help="Inference JSON preset; explicit CLI options override it")
    selected, _ = config_parser.parse_known_args(argv)
    p = argparse.ArgumentParser(description=__doc__, parents=[config_parser], allow_abbrev=False)
    p.add_argument("--policy", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model", default="${VIDEO_REWARD_CKPT_PATH}/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--prompts", type=Path, default=Path("data/video_motion/test.txt"))
    p.add_argument("--num-prompts", type=int, default=2)
    p.add_argument("--seed-offsets", type=int, nargs="+", default=[0, 1000])
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=17)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--policy-device", default="cuda:0")
    p.add_argument("--reward-device", default="cuda:1")
    p.add_argument("--vae-device", default="cuda:2")
    config_sha256 = None
    if selected.config is not None:
        try:
            contents = selected.config.read_bytes()
            overrides = json.loads(contents)
        except (OSError, ValueError) as exc:
            p.error(f"Cannot read inference config: {exc}")
        types = dict(height=int, width=int, num_frames=int, num_steps=int, guidance_scale=(int, float))
        if not isinstance(overrides, dict):
            p.error("Inference config must be a JSON object")
        unknown = overrides.keys() - types.keys()
        if unknown:
            p.error(f"Unknown inference config keys: {', '.join(sorted(unknown))}")
        for key, value in overrides.items():
            if isinstance(value, bool) or not isinstance(value, types[key]):
                p.error(f"Invalid numeric type for inference config key: {key}")
        p.set_defaults(**overrides)
        config_sha256 = hashlib.sha256(contents).hexdigest()
    a = p.parse_args(argv)
    a.config = a.config.resolve() if a.config is not None else None
    a.config_sha256 = config_sha256
    a.model = os.path.expandvars(a.model)
    if any(v <= 0 for v in [a.num_prompts, a.height, a.width, a.num_frames, a.num_steps, a.fps]):
        p.error("Counts, dimensions, steps, and FPS must be positive")
    if a.height % 32 or a.width % 32 or (a.num_frames - 1) % 4:
        p.error("Wan2.2 requires spatial multiples of 32 and 4k+1 frames")
    if a.num_frames < 2 or not 1 < a.guidance_scale < float("inf"):
        p.error("Use at least 5 frames and finite guidance > 1 for matched CFG paths")
    if min(a.seed_offsets) < 0 or len(set(a.seed_offsets)) != len(a.seed_offsets):
        p.error("Seed offsets must be distinct and nonnegative")
    if max(a.seed_offsets) + a.num_prompts - 1 >= 2**63:
        p.error("Seeds must fit in a signed 64-bit integer")
    for path in (a.policy, a.prompts, Path(a.model) / "model_index.json"):
        if not path.is_file():
            p.error(f"Required local file missing: {path}")
    if "VIDEO_REWARD_CKPT_PATH" not in os.environ:
        p.error("Set VIDEO_REWARD_CKPT_PATH to the local scorer tree")
    root = Path(os.environ["VIDEO_REWARD_CKPT_PATH"])
    for relative in ("depth-anything-3-large-v1.1/model.safetensors", "waft_tar_c_t.pth",
                     "dinov2-base/config.json", "VideoReward/model_config.json",
                     "Qwen2-VL-2B-Instruct/config.json", "VideoAlign/inference.py"):
        if not (root / relative).is_file():
            p.error(f"Required local scorer input missing: {root / relative}")
    return a


def main(argv=None):
    cfg = parse_args(argv)
    prompts = [s.strip() for s in cfg.prompts.read_text().splitlines() if s.strip()]
    if len(prompts) < cfg.num_prompts:
        raise ValueError("Prompt file contains fewer prompts than requested")
    prompts = prompts[:cfg.num_prompts]
    output = prepare_output(cfg.output_dir)
    run = None
    try:
        import torch
        from diffusers import UniPCMultistepScheduler, WanPipeline
        from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
        from diffusionopsd.video.geo_reward import GeoReward
        from diffusionopsd.video.quality_judge import load_video_reward
        from diffusionopsd.video.wan_geometry import require_expand_timesteps
        from diffusionopsd.video.wan_policy import attach_lora
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        import wandb

        def git(*args):
            result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
            return result.stdout.strip() if result.returncode == 0 else "unavailable"

        manifest = {
            "arguments": vars(cfg), "prompts": prompts,
            "seeds": [[i + offset for offset in cfg.seed_offsets] for i in range(len(prompts))],
            "checkpoint_sha256": file_hash(cfg.policy), "prompt_file_sha256": file_hash(cfg.prompts),
            "git_sha": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "packages": {k: importlib.metadata.version(k) for k in ["torch", "diffusers", "transformers", "peft", "wandb"]},
            "model_files": {str(p.relative_to(cfg.model)): {"size_bytes": p.stat().st_size,
                            "mtime_ns": p.stat().st_mtime_ns} for p in sorted(Path(cfg.model).rglob("*.safetensors"))},
            "model_config_hashes": {str(p.relative_to(cfg.model)): file_hash(p)
                                    for p in sorted(Path(cfg.model).rglob("*.json"))},
            "precision": {"transformer": "bfloat16", "text_encoder": "bfloat16", "vae": "bfloat16",
                          "initial_latents": "float32"},
            "initial_latents": [], "expected_videos": len(prompts) * len(cfg.seed_offsets) * 4,
            "status": "running",
        }
        write_json(output / "manifest.json", manifest)
        run = wandb.init(project="diffusionopsd", name=f"wan22-inference-{os.environ.get('SLURM_JOB_ID', output.name)}",
                         dir=str(output), mode="offline", config={k: str(v) if isinstance(v, Path) else v
                                                                for k, v in vars(cfg).items()})
        pipe = WanPipeline.from_pretrained(cfg.model, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.to(cfg.policy_device)
        pipe.vae.to(cfg.vae_device)
        require_expand_timesteps(pipe, cfg.model)
        if not isinstance(pipe.scheduler, UniPCMultistepScheduler):
            raise ValueError("Expected the checkpoint's UniPC scheduler")
        scheduler_config = dict(pipe.scheduler.config)
        policy = attach_lora(pipe.transformer, None)
        state = torch.load(cfg.policy, map_location="cpu", weights_only=True)
        validate_adapter(get_peft_model_state_dict(policy, adapter_name="default"), state)
        set_peft_model_state_dict(policy, state, adapter_name="default")
        del state
        pipe.transformer = policy
        for module in (policy, pipe.vae, pipe.text_encoder):
            module.eval().requires_grad_(False)
        reward = GeoReward(load_depth_anything3(cfg.reward_device), load_waft(cfg.reward_device),
                           load_dinov2(cfg.reward_device)).eval().requires_grad_(False)
        judge = load_video_reward(cfg.reward_device, num_frames=8)
        run_investigation(pipe, reward, judge, cfg, prompts, output,
                          lambda: UniPCMultistepScheduler.from_config(scheduler_config), run)
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["status"] = "complete"
        write_json(output / "manifest.json", manifest)
        run.finish()
    except Exception as exc:
        if not (output / "failure.json").exists():
            write_json(output / "failure.json", {"phase": "setup_or_finalization", "error": str(exc),
                                                 "error_type": type(exc).__name__})
        (output / "summary.json").unlink(missing_ok=True)
        if (output / "manifest.json").exists():
            manifest = json.loads((output / "manifest.json").read_text())
            manifest["status"] = "failed"
            write_json(output / "manifest.json", manifest)
        if run is not None:
            run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
