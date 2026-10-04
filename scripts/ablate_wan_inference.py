#!/usr/bin/env python3
"""Six base-Wan2.2 videos: native UniPC control, fewer steps, and lower resolution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

from diffusionopsd.video.inference_ablation import run_ablation, validate_experiment
from diffusionopsd.video.inference_investigation import file_hash, prepare_output, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/wan22_inference_ablation.json"))
    parser.add_argument("--model", default="${VIDEO_REWARD_CKPT_PATH}/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--prompts", type=Path, default=Path("data/video_motion/test.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vae-device", default="cuda:0")
    parser.add_argument("--reward-device", default="cuda:1")
    args = parser.parse_args(argv)
    args.model = os.path.expandvars(args.model)
    try:
        contents = args.config.read_bytes()
        experiment = validate_experiment(json.loads(contents))
        prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
        if not prompts:
            raise ValueError("Prompt file is empty")
        root = Path(os.environ["VIDEO_REWARD_CKPT_PATH"])
        for path in [Path(args.model) / "model_index.json", *[root / name for name in (
            "depth-anything-3-large-v1.1/model.safetensors", "waft_tar_c_t.pth", "dinov2-base/config.json",
            "VideoReward/model_config.json", "Qwen2-VL-2B-Instruct/config.json", "VideoAlign/inference.py",
        )]]:
            if not path.is_file():
                raise ValueError(f"Required local input missing: {path}")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    args.config = args.config.resolve()
    args.config_sha256 = hashlib.sha256(contents).hexdigest()
    return args, experiment, prompts[0]


def main(argv=None):
    cfg, experiment, prompt = parse_args(argv)
    output = prepare_output(cfg.output_dir)
    run = None
    manifest = {"status": "initializing", "arguments": vars(cfg), "experiment": experiment,
                "prompt": prompt, "expected_videos": 6, "weights": "base_only", "negative_prompt": "",
                "precision": {"transformer": "bfloat16", "text_encoder": "bfloat16", "vae": "bfloat16",
                              "initial_latents": "float32"}}
    try:
        import torch
        import wandb
        from diffusers import UniPCMultistepScheduler, WanPipeline
        from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
        from diffusionopsd.video.geo_reward import GeoReward
        from diffusionopsd.video.quality_judge import load_video_reward
        from diffusionopsd.video.wan_geometry import require_expand_timesteps

        def git(*args):
            result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
            return result.stdout.strip() if result.returncode == 0 else "unavailable"

        manifest.update({
            "git_sha": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "prompt_file_sha256": file_hash(cfg.prompts),
            "packages": {k: importlib.metadata.version(k) for k in ["torch", "diffusers", "transformers", "wandb"]},
            "model_files": {str(p.relative_to(cfg.model)): {"size_bytes": p.stat().st_size,
                            "mtime_ns": p.stat().st_mtime_ns} for p in sorted(Path(cfg.model).rglob("*.safetensors"))},
            "model_config_hashes": {str(p.relative_to(cfg.model)): file_hash(p)
                                    for p in sorted(Path(cfg.model).rglob("*.json"))}, "status": "running",
        })
        write_json(output / "manifest.json", manifest)
        run = wandb.init(project="diffusionopsd", name=f"wan22-ablation-{os.environ.get('SLURM_JOB_ID', output.name)}",
                         dir=str(output), mode="offline",
                         config={**{k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()},
                                 "experiment": experiment, "weights": "base_only", "negative_prompt": "",
                                 "sampler": "native_unipc", "precision": manifest["precision"]})
        pipe = WanPipeline.from_pretrained(cfg.model, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.to(cfg.policy_device)
        pipe.vae.to(cfg.vae_device)
        require_expand_timesteps(pipe, cfg.model)
        if not isinstance(pipe.scheduler, UniPCMultistepScheduler):
            raise ValueError("Expected the checkpoint's UniPC scheduler")
        scheduler_config = dict(pipe.scheduler.config)
        for module in (pipe.transformer, pipe.vae, pipe.text_encoder):
            module.eval().requires_grad_(False)
        reward = GeoReward(load_depth_anything3(cfg.reward_device), load_waft(cfg.reward_device),
                           load_dinov2(cfg.reward_device)).eval().requires_grad_(False)
        judge = load_video_reward(cfg.reward_device, num_frames=8)
        run_ablation(pipe, reward, judge, cfg, experiment, prompt, output,
                     lambda: UniPCMultistepScheduler.from_config(scheduler_config), run)
        run.finish()
        manifest["status"] = "complete"
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        if not (output / "failure.json").exists():
            write_json(output / "failure.json", {"phase": "setup_or_finalization", "error": str(exc),
                                                 "error_type": type(exc).__name__})
        (output / "summary.json").unlink(missing_ok=True)
        manifest["status"] = "failed"
        write_json(output / "manifest.json", manifest)
        if run is not None:
            run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
