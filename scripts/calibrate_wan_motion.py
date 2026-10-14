#!/usr/bin/env python3
# ruff: noqa: E402
"""Estimate a base-model motion floor from representative training prompts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from diffusers import WanPipeline
import wandb

from diffusionopsd.video.estimators import load_waft
from diffusionopsd.video.gates import percentile_threshold
from diffusionopsd.video.inference_investigation import file_hash, prepare_output, write_json
from diffusionopsd.video.motion_calibration import calibration_indices, motion_score
from diffusionopsd.video.training_eval import save_video
from diffusionopsd.video.wan_clean_output import WanRollout
from diffusionopsd.video.wan_geometry import latent_shape_from_pipe, require_expand_timesteps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("wan22_ti2v", "wan_video"), default="wan22_ti2v")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = importlib.import_module(f"config.{args.config}").get_config()
    prompt_file = ROOT / cfg.prompt_fn_kwargs["path"]
    prompts = [line.strip() for line in prompt_file.read_text().splitlines() if line.strip()]
    indices = calibration_indices(len(prompts), args.num_prompts)
    output = prepare_output(args.output_dir)
    manifest = dict(
        status="running",
        job_id=os.environ.get("SLURM_JOB_ID"),
        cluster=os.environ.get("SLURM_CLUSTER_NAME", "unknown"),
        hostname=socket.gethostname(),
        purpose=f"base-motion-calibration-{args.num_prompts}-prompts",
        num_prompts=args.num_prompts,
        prompt_indices=indices,
        prompt_file=str(prompt_file),
        prompt_sha256=file_hash(prompt_file),
        settings=dict(
            width=cfg.video.width,
            height=cfg.video.height,
            num_frames=cfg.video.num_frames,
            num_steps=cfg.sample.num_steps,
            guidance_scale=cfg.sample.guidance_scale,
            query_sigma=cfg.opa.query_sigma,
            quantile=0.10,
            fps=8,
        ),
        statistic="mean adjacent-frame optical-flow L2 magnitude in pixels per frame",
        model=os.path.expandvars(cfg.pretrained.model),
        unchanged_training_threshold=True,
    )
    write_json(output / "manifest.json", manifest)
    run = None
    rows = []
    started = time.monotonic()
    try:
        if torch.cuda.get_device_properties(0).total_memory / 2**30 < 79:
            raise ValueError("This job requires one H100 with at least 79 GiB usable VRAM")
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT", "diffusionopsd"),
            mode="offline",
            dir=str(output),
            config=manifest,
            name=f"job-{manifest['job_id']}-base-motion-calibration-{args.num_prompts}-prompts-{manifest['cluster']}",
        )
        manifest["wandb"] = dict(id=run.id, entity=run.entity, project=run.project)
        write_json(output / "manifest.json", manifest)
        pipe = WanPipeline.from_pretrained(manifest["model"], torch_dtype=torch.bfloat16, local_files_only=True).to(
            "cuda:0"
        )
        require_expand_timesteps(pipe, manifest["model"])
        # No adapter is attached: every calibration video comes from the base model.
        for module in (pipe.transformer, pipe.vae, pipe.text_encoder):
            module.eval().requires_grad_(False)
        roll = WanRollout(pipe, cfg.sample.num_steps, cfg.sample.guidance_scale)
        flow = load_waft("cuda:0").eval().requires_grad_(False)
        torch.cuda.reset_peak_memory_stats(0)
        for index in indices:
            sample_started = time.monotonic()
            prompt = prompts[index]
            seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
            with torch.no_grad():
                pe, ne = pipe.encode_prompt(
                    prompt, negative_prompt="", do_classifier_free_guidance=True, device="cuda:0"
                )[:2]
                noise = torch.randn(
                    latent_shape_from_pipe(pipe, cfg.video.num_frames, cfg.video.height, cfg.video.width),
                    device="cuda:0",
                    dtype=torch.float32,
                    generator=torch.Generator("cuda:0").manual_seed(seed),
                )
                # Match trainer._calibrate's autocast and custom Euler rollout.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    rec = roll.rollout(pe, ne, noise, cfg.opa.query_sigma)
                    clip = roll.decode01(rec.x0)
                    motion = motion_score(clip, flow)
                if not torch.isfinite(clip).all() or not torch.isfinite(motion).all():
                    raise ValueError("Nonfinite calibration output")
                video = f"base_prompt_{index:03d}.mp4"
                save_video(clip[0], output / video, 8)
                torch.cuda.synchronize()
                row = dict(
                    prompt_index=index,
                    prompt=prompt,
                    seed=seed,
                    motion=float(motion.item()),
                    video=video,
                    duration_seconds=time.monotonic() - sample_started,
                )
            with (output / "samples.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(row)
            run.log(
                {
                    "calibration/motion": row["motion"],
                    "calibration/prompt_index": index,
                    "calibration/video": wandb.Video(str(output / video), format="mp4", caption=prompt),
                },
                step=len(rows),
                commit=True,
            )
            print(json.dumps(dict(completed=len(rows), **row)), flush=True)
            del pe, ne, noise, rec, clip, motion
        scores = torch.tensor([row["motion"] for row in rows], dtype=torch.float32)
        summary = dict(
            status="complete",
            num_prompts=len(rows),
            tau_motion=percentile_threshold(scores, 0.10),
            previous_tau_motion=1.127976417541504,
            quantiles={str(q): percentile_threshold(scores, q) for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)},
            mean_motion=float(scores.mean()),
            reconstructed_two_prompt_threshold=percentile_threshold(scores[:2], 0.10),
            fraction_below_previous=float((scores < 1.127976417541504).float().mean()),
            fraction_below_new=float((scores < percentile_threshold(scores, 0.10)).float().mean()),
            duration_seconds=time.monotonic() - started,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        write_json(output / "summary.json", summary)
        manifest["status"] = "complete"
        write_json(output / "manifest.json", manifest)
        run.summary.update(summary)
        for name in ("manifest.json", "summary.json", "samples.jsonl"):
            run.save(str(output / name), base_path=str(output), policy="now")
        run.finish()
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        manifest["status"] = "failed"
        write_json(output / "manifest.json", manifest)
        write_json(output / "failure.json", dict(error=str(exc), completed=len(rows)))
        if run is not None:
            run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
