#!/usr/bin/env python3
"""Compare four text-to-video inference presets on the three saved Step 6 prompts."""

import argparse
import json
import os
from pathlib import Path
import time

DOCS_URL = "https://huggingface.co/docs/diffusers/en/api/pipelines/cosmos3#text-to-video"
MODEL_ID = "nvidia/Cosmos3-Nano"
MODEL_REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
PROMPT_SEEDS = {"train_dolly_0_1": 142, "train_orbit_0_2": 242, "train_orbit_0_1": 445}

# Sigma identifies the historical audit query. Full generation does not substitute a prediction.
PRESETS = {
    "initial_probe": dict(width=384, height=224, num_frames=5, num_inference_steps=8, historical_query_sigma=0.278),
    "validated_replay": dict(width=384, height=224, num_frames=5, num_inference_steps=35, historical_query_sigma=0.60),
    "training_default": dict(width=832, height=480, num_frames=17, num_inference_steps=35, historical_query_sigma=0.278),
    "diffusers_documented": dict(width=1280, height=720, num_frames=189, num_inference_steps=35,
                                 historical_query_sigma=None),
}
COMMON = dict(fps=24.0, guidance_scale=6.0, flow_shift=10.0, use_karras_sigmas=False,
              scheduler="UniPCMultistepScheduler", dtype="bfloat16", negative_prompt=None,
              cpu_offload=True, enable_sound=False, adapters="none", safety_checker_loaded=False)


def load_prompts(path):
    rows = [query["metadata"] for query in json.loads(path.read_text())["queries"]]
    if len(rows) != 3 or {row["prompt_id"]: row["seed"] for row in rows} != PROMPT_SEEDS:
        raise ValueError("Expected the three original prompt/seed pairs from replay job 9659456")
    return [
        {"prompt_id": row["prompt_id"], "prompt": row["prompt"], "seed": row["seed"],
         "conditioning": json.dumps({"description": row["prompt"]})}
        for row in rows
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-report", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    prompts = load_prompts(args.probe_report)
    settings = [{"preset": name, **values, **COMMON, "vae_tiling": name == "diffusers_documented"}
                for name, values in PRESETS.items()]
    if args.check_inputs:
        print(json.dumps({"prompts": prompts, "settings": settings, "expected_videos": 12}, indent=2))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required for generation")

    import diffusers
    from diffusers import Cosmos3OmniPipeline, UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    import torch
    import wandb

    model = Path(os.environ["VIDEO_REWARD_CKPT_PATH"]) / "Cosmos3-Nano"
    if (model / "opsd_revision.txt").read_text().strip() != MODEL_REVISION:
        raise ValueError("Unexpected Cosmos3-Nano checkpoint revision")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    video_dir = run_dir / "videos"
    video_dir.mkdir(exist_ok=False)
    comparisons = []
    video_rows = wandb.Table(columns=["preset", "prompt_id", "seed", "video", "elapsed_seconds"])
    prompt_rows = wandb.Table(columns=list(prompts[0]), data=[list(row.values()) for row in prompts])
    config_rows = wandb.Table(columns=list(settings[0]), data=[list(row.values()) for row in settings])
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    cluster = os.environ.get("SLURM_CLUSTER_NAME", "narval")
    config = dict(model_id=MODEL_ID, model_revision=MODEL_REVISION, diffusers_version=diffusers.__version__,
                  docs_url=DOCS_URL, source_replay_job=9659456, presets=settings, prompts=prompts)
    (run_dir / "inputs.json").write_text(json.dumps(config, indent=2) + "\n")
    with wandb.init(mode="offline", dir=str(run_dir), project=os.environ.get("WANDB_PROJECT", "diffusionopsd"),
                    entity=os.environ.get("WANDB_ENTITY", "cvis_tmu"),
                    name=f"job-{job_id}-cosmos-quality-comparison-{cluster}", config=config) as run:
        run.log({"prompts": prompt_rows, "config_settings": config_rows})
        try:
            pipe = Cosmos3OmniPipeline.from_pretrained(
                model, dtype=torch.bfloat16, local_files_only=True, sound_tokenizer=None,
                safety_checker=None, enable_safety_checker=False,
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config, flow_shift=COMMON["flow_shift"], use_karras_sigmas=False,
            )
            pipe.enable_model_cpu_offload()
            for setting in settings:
                if setting["vae_tiling"]:
                    pipe.vae.enable_tiling()
                kwargs = {key: setting[key] for key in (
                    "height", "width", "num_frames", "num_inference_steps", "guidance_scale", "fps",
                    "negative_prompt", "enable_sound",
                )}
                for prompt in prompts:
                    key = f'{setting["preset"]}--{prompt["prompt_id"]}'
                    print(f"Generating {key}, seed {prompt['seed']}", flush=True)
                    started = time.monotonic()
                    result = pipe(prompt=prompt["conditioning"], **kwargs,
                                  generator=torch.Generator(device="cuda").manual_seed(prompt["seed"]))
                    frames = result.video
                    expected_size = (setting["width"], setting["height"])
                    if len(frames) != setting["num_frames"] or any(frame.size != expected_size for frame in frames):
                        raise RuntimeError(f"Unexpected output geometry for {key}")
                    path = video_dir / f"{key}.mp4"
                    export_to_video(frames, str(path), fps=setting["fps"], macro_block_size=1)
                    del result, frames
                    elapsed = time.monotonic() - started
                    caption = f'{setting["preset"]} | seed {prompt["seed"]} | {prompt["prompt"]}'
                    video_rows.add_data(setting["preset"], prompt["prompt_id"], prompt["seed"],
                                        wandb.Video(str(path), format="mp4", caption=caption), elapsed)
                    comparisons.append({"preset": setting["preset"], "prompt_id": prompt["prompt_id"],
                                        "seed": prompt["seed"], "video": str(path), "elapsed_seconds": elapsed})
                    (run_dir / "videos.json").write_text(json.dumps(comparisons, indent=2) + "\n")
        finally:
            run.log({"videos": video_rows, "generated_video_count": len(comparisons)})
    if len(comparisons) != 12:
        raise RuntimeError(f"Expected 12 videos, got {len(comparisons)}")


if __name__ == "__main__":
    main()
