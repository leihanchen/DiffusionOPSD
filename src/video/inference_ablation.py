"""Base Wan native-UniPC ablations; independent of OPSD training."""

from __future__ import annotations

import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image

from diffusionopsd.video.inference_investigation import SCORES, tensor_hash, write_json
from diffusionopsd.video.training_eval import save_video
from diffusionopsd.video.wan_clean_output import WanRollout
from diffusionopsd.video.wan_geometry import latent_shape_from_pipe


CONDITIONS = ("control", "steps30", "resolution480")


def euler_latents(roll, pe, ne, initial):
    """Match the trainer's bf16 model forwards while keeping paired noise in fp32."""
    context = torch.autocast("cuda", dtype=torch.bfloat16) if initial.is_cuda else nullcontext()
    with context:
        return roll.rollout(pe.clone(), ne.clone(), initial.clone(), 0.278).x0


def validate_experiment(config):
    """Reject accidental changes to the agreed one-factor comparisons."""
    expected = {
        "num_frames": 121,
        "guidance_scale": 5.0,
        "fps": 8,
        "seeds": [0, 1000],
        "conditions": {
            "control": {"height": 704, "width": 1280, "num_steps": 50},
            "steps30": {"height": 704, "width": 1280, "num_steps": 30},
            "resolution480": {"height": 480, "width": 832, "num_steps": 50},
        },
    }
    scheduler_expected = {
        **expected,
        "seeds": [1],
        "conditions": {
            f"{sampler}_{name}": {**params, "sampler": sampler}
            for name, params in expected["conditions"].items()
            for sampler in ("euler", "native")
        },
    }
    duration_expected = {
        **scheduler_expected,
        "conditions": {
            **scheduler_expected["conditions"],
            **{
                f"euler17_{name}": {**params, "sampler": "euler", "num_frames": 17}
                for name, params in expected["conditions"].items()
            },
        },
    }
    if config not in (expected, scheduler_expected, duration_expected):
        raise ValueError("Config must specify an agreed base-model ablation")
    return config


def summarize(rows, seeds, conditions=CONDITIONS):
    grouped = {seed: {} for seed in seeds}
    for row in rows:
        seed, condition = row["seed"], row["condition"]
        if seed not in grouped or condition not in conditions or condition in grouped[seed]:
            raise ValueError("Unexpected or duplicate ablation sample")
        grouped[seed][condition] = row
    if not grouped or any(set(group) != set(conditions) for group in grouped.values()):
        raise ValueError("Incomplete ablation")
    metrics = (
        *SCORES,
        *(
            key
            for key in ("compute_seconds", "duration_seconds", "playback_duration_seconds")
            if all(key in row for row in rows)
        ),
    )
    if conditions == CONDITIONS:
        comparisons = [(name, name, "control") for name in CONDITIONS[1:]]
    else:
        comparisons = [(f"native_{name}-euler_{name}", f"native_{name}", f"euler_{name}") for name in CONDITIONS]
        comparisons += [
            (f"{sampler}_{name}-{sampler}_control", f"{sampler}_{name}", f"{sampler}_control")
            for sampler in ("euler", "native")
            for name in CONDITIONS[1:]
        ]
    comparisons += [
        (f"euler17_{name}-euler_{name}", f"euler17_{name}", f"euler_{name}")
        for name in CONDITIONS
        if f"euler17_{name}" in conditions
    ]
    pairs = [
        {
            "seed": seed,
            "differences": {
                label: {metric: group[left][metric] - group[right][metric] for metric in metrics}
                for label, left, right in comparisons
            },
        }
        for seed, group in grouped.items()
    ]
    means = {
        name: {metric: sum(g[name][metric] for g in grouped.values()) / len(grouped) for metric in metrics}
        for name in conditions
    }
    differences = {
        name: {metric: sum(p["differences"][name][metric] for p in pairs) / len(pairs) for metric in metrics}
        for name, _, _ in comparisons
    }
    return {
        "status": "complete",
        "num_videos": len(rows),
        "means": means,
        "differences": differences,
        "pairs": pairs,
        "interpretation": f"One prompt, {len(seeds)} seed(s); descriptive results. Resolution and frame count change "
        "noise shape; pixel motion scales with resolution. Duration comparisons use the same seed, not identical noise.",
    }


@torch.no_grad()
def run_ablation(pipe, reward, judge, cfg, experiment, prompt, output, scheduler_factory, wandb_run):
    if getattr(pipe.transformer, "peft_config", None):
        raise ValueError("Ablation requires the base transformer without LoRA adapters")
    output = Path(output)
    rows = []
    devices = {torch.device(d) for d in (cfg.policy_device, cfg.reward_device, cfg.vae_device)}
    cuda_devices = sorted((d for d in devices if d.type == "cuda"), key=str)

    def sync():
        for device in cuda_devices:
            torch.cuda.synchronize(device)

    current = {"phase": "prompt_encoding"}
    try:
        pe, ne = pipe.encode_prompt(
            prompt, negative_prompt="", do_classifier_free_guidance=True, device=cfg.policy_device
        )[:2]
        for seed in experiment["seeds"]:
            noises = {}
            for condition in experiment["conditions"]:
                params = experiment["conditions"][condition]
                height, width = params["height"], params["width"]
                frames, steps = params.get("num_frames", experiment["num_frames"]), params["num_steps"]
                current = {"phase": "generation", "seed": seed, "condition": condition}
                print(json.dumps({"event": "condition_start", **current}), flush=True)
                shape = latent_shape_from_pipe(pipe, frames, height, width)
                if shape not in noises:
                    noise = torch.randn(
                        shape,
                        device=cfg.policy_device,
                        dtype=torch.float32,
                        generator=torch.Generator(cfg.policy_device).manual_seed(seed),
                    )
                    noise_file = (
                        f"seed_{seed}_{width}x{height}_{frames}frames_initial.pt"
                        if "num_frames" in params
                        else f"seed_{seed}_{width}x{height}_initial.pt"
                    )
                    torch.save(noise.cpu(), output / noise_file)
                    noises[shape] = noise, noise_file
                initial, noise_file = noises[shape]
                hashes = {
                    "initial_latent_hash": tensor_hash(initial),
                    "prompt_embedding_hash": tensor_hash(pe),
                    "negative_embedding_hash": tensor_hash(ne),
                }
                pipe.scheduler = scheduler_factory()
                sync()
                for device in cuda_devices:
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                sampler = params.get("sampler", "native")
                if sampler == "euler":
                    latents = euler_latents(WanRollout(pipe, steps, experiment["guidance_scale"]), pe, ne, initial)
                else:
                    latents = pipe(
                        prompt_embeds=pe.clone(),
                        negative_prompt_embeds=ne.clone(),
                        latents=initial.clone(),
                        height=height,
                        width=width,
                        num_frames=frames,
                        num_inference_steps=steps,
                        guidance_scale=experiment["guidance_scale"],
                        output_type="latent",
                        return_dict=False,
                    )[0]
                if latents.shape != initial.shape or not torch.isfinite(latents).all():
                    raise ValueError("Invalid generated latents")
                current["phase"] = "decode"
                clip = WanRollout(pipe, steps, experiment["guidance_scale"]).decode01(latents)
                if clip.shape != (1, frames, 3, height, width) or not torch.isfinite(clip).all():
                    raise ValueError("Invalid decoded video")
                current["phase"] = "scoring"
                geometry = reward(clip.to(cfg.reward_device))
                quality = judge.score(clip[0].to(cfg.reward_device), prompt)
                scores = {k: float(getattr(geometry, k)) for k in SCORES[:5]}
                scores.update({k: float(quality[k]) for k in SCORES[5:]})
                if not all(math.isfinite(v) for v in scores.values()):
                    raise ValueError("Nonfinite scores")
                sync()
                compute_seconds = time.perf_counter() - started
                current["phase"] = "artifacts"
                name = f"seed_{seed}_{condition}"
                video = f"{name}.mp4"
                save_video(clip[0], output / video, experiment["fps"])
                for label, frame in [("first", 0), ("middle", frames // 2), ("last", frames - 1)]:
                    pixels = (clip[0, frame] * 255).round().byte().permute(1, 2, 0).cpu().numpy()
                    Image.fromarray(pixels).save(output / f"{name}_{label}.png")
                torch.save(latents.cpu(), output / f"{name}_final.pt")
                row = {
                    "condition": condition,
                    "seed": seed,
                    "prompt": prompt,
                    **params,
                    "num_frames": frames,
                    "guidance_scale": experiment["guidance_scale"],
                    "playback_duration_seconds": frames / experiment["fps"],
                    "fps": experiment["fps"],
                    "video": video,
                    "initial_latents": noise_file,
                    **hashes,
                    **scores,
                    "compute_seconds": compute_seconds,
                    "duration_seconds": time.perf_counter() - started,
                }
                for device in cuda_devices:
                    row[f"{device}_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    row[f"{device}_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                trace = {
                    "sampler": sampler,
                    "scheduler_class": type(pipe.scheduler).__name__,
                    "timesteps": pipe.scheduler.timesteps.float().cpu().tolist(),
                    "sigmas": pipe.scheduler.sigmas.float().cpu().tolist(),
                }
                write_json(output / f"{name}.json", {**row, "trace": trace})
                with (output / "samples.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                with (output / "samples.csv").open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    if not rows:
                        writer.writeheader()
                    writer.writerow(row)
                rows.append(row)
                current["phase"] = "wandb"
                import wandb

                caption = (
                    f"base Wan2.2 | {sampler} | {condition} | seed={seed} | {width}x{height} | "
                    f"steps={steps} | frames={frames} | {prompt}"
                )
                wandb_run.log(
                    {
                        "sample_index": len(rows),
                        "condition": condition,
                        "seed": seed,
                        "sampler": sampler,
                        "num_frames": frames,
                        "num_steps": steps,
                        "height": height,
                        "width": width,
                        "runtime/seconds_per_video": row["duration_seconds"],
                        "memory/peak_allocated_gib": max(
                            (v for k, v in row.items() if k.endswith("peak_allocated_bytes")), default=0
                        )
                        / 2**30,
                        "memory/peak_reserved_gib": max(
                            (v for k, v in row.items() if k.endswith("peak_reserved_bytes")), default=0
                        )
                        / 2**30,
                        f"ablation/videos/{condition}": wandb.Video(str(output / video), format="mp4", caption=caption),
                    },
                    step=len(rows),
                    commit=True,
                )
                print(json.dumps({"event": "condition_complete", **row}), flush=True)
                del latents, clip, geometry, quality
        current = {"phase": "summary"}
        summary = summarize(rows, experiment["seeds"], tuple(experiment["conditions"]))
        score_columns = list(SCORES)
        sample_columns = [
            "condition",
            "seed",
            "num_frames",
            "num_steps",
            "width",
            "height",
            *score_columns,
            "compute_seconds",
            "duration_seconds",
            "playback_duration_seconds",
        ]
        samples = wandb.Table(columns=sample_columns, data=[[row[column] for column in sample_columns] for row in rows])
        metric_columns = list(next(iter(summary["means"].values())))
        tables = {"ablation/samples": samples}
        for category in ("means", "differences"):
            tables[f"ablation/{category}"] = wandb.Table(
                columns=["condition_or_comparison", *metric_columns],
                data=[[name, *[metrics[k] for k in metric_columns]] for name, metrics in summary[category].items()],
            )
        wandb_run.log(tables, step=len(rows) + 1, commit=True)
        wandb_run.summary["ablation/num_videos"] = len(rows)
        wandb_run.summary["ablation/num_comparisons"] = len(summary["differences"])
        write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            output / "failure.json",
            {**current, "completed": len(rows), "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
