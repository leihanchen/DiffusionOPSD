"""Standalone paired inference experiments; never imported by OPSD training."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import resource
import shutil
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from diffusionopsd.video.eval_table import success_table
from diffusionopsd.video.quality_judge import gap_to_probability, quality_gap
from diffusionopsd.video.training_eval import save_video
from diffusionopsd.video.wan_clean_output import WanRollout
from diffusionopsd.video.wan_geometry import latent_shape_from_pipe
from PIL import Image

CONDITIONS = {"A": (False, "euler"), "B": (False, "native"), "C": (True, "euler"), "D": (True, "native")}
SCORES = ("geo", "rigid", "dino", "s_id", "motion", "VQ", "MQ", "TA", "Overall")


def selected_conditions(sampler="both"):
    if sampler not in ("both", "euler", "native"):
        raise ValueError(f"Unknown sampler: {sampler}")
    return tuple(c for c, (_, name) in CONDITIONS.items() if sampler == "both" or name == sampler)


def load_adapter_checkpoint(path, checkpoint_format="adapter", expected_update=None):
    """Full training state uses pickle: only opt in for a trusted local checkpoint."""
    if checkpoint_format not in ("adapter", "training"):
        raise ValueError("Checkpoint format must be adapter or training")
    if expected_update is not None and checkpoint_format != "training":
        raise ValueError("An expected update requires a training checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=checkpoint_format == "adapter")
    metadata = {"format": checkpoint_format}
    if checkpoint_format == "training":
        if not isinstance(state, dict) or not isinstance(state.get("policy_default"), dict):
            raise ValueError("Training checkpoint must contain policy_default adapter weights")
        update = state.get("next_update")
        if type(update) is not int or update < 0:
            raise ValueError("Training checkpoint must contain a nonnegative integer next_update")
        if expected_update is not None and update != expected_update:
            raise ValueError(f"Expected checkpoint update {expected_update}, got {update}")
        metadata.update(adapter_key="policy_default", next_update=update)
        state = state["policy_default"]  # Release optimizer, EMA, and RNG state immediately.
    if not isinstance(state, dict):
        raise ValueError("Adapter weights must be a dictionary")
    return state, metadata


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint_list(path):
    """Load an explicit list of trusted local training checkpoints, never discover implicitly."""
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Checkpoint manifest must be a nonempty list")
    states, metadata = {}, []
    for entry in entries:
        label = entry["label"]
        if not isinstance(label, str) or not label.isalnum() or label == "base" or label in states:
            raise ValueError("Checkpoint labels must be unique alphanumeric strings other than base")
        if type(entry.get("expected_update")) is not int or entry["expected_update"] < 0:
            raise ValueError("Checkpoint entry requires an expected update")
        state, info = load_adapter_checkpoint(entry["path"], "training", entry["expected_update"])
        states[label] = state
        metadata.append({**info, "label": label, "path": entry["path"], "sha256": file_hash(entry["path"])})
    return states, metadata


def tensor_hash(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def prepare_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_adapter(expected, actual):
    if expected.keys() != actual.keys():
        raise ValueError("Checkpoint adapter keys do not match the Wan LoRA configuration")
    for key in expected:
        if not isinstance(actual[key], torch.Tensor) or expected[key].shape != actual[key].shape:
            raise ValueError(f"Checkpoint adapter shape mismatch: {key}")
        if not torch.isfinite(actual[key]).all():
            raise ValueError(f"Nonfinite checkpoint adapter: {key}")


@contextmanager
def adapter_state(policy, trained):
    original = list(policy.active_adapters)
    flags = {p: p.requires_grad for p in policy.parameters()}
    try:
        policy.set_adapter("default")
        policy.requires_grad_(False)
        with nullcontext() if trained else policy.disable_adapter():
            yield
    finally:
        policy.set_adapter(original[0] if len(original) == 1 else original)
        for parameter, flag in flags.items():
            parameter.requires_grad_(flag)


def generate_latents(pipe, roll, cfg, condition, initial, pe, ne, scheduler_factory):
    """Pair on exact input noise/embeddings; retain each path's native numerical choices."""
    trained, sampler = CONDITIONS[condition]
    pipe.scheduler = scheduler_factory()
    with adapter_state(pipe.transformer, trained), torch.no_grad():
        if sampler == "euler":
            context = torch.autocast("cuda", dtype=torch.bfloat16) if initial.is_cuda else nullcontext()
            with context:
                result = roll.rollout(pe.clone(), ne.clone(), initial.clone(), 0.278)
            latents = result.x0
        else:
            latents = pipe(
                prompt_embeds=pe.clone(),
                negative_prompt_embeds=ne.clone(),
                latents=initial.clone(),
                height=cfg.height,
                width=cfg.width,
                num_frames=cfg.num_frames,
                num_inference_steps=cfg.num_steps,
                guidance_scale=cfg.guidance_scale,
                output_type="latent",
                return_dict=False,
            )[0]
    trace = {
        "sampler": sampler,
        "scheduler_class": type(pipe.scheduler).__name__,
        "scheduler_timesteps": pipe.scheduler.timesteps.detach().float().cpu().tolist(),
        "sigmas": pipe.scheduler.sigmas.detach().float().cpu().tolist(),
        "model_timesteps": ((pipe.scheduler.sigmas[:-1] * 1000) if sampler == "euler" else pipe.scheduler.timesteps)
        .detach()
        .float()
        .cpu()
        .tolist(),
        "input_latent_dtype": str(initial.dtype),
        "output_latent_dtype": str(latents.dtype),
        "transformer_input_cast": "autocast BF16" if sampler == "euler" else "native transformer dtype",
    }
    if latents.shape != initial.shape or not torch.isfinite(latents).all():
        raise ValueError(f"Invalid output latents for condition {condition}")
    return latents, trace


def summarize(rows, conditions=tuple(CONDITIONS), checkpoint_comparisons=None):
    grouped = {}
    for row in rows:
        key = (row["prompt_index"], row["seed"])
        group = grouped.setdefault(key, {})
        if row["condition"] in group:
            raise ValueError("Duplicate condition")
        group[row["condition"]] = row
    if not grouped or any(set(g) != set(conditions) for g in grouped.values()):
        raise ValueError("Cannot summarize incomplete condition pairs")
    comparisons = [
        (name, left, right)
        for name, left, right in [("B-A", "B", "A"), ("C-A", "C", "A"), ("D-B", "D", "B")]
        if left in conditions and right in conditions
    ]
    if checkpoint_comparisons is not None:
        comparisons = checkpoint_comparisons
    training_pairs = (
        comparisons
        if checkpoint_comparisons is not None
        else [pair for pair in comparisons if pair[0] in ("C-A", "D-B")]
    )
    interaction = set(conditions) == set(CONDITIONS)
    paired = []
    for (index, seed), group in grouped.items():
        record = {"prompt_index": index, "seed": seed, "differences": {}, "training": {}}
        for name, left, right in comparisons:
            record["differences"][name] = {k: group[left][k] - group[right][k] for k in SCORES}
        if interaction:
            record["differences"]["interaction"] = {
                k: record["differences"]["D-B"][k] - record["differences"]["C-A"][k] for k in SCORES
            }
        for name, trained, base in training_pairs:
            pq = gap_to_probability(quality_gap(group[trained], group[base]))
            record["training"][name] = {"p_q": pq, **success_table(group[base], {**group[trained], "p_q": pq})}
        paired.append(record)

    def prompt_mean(items, value):
        per_prompt = {}
        for item in items:
            per_prompt.setdefault(item["prompt_index"], []).append(value(item))
        return sum(sum(v) / len(v) for v in per_prompt.values()) / len(per_prompt)

    means = {
        c: {k: prompt_mean([r for r in rows if r["condition"] == c], lambda r: r[k]) for k in SCORES}
        for c in conditions
    }
    differences = {}
    for name in [c[0] for c in comparisons] + (["interaction"] if interaction else []):
        differences[name] = {k: prompt_mean(paired, lambda r: r["differences"][name][k]) for k in SCORES}
    training = {}
    for name, trained, base in training_pairs:
        pq = prompt_mean(paired, lambda r: r["training"][name]["p_q"])
        training[name] = {"p_q": pq, **success_table(means[base], {**means[trained], "p_q": pq})}
    return {
        "status": "complete",
        "num_videos": len(rows),
        "means": means,
        "differences": differences,
        "training": training,
        "pairs": paired,
        "interpretation": "Descriptive paired generation-path comparisons; no significance claim.",
    }


def prompt_strata(rows, conditions, comparisons):
    """Separate repeatedly monitored prompts from earlier screening and remaining prompts."""
    result = {}
    for name, start, stop in (("prompts_1_2", 0, 2), ("prompts_3_16", 2, 16), ("prompts_17_64", 16, 64)):
        selected = [r for r in rows if start <= r["prompt_index"] < stop]
        if selected:
            result[name] = {
                "num_prompts": len({r["prompt_index"] for r in selected}),
                **summarize(selected, conditions, comparisons),
            }
    return result


def verify_video(path, num_frames, width, height, fps):
    """Decode the exported file so completeness does not rely on file size alone."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(
        str(path), input_params=["-threads", "1"], output_params=["-threads", "1", "-filter_threads", "1"]
    )
    try:
        metadata = next(reader)
        if tuple(metadata["size"]) != (width, height) or not math.isclose(metadata["fps"], fps, abs_tol=0.01):
            raise ValueError(f"Unexpected video metadata for {path}: {metadata}")
        count = sum(1 for _ in reader)
        if count != num_frames:
            raise ValueError(f"Expected {num_frames} frames in {path}, got {count}")
    finally:
        reader.close()


def write_review(output, rows, seed=2026, conditions=tuple(CONDITIONS), checkpoint_comparisons=None):
    """Copy anonymously named media; keep the unblinding map outside the review folder."""
    review = output / "review"
    review.mkdir()
    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    mapping, entries = {}, []
    for i, row in enumerate(ordered):
        label = f"clip_{i + 1:03d}"
        shutil.copyfile(output / row["video"], review / f"{label}.mp4")
        mapping[label] = {k: row[k] for k in ("condition", "prompt_index", "seed")}
        entries.append({"clip_id": label, "prompt": row["prompt"], "video": f"{label}.mp4"})
    write_json(review / "manifest.json", entries)
    write_json(output / "review_condition_map.json", mapping)
    with (review / "ratings.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "clarity", "prompt_adherence", "motion", "temporal_artifacts", "notes"])
        writer.writerows([[e["clip_id"], "", "", "", "", ""] for e in entries])
    inverse = {(r["prompt_index"], r["seed"], r["condition"]): label for label, r in mapping.items()}
    pairs = []
    for index, seed_value in sorted({(r["prompt_index"], r["seed"]) for r in rows}):
        review_pairs = (
            [(base, trained) for _, trained, base in checkpoint_comparisons]
            if checkpoint_comparisons is not None
            else [("A", "B"), ("A", "C"), ("B", "D")]
        )
        for a, b in review_pairs:
            if a not in conditions or b not in conditions:
                continue
            labels = [inverse[index, seed_value, a], inverse[index, seed_value, b]]
            rng.shuffle(labels)
            pairs.append(labels)
    rng.shuffle(pairs)
    with (review / "preferences.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left_clip", "right_clip", "preferred_clip_or_tie", "reason"])
        writer.writerows([[*pair, "", ""] for pair in pairs])


def run_investigation(
    pipe, reward, judge, cfg, prompts, output, scheduler_factory, wandb_run=None, checkpoint_states=None
):
    rows = []
    conditions = selected_conditions(getattr(cfg, "sampler", "both"))
    comparisons = None
    if checkpoint_states is not None:
        conditions = ("base", *checkpoint_states)
        comparisons = [(f"{label}-base", label, "base") for label in checkpoint_states]
        if getattr(cfg, "compare_checkpoints", False):
            labels = list(checkpoint_states)
            comparisons += [
                (f"{later}-{earlier}", later, earlier) for i, later in enumerate(labels) for earlier in labels[:i]
            ]
    media = {}
    roll = WanRollout(pipe, cfg.num_steps, cfg.guidance_scale)
    devices = sorted({torch.device(x) for x in (cfg.policy_device, cfg.reward_device, cfg.vae_device)}, key=str)
    cuda_devices = [d for d in devices if d.type == "cuda"]

    def sync():
        for device in cuda_devices:
            torch.cuda.synchronize(device)

    current = {"phase": "initialization"}
    try:
        with torch.no_grad():
            for index, prompt in enumerate(prompts, start=getattr(cfg, "prompt_start", 0)):
                current = {"phase": "prompt_encoding", "prompt_index": index}
                pe, ne = pipe.encode_prompt(
                    prompt, negative_prompt="", do_classifier_free_guidance=True, device=cfg.policy_device
                )[:2]
                shape = latent_shape_from_pipe(pipe, cfg.num_frames, cfg.height, cfg.width)
                for offset in cfg.seed_offsets:
                    seed = index + offset
                    pair_id = f"prompt_{index:03d}_seed_{seed}"
                    initial = torch.randn(
                        shape,
                        device=cfg.policy_device,
                        dtype=torch.float32,
                        generator=torch.Generator(cfg.policy_device).manual_seed(seed),
                    )
                    torch.save(initial.cpu(), output / f"{pair_id}_initial.pt")
                    hashes = {
                        "initial_latent_hash": tensor_hash(initial),
                        "prompt_embedding_hash": tensor_hash(pe),
                        "negative_embedding_hash": tensor_hash(ne),
                    }
                    manifest_path = output / "manifest.json"
                    if manifest_path.exists():
                        manifest = json.loads(manifest_path.read_text())
                        manifest["initial_latents"].append(
                            {
                                "file": f"{pair_id}_initial.pt",
                                "prompt_index": index,
                                "seed": seed,
                                **hashes,
                                "file_sha256": file_hash(output / f"{pair_id}_initial.pt"),
                            }
                        )
                        write_json(manifest_path, manifest)
                    for condition in conditions:
                        current = {"phase": "generation", "prompt_index": index, "seed": seed, "condition": condition}
                        print(json.dumps({"event": "condition_start", **current}), flush=True)
                        sync()
                        for device in cuda_devices:
                            torch.cuda.reset_peak_memory_stats(device)
                        started = time.perf_counter()
                        generation_condition = condition
                        if checkpoint_states is not None:
                            from peft import set_peft_model_state_dict

                            if condition != "base":
                                set_peft_model_state_dict(
                                    pipe.transformer, checkpoint_states[condition], adapter_name="default"
                                )
                            if cfg.sampler == "native":
                                generation_condition = "B" if condition == "base" else "D"
                            else:
                                generation_condition = "A" if condition == "base" else "C"
                        latents, trace = generate_latents(
                            pipe, roll, cfg, generation_condition, initial, pe, ne, scheduler_factory
                        )
                        clip = roll.decode01(latents)
                        if (
                            clip.shape != (1, cfg.num_frames, 3, cfg.height, cfg.width)
                            or not torch.isfinite(clip).all()
                        ):
                            raise ValueError("Invalid decoded video")
                        geometry = reward(clip.to(cfg.reward_device))
                        quality = judge.score(clip[0].to(cfg.reward_device), prompt)
                        scores = {k: float(getattr(geometry, k)) for k in SCORES[:5]}
                        scores.update({k: float(quality[k]) for k in SCORES[5:]})
                        if not all(math.isfinite(v) for v in scores.values()):
                            raise ValueError("Nonfinite evaluation score")
                        sync()
                        compute_seconds = time.perf_counter() - started
                        name = f"{pair_id}_{condition}"
                        video = f"{name}.mp4"
                        save_video(clip[0], output / video, cfg.fps)
                        if getattr(cfg, "verify_videos", False):
                            verify_video(output / video, cfg.num_frames, cfg.width, cfg.height, cfg.fps)
                        for label, frame in [
                            ("first", 0),
                            ("middle", cfg.num_frames // 2),
                            ("last", cfg.num_frames - 1),
                        ]:
                            pixels = (clip[0, frame].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
                            Image.fromarray(pixels).save(output / f"{name}_{label}.png")
                        torch.save(latents.cpu(), output / f"{name}_final.pt")
                        row = {
                            "condition": condition,
                            "prompt_index": index,
                            "prompt": prompt,
                            "seed": seed,
                            "video": video,
                            "sampler": trace["sampler"],
                            "num_frames": cfg.num_frames,
                            "num_steps": cfg.num_steps,
                            "width": cfg.width,
                            "height": cfg.height,
                            "fps": cfg.fps,
                            "video_verified": getattr(cfg, "verify_videos", False),
                            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                            **hashes,
                            **scores,
                            "compute_seconds": compute_seconds,
                            "duration_seconds": time.perf_counter() - started,
                        }
                        for device in cuda_devices:
                            row[f"{device}_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                            row[f"{device}_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                        write_json(output / f"{name}.json", {**row, "trace": trace})
                        with (output / "samples.jsonl").open("a") as f:
                            f.write(json.dumps(row, allow_nan=False) + "\n")
                            f.flush()
                            os.fsync(f.fileno())
                        with (output / "samples.csv").open("a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=list(row))
                            if not rows:
                                writer.writeheader()
                            writer.writerow(row)
                        rows.append(row)
                        if wandb_run is not None:
                            import wandb

                            video_media = wandb.Video(
                                str(output / video), format="mp4", caption=f"{condition} | {prompt} | seed={seed}"
                            )
                            media[index, seed, condition] = video_media
                            wandb_run.log(
                                {
                                    "sample_index": len(rows),
                                    "condition": condition,
                                    **(
                                        {}
                                        if getattr(cfg, "table_logging", False)
                                        else {f"investigation/{condition}/{k}": v for k, v in scores.items()}
                                    ),
                                    "investigation/compute_seconds": compute_seconds,
                                    **{f"investigation/{k}": v for k, v in row.items() if "peak_" in k},
                                    "investigation/video": video_media,
                                },
                                step=len(rows),
                                commit=True,
                            )
                        print(json.dumps({"event": "condition_complete", "completed": len(rows), **row}), flush=True)
                        del latents, clip, geometry, quality
        summary = summarize(rows, conditions, comparisons)
        if getattr(cfg, "report_prompt_strata", False):
            summary["prompt_strata"] = prompt_strata(rows, conditions, comparisons)
        if wandb_run is not None:
            for category in () if getattr(cfg, "table_logging", False) else ("means", "differences", "training"):
                for name, metrics in summary[category].items():
                    for metric, value in metrics.items():
                        wandb_run.summary[f"investigation/{category}/{name}/{metric}"] = value
            table = wandb.Table(
                columns=[
                    "prompt_index",
                    "prompt",
                    "seed",
                    "comparison",
                    "reference_video" if getattr(cfg, "table_logging", False) else "base_video",
                    "candidate_video" if getattr(cfg, "table_logging", False) else "trained_video",
                    *[f"delta/{k}" for k in SCORES],
                    "p_q",
                    "geometry_up",
                    "identity_ok",
                    "quality_ok",
                    "not_frozen",
                    "all_pass",
                ]
            )
            for pair in summary["pairs"]:
                index, seed = pair["prompt_index"], pair["seed"]
                for name, checks in pair["training"].items():
                    trained, base = name.split("-")
                    table.add_data(
                        index,
                        prompts[index - getattr(cfg, "prompt_start", 0)],
                        seed,
                        name,
                        media[index, seed, base],
                        media[index, seed, trained],
                        *[pair["differences"][name][k] for k in SCORES],
                        *[
                            checks[k]
                            for k in ("p_q", "geometry_up", "identity_ok", "quality_ok", "not_frozen", "all_pass")
                        ],
                    )
            tables = {"investigation/pairs": table}
            if getattr(cfg, "table_logging", False):
                columns = [
                    "prompt_index",
                    "prompt",
                    "seed",
                    "condition",
                    "sampler",
                    "num_frames",
                    "num_steps",
                    "width",
                    "height",
                    "fps",
                    *SCORES,
                    "compute_seconds",
                    "duration_seconds",
                ]
                tables["investigation/samples"] = wandb.Table(
                    columns=columns, data=[[row[k] for k in columns] for row in rows]
                )
                checks = ["p_q", "geometry_up", "identity_ok", "quality_ok", "not_frozen", "all_pass"]
                tables["investigation/aggregates"] = wandb.Table(
                    columns=["kind", "condition_or_comparison", *SCORES, *checks],
                    data=[
                        ["absolute", name, *[metrics[k] for k in SCORES], *[None] * len(checks)]
                        for name, metrics in summary["means"].items()
                    ]
                    + [
                        ["delta", name, *[metrics[k] for k in SCORES], *[summary["training"][name][k] for k in checks]]
                        for name, metrics in summary["differences"].items()
                        if name in summary["training"]
                    ],
                )
                if "prompt_strata" in summary:
                    tables["investigation/prompt_strata"] = wandb.Table(
                        columns=["stratum", "num_prompts", "comparison", *[f"delta/{k}" for k in SCORES], *checks],
                        data=[
                            [
                                group,
                                result["num_prompts"],
                                name,
                                *[metrics[k] for k in SCORES],
                                *[result["training"][name][k] for k in checks],
                            ]
                            for group, result in summary["prompt_strata"].items()
                            for name, metrics in result["differences"].items()
                            if name in result["training"]
                        ],
                    )
            wandb_run.log(tables, step=len(rows) + 1, commit=True)
            wandb_run.summary["investigation/num_videos"] = len(rows)
            wandb_run.summary["investigation/num_pairs"] = len(summary["pairs"])
            wandb_run.summary["investigation/num_comparisons"] = sum(len(pair["training"]) for pair in summary["pairs"])
        write_review(output, rows, conditions=conditions, checkpoint_comparisons=comparisons)
        write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            output / "failure.json",
            {**current, "completed": len(rows), "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
