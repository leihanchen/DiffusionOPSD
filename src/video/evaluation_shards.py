"""Validate and combine the eight fixed full-epoch evaluation shards."""

from __future__ import annotations

import json
import math
from pathlib import Path

from diffusionopsd.video.inference_investigation import SCORES, prompt_strata, summarize


CONDITIONS = ("base", "step7", "step50")
COMPARISONS = [("step7-base", "step7", "base"), ("step50-base", "step50", "base"), ("step50-step7", "step50", "step7")]


def collect_shards(root: Path, if_ready: bool = False):
    """Return records only after all eight manifests declare successful completion."""
    manifests = []
    for task in range(8):
        path = root / f"task_{task}" / "manifest.json"
        try:
            manifest = json.loads(path.read_text()) if path.exists() else {}
        except json.JSONDecodeError:
            if if_ready:
                return None  # Another task may currently be writing its manifest.
            raise
        if manifest.get("status") == "failed":
            raise ValueError(f"Shard {task} failed; inspect its failure.json")
        if manifest.get("status") != "complete":
            if if_ready:
                return None
            raise ValueError(f"Shard {task} is incomplete")
        manifests.append(manifest)
    rows = []
    settings = (
        "config_sha256",
        "sampler",
        "seed_offsets",
        "height",
        "width",
        "num_frames",
        "num_steps",
        "guidance_scale",
        "fps",
    )
    for task, manifest in enumerate(manifests):
        args = manifest["arguments"]
        expected_indices = list(range(task * 8, (task + 1) * 8))
        if (
            manifest["conditions"] != list(CONDITIONS)
            or manifest["prompt_indices"] != expected_indices
            or manifest.get("completed_videos") != 24
            or not manifest.get("videos_verified")
            or args["prompt_start"] != task * 8
            or args["num_prompts"] != 8
            or args["seed_offsets"] != [0]
        ):
            raise ValueError(f"Invalid shard {task} coverage")
        for key in (
            "checkpoint_sha256",
            "checkpoint",
            "prompt_file_sha256",
            "scheduler",
            "git_sha",
            "model_config_hashes",
        ):
            if manifest[key] != manifests[0][key]:
                raise ValueError(f"Shard {task} has mismatched {key}")
        if any(args[key] != manifests[0]["arguments"][key] for key in settings):
            raise ValueError(f"Shard {task} has mismatched generation settings")
        path = root / f"task_{task}"
        records = [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines() if line.strip()]
        expected = {(i, i, c) for i in expected_indices for c in CONDITIONS}
        actual = [(r["prompt_index"], r["seed"], r["condition"]) for r in records]
        if len(actual) != 24 or set(actual) != expected:
            raise ValueError(f"Shard {task} has missing or duplicate records")
        for row in records:
            video = Path(row["video"])
            if video.name != str(video) or not (path / video).is_file() or (path / video).stat().st_size == 0:
                raise ValueError(f"Invalid video in shard {task}")
            if not row.get("video_verified") or not all(math.isfinite(row[k]) for k in SCORES):
                raise ValueError(f"Unverified video or nonfinite score in shard {task}")
            if row["prompt"] != manifest["prompts"][row["prompt_index"] - task * 8]:
                raise ValueError(f"Prompt mismatch in shard {task}")
            if any(row[key] != args[key] for key in ("height", "width", "num_frames", "num_steps", "fps")):
                raise ValueError(f"Record settings mismatch in shard {task}")
            row["source_task"] = task
            row["source_wandb"] = manifest["wandb"]
        for index in expected_indices:
            matched = [r for r in records if r["prompt_index"] == index]
            for key in ("initial_latent_hash", "prompt_embedding_hash", "negative_embedding_hash"):
                if len({r[key] for r in matched}) != 1:
                    raise ValueError(f"Paired inputs differ for prompt {index}")
        rows.extend(records)
    rows.sort(key=lambda r: (r["prompt_index"], CONDITIONS.index(r["condition"])))
    summary = summarize(rows, CONDITIONS, COMPARISONS)
    summary["prompt_strata"] = prompt_strata(rows, CONDITIONS, COMPARISONS)
    return rows, summary, manifests
