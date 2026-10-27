#!/usr/bin/env python3
"""Combine completed array shards locally and create a grouped offline W&B run."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
from pathlib import Path

from diffusionopsd.video.evaluation_shards import CONDITIONS, COMPARISONS, collect_shards
from diffusionopsd.video.inference_investigation import SCORES, write_json, write_review


def aggregate(root, if_ready=False, log_wandb=False):
    root = root.resolve()
    with (root / ".aggregate.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        output = root / "combined"
        if (output / "manifest.json").exists():
            manifest = json.loads((output / "manifest.json").read_text())
            if manifest.get("status") == "complete":
                return output
        collected = collect_shards(root, if_ready)
        if collected is None:
            print("Other array tasks are incomplete; aggregation deferred.", flush=True)
            return None
        rows, summary, manifests = collected
        staging = root / f"combined-building-{os.getpid()}"
        staging.mkdir()
        for row in rows:
            source = root / f"task_{row['source_task']}" / row["video"]
            try:
                os.link(source, staging / row["video"])
            except OSError:
                shutil.copyfile(source, staging / row["video"])
        (staging / "samples.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
        with (staging / "samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(staging / "summary.json", summary)
        write_review(staging, rows, conditions=CONDITIONS, checkpoint_comparisons=COMPARISONS)
        manifest = {
            "status": "complete",
            "expected_videos": 192,
            "completed_videos": len(rows),
            "completed_pairs": len(summary["pairs"]),
            "completed_comparisons": 192,
            "videos_verified": True,
            "conditions": CONDITIONS,
            "source_manifests": manifests,
        }
        if log_wandb:
            import wandb

            run = wandb.init(
                entity=os.environ["WANDB_ENTITY"],
                project=os.environ["WANDB_PROJECT"],
                mode="offline",
                name=f"job-{os.environ['SLURM_ARRAY_JOB_ID']}-base-step7-step50-combined-eval-narval",
                group=os.environ.get("WANDB_RUN_GROUP"),
                dir=str(root),
                config={
                    "array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
                    "source_runs": [m["wandb"] for m in manifests],
                },
            )
            try:
                media = {
                    (r["prompt_index"], r["condition"]): wandb.Video(str(staging / r["video"]), format="mp4")
                    for r in rows
                }
                for number, row in enumerate(rows, start=1):
                    run.log(
                        {
                            "comparison/video": media[row["prompt_index"], row["condition"]],
                            "prompt_index": row["prompt_index"],
                            "condition": row["condition"],
                        },
                        step=number,
                        commit=True,
                    )
                score_columns = [
                    "prompt_index",
                    "prompt",
                    "seed",
                    "condition",
                    *SCORES,
                    "compute_seconds",
                    "duration_seconds",
                    *[key for key in sorted(rows[0]) if "peak_" in key],
                ]
                check_columns = ["p_q", "geometry_up", "identity_ok", "quality_ok", "not_frozen", "all_pass"]
                pair_columns = [
                    "prompt_index",
                    "seed",
                    "comparison",
                    "reference_video",
                    "candidate_video",
                    *[f"delta/{k}" for k in SCORES],
                    *check_columns,
                ]
                pairs = []
                for pair in summary["pairs"]:
                    i = pair["prompt_index"]
                    for name, candidate, reference in COMPARISONS:
                        checks = pair["training"][name]
                        pairs.append(
                            [
                                i,
                                pair["seed"],
                                name,
                                media[i, reference],
                                media[i, candidate],
                                *[pair["differences"][name][k] for k in SCORES],
                                *[checks[key] for key in check_columns],
                            ]
                        )
                run.log(
                    {
                        "comparison/samples": wandb.Table(
                            columns=score_columns, data=[[r[k] for k in score_columns] for r in rows]
                        ),
                        "comparison/pairs": wandb.Table(columns=pair_columns, data=pairs),
                        "comparison/aggregates": wandb.Table(
                            columns=["kind", "condition_or_comparison", *SCORES, *check_columns],
                            data=[
                                [
                                    kind,
                                    name,
                                    *[metrics[k] for k in SCORES],
                                    *[summary["training"].get(name, {}).get(key) for key in check_columns],
                                ]
                                for kind, values in (("absolute", summary["means"]), ("delta", summary["differences"]))
                                for name, metrics in values.items()
                            ],
                        ),
                        "comparison/prompt_strata": wandb.Table(
                            columns=[
                                "stratum",
                                "num_prompts",
                                "comparison",
                                *[f"delta/{key}" for key in SCORES],
                                *check_columns,
                            ],
                            data=[
                                [
                                    stratum,
                                    values["num_prompts"],
                                    name,
                                    *[metrics[key] for key in SCORES],
                                    *[values["training"][name][key] for key in check_columns],
                                ]
                                for stratum, values in summary["prompt_strata"].items()
                                for name, metrics in values["differences"].items()
                            ],
                        ),
                    },
                    step=len(rows) + 1,
                    commit=True,
                )
                for key, value in {"num_videos": 192, "num_pairs": 64, "num_comparisons": 192}.items():
                    run.summary[f"comparison/{key}"] = value
                manifest["wandb"] = {"id": run.id, "entity": run.entity, "project": run.project}
                write_json(staging / "manifest.json", manifest)
                for filename in ("manifest.json", "summary.json", "samples.jsonl", "samples.csv"):
                    shutil.copyfile(staging / filename, Path(run.dir) / filename)
                run.finish()
            except Exception:
                run.finish(exit_code=1)
                raise
        write_json(staging / "manifest.json", manifest)
        staging.rename(output)
        print(f"Combined evaluation complete: {output}", flush=True)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-root", type=Path, required=True)
    parser.add_argument("--if-ready", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    aggregate(args.array_root, args.if_ready, args.wandb)


if __name__ == "__main__":
    main()
