#!/usr/bin/env python3
"""Submit fixed-batch Wan training with exclusive two-GPU workers on Narval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--nodes", type=int, default=4)
    result.add_argument("--gpus-per-node", type=int, default=4)
    result.add_argument("--cpus-per-worker", type=int, default=12)
    result.add_argument("--mem", default="", help="Per-node Slurm memory; default 448G/768G for one/two workers")
    result.add_argument("--time", default="04:00:00", help="HH:MM:SS allocation limit")
    result.add_argument("--job-budget-hours", type=float, default=None)
    result.add_argument("--account", default="def-wangcs")
    result.add_argument("--partition", default="")
    result.add_argument("--config", default="config/wan22_ti2v_epoch_two_gpu.py")
    result.add_argument("--run-id", default="")
    result.add_argument("--resume-from", default="")
    result.add_argument("--allow-topology-change", action="store_true")
    result.add_argument("--expected-resume-update", type=int, default=-1)
    result.add_argument("--max-updates", type=int, default=2, help="New updates per segment; 0 uses the time budget")
    result.add_argument("--eval-freq", type=int, default=0, help="Override interval; 0 retains configured default")
    result.add_argument("--evaluate-on-resume", action="store_true")
    result.add_argument("--chain", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--dependency", default="")
    result.add_argument("--resubmit", default="", help="Internal successor: load saved launch options")
    return result


def submission_options(args: argparse.Namespace) -> dict:
    if args.resubmit:
        options = json.loads(Path(args.resubmit).read_text())
        options.update(
            resume_from="",
            expected_resume_update=-1,
            allow_topology_change=False,
            dependency=args.dependency,
            dry_run=args.dry_run,
            resubmit=args.resubmit,
        )
    else:
        options = vars(args).copy()
    return options


def build_submission(options: dict) -> dict:
    nodes, gpus = options["nodes"], options["gpus_per_node"]
    if nodes <= 0 or gpus not in (2, 4):
        raise ValueError("Invalid allocation: Narval requires positive nodes and 2 or 4 full A100 GPUs per node")
    per_node, world = gpus // 2, nodes * (gpus // 2)
    if world not in (4, 8, 16):
        raise ValueError("Invalid allocation: require 4, 8 or 16 two-GPU workers")
    if options["cpus_per_worker"] <= 0 or options["max_updates"] < 0 or options["eval_freq"] < 0:
        raise ValueError("Invalid CPU request or update/evaluation interval")
    if options["expected_resume_update"] < -1:
        raise ValueError("Invalid expected resume update")
    clock = options["time"].split(":")
    if len(clock) != 3:
        raise ValueError("Time must use HH:MM:SS")
    hours, minutes, seconds = map(int, clock)
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError("Invalid allocation time")
    duration = hours + minutes / 60 + seconds / 3600
    budget = options["job_budget_hours"]
    if budget is None:
        budget = max(0.25, duration - 1)
    if not 0 < budget < duration:
        raise ValueError("Job budget must be positive and shorter than the allocation")
    run_id = options["run_id"] or time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("Run ID may contain only letters, numbers, hyphens and underscores")
    run_dir = ROOT / "logs" / f"wan22-scaled-{run_id}"
    resume = str(Path(options["resume_from"]).expanduser().resolve()) if options["resume_from"] else ""
    config = str(Path(options["config"]).expanduser().resolve())
    memory = options["mem"] or ("448G" if per_node == 1 else "768G")
    environment = {
        "WAN_SCALE_NODES": str(nodes),
        "WAN_SCALE_GPUS_PER_NODE": str(gpus),
        "WAN_SCALE_WORKERS_PER_NODE": str(per_node),
        "WAN_SCALE_WORLD_SIZE": str(world),
        "WAN_SCALE_OPTIONS": str(run_dir / "launch_options.json"),
        "WAN_EPOCH_RUN_ID": run_id,
        "WAN_EPOCH_CONFIG": config,
        "WAN_EPOCH_RUN_DIR": str(run_dir),
        "WAN_EPOCH_CHAIN_JOBS": str(int(options["chain"])),
        "WAN_EPOCH_JOB_BUDGET_HOURS": str(budget),
        "WAN_EPOCH_MAX_UPDATES": str(options["max_updates"]),
        "WAN_EPOCH_MONITOR_EVAL_FREQ": str(options["eval_freq"]),
        "WAN_EPOCH_EVALUATE_ON_RESUME": str(options["evaluate_on_resume"]).lower(),
        "WAN_EPOCH_EXPECTED_RESUME_UPDATE": str(options["expected_resume_update"]),
        "WAN_EPOCH_RESUME_FROM": resume,
        "WAN_EPOCH_ALLOW_TOPOLOGY_CHANGE": str(options["allow_topology_change"]).lower(),
    }
    if any("," in value or "\n" in value for value in environment.values()):
        raise ValueError("Slurm export values cannot contain commas or newlines")
    command = [
        "sbatch",
        "--parsable",
        f"--account={options['account']}",
        f"--nodes={nodes}",
        f"--ntasks={world}",
        f"--ntasks-per-node={per_node}",
        f"--gpus-per-node=a100:{gpus}",
        "--gpu-bind=per_task:2",
        f"--cpus-per-task={options['cpus_per_worker']}",
        f"--mem={memory}",
        f"--time={options['time']}",
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in environment.items()),
    ]
    if options["partition"]:
        command.append(f"--partition={options['partition']}")
    if options["dependency"]:
        command.append(f"--dependency={options['dependency']}")
    command.append(str(ROOT / "scripts/train_opsd_video_wan_scaled.slurm"))
    return {
        "command": command,
        "world_size": world,
        "workers_per_node": per_node,
        "global_rollouts": 16,
        "run_dir": str(run_dir),
        "environment": environment,
        "options": {**options, "run_id": run_id, "config": config},
    }


def main() -> None:
    args = parser().parse_args()
    try:
        payload = build_submission(submission_options(args))
        if args.dry_run:
            print(json.dumps(payload, indent=2))
            return
        options = payload["options"]
        if not Path(options["config"]).is_file():
            raise ValueError("Missing configuration file")
        if options["resume_from"]:
            source = Path(options["resume_from"]).expanduser().resolve()
            if not source.is_file():
                raise ValueError("Missing trusted resume checkpoint")
            if source.parent == Path(payload["run_dir"]) or (Path(payload["run_dir"]) / "training_state.pt").exists():
                raise ValueError("Migration requires a new destination run directory")
        if options["resubmit"] and not (Path(payload["run_dir"]) / "training_state.pt").is_file():
            raise ValueError("Successor requires a durable destination checkpoint")
        run_dir = Path(payload["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        if not options["resubmit"]:
            (run_dir / "launch_options.json").write_text(json.dumps(options, indent=2) + "\n")
        job_id = subprocess.check_output(payload["command"], cwd=ROOT, text=True).strip().split(";")[0]
        (run_dir / f"submission_{job_id}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"job_id": job_id, "run_dir": str(run_dir), "world_size": payload["world_size"]}))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
