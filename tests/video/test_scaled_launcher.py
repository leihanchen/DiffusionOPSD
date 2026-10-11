import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/submit_opsd_video_wan_scaled.py"


def launcher():
    spec = importlib.util.spec_from_file_location("scaled_launcher", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("nodes,gpus,workers,per_node", [(4, 2, 4, 1), (4, 4, 8, 2), (8, 4, 16, 2)])
def test_submission_dry_run_has_consistent_gpu_binding(nodes, gpus, workers, per_node):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--nodes", str(nodes), "--gpus-per-node", str(gpus), "--dry-run"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    args = payload["command"]
    assert not any(arg.startswith("--partition=") for arg in args)
    assert f"--ntasks={workers}" in args
    assert f"--ntasks-per-node={per_node}" in args
    assert f"--gpus-per-node=a100:{gpus}" in args
    assert not any(arg.startswith("--gpus-per-task=") for arg in args)
    assert "--gpu-bind=per_task:2" in args
    assert payload["world_size"] == workers
    assert payload["global_rollouts"] == 16
    assert f"--mem={'448G' if per_node == 1 else '768G'}" in args


@pytest.mark.parametrize("nodes,gpus", [(4, 3), (3, 2), (4, 6), (4, 8)])
def test_invalid_narval_request_is_rejected_without_submission(nodes, gpus):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--nodes", str(nodes), "--gpus-per-node", str(gpus), "--dry-run"],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "Invalid allocation" in result.stderr


def test_successor_preserves_settings_and_resumes_destination(tmp_path):
    module = launcher()
    args = module.parser().parse_args(
        [
            "--nodes",
            "4",
            "--gpus-per-node",
            "4",
            "--time",
            "24:00:00",
            "--chain",
            "--max-updates",
            "0",
            "--eval-freq",
            "25",
            "--resume-from",
            "/source/training_state_step4.pt",
            "--allow-topology-change",
            "--expected-resume-update",
            "4",
            "--run-id",
            "probe",
        ]
    )
    options = module.submission_options(args)
    path = tmp_path / "options.json"
    path.write_text(json.dumps(options))
    successor = module.parser().parse_args(["--resubmit", str(path), "--dependency", "afterok:123", "--dry-run"])
    options2 = module.submission_options(successor)
    for key in ("nodes", "gpus_per_node", "time", "chain", "max_updates", "eval_freq", "run_id", "config"):
        assert options2[key] == options[key]
    assert options2["resume_from"] == ""
    assert options2["expected_resume_update"] == -1
    assert not options2["allow_topology_change"]
    assert options2["dependency"] == "afterok:123"


def test_slurm_step_uses_allocated_tasks_and_explicit_visibility():
    script = (ROOT / "scripts/train_opsd_video_wan_scaled.slurm").read_text()
    assert '--ntasks="$SLURM_NTASKS"' in script
    assert '--ntasks-per-node="$WAN_SCALE_WORKERS_PER_NODE"' in script
    assert "--gpus-per-task=a100:2" in script
    assert "--gpu-bind=per_task:2" in script
    assert "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" in script
    assert "WANDB_MODE=offline" in script
    assert "--scalable_two_gpu=true" in script


def test_container_launch_preserves_bound_devices_and_checkpoint_argument(tmp_path):
    import os
    import re

    script = (ROOT / "scripts/train_opsd_video_wan_scaled.slurm").read_text()
    command = re.search(r"bash -c '(.*?)' scaled-worker", script, re.S).group(1)
    fake = tmp_path / "apptainer"
    fake.write_text("#!" + sys.executable + "\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    fake.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": str(tmp_path) + ":" + os.environ["PATH"],
        "CUDA_VISIBLE_DEVICES": "2,3",
        "WAN_SCALE_REPO": str(ROOT),
        "WAN_SCALE_SIF": "image.sif",
        "WAN_EPOCH_CONFIG": "config.py",
        "WAN_EPOCH_RUN_DIR": "destination",
        "WANDB_DIR": "destination",
        "HF_HOME": "hf",
        "TORCH_HOME": "weights",
        "VIDEO_REWARD_CKPT_PATH": "weights",
        "XDG_CACHE_HOME": "cache",
        "MASTER_ADDR": "node",
        "MASTER_PORT": "23456",
        "WAN_EPOCH_JOB_BUDGET_HOURS": "3",
        "WAN_EPOCH_MAX_UPDATES": "2",
        "WAN_EPOCH_MONITOR_EVAL_FREQ": "2",
        "WAN_EPOCH_EVALUATE_ON_RESUME": "false",
        "WAN_EPOCH_EXPECTED_RESUME_UPDATE": "4",
        "WAN_EPOCH_ALLOW_TOPOLOGY_CHANGE": "true",
        "WAN_EPOCH_RESUME_FROM": "/tmp/checkpoint with spaces.pt",
    }
    result = subprocess.run(
        ["bash", "-c", command, "scaled-worker", "--bind", str(ROOT)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    args = json.loads(result.stdout)
    assert "CUDA_VISIBLE_DEVICES=2,3" in args
    assert "--resume_from=/tmp/checkpoint with spaces.pt" in args
    assert "--scalable_two_gpu=true" in args
    assert "--allow_topology_change=true" in args
    assert "--config.logdir=destination" in args
    environment["WAN_EPOCH_RESUME_FROM"] = ""
    result = subprocess.run(
        ["bash", "-c", command, "scaled-worker"], env=environment, check=True, capture_output=True, text=True
    )
    assert not any(a.startswith("--resume_from=") for a in json.loads(result.stdout))


def test_single_node_smoke_allocation_is_explicit_and_one_update():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--nodes",
            "1",
            "--gpus-per-node",
            "4",
            "--single-node-smoke",
            "--max-updates",
            "1",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["world_size"] == 2
    assert payload["global_rollouts"] == 16
    assert "--nodes=1" in payload["command"]
    assert "--ntasks=2" in payload["command"]
    assert payload["environment"]["WAN_SCALE_SINGLE_NODE_SMOKE"] == "true"
    assert payload["environment"]["WAN_EPOCH_CHAIN_JOBS"] == "0"
    assert payload["environment"]["WAN_EPOCH_MAX_UPDATES"] == "1"


@pytest.mark.parametrize(
    "extra", [[], ["--single-node-smoke"], ["--single-node-smoke", "--max-updates", "1", "--chain"]]
)
def test_single_node_smoke_rejects_missing_mode_or_multi_update_or_chain(extra):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--nodes", "1", "--gpus-per-node", "4", *extra, "--dry-run"],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
