import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def launcher(root):
    spec = importlib.util.spec_from_file_location("portable_launcher", ROOT / "scripts/submit_opsd_video_wan_scaled.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    return module


def test_launcher_stores_repository_relative_paths(tmp_path):
    module = launcher(tmp_path)
    args = module.parser().parse_args(["--run-id", "portable", "--resume-from", str(tmp_path / "logs/source/state.pt")])
    payload = module.build_submission(module.submission_options(args))
    env = payload["environment"]
    assert env["WAN_EPOCH_CONFIG"] == "config/wan22_ti2v_epoch_two_gpu.py"
    assert env["WAN_EPOCH_RUN_DIR"] == "logs/wan22-scaled-portable"
    assert env["WAN_SCALE_OPTIONS"] == "logs/wan22-scaled-portable/launch_options.json"
    assert env["WAN_EPOCH_RESUME_FROM"] == "logs/source/state.pt"
    assert payload["options"]["config"] == env["WAN_EPOCH_CONFIG"]
    assert payload["options"]["resume_from"] == env["WAN_EPOCH_RESUME_FROM"]


def test_legacy_successor_options_follow_moved_repository(tmp_path):
    module = launcher(tmp_path)
    config = tmp_path / "config/wan22_ti2v_epoch_two_gpu.py"
    config.parent.mkdir()
    config.write_text("config")
    options = vars(module.parser().parse_args(["--run-id", "moved"]))
    options["config"] = "/old/repository/config/wan22_ti2v_epoch_two_gpu.py"
    saved = tmp_path / "logs/wan22-scaled-moved/launch_options.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps(options))
    args = module.parser().parse_args(["--resubmit", str(saved), "--dry-run"])
    payload = module.build_submission(module.submission_options(args))
    assert payload["environment"]["WAN_EPOCH_CONFIG"] == "config/wan22_ti2v_epoch_two_gpu.py"
    assert str(saved.parent) == payload["run_dir"]


def test_legacy_resume_reconstructs_and_checks_original_hash_after_move(tmp_path):
    from diffusionopsd.video.distributed_epoch import prepare_resume

    config = {"logdir": "/original/repo/logs/source", "width": 1280}
    state = {
        "version": 1,
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "prompt_hash": "prompts",
        "rng_states": [{"rank": i} for i in range(4)],
        "next_update": 4,
        "max_update_seconds": 10,
        "reference_files": {0: "/original/repo/logs/source/calibration/reference_000.pt"},
    }
    topology = {"world_size": 4, "workers_per_node": 1, "nodes": 4}
    moved_source = str(tmp_path / "logs/source")
    result, changed = prepare_resume(state, {**config, "logdir": "logs/new"}, "prompts", topology, False, moved_source)
    assert result["next_update"] == 4 and not changed
    with pytest.raises(ValueError, match="config_hash"):
        prepare_resume(state, {**config, "width": 720}, "prompts", topology, False, moved_source)


def test_reference_paths_prefer_copied_files_and_round_trip(tmp_path):
    from scripts.wan_paths import resolve_reference_files, portable_reference_files

    old = tmp_path / "old"
    moved = tmp_path / "moved"
    relative = Path("logs/source/calibration/reference_000.pt")
    for root in (old, moved):
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_text(str(root))
    refs = resolve_reference_files({0: str(old / relative)}, moved)
    assert refs == {0: str(moved / relative)}
    stored = portable_reference_files(refs, moved)
    assert stored == {0: str(relative)}
    assert resolve_reference_files(stored, moved) == refs
    (moved / relative).unlink()
    with pytest.raises(FileNotFoundError):
        resolve_reference_files(stored, moved)
    assert resolve_reference_files(stored, moved, source_root=old) == {0: str(old / relative)}


def test_model_config_is_normalized_without_downloads(tmp_path):
    from scripts.wan_paths import normalize_video_reward_config

    reward = tmp_path / "video_reward_ckpts/VideoReward"
    qwen = tmp_path / "video_reward_ckpts/Qwen2-VL-2B-Instruct"
    reward.mkdir(parents=True)
    qwen.mkdir()
    config = reward / "model_config.json"
    config.write_text(json.dumps({"model_config": {"model_name_or_path": "/old/repo/Qwen", "model_revision": "main"}}))
    normalize_video_reward_config(tmp_path)
    saved = json.loads(config.read_text())["model_config"]
    assert saved["model_name_or_path"] == "video_reward_ckpts/Qwen2-VL-2B-Instruct"
    assert saved["model_revision"] is None
    before = config.read_text()
    normalize_video_reward_config(tmp_path)
    assert config.read_text() == before


def test_submission_and_successor_work_after_directory_move(tmp_path):
    import os
    import shutil
    import subprocess
    import sys

    original = tmp_path / "original"
    copied = tmp_path / "relocated"
    (original / "scripts").mkdir(parents=True)
    (original / "config").mkdir()
    for name in ("submit_opsd_video_wan_scaled.py", "wan_paths.py", "train_opsd_video_wan_scaled.slurm"):
        shutil.copyfile(ROOT / "scripts" / name, original / "scripts" / name)
    (original / "config/wan22_ti2v_epoch_two_gpu.py").write_text("config")
    fake = tmp_path / "sbatch"
    fake.write_text("#!/bin/sh\nprintf '700001\\n'\n")
    fake.chmod(0o755)
    environment = {**os.environ, "PATH": str(tmp_path) + ":" + os.environ["PATH"]}
    command = [sys.executable, str(original / "scripts/submit_opsd_video_wan_scaled.py"), "--run-id", "portable"]
    first = subprocess.run(command, cwd=tmp_path, env=environment, check=True, capture_output=True, text=True)
    assert json.loads(first.stdout)["job_id"] == "700001"
    original.rename(copied)
    destination = copied / "logs/wan22-scaled-portable"
    (destination / "training_state.pt").write_text("checkpoint")
    second = subprocess.run(
        [
            sys.executable,
            str(copied / "scripts/submit_opsd_video_wan_scaled.py"),
            "--resubmit",
            "logs/wan22-scaled-portable/launch_options.json",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(second.stdout)["run_dir"] == str(destination)
    payload = json.loads((destination / "submission_700001.json").read_text())
    assert str(original) not in json.dumps(payload)
    assert payload["environment"]["WAN_EPOCH_RUN_DIR"] == "logs/wan22-scaled-portable"
