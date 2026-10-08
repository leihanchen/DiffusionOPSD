"""Inference-only configuration, independent of training assets and presets."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("investigation_cli", ROOT / "scripts/investigate_wan_inference.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture
def arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_REWARD_CKPT_PATH", str(tmp_path))
    for name in (
        "policy.pt", "prompts.txt", "Wan2.2-TI2V-5B-Diffusers/model_index.json",
        "depth-anything-3-large-v1.1/model.safetensors", "waft_tar_c_t.pth",
        "dinov2-base/config.json", "VideoReward/model_config.json",
        "Qwen2-VL-2B-Instruct/config.json", "VideoAlign/inference.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return ["--policy", str(tmp_path / "policy.pt"), "--prompts", str(tmp_path / "prompts.txt"),
            "--output-dir", str(tmp_path / "output")]


def test_preset_and_cli_precedence(arguments, tmp_path):
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps(dict(height=704, width=1280, num_frames=121, num_steps=50, guidance_scale=5.)))
    cfg = cli.parse_args(arguments + ["--config", str(preset)])
    assert (cfg.height, cfg.width, cfg.num_frames, cfg.num_steps, cfg.guidance_scale) == (704, 1280, 121, 50, 5.)
    assert cfg.config == preset.resolve()
    assert cfg.config_sha256 == hashlib.sha256(preset.read_bytes()).hexdigest()
    for options in (["--num-steps", "12", "--config", str(preset)],
                    ["--config", str(preset), "--num-steps", "12"]):
        assert cli.parse_args(arguments + options).num_steps == 12
    assert cfg.fps == 8


def test_legacy_defaults(arguments):
    cfg = cli.parse_args(arguments)
    assert (cfg.height, cfg.width, cfg.num_frames, cfg.num_steps, cfg.guidance_scale) == (480, 832, 17, 30, 5.)
    assert cfg.config is None and cfg.config_sha256 is None
    assert cfg.num_prompts == 2 and cfg.seed_offsets == [0, 1000]


def test_repository_preset_four_video_setup(arguments):
    cfg = cli.parse_args(arguments + ["--config", str(ROOT / "config/wan22_inference_debug.json"),
                                     "--num-prompts", "1", "--seed-offsets", "0"])
    assert (cfg.height, cfg.width, cfg.num_frames, cfg.num_steps, cfg.guidance_scale) == (704, 1280, 121, 50, 5.)
    assert cfg.num_prompts == 1 and cfg.seed_offsets == [0]
    assert cfg.fps == 8


@pytest.mark.parametrize("contents", [
    '{', '[]', '{"fps": 24}', '{"height": true}', '{"width": "1280"}',
    '{"num_steps": 1.5}', '{"height": 705}', '{"num_frames": 120}',
    '{"num_steps": 0}', '{"guidance_scale": NaN}', '{"guidance_scale": 1}',
])
def test_invalid_config_rejected(arguments, tmp_path, contents):
    preset = tmp_path / "bad.json"
    preset.write_text(contents)
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(arguments + ["--config", str(preset)])
    assert exc.value.code == 2


def test_missing_config_rejected(arguments, tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args(arguments + ["--config", str(tmp_path / "missing.json")])
