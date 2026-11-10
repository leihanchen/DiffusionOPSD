"""Standalone evaluation retains the exact records behind reported metrics."""

import json
from types import SimpleNamespace

import torch
import pytest


@pytest.mark.parametrize("count,expected", [(None, 64), (0, 65), (1, 1)])
def test_standalone_evaluation_exports_record_identity(tmp_path, monkeypatch, count, expected):
    from scripts import eval_video_consistency as evaluation

    cfg = SimpleNamespace(pretrained=SimpleNamespace(model="stub"), use_lora=False,
                          sample=SimpleNamespace(num_steps=1, guidance_scale=1),
                          reward_device="cpu", judge=SimpleNamespace(device="cpu", num_frames=2))
    transformer = torch.nn.Linear(1, 1)
    pipe = SimpleNamespace(transformer=transformer, vae=torch.nn.Identity())
    pipe.to = lambda device: pipe
    records = [dict(text="caption", sample_id=f"source-id-{i}", video_source="relative/video.mp4") for i in range(65)]
    monkeypatch.setattr(evaluation, "_get_config", lambda name: cfg)
    monkeypatch.setattr(evaluation, "load_training_data", lambda cfg: SimpleNamespace(
        evaluation=records, metadata={"revision": "pinned-sha", "eval_hash": "test-hash"}))
    monkeypatch.setattr(evaluation.WanPipeline, "from_pretrained", lambda *a, **kw: pipe)
    monkeypatch.setattr("diffusionopsd.video.wan_geometry.require_expand_timesteps", lambda *a: None)
    monkeypatch.setattr(evaluation.torch, "load", lambda *a, **kw: transformer.state_dict())
    monkeypatch.setattr(evaluation, "WanRollout", lambda *a: None)
    for name in ("load_depth_anything3", "load_waft", "load_dinov2"):
        monkeypatch.setattr(evaluation, name, lambda *a: None)
    class Reward(torch.nn.Module):
        def forward(self, clip):
            return SimpleNamespace(geo=clip.mean(), s_id=clip.mean(), motion=clip.mean())
    monkeypatch.setattr(evaluation, "GeoReward", lambda *a: Reward())
    monkeypatch.setattr(evaluation, "load_video_reward", lambda *a, **kw: SimpleNamespace(p_win=lambda *a: .5))
    monkeypatch.setattr(evaluation, "_gen", lambda *a: torch.ones(1, 2, 3, 2, 2))
    output = tmp_path / "result.json"
    arguments = ["eval", "--policy", "stub.pt", "--out", str(output)]
    if count is not None:
        arguments += ["--num-prompts", str(count)]
    monkeypatch.setattr("sys.argv", arguments)
    evaluation.main()
    result = json.loads(output.read_text())
    assert result["records"][0]["sample_id"] == "source-id-0"
    assert result["records"][0]["video_source"] == "relative/video.mp4"
    assert result["records"][0]["seed"] == 0
    assert result["records"][0]["tuned"]["p_q"] == .5

    assert result["n"] == expected
    assert result["dataset"]["revision"] == "pinned-sha"
