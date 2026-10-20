import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from diffusionopsd.video import inference_investigation as inv


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.active_adapters = ["old"]
        self.disabled = False

    def set_adapter(self, name):
        self.active_adapters = [name] if isinstance(name, str) else name
        self.weight.requires_grad_(True)

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False


class Scheduler:
    def __init__(self):
        self.timesteps = torch.tensor([1000.0, 500.0])
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.used = False


class Pipe:
    def __init__(self):
        self.transformer = Policy()
        self.calls = []
        self.encodes = 0

    def encode_prompt(self, prompt, **kwargs):
        self.encodes += 1
        return torch.ones(1, 2), torch.zeros(1, 2)

    def sample(self, latents, pe, ne, native):
        assert not self.scheduler.used
        self.scheduler.used = True
        assert not torch.is_grad_enabled()
        assert not self.transformer.weight.requires_grad
        assert self.transformer.active_adapters == ["default"]
        self.calls.append((latents.clone(), pe.clone(), ne.clone(), self.transformer.disabled, native))
        value = 0 if self.transformer.disabled else self.transformer.weight
        # Mutating the supplied copy verifies the caller protects paired inputs.
        latents.add_(value + (0.01 if native else 0))
        return latents

    def __call__(self, **kwargs):
        assert kwargs["output_type"] == "latent"
        return (self.sample(kwargs["latents"], kwargs["prompt_embeds"], kwargs["negative_prompt_embeds"], True),)


class Roll:
    def __init__(self, pipe, *args):
        self.pipe = pipe

    def rollout(self, pe, ne, latents, sigma):
        return SimpleNamespace(x0=self.pipe.sample(latents, pe, ne, False))

    def decode01(self, latents):
        return latents.sigmoid()


class Reward:
    def __call__(self, clip):
        v = clip.mean()
        return SimpleNamespace(geo=v, rigid=v, dino=v, s_id=v, motion=v)


class Judge:
    def score(self, clip, prompt):
        v = float(clip.mean())
        return dict(VQ=v, MQ=v, TA=v, Overall=v * 3)


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(inv, "WanRollout", Roll)
    monkeypatch.setattr(inv, "latent_shape_from_pipe", lambda *args: (1, 5, 3, 32, 32))
    cfg = SimpleNamespace(
        num_steps=2,
        guidance_scale=5.0,
        policy_device="cpu",
        vae_device="cpu",
        reward_device="cpu",
        seed_offsets=[0, 1000],
        num_frames=5,
        height=32,
        width=32,
        fps=8,
    )
    return Pipe(), cfg, inv.prepare_output(tmp_path / "run")


def test_full_paired_run_offline_media_and_parameter_safety(tmp_path, monkeypatch):
    import wandb

    pipe, cfg, out = setup(tmp_path, monkeypatch)
    before = pipe.transformer.weight.detach().clone()
    run = wandb.init(project="diffusionopsd-test", mode="offline", dir=str(out))
    try:
        summary = inv.run_investigation(pipe, Reward(), Judge(), cfg, ["first", "second"], out, Scheduler, run)
        assert run.settings.mode == "offline"
    finally:
        run.finish()
    assert summary["num_videos"] == 16
    assert pipe.encodes == 2
    assert pipe.transformer.active_adapters == ["old"]
    assert not pipe.transformer.weight.requires_grad
    assert pipe.transformer.weight.grad is None
    assert torch.equal(before, pipe.transformer.weight)
    for start in range(0, 16, 4):
        group = pipe.calls[start : start + 4]
        assert [(x[3], x[4]) for x in group] == [(True, False), (True, True), (False, False), (False, True)]
        assert all(torch.equal(x[j], group[0][j]) for x in group for j in range(3))
    rows = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert sorted({(r["prompt_index"], r["seed"]) for r in rows}) == [(0, 0), (0, 1000), (1, 1), (1, 1001)]
    assert len(list(out.glob("*.mp4"))) == 16
    assert len(list(out.glob("*.png"))) == 48
    assert len(list(out.glob("*_final.pt"))) == 16
    assert len(list((out / "review").glob("*.mp4"))) == 16
    assert len(list((out / "wandb").glob("offline-run-*/files/media/videos/**/*.mp4"))) == 16
    assert summary["training"]["C-A"]["p_q"] > 0.5
    assert "condition" not in (out / "review" / "manifest.json").read_text()
    assert (out / "review" / "preferences.csv").exists()
    assert not (out / "failure.json").exists()


@pytest.mark.parametrize("failure", ["nan", "generation"])
def test_failure_preserves_completed_sample_and_restores_adapter(tmp_path, monkeypatch, failure):
    pipe, cfg, out = setup(tmp_path, monkeypatch)
    original = pipe.sample

    def sample(*args):
        if len(pipe.calls) == 1 and failure == "generation":
            raise RuntimeError("failed native generation")
        return original(*args)

    pipe.sample = sample
    judge = Judge()
    original_score = judge.score

    def score(*args):
        value = original_score(*args)
        if len(pipe.calls) == 2 and failure == "nan":
            value["VQ"] = float("nan")
        return value

    judge.score = score
    with pytest.raises((RuntimeError, ValueError)):
        inv.run_investigation(pipe, Reward(), judge, cfg, ["first"], out, Scheduler)
    failure_record = json.loads((out / "failure.json").read_text())
    assert failure_record["condition"] == "B" and failure_record["completed"] == 1
    assert not (out / "summary.json").exists()
    assert len(list(out.glob("*.mp4"))) == 1
    assert pipe.transformer.active_adapters == ["old"] and not pipe.transformer.disabled
    assert not pipe.transformer.weight.requires_grad


def test_summary_arithmetic_and_prompt_weighting():
    rows = []
    for prompt, seeds, multiplier in [(0, [0, 1000], 1), (1, [1], 3)]:
        for seed in seeds:
            for condition, value in zip("ABCD", [1, 2, 4, 8]):
                rows.append(
                    dict(
                        prompt_index=prompt,
                        seed=seed,
                        condition=condition,
                        **{k: value * multiplier for k in inv.SCORES},
                    )
                )
    summary = inv.summarize(rows)
    assert summary["differences"]["B-A"]["geo"] == 2
    assert summary["differences"]["C-A"]["geo"] == 6
    assert summary["differences"]["D-B"]["geo"] == 12
    assert summary["differences"]["interaction"]["geo"] == 6
    with pytest.raises(ValueError, match="incomplete"):
        inv.summarize(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        inv.summarize(rows + rows[:1])


def test_euler_summary_keeps_pairs_and_weights_prompts_equally():
    rows = []
    for prompt, seeds, delta in [(0, [0, 1000], 1), (1, [1], 3)]:
        for seed in seeds:
            for condition, value in [("A", 1), ("C", 1 + delta)]:
                rows.append(dict(prompt_index=prompt, seed=seed, condition=condition, **{k: value for k in inv.SCORES}))
    summary = inv.summarize(rows, conditions=("A", "C"))
    assert set(summary["means"]) == {"A", "C"}
    assert summary["differences"]["C-A"]["geo"] == 2
    assert len(summary["pairs"]) == 3
    assert set(summary["training"]) == {"C-A"}
    with pytest.raises(ValueError, match="incomplete"):
        inv.summarize(rows[:-1], conditions=("A", "C"))


def test_training_checkpoint_selects_default_and_checks_update(tmp_path):
    import numpy as np

    path = tmp_path / "training.pt"
    torch.save(
        {
            "policy_default": {"weight": torch.tensor([4.0])},
            "policy_old": {"weight": torch.tensor([2.0])},
            "next_update": 4,
            "rng": np.random.get_state(),
        },
        path,
    )
    state, metadata = inv.load_adapter_checkpoint(path, "training", expected_update=4)
    assert state["weight"].item() == 4
    assert metadata == {"format": "training", "adapter_key": "policy_default", "next_update": 4}
    with pytest.raises(ValueError, match="update"):
        inv.load_adapter_checkpoint(path, "training", expected_update=2)
    torch.save({"next_update": 4, "policy_old": {}}, path)
    with pytest.raises(ValueError, match="policy_default"):
        inv.load_adapter_checkpoint(path, "training", expected_update=4)


def test_flat_checkpoint_remains_supported(tmp_path):
    path = tmp_path / "adapter.pt"
    torch.save({"weight": torch.ones(2)}, path)
    state, metadata = inv.load_adapter_checkpoint(path)
    assert torch.equal(state["weight"], torch.ones(2))
    assert metadata["format"] == "adapter"
    with pytest.raises(ValueError, match="training"):
        inv.load_adapter_checkpoint(path, expected_update=4)


def test_euler_64_video_run_logs_32_pairs(tmp_path, monkeypatch):
    import csv
    import wandb

    pipe, cfg, out = setup(tmp_path, monkeypatch)
    cfg.sampler = "euler"
    run = wandb.init(project="diffusionopsd-test", mode="offline", dir=str(out))
    try:
        summary = inv.run_investigation(
            pipe, Reward(), Judge(), cfg, [f"prompt {i}" for i in range(16)], out, Scheduler, run
        )
    finally:
        run.finish()
    assert summary["num_videos"] == 64
    assert len(summary["pairs"]) == 32
    for start in range(0, 64, 2):
        base, tuned = pipe.calls[start : start + 2]
        assert (base[3:], tuned[3:]) == ((True, False), (False, False))
        assert all(torch.equal(base[j], tuned[j]) for j in range(3))
    assert len(list(out.glob("*.mp4"))) == 64
    with (out / "review/preferences.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 32
    table_paths = list((out / "wandb").glob("offline-run-*/files/media/table/**/*.table.json"))
    assert len(table_paths) == 1
    table = json.loads(table_paths[0].read_text())
    assert len(table["data"]) == 32
    assert "base_video" in table["columns"] and "trained_video" in table["columns"]
    assert pipe.transformer.active_adapters == ["old"] and pipe.transformer.weight.grad is None


def test_inputs_and_adapter_checkpoint_rejected(tmp_path):
    inv.prepare_output(tmp_path)
    (tmp_path / "existing").write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        inv.prepare_output(tmp_path)
    assert (tmp_path / "existing").read_text() == "keep"
    expected = {"weight": torch.ones(2)}
    inv.validate_adapter(expected, expected)
    for bad in [{}, {"weight": torch.ones(3)}, {"weight": torch.tensor([1.0, float("nan")])}]:
        with pytest.raises(ValueError):
            inv.validate_adapter(expected, bad)


def test_actual_peft_adapter_restoration():
    from diffusionopsd.video.wan_policy import attach_lora
    from peft import get_peft_model_state_dict

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):
            return self.to_q(x)

    policy = attach_lora(Tiny(), None)
    policy.set_adapter("old")
    policy.eval().requires_grad_(False)
    state = get_peft_model_state_dict(policy, adapter_name="default")
    inv.validate_adapter(state, {k: v.clone() for k, v in state.items()})
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    with pytest.raises(RuntimeError), torch.no_grad(), inv.adapter_state(policy, False):
        assert policy.active_adapters == ["default"]
        assert not any(p.requires_grad for p in policy.parameters())
        policy(torch.ones(1, 2))
        raise RuntimeError("test restoration")
    assert policy.active_adapters == ["old"]
    assert not any(p.requires_grad or p.grad is not None for p in policy.parameters())
    assert all(torch.equal(before[k], v) for k, v in policy.state_dict().items())


def test_cli_validates_before_loading_models(tmp_path, monkeypatch):
    from scripts.investigate_wan_inference import parse_args

    root = tmp_path / "weights"
    root.mkdir()
    files = [
        "model_index.json",
        "policy.pt",
        "prompts.txt",
        "depth-anything-3-large-v1.1/model.safetensors",
        "waft_tar_c_t.pth",
        "dinov2-base/config.json",
        "VideoReward/model_config.json",
        "Qwen2-VL-2B-Instruct/config.json",
        "VideoAlign/inference.py",
    ]
    for name in files:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    monkeypatch.setenv("VIDEO_REWARD_CKPT_PATH", str(root))
    args = [
        "--policy",
        str(root / "policy.pt"),
        "--model",
        str(root),
        "--prompts",
        str(root / "prompts.txt"),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    cfg = parse_args(args)
    assert cfg.num_prompts == 2 and cfg.seed_offsets == [0, 1000]
    for flags in [["--seed-offsets", "0", "0"], ["--width", "33"], ["--num-frames", "1"], ["--guidance-scale", "nan"]]:
        with pytest.raises(SystemExit):
            parse_args(args + flags)


def test_checkpoint_list_comparisons():
    from diffusionopsd.video.inference_investigation import summarize, SCORES

    rows = [
        dict(prompt_index=i, seed=i, condition=c, **{k: 1.0 for k in SCORES})
        for i in range(2)
        for c in ("base", "step5", "step7")
    ]
    result = summarize(
        rows, ("base", "step5", "step7"), [("step5-base", "step5", "base"), ("step7-base", "step7", "base")]
    )
    assert result["num_videos"] == 6
    assert sum(len(p["training"]) for p in result["pairs"]) == 4


def test_checkpoint_list_validation(tmp_path):
    checkpoint = tmp_path / "state.pt"
    torch.save({"policy_default": {"x": torch.ones(2)}, "next_update": 5}, checkpoint)
    manifest = tmp_path / "checkpoints.json"
    entry = dict(label="step5", path=str(checkpoint), expected_update=5)
    manifest.write_text(json.dumps([entry]))
    states, metadata = inv.load_checkpoint_list(manifest)
    assert list(states) == ["step5"]
    assert metadata[0]["next_update"] == 5
    manifest.write_text(json.dumps([entry, entry]))
    with pytest.raises(ValueError, match="unique"):
        inv.load_checkpoint_list(manifest)
    entry["expected_update"] = 7
    manifest.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="Expected checkpoint update"):
        inv.load_checkpoint_list(manifest)


def test_three_model_run_shares_inputs_and_replaces_adapter(tmp_path, monkeypatch):
    import csv
    import peft

    pipe, cfg, out = setup(tmp_path, monkeypatch)
    cfg.sampler = "euler"
    cfg.seed_offsets = [0]
    loaded = []

    def replace(policy, state, adapter_name):
        assert adapter_name == "default"
        loaded.append(state["weight"].item())
        policy.weight.data.fill_(state["weight"].item())

    monkeypatch.setattr(peft, "set_peft_model_state_dict", replace)
    summary = inv.run_investigation(
        pipe,
        Reward(),
        Judge(),
        cfg,
        ["first", "second"],
        out,
        Scheduler,
        checkpoint_states={"step5": {"weight": torch.tensor(5.0)}, "step7": {"weight": torch.tensor(7.0)}},
    )
    assert loaded == [5.0, 7.0, 5.0, 7.0]
    assert summary["num_videos"] == 6
    for start in (0, 3):
        base, five, seven = pipe.calls[start : start + 3]
        assert base[3] and not five[3] and not seven[3]
        assert all(torch.equal(base[j], five[j]) and torch.equal(base[j], seven[j]) for j in range(3))
    with (out / "review/preferences.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 4


def test_native_three_model_tables_and_video_verification(tmp_path, monkeypatch):
    import imageio.v2 as imageio
    import peft
    import wandb

    pipe, cfg, out = setup(tmp_path, monkeypatch)
    cfg.sampler = "native"
    cfg.prompt_start = 24
    cfg.seed_offsets = [0]
    cfg.compare_checkpoints = True
    cfg.table_logging = True
    cfg.report_prompt_strata = True
    cfg.verify_videos = True
    cfg.fps = 24
    loaded = []

    def replace(policy, state, adapter_name):
        loaded.append(state["weight"].item())
        policy.weight.data.fill_(state["weight"].item())

    monkeypatch.setattr(peft, "set_peft_model_state_dict", replace)
    run = wandb.init(project="diffusionopsd-test", mode="offline", dir=str(out))
    try:
        summary = inv.run_investigation(
            pipe,
            Reward(),
            Judge(),
            cfg,
            ["first", "second"],
            out,
            Scheduler,
            run,
            checkpoint_states={"step7": {"weight": torch.tensor(0.07)}, "step50": {"weight": torch.tensor(0.5)}},
        )
        assert run.summary["investigation/num_comparisons"] == 6
        assert not any(k.startswith("investigation/means/") for k in run.summary._as_dict())
    finally:
        run.finish()
    assert loaded == [0.07, 0.5, 0.07, 0.5] or loaded == pytest.approx([0.07, 0.5, 0.07, 0.5])
    assert summary["num_videos"] == 6
    assert "step50-step7" in summary["differences"]
    for start in (0, 3):
        group = pipe.calls[start : start + 3]
        assert [(x[3], x[4]) for x in group] == [(True, True), (False, True), (False, True)]
        assert all(torch.equal(x[j], group[0][j]) for x in group for j in range(3))
    rows = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert {(r["prompt_index"], r["seed"], r["prompt"]) for r in rows} == {(24, 24, "first"), (25, 25, "second")}
    assert all(r["video_verified"] and r["fps"] == 24 for r in rows)
    with imageio.get_reader(str(out / rows[0]["video"])) as reader:
        assert reader.get_meta_data()["fps"] == 24
        assert reader.count_frames() == 5
    table_paths = list((out / "wandb").glob("offline-run-*/files/media/table/**/*.table.json"))
    assert len(table_paths) == 4
    pair_table = next(json.loads(p.read_text()) for p in table_paths if "pairs_" in p.name)
    assert len(pair_table["data"]) == 6
    assert "reference_video" in pair_table["columns"]
    with pytest.raises(ValueError, match="frames"):
        inv.verify_video(out / rows[0]["video"], 6, 32, 32, 24)
    with pytest.raises(ValueError, match="metadata"):
        inv.verify_video(out / rows[0]["video"], 5, 32, 32, 8)


def test_full_prompt_strata_equal_weighting():
    rows = [
        dict(condition=c, prompt_index=i, seed=i, **{k: float(i) for k in inv.SCORES})
        for i in range(64)
        for c in ("base", "step7", "step50")
    ]
    summary = inv.prompt_strata(
        rows,
        ("base", "step7", "step50"),
        [("step7-base", "step7", "base"), ("step50-base", "step50", "base"), ("step50-step7", "step50", "step7")],
    )
    assert [s["num_prompts"] for s in summary.values()] == [2, 14, 48]
    assert summary["prompts_17_64"]["means"]["base"]["geo"] == 39.5
