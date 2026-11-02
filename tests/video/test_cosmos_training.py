from types import SimpleNamespace

import pytest
import torch

from diffusionopsd.video.cosmos_training import fit_queries, load_training_state, save_training_state


def test_rejected_prompt_does_not_block_next_prompt_or_consume_update(tmp_path):
    from diffusionopsd.video.cosmos_training import PromptCursor

    sampler = PromptCursor(num_prompts=3)
    model = QueryModel()
    optimizer = torch.optim.AdamW([model.weight], lr=0.01)
    before = model.weight.detach().clone()
    index, nonce = sampler.take()
    assert (index, nonce) == (0, 0)
    assert not sampler.record_batch(retained=0)
    assert torch.equal(model.weight, before)
    index, nonce = sampler.take()
    assert (index, nonce) == (1, 1)
    assert sampler.record_batch(retained=1)
    record = SimpleNamespace(y0=torch.full((1, 1, 3, 2, 2), 0.5))
    fit_queries(model, [record], [.7], lambda y: -(y - 1).square().flatten(1).mean(1), optimizer, config())
    assert not torch.equal(model.weight, before)
    path = tmp_path / "state.pt"
    save_training_state(path, adapters={}, optimizer=optimizer.state_dict(), update=1,
                        config={"model": "nano"}, extra={"prompt_cursor": sampler.state_dict()})
    state = load_training_state(path, config={"model": "nano"})
    assert state["update"] == 1
    restored = PromptCursor(**state["extra"]["prompt_cursor"])
    assert restored.take() == sampler.take() == (2, 2)


def test_identity_gate_failure_is_bounded_by_one_full_prompt_pass():
    from diffusionopsd.video.cosmos_training import PromptCursor

    sampler = PromptCursor(num_prompts=2)
    assert sampler.take() == (0, 0)
    assert not sampler.record_batch(retained=0)
    assert sampler.take() == (1, 1)
    with pytest.raises(RuntimeError, match="full pass"):
        sampler.record_batch(retained=0)


def test_shared_gpu_layout_validates_actual_configured_devices():
    from diffusionopsd.video.cosmos_training import validate_cuda_devices

    validate_cuda_devices(["cuda:0", "cuda:1", "cuda:1", "cuda:1"], visible_count=2)
    validate_cuda_devices(["cuda", "cuda:1", "cuda:2"], visible_count=3)
    with pytest.raises(ValueError, match="3 visible"):
        validate_cuda_devices(["cuda:0", "cuda:2"], visible_count=2)
    with pytest.raises(ValueError, match="CUDA"):
        validate_cuda_devices(["cpu", "cuda:0"], visible_count=2)


class QueryModel:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def prediction(self, record):
        return record.y0 + self.weight


def config():
    return SimpleNamespace(beta=1.0, rho=0.1, n_ascent=2, eta=1.0, max_grad_norm=1.0, loss_scale=5.0)


def test_real_branch_loss_updates_policy_with_reward_targets():
    model = QueryModel()
    opt = torch.optim.AdamW([model.weight], lr=0.01)
    record = SimpleNamespace(y0=torch.full((1, 1, 3, 2, 2), 0.5))
    before = model.weight.detach().clone()
    stages = []
    metrics = fit_queries(
        model, [record], [0.7], lambda y: -(y - 1).square().flatten(1).mean(1), opt, config(),
        progress=lambda stage, index: stages.append((stage, index)),
    )
    assert model.weight != before
    assert metrics["grad_norm"] > 0
    assert metrics["target_grad_norm"] > 0
    assert metrics["retained"] == 1
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert stages[0] == ("target_forward", 0)
    assert ("target_backward", 0) in stages and ("policy_backward", 0) in stages
    assert stages[-1] == ("optimizer_step", None)


def test_nonfinite_reward_prevents_optimizer_update():
    model = QueryModel()
    opt = torch.optim.AdamW([model.weight], lr=0.01)
    record = SimpleNamespace(y0=torch.ones(1, 1, 3, 2, 2))
    before = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="finite"):
        fit_queries(model, [record], [0.5], lambda y: y.flatten(1).mean(1) * float("nan"), opt, config())
    assert torch.equal(model.weight, before)


def test_empty_batch_rejected():
    model = QueryModel()
    opt = torch.optim.AdamW([model.weight])
    with pytest.raises(ValueError, match="queries"):
        fit_queries(model, [], [], lambda y: y.sum(), opt, config())


def test_checkpoint_roundtrip_rng_and_config_guard(tmp_path):
    path = tmp_path / "state.pt"
    torch.manual_seed(42)
    state = {"default": {"weight": torch.tensor([1.0])}, "old": {"weight": torch.tensor([0.9])}}
    save_training_state(path, adapters=state, optimizer={"state": {}}, update=3, config={"model": "nano"})
    expected_next_noise = torch.randn(3)
    loaded = load_training_state(path, config={"model": "nano"})
    assert loaded["update"] == 3
    assert torch.equal(loaded["adapters"]["old"]["weight"], state["old"]["weight"])
    assert torch.equal(torch.randn(3), expected_next_noise)
    with pytest.raises(ValueError, match="configuration"):
        load_training_state(path, config={"model": "different"})
