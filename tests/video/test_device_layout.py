from types import SimpleNamespace
from pathlib import Path
import re

import pytest
import torch

from diffusionopsd.video.device_layout import InferenceReplica, validate_two_gpu_layout, encode_wan_prompt


def test_module_device_fallback_only_applies_to_stateless_modules():
    from diffusionopsd.video.device_layout import module_device
    assert module_device(lambda x: x, fallback="cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="stateless"):
        module_device(lambda x: x)
    model = torch.nn.Module()
    model.register_buffer("state", torch.zeros(1, device="meta"))
    assert module_device(model, fallback="cpu") == torch.device("meta")
    assert module_device(torch.nn.Linear(1, 1), fallback="meta") == torch.device("cpu")


@pytest.mark.parametrize("count", [0, 1, 3, 4])
def test_worker_requires_exactly_two_visible_gpus(count):
    with pytest.raises(ValueError, match="exactly two"):
        validate_two_gpu_layout("cuda:0", ["cuda:1"] * 3, visible_count=count)


@pytest.mark.parametrize("training,frozen", [("cuda:1", "cuda:1"), ("cuda:0", "cuda:0"),
                                            ("cuda:0", "cuda:2"), ("cuda:0", "cpu")])
def test_invalid_component_placement_rejected(training, frozen):
    with pytest.raises(ValueError):
        validate_two_gpu_layout(training, [frozen], visible_count=2)


def test_valid_worker_layout():
    validate_two_gpu_layout("cuda:0", ["cuda:1"] * 4, visible_count=2)


def test_frozen_replica_sync_is_independent_and_never_trains():
    policy = torch.nn.Linear(2, 2)
    replica = InferenceReplica(policy, device="cpu", lora=False)
    assert not any(p.requires_grad for p in replica.model.parameters())
    initial = replica.model.weight.clone()
    optimizer = torch.optim.SGD(policy.parameters(), lr=.1)
    policy(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    assert torch.equal(replica.model.weight, initial)
    replica.sync(policy)
    assert torch.equal(replica.model.weight, policy.weight)
    assert replica.model.weight.data_ptr() != policy.weight.data_ptr()
    assert not replica.model.training
    assert all(p.grad is None for p in replica.model.parameters())


def test_replica_copies_parameters_to_destination_before_deepcopy():
    class DestinationOnlyParameter(torch.nn.Parameter):
        def __deepcopy__(self, memo):
            raise AssertionError("Do not duplicate parameters on the training GPU")

    policy = torch.nn.Linear(2, 2)
    policy.weight = DestinationOnlyParameter(policy.weight.detach())
    replica = InferenceReplica(policy, device="cpu", lora=False)
    assert torch.equal(replica.model.weight, policy.weight)
    assert replica.model.weight.data_ptr() != policy.weight.data_ptr()


def test_replica_active_context_restores_idle_state_after_failure():
    policy = torch.nn.Linear(2, 2)
    replica = InferenceReplica(policy, device="cpu", lora=False)
    expected = policy(torch.ones(1, 2)).detach()
    with pytest.raises(RuntimeError, match="generation failed"):
        with replica.active_on("cpu") as model:
            assert model is replica.model
            assert torch.equal(model(torch.ones(1, 2)), expected)
            raise RuntimeError("generation failed")
    assert all(p.device.type == "cpu" and not p.requires_grad for p in replica.model.parameters())
    assert all(p.requires_grad for p in policy.parameters())


@pytest.mark.skipif(torch.cuda.device_count() != 2, reason="requires an isolated two-GPU allocation")
def test_replica_stages_to_cpu_between_gpu_generation_calls():
    policy = torch.nn.Linear(2, 2).to("cuda:0")
    replica = InferenceReplica(policy, device="cpu", lora=False)
    for fail in (False, True):
        try:
            with replica.active_on("cuda:1") as model:
                assert all(p.device == torch.device("cuda:1") for p in model.parameters())
                expected = policy(torch.ones(1, 2, device="cuda:0")).detach().cpu()
                assert torch.allclose(model(torch.ones(1, 2, device="cuda:1")).cpu(), expected)
                if fail:
                    raise RuntimeError("generation failed")
        except RuntimeError:
            if not fail:
                raise
        assert all(p.device.type == "cpu" and not p.requires_grad for p in replica.model.parameters())
        assert all(p.device == torch.device("cuda:0") for p in policy.parameters())


def test_prompt_encoder_runs_on_frozen_device_and_returns_policy_inputs():
    calls = []

    def encode(*args, **kwargs):
        calls.append(kwargs["device"])
        return torch.ones(1, 2), torch.zeros(1, 2)

    pipe = SimpleNamespace(text_encoder=SimpleNamespace(device=torch.device("cpu")), encode_prompt=encode)
    pe, ne = encode_wan_prompt(pipe, "prompt", negative_prompt="", device="cpu")
    assert calls == [torch.device("cpu")]
    assert pe.device == ne.device == torch.device("cpu")


@pytest.mark.parametrize("name", ["wan_video", "wan22_ti2v", "wan22_ti2v_debug", "wan22_ti2v_epoch",
                                  "wan22_ti2v_epoch_two_gpu", "wan22_logging_smoke", "cosmos_nano",
                                  "cosmos_logging_smoke"])
def test_all_training_presets_have_one_policy_and_one_frozen_device(name):
    import importlib
    cfg = importlib.import_module(f"config.{name}").get_config()
    assert cfg.policy_device == "cuda:0"
    assert cfg.inference_device == cfg.vae_device == cfg.reward_device == "cuda:1"
    judge = cfg.judge_device if name.startswith("cosmos") else cfg.judge.device
    assert judge == "cuda:1"


def test_training_slurm_requests_only_paired_gpus():
    root = Path(__file__).resolve().parents[2]
    for path in (root / "scripts").glob("train_opsd_video_*.slurm"):
        text = path.read_text()
        for count in re.findall(r"#SBATCH --(?:gpus-per-node=a100:|gres=gpu:a100:)(\d+)", text):
            assert int(count) % 2 == 0, path
        if "srun " in text:
            assert "--gpus-per-task=a100:2" in text, path


def test_lora_replica_sync_preserves_active_policy_and_freezes_selected_adapter():
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    policy = get_peft_model(torch.nn.Sequential(torch.nn.Linear(2, 2)), LoraConfig(r=1, target_modules=["0"]))
    policy.add_adapter("old", LoraConfig(r=1, target_modules=["0"]))
    policy.set_adapter("default")
    replica = InferenceReplica(policy, device="cpu", lora=True)
    with torch.no_grad():
        for name, p in policy.named_parameters():
            if "lora_" in name:
                p.add_(.1)
    replica.sync(policy)
    replica.select("old")
    assert policy.active_adapters == ["default"]
    assert all(not p.requires_grad for p in replica.model.parameters())
    for key, value in get_peft_model_state_dict(policy, adapter_name="old").items():
        assert torch.equal(value, get_peft_model_state_dict(replica.model, adapter_name="old")[key])
    policy(torch.ones(1, 2)).sum().backward()
    assert any(p.grad is not None for p in policy.parameters())
    assert all(p.grad is None for p in replica.model.parameters())


@pytest.mark.skipif(torch.cuda.device_count() != 2, reason="requires an isolated two-GPU allocation")
def test_cuda_query_replay_preserves_policy_gradients_and_frozen_isolation():
    from diffusionopsd.video.cosmos_rollout import CosmosRollout, CosmosQuery
    from diffusionopsd.video.cosmos_training import fit_queries
    from diffusionopsd.video.wan_clean_output import WanRollout

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(.12))

        @property
        def dtype(self):
            return self.scale.dtype

        def forward(self, vision_tokens=None, hidden_states=None, **kwargs):
            value = hidden_states if vision_tokens is None else vision_tokens[0]
            assert value.device == self.scale.device
            if vision_tokens is None:
                assert kwargs["encoder_hidden_states"].device == self.scale.device
                return (self.scale * value.sin(),)
            assert kwargs["input_ids"].device == self.scale.device
            return [self.scale * value.sin()], None, None

    policy = Model().to("cuda:0")
    inference = InferenceReplica(policy, device="cuda:1", lora=False)
    pipe = SimpleNamespace(transformer=inference.model,
                           scheduler=SimpleNamespace(config=SimpleNamespace(prediction_type="flow_prediction")))
    roll = CosmosRollout(pipe, num_steps=4, guidance_scale=1.)
    z = torch.full((1, 1, 1, 2, 2), .2, device="cuda:1")
    rec = CosmosQuery(z, z, .3, 0, z * .1, torch.tensor([.3, 0.]), torch.tensor([300.]),
                      [{"vision_tokens": [z], "input_ids": torch.ones(1, device="cuda:1")}], None, False)
    cfg = SimpleNamespace(rho=.1, n_ascent=2, eta=1., beta=1., loss_scale=5., max_grad_norm=1.)
    before = policy.scale.clone()
    metrics = fit_queries(roll, [rec], [.75], lambda y: y.sin().flatten(1).mean(1),
                          torch.optim.SGD(policy.parameters(), lr=.1), cfg, policy=policy)
    assert metrics["grad_norm"] > 0 and not torch.equal(before, policy.scale)
    assert all(p.device == torch.device("cuda:0") for p in policy.parameters())
    assert all(p.device == torch.device("cuda:1") and not p.requires_grad and p.grad is None
               for p in inference.model.parameters())
    inference.sync(policy)
    assert torch.equal(inference.model.scale.cpu(), policy.scale.cpu())
    pipe.config = SimpleNamespace(expand_timesteps=False)
    wan = WanRollout(pipe, 4, 1.)
    embed = torch.ones(1, 2, device="cuda:1")
    velocity = wan.velocity(z, torch.tensor(.3, device="cuda:1"), embed, embed, transformer=policy)
    assert velocity.device == z.device
    velocity.sum().backward()
    assert policy.scale.grad is not None and policy.scale.grad.abs() > 0
