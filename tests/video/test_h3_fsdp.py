"""CPU contract checks; opt-in CUDA tests run with torchrun (three/five ranks)."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.filterwarnings(
    "ignore:Already found a `peft_config` attribute:UserWarning"
)


@pytest.mark.parametrize('values,expected', [
    ({}, {'collective_timeout_s': 3600., 'service_timeout_s': 600.}),
    ({'collective_timeout_s': 4200., 'service_timeout_s': 900.},
     {'collective_timeout_s': 4200., 'service_timeout_s': 900.}),
])
def test_resolve_h3_timeouts_for_legacy_and_explicit_configs(values, expected):
    from diffusionopsd.video.h3_fsdp import resolve_h3_timeouts
    assert resolve_h3_timeouts(SimpleNamespace(**values)) == expected


@pytest.mark.parametrize('name,value', [
    (name, value) for name in ('collective_timeout_s', 'service_timeout_s')
    for value in (float('nan'), float('inf'), 0., -1.)
] + [('collective_timeout_s', 600.), ('collective_timeout_s', 599.),
     ('service_timeout_s', 3600.), ('service_timeout_s', 4000.)])
def test_resolve_h3_timeouts_rejects_invalid_or_inconsistent_budgets(name, value):
    from diffusionopsd.video.h3_fsdp import resolve_h3_timeouts
    values = {name: value}
    if name == 'collective_timeout_s' and value in (599., 600.):
        values['service_timeout_s'] = 1.  # Judge readiness is an independent floor.
    with pytest.raises(ValueError, match='timeout'):
        resolve_h3_timeouts(SimpleNamespace(**values))


@pytest.mark.parametrize('timeout', [float('nan'), float('inf'), 0., -1.])
def test_distributed_timeout_validation_precedes_cuda(timeout, monkeypatch):
    from diffusionopsd.video.h3_fsdp import init_h3_distributed
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: pytest.fail('CUDA touched'))
    with pytest.raises(ValueError, match='timeout.*finite.*positive'):
        init_h3_distributed(policy_gpus=(0, 1), target_gpus=(2,), timeout_s=timeout)


def tiny_model(*, loading_probe=False):
    from diffusers import MiniMaxH3Transformer3DModel
    if loading_probe:
        # Larger than allocator rounding, with uneven FFN/audio projection rows.
        return MiniMaxH3Transformer3DModel(num_attention_heads=4, attention_head_dim=16,
            hidden_size=64, num_layers=2, num_refiner_layers=1, ffn_dim=97, in_channels=2,
            audio_in_channels=3, patch_size=(1, 1, 1), text_dim=5, freq_dim=8,
            time_embed_hidden_dim=64, time_embed_dim=32, rope_freq_dim=2)
    return MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=8,
        hidden_size=16, num_layers=2, num_refiner_layers=1, ffn_dim=24, in_channels=2,
        audio_in_channels=3, patch_size=(1, 1, 1), text_dim=5, freq_dim=8,
        time_embed_hidden_dim=16, time_embed_dim=8, rope_freq_dim=1)


def packed_inputs(device='cpu', seed=91):
    g = torch.Generator(device=device).manual_seed(seed)
    return dict(hidden_states=torch.randn(1, 2, 2, device=device, generator=g),
        audio_hidden_states=torch.randn(1, 2, 3, device=device, generator=g),
        encoder_hidden_states=torch.randn(1, 1, 5, device=device, generator=g),
        timestep=torch.tensor([0.5], device=device), timestep_indices=torch.zeros(5, dtype=torch.long, device=device),
        token_tags=torch.tensor([1, 0, 0, 2, 2], device=device),
        position_ids=torch.zeros(5, 3, device=device), video_indices=torch.tensor([1, 2], device=device),
        audio_indices=torch.tensor([3, 4], device=device), text_indices=torch.tensor([0], device=device))



def forbid_whole_checkpoint_reads(monkeypatch):
    """Guard loading only: checkpoint files stay CPU-sliced, never get_tensor."""
    import safetensors
    original_open = safetensors.safe_open

    class SlicedCPUCheckpoint:
        def __init__(self, filename, *args, **kwargs):
            assert kwargs.get('device', 'cpu') == 'cpu', 'checkpoint opened directly on CUDA'
            self.handle = original_open(filename, *args, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_slice(self, key):
            return self.handle.get_slice(key)

        def get_tensor(self, key):
            raise AssertionError(f'whole checkpoint tensor read forbidden during loading: {key}')

    monkeypatch.setattr(safetensors, 'safe_open', SlicedCPUCheckpoint)


def test_loading_guard_rejects_whole_checkpoint_reads(tmp_path, monkeypatch):
    import safetensors
    tiny_model().save_pretrained(tmp_path)
    forbid_whole_checkpoint_reads(monkeypatch)
    with safetensors.safe_open(tmp_path / 'diffusion_pytorch_model.safetensors',
                               framework='pt', device='cpu') as handle:
        with pytest.raises(AssertionError, match='whole checkpoint tensor read forbidden'):
            handle.get_tensor('proj_in.weight')
    with pytest.raises(AssertionError, match='opened directly on CUDA'):
        safetensors.safe_open(tmp_path / 'diffusion_pytorch_model.safetensors', framework='pt', device='cuda')

def test_native_attention_selection_excludes_refiner():
    from diffusionopsd.video.h3_fsdp import attention_lora_targets
    names = attention_lora_targets(tiny_model())
    assert len(names) == 8
    assert all(n.startswith('transformer_blocks.') and '.attn.' in n for n in names)
    assert names[:4] == ('transformer_blocks.0.attn.to_q', 'transformer_blocks.0.attn.to_k',
                        'transformer_blocks.0.attn.to_v', 'transformer_blocks.0.attn.to_out.0')


def test_streamed_loading_matches_unsharded_weights(tmp_path):
    from diffusionopsd.video.h3_fsdp import load_h3_policy, H3Devices
    torch.manual_seed(12)
    original = tiny_model()
    original.save_pretrained(tmp_path, max_shard_size='8KB')
    policy = load_h3_policy(tmp_path, 'test-local', H3Devices.cpu(), rank=2, alpha=4)
    assert policy.checkpoint_missing == () and policy.checkpoint_unexpected == ()
    original.eval()
    policy.model.eval()
    with torch.no_grad():
        expected, actual = original(**packed_inputs()), policy.model(**packed_inputs())
    torch.testing.assert_close(expected.sample, actual.sample)
    torch.testing.assert_close(expected.audio_sample, actual.audio_sample)
    torch.testing.assert_close(original.rope.inv_freq, policy.model.rope.inv_freq)
    assert policy.model.rope.inv_freq.dtype == torch.float32
    assert policy.model.gradient_checkpointing
    assert not any(p.is_meta for p in policy.model.parameters())
    for name, value in original.state_dict().items():
        loaded = dict(policy.model.named_parameters()).get(name.replace('.attn.to_', '.attn.to_'))
        # PEFT inserts base_layer only at the selected native attention projections.
        if loaded is None:
            prefix, leaf = name.rsplit('.', 1)
            loaded = dict(policy.model.named_parameters())[prefix + '.base_layer.' + leaf]
        torch.testing.assert_close(value, loaded)


def test_configured_seed_controls_adapters_and_first_native_update(tmp_path, monkeypatch):
    from diffusionopsd.video.h3_fsdp import H3Devices, load_h3_policy
    tiny_model().save_pretrained(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('foreign CUDA RNG access')
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', forbidden)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', forbidden)
    monkeypatch.setattr(torch.cuda, 'device_count', forbidden)
    def run(ambient, seed):
        torch.random.default_generator.manual_seed(ambient)
        before_rng = torch.get_rng_state().clone()
        policy = load_h3_policy(tmp_path, 'test-local', H3Devices.cpu(), rank=2, alpha=4, seed=seed)
        assert torch.equal(before_rng, torch.get_rng_state())
        initial = {name: p.detach().clone() for name, p in policy.model.named_parameters() if '.lora_' in name}
        optimizer = torch.optim.AdamW(policy.trainable_parameters, lr=.01)
        output = policy.model(**packed_inputs())
        (output.sample.square().sum()+output.audio_sample.square().sum()).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.trainable_parameters)
        optimizer.step()
        return initial, [p.detach().clone() for p in policy.trainable_parameters]
    first, first_update = run(123, 42)
    second, second_update = run(456, 42)
    different, _ = run(789, 43)
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert all(torch.equal(left, right) for left, right in zip(first_update, second_update))
    assert any(not torch.equal(first[name], different[name]) for name in first if '.lora_A.default.' in name)


def test_adapter_switch_preserves_trainability(tmp_path):
    from diffusionopsd.video.h3_fsdp import load_h3_policy, H3Devices
    tiny_model().save_pretrained(tmp_path)
    policy = load_h3_policy(tmp_path, 'test-local', H3Devices.cpu(), rank=2, alpha=4)
    before = {n: p.requires_grad for n, p in policy.model.named_parameters()}
    for name in ('old', 'default', 'old', 'default'):
        policy.select_adapter(name)
        assert before == {n: p.requires_grad for n, p in policy.model.named_parameters()}
    assert all(p.dtype == torch.float32 for p in policy.trainable_parameters)
    for p in policy.trainable_parameters:
        p.data.add_(1)
    policy.ema_(0)
    params = dict(policy.model.named_parameters())
    for name, p in params.items():
        if '.lora_' in name and '.old.' in name:
            torch.testing.assert_close(p, params[name.replace('.old.', '.default.')])


def test_gradient_normalization_and_finite_validation():
    from diffusionopsd.video.h3_fsdp import reduce_adapter_gradients_
    p = torch.nn.Parameter(torch.ones(2))
    p.grad = torch.tensor([4., 8.])
    reduce_adapter_gradients_((p,), None, 4)
    torch.testing.assert_close(p.grad, torch.tensor([1., 2.]))
    p.grad.fill_(float('nan'))
    with pytest.raises(FloatingPointError):
        reduce_adapter_gradients_((p,), None, 4)
    with pytest.raises(ValueError):
        reduce_adapter_gradients_((p,), None, 0)


@pytest.fixture(scope='module')
def h3_gpu_devices():
    """One rendezvous generation for all cases in this torchrun invocation.

    Destroy/reinit under env:// can reuse stale store keys while service ranks
    advance ahead of policy NCCL teardown. Keep every group alive until all
    parametrized cases and their per-case control barriers have completed.
    """
    from diffusionopsd.video.h3_fsdp import init_h3_distributed
    import torch.distributed as dist
    count = int(os.environ['LOCAL_WORLD_SIZE']) - 1
    devices = init_h3_distributed(policy_gpus=tuple(range(count)), target_gpus=(count,))
    yield devices
    dist.barrier(group=devices.control_group)
    dist.destroy_process_group()


@pytest.mark.parametrize('mixed_precision', [False, True], ids=['fp32', 'mixed-bf16-fp32'])
@pytest.mark.h3_gpu
@pytest.mark.skipif(os.environ.get('H3_RUN_GPU_TESTS') != '1', reason='set H3_RUN_GPU_TESTS=1 under torchrun')
def test_frozen_fsdp_backbone_propagates_ignored_lora_gradients(
        tmp_path, monkeypatch, mixed_precision, h3_gpu_devices):
    from diffusionopsd.video.h3_fsdp import load_h3_policy, reduce_adapter_gradients_
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    devices = h3_gpu_devices
    count = devices.policy_count
    # Shared filesystem fixture chosen outside pytest's per-process temporary directory.
    path = Path(os.environ['H3_TEST_CHECKPOINT']) / ('mixed' if mixed_precision else 'fp32')
    if devices.rank == 0:
        torch.manual_seed(17)
        fixture = tiny_model(loading_probe=True)
        if mixed_precision:
            for name, param in fixture.named_parameters():
                if not any(part in name for part in fixture._keep_in_fp32_modules):
                    param.data = param.data.bfloat16()
        fixture.save_pretrained(path)
        del fixture
    dist.barrier(group=devices.control_group)
    if devices.role == 'policy':
        # The guard and peak assertion are scoped BEFORE the intentionally full
        # unsharded CUDA reference. A regression to whole-model materialization
        # must fail even if output/gradient parity would still pass.
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with monkeypatch.context() as patch:
            forbid_whole_checkpoint_reads(patch)
            policy = load_h3_policy(path, 'test-local', devices, rank=2, alpha=4)
        torch.cuda.synchronize()
        loading_peak = torch.cuda.max_memory_allocated() - before
        full_base_bytes = sum(p.numel() * p.element_size() for n, p in policy.model.named_parameters()
                              if '.lora_' not in n)
        replicated_bytes = sum(p.numel() * p.element_size() for n, p in policy.model.named_parameters()
                               if '.lora_' in n)
        assert loading_peak < full_base_bytes + replicated_bytes, (
            f'loading allocated {loading_peak} bytes, sufficient for a whole frozen checkpoint '
            f'({full_base_bytes} bytes) plus replicated adapters ({replicated_bytes} bytes)')
        assert policy.checkpoint_missing == () and policy.checkpoint_unexpected == ()
        assert policy.model.proj_in.weight.dtype == torch.float32
        assert policy.model.proj_out.weight.dtype == torch.float32
        assert policy.model.transformer_blocks[0].attn.to_q.base_layer.weight.dtype == (
            torch.bfloat16 if mixed_precision else torch.float32)
        assert all(p.dtype == torch.float32 for n, p in policy.model.named_parameters() if '.lora_' in n)
        # FP32 projections and the root's norm/output parameters are also sharded.
        assert isinstance(policy.model.norm_out.linear.weight, DTensor)
        assert isinstance(policy.model.audio_proj_out.weight, DTensor)
        torch.testing.assert_close(policy.model.rope.inv_freq.cpu(),
            tiny_model(loading_probe=True).rope.inv_freq)
        reference = load_h3_policy(path, 'test-local', devices.cpu(), rank=2, alpha=4)
        reference.model.to(devices.device)
        # Identical initialized adapters make the independent reference meaningful.
        reference.model.load_state_dict({n: p.detach().cpu() for n, p in policy.model.named_parameters()
            if '.lora_' in n}, strict=False)
        with torch.no_grad():
            output = policy.model(**packed_inputs(devices.device))
            ref_output = reference.model(**packed_inputs(devices.device))
            torch.testing.assert_close(output.sample, ref_output.sample)
            torch.testing.assert_close(output.audio_sample, ref_output.audio_sample)
        optimizer = torch.optim.SGD(policy.trainable_parameters, lr=0.01)
        ref_optimizer = torch.optim.SGD(reference.trainable_parameters, lr=0.01)
        for seed in range(4):
            output = reference.model(**packed_inputs(devices.device, seed))
            ((output.sample.square().sum() + output.audio_sample.square().sum()) / 4).backward()
        for seed in range(devices.policy_index, 4, count):
            output = policy.model(**packed_inputs(devices.device, seed))
            (output.sample.square().sum() + output.audio_sample.square().sum()).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.trainable_parameters)
        reduce_adapter_gradients_(policy.trainable_parameters, devices.all_policy_group, 4)
        # BF16 matmul backward rounds each independent candidate before SUM.
        grad_atol, grad_rtol = (2e-3, 2e-2) if mixed_precision else (2e-5, 2e-4)
        for p, ref in zip(policy.trainable_parameters, reference.trainable_parameters):
            torch.testing.assert_close(p.grad, ref.grad, atol=grad_atol, rtol=grad_rtol)
        optimizer.step()
        ref_optimizer.step()
        for p, ref in zip(policy.trainable_parameters, reference.trainable_parameters):
            assert not isinstance(p, DTensor)
            torch.testing.assert_close(p, ref, atol=grad_atol * 0.01, rtol=grad_rtol)
            replicas = [torch.empty_like(p) for _ in range(count)]
            dist.all_gather(replicas, p, group=devices.policy_group)
            assert all(torch.equal(p, replica) for replica in replicas)
        assert all(p.grad is None for n, p in policy.model.named_parameters() if '.lora_' not in n)
        assert all(isinstance(p, DTensor) for n, p in policy.model.named_parameters() if '.lora_' not in n)
        for adapter in ('old', 'default', 'old', 'default'):
            policy.select_adapter(adapter)
        policy.model.eval()
        with torch.no_grad():
            policy.model(**packed_inputs(devices.device))
        policy.model.train()
        policy.model(**packed_inputs(devices.device)).sample.sum().backward()
    dist.barrier(group=devices.control_group)


def test_streamed_loading_preserves_mixed_checkpoint_dtypes(tmp_path):
    from diffusionopsd.video.h3_fsdp import H3Devices, load_h3_policy
    original = tiny_model()
    for name, param in original.named_parameters():
        if not any(part in name for part in original._keep_in_fp32_modules):
            param.data = param.data.bfloat16()
    original.save_pretrained(tmp_path)
    policy = load_h3_policy(tmp_path, 'test-local', H3Devices.cpu(), rank=2, alpha=4)
    expected = dict(original.named_parameters())
    for name, param in policy.model.named_parameters():
        if '.lora_' in name:
            assert param.dtype == torch.float32
        else:
            assert param.dtype == expected[name.replace('.base_layer.', '.')].dtype
    assert policy.model.proj_in.weight.dtype == torch.float32
    assert policy.model.transformer_blocks[0].attn.to_q.base_layer.weight.dtype == torch.bfloat16
    assert policy.model.rope.inv_freq.dtype == torch.float32


def test_streamed_loading_rejects_checkpoint_key_mismatch(tmp_path):
    from diffusionopsd.video.h3_fsdp import H3Devices, load_h3_policy
    from safetensors.torch import save_file
    model = tiny_model()
    model.save_config(tmp_path)
    state = model.state_dict()
    state.pop('proj_in.weight')
    state['alien.weight'] = torch.zeros(1)
    save_file(state, tmp_path / 'diffusion_pytorch_model.safetensors')
    with pytest.raises(ValueError, match='checkpoint key mismatch'):
        load_h3_policy(tmp_path, 'test-local', H3Devices.cpu(), rank=2, alpha=4)
