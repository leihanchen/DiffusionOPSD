"""Frozen native H3 FSDP2 backbone with explicitly replicated FP32 adapters.

The policy mesh is node-local. Adapter reductions use ``all_policy_group``;
service ranks only use the Gloo control groups and never enter policy collectives.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
import inspect
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint


DEFAULT_H3_COLLECTIVE_TIMEOUT_S = 3600.
DEFAULT_H3_SERVICE_TIMEOUT_S = 600.
# IsolatedVideoJudge permits 600 seconds for readiness, after earlier loaders.
H3_JUDGE_READINESS_TIMEOUT_S = 600.


def resolve_h3_timeouts(cfg) -> dict[str, float]:
    """Validate shared role-work budgets before groups, CUDA, or asset access.

    These floors prevent premature peer abandonment of one permitted service
    wait; sequential loading, requests, and policy work still need timing evidence.
    """
    timeouts = {}
    for name, default in (('collective_timeout_s', DEFAULT_H3_COLLECTIVE_TIMEOUT_S),
                          ('service_timeout_s', DEFAULT_H3_SERVICE_TIMEOUT_S)):
        try:
            value = float(getattr(cfg, name, default))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f'{name} timeout must be finite and positive') from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} timeout must be finite and positive')
        timeouts[name] = value
    if timeouts['collective_timeout_s'] <= max(timeouts['service_timeout_s'], H3_JUDGE_READINESS_TIMEOUT_S):
        raise ValueError('collective_timeout_s must exceed service_timeout_s and the 600s judge-readiness timeout')
    return timeouts


@dataclass(frozen=True)
class H3Devices:
    rank: int
    role: str
    policy_count: int
    reward_ranks: tuple[int, ...]
    device: torch.device
    policy_group: object = None
    control_group: object = None
    all_policy_group: object = None
    node_control_group: object = None
    policy_index: int = 0
    num_nodes: int = 1
    node_index: int = 0

    @property
    def node_rank(self):
        return self.node_index

    @property
    def node_count(self):
        return self.num_nodes

    @property
    def node_policy_rank(self):
        return self.policy_index

    @property
    def node_policy_group(self):
        return self.policy_group

    @classmethod
    def cpu(cls):
        """Unsharded CPU loading for serialization/reference checks only."""
        return cls(0, 'policy', 1, (), torch.device('cpu'))


def init_h3_distributed(*, policy_gpus: tuple[int, ...], target_gpus: tuple[int, ...],
                        timeout_s: float = DEFAULT_H3_COLLECTIVE_TIMEOUT_S) -> H3Devices:
    """Build every group on every rank in deterministic global order.

    Indices are scheduler-visible local CUDA indices; global ranks follow the
    torchrun node-major layout, with LOCAL_WORLD_SIZE ranks on every node.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('collective timeout must be finite and positive')
    timeout = timedelta(seconds=timeout_s)
    n = len(policy_gpus) + len(target_gpus)
    if not policy_gpus or not target_gpus or len(set(policy_gpus + target_gpus)) != n:
        raise ValueError('policy and target indices must be nonempty, unique and disjoint')
    if sorted(policy_gpus + target_gpus) != list(range(n)):
        raise ValueError('policy and target indices must partition scheduler-visible devices 0..n-1')
    local_size = int(os.environ.get('LOCAL_WORLD_SIZE', n))
    world_size = int(os.environ.get('WORLD_SIZE', n))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_size != n or world_size % n or torch.cuda.device_count() < n:
        raise ValueError('expected n=P+R visible devices and ranks per node')
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group('gloo', timeout=timeout)
    rank = dist.get_rank()
    if dist.get_world_size() != world_size or rank % n != local_rank:
        raise ValueError('expected a node-major torchrun rank layout')
    nodes, node = world_size // n, rank // n
    control = dist.new_group(list(range(world_size)), backend='gloo', timeout=timeout)
    all_ranks = [i * n + p for i in range(nodes) for p in policy_gpus]
    all_policy = dist.new_group(all_ranks, backend='nccl', timeout=timeout)
    policy_group = node_control = None
    for i in range(nodes):
        group = dist.new_group([i * n + p for p in policy_gpus], backend='nccl', timeout=timeout)
        cpu_group = dist.new_group(list(range(i * n, (i + 1) * n)), backend='gloo', timeout=timeout)
        if node == i:
            policy_group, node_control = group, cpu_group
    return H3Devices(rank, 'policy' if local_rank in policy_gpus else 'reward', len(policy_gpus),
        tuple(node * n + t for t in target_gpus), torch.device('cuda', local_rank), policy_group,
        control, all_policy, node_control, policy_gpus.index(local_rank) if local_rank in policy_gpus else -1,
        nodes, node)


def attention_lora_targets(model: nn.Module) -> tuple[str, ...]:
    """Select only the four native attention projections of each main block."""
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention
    names = []
    for i, block in enumerate(model.transformer_blocks):
        if not isinstance(block.attn, MiniMaxH3Attention):
            raise TypeError('expected native MiniMaxH3Attention')
        names.extend(f'transformer_blocks.{i}.attn.{leaf}' for leaf in ('to_q', 'to_k', 'to_v', 'to_out.0'))
    return tuple(names)


class H3Policy:
    def __init__(self, model, devices, missing=(), unexpected=()):
        self.model, self.devices = model, devices
        self.checkpoint_missing, self.checkpoint_unexpected = tuple(missing), tuple(unexpected)
        self._mask = {n: p.requires_grad for n, p in model.named_parameters()}
        self.trainable_parameters = tuple(p for _, p in sorted(model.named_parameters()) if p.requires_grad)

    def select_adapter(self, name):
        if name not in ('default', 'old'):
            raise ValueError(f'unknown adapter: {name}')
        self.model.set_adapter(name)
        # PEFT set_adapter changes requires_grad; behavior/evaluation must not do so.
        for n, p in self.model.named_parameters():
            p.requires_grad_(self._mask[n])

    @contextmanager
    def generation_adapter(self, name):
        """Select behavior/current generation or the frozen base, restoring state."""
        active = self.model.active_adapters
        previous = active() if callable(active) else active
        try:
            if name is None:
                self.model.disable_adapters()
            else:
                self.select_adapter(name)
            yield
        finally:
            if name is None:
                self.model.enable_adapters()
            if previous:
                self.select_adapter(previous[0] if isinstance(previous, list) else previous)

    @torch.no_grad()
    def ema_(self, decay):
        if not 0 <= decay <= 1:
            raise ValueError('EMA decay must be in [0, 1]')
        params = dict(self.model.named_parameters())
        for n, p in params.items():
            if '.lora_' in n and '.old.' in n:
                p.lerp_(params[n.replace('.old.', '.default.')], 1 - decay)


def _checkpoint_index(path):
    from safetensors import safe_open
    index = path / 'diffusion_pytorch_model.safetensors.index.json'
    if index.exists():
        mapping = json.loads(index.read_text())['weight_map']
        return {k: path / v for k, v in mapping.items()}
    files = sorted(path.glob('*.safetensors'))
    if not files:
        raise FileNotFoundError(f'no safetensors checkpoint at {path}')
    mapping = {}
    for file in files:
        with safe_open(file, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                if key in mapping:
                    raise ValueError(f'duplicate checkpoint key: {key}')
                mapping[key] = file
    return mapping


def _base_key(name):
    return name.replace('.base_layer.', '.')


def _stream_weights_(model, mapping):
    """Read only each DTensor's local slice, never a full checkpoint tensor on CUDA."""
    from safetensors import safe_open
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    loaded = set()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if '.lora_' in name:
                continue
            key = _base_key(name)
            with safe_open(mapping[key], framework='pt', device='cpu') as handle:
                view = handle.get_slice(key)
                if tuple(view.get_shape()) != tuple(param.shape):
                    raise ValueError(f'checkpoint shape mismatch: {key}')
                if isinstance(param, DTensor):
                    shape, offsets = compute_local_shape_and_global_offset(
                        param.shape, param.device_mesh, param.placements)
                    value = view[tuple(slice(offset, offset + size) for offset, size in zip(offsets, shape))]
                    local = param.to_local()
                    # FSDP may keep padded storage; only logical elements are checkpoint data.
                    if tuple(local.shape) != tuple(shape):
                        raise ValueError(f'invalid local shard/padding shape for {key}: {local.shape} != {shape}')
                    local.copy_(value)
                else:
                    param.copy_(view[:])
            loaded.add(key)
    return tuple(sorted(set(mapping) - loaded))


def load_h3_policy(model_path: Path, revision: str, devices: H3Devices, *, rank: int, alpha: int,
                   seed: int = 42) -> H3Policy:
    """Meta construct, shard frozen leaves bottom-up, then stream local weights.

    ``rank`` is the LoRA rank, not a distributed rank. The local checkpoint must
    already be downloaded; its revision marker is verified when present.
    """
    from diffusers import MiniMaxH3Transformer3DModel
    from peft import LoraConfig
    from safetensors import safe_open
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.device_mesh import DeviceMesh
    if devices.role != 'policy':
        raise ValueError('service ranks must not load or shard the policy')
    if not revision or rank <= 0 or alpha <= 0:
        raise ValueError('revision and positive LoRA rank/alpha are required')
    path = Path(model_path)
    marker = path / 'opsd_revision.txt'
    if marker.exists() and marker.read_text().strip() != revision:
        raise ValueError('local checkpoint revision does not match requested revision')
    if (path / 'transformer').is_dir():
        path = path / 'transformer'
    mapping = _checkpoint_index(path)
    config = MiniMaxH3Transformer3DModel.load_config(path)
    with torch.device('meta'):
        model = MiniMaxH3Transformer3DModel.from_config(config)
    expected = set(model.state_dict())
    missing, unexpected = tuple(sorted(expected - mapping.keys())), tuple(sorted(mapping.keys() - expected))
    if missing or unexpected:
        raise ValueError(f'checkpoint key mismatch; missing={missing}, unexpected={unexpected}')
    dtypes = {'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16}
    for name, param in model.named_parameters():
        with safe_open(mapping[name], framework='pt', device='cpu') as handle:
            dtype = dtypes[handle.get_slice(name).get_dtype()]
        param.data = param.data.to(dtype=dtype)
    model.requires_grad_(False)
    targets = list(attention_lora_targets(model))
    adapter_config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0., target_modules=targets,
                                init_lora_weights=True, bias='none')
    with torch.device('meta'):
        model.add_adapter(adapter_config, adapter_name='default')
        model.add_adapter(adapter_config, adapter_name='old')
    ignored = {p for n, p in model.named_parameters() if '.lora_' in n}
    for p in ignored:
        p.data = p.data.float()
    for n, p in model.named_parameters():
        p.requires_grad_('.lora_' in n and '.default.' in n)
    if devices.device.type == 'cuda':
        if 'ignored_params' not in inspect.signature(fully_shard).parameters:
            raise RuntimeError('H3 requires FSDP2 fully_shard(ignored_params=...)')
        ranks = dist.get_process_group_ranks(devices.policy_group)
        mesh = DeviceMesh.from_group(devices.policy_group, 'cuda', mesh=ranks)
        # Mixed checkpoint dtypes cannot share an FSDP param group. Leaf groups
        # preserve FP32 projection heads and BF16 attention/MLP weights exactly.
        for _, module in reversed(list(model.named_modules())):
            direct = list(module.parameters(recurse=False))
            if direct and any(p not in ignored for p in direct):
                fully_shard(module, mesh=mesh, ignored_params=ignored, reshard_after_forward=True)
        for block in model.token_refiner.refiner_blocks:
            fully_shard(block, mesh=mesh, ignored_params=ignored, reshard_after_forward=True)
        for block in model.transformer_blocks:
            fully_shard(block, mesh=mesh, ignored_params=ignored, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, ignored_params=ignored, reshard_after_forward=True)
    model.to_empty(device=devices.device)
    unexpected = _stream_weights_(model, mapping)
    # All constructor-derived buffers are rebuilt outside meta, without casting
    # the FP32 rotary frequencies to the backbone's checkpoint dtype.
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
    model.rope.inv_freq = MiniMaxH3RotaryPosEmbed(
        rope_freq_dim=config['rope_freq_dim'], rope_theta=config['rope_theta']).inv_freq.to(devices.device)
    # to_empty intentionally discarded meta initialization, so initialize adapters
    # explicitly and then copy behavior tensors from default before broadcasting.
    from peft.tuners.lora.layer import LoraLayer
    owned = []
    if devices.device.type == 'cuda':
        owned = [devices.device.index if devices.device.index is not None else torch.cuda.current_device()]
    # Fork only CPU and the owned GPU. Never seed/enumerate foreign CUDA generators.
    # Checkpoint restoration still happens later, after policy/optimizer construction.
    with torch.random.fork_rng(devices=owned):
        torch.random.default_generator.manual_seed(seed)
        for index in owned:
            torch.cuda.default_generators[index].manual_seed(seed)
        for module in model.modules():
            if isinstance(module, LoraLayer):
                module.reset_lora_parameters('default', init_lora_weights=True)
                for collection in (module.lora_A, module.lora_B):
                    collection['old'].weight.data.copy_(collection['default'].weight.data)
    if devices.device.type == 'cuda':
        src = dist.get_process_group_ranks(devices.all_policy_group)[0]
        for name, param in sorted(model.named_parameters()):
            if '.lora_' in name:
                dist.broadcast(param.data, src=src, group=devices.all_policy_group)
    model.enable_gradient_checkpointing(gradient_checkpointing_func=partial(checkpoint, use_reentrant=False))
    policy = H3Policy(model, devices, missing, unexpected)
    policy.select_adapter('default')
    return policy


def reduce_adapter_gradients_(parameters, group, kept_count: int) -> None:
    """SUM independent candidate gradients exactly once, then divide by global N.

    All policy ranks must call with the same deterministic parameter tuple, even
    ranks with zero kept candidates. No clipping is performed in this helper.
    """
    if kept_count <= 0:
        raise ValueError('kept_count must be positive')
    parameters = tuple(parameters)
    for p in parameters:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if group is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)
        p.grad.div_(kept_count)
    if any(not torch.isfinite(p.grad).all().item() for p in parameters):
        raise FloatingPointError('nonfinite synchronized H3 adapter gradient')
