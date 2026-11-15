"""Atomic CPU-only replicated LoRA snapshots; frozen shards reload from pinned assets.

All ranks call both APIs at an idle service boundary. Collective timeouts are owned
by H3Devices; a failed collective is fatal and must be handled by torchrun.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import random
import signal
import tempfile
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist

from diffusionopsd.video.h3_conditioning import validate_h3_topology

if TYPE_CHECKING:
    from diffusionopsd.video.h3_fsdp import H3Policy


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _gather(value, group):
    if not _distributed():
        return [value]
    values = [None]*dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def _checked(operation, group):
    value, error = None, None
    try:
        value = operation()
    except Exception as exception:
        error = f'{type(exception).__name__}: {exception}'
    errors = _gather(error, group)
    if any(errors):
        raise RuntimeError(f'H3 checkpoint failed: {errors}')
    return value


def _broadcast(value, group):
    values = [value]
    if _distributed():
        dist.broadcast_object_list(values, src=0, group=group)
    return values[0]


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return value


def _adapter_parameters(policy):
    return {name: parameter for name, parameter in policy.model.named_parameters()
            if '.lora_' in name and ('.default.' in name or '.old.' in name)}


def _digest(value):
    """Stable recursive tensor digest, independent of pickle storage identities."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
        digest.update(b'\0')
    visit(value)
    return digest.hexdigest()


def _rng_state():
    state = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state())
    # Never enumerate visible CUDA devices or initialize a foreign-role context.
    if torch.cuda.is_initialized():
        state['cuda'] = torch.cuda.get_rng_state(torch.cuda.current_device())
    return state


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'], torch.cuda.current_device())


def _atomic_bytes(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _checkpoint_groups(policy, policy_group, all_policy_group, node_policy_group, topology, signature):
    """Legacy policy_group means global replicas; FSDP's local group stays separate."""
    if topology is not None:
        if validate_h3_topology(topology) != signature['topology']:
            raise ValueError('explicit checkpoint topology differs from signature')
        if policy is not None and node_policy_group is not None and policy.devices.policy_group is not node_policy_group:
            raise ValueError('checkpoint node FSDP group differs from policy devices')
    if policy_group is not None and all_policy_group is not None and policy_group is not all_policy_group:
        raise ValueError('legacy/global checkpoint policy groups disagree')
    group = all_policy_group if all_policy_group is not None else policy_group
    if topology is not None and policy is not None and _distributed():
        if group is None or dist.get_world_size(group) != topology['num_nodes']*topology['policy_ranks']:
            raise ValueError('checkpoint global adapter group must contain exactly M*P policy ranks')
    return group


def save_h3_checkpoint(path: Path, policy: H3Policy | None, optimizer, *, state: dict, rank_state: dict,
                       policy_group=None, control_group=None, all_policy_group=None, node_policy_group=None,
                       topology=None) -> None:
    """Publish adapters/moments after policy agreement and gather every rank's RNG."""
    rank = dist.get_rank() if _distributed() else 0
    policy_group = _checked(lambda: _checkpoint_groups(policy, policy_group, all_policy_group,
        node_policy_group, topology, state['signature']), control_group)
    def prepare():
        validate_h3_topology(state['signature']['topology'])
        if (policy is None) != (optimizer is None):
            raise ValueError('policy and optimizer must both be present or absent')
        if policy is None:
            return None
        adapters = _cpu(_adapter_parameters(policy))
        if not adapters:
            raise ValueError('no H3 default/old adapter parameters')
        return dict(adapters=adapters, optimizer=_cpu(optimizer.state_dict()))
    local = _checked(prepare, control_group)
    # Service ranks never enter policy collectives, including failure agreement.
    def agreement():
        if policy is not None:
            hashes = _gather(_digest(local), policy_group)
            if len(set(hashes)) != 1:
                raise ValueError('divergent policy adapter/optimizer replicas')
    _checked(agreement, control_group)
    caller = _checked(lambda: dict(control=_cpu(rank_state), rng=_rng_state()), control_group)
    ranks = _gather(caller, control_group)
    common_hashes = _gather(_digest(state), control_group)
    def publish():
        if len(set(common_hashes)) != 1:
            raise ValueError('divergent common checkpoint state')
        if rank != 0:
            return
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError('checkpoint destination already exists; published snapshots are immutable')
        publication_started = time.monotonic()
        payload = dict(schema=1, common=_cpu(state), ranks=ranks, **local)
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getvalue()
        manifest = dict(file=destination.name, sha256=hashlib.sha256(data).hexdigest(), update=state['update'])
        _atomic_bytes(destination, data)
        # Current payload serialization/fsync is persisted without rewriting the
        # immutable payload. Pointer fsync and final control exchange are excluded.
        manifest['payload_publication_seconds'] = time.monotonic()-publication_started
        _atomic_bytes(destination.with_suffix(destination.suffix+'.json'), json.dumps(manifest).encode())
        # This is the commit record: readers never follow an incomplete publication.
        _atomic_bytes(destination.parent/'latest.json', json.dumps(manifest).encode())
    _checked(publish, control_group)


def load_h3_checkpoint(path: Path, policy: H3Policy | None, optimizer, *, expected_signature: dict,
                       policy_group=None, control_group=None, all_policy_group=None, node_policy_group=None,
                       topology=None) -> tuple[dict, dict]:
    """Validate before mutation, broadcast only adapter tensors among policy ranks."""
    rank = dist.get_rank() if _distributed() else 0
    policy_group = _checked(lambda: _checkpoint_groups(policy, policy_group, all_policy_group,
        node_policy_group, topology, expected_signature), control_group)
    signatures = _gather(_digest(expected_signature), control_group)
    if len(set(signatures)) != 1:
        raise RuntimeError('H3 checkpoint expected signature differs across ranks')
    def read():
        validate_h3_topology(expected_signature['topology'])
        if rank != 0:
            return None
        source = Path(path)
        pointer = source if source.name == 'latest.json' else source.with_suffix(source.suffix+'.json')
        manifest = json.loads(pointer.read_text())
        if source.name == 'latest.json':
            source = source.parent/manifest['file']
        elif manifest['file'] != source.name:
            raise ValueError('checkpoint integrity filename mismatch')
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['sha256']:
            raise ValueError('checkpoint integrity digest mismatch')
        payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
        if payload['schema'] != 1 or payload['common']['signature'] != expected_signature:
            raise ValueError('checkpoint signature mismatch')
        if manifest['update'] != payload['common']['update']:
            raise ValueError('stale latest pointer update')
        if len(payload['ranks']) != (dist.get_world_size(control_group) if _distributed() else 1):
            raise ValueError('rank topology mismatch')
        if 'longest_publication_seconds' in payload['common']:
            payload['common']['longest_publication_seconds'] = max(
                payload['common']['longest_publication_seconds'], manifest['payload_publication_seconds'])
        return payload
    payload = _checked(read, control_group)
    common, ranks = _broadcast((payload['common'], payload['ranks']) if rank == 0 else None, control_group)
    replicas = None
    if policy is not None:
        replicas = _broadcast((payload['adapters'], payload['optimizer']) if rank == 0 else None, policy_group)
    def validate_layout():
        if policy is not None:
            parameters = _adapter_parameters(policy)
            adapters, moments = replicas
            if parameters.keys() != adapters.keys() or any(
                    parameters[key].shape != adapters[key].shape for key in parameters):
                raise ValueError('adapter layout mismatch')
    _checked(validate_layout, control_group)
    def restore():
        if policy is not None:
            parameters = _adapter_parameters(policy)
            adapters, moments = replicas
            with torch.no_grad():
                for key, parameter in parameters.items():
                    parameter.copy_(adapters[key])
            optimizer.load_state_dict(moments)
        _restore_rng(ranks[rank]['rng'])
        return common, ranks[rank]['control']
    return _checked(restore, control_group)


class H3SafeBoundary:
    """Collective stop vote with enough reserve for one update and publication."""
    def __init__(self, *, walltime_seconds=0., publication_reserve_seconds=60., clock=time.monotonic,
                 deadline=None):
        self.clock = clock
        self.started = clock()
        self.limited = deadline is not None or walltime_seconds > 0
        self.walltime_seconds = float(walltime_seconds) if deadline is None else float(deadline)-time.time()
        self.publication_reserve_seconds = float(publication_reserve_seconds)
        self.termination_requested = False

    @contextmanager
    def signals(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
        def terminate(signum, frame):
            self.termination_requested = True
        try:
            for sig in previous:
                signal.signal(sig, terminate)
            yield self
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def should_stop(self, longest_update, longest_publication, control_group):
        reserve = self.publication_reserve_seconds + longest_publication
        remaining = self.walltime_seconds-(self.clock()-self.started)
        local = self.termination_requested or (self.limited and remaining <= longest_update+reserve)
        return any(_gather(local, control_group))
