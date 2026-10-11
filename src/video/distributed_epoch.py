"""One-pass scheduling and durable state for distributed Wan training."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist


def active_cuda_devices(cfg, available_count: int) -> tuple[int, ...]:
    """Validate configured model placement and return each used GPU once."""
    devices = {0}  # Policy and NCCL always use local GPU 0.
    for placement in (cfg.vae_device, cfg.reward_device, cfg.judge.device):
        device = torch.device(placement)
        if device.type != "cuda" or device.index is None:
            raise ValueError(f"Expected explicit CUDA device, got {placement}")
        devices.add(device.index)
    if max(devices) >= available_count:
        raise ValueError(f"Configured GPUs {sorted(devices)} exceed {available_count} visible CUDA devices")
    return tuple(sorted(devices))


def prompt_schedule(prompts: list[str], nonce: int, workers: int) -> list[int]:
    if not prompts or len(prompts) % workers:
        raise ValueError("Prompt count must be nonzero and divisible by worker count")
    if len(set(prompts)) != len(prompts):
        raise ValueError("One-pass training requires distinct prompt lines")
    order = list(range(len(prompts)))
    random.Random(nonce).shuffle(order)
    return order


def split_update(order: list[int], update: int, rank: int, workers: int) -> list[int]:
    return order[update * workers + rank : update * workers + rank + 1]


def deterministic_seed(nonce: int, prompt_index: int, repetition: int) -> int:
    value = f"{nonce}:{prompt_index}:{repetition}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**63)


def atomic_checkpoint(path: Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_checkpoint(path: Path, prompt_hash: str, config_hash: str) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (("prompt_hash", prompt_hash), ("config_hash", config_hash)):
        if state[key] != expected:
            raise ValueError(f"Resume {key} mismatch")
    return state


def should_start_update(
    completed: int, total: int, elapsed: float, max_update_seconds: float | None, budget: float
) -> bool:
    if completed >= total:
        return False
    if completed == 0:
        return True
    allowance = 1800 + 1.25 * (max_update_seconds or 3600)
    return elapsed + allowance < budget


def synchronize_gradients(parameters, total_kept: int) -> None:
    """Sum sample gradients across workers and scale by global retained count."""
    if total_kept <= 0:
        raise ValueError("No retained rollout in global update")
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(total_kept)


def allocation_layout(nodes: int, gpus_per_node: int) -> tuple[int, int]:
    """Return processes per node and world size for exclusive two-GPU workers."""
    if nodes <= 0 or gpus_per_node <= 0 or gpus_per_node % 2:
        raise ValueError("Require positive nodes and an even positive GPU count per node")
    per_node = gpus_per_node // 2
    world = nodes * per_node
    if world not in (4, 8, 16):
        raise ValueError("The fixed 16-rollout batch supports 4, 8 or 16 workers")
    return per_node, world


def assign_rollouts(order: list[int], update: int, rank: int, workers: int) -> tuple[int, tuple[int, ...]]:
    """Keep four logical prompts per update, independent of execution topology."""
    if workers not in (4, 8, 16) or not 0 <= rank < workers or update < 0:
        raise ValueError("Invalid worker layout or update")
    per_prompt = workers // 4
    group, member = divmod(rank, per_prompt)
    position = update * 4 + group
    if position >= len(order):
        raise ValueError("Update exceeds the prompt schedule")
    count = 4 // per_prompt
    return order[position], tuple(range(member * count, (member + 1) * count))


def worker_sample_assignments(
    order: list[int], update: int, rank: int, workers: int, *, single_node_smoke: bool = False
) -> list[tuple[int, tuple[int, ...]]]:
    """A smoke worker handles half of each prompt; production workers handle one prompt."""
    if not single_node_smoke:
        return [assign_rollouts(order, update, rank, workers)]
    if workers != 2 or rank not in (0, 1) or update < 0:
        raise ValueError("Single-node smoke requires exactly two workers")
    prompts = order[update * 4 : update * 4 + 4]
    if len(prompts) != 4:
        raise ValueError("Update exceeds the prompt schedule")
    return [(prompt, tuple(range(rank * 2, (rank + 1) * 2))) for prompt in prompts]


def canonical_samples(
    gathered: list[list[dict]], order: list[int], update: int, workers: int, *, single_node_smoke: bool = False
) -> list[dict]:
    """Validate ownership and return all samples in logical prompt/rollout order."""
    if len(gathered) != workers:
        raise ValueError("Missing worker sample records")
    by_id = {}
    prompt_text = {}
    for rank, samples in enumerate(gathered):
        assignments = worker_sample_assignments(order, update, rank, workers, single_node_smoke=single_node_smoke)
        expected = {(prompt, rollout) for prompt, rollouts in assignments for rollout in rollouts}
        actual = [(s["prompt_index"], s["rollout_index"]) for s in samples]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f"Invalid or duplicate sample identities for rank {rank}")
        for sample, key in zip(samples, actual):
            prompt = key[0]
            if key in by_id:
                raise ValueError("Duplicate global sample identity")
            text = sample["prompt"]
            if prompt in prompt_text and prompt_text[prompt] != text:
                raise ValueError("Inconsistent text within a prompt group")
            prompt_text[prompt] = text
            by_id[key] = sample
    keys = [(prompt, rollout) for prompt in order[update * 4 : update * 4 + 4] for rollout in range(4)]
    if len(keys) != 16 or set(keys) != set(by_id):
        raise ValueError("Expected exactly 16 unique global rollouts")
    return [by_id[key] for key in keys]


def route_sample_weights(canonical: list[dict], weights: list[float], gathered: list[list[dict]]) -> list[list[float]]:
    if len(canonical) != len(weights):
        raise ValueError("Sample and weight counts differ")
    by_id = {(s["prompt_index"], s["rollout_index"]): w for s, w in zip(canonical, weights)}
    return [[by_id[(s["prompt_index"], s["rollout_index"])] for s in samples] for samples in gathered]


def validate_worker_devices(records: list[dict], workers: int) -> None:
    """GPU UUIDs, rather than remapped CUDA indices, identify device ownership."""
    if len(records) != workers or {r["rank"] for r in records} != set(range(workers)):
        raise ValueError("Missing or duplicated worker ranks")
    used = {}
    local_ranks = set()
    for record in records:
        host = record["host"]
        uuids = record["gpu_uuids"]
        if len(uuids) != 2 or len(set(uuids)) != 2 or any(not uuid for uuid in uuids):
            raise ValueError("Every scalable worker must see two distinct GPU UUIDs")
        local_key = (host, record["local_rank"])
        if local_key in local_ranks:
            raise ValueError("Duplicated local rank on a host")
        local_ranks.add(local_key)
        host_used = used.setdefault(host, set())
        if host_used.intersection(uuids):
            raise ValueError(f"GPU overlap on host {host}")
        host_used.update(uuids)


def training_config_hash(config: dict) -> str:
    """Only output location is excluded; training settings remain strict."""
    import json

    normalized = {key: value for key, value in config.items() if key != "logdir"}
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def prepare_resume(
    state: dict,
    config: dict,
    prompt_hash: str,
    topology: dict,
    allow_topology_change: bool,
    source_run_dir: str | None = None,
) -> tuple[dict, bool]:
    """Validate a trusted checkpoint and prepare a non-destructive topology migration."""
    import json

    if state.get("prompt_hash") != prompt_hash:
        raise ValueError("Resume prompt_hash mismatch")
    version = state.get("version", 1)
    if version == 1:
        legacy_config = dict(config)
        if source_run_dir is not None:
            legacy_config["logdir"] = source_run_dir
        legacy_hash = hashlib.sha256(json.dumps(legacy_config, sort_keys=True).encode()).hexdigest()
        if state.get("config_hash") != legacy_hash:
            raise ValueError("Legacy resume config_hash mismatch")
        if len(state["rng_states"]) != 4:
            raise ValueError("Legacy checkpoint must contain four worker RNG states")
        previous = {"world_size": 4, "workers_per_node": 1, "nodes": 4}
    elif version == 2:
        expected = training_config_hash(config)
        if state.get("training_config_hash") != expected or training_config_hash(state["training_config"]) != expected:
            raise ValueError("Resume training config mismatch")
        previous = state["topology"]
        if len(state["rng_states"]) != previous["world_size"]:
            raise ValueError("Checkpoint RNG count differs from its topology")
    else:
        raise ValueError(f"Unsupported training checkpoint version {version}")
    changed = previous != topology
    if changed and not allow_topology_change:
        raise ValueError("Resume topology differs; explicitly allow topology change")
    result = dict(state)
    if changed:
        result["max_update_seconds"] = None
        result["topology_migrations"] = [
            *state.get("topology_migrations", []),
            {
                "from": previous,
                "to": topology,
                "at_update": state["next_update"],
                "rng_policy": "nonce-update-rank-reseed",
            },
        ]
    return result, changed


def topology_rng_seed(nonce: int, completed: int, rank: int) -> int:
    value = f"topology-rng:{nonce}:{completed}:{rank}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**63)


def adapter_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """Hash adapter tensors without requiring NumPy support for BF16."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{cpu.dtype}:{tuple(cpu.shape)}:".encode())
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
