"""One-pass scheduling and durable state for distributed Wan training."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist


def prompt_schedule(prompts: list[str], nonce: int, workers: int) -> list[int]:
    if not prompts or len(prompts) % workers:
        raise ValueError("Prompt count must be nonzero and divisible by worker count")
    if len(set(prompts)) != len(prompts):
        raise ValueError("One-pass training requires distinct prompt lines")
    order = list(range(len(prompts)))
    random.Random(nonce).shuffle(order)
    return order


def split_update(order: list[int], update: int, rank: int, workers: int) -> list[int]:
    return order[update * workers + rank:update * workers + rank + 1]


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


def should_start_update(completed: int, total: int, elapsed: float,
                        max_update_seconds: float | None, budget: float) -> bool:
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
