"""OPSD fitting and resumable state independent of Cosmos checkpoint size."""

from __future__ import annotations

import json
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from diffusionopsd.video.branch_loss import branch_loss
from diffusionopsd.video.opa_video import opa_tr_step_nd


@dataclass
class PromptCursor:
    """Sampling attempts advance independently of successful optimizer updates."""

    num_prompts: int
    cursor: int = 0
    consecutive_empty: int = 0

    def __post_init__(self):
        if self.num_prompts <= 0 or self.cursor < 0 or self.consecutive_empty < 0:
            raise ValueError("Prompt cursor requires a nonempty dataset and nonnegative counters")

    def take(self) -> tuple[int, int]:
        attempt = self.cursor
        self.cursor += 1
        return attempt % self.num_prompts, attempt

    def record_batch(self, retained: int) -> bool:
        if retained < 0:
            raise ValueError("Retained sample count must be nonnegative")
        self.consecutive_empty = self.consecutive_empty + 1 if retained == 0 else 0
        if self.consecutive_empty >= self.num_prompts:
            raise RuntimeError("Identity gate rejected a full pass over the prompt set; no optimizer update performed")
        return retained > 0

    def state_dict(self) -> dict:
        return asdict(self)


def validate_cuda_devices(devices, *, visible_count):
    """Validate policy first, then all frozen components on the inference GPU."""
    from diffusionopsd.video.device_layout import validate_two_gpu_layout

    if not devices:
        raise ValueError("Trainable policy must use explicit cuda:0")
    validate_two_gpu_layout(devices[0], devices[1:], visible_count=visible_count)


@contextmanager
def preserve_rng():
    """Keep optional diagnostic observations out of training RNG."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def fit_queries(rollout, records, weights, reward_fn, optimizer, cfg, progress=None, policy=None, target_observer=None):
    """Fit detached reward targets at saved queries; never differentiate rollouts."""
    if not records or len(records) != len(weights):
        raise ValueError("queries and weights must be nonempty and have the same length")
    optimizer.zero_grad(set_to_none=True)
    loss_sum, target_grad_sum = 0.0, 0.0
    for index, (rec, weight) in enumerate(zip(records, weights)):
        def report(stage):
            if progress is not None:
                progress(stage, index)

        if not 0 <= float(weight) <= 1:
            raise ValueError("OPSD weight must lie in [0,1]")
        y0 = rec.y0.detach().float()
        yg = y0.clone().requires_grad_(True)
        report("target_forward")
        reward = reward_fn(yg)
        report("target_backward")
        (gradient,) = torch.autograd.grad(reward.sum(), yg)
        norm = gradient.norm()
        if not torch.isfinite(reward).all() or not torch.isfinite(norm) or norm <= 0:
            raise RuntimeError("Target reward must have a finite, nonzero latent gradient")
        target_grad_sum += float(norm)
        args = (y0, reward_fn, cfg.rho, cfg.n_ascent, cfg.eta)
        report("positive_target")
        positive = opa_tr_step_nd(*args, direction=1, first_grad=gradient)
        report("negative_target")
        negative = opa_tr_step_nd(*args, direction=-1, first_grad=gradient)
        if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
            raise RuntimeError("Targets must be finite")
        if target_observer is not None:
            with preserve_rng():
                target_observer(rec, positive, negative)
        report("policy_forward")
        prediction = rollout.prediction(rec) if policy is None else rollout.prediction(rec, transformer=policy)
        loss = branch_loss(prediction, y0, positive, negative, y0.new_tensor([weight]), cfg.beta).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Policy loss must be finite")
        report("policy_backward")
        (loss * cfg.loss_scale / len(records)).backward()
        loss_sum += float(loss.detach())
        report("sample_complete")
    params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
    if not torch.isfinite(norm) or norm <= 0:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Policy gradient must be finite and nonzero")
    if progress is not None:
        progress("optimizer_step", None)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss_sum / len(records),
        "grad_norm": float(norm),
        "target_grad_norm": target_grad_sum / len(records),
        "retained": len(records),
    }


def save_training_state(path, *, adapters, optimizer, update, config, extra=None):
    """Atomically save trusted local state, including RNG for exact resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1,
        "adapters": adapters,
        "optimizer": optimizer,
        "update": update,
        "config": config,
        "extra": extra or {},
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_training_state(path, *, config):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1 or json.dumps(state["config"], sort_keys=True) != json.dumps(config, sort_keys=True):
        raise ValueError("Checkpoint configuration differs from this run")
    rng = state["rng"]
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    return state


@dataclass
class RoundCursor:
    """A shuffled permutation visited once, including entirely rejected batches."""

    order: list[int]
    next_index: int = 0

    def __post_init__(self):
        if (not self.order or any(type(i) is not int for i in self.order)
                or sorted(self.order) != list(range(len(self.order)))
                or type(self.next_index) is not int or not 0 <= self.next_index <= len(self.order)):
            raise ValueError('Round cursor requires a permutation and a valid committed position')

    @property
    def cursor(self) -> int:
        return self.next_index

    def take(self) -> int | None:
        if self.next_index == len(self.order):
            return None
        index = self.order[self.next_index]
        self.next_index += 1
        return index

    def record_batch(self, retained: int) -> bool:
        if retained < 0:
            raise ValueError('Retained sample count must be nonnegative')
        return retained > 0

    def state_dict(self) -> dict:
        return asdict(self)


def validate_round_transition(state: dict, config: dict) -> None:
    """Allow manifest replacement only after a committed complete previous round."""
    old = state['config']
    mutable = {'train_prompt_sha256', 'curriculum_manifest_hash', 'curriculum_round', 'curriculum_history_hash'}
    def invariants(c):
        return {k: v for k, v in c.items() if k not in mutable}
    if invariants(old) != invariants(config):
        raise ValueError('Round transition changes invariant training configuration')
    if old.get('curriculum_round', -2) + 1 != config.get('curriculum_round'):
        raise ValueError('Round transition must advance exactly one round')
    cursor = state['extra'].get('round_cursor')
    if cursor is None or cursor['next_index'] != len(cursor['order']):
        raise ValueError('Previous round is not complete')

    if state['extra'].get('development_done') is not True:
        raise ValueError('Previous round development evaluation is not committed')

    if old.get('paired_probe_count', 0) and not state['extra'].get('round_probe', {}).get('done'):
        raise ValueError('Previous round paired probe evaluation is not committed')
