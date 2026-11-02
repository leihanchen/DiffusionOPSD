"""OPSD fitting and resumable state independent of Cosmos checkpoint size."""

from __future__ import annotations

import json
import random
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
    """Permit shared GPUs while rejecting unavailable configured indices."""
    parsed = [torch.device(device) for device in devices]
    if not parsed or any(device.type != "cuda" for device in parsed):
        raise ValueError("Cosmos training devices must be CUDA devices")
    required = max(device.index or 0 for device in parsed) + 1
    if visible_count < required:
        raise ValueError(f"Configured Cosmos devices require at least {required} visible GPUs; found {visible_count}")


def fit_queries(rollout, records, weights, reward_fn, optimizer, cfg, progress=None):
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
        report("policy_forward")
        prediction = rollout.prediction(rec)
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
