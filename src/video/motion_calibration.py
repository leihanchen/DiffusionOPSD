"""Motion-only calibration using the same pixel-flow statistic as GeoReward."""

from __future__ import annotations

import random

import torch


def calibration_indices(total: int, count: int) -> list[int]:
    if not 1 <= count <= total:
        raise ValueError("Calibration count must be between one and the number of prompts")
    return random.Random(0).sample(range(total), count)


@torch.no_grad()
def motion_score(frames01: torch.Tensor, flow) -> torch.Tensor:
    """Mean pixel displacement per adjacent frame, matching GeoReward.motion."""
    if frames01.ndim != 5 or frames01.shape[1] < 2:
        raise ValueError("Expected [B,T,C,H,W] with at least two frames")
    terms = [
        flow(frames01[:, t], frames01[:, t + 1]).norm(dim=1).flatten(1).mean(1) for t in range(frames01.shape[1] - 1)
    ]
    return torch.stack(terms, 1).mean(1)
