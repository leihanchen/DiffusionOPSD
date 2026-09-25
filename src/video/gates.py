"""Identity gate, motion floor, quality mask, and the masked endpoint weight (spec Sec. 4.2-4.5)."""

from __future__ import annotations

import torch


def identity_keep(s_id: torch.Tensor, tau_id: float) -> torch.Tensor:
    return s_id >= tau_id


def motion_mask(m: torch.Tensor, tau_motion: float) -> torch.Tensor:
    return (m >= tau_motion).to(m.dtype)


def quality_mask(p_q: torch.Tensor, tau_q: float) -> torch.Tensor:
    return (p_q >= tau_q).to(p_q.dtype)


def endpoint_weight(adv: torch.Tensor, adv_clip_max: float) -> torch.Tensor:
    """Repo convention from scripts/train_opsd_ri_sd3.py: r1 in [0,1] from a clipped advantage."""
    adv_clip = torch.clamp(adv, -adv_clip_max, adv_clip_max)
    return torch.clamp((adv_clip / adv_clip_max) / 2.0 + 0.5, 0, 1)


def effective_weight(adv, adv_clip_max, m, tau_motion, p_q, tau_q) -> torch.Tensor:
    return endpoint_weight(adv, adv_clip_max) * motion_mask(m, tau_motion) * quality_mask(p_q, tau_q)


def percentile_threshold(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float().flatten(), q).item())
