"""N-D port of scripts/train_opsd_ri_sd3.py::_opa_tr_step (dir_mode='grad') for video latents."""

from __future__ import annotations

from typing import Callable, Optional

import torch


def _bshape(x: torch.Tensor):
    return (-1,) + (1,) * (x.ndim - 1)


def opa_tr_step_nd(
    y0: torch.Tensor,
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    rho: float,
    n_ascent: int,
    eta: float,
    direction: float,
    first_grad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x0 = y0.detach().float()
    budget = (rho * x0.flatten(1).norm(dim=1)).view(_bshape(x0))
    step_len = eta * budget / max(n_ascent, 1)
    x = x0.clone()
    for i in range(n_ascent):
        if i == 0 and first_grad is not None:
            g = first_grad.float()
        else:
            x = x.detach().requires_grad_(True)
            r = reward_fn(x)
            (g,) = torch.autograd.grad(r.sum(), x)
        gn = g.flatten(1).norm(dim=1).view(_bshape(x0)) + 1e-12
        x = x.detach() + float(direction) * step_len * (g / gn)
        d = x - x0
        dn = d.flatten(1).norm(dim=1).view(_bshape(x0))
        x = (x0 + d * torch.clamp(budget / (dn + 1e-12), max=1.0)).detach()
    return x.detach()
