"""Paper Eq. 11-13 with the masked endpoint weight from gates.effective_weight."""
from __future__ import annotations

import torch


def branch_loss(y_theta, y0, y_plus, y_minus, w_eff, beta: float) -> torch.Tensor:
    rd = tuple(range(1, y0.ndim))
    y_pos = beta * y_theta + (1 - beta) * y0
    y_neg = (1 + beta) * y0 - beta * y_theta
    with torch.no_grad():
        wf_p = (y_pos - y_plus).abs().mean(dim=rd, keepdim=True).clamp(min=1e-5)
        wf_n = (y_neg - y_minus).abs().mean(dim=rd, keepdim=True).clamp(min=1e-5)
    pos = ((y_pos - y_plus) ** 2 / wf_p).mean(dim=rd)
    neg = ((y_neg - y_minus) ** 2 / wf_n).mean(dim=rd)
    return w_eff * pos + (1 - w_eff) * neg
