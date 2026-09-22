"""Rigid-scene geometry in pure torch: depth + camera motion -> induced optical flow."""

from __future__ import annotations

import torch


def pixel_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    v, u = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([u, v, torch.ones_like(u)], dim=0)  # [3,H,W]


def rigid_flow(depth: torch.Tensor, K: torch.Tensor, T_rel: torch.Tensor) -> torch.Tensor:
    """Flow induced by a rigid camera motion.

    depth: [B,H,W] metric depth of frame t.  K: [B,3,3].  T_rel: [B,4,4], frame-t camera -> frame-(t+1) camera.
    Returns flow [B,2,H,W] in pixels (u' - u, v' - v).
    """
    B, H, W = depth.shape
    grid = pixel_grid(H, W, depth.device, depth.dtype).reshape(1, 3, -1).expand(B, 3, -1)  # [B,3,N]
    rays = torch.linalg.solve(K, grid)  # K^-1 [u,v,1]
    X = rays * depth.reshape(B, 1, -1)  # [B,3,N]
    R, t = T_rel[:, :3, :3], T_rel[:, :3, 3:4]
    X2 = R @ X + t
    z2 = X2[:, 2:3].clamp(min=1e-6)
    p2 = (K @ (X2 / z2))[:, :2]  # [B,2,N]
    flow = p2 - grid[:, :2]
    return flow.reshape(B, 2, H, W)


def reprojection_residual(flow: torch.Tensor, rigid: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    """Confidence-weighted mean L1 residual between observed and rigid flow, per sample -> [B]."""
    l1 = (flow - rigid).abs().sum(dim=1)  # [B,H,W]
    w = conf.clamp(min=0.0)
    return (w * l1).flatten(1).sum(1) / (w.flatten(1).sum(1) + 1e-8) * (w.flatten(1).sum(1) > 0)
