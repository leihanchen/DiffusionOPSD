"""Tensor scoring matching GeoFlow reward_server.py at fac2426820e3.

DA3 geometry is detached by the caller. Hard visibility decisions retain the
reference's piecewise behavior. Vector norms use PyTorch's zero subgradient,
which avoids the undefined sqrt backward at exactly matching/static flow.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def warp_image(images: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample target images on the source grid at source + forward flow."""
    height, width = images.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=images.device, dtype=flow.dtype),
        torch.arange(width, device=images.device, dtype=flow.dtype), indexing="ij",
    )
    grid = torch.stack((2 * (x + flow[:, 0]) / (width - 1) - 1,
                        2 * (y + flow[:, 1]) / (height - 1) - 1), dim=-1)
    return F.grid_sample(images, grid, align_corners=True, padding_mode="zeros")


def project_depth(depth, source_k, target_k, pose):
    """Return rigid flow [P,2,H,W] and target-camera z [P,H,W]."""
    count, height, width = depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype), indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x))).reshape(3, -1)
    points = (torch.linalg.inv(source_k) @ pixels) * depth.reshape(count, 1, -1)
    target_points = pose[:, :3, :3] @ points + pose[:, :3, 3:4]
    projected = target_k @ target_points
    z = projected[:, 2]
    valid = z > 1e-6
    xy = torch.where(valid[:, None], projected[:, :2] / (z[:, None] + 1e-12), pixels[None, :2])
    return (xy - pixels[None, :2]).reshape(count, 2, height, width), z.reshape(count, height, width)


def warp_depth(flow, projected_depth):
    """Nearest-pixel forward projection with the reference's minimum-z buffer."""
    count, height, width = projected_depth.shape
    y, x = torch.meshgrid(torch.arange(height, device=flow.device),
                          torch.arange(width, device=flow.device), indexing="ij")
    u = (x + flow[:, 0]).round().long()
    v = (y + flow[:, 1]).round().long()
    valid = (projected_depth > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    indices = (v * width + u).clamp(0, height * width - 1).flatten(1)
    values = torch.where(valid, projected_depth, float("inf")).flatten(1)
    buffer = torch.full_like(values, float("inf"))
    buffer = buffer.scatter_reduce(1, indices, values, reduce="amin", include_self=True)
    return torch.where(torch.isinf(buffer), 0, buffer).reshape(count, height, width)


def depth_visibility(projected_depth, target_depth, rigid_flow):
    sampled = warp_image(target_depth[:, None], rigid_flow)[:, 0]
    valid = (projected_depth > 1e-6) & (sampled > 1e-6)
    return (valid & ((projected_depth - sampled) <= 0.1 * sampled.clamp(min=1e-6))).float()


def pair_rewards(flow, rigid, target_depth, warped_depth, source_features, warped_features,
                 warp_valid, confidence=None, visibility=None):
    """Pinned default pair score, with optional confidence and long-pair visibility.

    Features use [P,D,ph,pw]. Visibility has the reference denominator floor 0.3.
    Empty geometry or feature support scores -1, as in the released server.
    """
    error = torch.linalg.vector_norm(flow - rigid, dim=1)
    magnitude = torch.linalg.vector_norm(flow, dim=1) + torch.linalg.vector_norm(rigid, dim=1)
    flow_error = (error / (magnitude + 5)).clamp(0, 1)
    valid = (warped_depth > 0) & (target_depth > 1e-6)
    depth_error = torch.where(valid, (warped_depth - target_depth).abs() / (target_depth + 1e-6), 1)
    quality = (1 - flow_error) * (1 - depth_error.clamp(0, 1))
    weights = valid.float()
    if confidence is not None:
        confidence = confidence.clamp(0, 1)
        weights = weights * confidence
    full_count = weights.sum((1, 2))
    if visibility is not None:
        weights = weights * visibility
    denominator = weights.sum((1, 2)).clamp(min=1e-6)
    if visibility is not None:
        denominator = torch.maximum(denominator, 0.3 * full_count)
    geometry = (quality * weights).sum((1, 2)) / denominator - 1

    cosine = (F.normalize(source_features, dim=1) * F.normalize(warped_features, dim=1)).sum(1)
    weights = warp_valid
    if confidence is not None:
        weights = weights * F.interpolate(confidence[:, None], size=cosine.shape[-2:],
                                           mode="bilinear", align_corners=False)[:, 0]
    full_count = weights.sum((1, 2))
    if visibility is not None:
        visible_patches = F.interpolate(visibility[:, None], size=cosine.shape[-2:],
                                        mode="bilinear", align_corners=False)[:, 0]
        weights = weights * (visible_patches > 0.5).float()
    count = weights.sum((1, 2))
    denominator = count.clamp(min=1e-6)
    if visibility is not None:
        denominator = torch.maximum(denominator, 0.3 * full_count)
    # Preserve the released expression's float32 arithmetic order.
    distance = 1 - cosine
    dino = ((1 - distance) * weights).sum((1, 2)) / denominator - 1
    return {"r_geo": geometry, "r_dino": dino, "r_combined": 0.5 * geometry + 0.5 * dino,
            "dino_counts": count}
