"""Pinned GeoFlow training-server reward with partial input gradients."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from diffusionopsd.video.estimators import DepthEstimator, FlowEstimator, PatchFeatureExtractor


class GeoRewardOutput(NamedTuple):
    geo: torch.Tensor
    rigid: torch.Tensor
    dino: torch.Tensor
    s_id: torch.Tensor
    motion: torch.Tensor


def warp_features(feat_next: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp next-frame patch features to the current frame along pixel flow."""
    B, D, h, w = feat_next.shape
    H, W = flow.shape[-2:]
    flow_p = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
    flow_p = torch.stack([flow_p[:, 0] * (w / W), flow_p[:, 1] * (h / H)], dim=1)  # to patch units
    ys, xs = torch.meshgrid(torch.arange(h, device=flow.device), torch.arange(w, device=flow.device), indexing="ij")
    grid = torch.stack([xs, ys], dim=0).float()[None] + flow_p  # [B,2,h,w] sample positions in next frame
    gx = 2 * grid[:, 0] / max(w - 1, 1) - 1
    gy = 2 * grid[:, 1] / max(h - 1, 1) - 1
    return F.grid_sample(
        feat_next, torch.stack([gx, gy], dim=-1), mode="bilinear", padding_mode="border", align_corners=True
    )


GEOFLOW_REVISION = "fac2426820e303eb1c0ffd1cc43e58585f99f87b"
REWARD_VERSION = "geoflow-server-" + GEOFLOW_REVISION
GRADIENT_POLICY = "partial-da3-detached-uint8-ste"


def reward_metadata():
    return {"version": REWARD_VERSION, "gradient_policy": GRADIENT_POLICY,
            "process_res": 504, "confidence_gating": False,
            "long_pair_gap": 8, "long_pair_stride": 4, "min_weight": 0.3,
            "waft_revision": "b152ff1cad1af8c185ee7b141997c48ff3334c87",
            "da3_revision": "2c21ea849ceec7b469a3e62ea0c0e270afc3281a"}


def configure_reward(config):
    """Stamp custom configs and reject settings that contradict the pinned scorer."""
    metadata = reward_metadata()
    configured = config.get("geometry_reward", metadata)
    if dict(configured) != metadata:
        raise ValueError("GeoFlow settings differ from the pinned server defaults; remove geometry_reward overrides")
    with config.unlocked():
        config.geometry_reward = metadata


def quantize_frames(frames):
    """PNG-equivalent rounded RGB8 forward values with a straight-through gradient."""
    clipped = frames.float().clamp(0, 1)
    rounded = clipped.mul(255).round().div(255)
    return clipped + (rounded - clipped).detach()


class GeoReward(torch.nn.Module):
    """Released server defaults, including long pairs and worst-adjacent scoring.

    Input frames are the server's already-selected frame sequence. The server's
    fps argument does not subsample. DA3 is evaluated anew and detached on every
    call. The two frozen image estimators propagate input gradients.
    """
    def __init__(self, depth: DepthEstimator, flow: FlowEstimator, feats: PatchFeatureExtractor,
                 w_rigid: float = 0.5, w_dino: float = 0.5, residual_scale: float = 1.0):
        super().__init__()
        if (w_rigid, w_dino, residual_scale) != (0.5, 0.5, 1.0):
            raise ValueError("GeoFlow requires w_rigid=w_dino=0.5 and residual_scale=1; remove legacy overrides")
        self.depth, self.flow, self.feats = depth, flow, feats
        for estimator in (depth, flow, feats):
            if isinstance(estimator, torch.nn.Module):
                estimator.eval().requires_grad_(False)

    def _pair_terms(self, source, target, source_depth, target_depth, source_k, target_k, pose, long_pair):
        from diffusionopsd.video.geoflow_math import (
            depth_visibility, pair_rewards, project_depth, warp_depth, warp_image,
        )
        rigid, projected = project_depth(source_depth, source_k, target_k, pose)
        warped_depth = warp_depth(rigid, projected)
        flow = rigid if long_pair else self.flow(source, target)
        source_features = self.feats(source)
        warped_features = self.feats(warp_image(target, flow))
        patch_size = source_features.shape[-2:]
        valid_pixels = warp_image(torch.ones_like(source[:, :1]), flow)
        valid = F.interpolate(valid_pixels, size=patch_size, mode="bilinear", align_corners=False)[:, 0] > 0.5
        visibility = None
        if long_pair:
            low_flow = (torch.linalg.vector_norm(rigid, dim=1) < 4).float()
            low_patches = F.interpolate(low_flow[:, None], size=patch_size,
                                        mode="bilinear", align_corners=False)[:, 0]
            valid = valid & (low_patches > 0.5)
            visibility = depth_visibility(projected, target_depth, rigid)
        result = pair_rewards(flow, rigid, target_depth, warped_depth, source_features,
                              warped_features, valid.float(), visibility=visibility)
        dino = result["r_dino"]
        if long_pair:
            dino = torch.where(result["dino_counts"] >= 1, dino, torch.zeros_like(dino))
        motion = torch.linalg.vector_norm(flow, dim=1).mean((1, 2)).detach()
        return result["r_geo"], dino, motion

    def forward(self, frames01: torch.Tensor) -> GeoRewardOutput:
        if frames01.ndim != 5 or frames01.shape[2] != 3 or frames01.shape[1] < 2:
            raise ValueError("GeoFlow expects [B,T,3,H,W] with at least two frames")
        # Reward computations use float32, even under the generator's autocast.
        with torch.autocast(device_type=frames01.device.type, enabled=False):
            return self._score(quantize_frames(frames01))

    def _score(self, frames):
        batch, length = frames.shape[:2]
        with torch.no_grad():
            prediction = self.depth(frames.detach())
            depth, intrinsics, poses, _ = (value.detach() for value in prediction)
        height, width = depth.shape[-2:]
        if min(height, width) < 2:
            raise ValueError("GeoFlow processing dimensions must be at least two pixels")
        frames = F.interpolate(frames.flatten(0, 1), size=(height, width), mode="bilinear",
                               align_corners=False, antialias=True).reshape(batch, length, 3, height, width)
        if intrinsics.ndim == 3:
            intrinsics = intrinsics[:, None].expand(-1, length, -1, -1)
        # Reference computes relative world-to-camera transforms in float64.
        world_to_camera = torch.linalg.inv(poses.double())
        pairs = [(t, t + 1, False) for t in range(length - 1)]
        pairs += [(t, t + 8, True) for t in range(0, length - 8, 4)]
        geometry, semantic, motions = [], [], []
        for source, target, long_pair in pairs:
            pose = (world_to_camera[:, target] @ torch.linalg.inv(world_to_camera[:, source])).float()
            inputs = (frames[:, source], frames[:, target], depth[:, source], depth[:, target],
                      intrinsics[:, source], intrinsics[:, target], pose, long_pair)
            if torch.is_grad_enabled() and frames.requires_grad:
                terms = checkpoint(self._pair_terms, *inputs, use_reentrant=False)
            else:
                terms = self._pair_terms(*inputs)
            geometry.append(terms[0])
            semantic.append(terms[1])
            motions.append(terms[2])
        geometry, semantic = torch.stack(geometry, 1), torch.stack(semantic, 1)
        adjacent = length - 1
        combined = 0.5 * geometry[:, :adjacent] + 0.5 * semantic[:, :adjacent]
        all_scores = torch.cat((combined, 0.25 * geometry[:, adjacent:] + 0.75 * semantic[:, adjacent:]), dim=1)
        score = 0.7 * all_scores.mean(1) + 0.3 * combined.min(1).values
        # Components and identity summarize adjacent pairs, matching server diagnostics.
        rigid, dino = geometry[:, :adjacent].mean(1), semantic[:, :adjacent].mean(1)
        motion = torch.stack(motions[:adjacent], 1).mean(1)
        return GeoRewardOutput(score, rigid, dino, (1 + dino).detach(), motion)
