"""GeoFlow-style geometry reward on decoded clips (spec Sec. 4.1-4.3)."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from diffusionopsd.video.estimators import DepthEstimator, FlowEstimator, PatchFeatureExtractor, relative_pose
from diffusionopsd.video.rigid_flow import reprojection_residual, rigid_flow


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
    return F.grid_sample(feat_next, torch.stack([gx, gy], dim=-1), mode="bilinear",
                         padding_mode="border", align_corners=True)


class GeoReward(torch.nn.Module):
    def __init__(self, depth: DepthEstimator, flow: FlowEstimator, feats: PatchFeatureExtractor,
                 w_rigid: float = 0.5, w_dino: float = 0.5, residual_scale: float = 1.0):
        super().__init__()
        self.depth, self.flow, self.feats = depth, flow, feats
        self.w_rigid, self.w_dino, self.residual_scale = w_rigid, w_dino, residual_scale

    def forward(self, frames01: torch.Tensor) -> GeoRewardOutput:
        B, T, _, H, W = frames01.shape
        d = self.depth(frames01)
        rigid_terms, dino_terms, motion_terms = [], [], []
        for t in range(T - 1):
            a, b = frames01[:, t], frames01[:, t + 1]
            flow = self.flow(a, b)
            rig = rigid_flow(d.depth[:, t], d.K, relative_pose(d.poses, t))
            res = reprojection_residual(flow, rig, d.conf[:, t])
            rigid_terms.append(-self.residual_scale * res)
            fa, fb = self.feats(a), self.feats(b)
            sim = (fa * warp_features(fb, flow)).sum(dim=1)  # cosine, [B,h,w]
            dino_terms.append(sim.flatten(1).mean(1))
            motion_terms.append(flow.norm(dim=1).flatten(1).mean(1))
        rigid = torch.stack(rigid_terms, 1).mean(1)
        dino = torch.stack(dino_terms, 1).mean(1)
        motion = torch.stack(motion_terms, 1).mean(1).detach()
        geo = self.w_rigid * rigid + self.w_dino * dino
        return GeoRewardOutput(geo, rigid, dino, dino.detach(), motion)
