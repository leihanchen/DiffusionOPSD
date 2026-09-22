"""Frozen vision estimators behind small protocols so GeoReward can be unit-tested with stubs."""

from __future__ import annotations

import os
from typing import NamedTuple, Protocol

import torch
import torch.nn.functional as F

_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DepthOutput(NamedTuple):
    depth: torch.Tensor  # [B,T,H,W]
    K: torch.Tensor      # [B,3,3]
    poses: torch.Tensor  # [B,T,4,4] camera-to-world
    conf: torch.Tensor   # [B,T,H,W] in [0,1]


class DepthEstimator(Protocol):
    def __call__(self, frames01: torch.Tensor) -> DepthOutput: ...


class FlowEstimator(Protocol):
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


class PatchFeatureExtractor(Protocol):
    def __call__(self, imgs: torch.Tensor) -> torch.Tensor: ...


def relative_pose(poses: torch.Tensor, t: int) -> torch.Tensor:
    """camera(t) coords -> camera(t+1) coords given camera-to-world poses."""
    return torch.linalg.inv(poses[:, t + 1]) @ poses[:, t]


def _ckpt_root() -> str:
    root = os.environ.get("VIDEO_REWARD_CKPT_PATH")
    if not root:
        raise EnvironmentError("Set VIDEO_REWARD_CKPT_PATH (see scripts/download_video_reward_weights.sh)")
    return root


class DinoV2Patches(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, patch: int = 14):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.patch = patch

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        B, _, H, W = imgs.shape
        h, w = H // self.patch, W // self.patch
        x = F.interpolate(imgs, size=(h * self.patch, w * self.patch), mode="bilinear", align_corners=False)
        x = (x - _IMNET_MEAN.to(x)) / _IMNET_STD.to(x)
        tok = self.backbone.forward_features(x)["x_norm_patchtokens"]  # [B,h*w,D]
        feat = tok.transpose(1, 2).reshape(B, -1, h, w)
        return F.normalize(feat, dim=1)


def load_dinov2(device) -> PatchFeatureExtractor:
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device)
    return DinoV2Patches(backbone).to(device)


class WaftFlow(torch.nn.Module):
    """WAFT adapter. Wraps the model object returned by the official WAFT repo (princeton-vl/WAFT)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.eval().requires_grad_(False)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # WAFT consumes images in [0,255]; its forward returns a list of flow predictions, last is final.
        out = self.model(a * 255.0, b * 255.0)
        flow = out[-1] if isinstance(out, (list, tuple)) else out
        return flow.float()


def load_waft(device) -> FlowEstimator:
    import sys
    root = os.path.join(_ckpt_root(), "WAFT")
    sys.path.insert(0, root)
    from core.waft import WAFT  # official repo module
    import json
    with open(os.path.join(root, "config", "eval", "sintel.json")) as f:
        cfg = json.load(f)
    model = WAFT(cfg)
    sd = torch.load(os.path.join(_ckpt_root(), "waft_tar_c_t.pth"), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    return WaftFlow(model).to(device)


class DepthAnything3(torch.nn.Module):
    """Depth Anything 3 adapter. Wraps the official inference model (ByteDance-Seed/Depth-Anything-3)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.eval().requires_grad_(False)

    def forward(self, frames01: torch.Tensor) -> DepthOutput:
        B, T, _, H, W = frames01.shape
        pred = self.model.inference(frames01.reshape(B * T, 3, H, W), batch_size=T)
        depth = pred["depth"].reshape(B, T, H, W)
        conf = pred["conf"].reshape(B, T, H, W).clamp(0, 1)
        K = pred["intrinsics"].reshape(B, T, 3, 3)[:, 0]
        poses = pred["extrinsics_c2w"].reshape(B, T, 4, 4)
        return DepthOutput(depth.float(), K.float(), poses.float(), conf.float())


def load_depth_anything3(device) -> DepthEstimator:
    from depth_anything_3.api import DepthAnything3 as DA3
    model = DA3.from_pretrained(os.path.join(_ckpt_root(), "depth-anything-3-large-v1.1")).to(device)
    return DepthAnything3(model).to(device)
