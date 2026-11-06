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
    K: torch.Tensor  # [B,T,3,3], or shared [B,3,3] for legacy estimators
    poses: torch.Tensor  # [B,T,4,4] camera-to-world
    conf: torch.Tensor  # [B,T,H,W] in [0,1]


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
        h, w = max(1, H // self.patch), max(1, W // self.patch)
        x = F.interpolate(imgs, size=(h * self.patch, w * self.patch), mode="bilinear", align_corners=False, antialias=True)
        x = (x - _IMNET_MEAN.to(x)) / _IMNET_STD.to(x)
        if hasattr(self.backbone, "forward_features"):
            tok = self.backbone.forward_features(x)["x_norm_patchtokens"]
        else:
            hidden = self.backbone(pixel_values=x).last_hidden_state
            n_reg = int(getattr(self.backbone.config, "num_register_tokens", 0) or 0)
            tok = hidden[:, 1 + n_reg :, :]
        feat = tok.transpose(1, 2).reshape(B, -1, h, w)
        return F.normalize(feat, dim=1)


def load_dinov2(device) -> PatchFeatureExtractor:
    from transformers import AutoModel

    path = os.path.join(_ckpt_root(), "dinov2-base")
    backbone = AutoModel.from_pretrained(path, local_files_only=True)
    patch = int(getattr(backbone.config, "patch_size", 14))
    return DinoV2Patches(backbone, patch=patch).to(device)


class WaftFlow(torch.nn.Module):
    """WAFT adapter. Wraps the model object returned by the official WAFT repo (princeton-vl/WAFT)."""

    def __init__(self, model: torch.nn.Module, inference=None):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.inference = inference

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Official a1 ViTWarpV8 consumes images in [0,255] and returns {"flow": [pred, ...]}.
        predict = self.inference.calc_flow if self.inference is not None else self.model
        out = predict(a * 255.0, b * 255.0)
        if isinstance(out, dict):
            flow = out["flow"][-1]
        elif isinstance(out, (list, tuple)):
            flow = out[-1]
        else:
            flow = out
        return flow.float()


def _waft_state_dict(sd):
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in sd.items()}


def load_waft(device) -> FlowEstimator:
    import sys

    root = os.environ.get("GEOFLOW_WAFT_DIR", os.path.join(_ckpt_root(), "WAFT"))
    _require_revision(root, "b152ff1cad1af8c185ee7b141997c48ff3334c87", "GEOFLOW_WAFT_DIR")
    if root not in sys.path:
        sys.path.insert(0, root)
    # a1 code resolves depth-anything-ckpts/ relative to the WAFT checkout.
    import argparse
    import json
    from model import fetch_model
    from inference_tools import InferenceWrapper

    cfg = os.path.join(root, "config", "a1", "tar-c-t.json")
    with open(cfg) as handle:
        args = argparse.Namespace(**json.load(handle))
    previous = os.getcwd()
    os.chdir(root)
    try:
        model = fetch_model(args)
    finally:
        os.chdir(previous)
    ckpt = os.path.join(_ckpt_root(), "waft_tar_c_t.pth")
    try:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    result = model.load_state_dict(_waft_state_dict(sd), strict=False)
    if result.missing_keys:
        raise RuntimeError(f"WAFT checkpoint is missing {len(result.missing_keys)} keys: {result.missing_keys[:5]}...")
    model = model.to(device).eval()
    inference = InferenceWrapper(model, scale=0, train_size=args.image_size,
                                 pad_to_train_size=False, tiling=False)
    return WaftFlow(model, inference).to(device)


def _uint8_frames(clip: torch.Tensor) -> list:
    """[T,3,H,W] float in [0,1] -> HxWx3 uint8 RGB arrays. DA3 rejects torch tensors."""
    frames = clip.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
    return [frame.permute(1, 2, 0).numpy() for frame in frames]


def _resize_map(values: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """[T,H,W] -> [T,height,width]."""
    if values.shape[-2:] == (height, width):
        return values
    return F.interpolate(values[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]


def _scale_intrinsics(K: torch.Tensor, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> torch.Tensor:
    """Scale a pinhole K from the resolution DA3 processed to the decoded frame."""
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scaled = K.clone()
    scaled[0, 0] *= dst_w / src_w
    scaled[0, 2] *= dst_w / src_w
    scaled[1, 1] *= dst_h / src_h
    scaled[1, 2] *= dst_h / src_h
    return scaled


class DepthAnything3(torch.nn.Module):
    """Depth Anything 3 adapter. Wraps the official inference model (ByteDance-Seed/Depth-Anything-3)."""

    def __init__(self, model: torch.nn.Module, process_res: int = 504):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.process_res = process_res

    @torch.no_grad()
    def forward(self, frames01: torch.Tensor) -> DepthOutput:
        B, T, _, H, W = frames01.shape
        # One clip at a time so unrelated videos never share a scene.
        # inference() wants a list of RGB images and returns a Prediction, not a dict.
        # Its extrinsics are world-to-camera; rigid_flow needs camera-to-world.
        depths, confs, Ks, poses_list = [], [], [], []
        for b in range(B):
            kwargs = dict(process_res=self.process_res, process_res_method="upper_bound_resize", export_dir=None)
            pred = self.model.inference(_uint8_frames(frames01[b]), **kwargs)
            depth = torch.as_tensor(pred.depth, device=frames01.device, dtype=torch.float32)
            if depth.ndim == 4:
                depth = depth[..., 0]
            if depth.shape[0] != T:
                raise RuntimeError(f"DA3 returned {depth.shape[0]} frames, expected {T}")
            conf = pred.conf
            if conf is None:
                conf_t = torch.ones_like(depth)
            else:
                conf_t = torch.as_tensor(conf, device=frames01.device, dtype=torch.float32)
                if conf_t.ndim == 4:
                    conf_t = conf_t[..., 0]
            src_hw = (int(depth.shape[-2]), int(depth.shape[-1]))
            processed_h, processed_w = (src_hw[0] // 8) * 8, (src_hw[1] // 8) * 8
            if min(processed_h, processed_w) < 8:
                raise ValueError("DA3 output must be at least 8 pixels on each axis")
            depths.append(_resize_map(depth, processed_h, processed_w))
            confs.append(_resize_map(conf_t, processed_h, processed_w).clamp(0, 1))
            intrinsics = torch.as_tensor(pred.intrinsics, device=frames01.device, dtype=torch.float32)
            intrinsics = intrinsics.clone()
            intrinsics[:, 0, :] *= processed_w / src_hw[1]
            intrinsics[:, 1, :] *= processed_h / src_hw[0]
            Ks.append(intrinsics)
            w2c = torch.as_tensor(pred.extrinsics, device=frames01.device, dtype=torch.float64)
            if w2c.shape == (T, 3, 4):
                homogeneous = torch.eye(4, device=w2c.device, dtype=w2c.dtype).expand(T, -1, -1).clone()
                homogeneous[:, :3] = w2c
                w2c = homogeneous
            elif w2c.shape != (T, 4, 4):
                raise RuntimeError(
                    f"DA3 returned extrinsics with shape {tuple(w2c.shape)}, expected [{T},3,4] or [{T},4,4]"
                )
            poses_list.append(torch.linalg.inv(w2c))
        return DepthOutput(
            torch.stack(depths),
            torch.stack(Ks),
            torch.stack(poses_list),
            torch.stack(confs),
        )


def _require_revision(root, expected, variable):
    import subprocess

    try:
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        actual = "unverified"
    if actual != expected:
        raise RuntimeError(f"GeoFlow requires {expected}, found {actual} at {root}. "
                           f"Set {variable} to the pinned source checkout; see docs/geoflow_reward.md")


def load_depth_anything3(device) -> DepthEstimator:
    import sys
    from pathlib import Path

    root = os.environ.get("GEOFLOW_DA3_DIR")
    if root:
        sys.path.insert(0, str(Path(root) / "src"))
    from depth_anything_3 import api as da3_api
    _require_revision(Path(da3_api.__file__).parent,
                      "2c21ea849ceec7b469a3e62ea0c0e270afc3281a", "GEOFLOW_DA3_DIR")
    from depth_anything_3.api import DepthAnything3 as DA3

    path = os.path.join(_ckpt_root(), "depth-anything-3-large-v1.1")
    weights = os.path.join(path, "model.safetensors")
    if not os.path.isfile(weights):
        raise FileNotFoundError(
            f"Local Depth Anything 3 weights missing: {weights}. Run scripts/prefetch_wan22_login.sh on a login node."
        )
    saved = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        model = DA3.from_pretrained(path, local_files_only=True)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return DepthAnything3(model).to(device)
