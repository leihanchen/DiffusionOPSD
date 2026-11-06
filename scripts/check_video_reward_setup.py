#!/usr/bin/env python3
"""Load the real estimators, score one synthetic clip, and verify a finite frame-space gradient."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
from diffusionopsd.video.geo_reward import GeoReward, GEOFLOW_REVISION, quantize_frames, reward_metadata
from diffusionopsd.video.opa_video import opa_tr_step_nd


def reference_score(reference, reward, clip):
    """Run the original server with the same loaded backbones and decoded RGB8."""
    import inspect
    import subprocess
    from types import SimpleNamespace
    import cv2
    import torch.nn.functional as F
    import torchvision.transforms.functional as TVF

    actual = subprocess.check_output(["git", "-C", reference, "rev-parse", "HEAD"], text=True).strip()
    if actual != GEOFLOW_REVISION:
        raise ValueError(f"Expected GeoFlow {GEOFLOW_REVISION}, found {actual}")
    sys.path.insert(0, reference)
    from geoflow.reward_server import FastRewardWorker

    worker = FastRewardWorker.__new__(FastRewardWorker)
    for name, parameter in inspect.signature(FastRewardWorker).parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            setattr(worker, name, parameter.default)
    worker.device = str(clip.device)
    worker.torch, worker.TF, worker.TVF = torch, F, TVF
    worker._da3, worker._flow = reward.depth.model, reward.flow.inference
    worker._dino = SimpleNamespace(
        patch_size=reward.feats.patch, model=reward.feats.backbone,
        mean=torch.tensor([0.485, 0.456, 0.406], device=clip.device).view(3, 1, 1),
        std=torch.tensor([0.229, 0.224, 0.225], device=clip.device).view(3, 1, 1),
        _num_prefix_tokens=1 + int(getattr(reward.feats.backbone.config, "num_register_tokens", 0) or 0),
    )
    rgb = quantize_frames(clip.detach())[0].mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    pngs = [cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1].tobytes() for frame in rgb]
    with torch.no_grad():
        return worker.score_video(pngs, fps=8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--side", type=int, default=224)
    p.add_argument("--reference", help="Pinned GeoFlow checkout for real-model forward parity")
    p.add_argument("--clip-tensor", help="Saved [1,T,3,H,W] float RGB clip, loaded with weights_only=True")
    a = p.parse_args()
    t0 = time.perf_counter()
    reward = GeoReward(load_depth_anything3(a.device), load_waft(a.device), load_dinov2(a.device))
    clip = torch.rand(1, a.frames, 3, a.side, a.side, device=a.device, requires_grad=True)
    if a.clip_tensor:
        clip = torch.load(a.clip_tensor, map_location=a.device, weights_only=True).float().requires_grad_(True)
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    out = reward(clip)
    (g,) = torch.autograd.grad(out.geo.sum(), clip)
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    gradient_seconds = time.perf_counter() - started
    parity = None
    if a.reference:
        reference = reference_score(a.reference, reward, clip)
        parity = {}
        for field, ref_field in (("geo", "score"), ("rigid", "mean_r_geo"), ("dino", "mean_r_dino")):
            value = getattr(out, field).detach().cpu()
            expected = torch.tensor([reference[ref_field]])
            torch.testing.assert_close(value, expected, atol=1e-5, rtol=1e-4)
            parity[field] = float((value - expected).abs().max())
    started = time.perf_counter()
    for direction in (1, -1):
        target = opa_tr_step_nd(clip.detach(), lambda x: reward(x).geo, rho=0.01, n_ascent=2,
                                eta=1.0, direction=direction, first_grad=g)
        if not torch.isfinite(target).all():
            raise RuntimeError("Nonfinite reward target")
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    rec = {
        "reward": reward_metadata(),
        "parity_absolute_errors": parity,
        "forward_backward_s": gradient_seconds,
        "target_construction_s": time.perf_counter() - started,
        "geo": float(out.geo),
        "rigid": float(out.rigid),
        "dino": float(out.dino),
        "motion": float(out.motion),
        "frame_grad_norm": float(g.norm()),
        "elapsed_s": time.perf_counter() - t0,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30 if a.device.startswith("cuda") else None,
    }
    for k in ("geo", "frame_grad_norm"):
        if not math.isfinite(rec[k]) or (k == "frame_grad_norm" and rec[k] <= 0):
            raise RuntimeError(f"bad {k}: {rec[k]}")
    print(json.dumps(rec, sort_keys=True))


if __name__ == "__main__":
    main()
