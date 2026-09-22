#!/usr/bin/env python3
"""Load the real estimators, score one synthetic clip, and verify a finite frame-space gradient."""
from __future__ import annotations
import argparse
import json
import math
import time
import torch
from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
from diffusionopsd.video.geo_reward import GeoReward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--side", type=int, default=224)
    a = p.parse_args()
    t0 = time.perf_counter()
    reward = GeoReward(load_depth_anything3(a.device), load_waft(a.device), load_dinov2(a.device))
    clip = torch.rand(1, a.frames, 3, a.side, a.side, device=a.device, requires_grad=True)
    out = reward(clip)
    (g,) = torch.autograd.grad(out.geo.sum(), clip)
    rec = {"geo": float(out.geo), "rigid": float(out.rigid), "dino": float(out.dino), "motion": float(out.motion),
           "frame_grad_norm": float(g.norm()), "elapsed_s": time.perf_counter() - t0,
           "peak_gib": torch.cuda.max_memory_allocated() / 2**30 if a.device.startswith("cuda") else None}
    for k in ("geo", "frame_grad_norm"):
        if not math.isfinite(rec[k]) or (k == "frame_grad_norm" and rec[k] <= 0):
            raise RuntimeError(f"bad {k}: {rec[k]}")
    print(json.dumps(rec, sort_keys=True))


if __name__ == "__main__":
    main()
