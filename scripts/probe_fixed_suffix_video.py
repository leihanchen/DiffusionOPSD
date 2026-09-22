#!/usr/bin/env python3
"""Same-query probe: G_construct, alignment, G_realized, G_fit with the geometry reward on a Wan 2.1 policy."""
from __future__ import annotations

import argparse
import copy
import json
import os

import torch
from diffusers import WanPipeline

from config.wan_video import get_config
from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
from diffusionopsd.video.geo_reward import GeoReward
from diffusionopsd.video.opa_video import opa_tr_step_nd
from diffusionopsd.video.probe_stats import probe_summary
from diffusionopsd.video.wan_clean_output import WanRollout, clean_output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--out", default="probe_fixed_suffix.jsonl")
    p.add_argument("--lr", type=float, default=1e-4)
    a = p.parse_args()
    cfg = get_config()
    dev = "cuda:0"
    pipe = WanPipeline.from_pretrained(os.path.expandvars(cfg.pretrained.model), torch_dtype=torch.bfloat16).to(dev)
    roll = WanRollout(pipe, cfg.sample.num_steps, cfg.sample.guidance_scale)
    reward = GeoReward(
        load_depth_anything3(cfg.reward_device), load_waft(cfg.reward_device), load_dinov2(cfg.reward_device)
    )
    R = lambda lat: reward(roll.decode01(lat).to(cfg.reward_device)).geo.to(lat.device)
    base_state = copy.deepcopy(pipe.transformer.state_dict())
    prompts = [l.strip() for l in open(a.prompts) if l.strip()][: a.n]
    recs = []
    for prompt in prompts:
        pe, ne = pipe.encode_prompt(prompt, negative_prompt="", do_classifier_free_guidance=True, device=dev)[:2]
        shape = (
            1,
            pipe.transformer.config.in_channels,
            (cfg.video.num_frames - 1) // 4 + 1,
            cfg.video.height // 8,
            cfg.video.width // 8,
        )
        z_T = torch.randn(shape, device=dev, dtype=torch.bfloat16)
        rec = roll.rollout(pe, ne, z_T, cfg.opa.query_sigma)
        y0 = clean_output(rec.z_q, rec.v_old_q, torch.tensor([rec.sigma_q], device=dev)).float()
        Fq = lambda y: R(roll.continue_from(rec.z_q, rec.q_index, y.to(rec.z_q.dtype), pe, ne))
        # construction
        y_plus = opa_tr_step_nd(y0, R, cfg.opa.rho, cfg.opa.n_ascent, cfg.opa.eta, +1.0)
        F0, Fplus = float(Fq(y0)), float(Fq(y_plus))
        yg = y0.clone().requires_grad_(True)
        (g_local,) = torch.autograd.grad(R(yg).sum(), yg)
        yf = y0.clone().requires_grad_(True)
        (g_suffix,) = torch.autograd.grad(Fq(yf).sum(), yf)
        align = float(torch.nn.functional.cosine_similarity(g_local.flatten(), g_suffix.flatten(), dim=0))
        # one fresh fitting update on a copy of the policy (positive branch only, paper Sec. 4.5 protocol)
        pipe.transformer.load_state_dict(base_state)
        opt = torch.optim.AdamW(pipe.transformer.parameters(), lr=a.lr)
        v_theta = roll.velocity(rec.z_q, torch.tensor(rec.sigma_q, device=dev), pe, ne)
        y_theta = clean_output(rec.z_q, v_theta, torch.tensor([rec.sigma_q], device=dev)).float()
        wf = (y_theta - y_plus).abs().mean().clamp(min=1e-5).detach()
        (((y_theta - y_plus) ** 2) / wf).mean().backward()
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            v_after = roll.velocity(rec.z_q, torch.tensor(rec.sigma_q, device=dev), pe, ne)
            y_after = clean_output(rec.z_q, v_after, torch.tensor([rec.sigma_q], device=dev)).float()
            Fafter = float(Fq(y_after))
        pipe.transformer.load_state_dict(base_state)
        r = {
            "prompt": prompt,
            "G_construct": Fplus - F0,
            "alignment": align,
            "G_realized": Fafter - F0,
            "G_fit": Fplus - Fafter,
        }
        recs.append(r)
        print(json.dumps(r))
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(probe_summary(recs), indent=2))


if __name__ == "__main__":
    main()
