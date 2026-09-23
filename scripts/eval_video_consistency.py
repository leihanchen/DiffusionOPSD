#!/usr/bin/env python3
"""Held-out comparison of a tuned policy against the base model on the same prompts and seeds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
from diffusers import WanPipeline

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
from diffusionopsd.video.eval_table import success_table
from diffusionopsd.video.geo_reward import GeoReward
from diffusionopsd.video.quality_judge import load_video_reward
from diffusionopsd.video.wan_clean_output import WanRollout


def _get_config(name):
    if name == "wan22":
        from config.wan22_ti2v import get_config
    elif name == "wan21":
        from config.wan_video import get_config
    else:
        raise SystemExit(f"unknown --config {name}")
    return get_config()


def _latent_shape(pipe, cfg):
    from diffusionopsd.video.wan_geometry import latent_shape_from_pipe

    return latent_shape_from_pipe(pipe, cfg.video.num_frames, cfg.video.height, cfg.video.width)


def _gen(pipe, roll, cfg, prompt, seed, dev):
    pe, ne = pipe.encode_prompt(
        prompt, negative_prompt="", do_classifier_free_guidance=True, device=dev
    )[:2]
    shape = _latent_shape(pipe, cfg)
    gen = torch.Generator(dev).manual_seed(seed)
    z = torch.randn(shape, device=dev, dtype=torch.float32, generator=gen)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rec = roll.rollout(pe, ne, z, cfg.opa.query_sigma)
    return roll.decode01(rec.x0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True)
    p.add_argument("--prompts", default=None)
    p.add_argument("--out", default="eval_video.json")
    p.add_argument("--config", choices=("wan21", "wan22"), default="wan21")
    a = p.parse_args()
    cfg = _get_config(a.config)
    dev = "cuda:0"
    pipe = WanPipeline.from_pretrained(
        os.path.expandvars(cfg.pretrained.model), torch_dtype=torch.bfloat16
    ).to(dev)
    if cfg.use_lora:
        from diffusionopsd.video.wan_policy import attach_lora
        from peft import set_peft_model_state_dict

        pipe.transformer.requires_grad_(False)
        policy = attach_lora(pipe.transformer, None)
        set_peft_model_state_dict(
            policy, torch.load(a.policy, map_location="cpu"), adapter_name="default"
        )
        policy.set_adapter("default")
        pipe.transformer = policy
    else:
        pipe.transformer.to(torch.float32)
        base_sd = {k: v.clone() for k, v in pipe.transformer.state_dict().items()}
        tuned_sd = torch.load(a.policy, map_location=dev)
    roll = WanRollout(pipe, cfg.sample.num_steps, cfg.sample.guidance_scale)
    reward = GeoReward(
        load_depth_anything3(cfg.reward_device),
        load_waft(cfg.reward_device),
        load_dinov2(cfg.reward_device),
    )
    pipe.vae.requires_grad_(False)
    reward.requires_grad_(False)
    judge = load_video_reward(cfg.judge.device, num_frames=cfg.judge.num_frames)
    prompt_path = a.prompts or cfg.eval_prompts
    with open(prompt_path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    acc = {
        "base": {"geo": [], "s_id": [], "motion": []},
        "tuned": {"geo": [], "s_id": [], "motion": [], "p_q": []},
    }
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            if cfg.use_lora:
                with pipe.transformer.disable_adapter():
                    cb = _gen(pipe, roll, cfg, prompt, i, dev)
                ct = _gen(pipe, roll, cfg, prompt, i, dev)
            else:
                pipe.transformer.load_state_dict(base_sd)
                cb = _gen(pipe, roll, cfg, prompt, i, dev)
                pipe.transformer.load_state_dict(tuned_sd)
                ct = _gen(pipe, roll, cfg, prompt, i, dev)
            ob, ot = reward(cb.to(cfg.reward_device)), reward(ct.to(cfg.reward_device))
            for k in ("geo", "s_id", "motion"):
                acc["base"][k].append(float(getattr(ob, k)))
                acc["tuned"][k].append(float(getattr(ot, k)))
            acc["tuned"]["p_q"].append(
                judge.p_win(ct[0].to(cfg.judge.device), cb[0].to(cfg.judge.device), prompt)
            )
    means = {s: {k: sum(v) / len(v) for k, v in d.items()} for s, d in acc.items()}
    result = {"n": len(prompts), "means": means, "success": success_table(means["base"], means["tuned"])}
    with open(a.out, "w") as out_f:
        json.dump(result, out_f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
