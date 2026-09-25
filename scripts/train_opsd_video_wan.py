#!/usr/bin/env python3
"""DiffusionOPSD for Wan 2.1 with the geometry reward (spec Sec. 5.2).

Single-node: policy on cuda:0, rewards on cuda:1.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from absl import app, flags
from diffusers import WanPipeline
from ml_collections import config_flags
from peft import get_peft_model_state_dict
import torch

from diffusionopsd.ema import EMAModuleWrapper
from diffusionopsd.stat_tracking import PerPromptStatTracker
from diffusionopsd.video.branch_loss import branch_loss
from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
from diffusionopsd.video.gates import effective_weight, identity_keep, percentile_threshold
from diffusionopsd.video.geo_reward import GeoReward
from diffusionopsd.video.opa_video import opa_tr_step_nd
from diffusionopsd.video.quality_judge import load_video_reward
from diffusionopsd.video.wan_clean_output import WanRollout, clean_output
from diffusionopsd.video.wan_policy import attach_lora, ema_adapter_

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/wan_video.py")


def _latent_shape(pipe, cfg):
    from diffusionopsd.video.wan_geometry import latent_shape_from_pipe

    return latent_shape_from_pipe(pipe, cfg.video.num_frames, cfg.video.height, cfg.video.width)


def _prompt_seed(prompt: str) -> int:
    """Deterministic per-prompt seed (Python hash() is salted per process)."""
    return int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)


def main(_):
    cfg = FLAGS.config
    dev = "cuda:0"
    model_path = os.path.expandvars(cfg.pretrained.model)
    pipe = WanPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(dev)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    from diffusionopsd.video.wan_geometry import require_expand_timesteps

    require_expand_timesteps(pipe, model_path)
    if cfg.use_lora:
        pipe.transformer.requires_grad_(False)
        pipe.transformer.enable_gradient_checkpointing()
        policy = attach_lora(pipe.transformer, cfg.train.lora_path)
        pipe.transformer = policy
        behavior = None
    else:
        policy = pipe.transformer.to(torch.float32)
        policy.enable_gradient_checkpointing()
        behavior = copy.deepcopy(policy).requires_grad_(False)
    trainable = [param for param in policy.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(
        trainable, lr=cfg.train.learning_rate, weight_decay=cfg.train.adam_weight_decay
    )
    ema = None if cfg.use_lora else EMAModuleWrapper(
        policy.parameters(), decay=0.99, update_step_interval=1, device=dev
    )
    roll_old = WanRollout(pipe, cfg.sample.num_steps, cfg.sample.guidance_scale)
    reward = GeoReward(
        load_depth_anything3(cfg.reward_device),
        load_waft(cfg.reward_device),
        load_dinov2(cfg.reward_device),
    )
    reward.requires_grad_(False)
    judge = load_video_reward(cfg.judge.device, num_frames=cfg.judge.num_frames)

    def R_geo(lat):
        return reward(roll_old.decode01(lat).to(cfg.reward_device)).geo.to(lat.device)

    tracker = PerPromptStatTracker(cfg.sample.global_std)
    with open(cfg.prompt_fn_kwargs["path"]) as f:
        prompts = [line.strip() for line in f if line.strip()]
    K = cfg.sample.num_image_per_prompt

    refs, s_ids, motions = {}, [], []
    with torch.no_grad():
        for prompt in random.Random(0).sample(prompts, min(cfg.gates.calib_prompts, len(prompts))):
            pe, ne = pipe.encode_prompt(
                prompt, negative_prompt="", do_classifier_free_guidance=True, device=dev
            )[:2]
            g = torch.Generator(dev).manual_seed(_prompt_seed(prompt))
            latents = torch.randn(_latent_shape(pipe, cfg), device=dev, dtype=torch.float32, generator=g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rec = roll_old.rollout(pe, ne, latents, cfg.opa.query_sigma)
            clip = roll_old.decode01(rec.x0)
            refs[prompt] = clip[0].cpu()
            out = reward(clip.to(cfg.reward_device))
            s_ids.append(out.s_id.cpu())
            motions.append(out.motion.cpu())
    tau_id = cfg.gates.tau_id if cfg.gates.tau_id >= 0 else percentile_threshold(torch.cat(s_ids), 0.10)
    tau_motion = (
        cfg.gates.tau_motion if cfg.gates.tau_motion >= 0 else percentile_threshold(torch.cat(motions), 0.10)
    )
    print(json.dumps({"tau_id": tau_id, "tau_motion": tau_motion, "tau_q": cfg.gates.tau_q}))

    for epoch in range(cfg.num_epochs):
        batch_prompts = random.sample(prompts, cfg.sample.num_batches_per_epoch)
        tuples = []
        if cfg.use_lora:
            policy.set_adapter("old")
        else:
            pipe.transformer = behavior
        for prompt in batch_prompts:
            pe, ne = pipe.encode_prompt(
                prompt, negative_prompt="", do_classifier_free_guidance=True, device=dev
            )[:2]
            if prompt not in refs:
                with torch.no_grad():
                    g = torch.Generator(dev).manual_seed(_prompt_seed(prompt))
                    latents = torch.randn(
                        _latent_shape(pipe, cfg), device=dev, dtype=torch.float32, generator=g
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        rec_ref = roll_old.rollout(pe, ne, latents, cfg.opa.query_sigma)
                    refs[prompt] = roll_old.decode01(rec_ref.x0)[0].cpu()
            for _ in range(K):
                latents = torch.randn(_latent_shape(pipe, cfg), device=dev, dtype=torch.float32)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    rec = roll_old.rollout(pe, ne, latents, cfg.opa.query_sigma)
                with torch.no_grad():
                    clip = roll_old.decode01(rec.x0)
                    out = reward(clip.to(cfg.reward_device))
                    p_q = judge.p_win(
                        clip[0].to(cfg.judge.device), refs[prompt].to(cfg.judge.device), prompt
                    )
                tuples.append(
                    {
                        "prompt": prompt,
                        "pe": pe,
                        "ne": ne,
                        "rec": rec,
                        "r": float(out.geo),
                        "m": float(out.motion),
                        "p_q": p_q,
                    }
                )
        adv = tracker.update([t["prompt"] for t in tuples], torch.tensor([t["r"] for t in tuples]).numpy())
        adv = torch.tensor(adv, dtype=torch.float32)
        for t, a in zip(tuples, adv):
            t["w"] = float(
                effective_weight(
                    a.view(1),
                    cfg.train.adv_clip_max,
                    torch.tensor([t["m"]]),
                    tau_motion,
                    torch.tensor([t["p_q"]]),
                    cfg.gates.tau_q,
                )
            )
        kept = []
        for t in tuples:
            rec = t["rec"]
            y0 = clean_output(rec.z_q, rec.v_old_q, torch.tensor([rec.sigma_q], device=dev)).float()
            with torch.no_grad():
                s_id = reward(roll_old.decode01(y0).to(cfg.reward_device)).s_id
            if not bool(identity_keep(s_id, tau_id).all()):
                continue
            yg = y0.clone().requires_grad_(True)
            (g0,) = torch.autograd.grad(R_geo(yg).sum(), yg)
            t["y0"] = y0
            t["y_plus"] = opa_tr_step_nd(
                y0, R_geo, cfg.opa.rho, cfg.opa.n_ascent, cfg.opa.eta, +1.0, first_grad=g0
            )
            t["y_minus"] = opa_tr_step_nd(
                y0, R_geo, cfg.opa.rho, cfg.opa.n_ascent, cfg.opa.eta, -1.0, first_grad=g0
            )
            kept.append(t)
        if cfg.use_lora:
            policy.set_adapter("default")
        else:
            pipe.transformer = policy
        policy.train()
        opt.zero_grad()
        for t in kept:
            rec = t["rec"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_theta = roll_old.velocity(
                    rec.z_q,
                    torch.tensor(rec.sigma_q, device=dev),
                    t["pe"],
                    t["ne"],
                    transformer=policy,
                )
            y_theta = clean_output(rec.z_q, v_theta, torch.tensor([rec.sigma_q], device=dev)).float()
            loss = branch_loss(
                y_theta,
                t["y0"],
                t["y_plus"],
                t["y_minus"],
                torch.tensor([t["w"]], device=dev),
                cfg.beta,
            ).mean()
            (loss * cfg.train.adv_clip_max / max(len(kept), 1)).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.train.max_grad_norm)
        if cfg.debug and (not torch.isfinite(grad_norm) or grad_norm <= 0):
            raise RuntimeError(
                f"Debug run did not produce finite nonzero gradients: "
                f"n_kept={len(kept)}, grad_norm={float(grad_norm)}"
            )
        opt.step()
        if cfg.debug:
            print(json.dumps({"optimizer_step_completed": True, "grad_norm": float(grad_norm)}), flush=True)
        if cfg.use_lora:
            ema_adapter_(policy, src="default", dst="old", decay=0.99)
        else:
            ema.step(policy.parameters(), epoch)
            with torch.no_grad():
                for pb, pp in zip(behavior.parameters(), policy.parameters()):
                    pb.mul_(0.99).add_(pp.detach(), alpha=0.01)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "n_rollouts": len(tuples),
                    "n_kept": len(kept),
                    "mean_r": sum(t["r"] for t in tuples) / len(tuples),
                    "mean_w": sum(t["w"] for t in tuples) / len(tuples),
                    "frac_motion_masked": sum(t["m"] < tau_motion for t in tuples) / len(tuples),
                    "frac_quality_masked": sum(t["p_q"] < cfg.gates.tau_q for t in tuples) / len(tuples),
                }
            )
        )
        if (epoch + 1) % cfg.save_freq == 0:
            os.makedirs(cfg.logdir, exist_ok=True)
            if cfg.use_lora:
                state = get_peft_model_state_dict(policy, adapter_name="default")
            else:
                state = policy.state_dict()
            checkpoint_path = os.path.join(cfg.logdir, f"policy_{epoch+1}.pt")
            torch.save(state, checkpoint_path)
            print(json.dumps({"checkpoint_saved": checkpoint_path}), flush=True)


if __name__ == "__main__":
    app.run(main)
