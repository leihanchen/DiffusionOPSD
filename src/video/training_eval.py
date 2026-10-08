"""Periodic evaluation using the policy and scorers already loaded by training."""

from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from diffusionopsd.video.eval_table import success_table
from diffusionopsd.video.quality_judge import gap_to_probability, quality_gap
from diffusionopsd.video.wan_geometry import latent_shape_from_pipe


def evaluation_due(update: int, total_updates: int, frequency: int) -> bool:
    return frequency > 0 and (update == 0 or update % frequency == 0 or update == total_updates)


@contextmanager
def evaluation_state(pipe, policy, reward, use_lora):
    """Restore adapter, module modes, parameter flags, and RNG even after failure."""
    original_transformer = pipe.transformer
    modules = {m: m.training for root in (policy, pipe.vae, pipe.text_encoder, reward) for m in root.modules()}
    requires_grad = {p: p.requires_grad for p in policy.parameters()}
    adapters = list(policy.active_adapters) if use_lora else None
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        try:
            pipe.transformer = policy
            if use_lora:
                policy.set_adapter("default")
            for root in (policy, pipe.vae, pipe.text_encoder, reward):
                root.eval()
            with torch.no_grad():
                yield
        finally:
            pipe.transformer = original_transformer
            if use_lora:
                policy.set_adapter(adapters[0] if len(adapters) == 1 else adapters)
            for parameter, enabled in requires_grad.items():
                parameter.requires_grad_(enabled)
            for module, training in modules.items():
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)


def generate_clip(pipe, roll, cfg, prompt, seed, device):
    """Return a decoded clip [1,T,3,H,W] using a local noise generator."""
    pe, ne = pipe.encode_prompt(prompt, negative_prompt="", do_classifier_free_guidance=True, device=device)[:2]
    shape = latent_shape_from_pipe(pipe, cfg.video.num_frames, cfg.video.height, cfg.video.width)
    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with autocast:
        rec = roll.rollout(pe, ne, latents, cfg.opa.query_sigma)
    return roll.decode01(rec.x0)


def save_video(clip, path, fps):
    """Encode [T,3,H,W] frames without keeping decoded clips in the monitor."""
    import imageio.v2 as imageio

    frames = (clip.detach().clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264", macro_block_size=1)


class EvaluationMonitor:
    """Fixed prompts/seeds and scalar baseline scores, independent of checkpoints."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.baseline = None
        self.prompts = []
        if cfg.eval_freq < 0:
            raise ValueError("eval_freq must be nonnegative; 0 disables evaluation")
        if cfg.eval_freq == 0:
            return
        if cfg.eval_num_prompts <= 0 or not 0 <= cfg.eval_num_videos <= cfg.eval_num_prompts:
            raise ValueError("evaluation requires positive eval_num_prompts and 0 <= eval_num_videos <= it")
        if cfg.eval_video_fps <= 0 or cfg.eval_seed < 0:
            raise ValueError("eval_video_fps must be positive and eval_seed nonnegative")
        prompts = [line.strip() for line in Path(cfg.eval_prompts).read_text().splitlines() if line.strip()]
        if len(prompts) < cfg.eval_num_prompts:
            raise ValueError(f"evaluation requested {cfg.eval_num_prompts} prompts but found {len(prompts)}")
        self.prompts = prompts[:cfg.eval_num_prompts]

    def evaluate(self, pipe, policy, roll, reward, judge, update):
        if not self.prompts:
            return {}, []
        if self.baseline is None and update != 0:
            raise ValueError("evaluate update 0 before comparing subsequent updates")
        started = time.perf_counter()
        cfg = self.cfg
        output = Path(cfg.logdir) / "eval"
        output.mkdir(parents=True, exist_ok=True)
        rows, videos = [], []
        with evaluation_state(pipe, policy, reward, cfg.use_lora):
            device = next(policy.parameters()).device
            for index, prompt in enumerate(self.prompts):
                seed = cfg.eval_seed + index
                clip = generate_clip(pipe, roll, cfg, prompt, seed, device)
                geometry = reward(clip.to(cfg.reward_device))
                scores = judge.score(clip[0].to(cfg.judge.device), prompt)
                row = {"prompt": prompt, "seed": seed}
                row.update({key: float(getattr(geometry, key)) for key in ("geo", "s_id", "motion")})
                row.update({key: float(scores[key]) for key in ("VQ", "MQ", "TA", "Overall")})
                reference = row if self.baseline is None else self.baseline[index]
                row["p_q"] = gap_to_probability(quality_gap(row, reference))
                rows.append(row)
                if index < cfg.eval_num_videos:
                    path = output / f"update_{update:06d}_prompt_{index:03d}.mp4"
                    save_video(clip[0], path, cfg.eval_video_fps)
                    videos.append({"path": str(path), "prompt": prompt, "seed": seed})
                del clip, geometry, scores
        baseline = rows if self.baseline is None else self.baseline
        keys = ("geo", "s_id", "motion", "VQ", "MQ", "TA", "Overall", "p_q")
        means = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
        base_means = {key: sum(row[key] for row in baseline) / len(baseline) for key in keys}
        success = success_table(base_means, means)
        metrics = {f"eval/{key}": value for key, value in means.items()}
        metrics.update({f"eval/{key}_delta": means[key] - base_means[key] for key in keys})
        metrics.update({f"eval/{key}": value for key, value in success.items()})
        metrics.update({"eval/num_prompts": len(rows), "eval/duration_seconds": time.perf_counter() - started})
        result = {"optimizer_updates": update, "prompts": rows, "means": means,
                  "baseline_means": base_means, "success": success, "metrics": metrics, "videos": videos}
        (output / f"update_{update:06d}.json").write_text(json.dumps(result, indent=2) + "\n")
        # Publish a baseline only after every prompt and output has succeeded.
        if self.baseline is None:
            self.baseline = rows
        return metrics, videos
