"""Rectified-flow clean-output map, rollout with low-noise query capture, and fixed-suffix continuation for Wan 2.1."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from diffusionopsd.video.wan_geometry import flow_timestep


def clean_output(z: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    s = sigma.to(z).view(-1, *([1] * (z.ndim - 1)))
    return z - s * v


def select_query_index(sigmas: torch.Tensor, sigma_star: float) -> int:
    return int(torch.argmin((sigmas - sigma_star).abs()).item())


@dataclass
class RolloutRecord:
    x0: torch.Tensor
    z_q: torch.Tensor
    sigma_q: float
    q_index: int
    v_old_q: torch.Tensor
    sigmas: torch.Tensor


class WanRollout:
    def __init__(self, pipeline, num_steps: int, guidance_scale: float):
        self.pipe, self.num_steps, self.g = pipeline, num_steps, guidance_scale

    def _sigmas(self, device):
        self.pipe.scheduler.set_timesteps(self.num_steps, device=device)
        return self.pipe.scheduler.sigmas.to(device)  # length num_steps+1, ends at 0

    def velocity(self, z, sigma, prompt_embeds, negative_embeds, transformer=None):
        tr = transformer if transformer is not None else self.pipe.transformer
        expand = bool(getattr(self.pipe.config, "expand_timesteps", False))
        t = flow_timestep(sigma, z, expand)
        v_c = tr(hidden_states=z, timestep=t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
        if self.g == 1.0:
            return v_c
        v_u = tr(hidden_states=z, timestep=t, encoder_hidden_states=negative_embeds, return_dict=False)[0]
        return v_u + self.g * (v_c - v_u)

    @torch.no_grad()
    def rollout(self, prompt_embeds, negative_embeds, latents, sigma_star: float) -> RolloutRecord:
        sig = self._sigmas(latents.device)
        q = select_query_index(sig[:-1], sigma_star)
        z, z_q, v_old_q = latents, None, None
        for j in range(self.num_steps):
            v = self.velocity(z, sig[j], prompt_embeds, negative_embeds)
            if j == q:
                z_q, v_old_q = z.clone(), v.clone()
            z = z + (sig[j + 1] - sig[j]) * v  # Euler step on the rectified path
        return RolloutRecord(z, z_q, float(sig[q]), q, v_old_q, sig)

    def continue_from(self, z_q, q_index, forced_y, prompt_embeds, negative_embeds):
        """Differentiable w.r.t. forced_y; wrap in torch.no_grad() when gradients are not needed."""
        sig = self._sigmas(z_q.device)
        v = (z_q - forced_y) / sig[q_index]
        z = z_q + (sig[q_index + 1] - sig[q_index]) * v
        for j in range(q_index + 1, self.num_steps):
            z = z + (sig[j + 1] - sig[j]) * self.velocity(z, sig[j], prompt_embeds, negative_embeds)
        return z

    def decode01(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self.pipe.vae
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
        std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
        lat = latents / std + mean
        video = vae.decode(lat.to(vae.dtype), return_dict=False)[0]  # [B,3,T,H,W] in [-1,1]
        return (video / 2 + 0.5).clamp(0, 1).float().permute(0, 2, 1, 3, 4)
