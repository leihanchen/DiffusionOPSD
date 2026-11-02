"""Native Cosmos3 T2V trajectories with differentiable OPSD query replay.

Latents use [B,C,T,H,W]. Cosmos runs B=1 and retains that axis inside each
packed item, then adds an outer solver batch axis. No sampler is replaced.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint


def _detach_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _detach_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_detach_tree(v) for v in value)
    return value


def _copy_scheduler(scheduler):
    # Exclude the temporary per-instance step interceptor and its closure.
    result = copy.copy(scheduler)
    result.__dict__ = copy.deepcopy({k: v for k, v in vars(scheduler).items() if k != "step"})
    return result


@dataclass
class CosmosQuery:
    x0: torch.Tensor
    z_q: torch.Tensor
    sigma_q: float
    q_index: int
    v_old_q: torch.Tensor
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    forward_inputs: list[dict]
    scheduler: object
    solver_outer_axis: bool

    @property
    def y0(self):
        return self.z_q.float() - self.sigma_q * self.v_old_q.float()


class CosmosRollout:
    """Capture native UniPC state; scope is one text-only video, without caches."""

    def __init__(self, pipeline, num_steps=35, guidance_scale=6.0, offload_vae_activations=False):
        if num_steps < 2 or not math.isfinite(guidance_scale) or guidance_scale < 1:
            raise ValueError("Cosmos requires num_steps >= 2 and guidance_scale >= 1")
        if pipeline.scheduler.config.prediction_type != "flow_prediction":
            raise ValueError("Cosmos OPSD requires flow_prediction")
        self.pipe, self.num_steps, self.g = pipeline, num_steps, guidance_scale
        self.offload_vae_activations = offload_vae_activations

    @torch.no_grad()
    def rollout(self, prompt, generator=None, sigma_star=0.278, **geometry):
        if not math.isfinite(sigma_star) or not 0 < sigma_star <= 1:
            raise ValueError("query sigma must lie in (0, 1]")
        forbidden = {"image", "video", "action", "sound_latents", "action_latents", "enable_sound"} & geometry.keys()
        if forbidden:
            raise ValueError(f"Cosmos OPSD supports text-only video: unsupported {sorted(forbidden)}")
        scheduler = self.pipe.scheduler
        original_step = scheduler.step
        previous_instance_step = vars(scheduler).get("step")
        calls, captured = [], {}
        step_index = 0

        def capture_inputs(module, args, kwargs):
            q = int((scheduler.sigmas[:-1] - sigma_star).abs().argmin())
            if step_index == q:
                calls.append(_detach_tree(kwargs))

        def capture_step(model_output, timestep, sample, *args, **kwargs):
            nonlocal step_index
            q = int((scheduler.sigmas[:-1] - sigma_star).abs().argmin())
            if step_index == q:
                expected = 2 if self.g > 1 else 1
                if len(calls) != expected:
                    raise RuntimeError(f"Expected {expected} Cosmos CFG calls, captured {len(calls)}")
                captured.update(
                    z_q=(sample[0] if sample.ndim == 6 else sample).detach().clone(),
                    sigma_q=float(scheduler.sigmas[q]),
                    q_index=q,
                    v_old_q=(model_output[0] if model_output.ndim == 6 else model_output).detach().clone(),
                    sigmas=scheduler.sigmas.detach().clone(),
                    timesteps=scheduler.timesteps.detach().clone(),
                    forward_inputs=calls.copy(),
                    scheduler=_copy_scheduler(scheduler),
                    solver_outer_axis=sample.ndim == 6,
                )
            result = original_step(model_output, timestep, sample, *args, **kwargs)
            step_index += 1
            return result

        hook = self.pipe.transformer.register_forward_pre_hook(capture_inputs, with_kwargs=True)
        scheduler.step = capture_step
        original_encode = getattr(self.pipe, "_encode_video", None)
        previous_instance_encode = vars(self.pipe).get("_encode_video")
        if original_encode is not None:
            # Native prepare_latents encodes a zero video even in text-only mode.
            # Preserve its values while allowing a VAE on a separate GPU.
            def encode_on_vae(x):
                return original_encode(x.to(self.pipe.vae.device)).to(x.device)

            self.pipe._encode_video = encode_on_vae
        try:
            output = self.pipe(
                prompt=prompt,
                generator=generator,
                num_inference_steps=self.num_steps,
                guidance_scale=self.g,
                enable_sound=False,
                enable_safety_check=False,
                output_type="latent",
                return_dict=False,
                **geometry,
            )[0]
        finally:
            hook.remove()
            if previous_instance_step is None:
                del scheduler.step
            else:
                scheduler.step = previous_instance_step
            if original_encode is not None:
                if previous_instance_encode is None:
                    del self.pipe._encode_video
                else:
                    self.pipe._encode_video = previous_instance_encode
        if not captured:
            raise RuntimeError("Native Cosmos pipeline never reached the OPSD query")
        endpoint = output.unsqueeze(0) if output.ndim == 4 else output
        return CosmosQuery(x0=endpoint.detach(), **captured)

    def velocity(self, record, latents=None, timestep=None):
        z = record.z_q if latents is None else latents
        predictions = []
        for saved in record.forward_inputs:
            kwargs = dict(saved)
            tokens = z if saved["vision_tokens"][0].ndim == 5 else z[0]
            kwargs["vision_tokens"] = [tokens.to(self.pipe.transformer.dtype)]
            if timestep is not None:
                kwargs["vision_timesteps"] = torch.full_like(saved["vision_timesteps"], float(timestep))
            prediction = self.pipe.transformer(**kwargs)[0][0].float()
            predictions.append(prediction if prediction.ndim == 5 else prediction.unsqueeze(0))
        if self.g == 1:
            return predictions[0]
        cond, uncond = predictions
        return uncond + self.g * (cond - uncond)

    def prediction(self, record):
        return record.z_q.float() - record.sigma_q * self.velocity(record)

    def continue_from(self, record, forced_y):
        """Restore pre-query UniPC history; caller holds behavior weights fixed."""
        scheduler = _copy_scheduler(record.scheduler)
        q = record.q_index

        def step(v, t, z):
            if record.solver_outer_axis:
                return scheduler.step(v.unsqueeze(0), t, z.unsqueeze(0), return_dict=False)[0].squeeze(0)
            return scheduler.step(v, t, z, return_dict=False)[0]

        velocity = (record.z_q - forced_y) / record.sigma_q
        z = step(velocity, record.timesteps[q], record.z_q)
        for timestep in record.timesteps[q + 1 :]:
            velocity = self.velocity(record, latents=z, timestep=timestep)
            z = step(velocity, timestep, z)
        return z

    def decode01(self, latents):
        vae = self.pipe.vae
        # Match native Cosmos normalization, including rounding statistics to
        # VAE precision before the affine transformation.
        latents = latents.to(device=vae.device, dtype=vae.dtype)
        mean = torch.tensor(vae.config.latents_mean, device=vae.device, dtype=vae.dtype)
        inv_std = 1 / torch.tensor(vae.config.latents_std, device=vae.device, dtype=vae.dtype)
        raw = latents / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        if torch.is_grad_enabled() and raw.requires_grad:
            if self.offload_vae_activations:
                with torch.autograd.graph.save_on_cpu():
                    video = vae.decode(raw, return_dict=False)[0]
            else:
                video = checkpoint(lambda z: vae.decode(z, return_dict=False)[0], raw, use_reentrant=False)
        else:
            video = vae.decode(raw, return_dict=False)[0]
        return (video.float() / 2 + 0.5).clamp(0, 1).permute(0, 2, 1, 3, 4)
