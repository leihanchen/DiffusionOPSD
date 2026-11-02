"""Exercise native UniPC capture and replay without downloading a video model."""

from types import SimpleNamespace

import pytest
import torch
from diffusers import UniPCMultistepScheduler

from diffusionopsd.video.cosmos_rollout import CosmosRollout


class TinyPackedTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.12))

    @property
    def dtype(self):
        return self.scale.dtype

    def forward(self, vision_tokens, vision_timesteps, input_ids, **kwargs):
        z = vision_tokens[0]
        v = self.scale * z.sin() + input_ids.float().mean() * 0.02 + vision_timesteps[0] * 0.0001
        return [v], None, None


class TinyVAE:
    device = torch.device("cpu")
    dtype = torch.float32
    config = SimpleNamespace(latents_mean=[0.1], latents_std=[0.7])

    def decode(self, z, return_dict=False):
        return (z.sin(),)


class TinyPipeline:
    def __init__(self, native_axis=False):
        self.transformer = TinyPackedTransformer()
        self.vae = TinyVAE()
        self.scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=2.0, solver_order=2
        )
        self.fail = False
        self.native_axis = native_axis

    @torch.no_grad()
    def __call__(self, prompt, num_inference_steps, guidance_scale, generator=None, latents=None, **kwargs):
        self.scheduler.set_timesteps(num_inference_steps)
        z = torch.randn((1, 3, 2, 2), generator=generator) if latents is None else latents.clone()
        if self.native_axis:
            z = z.unsqueeze(0)
        for t in self.scheduler.timesteps:
            cond = self.transformer(
                vision_tokens=[z], vision_timesteps=t.reshape(1), input_ids=torch.tensor([3]), return_dict=False
            )[0][0]
            if self.fail:
                raise RuntimeError("generation failed")
            if guidance_scale > 1:
                uncond = self.transformer(
                    vision_tokens=[z], vision_timesteps=t.reshape(1), input_ids=torch.tensor([0]), return_dict=False
                )[0][0]
                cond = uncond + guidance_scale * (cond - uncond)
            z = self.scheduler.step(cond.unsqueeze(0), t, z.unsqueeze(0), return_dict=False)[0].squeeze(0)
        return z, None


@pytest.mark.parametrize("guidance", [1.0, 6.0])
@pytest.mark.parametrize("native_axis", [False, True])
def test_native_endpoint_query_replay_and_suffix_parity(guidance, native_axis):
    pipe = TinyPipeline(native_axis=native_axis)
    roll = CosmosRollout(pipe, num_steps=8, guidance_scale=guidance)
    noise = torch.linspace(-0.6, 0.6, 12).reshape(1, 3, 2, 2)
    native = pipe("prompt", num_inference_steps=8, guidance_scale=guidance, latents=noise)[0]
    rec = roll.rollout("prompt", latents=noise, sigma_star=0.4)
    assert rec.z_q.ndim == 5
    assert torch.equal(rec.x0, native if native.ndim == 5 else native.unsqueeze(0))
    assert rec.q_index == int((rec.sigmas[:-1] - 0.4).abs().argmin())
    prediction = roll.prediction(rec)
    assert torch.allclose(prediction, rec.z_q - rec.sigma_q * rec.v_old_q, atol=1e-6)
    continued = roll.continue_from(rec, prediction.detach())
    assert torch.allclose(continued, rec.x0, atol=2e-6)
    prediction.square().mean().backward()
    assert pipe.transformer.scale.grad is not None
    assert pipe.transformer.scale.grad.abs() > 0


def test_suffix_changes_with_target_and_is_differentiable():
    roll = CosmosRollout(TinyPipeline(), 8, 6.0)
    rec = roll.rollout("prompt", generator=torch.Generator().manual_seed(4), sigma_star=0.4)
    y = (rec.z_q - rec.sigma_q * rec.v_old_q).requires_grad_()
    endpoint = roll.continue_from(rec, y)
    endpoint.sum().backward()
    assert y.grad is not None and y.grad.abs().sum() > 0
    assert not torch.allclose(roll.continue_from(rec, y.detach() + 0.1), endpoint)
    # Repeated continuations must not consume the stored scheduler history.
    assert torch.equal(roll.continue_from(rec, y.detach()), endpoint.detach())


def test_capture_restores_hooks_and_scheduler_after_failure():
    pipe = TinyPipeline()
    pipe.fail = True
    roll = CosmosRollout(pipe, 8, 6.0)
    with pytest.raises(RuntimeError, match="generation failed"):
        roll.rollout("prompt")
    assert not pipe.transformer._forward_pre_hooks
    assert "step" not in vars(pipe.scheduler)
    pipe.fail = False
    assert torch.isfinite(roll.rollout("prompt").x0).all()


def test_decode_keeps_latent_gradient_and_normalization():
    roll = CosmosRollout(TinyPipeline(), 8, 1.0)
    y = torch.full((1, 1, 3, 2, 2), 0.2, requires_grad=True)
    video = roll.decode01(y)
    assert video.shape == (1, 3, 1, 2, 2)
    assert torch.allclose(video, torch.full_like(video, (torch.sin(torch.tensor(0.24)) + 1) / 2))
    video.sum().backward()
    assert y.grad.abs().sum() > 0


@pytest.mark.parametrize("sigma", [0.0, 1.1, float("nan")])
def test_invalid_query_rejected(sigma):
    with pytest.raises(ValueError, match="sigma"):
        CosmosRollout(TinyPipeline(), 8, 1.0).rollout("prompt", sigma_star=sigma)
