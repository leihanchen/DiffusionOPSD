from types import SimpleNamespace

import torch

from diffusionopsd.video.wan_clean_output import WanRollout, clean_output, select_query_index


def test_clean_output_recovers_data_on_rectified_path():
    torch.manual_seed(0)
    y = torch.randn(2, 4, 3, 5, 5)
    eps = torch.randn_like(y)
    sigma = torch.tensor([0.3, 0.7])
    s = sigma.view(-1, 1, 1, 1, 1)
    z = (1 - s) * y + s * eps
    v = eps - y
    assert torch.allclose(clean_output(z, v, sigma), y, atol=1e-5)


def test_select_query_index_nearest():
    sig = torch.tensor([1.0, 0.8, 0.5, 0.3, 0.1, 0.0])
    assert select_query_index(sig, 0.278) == 3


def test_decode01_recomputes_frozen_vae_for_latent_gradient():
    class VAE:
        dtype = torch.float32
        device = torch.device("cpu")
        config = SimpleNamespace(latents_mean=[0.0], latents_std=[1.0])

        def __init__(self):
            self.calls = 0

        def decode(self, latent, return_dict=False):
            self.calls += 1
            return (torch.sin(latent),)

    vae = VAE()
    rollout = WanRollout(SimpleNamespace(vae=vae), num_steps=1, guidance_scale=1.0)
    latent = torch.full((1, 1, 1, 2, 2), 0.2, requires_grad=True)

    rollout.decode01(latent).sum().backward()

    assert vae.calls == 2
    assert torch.allclose(latent.grad, torch.full_like(latent, torch.cos(torch.tensor(0.2)) / 2))


def test_decode01_places_latents_on_vae_device():
    class VAE:
        dtype = torch.float32
        device = torch.device("meta")
        config = SimpleNamespace(latents_mean=[0.0], latents_std=[1.0])

        def decode(self, latent, return_dict=False):
            assert latent.device == self.device
            return (latent,)

    rollout = WanRollout(SimpleNamespace(vae=VAE()), num_steps=1, guidance_scale=1.0)
    decoded = rollout.decode01(torch.ones(1, 1, 1, 2, 2))
    assert decoded.device.type == "meta"
