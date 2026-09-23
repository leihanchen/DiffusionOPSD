import pytest
import torch
from diffusionopsd.video.wan_geometry import flow_timestep, latent_shape


def test_wan21_latent_at_the_training_grid():
    assert latent_shape(1, 16, 17, 480, 832, 4, 8) == (1, 16, 5, 60, 104)


def test_ti2v_latent_at_the_training_grid():
    assert latent_shape(1, 48, 17, 480, 832, 4, 16) == (1, 48, 5, 30, 52)


def test_illegal_frame_count_and_side_raise():
    with pytest.raises(ValueError):
        latent_shape(1, 16, 16, 480, 832, 4, 8)
    with pytest.raises(ValueError):
        latent_shape(1, 48, 17, 480, 816, 4, 16)


def test_timestep_is_a_batch_vector_until_expand_is_on():
    latents = torch.zeros(1, 48, 5, 30, 52)
    sigma = torch.tensor(0.25)
    short = flow_timestep(sigma, latents, False)
    wide = flow_timestep(sigma, latents, True)
    assert short.shape == (1,)
    assert wide.shape == (1, 5 * 15 * 26)
    assert torch.allclose(short, torch.tensor([250.0]))
    assert torch.allclose(wide, torch.full((1, 1950), 250.0))
