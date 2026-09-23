import pytest
import torch
from diffusionopsd.video.wan_geometry import flow_timestep, latent_shape, latent_shape_from_pipe


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


def test_missing_vae_scales_mean_wan21():
    class Cfg:
        scale_factor_temporal = None

    class Vae:
        config = Cfg()

    class TrCfg:
        in_channels = 16
        patch_size = (1, 2, 2)

    class Tr:
        config = TrCfg()

    class Pipe:
        vae = Vae()
        transformer = Tr()

    # getattr default applies only when the attribute is absent, so drop it.
    delattr(Cfg, "scale_factor_temporal")
    assert latent_shape_from_pipe(Pipe(), 17, 480, 832) == (1, 16, 5, 60, 104)


def test_velocity_expands_when_the_pipeline_says_so():
    from diffusionopsd.video.wan_clean_output import WanRollout

    class Cfg:
        expand_timesteps = True

    class Pipe:
        config = Cfg()

        class transformer:
            def __call__(self, hidden_states, timestep, encoder_hidden_states, return_dict):
                assert timestep.shape == (1, 5 * 15 * 26)
                return (hidden_states,)

    pipe = Pipe()
    pipe.transformer = Pipe.transformer()
    roll = WanRollout(pipe, num_steps=2, guidance_scale=1.0)
    z = torch.zeros(1, 48, 5, 30, 52)
    out = roll.velocity(z, torch.tensor(0.25), torch.zeros(1, 2), torch.zeros(1, 2))
    assert out.shape == z.shape
