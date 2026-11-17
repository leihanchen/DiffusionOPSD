"""Native tiny decoder parity and latent derivatives, with no model downloads."""
import pytest
import torch


def tiny_vae():
    from diffusers import AutoencoderKLMiniMaxH3
    torch.manual_seed(5)
    return AutoencoderKLMiniMaxH3(latent_channels=2, block_out_channels=(8,), layers_per_block=1,
        spatial_downsample_factors=(1,), temporal_downsample_factors=(1,), norm_num_groups=2,
        decoder_num_layers=1, decoder_num_attention_heads=2, decoder_attention_head_dim=8,
        decoder_num_register_tokens=1, clip_length=3, token_drop=1,
        latents_mean=(.2, -.1), latents_std=(.7, 1.3))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('tiled', [False, True])
@pytest.mark.filterwarnings('ignore:Mismatch dtype between input and weight.*Cannot dispatch to fused implementation:UserWarning')
def test_native_decode_parity_and_gradients(dtype, tiled):
    from diffusionopsd.video.h3_auxiliary import H3RewardService
    vae = tiny_vae().to(dtype=dtype)
    service = H3RewardService(vae, lambda clip: clip.mean((1, 2, 3, 4)), device='cpu')
    if tiled:
        vae.enable_tiling(tile_sample_min_height=3, tile_sample_min_width=3,
            tile_sample_min_overlap_height=1, tile_sample_min_overlap_width=1)
    else:
        vae.disable_tiling()
    x = torch.randn(1, 2, 3, 4, 4, requires_grad=True)
    got = service.decode01(x)
    z = x * torch.tensor([.7, 1.3]).view(1, 2, 1, 1, 1) + torch.tensor([.2, -.1]).view(1, 2, 1, 1, 1)
    raw = vae.decode(z, return_dict=False)[0]
    expected = (raw.float() * torch.tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)
                + torch.tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)).clamp(0, 1).permute(0, 2, 1, 3, 4)
    torch.testing.assert_close(got, expected)
    assert got.shape == (1, 3, 3, 4, 4) and 0 <= got.min() <= got.max() <= 1
    gradient, = torch.autograd.grad(got.sum(), x)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert not any(p.requires_grad for p in vae.parameters())
