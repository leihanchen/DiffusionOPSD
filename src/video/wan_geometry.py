"""Latent geometry and the rectified-flow timestep for Wan 2.1 and Wan2.2-TI2V-5B."""

from __future__ import annotations

import torch


def latent_shape(batch, in_channels, num_frames, height, width, temporal_scale, spatial_scale, patch=(1, 2, 2)):
    """Pixel grid to [B, C, T_lat, H_lat, W_lat]. patch is (temporal, height, width)."""
    if patch[0] != 1:
        raise ValueError(f"temporal patch {patch[0]} is not 1")
    if (num_frames - 1) % temporal_scale != 0:
        raise ValueError(f"num_frames {num_frames} is not {temporal_scale}k+1")
    h_mult, w_mult = spatial_scale * patch[1], spatial_scale * patch[2]
    if height % h_mult != 0 or width % w_mult != 0:
        raise ValueError(f"frame {height}x{width} is not a multiple of {h_mult}x{w_mult}")
    frames = (num_frames - 1) // temporal_scale + 1
    return (batch, in_channels, frames, height // spatial_scale, width // spatial_scale)


def _base_transformer(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def latent_shape_from_pipe(pipe, num_frames, height, width, batch=1):
    vae = pipe.vae.config
    temporal = int(getattr(vae, "scale_factor_temporal", 4))
    spatial = int(getattr(vae, "scale_factor_spatial", 8))
    transformer = _base_transformer(pipe.transformer)
    patch = tuple(int(x) for x in transformer.config.patch_size)
    channels = int(transformer.config.in_channels)
    return latent_shape(batch, channels, num_frames, height, width, temporal, spatial, patch)


def flow_timestep(sigma, latents, expand_timesteps):
    """sigma is a scalar tensor. expand_timesteps fills one value per patch token."""
    base = (sigma.to(latents) * 1000.0).reshape(()).expand(latents.shape[0])
    if not expand_timesteps:
        return base
    _, _, frames, height, width = latents.shape
    seq = frames * (height // 2) * (width // 2)
    return base[:, None].expand(latents.shape[0], seq).contiguous()
