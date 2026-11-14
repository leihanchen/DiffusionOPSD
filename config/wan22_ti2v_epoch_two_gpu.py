"""Compatibility alias for the standard two-GPU-per-worker epoch preset."""

from config.wan22_ti2v_epoch import get_config as epoch_config


def get_config():
    config = epoch_config()
    config.vae_device = "cuda:1"
    return config
