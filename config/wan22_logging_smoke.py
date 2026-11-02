"""Small-video logging integration check; not a quality-training preset."""

from config.wan22_ti2v_epoch_two_gpu import get_config as epoch_config


def get_config():
    config = epoch_config()
    config.video.height = 224
    config.video.width = 384
    config.video.num_frames = 5
    config.sample.num_steps = 8
    return config
