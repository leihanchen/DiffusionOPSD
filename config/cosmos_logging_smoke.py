"""Small-video offline logging check; preserves the OPSD update algorithm."""

from config.cosmos_nano import get_config as nano_config


def get_config():
    config = nano_config()
    config.height = 224
    config.width = 384
    config.num_frames = 5
    config.reward_frames = 5
    config.num_steps = 8
    config.calibration_prompts = 2
    return config
