"""One-epoch debugging run at the normal video resolution and sampling steps."""

from config.wan22_ti2v import get_config as wan22_config


def get_config():
    config = wan22_config()
    config.debug = True
    config.gates.calib_prompts = 2
    config.num_epochs = 1
    config.save_freq = 1
    return config
