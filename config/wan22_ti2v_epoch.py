"""One pass over 200 Wan2.2 prompts, four distributed workers."""

from config.wan22_ti2v_debug import get_config as debug_config


def get_config():
    config = debug_config()
    config.sample.num_batches_per_epoch = 4
    config.num_epochs = 50
    config.save_freq = 50
    config.eval_freq = 25
    config.eval_num_prompts = 2
    config.eval_num_videos = 2
    return config
