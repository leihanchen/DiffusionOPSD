"""Reduced canvas and schedule, retaining native duration and P-rank protocol."""
from config.minimax_h3_t2va import get_config as pilot_config


def get_config():
    cfg = pilot_config()
    cfg.conditioning_cache = '${VIDEO_REWARD_CKPT_PATH}/minimax-h3-conditioning-smoke'
    cfg.height = cfg.width = 256
    cfg.num_steps = 16
    cfg.updates = 1
    cfg.eval_count = 4
    cfg.eval_every = 1
    return cfg
