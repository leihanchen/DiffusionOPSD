"""Unqualified opt-in two-node smoke. Requires single-node then two-node gates."""
from config.minimax_h3_t2va import get_config as base_config


def get_config():
    cfg = base_config()
    cfg.num_nodes = 2
    cfg.updates = 1
    cfg.eval_every = 1
    return cfg
