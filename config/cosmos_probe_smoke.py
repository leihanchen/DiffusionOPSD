"""Four-prompt Step 6 smoke with matched, original-EMA continuations."""
from config.cosmos_logging_smoke import get_config as smoke_config


def get_config():
    config = smoke_config()
    config.paired_probe_count = 4
    config.num_steps = 35
    # The 8-step schedule's nearest query was its final step. Leave a real suffix
    # for the continuation audit while keeping the same small video geometry.
    config.query_sigma = .60
    return config
