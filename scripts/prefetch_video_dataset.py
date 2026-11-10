#!/usr/bin/env python3
"""Cache and validate the pinned video dataset before an offline Slurm job."""

import argparse
from pathlib import Path
import json
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusionopsd.video.training_data import default_dataset_config, load_training_data  # noqa: E402


def main():
    settings = default_dataset_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=settings["repo_id"])
    parser.add_argument("--revision", default=settings["revision"])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--config", help="Use the exact dataset configuration of a training preset")
    args = parser.parse_args()
    settings.update(repo_id=args.repo_id, revision=args.revision)
    if args.config:
        import importlib.util

        spec = importlib.util.spec_from_file_location("prefetch_config", args.config)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = module.get_config()
    else:
        cfg = SimpleNamespace(dataset=settings)
    data = load_training_data(cfg, offline=args.check_only)
    print(json.dumps(data.metadata, indent=2))


if __name__ == "__main__":
    main()
