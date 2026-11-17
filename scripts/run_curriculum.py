#!/usr/bin/env python3
"""Run or resume a manifest-bound curriculum experiment from a JSON configuration."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusionopsd.curriculum.runner import run_experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run_experiment(args.config, args.run_dir)


if __name__ == '__main__':
    main()
