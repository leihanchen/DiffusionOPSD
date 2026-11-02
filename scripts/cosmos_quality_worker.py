#!/usr/bin/env python3
"""VideoReward RPC worker using the unchanged container's Transformers stack."""

import argparse
from contextlib import redirect_stdout
import json
import sys

import torch

from diffusionopsd.video.quality_judge import load_video_reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        judge = load_video_reward(args.device)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        with redirect_stdout(sys.stderr):
            clip = torch.load(request["clip"], map_location="cpu", weights_only=True)
            result = judge.score(clip, request["prompt"])
        print(json.dumps({"score": result}), flush=True)


if __name__ == "__main__":
    main()
