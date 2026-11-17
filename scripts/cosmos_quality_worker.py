#!/usr/bin/env python3
"""VideoReward RPC worker using the unchanged container's Transformers stack."""

import argparse
from contextlib import redirect_stdout
import json
import sys

import torch

from diffusionopsd.video.quality_judge import load_video_reward


def dispatch(judge, request, device, residency_protocol=False):
    """Execute one graph-free command; legacy score dictionaries stay unchanged."""
    if not residency_protocol:
        clip = torch.load(request["clip"], map_location="cpu", weights_only=True)
        return {"score": judge.score(clip, request["prompt"])}
    identity = dict(version=1, request_id=request.get('request_id'))
    try:
        if request.get('version') != 1:
            raise ValueError('unsupported judge protocol version')
        if request.get('command') == 'park':
            if str(device).startswith('cuda'):
                torch.cuda.synchronize(device)
            judge.to('cpu')
            import gc
            gc.collect()
            if str(device).startswith('cuda'):
                # Backend handle/stream maps own live allocator storage that
                # model.to('cpu'), gc, and empty_cache alone cannot release.
                clear_workspaces = getattr(torch._C, '_cuda_clearCublasWorkspaces', None)
                if not callable(clear_workspaces):
                    raise RuntimeError('CUDA judge parking requires torch._C._cuda_clearCublasWorkspaces')
                clear_workspaces()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                allocated, reserved = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
            else:
                allocated = reserved = 0
            return dict(identity, residency='cpu', allocated_bytes=allocated, reserved_bytes=reserved)
        if request.get('command') != 'score':
            raise ValueError('unknown judge command')
        judge.to(device)
        clip = torch.load(request['clip'], map_location='cpu', weights_only=True)
        result = judge.score(clip, request['prompt'])
        return dict(identity, score=result, residency=str(device))
    except Exception as exc:
        return dict(identity, error=f'{type(exc).__name__}: {exc}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True)
    parser.add_argument('--residency-protocol', action='store_true')
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        judge = load_video_reward('cpu' if args.residency_protocol else args.device)
    print(json.dumps({'ready': True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        with redirect_stdout(sys.stderr):
            result = dispatch(judge, request, args.device, args.residency_protocol)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
