"""Offline native H3 training; allocation-free validation precedes model loading."""
import argparse
from contextlib import nullcontext
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import runpy
import signal
import sys
import time

from scripts.prepare_minimax_h3 import load_config


def allocation_walltime_seconds(override=None, *, now=None):
    """Resolve a finite Slurm budget at worker startup, before preflight/loading."""
    if override is not None:
        if not math.isfinite(override) or override < 0:
            raise ValueError('walltime_seconds must be finite and nonnegative')
        return float(override)
    deadline = os.environ.get('SLURM_JOB_END_TIME')
    if deadline is None:
        return 0.
    try:
        remaining = float(deadline)-(time.time() if now is None else now)
    except ValueError as error:
        raise ValueError('malformed Slurm allocation deadline') from error
    if not math.isfinite(remaining) or remaining <= 0:
        raise ValueError('Slurm allocation deadline must be finite and in the future')
    return remaining


def configure_allocation_deadline(cfg, override=None, *, startup=None):
    """Keep an absolute startup deadline through preflight and model loading."""
    startup = time.time() if startup is None else startup
    if override is None and getattr(cfg, 'walltime_seconds', 0) > 0:
        override = cfg.walltime_seconds
    duration = allocation_walltime_seconds(override, now=startup)
    # User ConfigDicts may have declared walltime_seconds with integer zero.
    with cfg.ignore_type() if hasattr(cfg, 'ignore_type') else nullcontext():
        cfg.walltime_seconds = duration
        cfg.walltime_deadline = startup+duration if duration > 0 else None


def runtime_contract():
    from diffusionopsd.video.h3_conditioning import runtime_identity
    from torch.distributed.fsdp import fully_shard
    import pynvml
    if 'ignored_params' not in inspect.signature(fully_shard).parameters:
        raise RuntimeError('H3 requires FSDP2 fully_shard(ignored_params=...)')
    if not callable(pynvml.nvmlInit) or not callable(pynvml.nvmlDeviceGetComputeRunningProcesses):
        raise RuntimeError('H3 requires process-level NVML residency evidence')
    nvml_version = importlib.metadata.version('nvidia-ml-py')
    if nvml_version != '13.580.82':
        raise RuntimeError(f'H3 requires nvidia-ml-py==13.580.82, found {nvml_version}')
    return dict(runtime=runtime_identity(), fsdp_ignored_params=True,
                nvidia_ml_py=nvml_version, pynvml_path=pynvml.__file__)


def allocation(cfg, *, training=False):
    from diffusionopsd.video.h3_conditioning import validate_h3_topology
    from diffusionopsd.video.h3_fsdp import resolve_h3_timeouts
    resolve_h3_timeouts(cfg)
    topology = validate_h3_topology(cfg)
    for key in ('max_updates_this_job', 'walltime_seconds', 'checkpoint_publication_reserve_seconds'):
        value = float(getattr(cfg, key, 0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if cfg.num_frames != 124 or cfg.fps != 24 or any(v <= 0 or v % 32 for v in (cfg.height, cfg.width)):
        raise ValueError('H3 geometry requires 124 frames, 24 fps and positive canvas multiples of 32')
    n = len(cfg.policy_gpus) + len(cfg.target_gpus)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    tokens = [token.strip() for token in visible.split(',')] if visible is not None else [None] * n
    if len(tokens) != n or (visible is not None and len(set(tokens)) != n) or visible == '':
        raise ValueError(f'expected exactly P+R={n} scheduler-visible devices')
    scheduler = os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS')
    host_ids = []
    if scheduler:
        for token in scheduler.split(','):
            if '-' in token and all(part.isdigit() for part in token.split('-')):
                first, last = map(int, token.split('-'))
                host_ids.extend(str(index) for index in range(first, last+1))
            else:
                host_ids.append(token.strip())
        if len(host_ids) != n or len(set(host_ids)) != n:
            raise ValueError('scheduler physical GPU allocation does not match P+R')
    physical = list(tokens)
    mapping_source = 'CUDA_VISIBLE_DEVICES' if visible is not None else 'unknown'
    if host_ids and (visible is None or (tokens == [str(i) for i in range(n)] and set(tokens) != set(host_ids))):
        # Slurm cgroups may renumber host devices. Allocation order alone does
        # not prove the host-to-visible correspondence; retain both identities.
        physical, mapping_source = [None] * n, 'Slurm/CUDA correspondence unverified'
    elif host_ids and all(str(token).isdigit() for token in tokens + host_ids) and set(tokens) != set(host_ids):
        raise ValueError('scheduler physical and CUDA-visible GPU mappings disagree')
    for key, expected in (('LOCAL_WORLD_SIZE', n), ('WORLD_SIZE', n*cfg.num_nodes),
                          ('SLURM_JOB_NUM_NODES', cfg.num_nodes)):
        if key in os.environ and int(os.environ[key]) != expected:
            raise ValueError(f'{key} does not match H3 topology ({expected})')
    if training:
        import torch
        if torch.cuda.device_count() != n:
            raise ValueError(f'expected exactly P+R={n} CUDA-visible devices before model loading')
        if 'LOCAL_RANK' not in os.environ or not 0 <= int(os.environ['LOCAL_RANK']) < n:
            raise ValueError('Launch training with torchrun and one process per owned device')
        rank = int(os.environ.get('RANK', -1))
        if not 0 <= rank < n*cfg.num_nodes or rank % n != int(os.environ['LOCAL_RANK']):
            raise ValueError('RANK and LOCAL_RANK must follow node-major H3 topology')
    node = int(os.environ.get('RANK', 0))//n
    return topology, [dict(rank=node*n+i, node_rank=node, local_rank=i, visible_device=i, physical_device=physical[i],
                           cuda_visible_token=tokens[i], mapping_source=mapping_source,
                           scheduler_allocation_tokens=host_ids,
                           role='policy' if i in cfg.policy_gpus else 'reward') for i in range(n)]


def preflight(cfg, *, training=False):
    from diffusionopsd.video.h3_conditioning import validate_h3_assets
    from diffusionopsd.video.h3_fsdp import resolve_h3_timeouts
    timeouts = resolve_h3_timeouts(cfg)
    topology, ranks = allocation(cfg, training=training)
    cache = Path(os.path.expandvars(str(cfg.conditioning_cache)))
    # Assets include exact tensor bytes, indexes, dataset/cache identities and all reward weights.
    report = validate_h3_assets(cfg, cache)
    return dict(report, topology=topology, rank_map=ranks, timeouts=timeouts, runtime_contract=runtime_contract(),
                site={key: os.environ.get(key) for key in
                      ('SLURM_CLUSTER_NAME', 'SLURM_JOB_ACCOUNT', 'SLURM_JOB_PARTITION', 'SLURM_JOB_GPUS')})


def main(argv=None):
    startup = time.time()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='config/minimax_h3_t2va.py')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--max-updates-this-job', type=int)
    parser.add_argument('--walltime-seconds', type=float)
    parser.add_argument('--checkpoint-publication-reserve-seconds', type=float)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    for name in ('max_updates_this_job', 'walltime_seconds', 'checkpoint_publication_reserve_seconds'):
        value = getattr(args, name)
        if value is not None:
            if not math.isfinite(value) or value < 0:
                parser.error(f'{name} must be finite and nonnegative')
            setattr(cfg, name, value)
    configure_allocation_deadline(cfg, args.walltime_seconds, startup=startup)
    report = preflight(cfg, training=not args.preflight)
    if args.preflight:
        print(json.dumps(report, indent=2, sort_keys=True))
        return report
    if args.run_dir is None:
        parser.error('--run-dir is required for training')
    from diffusionopsd.video.h3_training import run_h3_training
    return run_h3_training(cfg, run_dir=args.run_dir, resume=args.resume)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--launch']:
        # Slurm signals the whole srun job step. Elastic handles TERM/INT itself;
        # ignore USR1 only in the agent, until each worker installs its flag handler.
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        del sys.argv[1]
        if 'SLURM_PROCID' in os.environ and not any(arg.startswith(('--node_rank', '--node-rank')) for arg in sys.argv):
            sys.argv.insert(1, '--node_rank='+os.environ['SLURM_PROCID'])
        runpy.run_module('torch.distributed.run', run_name='__main__')
        raise SystemExit(0)
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'H3 training failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
