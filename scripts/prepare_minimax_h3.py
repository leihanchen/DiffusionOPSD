"""Prepare native H3 assets and conditioning before distributed GPU training."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def load_config(path):
    spec = importlib.util.spec_from_file_location('minimax_h3_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='config/minimax_h3_t2va.py')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prefetch', action='store_true', help='Download only t2va components and pinned dataset')
    mode.add_argument('--condition', action='store_true', help='Load encoder once, stream missing cache entries, exit')
    mode.add_argument('--check-only', action='store_true', help='Offline validation, without model allocation')
    parser.add_argument('--device', default='cpu', help='Encoder device for --condition only')
    parser.add_argument('--cache-root', type=Path)
    args = parser.parse_args()
    from diffusionopsd.video.h3_conditioning import (
        prefetch_h3_assets, prepare_conditioning, validate_h3_assets, validate_h3_topology, _atomic_write, _file_hash,
    )
    cfg = load_config(args.config)
    validate_h3_topology(cfg)
    cache = args.cache_root or Path(os.path.expandvars(cfg.conditioning_cache))
    if '$' in str(cache):
        parser.error('Set VIDEO_REWARD_CKPT_PATH or supply --cache-root')
    try:
        if args.prefetch:
            result = {'model_root': str(prefetch_h3_assets(cfg))}
        elif args.condition:
            result = {'manifest': str(prepare_conditioning(cfg, device=args.device, cache_root=cache))}
        else:
            result = validate_h3_assets(cfg, cache)
            inventory = result.pop('weight_inventory')
            audit = cache / 'preflight-weight-inventory.json'
            _atomic_write(audit, lambda stream: stream.write(json.dumps(inventory, indent=2, sort_keys=True).encode()))
            result['weight_inventory_audit'] = dict(path=str(audit), sha256=_file_hash(audit),
                components={name: dict(tensor_count=value['tensor_count'], tensor_bytes=value['tensor_bytes'])
                            for name, value in inventory.items()})
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'H3 preparation failed: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
