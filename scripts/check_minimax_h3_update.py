"""Runtime probe and explicit opt-in tiny FSDP qualification entry point."""
import argparse
import inspect
import json
import os
from pathlib import Path
import time

REVISION = '0919fb684102ef115126612f3fbb84d1027e7317'


def main(argv=None):
    startup = time.time()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--runtime-only', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--tiny-gpu', action='store_true')
    parser.add_argument('--profile-targets', metavar='LATENTS_PT',
        help='Profile actual saved unpacked video latents on the reward GPU; no synthetic full-size claims')
    parser.add_argument('--vae-tiling', choices=('tiled', 'untiled', 'both'), default='both')
    parser.add_argument('--profile-run-dir', default='logs/minimax-h3-target-profile')
    parser.add_argument('--profile-prompt', default='')
    parser.add_argument('--config')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--phase', choices=('load', 'rollout', 'target', 'update', 'resume'))
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(argv)
    if args.phase:
        if not args.config or args.run_dir is None:
            parser.error('--phase requires --config and --run-dir')
        if args.runtime_only or args.tiny_gpu or args.profile_targets or args.download:
            parser.error('--phase cannot be combined with legacy probe/profile modes')
        if (args.phase == 'resume') != (args.resume is not None):
            parser.error('--resume is required exactly for --phase resume')
        from scripts.prepare_minimax_h3 import load_config
        from scripts.train_opsd_video_minimax_h3 import preflight, configure_allocation_deadline
        from diffusionopsd.video.h3_training import run_h3_training
        from diffusionopsd.video.h3_verification import H3Verification
        cfg = load_config(args.config)
        configure_allocation_deadline(cfg, startup=startup)
        cfg.save_videos = True
        preflight(cfg, training=True)
        result = run_h3_training(cfg, run_dir=args.run_dir, resume=args.resume,
                                 verification=H3Verification(args.phase))
        if result.get('status') != 'stage_measured' or result.get('reached_phase') != args.phase:
            raise RuntimeError('requested verification phase was not reached (no qualification claimed)')
        print(json.dumps(result), flush=True)
        return result
    if args.config or args.run_dir or args.resume:
        parser.error('--config/--run-dir/--resume require --phase')
    import diffusers
    import peft
    import torch
    import transformers
    from torch.distributed.fsdp import fully_shard
    from diffusers import MiniMaxH3Transformer3DModel
    assert MiniMaxH3Transformer3DModel._supports_gradient_checkpointing
    if 'ignored_params' not in inspect.signature(fully_shard).parameters:
        raise RuntimeError('installed FSDP2 lacks ignored_params')
    expected = {'diffusers': '0.42.1', 'transformers': '5.22.1', 'peft': '0.21.2'}
    actual = {'diffusers': diffusers.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__}
    if actual != expected or torch.__version__ != '2.14.0+cu130':
        raise RuntimeError(f'H3 runtime version mismatch: torch={torch.__version__}, packages={actual}')
    print(json.dumps(dict(torch=torch.__version__, diffusers=diffusers.__version__,
        transformers=transformers.__version__, peft=peft.__version__,
        ignored_params=True, cuda=torch.cuda.is_available())), flush=True)
    if args.download:
        from config.minimax_h3_t2va import get_config
        from diffusionopsd.video.h3_conditioning import prefetch_h3_assets
        prefetch_h3_assets(get_config())
    if args.profile_targets:
        from config.minimax_h3_t2va import get_config
        from diffusionopsd.video.h3_auxiliary import H3RewardClient, H3RewardService
        from diffusionopsd.video.h3_fsdp import H3Devices
        if not torch.cuda.is_available():
            raise RuntimeError('target profiling requires a CUDA GPU and local full reward/VAE checkpoints')
        devices = H3Devices(0, 'reward', 0, (0,), torch.device('cuda', 0))
        service = H3RewardService.from_config(get_config(), devices, args.profile_run_dir)
        payload = torch.load(args.profile_targets, map_location='cpu', weights_only=True)
        if isinstance(payload, torch.Tensor):
            payload = {'video': payload}
        modes = ('tiled', 'untiled') if args.vae_tiling == 'both' else (args.vae_tiling,)
        results = []
        try:
            for mode in modes:
                if mode == 'tiled':
                    service.vae.enable_tiling()
                else:
                    service.vae.disable_tiling()
                client = H3RewardClient(service=service)
                scores = client.request('score_endpoint', mode, payload, {'prompt': args.profile_prompt})
                targets = client.request('construct_targets', mode, {'video': payload['video']}, {'accepted': True})
                results.append(dict(mode=mode, geometry=list(payload['video'].shape),
                    score_metrics=scores['metrics'], target_metrics=targets['metrics'],
                    diagnostics=targets['diagnostics']))
            output = Path(args.profile_run_dir) / 'target-profile.json'
            output.write_text(json.dumps(results, indent=2) + '\n')
            print(str(output), flush=True)
        finally:
            if service.judge:
                service.judge.close()
    if args.tiny_gpu:
        import pytest
        os.environ['H3_RUN_GPU_TESTS'] = '1'
        raise SystemExit(pytest.main(['tests/video/test_h3_fsdp.py', '-q', '-m', 'h3_gpu',
            '-o', 'markers=h3_gpu: opt-in torchrun CUDA test']))


if __name__ == '__main__':
    main()
