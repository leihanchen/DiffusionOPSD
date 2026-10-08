import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from diffusionopsd.video import inference_ablation as ab


class Pipe:
    def __init__(self, fail_at=None):
        self.transformer = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        self.calls = []
        self.fail_at = fail_at

    def encode_prompt(self, *args, **kwargs):
        assert kwargs['negative_prompt'] == ''
        return torch.ones(1, 2), torch.zeros(1, 2)

    def __call__(self, **kwargs):
        assert not torch.is_grad_enabled()
        assert not self.scheduler.used
        self.scheduler.used = True
        if len(self.calls) == self.fail_at:
            raise RuntimeError('generation failed')
        self.calls.append({**kwargs, 'latents': kwargs['latents'].clone()})
        steps = kwargs['num_inference_steps']
        self.scheduler.timesteps = torch.linspace(1000, 1, steps)
        self.scheduler.sigmas = torch.linspace(1, 0, steps + 1)
        # Mutation must not affect the noise cached for the next condition.
        return (kwargs['latents'].add_(0.01),)


class Roll:
    def __init__(self, *args):
        pass

    def decode01(self, latent):
        return latent.sigmoid()


class Reward:
    def __call__(self, clip):
        return SimpleNamespace(**{key: clip.mean() for key in ab.SCORES[:5]})


class Judge:
    def score(self, clip, prompt):
        return {key: float(clip.mean()) for key in ab.SCORES[5:]}


def fixture_config(tmp_path, monkeypatch):
    real = json.loads((Path(__file__).resolve().parents[2] / 'config/wan22_inference_ablation.json').read_text())
    ab.validate_experiment(real)
    experiment = copy.deepcopy(real)
    experiment['num_frames'] = 5
    for condition, params in experiment['conditions'].items():
        params.update(height=32, width=32 if condition == 'resolution480' else 64)
    monkeypatch.setattr(ab, 'WanRollout', Roll)
    monkeypatch.setattr(ab, 'latent_shape_from_pipe', lambda p, t, h, w: (1, t, 3, h, w))
    cfg = SimpleNamespace(policy_device='cpu', vae_device='cpu', reward_device='cpu')
    return cfg, experiment


def test_six_videos_paired_noise_fresh_scheduler_and_offline_wandb(tmp_path, monkeypatch):
    import wandb

    cfg, experiment = fixture_config(tmp_path, monkeypatch)
    pipe = Pipe()
    before = copy.deepcopy(pipe.transformer.state_dict())
    run = wandb.init(project='diffusionopsd-test', mode='offline', dir=str(tmp_path))
    try:
        summary = ab.run_ablation(pipe, Reward(), Judge(), cfg, experiment, 'chessboard', tmp_path,
                                  lambda: SimpleNamespace(used=False), run)
        assert run.settings.mode == 'offline'
        assert run.summary['ablation/differences/steps30/VQ'] == 0.0
    finally:
        run.finish()
    assert summary['num_videos'] == 6
    rows = [json.loads(line) for line in (tmp_path / 'samples.jsonl').read_text().splitlines()]
    for i in [0, 3]:
        assert rows[i]['initial_latent_hash'] == rows[i+1]['initial_latent_hash']
        assert rows[i]['initial_latents'] == rows[i+1]['initial_latents']
        assert rows[i]['initial_latent_hash'] != rows[i+2]['initial_latent_hash']
        assert torch.equal(pipe.calls[i]['latents'], pipe.calls[i+1]['latents'])
    assert [x['num_inference_steps'] for x in pipe.calls] == [50, 30, 50] * 2
    assert len(list(tmp_path.glob('*.mp4'))) == 6
    assert len(list(tmp_path.glob('*.png'))) == 18
    assert len(list(tmp_path.glob('*_initial.pt'))) == 4
    assert len(list(tmp_path.glob('wandb/offline-run-*/files/media/videos/**/*.mp4'))) == 6
    assert all(torch.equal(v, pipe.transformer.state_dict()[k]) for k, v in before.items())
    assert all(p.grad is None for p in pipe.transformer.parameters())
    for condition, steps in [('control', 50), ('steps30', 30), ('resolution480', 50)]:
        trace = json.loads((tmp_path / f'seed_0_{condition}.json').read_text())['trace']
        assert len(trace['timesteps']) == steps


def test_generation_failure_retains_completed_media(tmp_path, monkeypatch):
    cfg, experiment = fixture_config(tmp_path, monkeypatch)
    run = SimpleNamespace(log=lambda *a, **k: None, summary={})
    with pytest.raises(RuntimeError, match='generation failed'):
        ab.run_ablation(Pipe(fail_at=1), Reward(), Judge(), cfg, experiment, 'chessboard', tmp_path,
                        lambda: SimpleNamespace(used=False), run)
    failure = json.loads((tmp_path / 'failure.json').read_text())
    assert failure['completed'] == 1 and failure['condition'] == 'steps30'
    assert len(list(tmp_path.glob('*.mp4'))) == 1
    assert not (tmp_path / 'summary.json').exists()


def test_summary_and_incomplete_or_duplicate_rejection():
    rows = [dict(seed=seed, condition=name, **{metric: value+seed for metric in ab.SCORES})
            for seed in [0, 1000] for name, value in zip(ab.CONDITIONS, [3., 2., 1.])]
    summary = ab.summarize(rows, [0, 1000])
    assert summary['differences']['steps30']['VQ'] == -1
    assert summary['differences']['resolution480']['VQ'] == -2
    for bad in [rows[:-1], rows + rows[:1]]:
        with pytest.raises(ValueError):
            ab.summarize(bad, [0, 1000])


def test_rejects_adapter_and_changed_experiment(tmp_path, monkeypatch):
    cfg, experiment = fixture_config(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='agreed'):
        ab.validate_experiment(experiment)
    pipe = Pipe()
    pipe.transformer.peft_config = {'default': {}}
    with pytest.raises(ValueError, match='without LoRA'):
        ab.run_ablation(pipe, Reward(), Judge(), cfg, experiment, 'chessboard', tmp_path, None, None)
