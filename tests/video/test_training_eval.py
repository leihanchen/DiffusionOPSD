import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusionopsd.video.training_eval import EvaluationMonitor, evaluation_due


def config(tmp_path, **kwargs):
    path = tmp_path / 'prompts.txt'
    path.write_text('first prompt\nsecond prompt\n')
    values = {'eval_freq': 2, 'eval_num_prompts': 2, 'eval_seed': 10, 'eval_num_videos': 0,
              'eval_video_fps': 8, 'eval_prompts': str(path), 'logdir': str(tmp_path), 'use_lora': True,
              'reward_device': 'cpu', 'judge': SimpleNamespace(device='cpu')}
    values.update(kwargs)
    return SimpleNamespace(**values)


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.active_adapters = ['old']
        self.child = torch.nn.Dropout()
        self.child.eval()

    def set_adapter(self, name):
        self.active_adapters = [name] if isinstance(name, str) else name
        self.weight.requires_grad_(False)


class Reward(torch.nn.Module):
    def forward(self, clip):
        value = clip.mean()
        return SimpleNamespace(geo=value, s_id=value, motion=value)


class Judge:
    def score(self, clip, prompt):
        return {'VQ': float(clip.mean()), 'MQ': float(clip.mean()), 'TA': 0., 'Overall': 0.}


def test_schedule_includes_baseline_and_final_once():
    assert [s for s in range(6) if evaluation_due(s, 5, 2)] == [0, 2, 4, 5]
    assert [s for s in range(5) if evaluation_due(s, 4, 2)] == [0, 2, 4]
    assert not any(evaluation_due(s, 4, 0) for s in range(5))


def test_settings_validate_before_models_are_needed(tmp_path):
    with pytest.raises(ValueError):
        EvaluationMonitor(config(tmp_path, eval_num_prompts=0))
    with pytest.raises(ValueError):
        EvaluationMonitor(config(tmp_path, eval_num_videos=3))
    cfg = config(tmp_path)
    (tmp_path / 'prompts.txt').write_text('\n')
    with pytest.raises(ValueError):
        EvaluationMonitor(cfg)
    # Disabled evaluation does not require a prompt file.
    EvaluationMonitor(config(tmp_path, eval_freq=0, eval_prompts='/does/not/exist'))


@pytest.mark.parametrize('fail', [False, True])
def test_evaluation_preserves_training_state_and_rng(tmp_path, monkeypatch, fail):
    monitor = EvaluationMonitor(config(tmp_path))
    policy = Policy()
    original = torch.nn.Linear(1, 1)
    pipe = SimpleNamespace(transformer=original, vae=torch.nn.Identity(), text_encoder=torch.nn.Identity())
    policy.weight.grad = torch.tensor(7.)
    torch_state = torch.get_rng_state().clone()
    py_state = random.getstate()
    np_state = np.random.get_state()
    observed = []

    def generate(pipe, roll, cfg, prompt, seed, device):
        assert pipe.transformer is policy
        assert not policy.training and not torch.is_grad_enabled()
        assert policy.active_adapters == ['default']
        observed.append((prompt, seed))
        random.random()
        np.random.rand()
        torch.rand(1)
        if fail:
            raise RuntimeError('scorer failure')
        return policy.weight.detach().expand(1, 2, 3, 16, 16).clone()

    monkeypatch.setattr('diffusionopsd.video.training_eval.generate_clip', generate)
    if fail:
        with pytest.raises(RuntimeError, match='scorer failure'):
            monitor.evaluate(pipe, policy, None, Reward(), Judge(), 0)
        assert monitor.baseline is None
    else:
        metrics, videos = monitor.evaluate(pipe, policy, None, Reward(), Judge(), 0)
        assert metrics['eval/geo'] == 1.
        assert metrics['eval/p_q'] == .5
        assert not videos
        with torch.no_grad():
            policy.weight.fill_(2.)
        metrics, _ = monitor.evaluate(pipe, policy, None, Reward(), Judge(), 2)
        assert metrics['eval/geo_delta'] == 1.
        assert metrics['eval/p_q'] == pytest.approx(torch.sigmoid(torch.tensor(1.)).item())
        assert observed == [('first prompt', 10), ('second prompt', 11)] * 2
        assert (tmp_path / 'eval' / 'update_000002.json').exists()
    assert pipe.transformer is original
    assert policy.training and not policy.child.training
    assert policy.active_adapters == ['old']
    assert policy.weight.requires_grad
    assert policy.weight.grad.item() == 7.
    assert torch.equal(torch_state, torch.get_rng_state())
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])


def test_full_policy_evaluation_and_video_files(tmp_path, monkeypatch):
    import imageio.v2 as imageio

    monitor = EvaluationMonitor(config(tmp_path, use_lora=False, eval_num_videos=2))
    policy = torch.nn.Linear(1, 1)
    pipe = SimpleNamespace(transformer=policy, vae=torch.nn.Identity(), text_encoder=torch.nn.Identity())
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    monkeypatch.setattr('diffusionopsd.video.training_eval.generate_clip',
                        lambda *args: torch.full((1, 2, 3, 16, 16), .5))
    metrics, videos = monitor.evaluate(pipe, policy, None, Reward(), Judge(), 0)
    assert metrics['eval/num_prompts'] == 2
    assert len(videos) == 2
    for video in videos:
        reader = imageio.get_reader(video['path'])
        try:
            assert reader.get_data(0).shape == (16, 16, 3)
        finally:
            reader.close()
    assert policy.training
    assert all(torch.equal(before[k], v) for k, v in policy.state_dict().items())
    assert all(p.grad is None for p in policy.parameters())


def test_training_continues_and_logs_offline_metrics_and_media(tmp_path, monkeypatch):
    import json

    from diffusionopsd.metrics import install_wandb_jsonl_tee

    import wandb
    from scripts.train_opsd_video_wan import _log_record

    monitor = EvaluationMonitor(config(tmp_path, use_lora=False, eval_num_videos=2))
    policy = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.weight.fill_(.1)
    pipe = SimpleNamespace(transformer=policy, vae=torch.nn.Identity(), text_encoder=torch.nn.Identity())
    def generate(pipe, roll, cfg, prompt, seed, device):
        value = policy.weight.detach() + (.05 if prompt == 'second prompt' else 0.)
        return value.expand(1, 2, 3, 16, 16).clone()

    monkeypatch.setattr('diffusionopsd.video.training_eval.generate_clip', generate)
    monkeypatch.setattr(wandb, 'log', wandb.log)
    run = wandb.init(project='diffusionopsd-test', dir=str(tmp_path), mode='offline')
    try:
        install_wandb_jsonl_tee(wandb, tmp_path / 'metrics.jsonl')
        optimizer = torch.optim.SGD(policy.parameters(), lr=.1)
        for update in range(5):
            if update:
                assert policy.training
                optimizer.zero_grad()
                ((policy.weight - .9) ** 2).sum().backward()
                optimizer.step()
            if evaluation_due(update, 4, 2):
                metrics, videos = monitor.evaluate(pipe, policy, None, Reward(), Judge(), update)
            else:
                metrics, videos = {}, []
            record = {'optimizer_updates': update, **metrics}
            if update:
                record['train/rl_distillation_loss'] = float(((policy.weight.detach() - .9) ** 2).sum())
            _log_record(record, videos)
        assert policy.weight.item() > .1
        assert run.settings.mode == 'offline'
    finally:
        wandb.finish()
    rows = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert [row['_step'] for row in rows] == [0, 1, 2, 3, 4]
    assert all('train/rl_distillation_loss' in row for row in rows[1:])
    assert ['eval/geo' in row for row in rows] == [True, False, True, False, True]
    assert rows[0]['eval/geo_delta'] == 0.
    assert rows[-1]['eval/geo_delta'] > 0.
    assert all('eval/videos' not in row for row in rows)
    assert len(list((tmp_path / 'eval').glob('*.mp4'))) == 6
    assert list((tmp_path / 'wandb').glob('offline-run-*/run-*.wandb'))
    assert len(list((tmp_path / 'wandb').glob('offline-run-*/files/media/videos/**/*.mp4'))) == 6
