"""Exercise the actual CLI loop with tiny CPU policy/reward boundaries."""
from contextlib import contextmanager
import copy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from diffusionopsd.curriculum.records import PromptRecord


@pytest.fixture
def trainer(monkeypatch, tmp_path):
    import diffusers
    import scripts.train_opsd_video_cosmos as m
    from config.cosmos_nano import get_config

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.ParameterDict()
            self.active = 'default'

        def add_adapter(self, config, adapter_name):
            self.weights[adapter_name] = torch.nn.Parameter(torch.tensor(.1))

        def set_adapter(self, name):
            self.active = name
            for k, p in self.weights.items():
                p.requires_grad_(k == name)

        def enable_gradient_checkpointing(self):
            pass

        def disable_adapters(self):
            pass

        def enable_adapters(self):
            pass

    class Pipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(transformer=Policy(), vae=torch.nn.Identity(), scheduler=SimpleNamespace(config={}))

    class Rollout:
        def __init__(self, pipe, *args, **kw):
            self.pipe = pipe

        def rollout(self, prompt, **kw):
            # Five video frames in [B,T,C,H,W].
            y = torch.full((1, 5, 3, 2, 2), .5)
            return SimpleNamespace(y0=y, x0=y)

        def decode01(self, x):
            return x

        def prediction(self, rec, transformer=None):
            return rec.y0 + (transformer or self.pipe.transformer).weights['default']

    class Reward(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()

        def forward(self, x):
            score = -(x - 1).square().flatten(1).mean(1)
            return SimpleNamespace(geo=score, s_id=torch.tensor([1.]), motion=torch.tensor([1.]))

    class Judge:
        def __init__(self, *args):
            pass

        def score(self, *args):
            return {'quality': 1.}

        def close(self):
            pass

    class Logger:
        experiment_group = 'test'

        def __init__(self, *args, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def log_event(self, *args):
            pass

        log_train = log_eval = log_event

    class Replica:
        def __init__(self, policy, **kwargs):
            self.model = copy.deepcopy(policy).requires_grad_(False)

        def select(self, name):
            self.model.set_adapter(name)
            self.model.requires_grad_(False)

        @contextmanager
        def active_on(self, device):
            yield self.model

        def sync(self, policy):
            self.model.load_state_dict(policy.state_dict())
            self.model.requires_grad_(False)

    monkeypatch.setattr(m, 'InferenceReplica', Replica)
    cfg = get_config()
    cfg.dataset.source = 'files'
    cfg.inference_device = 'cpu'
    cfg.policy_device = cfg.vae_device = cfg.reward_device = cfg.judge_device = 'cpu'
    cfg.num_frames = cfg.reward_frames = 5
    cfg.reward_height = cfg.reward_width = 2
    cfg.eval_count = cfg.calibration_prompts = 1
    cfg.save_videos = cfg.offload_saved_tensors = False
    cfg.tau_id = .5
    cfg.tau_motion = 0.
    cfg.train_prompts = str(tmp_path / 'legacy.txt')
    cfg.eval_prompts = str(tmp_path / 'eval.txt')
    (tmp_path / 'legacy.txt').write_text('A static statue.\nAnother static statue.\n')
    (tmp_path / 'eval.txt').write_text('An independent static scene.\n')
    monkeypatch.setattr(m, 'read_config', lambda _: cfg)
    monkeypatch.setattr(m, 'configure_reward', lambda _: None)
    monkeypatch.setattr(m, 'preflight', lambda _: (tmp_path, {}))
    monkeypatch.setattr(m, 'validate_cuda_devices', lambda *a, **kw: None)
    monkeypatch.setattr(diffusers, 'Cosmos3OmniPipeline', Pipeline)
    monkeypatch.setattr(diffusers, 'UniPCMultistepScheduler', SimpleNamespace(from_config=lambda *a, **kw: None))
    monkeypatch.setattr(m, 'CosmosRollout', Rollout)
    monkeypatch.setattr(m, 'GeoReward', Reward)
    monkeypatch.setattr(m, 'IsolatedVideoJudge', Judge)
    monkeypatch.setattr(m, 'VideoTrainingLogger', Logger)
    for name in ('load_depth_anything3', 'load_waft', 'load_dinov2'):
        monkeypatch.setattr(m, name, lambda _: None)
    monkeypatch.setattr(m, 'get_peft_model_state_dict',
                        lambda p, adapter_name='default': {'w': p.weights[adapter_name].detach().clone()})

    def set_state(p, state, adapter_name='default'):
        with torch.no_grad():
            p.weights[adapter_name].copy_(state['w'])

    def ema(p, decay):
        with torch.no_grad():
            p.weights['old'].lerp_(p.weights['default'], 1-decay)

    monkeypatch.setattr(m, 'set_peft_model_state_dict', set_state)
    monkeypatch.setattr(m, 'ema_adapter_', ema)
    monkeypatch.setattr(m, 'quality_gap', lambda *a: 1.)
    monkeypatch.setattr(m, 'evaluation_metrics', lambda *a: {})
    return m, cfg


def manifest(path, round_index=0):
    rows = [PromptRecord(str(i), str(i), None, 'train', f'Static statue {i}; camera orbits.',
                         'orbit', {}, 'initial' if round_index == 0 else 'new', round_index)
            for i in range(2)]
    path.write_text(json.dumps([asdict(r) for r in rows]))


def run(monkeypatch, m, run_dir, path=None, extra=()):
    argv = ['train', '--run-dir', str(run_dir)]
    if path:
        dev = path.parent / 'development.json'
        if not dev.exists():
            record = PromptRecord('dev', 'dev', None, 'development', 'Independent rigid scene',
                                  'dolly', {}, 'initial', 0)
            dev.write_text(json.dumps([asdict(record)]))
        argv += ['--curriculum-manifest', str(path), '--curriculum-round', '0',
                 '--development-manifest', str(dev), '--development-count', '1']
    monkeypatch.setattr('sys.argv', argv + list(extra))
    m.main()
    return torch.load(run_dir / 'training_state.pt', weights_only=False)


def test_actual_loop_rejected_round_commits_all_visits(trainer, monkeypatch, tmp_path):
    m, cfg = trainer
    cfg.tau_id = 2.
    path = tmp_path / 'train.json'
    manifest(path)
    state = run(monkeypatch, m, tmp_path / 'run', path)
    assert state['update'] == 0
    assert state['extra']['round_stats'] == {'attempted_clips': 8, 'successful_updates': 0, 'prompt_visits': 2}
    assert state['extra']['round_cursor']['next_index'] == 2
    resumed = run(monkeypatch, m, tmp_path / 'run', path, ['--resume'])
    assert resumed['extra']['round_stats'] == state['extra']['round_stats']


def test_actual_loop_resume_matches_uninterrupted(trainer, monkeypatch, tmp_path):
    m, cfg = trainer
    path = tmp_path / 'train.json'
    manifest(path)
    full = run(monkeypatch, m, tmp_path / 'full', path)
    partial = run(monkeypatch, m, tmp_path / 'resumed', path, ['--max-updates-this-job', '1'])
    assert partial['update'] == 1
    resumed = run(monkeypatch, m, tmp_path / 'resumed', path, ['--resume'])
    assert full['update'] == resumed['update'] == 2
    for name in ('default', 'old'):
        assert torch.equal(full['adapters'][name]['w'], resumed['adapters'][name]['w'])
    rows = json.loads(path.read_text())
    rows[0]['text'] = 'Changed scene'
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='configuration'):
        run(monkeypatch, m, tmp_path / 'resumed', path, ['--resume'])


def test_actual_legacy_loop_still_obeys_update_budget(trainer, monkeypatch, tmp_path):
    m, cfg = trainer
    cfg.updates = 1
    state = run(monkeypatch, m, tmp_path / 'legacy')
    assert state['update'] == 1
    assert 'prompt_cursor' in state['extra']


def test_actual_round_transition_preserves_state(trainer, monkeypatch, tmp_path):
    m, cfg = trainer
    path = tmp_path / 'train.json'
    manifest(path)
    old = run(monkeypatch, m, tmp_path / 'round0', path)
    rows = json.loads(path.read_text())
    for i, row in enumerate(rows):
        row.update(prompt_id=f'new{i}', lineage_id=f'new{i}', text=f'New rigid scene {i}',
                   source='new', round_index=1)
    path.write_text(json.dumps(rows))
    new = run(monkeypatch, m, tmp_path / 'round1', path,
              ['--curriculum-round', '1', '--previous-round-checkpoint', str(tmp_path / 'round0/training_state.pt')])
    assert new['update'] == 4
    assert new['extra']['round_stats']['successful_updates'] == 2
    assert new['extra']['thresholds'] == old['extra']['thresholds']
    assert set(old['extra']['references']) <= set(new['extra']['references'])
    assert new['optimizer']['state'][0]['step'] == 4


def test_round_feedback_has_all_attempts_and_fixed_development(trainer, monkeypatch, tmp_path):
    m, cfg = trainer
    cfg.tau_id = 2.
    path = tmp_path / 'train.json'
    manifest(path)
    run_dir = tmp_path / 'run'
    state = run(monkeypatch, m, run_dir, path)
    assert len(state['extra']['training_rows']) == 8
    assert len(state['extra']['development_baseline']) == 2
    assert len(state['extra']['development_rows']) == 2
    costs = json.loads((run_dir / 'costs.json').read_text())
    assert costs['training_clips'] == 8
    assert costs['development_clips'] == 4
    resumed = run(monkeypatch, m, run_dir, path, ['--resume'])
    assert resumed['extra']['costs'] == state['extra']['costs']
    dev_path = tmp_path / 'development.json'
    dev = json.loads(dev_path.read_text())
    dev[0]['text'] = 'Changed diagnostic prompt'
    dev_path.write_text(json.dumps(dev))
    with pytest.raises(ValueError, match='configuration'):
        run(monkeypatch, m, run_dir, path, ['--resume'])


def test_nonfinite_rejected_feedback_exports_as_unknown(trainer, monkeypatch, tmp_path):
    m, cfg = trainer
    cfg.tau_id = 2.

    class MissingReward(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()

        def forward(self, x):
            return SimpleNamespace(geo=torch.tensor([float('nan')]), s_id=torch.tensor([1.]),
                                   motion=torch.tensor([1.]))

    monkeypatch.setattr(m, 'GeoReward', MissingReward)
    path = tmp_path / 'train.json'
    manifest(path)
    run_dir = tmp_path / 'run'
    run(monkeypatch, m, run_dir, path)
    report = json.loads((run_dir / 'round_feedback.json').read_text())
    assert all(r['mean_geo'] is None for r in report['prompts'].values())
    assert json.loads((run_dir / 'training_feedback.json').read_text())[0]['geo'] is None


def test_curriculum_does_not_load_default_hf_training_data(trainer, monkeypatch, tmp_path):
    m, cfg = trainer
    cfg.dataset.source = 'hf'
    cfg.tau_id = 2.

    def unwanted_dataset_load(*args, **kwargs):
        raise AssertionError('Curriculum manifests must not require the default HF dataset')

    monkeypatch.setattr(m, 'load_training_data', unwanted_dataset_load)
    path = tmp_path / 'train.json'
    manifest(path)
    state = run(monkeypatch, m, tmp_path / 'run', path)
    assert state['extra']['round_cursor']['next_index'] == 2


def test_legacy_hf_dataset_keeps_identity_metadata(trainer, monkeypatch, tmp_path):
    from diffusionopsd.video.training_data import TrainingData

    m, cfg = trainer
    cfg.updates = 1
    data = TrainingData(train=[{'text': 'Pinned training scene', 'sample_id': 'sample-1'}],
                        evaluation=[{'text': 'Pinned independent scene', 'sample_id': 'eval-1'}],
                        metadata={'seed': 42, 'revision': 'a' * 40},
                        train_hash='train-content-hash', eval_hash='eval-content-hash')
    monkeypatch.setattr(m, 'load_training_data', lambda *args, **kw: data)
    state = run(monkeypatch, m, tmp_path / 'hf-run')
    assert state['update'] == 1
    assert state['config']['dataset_identity']['revision'] == 'a' * 40
    assert state['config']['train_prompt_sha256'] == 'train-content-hash'
    assert state['config']['eval_prompt_sha256'] == 'eval-content-hash'


def enable_probes(monkeypatch, m, cfg):
    import diffusionopsd.video.curriculum_probe as probes
    cfg.paired_probe_count = 2
    monkeypatch.setattr(probes, 'get_peft_model_state_dict', m.get_peft_model_state_dict)
    monkeypatch.setattr(probes, 'set_peft_model_state_dict', m.set_peft_model_state_dict)
    cls = m.CosmosRollout
    original = cls.rollout
    def rollout(self, *args, **kwargs):
        rec = original(self, *args, **kwargs)
        rec.sigma_q, rec.q_index = .5, 1
        rec.timesteps = torch.arange(3)
        rec.base = rec.y0.clone()
        model = self.pipe.transformer
        rec.y0 = rec.base + model.weights[model.active].detach().clone()
        rec.x0 = rec.y0.clone()
        return rec
    monkeypatch.setattr(cls, 'rollout', rollout)
    def prediction(self, rec, transformer=None):
        model = transformer or self.pipe.transformer
        return rec.base + model.weights[model.active]
    monkeypatch.setattr(cls, 'prediction', prediction)
    monkeypatch.setattr(cls, 'continue_from', lambda self, rec, y: y.clone(), raising=False)
    # Mirror the production Diffusers adapter API at the heavyweight model boundary.
    previous = m.InferenceReplica.__init__
    def initialize(self, policy, **kwargs):
        previous(self, policy, **kwargs)
        self.model.get_active_adapters = lambda: [self.model.active]
    monkeypatch.setattr(m.InferenceReplica, '__init__', initialize)


def test_round_probes_capture_actual_targets_and_resume_without_repeating(trainer, monkeypatch, tmp_path):
    m, cfg = trainer
    enable_probes(monkeypatch, m, cfg)
    path = tmp_path / 'train.json'
    manifest(path)
    full = run(monkeypatch, m, tmp_path / 'full', path)
    report = json.loads((tmp_path / 'full/round_probe.json').read_text())
    assert report['done'] and report['selected_count'] == report['audited_count'] == 2
    assert all(q['baseline_parity']['passed'] for q in report['queries'])
    assert all(q['final_gains']['g_plus'] > 0 for q in report['queries'])
    # The native-consistent fixture fits .6 toward .66, closer to optimum1.
    assert all(q['final_gains']['delta_round'] > 0 for q in report['queries'])
    assert all(q['metadata']['original_behavior_update'] < full['update'] for q in report['queries'])
    interrupted = run(monkeypatch, m, tmp_path / 'resumed', path, ['--max-updates-this-job', '1'])
    assert not interrupted['extra']['round_probe']['done']
    resumed = run(monkeypatch, m, tmp_path / 'resumed', path, ['--resume'])
    again = run(monkeypatch, m, tmp_path / 'resumed', path, ['--resume'])
    assert again['extra']['round_probe']['results'] == resumed['extra']['round_probe']['results']
    assert full['extra']['round_probe']['results'] == resumed['extra']['round_probe']['results']
    for name in ['default', 'old']:
        assert torch.equal(full['adapters'][name]['w'], resumed['adapters'][name]['w'])


def test_round_probes_record_missing_identity_rejected_queries(trainer, monkeypatch, tmp_path):
    m, cfg = trainer
    enable_probes(monkeypatch, m, cfg)
    cfg.tau_id = 2.
    path = tmp_path / 'train.json'
    manifest(path)
    run(monkeypatch, m, tmp_path / 'rejected', path)
    report = json.loads((tmp_path / 'rejected/round_probe.json').read_text())
    assert report['done'] and report['selected_count'] == 2 and report['audited_count'] == 0
    assert len(report['missing_prompt_ids']) == 2


def test_probe_resume_preserves_completed_query_after_diagnostic_failure(trainer, monkeypatch, tmp_path):
    m, cfg = trainer
    enable_probes(monkeypatch, m, cfg)
    path = tmp_path / 'train.json'
    manifest(path)
    audit = m.audit_query
    calls = []
    def interrupted(entry, **kwargs):
        calls.append(entry['metadata']['prompt_id'])
        if len(calls) == 2:
            raise RuntimeError('injected diagnostic failure')
        return audit(entry, **kwargs)
    monkeypatch.setattr(m, 'audit_query', interrupted)
    with pytest.raises(RuntimeError, match='injected diagnostic'):
        run(monkeypatch, m, tmp_path / 'run', path)
    state = torch.load(tmp_path / 'run/training_state.pt', weights_only=False)
    assert state['update'] == 2 and len(state['extra']['round_probe']['results']) == 1
    first = state['extra']['round_probe']['results'][0]
    monkeypatch.setattr(m, 'audit_query', audit)
    resumed = run(monkeypatch, m, tmp_path / 'run', path, ['--resume'])
    assert resumed['update'] == 2
    assert len(resumed['extra']['round_probe']['results']) == 2
    assert resumed['extra']['round_probe']['results'][0] == first


def test_completed_round_can_recompute_probes_without_training_again(trainer, monkeypatch, tmp_path):
    m, cfg = trainer
    enable_probes(monkeypatch, m, cfg)
    path = tmp_path / 'train.json'
    manifest(path)
    original = run(monkeypatch, m, tmp_path / 'run', path)
    audit = m.audit_query
    calls = []
    def observed(entry, **kwargs):
        calls.append(entry['metadata']['prompt_id'])
        return audit(entry, **kwargs)
    monkeypatch.setattr(m, 'audit_query', observed)
    repeated = run(monkeypatch, m, tmp_path / 'run', path, ['--resume', '--recompute-probes'])
    assert len(calls) == 2
    assert repeated['update'] == original['update'] == 2
    assert repeated['extra']['round_stats'] == original['extra']['round_stats']
    assert repeated['extra']['round_probe']['results'] == original['extra']['round_probe']['results']
    assert all(torch.equal(v, original['adapters']['default'][k]) for k,v in repeated['adapters']['default'].items())


def test_recompute_probes_rejects_an_unfinished_training_round(trainer, monkeypatch, tmp_path):
    m, cfg = trainer
    enable_probes(monkeypatch, m, cfg)
    path = tmp_path / 'train.json'
    manifest(path)
    run(monkeypatch, m, tmp_path / 'run', path, ['--max-updates-this-job', '1'])
    with pytest.raises(ValueError, match='completed training'):
        run(monkeypatch, m, tmp_path / 'run', path, ['--resume', '--recompute-probes'])


def test_optional_probes_preserve_training_state_and_rng(trainer, monkeypatch, tmp_path):
    """Extra diagnostic reward calls must leave training and its RNG intact."""
    m, cfg = trainer
    enable_probes(monkeypatch, m, cfg)
    base_reward = m.GeoReward
    class RandomConsumingReward(base_reward):
        def forward(self, video):
            torch.rand(())
            return super().forward(video)
    monkeypatch.setattr(m, 'GeoReward', RandomConsumingReward)
    path = tmp_path / 'train.json'
    manifest(path)
    observed = run(monkeypatch, m, tmp_path / 'observed', path)
    cfg.paired_probe_count = 0
    plain = run(monkeypatch, m, tmp_path / 'plain', path)
    assert observed['update'] == plain['update'] == 2
    assert observed['extra']['round_stats'] == plain['extra']['round_stats']
    for name in ['default', 'old']:
        assert torch.equal(observed['adapters'][name]['w'], plain['adapters'][name]['w'])
    assert torch.equal(observed['rng']['torch'], plain['rng']['torch'])
    for param_id, state in observed['optimizer']['state'].items():
        for key, value in state.items():
            assert torch.equal(value, plain['optimizer']['state'][param_id][key])
