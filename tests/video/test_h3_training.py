"""Global H3 candidate ownership, summed gradients and bounded failure tests."""
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import time
import sys
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusionopsd.video.branch_loss import branch_loss
from diffusionopsd.video.h3_auxiliary import H3Targets
from diffusionopsd.video.h3_fsdp import H3Devices
from diffusionopsd.video import h3_training


def test_phase_prints_original_error_before_completion_exchange(monkeypatch, capsys):
    """A stuck completion must not hide the original rank-local exception."""
    def ready(name, group, error=None):
        if name.endswith(':complete'):
            captured = capsys.readouterr().err
            rows = [json.loads(line) for line in captured.splitlines()]
            failure = next(row for row in rows if row['event'] == 'error')
            assert failure['phase'] == 'backward:0'
            assert failure['rank'] == 1
            assert failure['error'] == 'RuntimeError: original allocation failure'
            assert 'Traceback (most recent call last)' in failure['traceback']
            raise RuntimeError('completion exchange unavailable')

    monkeypatch.setattr(h3_training, '_ready', ready)
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'get_rank', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: pytest.fail('diagnostics synchronized CUDA'))

    def fail():
        raise RuntimeError('original allocation failure')

    with pytest.raises(RuntimeError, match='completion exchange unavailable'):
        h3_training._phase('backward:0', fail, object())


def test_phase_prints_operation_boundaries_without_changing_result(capsys):
    result = object()
    assert h3_training._phase('replay:1', lambda: result, None) is result
    rows = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [row['event'] for row in rows] == ['start', 'returned']
    assert all(row['phase'] == 'replay:1' for row in rows)
    assert all(row['rank'] == 0 and row['time_unix_s'] > 0 for row in rows)


@pytest.mark.parametrize('name,value', [
    (name, value) for name in ('collective_timeout_s', 'service_timeout_s')
    for value in (float('nan'), float('inf'), 0., -1.)
] + [('collective_timeout_s', 600.), ('collective_timeout_s', 599.),
     ('service_timeout_s', 3600.), ('service_timeout_s', 4000.)])
def test_runner_rejects_timeout_budgets_before_group_or_model_initialization(name, value, tmp_path, monkeypatch):
    cfg = SimpleNamespace(policy_gpus=(0, 1), target_gpus=(2,), **{name: value})
    monkeypatch.setattr(h3_training, 'init_h3_distributed', lambda **kw: pytest.fail('group initialization'))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: pytest.fail('CUDA touched'))
    with pytest.raises(ValueError, match='timeout'):
        h3_training.run_h3_training(cfg, run_dir=tmp_path)


def _delayed_phase_worker(rank, rendezvous, output, timeout, delayed_rank, fail):
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=20))
    group = dist.new_group([0, 1], backend='gloo', timeout=timedelta(seconds=timeout))
    started = time.monotonic()
    try:
        def work():
            if rank == delayed_rank:
                if fail:
                    raise RuntimeError('injected role failure')
                time.sleep(1.)
            return rank
        result = h3_training._phase('service_loading' if delayed_rank else 'behavior_rollout', work, group)
        peers = h3_training._exchange(result, group)
        outcome = dict(status='complete', result=result, peers=peers)
    except RuntimeError as error:
        outcome = dict(status='failed', error=str(error))
    finally:
        elapsed = time.monotonic()-started
        dist.destroy_process_group()
    Path(output, f'{rank}.json').write_text(json.dumps(dict(outcome, elapsed=elapsed)))


@pytest.mark.parametrize('timeout,delayed_rank,fail,expected', [
    (.25, 1, False, 'failed'),
    (5., 1, False, 'complete'),
    (5., 0, False, 'complete'),
    (5., 1, True, 'failed'),
])
def test_real_gloo_phase_bounds_delayed_or_failed_role(tmp_path, timeout, delayed_rank, fail, expected):
    context = mp.spawn(_delayed_phase_worker,
        args=(str(tmp_path/'rendezvous'), str(tmp_path), timeout, delayed_rank, fail),
        nprocs=2, join=False)
    deadline = time.monotonic()+45
    try:
        while not context.join(timeout=.2):
            if time.monotonic() >= deadline:
                pytest.fail('phase workers exceeded parent deadline')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    for rank in range(2):
        outcome = json.loads((tmp_path/f'{rank}.json').read_text())
        assert outcome['status'] == expected
        assert outcome['elapsed'] < 10
        if expected == 'complete':
            assert outcome['result'] == rank
            assert outcome['peers'] == [0, 1]
        elif fail:
            assert 'injected role failure' in outcome['error']
        else:
            assert any(word in outcome['error'].lower() for word in ('timed out', 'timeout', 'closed'))


def test_duplicate_file_reference_reuse_calibration_training_and_nodes(monkeypatch):
    devices = H3Devices.cpu()
    references, calls = {}, []
    rollout = SimpleNamespace(rollout=lambda *a, **kw: SimpleNamespace(video_endpoint=torch.ones(1)))
    def score(*args):
        calls.append(args)
        return dict(geo=[1.], s_id=[1.], motion=[1.], quality=dict(VQ=1., MQ=1., TA=1., Overall=1.))
    client = SimpleNamespace(request=score)
    def exchange(value, group):
        if isinstance(value, tuple):
            key, identity, node = value
            other = dict(identity)
            if 'sample_id' in other:
                other['sample_id'] = 'train:1:hash'
            return [value, (key, other, 1)]  # same caption assigned to two nodes
        return [value]
    monkeypatch.setattr(h3_training, '_exchange', exchange)
    request = dict(record_key='duplicate', record_source='file',
        record=dict(sample_id='train:0:hash', text='duplicate'), conditioning={}, geometry={})
    kwargs = dict(cfg=SimpleNamespace(seed=42), devices=devices, policy=True,
        rollout=rollout, client=client, manifest={'identity': {'pinned': 1}})
    first = h3_training._training_reference(request, references, **kwargs)
    second = h3_training._training_reference(dict(request, attempt=1,
        record=dict(sample_id='train:1:hash', text='duplicate')), references, **kwargs)
    assert first is second and len(calls) == 1
    assert list(references) == ['duplicate']
    # HF keys continue to isolate records with identical caption text.
    monkeypatch.setattr(h3_training, '_exchange', lambda value, group: [value])
    for index in range(2):
        hf = dict(request, record_source='hf', record_key=f'hf-{index}',
            record=dict(sample_id=f'hf-{index}', text='duplicate'))
        result = h3_training._training_reference(hf, references, **kwargs)
        assert result['identity']['sample_id'] == f'hf-{index}'
    with pytest.raises(RuntimeError, match='identity mismatch'):
        h3_training._training_reference(dict(hf, record=dict(sample_id='changed', text='duplicate')),
            references, **kwargs)


def test_runner_uses_explicit_collective_and_service_timeout_budgets(tmp_path, monkeypatch):
    test_runner_eight_global_evaluation_records_and_checkpoint_boundary(
        tmp_path, monkeypatch, total_updates=1, walltime_stop=False, checkpoint_interval=1,
        timeout_budget=(4200., 900.))


class TinyPolicy:
    def __init__(self, devices):
        self.devices = devices
        self.parameter = torch.nn.Parameter(torch.tensor([.2, -.3]))
        self.trainable_parameters = (self.parameter,)
        self.old = self.parameter.detach().clone()
        self.ema_calls = 0

    def ema_(self, decay):
        self.old.lerp_(self.parameter.detach(), 1-decay)
        self.ema_calls += 1


class TinyRollout:
    def __init__(self, policy, failure=None):
        self.policy, self.failure = policy, failure
        self.calls = 0
        self.generation_calls = 0
        self.adapters = []
        self.seeds = []

    def rollout(self, conditioning, *, seed, geometry, adapter='old'):
        self.generation_calls += 1
        self.adapters.append(adapter)
        self.seeds.append(seed)
        anchor = torch.tensor([[float(seed % 7), 1.]])
        return SimpleNamespace(y0=anchor, seed=seed, video_endpoint=anchor+10,
                               audio_endpoint=torch.ones(1, 2, 3),
                               video_schedule=torch.tensor([1., .6, 0.]),
                               audio_schedule=torch.tensor([1., .3, 0.]),
                               query_index=1, sigma_video=.6, sigma_audio=.3)

    def prediction(self, query):
        self.calls += 1
        if self.failure == 'forward':
            raise RuntimeError('injected forward failure')
        if self.failure == 'collective':
            dist.all_reduce(torch.ones(1), group=self.policy.devices.policy_group)
        return self.policy.parameter[None] * query.y0


def settings():
    return SimpleNamespace(beta=1., loss_scale=5., max_grad_norm=1., ema_decay=.99)


def test_historical_endpoint_weights_and_rejection():
    from diffusionopsd.stat_tracking import PerPromptStatTracker
    tracker = PerPromptStatTracker()
    cfg = SimpleNamespace(tau_id=.5, tau_motion=.1, tau_quality=.4, adv_clip_max=5.)
    rows = [dict(candidate_index=index, prompt='repeat', record_key='record-A', geo=float(index),
                 s_id=1., anchor_s_id=1., motion=1.,
                 quality=dict(VQ=0., MQ=0.), reference_quality=dict(VQ=0., MQ=0.)) for index in range(4)]
    masks, weights = h3_training.endpoint_decisions(rows, tracker, cfg)
    assert list(masks.values()) == [True] * 4
    assert weights[0] < weights[3]
    rows[0]['geo'] = 10.
    _, next_weights = h3_training.endpoint_decisions(rows[:1], tracker, cfg)
    assert next_weights[0] > .5
    assert len(tracker.stats['record-A']) == 5
    for row in rows:
        row['anchor_s_id'] = 0.
    rejected, _ = h3_training.endpoint_decisions(rows, tracker, cfg)
    assert not any(rejected.values())


def test_anchor_gate_excludes_rejected_extremes_from_present_and_future_history():
    from diffusionopsd.stat_tracking import PerPromptStatTracker
    cfg = SimpleNamespace(tau_id=.5, tau_motion=.1, tau_quality=.4, adv_clip_max=5.)
    def row(index, reward, anchor_identity, endpoint_identity):
        return dict(candidate_index=index, prompt='same caption', record_key='record-A', geo=reward,
            anchor_s_id=anchor_identity, s_id=endpoint_identity, motion=1.,
            quality=dict(VQ=0., MQ=0.), reference_quality=dict(VQ=0., MQ=0.))
    tracker, reference = PerPromptStatTracker(), PerPromptStatTracker()
    accepted = row(1, 2., 1., 0.)
    masks, weights = h3_training.endpoint_decisions([row(0, 1e9, 0., 1.), accepted], tracker, cfg)
    assert masks == {0: False, 1: True}
    assert weights == {1: .5}
    assert tracker.stats == {'record-A': [2.]}
    h3_training.endpoint_decisions([accepted], reference, cfg)
    rejected, rejected_weights = h3_training.endpoint_decisions([row(2, -1e9, 0., 1.)], tracker, cfg)
    assert rejected == {2: False} and rejected_weights == {}
    assert tracker.stats == {'record-A': [2.]}
    assert tracker.history_prompts == reference.history_prompts
    next_row = row(3, 4., 1., 0.)
    _, actual = h3_training.endpoint_decisions([next_row], tracker, cfg)
    _, expected = h3_training.endpoint_decisions([next_row], reference, cfg)
    assert actual == expected


def test_equal_caption_distinct_record_histories_remain_separate():
    from diffusionopsd.stat_tracking import PerPromptStatTracker
    cfg = SimpleNamespace(tau_id=.5, tau_motion=.1, tau_quality=.4, adv_clip_max=5.)
    tracker = PerPromptStatTracker()
    rows = [dict(candidate_index=index, prompt='duplicate caption', record_key=record_key,
        geo=reward, anchor_s_id=1., s_id=1., motion=1., quality=dict(VQ=0., MQ=0.),
        reference_quality=dict(VQ=0., MQ=0.)) for index, (record_key, reward) in enumerate(
            [('sample-A', 1.), ('sample-A', 3.), ('sample-B', 100.), ('sample-B', 200.)])]
    h3_training.endpoint_decisions(rows, tracker, cfg)
    assert tracker.stats == {'sample-A': [1., 3.], 'sample-B': [100., 200.]}
    rows[0]['geo'] = 2.
    _, repeated = h3_training.endpoint_decisions(rows[:1], tracker, cfg)
    assert repeated == {0: .5}
    assert tracker.stats == {'sample-A': [1., 3., 2.], 'sample-B': [100., 200.]}


def _worker(rank, count, rendezvous, output, failure):
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=count+1, timeout=timedelta(seconds=5))
    policy_group = dist.new_group(list(range(count)), timeout=timedelta(seconds=3))
    devices = H3Devices(rank, 'policy' if rank < count else 'reward', count, (count,),
                        torch.device('cpu'), policy_group, dist.group.WORLD, policy_group,
                        dist.group.WORLD, rank if rank < count else -1)
    results = []
    try:
        for mask in ([1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]):
            policy = TinyPolicy(devices) if rank < count else None
            rollout = TinyRollout(policy, 'forward' if failure == 'forward' and rank == 0 else
                                  'collective' if failure == 'forward' and rank == 1 else None)
            candidates = h3_training.collect_candidates(policy, rollout,
                dict(conditioning={}, geometry={}, seed=20, attempt=0), rank=rank, policy_count=count)
            targets, weights = {}, {}
            for candidate in candidates:
                candidate.dummy = not bool(mask[candidate.candidate_index])
                if not candidate.dummy:
                    anchor = candidate.query.y0
                    targets[candidate.candidate_index] = H3Targets(anchor+.4, anchor-.2, {})
                    weights[candidate.candidate_index] = .8 - candidate.candidate_index*.1
            if failure == 'target' and rank == 1:
                targets[1].positive.fill_(float('nan'))
            if rank < count:
                optimizer = torch.optim.SGD(policy.trainable_parameters, lr=.03)
                before = policy.parameter.detach().clone()
                try:
                    metrics = h3_training.fit_h3_update(policy, rollout, candidates, targets=targets,
                        weights=weights, global_kept=sum(mask), optimizer=optimizer, cfg=settings())
                    assert failure is None
                    reference = torch.nn.Parameter(before.clone())
                    if sum(mask):
                        losses = []
                        for index, kept in enumerate(mask):
                            if kept:
                                anchor = torch.tensor([[float((20+index) % 7), 1.]])
                                losses.append(branch_loss(reference[None]*anchor, anchor, anchor+.4,
                                    anchor-.2, torch.tensor([.8-index*.1]), 1.).mean())
                        (sum(losses)*5/sum(mask)).backward()
                        torch.nn.utils.clip_grad_norm_((reference,), 1.)
                        reference.data.add_(reference.grad, alpha=-.03)
                    torch.testing.assert_close(policy.parameter, reference)
                    assert rollout.calls == (4//count if sum(mask) else 0)
                    assert policy.ema_calls == bool(sum(mask))
                    assert metrics['retained'] == sum(mask)
                    results.append([candidate.candidate_index for candidate in candidates])
                except RuntimeError:
                    if failure is None:
                        raise
                    torch.testing.assert_close(policy.parameter, before)
                    assert policy.ema_calls == 0
                    results.append('failed_before_step')
            else:
                assert not candidates and rollout.calls == 0
            if failure:
                break
        torch.save(results, Path(output)/f'{rank}.pt')
    finally:
        dist.destroy_process_group()


def _distributed(tmp_path, count, failure=None):
    context = mp.get_context('spawn')
    processes = [context.Process(target=_worker, args=(rank, count, str(tmp_path/'rendezvous'),
                                                     str(tmp_path), failure)) for rank in range(count+1)]
    started = time.monotonic()
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(max(0, 60-(time.monotonic()-started)))
        assert all(not process.is_alive() for process in processes), 'distributed test exceeded 60 seconds'
        assert all(process.exitcode == 0 for process in processes)
        return [torch.load(tmp_path/f'{rank}.pt', weights_only=True) for rank in range(count+1)]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(5)


@pytest.mark.parametrize('count', [2, 4])
def test_gloo_reference_parity_ownership_and_rejection(tmp_path, count):
    results = _distributed(tmp_path, count)
    assert sorted(index for rank in results[:-1] for index in rank[0]) == [0, 1, 2, 3]


@pytest.mark.parametrize('failure', ['target', 'forward'])
def test_gloo_one_rank_failure_never_steps_and_is_bounded(tmp_path, failure):
    results = _distributed(tmp_path, 2, failure)
    assert all(result == ['failed_before_step'] for result in results[:-1])


def test_zero_gradient_does_not_step_or_update_ema():
    policy = TinyPolicy(H3Devices.cpu())
    rollout = TinyRollout(policy)
    candidates = h3_training.collect_candidates(policy, rollout,
        dict(conditioning={}, geometry={}, seed=20, attempt=0), rank=0, policy_count=1)
    targets = {candidate.candidate_index: H3Targets(candidate.query.y0, candidate.query.y0, {})
               for candidate in candidates}
    for candidate in candidates:
        candidate.query.y0.zero_()
    with pytest.raises(RuntimeError, match='nonzero'):
        h3_training.fit_h3_update(policy, rollout, candidates, targets=targets,
            weights=dict.fromkeys(range(4), .5), global_kept=4,
            optimizer=torch.optim.SGD(policy.trainable_parameters, lr=.03), cfg=settings())
    assert policy.ema_calls == 0


@pytest.mark.parametrize('total_updates', [1, 2])
@pytest.mark.parametrize('walltime_stop', [False, True])
@pytest.mark.parametrize('checkpoint_interval', [1, 2])
def test_runner_eight_global_evaluation_records_and_checkpoint_boundary(tmp_path, monkeypatch, total_updates,
                                                                       walltime_stop, checkpoint_interval,
                                                                       entrypoint=False, verification_phase=None,
                                                                       exhausted_resume=False, timeout_budget=None, acceptance_retry=None):
    from ml_collections import ConfigDict
    # Resolve audit helpers against the real checkpoint module before injection.
    __import__('diffusionopsd.video.h3_verification')
    records = [dict(sample_id=f'row-{index}', text=f'prompt-{index}') for index in range(8)]
    cfg = ConfigDict(dict(policy_gpus=[0], target_gpus=[], conditioning_cache=str(tmp_path),
        model='unused', model_revision='revision', lora_rank=2, lora_alpha=4, learning_rate=.03,
        query_sigma=.278, num_steps=3, video_shift=12., audio_shift=3., seed=20, updates=total_updates,
        num_frames=124, height=32, width=32, fps=24, tau_id=.5, tau_motion=.1, tau_quality=.4,
        beta=1., loss_scale=5., max_grad_norm=1., ema_decay=.99, eval_every=1,
        checkpoint_every=checkpoint_interval, save_videos=True,
        max_updates_this_job=1 if checkpoint_interval == 1 else 0))
    cfg.walltime_seconds = 1e-9 if walltime_stop else 0
    if timeout_budget is not None:
        cfg.collective_timeout_s, cfg.service_timeout_s = timeout_budget
    cfg.model = str(tmp_path/'model')
    audio_config = Path(cfg.model)/'audio_vae/config.json'
    audio_config.parent.mkdir(parents=True)
    audio_config.write_text(__import__('json').dumps(dict(sampling_rate=32000,
        encoder_rates=[2, 4, 4, 5, 5], decoder_rates=[5, 5, 2, 2, 2, 2, 2])))
    if entrypoint:
        cfg.update(dict(policy_gpus=[0, 1], target_gpus=[2], num_nodes=1, train_pool_size=8,
            workflow='t2va', clips_per_prompt=4, candidates_per_node=4, policy_microbatch_size=1, eval_count=8))
    manifest = dict(identity={'pinned': 'identity'}, geometry={'num_frames': 124},
                    records={row['sample_id']: {'identity': {}} for row in records})
    (tmp_path/'manifest.json').write_text(__import__('json').dumps(manifest))
    devices = H3Devices.cpu()
    def initialize(**kwargs):
        assert kwargs['timeout_s'] == (3600. if timeout_budget is None else timeout_budget[0])
        return devices
    monkeypatch.setattr(h3_training, 'init_h3_distributed', initialize)
    from diffusionopsd.video.h3_conditioning import validate_h3_topology
    topology = validate_h3_topology(cfg) if entrypoint else {'policy_ranks': 1}
    if acceptance_retry:
        topology = validate_h3_topology(SimpleNamespace(num_nodes=1, policy_gpus=[0, 1], target_gpus=[2],
            train_pool_size=8, workflow='t2va', candidates_per_node=4, clips_per_prompt=4,
            policy_microbatch_size=1, eval_count=8))
    monkeypatch.setattr(h3_training, 'validate_h3_assets', lambda *args: dict(
        identity=manifest['identity'], topology=topology, manifest_sha256='hash',
        manifest=str(tmp_path/'manifest.json')))
    monkeypatch.setattr(h3_training, 'load_training_data', lambda *args, **kwargs: SimpleNamespace(is_hf=True))
    retry_pool = bool(acceptance_retry)
    monkeypatch.setattr(h3_training, 'select_records', lambda *args: dict(
        train=records[:2] if retry_pool else records[:1], calibration=records[:1], evaluation=records))
    monkeypatch.setattr(h3_training, 'load_conditioning', lambda *args, **kwargs: {})
    policy = TinyPolicy(devices)
    if entrypoint or acceptance_retry:
        policy.model = torch.nn.Module()
        policy.model.block = torch.nn.Module()
        adapters = torch.nn.ModuleDict({'default': torch.nn.Module(), 'old': torch.nn.Module()})
        adapters['default'].register_parameter('weight', policy.parameter)
        adapters['old'].register_parameter('weight', torch.nn.Parameter(policy.old, requires_grad=False))
        policy.model.block.lora_A = adapters
    monkeypatch.setattr(h3_training, 'load_h3_policy', lambda *args, **kwargs: policy)
    monkeypatch.setattr(h3_training, 'build_h3_core', lambda *args, **kwargs: policy)
    rollouts = []
    def make_rollout(core, **kwargs):
        result = TinyRollout(core)
        rollouts.append(result)
        return result
    monkeypatch.setattr(h3_training, 'H3Rollout', make_rollout)
    requests = []
    rejected_snapshot = None
    rejected_attempt = 1 if verification_phase == 'resume' else 0

    class LocalClient:
        def __init__(self, *args, **kwargs):
            assert kwargs['timeout_s'] == (600. if timeout_budget is None else timeout_budget[1])
            self.sequence = 0
            self.run_id = kwargs['run_id']

        def request(self, kind, candidate_id, payload, metadata):
            self.sequence += 1
            requests.append((kind, candidate_id, dict(metadata)))
            if kind == 'score_endpoint':
                identity = 0. if candidate_id.startswith('endpoint/') and metadata['candidate_index'] == 1 else 1.
                return dict(geo=[float(metadata['candidate_index'])], s_id=[identity], motion=[1.],
                            quality=dict(VQ=0., MQ=0., TA=0., Overall=0.))
            if kind == 'score_anchor':
                if acceptance_retry:
                    nonlocal rejected_snapshot
                    if metadata['attempt'] == rejected_attempt+1:
                        rejected_snapshot = (policy.parameter.detach().clone(), policy.old.clone(), policy.ema_calls)
                    if acceptance_retry == 'full_pool' or metadata['attempt'] == rejected_attempt:
                        return dict(s_id=[0.])
                assert not payload['video'].requires_grad
                assert payload['video'][0, 0] < 10
                return dict(s_id=[0. if metadata['candidate_index'] == 0 else 1.])
            if kind == 'construct_targets':
                return dict(positive=payload['video']+.4, negative=payload['video']-.2, diagnostics={'gradient_norm': [1.]})
            if kind == 'decode_evaluation':
                return dict(video=torch.ones(1, 2, 3, 2, 2)*.5,
                            audio=torch.ones(1, 2, 165600)*.1, sampling_rate=32000)
            raise AssertionError(kind)
    monkeypatch.setattr(h3_training, 'H3RewardClient', LocalClient)
    evaluation_updates = []
    logger_instances = []
    sdk_instances = []
    logger_finishes = []

    class LocalLogger:
        def __init__(self, *args, **kwargs):
            logger_instances.append(kwargs)
            assert kwargs['rank'] == 0
            assert kwargs['owned_cuda_device'] is None
            assert args[1]['h3_provenance']['identity'] == manifest['identity']
            self.experiment_group = 'group'
            self.real = None
            if entrypoint:
                import runpy
                from diffusionopsd.video.training_logging import VideoTrainingLogger
                sdk = runpy.run_path('tests/video/test_training_logging.py')['FakeWandb']()
                sdk_instances.append(sdk)
                self.real = VideoTrainingLogger(*args, **dict(kwargs, experiment_group='group'), wandb_module=sdk)

        def start(self):
            if self.real:
                self.real.start()

        def log_eval(self, update, metrics, videos):
            evaluation_updates.append(update)
            assert metrics['eval/num_prompts'] == 8
            assert len(videos) == 8
            assert all(Path(video['audio_path']).exists() for video in videos)
            if self.real:
                self.real.log_eval(update, metrics, videos)

        def log_train(self, update, metrics):
            assert metrics['retained'] == 3
            if update == 2:
                assert (tmp_path/'run/update-000001.pt').exists(), 'successful update 1 must already be published'
            if self.real:
                self.real.log_train(update, metrics)

        def finish(self, *args):
            logger_finishes.append(args)
            if self.real:
                self.real.finish(*args)
    monkeypatch.setattr(h3_training, 'VideoTrainingLogger', LocalLogger)
    monkeypatch.setattr(h3_training, 'save_video', lambda clip, path, fps: Path(path).write_bytes(b'video'))
    module = types.ModuleType('diffusionopsd.video.h3_checkpoint')

    def save_checkpoint(path, policy, optimizer, *, state, rank_state, all_policy_group, node_policy_group,
                        topology, control_group):
        assert state['update'] in (0, 1, 2)
        assert state['prompt_cursor']['cursor'] == state['update']
        if state['update']:
            assert len(state['baselines']['evaluation']) == 8
        else:
            assert state['baselines'] == {}
        assert 'max_updates_this_job' not in state['signature']['training']
        assert rank_state['service_sequence'] >= 0
        assert state['tracker'] == ({'row-0': [1., 2., 3.] * state['update']} if state['update'] else {})
        assert rank_state['service_run_id']
        if state['update']:
            reference = state['training_references']['row-0']
            assert reference['identity']['seed'] == 20
            assert reference['identity']['sample_id'] == 'row-0'
        torch.save(dict(state=state, rank_state=rank_state), path)

    def load_checkpoint(path, policy, optimizer, *, expected_signature, all_policy_group, node_policy_group,
                        topology, control_group):
        saved = torch.load(path, weights_only=False)
        assert saved['state']['signature'] == expected_signature
        return saved['state'], saved['rank_state']
    module.save_h3_checkpoint = save_checkpoint
    module.load_h3_checkpoint = load_checkpoint
    if not entrypoint and not acceptance_retry:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    hooks = []
    class AuditSpy:
        phase = verification_phase

        def __getattr__(self, name):
            def record(*args, **kwargs):
                hooks.append(name)
            return record

        def write(self, *args, reached, **kwargs):
            return dict(status='stage_measured', reached_phase=reached)
    verification = AuditSpy() if verification_phase else None
    if acceptance_retry:
        from diffusionopsd.video.h3_verification import H3Verification
        if verification_phase == 'resume':
            acceptance_retry = None
            h3_training.run_h3_training(cfg, run_dir=tmp_path/'run')
            acceptance_retry = 'then_accept'
        verification = H3Verification(verification_phase)
        initial_ema_calls = policy.ema_calls
        initial_parameter, initial_old = policy.parameter.detach().clone(), policy.old.clone()
        if acceptance_retry == 'after_reject_walltime':
            checks = iter([False, False, True])
            monkeypatch.setattr(h3_training.H3SafeBoundary, 'should_stop', lambda *args: next(checks))
        if acceptance_retry == 'full_pool':
            with pytest.raises(RuntimeError, match='full pass'):
                h3_training.run_h3_training(cfg, run_dir=tmp_path/'run', verification=verification)
            report = json.loads((tmp_path/f'run/verification-{verification_phase}-rank-0.json').read_text())
            assert report['reached_phase'] == 'failed'
            assert report['prompt_cursor'] == dict(num_prompts=2, cursor=2, consecutive_empty=2)
            assert report['update'] == 0
            assert policy.ema_calls == 0
            assert torch.equal(policy.parameter, initial_parameter) and torch.equal(policy.old, initial_old)
            assert not any(event['stage'] in ('target', 'update', 'checkpoint') for event in report['events'])
            assert len([request for request in requests if request[0] == 'score_anchor']) == 8
            assert not any(request[0] == 'construct_targets' for request in requests)
            assert not list((tmp_path/'run').glob('update-*.pt'))
        elif acceptance_retry in ('walltime', 'after_reject_walltime'):
            result = h3_training.run_h3_training(cfg, run_dir=tmp_path/'run', verification=verification)
            assert result['reached_phase'] == 'incomplete'
            assert result['status'] != 'stage_measured'
            assert policy.ema_calls == 0
            assert result['prompt_cursor']['cursor'] == (1 if acceptance_retry == 'after_reject_walltime' else 0)
            assert not any(event['stage'] in ('target', 'update', 'checkpoint') for event in result['events'])
            return
        else:
            result = h3_training.run_h3_training(cfg, run_dir=tmp_path/'run', verification=verification,
                resume=tmp_path/'run/update-000001.pt' if verification_phase == 'resume' else None)
            report = json.loads((tmp_path/f'run/verification-{verification_phase}-rank-0.json').read_text())
            assert result['reached_phase'] == verification_phase
            assert report['prompt_cursor'] == dict(num_prompts=2, cursor=2+rejected_attempt, consecutive_empty=0)
            assert rejected_snapshot is not None
            assert torch.equal(rejected_snapshot[0], initial_parameter)
            assert torch.equal(rejected_snapshot[1], initial_old) and rejected_snapshot[2] == initial_ema_calls
            assert len([event for event in report['events'] if event['stage'] == 'target']) == 1
            assert policy.ema_calls == initial_ema_calls + (0 if verification_phase == 'target' else 1)
            if verification_phase in ('update', 'resume'):
                assert not torch.equal(policy.parameter, initial_parameter)
                checkpoint = torch.load(tmp_path/f'run/update-{1+rejected_attempt:06d}.pt', weights_only=False)
                assert checkpoint['optimizer']['state']
                assert all(float(item['step']) == 1+rejected_attempt
                           for item in checkpoint['optimizer']['state'].values())
                assert checkpoint['common']['update'] == 1+rejected_attempt
                assert checkpoint['common']['prompt_cursor']['cursor'] == 2+rejected_attempt
        assert report['calibration']['tau_id'] == .5
        decisions = [json.loads(line) for line in (tmp_path/'run/gate-decisions-rank-0.jsonl').read_text().splitlines()]
        assert len(decisions) == 2
        assert decisions[0]['attempt'] == rejected_attempt and decisions[0]['global_accepted'] == 0
        assert decisions[0]['candidates'][0] == dict(candidate_index=0, seed=20+4*rejected_attempt,
            record_key=f'row-{rejected_attempt}',
            sample_id=f'row-{rejected_attempt}', anchor_s_id=0., endpoint_s_id=1., accepted=False, margin=-.5)
        assert decisions[1]['candidates'][0]['record_key'] == f'row-{1-rejected_attempt}'
        assert decisions[1]['candidates'][0]['seed'] == 24+4*rejected_attempt
        return
    if entrypoint:
        import runpy
        entry = runpy.run_path('scripts/train_opsd_video_minimax_h3.py')
        monkeypatch.setitem(entry['main'].__globals__, 'load_config', lambda path: cfg)
        monkeypatch.setitem(entry['main'].__globals__, 'preflight', lambda config, **kwargs: {})
        result = entry['main'](['--run-dir', str(tmp_path/'run')])
    else:
        if verification_phase == 'resume':
            h3_training.run_h3_training(cfg, run_dir=tmp_path/'run')
        if exhausted_resume:
            # Re-enter as the service role with a real final-total fixture checkpoint.
            from dataclasses import replace
            devices = replace(devices, role='reward')
            closed = []
            service = SimpleNamespace(judge=SimpleNamespace(close=lambda: closed.append(True)))
            monkeypatch.setattr(h3_training.H3RewardService, 'from_config', lambda *args: service)
            profiler_path = tmp_path/'run/policy-phases-rank-0.json'
            profiler_path.unlink()
            with pytest.raises(ValueError, match='available optimizer update budget'):
                h3_training.run_h3_training(cfg, run_dir=tmp_path/'run', verification=verification,
                                           resume=tmp_path/'run/update-000001.pt')
            assert closed == [True]
            assert logger_finishes[-1] == ('failed',)
            assert profiler_path.exists()
            phases = __import__('json').loads(profiler_path.read_text())
            assert 'restore' in [row['phase'] for row in phases]
            return
        result = h3_training.run_h3_training(cfg, run_dir=tmp_path/'run', verification=verification,
            resume=tmp_path/'run/update-000001.pt' if verification_phase == 'resume' else None)
    if verification_phase:
        assert result == dict(status='stage_measured', reached_phase=verification_phase)
        assert 'loaded' in hooks
        assert ('rollout' in hooks) == (verification_phase != 'load')
        assert ('targets' in hooks) == (verification_phase in ('target', 'update', 'resume'))
        assert ('updated' in hooks) == (verification_phase in ('update', 'resume'))
        assert ('checkpoint' in hooks) == (verification_phase in ('update', 'resume'))
        assert ('restored' in hooks) == (verification_phase == 'resume')
        if verification_phase == 'load':
            assert requests == []
        if verification_phase in ('load', 'rollout', 'target'):
            assert policy.ema_calls == 0
            assert not list((tmp_path/'run').glob('update-*.pt'))
        assert (tmp_path/'run/policy-phases-rank-0.json').exists()
        return
    assert len(logger_instances) == 1
    if entrypoint:
        assert sdk_instances[0].options['mode'] == 'offline'
        assert (tmp_path/'run/latest.json').exists()
        assert (tmp_path/'run/update-000001.pt.json').exists()
    if walltime_stop:
        assert result['update'] == 0
        assert result['status'] == 'checkpointed'
        assert (tmp_path/'run/update-000000.pt').exists()
        assert requests == []
        return
    assert result['update'] == (1 if checkpoint_interval == 1 else total_updates)
    assert (tmp_path/'run/update-000001.pt').exists()
    report = __import__('json').loads((tmp_path/'run/eval/update-000001.json').read_text())
    assert [row['sample_id'] for row in report['rows']] == [f'row-{index}' for index in range(8)]
    assert len(list((tmp_path/'run/eval/update-000001').glob('*.wav'))) == 8
    if total_updates == 2 and checkpoint_interval == 1:
        resumed = h3_training.run_h3_training(cfg, run_dir=tmp_path/'run', resume=tmp_path/'run/update-000001.pt')
        assert resumed['update'] == 2
        assert evaluation_updates == [0, 1, 2]
    anchors = [request for request in requests if request[0] == 'score_anchor']
    assert len(anchors) == 4 * total_updates
    target_indices = [request[2]['candidate_index'] for request in requests if request[0] == 'construct_targets']
    assert target_indices == [1, 2, 3] * total_updates
    training_references = [request for request in requests if request[1].startswith('training-reference/')]
    assert len(training_references) == 1
    assert training_references[0][2]['seed'] == 20
    assert sum(adapter is None and seed == 20 for rollout in rollouts
               for adapter, seed in zip(rollout.adapters, rollout.seeds)) == 1


@pytest.mark.parametrize('phase', ['load', 'rollout', 'target', 'update', 'resume'])
def test_verification_uses_production_runner_stage_boundaries(tmp_path, monkeypatch, phase):
    test_runner_eight_global_evaluation_records_and_checkpoint_boundary(tmp_path, monkeypatch,
        total_updates=2 if phase == 'resume' else 1, walltime_stop=False, checkpoint_interval=1,
        verification_phase=phase)


def test_verification_exhausted_resume_closes_judge_and_persists_profiler(tmp_path, monkeypatch):
    test_runner_eight_global_evaluation_records_and_checkpoint_boundary(tmp_path, monkeypatch,
        total_updates=1, walltime_stop=False, checkpoint_interval=1,
        verification_phase='resume', exhausted_resume=True)


def test_verification_media_does_not_reset_enclosing_evaluation_peak(tmp_path, monkeypatch):
    peak, active, measurements, resets = [0], [], [], []

    class ResettingProfiler:
        def __init__(self, device, phase, **kwargs):
            self.phase = phase

        def __enter__(self):
            resets.append((self.phase, tuple(active)))
            peak[0] = 0
            active.append(self.phase)
            return self

        def __exit__(self, *args):
            self.metrics = dict(peak_allocated_bytes=peak[0])
            measurements.append((self.phase, peak[0]))
            assert active.pop() == self.phase

    native_rollout = TinyRollout.rollout

    def measured_rollout(self, *args, **kwargs):
        result = native_rollout(self, *args, **kwargs)
        peak[0] = 99  # transient generation allocation, gone before the media audit
        return result

    monkeypatch.setattr(h3_training, 'H3PhaseProfiler', ResettingProfiler)
    monkeypatch.setattr(TinyRollout, 'rollout', measured_rollout)
    test_runner_eight_global_evaluation_records_and_checkpoint_boundary(tmp_path, monkeypatch,
        total_updates=1, walltime_stop=False, checkpoint_interval=1, verification_phase='update')
    assert not [(phase, parents) for phase, parents in resets if 'evaluation' in parents]
    assert [value for phase, value in measurements if phase == 'evaluation'] == [99, 99]


def _evaluation_worker(rank, rendezvous, output):
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=3, timeout=timedelta(seconds=10))
    policy_group = dist.new_group([0, 1], timeout=timedelta(seconds=5))
    devices = H3Devices(rank, 'policy' if rank < 2 else 'reward', 2, (2,), torch.device('cpu'),
                        policy_group=policy_group, all_policy_group=policy_group,
                        control_group=dist.group.WORLD, node_control_group=dist.group.WORLD,
                        policy_index=rank if rank < 2 else -1)

    class ScoreService:
        def __init__(self):
            self.calls = 0

        def dispatch(self, kind, payload, metadata):
            self.calls += 1
            assert kind == 'score_endpoint'
            assert not payload['video'].requires_grad
            return dict(geo=[float(metadata['candidate_index'])], s_id=[1.], motion=[1.],
                        quality=dict(VQ=0., MQ=0., TA=0., Overall=0.))
    class CollectiveRollout(TinyRollout):
        def rollout(self, *args, **kwargs):
            dist.all_reduce(torch.ones(1), group=policy_group)
            return super().rollout(*args, **kwargs)

    policy = TinyPolicy(devices) if rank < 2 else None
    rollout = CollectiveRollout(policy) if policy else None
    h3_training._conditioning = lambda *args: {}
    records = [dict(sample_id=f'row-{index}', text=f'prompt-{index}') for index in range(8)]
    cfg = SimpleNamespace(seed=20, num_frames=124, height=32, width=32, fps=24, save_videos=False)
    try:
        service = ScoreService() if rank == 2 else None
        client = h3_training.H3RewardClient(devices, service,
                                            run_id='evaluation-test', timeout_s=5)
        references = {}
        request = dict(record=records[0], record_key='row-0', conditioning={}, seed=999,
                       geometry=dict(num_frames=124, height=32, width=32, fps=24))
        reference = h3_training._training_reference(request, references, cfg=cfg, devices=devices,
            policy=policy, rollout=rollout, client=client, manifest={'identity': {}})
        repeated = h3_training._training_reference(dict(request, seed=123), references, cfg=cfg,
            devices=devices, policy=policy, rollout=rollout, client=client, manifest={'identity': {}})
        assert repeated == reference
        assert reference['identity']['seed'] == 20
        assert client.sequence == 1
        rows, _ = h3_training._evaluate_records(records, adapter='default', label='heldout', cfg=cfg,
            devices=devices, policy=policy, rollout=rollout, client=client, cache=Path(output),
            manifest={'identity': {}}, run_dir=Path(output))
        assert [row['candidate_index'] for row in rows] == list(range(8))
        assert [row['seed'] for row in rows] == list(range(100020, 100028))
        if rollout:
            assert rollout.generation_calls == 5
            assert rollout.adapters == [None] + ['default'] * 4
            assert rollout.seeds[0] == 20
        if service:
            assert service.calls == 9
        torch.save(rows, Path(output)/f'{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_gloo_eight_prompt_evaluation_uses_owner_only_service_transport(tmp_path):
    context = mp.get_context('spawn')
    processes = [context.Process(target=_evaluation_worker,
        args=(rank, str(tmp_path/'rendezvous'), str(tmp_path))) for rank in range(3)]
    started = time.monotonic()
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(max(0, 60-(time.monotonic()-started)))
        assert all(not process.is_alive() for process in processes), 'evaluation exceeded 60 seconds'
        assert all(process.exitcode == 0 for process in processes)
        rows = [torch.load(tmp_path/f'{rank}.pt', weights_only=True) for rank in range(3)]
        assert rows[0] == rows[1] == rows[2]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(5)


@pytest.mark.parametrize('phase', ['target', 'update', 'resume'])
def test_verification_retries_rejected_attempt_before_real_success(tmp_path, monkeypatch, phase):
    test_runner_eight_global_evaluation_records_and_checkpoint_boundary(tmp_path, monkeypatch,
        total_updates=2 if phase == 'resume' else 1, walltime_stop=False, checkpoint_interval=1,
        verification_phase=phase, acceptance_retry='then_accept')


def test_verification_full_pool_rejection_persists_live_cursor_and_gate_evidence(tmp_path, monkeypatch):
    test_runner_eight_global_evaluation_records_and_checkpoint_boundary(tmp_path, monkeypatch,
        total_updates=1, walltime_stop=False, checkpoint_interval=1,
        verification_phase='target', acceptance_retry='full_pool')


def test_verification_walltime_cannot_claim_update_success(tmp_path, monkeypatch):
    test_runner_eight_global_evaluation_records_and_checkpoint_boundary(tmp_path, monkeypatch,
        total_updates=1, walltime_stop=True, checkpoint_interval=1,
        verification_phase='update', acceptance_retry='walltime')


def test_verification_walltime_after_rejection_cannot_claim_update_success(tmp_path, monkeypatch):
    test_runner_eight_global_evaluation_records_and_checkpoint_boundary(tmp_path, monkeypatch,
        total_updates=1, walltime_stop=False, checkpoint_interval=1,
        verification_phase='update', acceptance_retry='after_reject_walltime')
