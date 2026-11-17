"""Independent node prompts, shared scalar cache and one global update/restart."""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import runpy
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusionopsd.video import h3_training as training
from diffusionopsd.video.branch_loss import branch_loss
from diffusionopsd.video.h3_auxiliary import H3RewardClient, H3Targets
from diffusionopsd.video.h3_checkpoint import save_h3_checkpoint, load_h3_checkpoint, _digest
from diffusionopsd.video.h3_fsdp import H3Devices


@pytest.mark.parametrize('nodes', [1, 2, 4])
@pytest.mark.parametrize('policies', [2, 4])
def test_four_candidates_per_node_independent_of_policy_width(nodes, policies):
    scheduler = getattr(training, 'h3_candidate_assignments', None)
    assert callable(scheduler), 'missing per-node candidate scheduler'
    order = [5, 2, 7, 1, 6, 3, 0, 4]
    for attempt in (0, 1, 7, 8):
        prompts, total = [], []
        for node in range(nodes):
            local = [scheduler(attempt, node, nodes, rank, policies, order) for rank in range(policies)]
            assert all(len(slots) == 4//policies for slots in local)
            pairs = [pair for slots in local for pair in slots]
            assert sorted(candidate for _, candidate in pairs) == list(range(4))
            assert {record for record, _ in pairs} == {order[(attempt*nodes+node) % len(order)]}
            prompts.append(pairs[0][0])
            total.extend((node, *pair) for pair in pairs)
        assert len(set(prompts)) == nodes
        assert len(set(total)) == 4*nodes


@pytest.mark.parametrize('nodes,policies', [(2, 2), (4, 4)])
def test_evaluation_equal_rounds_and_eight_unique_global_records(nodes, policies):
    layout = getattr(training, 'h3_evaluation_assignments', None)
    assert callable(layout), 'missing explicit global evaluation layout'
    slots = [layout(node, nodes, rank, policies, 8) for node in range(nodes) for rank in range(policies)]
    assert {len(row) for row in slots} == {__import__('math').ceil(8/(nodes*policies))}
    assert sorted(index for row in slots for index in row if index is not None) == list(range(8))


def test_devices_named_node_interfaces():
    devices = H3Devices(4, 'policy', 2, (5,), torch.device('cpu'), policy_index=1, num_nodes=2, node_index=1)
    assert getattr(devices, 'node_rank', None) == 1
    assert getattr(devices, 'node_count', None) == 2
    assert getattr(devices, 'node_policy_rank', None) == 1
    assert hasattr(devices, 'node_policy_group')


@pytest.mark.parametrize('deadline', ['broken', 'nan', '999'])
def test_invalid_allocation_deadline_fails_closed(monkeypatch, deadline):
    entry = runpy.run_path('scripts/train_opsd_video_minimax_h3.py')
    resolve = entry.get('allocation_walltime_seconds')
    assert callable(resolve), 'missing allocation deadline default'
    monkeypatch.setenv('SLURM_JOB_END_TIME', deadline)
    with pytest.raises(ValueError, match='deadline'):
        resolve(None, now=1000.)


def test_allocation_deadline_counts_from_startup_and_explicit_override(monkeypatch):
    entry = runpy.run_path('scripts/train_opsd_video_minimax_h3.py')
    resolve = entry.get('allocation_walltime_seconds')
    assert callable(resolve), 'missing allocation deadline default'
    monkeypatch.setenv('SLURM_JOB_END_TIME', '1500')
    assert resolve(None, now=1000.) == 500
    assert resolve(0, now=1000.) == 0
    assert resolve(45, now=1000.) == 45
    monkeypatch.delenv('SLURM_JOB_END_TIME')
    assert resolve(None, now=1000.) == 0


def test_checkpoint_explicit_node_global_group_and_topology_interfaces(tmp_path):
    from test_h3_checkpoint import policy, signature
    p = policy()
    opt = torch.optim.AdamW(p.trainable_parameters)
    state = dict(signature=signature(), update=0)
    path = tmp_path/'explicit-topology.pt'
    save_h3_checkpoint(path, p, opt, state=state, rank_state={},
        all_policy_group=None, node_policy_group=None, topology=state['signature']['topology'], control_group=None)
    common, _ = load_h3_checkpoint(path, p, opt, expected_signature=state['signature'],
        all_policy_group=None, node_policy_group=None, topology=state['signature']['topology'], control_group=None)
    assert common == state


def test_absolute_deadline_expired_during_startup_is_still_limited(monkeypatch):
    import diffusionopsd.video.h3_checkpoint as checkpoint
    monkeypatch.setattr(checkpoint.time, 'time', lambda: 2000.)
    boundary = checkpoint.H3SafeBoundary(walltime_seconds=1000, deadline=1500, clock=lambda: 50.)
    assert boundary.should_stop(0, 0, None)


class Policy:
    def __init__(self, devices):
        self.devices = devices
        self.model = torch.nn.Module()
        self.model.block = torch.nn.Module()
        self.model.block.lora_A = torch.nn.ModuleDict({'default': torch.nn.Linear(2, 1, bias=False),
                                               'old': torch.nn.Linear(2, 1, bias=False)})
        with torch.no_grad():
            self.parameter.copy_(torch.tensor([[.2, -.3]]))
            self.old.copy_(self.parameter)
        self.old.requires_grad_(False)
        self.trainable_parameters = (self.parameter,)
        self.ema_calls = 0

    @property
    def parameter(self):
        return self.model.block.lora_A['default'].weight

    @property
    def old(self):
        return self.model.block.lora_A['old'].weight

    def ema_(self, decay):
        self.old.lerp_(self.parameter.detach(), 1-decay)
        self.ema_calls += 1


class Rollout:
    def __init__(self, policy):
        self.policy, self.calls = policy, 0

    def rollout(self, conditioning, *, seed, geometry, adapter='old'):
        # Actual node-group collective: every FSDP member must participate.
        dist.all_reduce(torch.ones(1), group=self.policy.devices.policy_group)
        self.calls += 1
        anchor = torch.tensor([[float(seed % 7)+1, 1.]])
        return SimpleNamespace(y0=anchor, video_endpoint=anchor+10, audio_endpoint=anchor)

    def prediction(self, query):
        dist.all_reduce(torch.ones(1), group=self.policy.devices.policy_group)
        return self.policy.parameter*query.y0


def _worker(rank, rendezvous, output, inject_failure):
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=6,
                            timeout=timedelta(seconds=20))
    all_policy = dist.new_group([0, 1, 3, 4], timeout=timedelta(seconds=15))
    node, local = divmod(rank, 3)
    policy_group = node_group = None
    for index in range(2):
        p = dist.new_group([index*3, index*3+1], timeout=timedelta(seconds=15))
        g = dist.new_group(list(range(index*3, index*3+3)), timeout=timedelta(seconds=15))
        if node == index:
            policy_group, node_group = p, g
    devices = H3Devices(rank, 'policy' if local < 2 else 'reward', 2, (node*3+2,), torch.device('cpu'),
        policy_group, dist.group.WORLD, all_policy, node_group, local if local < 2 else -1, 2, node)
    policy = Policy(devices) if local < 2 else None
    rollout = Rollout(policy) if policy else None
    cfg = SimpleNamespace(seed=20, beta=1., loss_scale=5., max_grad_norm=1., ema_decay=.99,
                          num_frames=124, height=32, width=32, fps=24, save_videos=False)
    records = [dict(sample_id=f'row-{i}', text=f'prompt-{i}') for i in range(8)]
    calls = []

    class Service:
        def dispatch(self, kind, payload, metadata):
            calls.append(metadata)
            if inject_failure and node == 1:
                raise RuntimeError('single-node injected service failure')
            assert metadata['owner_rank']//3 == node
            return dict(geo=[float(metadata['candidate_index'])], s_id=[1.], motion=[1.],
                        quality=dict(VQ=0., MQ=0., TA=0., Overall=0.))
    client = H3RewardClient(devices, Service() if not policy else None, run_id='two-node', timeout_s=10)
    references = {}
    try:
        request = dict(record=records[node], record_key=f'row-{node}', conditioning={},
                       geometry={}, attempt=0, seed=20+node*4)
        if inject_failure:
            with pytest.raises(RuntimeError, match='single-node injected service failure'):
                training._training_reference(request, references, cfg=cfg, devices=devices,
                    policy=policy, rollout=rollout, client=client, manifest={'identity': {}})
            assert policy is None or policy.ema_calls == 0
            return
        reference = training._training_reference(request, references, cfg=cfg, devices=devices,
            policy=policy, rollout=rollout, client=client, manifest={'identity': {}})
        assert sorted(references) == ['row-0', 'row-1'], 'all nodes must share both scalar references'
        assert reference['identity']['sample_id'] == f'row-{node}'
        assert len(set(training._exchange(_digest(references), dist.group.WORLD))) == 1
        before_calls = rollout.calls if rollout else 0
        training._training_reference(request, references, cfg=cfg, devices=devices, policy=policy,
            rollout=rollout, client=client, manifest={'identity': {}})
        assert not rollout or rollout.calls == before_calls
        training._conditioning = lambda *args: {}
        rows, _ = training._evaluate_records(records, adapter=None, label='base', cfg=cfg, devices=devices,
            policy=policy, rollout=rollout, client=client, cache=Path(output), manifest={'identity': {}},
            run_dir=Path(output))
        assert [row['candidate_index'] for row in rows] == list(range(8))
        assert [row['sample_id'] for row in rows] == [f'row-{i}' for i in range(8)]
        candidates = training.collect_candidates(policy, rollout, request, rank=rank, policy_count=2)
        for mask in ([1, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 0], [0]*8):
            p = Policy(devices) if policy else None
            opt = torch.optim.AdamW(p.trainable_parameters, lr=.03) if p else None
            targets, weights = {}, {}
            for c in candidates:
                c.dummy = not mask[c.candidate_index]
                if not c.dummy:
                    targets[c.candidate_index] = H3Targets(c.query.y0+.4, c.query.y0-.2, {})
                    weights[c.candidate_index] = .8-c.candidate_index*.05
            def fit():
                if p:
                    return training.fit_h3_update(p, Rollout(p), candidates, targets=targets,
                        weights=weights, global_kept=sum(mask), optimizer=opt, cfg=cfg)
            metrics = training._phase('global-fit', fit, dist.group.WORLD)
            if p:
                ref = Policy(H3Devices.cpu())
                ref_opt = torch.optim.AdamW(ref.trainable_parameters, lr=.03)
                losses = []
                for i, kept in enumerate(mask):
                    if kept:
                        anchor = torch.tensor([[float((20+i) % 7)+1, 1.]])
                        losses.append(branch_loss(ref.parameter*anchor, anchor, anchor+.4, anchor-.2,
                            torch.tensor([.8-i*.05]), 1.).mean())
                if losses:
                    (sum(losses)*cfg.loss_scale/sum(mask)).backward()
                    torch.nn.utils.clip_grad_norm_(ref.trainable_parameters, 1.)
                    ref_opt.step()
                    ref.ema_(.99)
                torch.testing.assert_close(p.parameter, ref.parameter)
                torch.testing.assert_close(p.old, ref.old)
                assert _digest(opt.state_dict()) == _digest(ref_opt.state_dict())
                assert p.ema_calls == bool(sum(mask))
                assert metrics['optimizer_step'] == bool(sum(mask))
                assert len(set(training._exchange(_digest((p.parameter, p.old, opt.state_dict())), all_policy))) == 1
            # Interrupted publication/reconstruction after the same fixed global update.
            if sum(mask):
                from config.minimax_h3_t2va import get_config
                from diffusionopsd.video.h3_conditioning import validate_h3_topology
                configuration = get_config()
                configuration.num_nodes = 2
                topology = validate_h3_topology(configuration)
                state = dict(signature={'topology': topology}, update=1, attempt=3,
                    node_positions=[dict(node_rank=j, next_record=(3*2+j) % 8) for j in range(2)],
                    tracker={'row-0': [1.]}, training_references=references)
                path = Path(output)/f'update-{mask.index(1)}.pt'
                save_h3_checkpoint(path, p, opt, state=state,
                    rank_state={'service_sequence': client.sequence, 'service_run_id': client.run_id},
                    policy_group=all_policy, control_group=dist.group.WORLD)
                next_rng = torch.rand(3)
                expected = deepcopy(opt.state_dict()) if p else None
                restored = Policy(devices) if p else None
                restored_opt = torch.optim.AdamW(restored.trainable_parameters, lr=.03) if p else None
                common, caller = load_h3_checkpoint(path, restored, restored_opt,
                    expected_signature=state['signature'], policy_group=all_policy, control_group=dist.group.WORLD)
                assert common == state
                assert caller['service_sequence'] == client.sequence
                assert torch.equal(torch.rand(3), next_rng)
                if p:
                    assert _digest(restored_opt.state_dict()) == _digest(expected)
                    assert torch.equal(restored.parameter, p.parameter) and torch.equal(restored.old, p.old)
                def resumed_fit():
                    if restored:
                        return training.fit_h3_update(restored, Rollout(restored), candidates, targets=targets,
                            weights=weights, global_kept=sum(mask), optimizer=restored_opt, cfg=cfg)
                training._phase('continuous-second-update', fit, dist.group.WORLD)
                training._phase('resumed-second-update', resumed_fit, dist.group.WORLD)
                if p:
                    assert _digest((p.parameter, p.old, opt.state_dict())) == _digest(
                        (restored.parameter, restored.old, restored_opt.state_dict()))
                for position in common['node_positions']:
                    next_record = training.h3_candidate_assignments(common['attempt'], position['node_rank'],
                        2, 0, 2, list(range(8)))[0][0]
                    assert next_record == position['next_record']
        torch.save({'references': references, 'service_calls': calls}, Path(output)/f'rank-{rank}.pt')
        _runner_segmented_resume(devices, Path(output)/'runner')
    finally:
        dist.destroy_process_group()


def _runner_segmented_resume(devices, directory):
    """Actual multi-node runner request broadcast/calibration/cache/checkpoint flow."""
    from config.minimax_h3_t2va import get_config
    from diffusionopsd.video.h3_conditioning import validate_h3_topology
    cfg = get_config()
    cfg.num_nodes, cfg.seed, cfg.updates, cfg.max_updates_this_job = 2, 20, 2, 1
    cfg.train_pool_size, cfg.calibration_prompts, cfg.eval_every = 8, 2, 1
    cfg.height, cfg.width, cfg.save_videos, cfg.tau_id, cfg.tau_motion = 32, 32, False, .5, .1
    cfg.conditioning_cache = str(directory)
    records = [dict(sample_id=f'row-{i}', text=f'prompt-{i}') for i in range(8)]
    manifest = {'identity': {}, 'geometry': {}, 'records': {r['sample_id']: {'identity': {}} for r in records}}
    if devices.rank == 0:
        directory.mkdir(parents=True)
        (directory/'manifest.json').write_text(json.dumps(manifest))
    dist.barrier(group=devices.control_group)
    requests = []

    class Service:
        judge = None

        def dispatch(self, kind, payload, metadata):
            requests.append((kind, dict(metadata)))
            if kind == 'score_anchor':
                return {'s_id': [1. if devices.node_index == 0 else 0.]}
            if kind == 'construct_targets':
                return dict(positive=payload['video']+.4, negative=payload['video']-.2, diagnostics={})
            return dict(geo=[float(metadata['candidate_index'])], s_id=[1.], motion=[1.],
                        quality=dict(VQ=0., MQ=0., TA=0., Overall=0.))

    class Logger:
        experiment_group = 'two-node-fixture'

        def __init__(self, *a, **kw):
            pass

        def start(self):
            pass

        def finish(self, *a):
            pass

        def log_eval(self, *a):
            pass

        def log_train(self, update, metrics):
            assert devices.rank == 0 and metrics['retained'] == 4

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(training, 'init_h3_distributed', lambda **kw: devices)
        patch.setattr(training, 'validate_h3_assets', lambda *a: dict(identity={},
            topology=validate_h3_topology(cfg), manifest_sha256='fixture', manifest=str(directory/'manifest.json')))
        patch.setattr(training, 'load_training_data', lambda *a, **kw: SimpleNamespace(is_hf=True))
        patch.setattr(training, 'select_records', lambda *a: dict(train=records,
            calibration=records[:2], evaluation=records))
        patch.setattr(training, '_conditioning', lambda *a: {})
        patch.setattr(training, 'load_h3_policy', lambda *a, **kw: Policy(devices))
        patch.setattr(training.H3RewardService, 'from_config', lambda *a: Service())
        patch.setattr(training, 'build_h3_core', lambda root, p: p)
        patch.setattr(training, 'H3Rollout', lambda core, **kw: Rollout(core))
        patch.setattr(training, 'VideoTrainingLogger', Logger)
        first = training.run_h3_training(cfg, run_dir=directory/'run', num_nodes=2, policy_count=2, reward_count=1)
        assert first['update'] == 1
        second = training.run_h3_training(cfg, run_dir=directory/'run', resume=directory/'run/latest.json',
                                         num_nodes=2, policy_count=2, reward_count=1)
        assert second['update'] == 2
    endpoint_requests = [metadata for kind, metadata in requests
                         if kind == 'score_endpoint' and metadata['candidate_id'].startswith('endpoint/')]
    if devices.role == 'reward':
        assert len(endpoint_requests) == 8
        for metadata in endpoint_requests:
            attempt = metadata['attempt']
            assert metadata['record_key'] == f'row-{attempt*2+devices.node_index}'
            assert metadata['node_slot'] == devices.node_index
            assert f'/node-{devices.node_index}/' in metadata['request_id']
    if devices.rank == 0:
        saved = torch.load(directory/'run/update-000002.pt', weights_only=False)
        assert saved['common']['prompt_cursor']['cursor'] == 2
        assert saved['common']['node_positions'] == training._node_positions(2, 2, 8)
        assert set(saved['common']['training_references']) == {'row-0', 'row-1', 'row-2', 'row-3'}
        assert set(saved['common']['tracker']) == {'row-0', 'row-2'}


@pytest.mark.parametrize('failure', [False, True])
def test_two_node_groups_cache_evaluation_global_update_and_restart(tmp_path, failure):
    context = mp.spawn(_worker, args=(str(tmp_path/'rdzv'), str(tmp_path), failure), nprocs=6, join=False)
    deadline = time.monotonic()+100
    try:
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, 'two-node CPU test exceeded 100s'
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)


def test_four_node_four_policy_evaluation_padded_dummy_nodes_and_media(tmp_path, monkeypatch):
    """Twenty in-process rank participants, bounded control barriers and real WAVs.

    Actual six-rank Gloo tests cover transport. This larger layout uses threads
    because twenty separate Gloo contexts exceed the login node thread limit.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading

    class Group:
        def __init__(self, size):
            self.barrier = threading.Barrier(size, timeout=15)
            self.values = {}

        def exchange(self, value):
            self.values[threading.get_ident()] = value
            self.barrier.wait()
            result = list(self.values.values())
            self.barrier.wait()
            # Each thread replaces its slot only after everybody copied it.
            return result

    global_control = Group(20)
    node_controls = [Group(5) for _ in range(4)]
    policy_controls = [Group(4) for _ in range(4)]
    monkeypatch.setattr(training, '_exchange', lambda value, group: group.exchange(value))
    monkeypatch.setattr(training, '_conditioning', lambda *a: {})
    monkeypatch.setattr(training, 'save_video', lambda clip, path, fps: path.write_bytes(b'fixture-video'))
    records = [dict(sample_id=f'row-{i}', text=f'prompt-{i}') for i in range(8)]
    cfg = SimpleNamespace(seed=20, num_frames=124, height=32, width=32, fps=24, save_videos=True)

    def participant(rank):
        node, local = divmod(rank, 5)
        devices = H3Devices(rank, 'policy' if local < 4 else 'reward', 4, (node*5+4,), torch.device('cpu'),
            policy_group=policy_controls[node], control_group=global_control,
            node_control_group=node_controls[node], policy_index=local if local < 4 else -1,
            num_nodes=4, node_index=node)
        calls, forwards = [], []

        class Rollout:
            def rollout(self, conditioning, *, seed, geometry, adapter):
                policy_controls[node].exchange(None)
                forwards.append((seed, adapter))
                return SimpleNamespace(video_endpoint=torch.ones(1, 2, 2), audio_endpoint=torch.ones(1, 2, 2))

        class Client:
            def request(self, kind, candidate_id, payload, metadata):
                # Payload ownership is checked on every virtual rank; the real
                # bounded node-local transport is covered by the M2 Gloo test.
                assert bool(payload) == (rank == metadata['owner_rank'])
                assert metadata['owner_rank']//5 == node
                calls.append((kind, metadata['candidate_index']))
                if rank != metadata['owner_rank']:
                    return {}
                if kind == 'decode_evaluation':
                    return dict(video=torch.ones(1, 2, 3, 2, 2)*.5,
                                audio=torch.ones(1, 2, 16), sampling_rate=32000)
                return dict(geo=[float(metadata['candidate_index'])], s_id=[1.], motion=[1.],
                            quality=dict(VQ=0., MQ=0., TA=0., Overall=0.))
        policy = object() if local < 4 else None
        matched = []
        for adapter, label in ((None, 'base'), ('default', 'current')):
            rows, media = training._evaluate_records(records, adapter=adapter, label=label, cfg=cfg,
                devices=devices, policy=policy, rollout=Rollout() if policy else None, client=Client(),
                cache=tmp_path, manifest={'identity': {}}, run_dir=tmp_path)
            assert [row['candidate_index'] for row in rows] == list(range(8))
            assert [row['candidate_index'] for row in media] == list(range(8))
            assert len({row['path'] for row in media}) == 8
            assert all(Path(row['audio_path']).is_file() for row in media)
            matched.append([(row['sample_id'], row['seed']) for row in rows])
        assert matched[0] == matched[1]
        assert len(forwards) == (2 if policy else 0)
        assert len(calls) == (16 if node < 2 else 0)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(participant, rank) for rank in range(20)]
        for future in futures:
            future.result(timeout=30)


@pytest.mark.h3_gpu
@pytest.mark.skipif(os.environ.get('H3_RUN_GPU_TESTS') != '1', reason='opt-in two-node torchrun native CUDA test')
def test_two_node_native_cuda_update_resume(monkeypatch):
    """Two torchrun agents, three workers/node; shared H3_TEST_CHECKPOINT directory.

    Native tiny H3 forwards/backwards, real node-local Gloo payloads, production
    fitter, global adapter/AdamW/EMA agreement and exact second-update restart.
    This deliberately does not load full H3 or production reward checkpoints.
    """
    from test_h3_fsdp import tiny_model, packed_inputs
    from config.minimax_h3_t2va import get_config
    from diffusionopsd.video.h3_conditioning import validate_h3_topology
    from diffusionopsd.video.h3_fsdp import init_h3_distributed, load_h3_policy
    assert int(os.environ['WORLD_SIZE']) == 6 and int(os.environ['LOCAL_WORLD_SIZE']) == 3
    devices = init_h3_distributed(policy_gpus=(0, 1), target_gpus=(2,), timeout_s=120)
    directory = Path(os.environ['H3_TEST_CHECKPOINT'])/'task9-native-two-node'
    started = time.monotonic()
    try:
        if devices.rank == 0:
            directory.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(17)
            tiny_model().save_pretrained(directory/'transformer')
        dist.barrier(group=devices.control_group)
        cfg = get_config()
        cfg.num_nodes, cfg.learning_rate = 2, 1e-4
        cfg.tau_id, cfg.tau_motion = .5, .1
        signature = {'topology': validate_h3_topology(cfg)}
        p = load_h3_policy(directory/'transformer', 'test-local', devices, rank=2, alpha=4) if devices.role == 'policy' else None
        opt = torch.optim.AdamW(p.trainable_parameters, lr=cfg.learning_rate) if p else None

        class NativeRollout:
            def __init__(self, policy):
                self.policy = policy

            def rollout(self, conditioning, *, seed, geometry, adapter='old'):
                inputs = packed_inputs(devices.device, seed)
                with self.policy.generation_adapter(adapter), torch.no_grad():
                    output = self.policy.model(**inputs)
                return SimpleNamespace(y0=output.sample.detach(), video_endpoint=output.sample.detach(),
                    audio_endpoint=output.audio_sample.detach(), inputs=inputs)

            def prediction(self, query):
                return self.policy.model(**query.inputs).sample

        class NativeTargetService:
            def dispatch(self, kind, payload, metadata):
                assert devices.role == 'reward' and metadata['owner_rank']//3 == devices.node_index
                anchor = payload['video']
                assert not anchor.requires_grad
                assert kind == 'construct_targets' and metadata['accepted']
                return dict(positive=anchor+.4, negative=anchor-.2, diagnostics={})

        rollout = NativeRollout(p) if p else None
        client = H3RewardClient(devices, NativeTargetService() if not p else None,
                                run_id='task9-native-two-node', timeout_s=90)
        request = dict(conditioning={}, geometry={}, attempt=0, record_key=f'node-record-{devices.node_index}',
                       seed=20+devices.node_index*4)
        candidates = training.collect_candidates(p, rollout, request, rank=devices.rank, policy_count=2)
        # Entire second node rejected: its real native queries still replay zero loss.
        mask = [1, 0, 1, 1, 0, 0, 0, 0]
        local = {c.candidate_index: c for c in candidates}
        targets, weights = {}, {i: .8-i*.05 for i, accepted in enumerate(mask) if accepted}
        def make_targets():
            for candidate_index in range(devices.node_index*4, devices.node_index*4+4):
                if not mask[candidate_index]:
                    continue
                owner = devices.node_index*3+candidate_index % 2
                metadata = dict(owner_rank=owner, candidate_index=candidate_index, accepted=True,
                    node_slot=devices.node_index, record_key=request['record_key'], attempt=0,
                    local_candidate_index=candidate_index % 4)
                reply = client.request('construct_targets', f'native/node-{devices.node_index}/{candidate_index}',
                    {'video': local[candidate_index].query.y0} if devices.rank == owner else {}, metadata)
                if devices.rank == owner:
                    targets[candidate_index] = H3Targets(reply['positive'], reply['negative'], reply['diagnostics'])
        training._phase('native-targets', make_targets, devices.control_group)
        for candidate in candidates:
            candidate.dummy = not mask[candidate.candidate_index]

        def update(policy, optimizer):
            def fit():
                if policy:
                    return training.fit_h3_update(policy, NativeRollout(policy), candidates,
                        targets=targets, weights=weights, global_kept=sum(mask), optimizer=optimizer, cfg=cfg)
            return training._phase('native-fit', fit, devices.control_group)

        metrics = update(p, opt)
        if p:
            assert metrics['optimizer_step'] and metrics['retained'] == 3 and metrics['replay_slots'] == 2
            assert len(set(training._exchange(_digest((
                {n: v for n, v in p.model.named_parameters() if '.lora_' in n}, opt.state_dict())),
                devices.all_policy_group))) == 1
        state = dict(signature=signature, update=1, prompt_cursor={'cursor': 1},
            node_positions=training._node_positions(1, 2, 8), tracker={}, training_references={})
        path = directory/'update-1.pt'
        with monkeypatch.context() as guard:
            if p:
                guard.setattr(p.model, 'state_dict', lambda *a, **k: pytest.fail('whole-model checkpoint gather'))
            save_h3_checkpoint(path, p, opt, state=state,
                rank_state={'service_run_id': client.run_id, 'service_sequence': client.sequence},
                all_policy_group=devices.all_policy_group, node_policy_group=devices.policy_group,
                topology=signature['topology'], control_group=devices.control_group)
        del guard
        next_rng = torch.rand(4, device=devices.device)
        update(p, opt)
        expected = _digest(({n: v for n, v in p.model.named_parameters() if '.lora_' in n},
                            opt.state_dict())) if p else None
        if p:
            del p, opt, rollout
            torch.cuda.empty_cache()
            p = load_h3_policy(directory/'transformer', 'test-local', devices, rank=2, alpha=4)
            opt = torch.optim.AdamW(p.trainable_parameters, lr=cfg.learning_rate)
        common, rank_state = load_h3_checkpoint(path, p, opt, expected_signature=signature,
            all_policy_group=devices.all_policy_group, node_policy_group=devices.policy_group,
            topology=signature['topology'], control_group=devices.control_group)
        assert common == state and rank_state['service_sequence'] == client.sequence
        assert torch.equal(torch.rand(4, device=devices.device), next_rng)
        update(p, opt)
        if p:
            assert _digest(({n: v for n, v in p.model.named_parameters() if '.lora_' in n}, opt.state_dict())) == expected
        torch.cuda.synchronize(devices.device)
        elapsed = time.monotonic()-started
        evidence = training._exchange(dict(rank=devices.rank, node=devices.node_index, role=devices.role,
            device=torch.cuda.get_device_name(devices.device), elapsed_seconds=elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(devices.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(devices.device)), devices.control_group)
        if devices.rank == 0:
            report = dict(global_candidates=8, accepted=3, nodes=2, policy_per_node=2, reward_per_node=1,
                optimizer_updates_per_fit=1, wall_seconds=max(row['elapsed_seconds'] for row in evidence),
                resources=evidence, full_model_qualified=False)
            report['worker_gpu_hours'] = report['wall_seconds']*6/3600
            report['billed_gpu_hours'] = None  # Controller supplies allocation accounting, including startup/teardown.
            (directory/'result.json').write_text(json.dumps(report, indent=2))
        dist.barrier(group=devices.control_group)
    finally:
        dist.destroy_process_group()
