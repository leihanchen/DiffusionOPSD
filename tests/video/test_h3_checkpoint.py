"""Adapter-only publication, exact replay, and fail-closed resume."""
from copy import deepcopy
from types import SimpleNamespace
from datetime import timedelta
import json
import os
import random
import signal

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusionopsd.video.h3_fsdp import H3Devices, H3Policy
from diffusionopsd.video.h3_conditioning import validate_h3_topology


def signature(nodes=1, policies=2):
    cfg = SimpleNamespace(num_nodes=nodes, policy_gpus=list(range(policies)), target_gpus=[policies],
        train_pool_size=8, workflow='t2va', candidates_per_node=4, clips_per_prompt=4,
        policy_microbatch_size=1, eval_count=8)
    return dict(identity=dict(model='pinned', reward='pinned', runtime='pinned'), manifest_sha256='cache',
                geometry=dict(height=16), topology=validate_h3_topology(cfg), training=dict(updates=2))


def policy():
    model = torch.nn.Module()
    model.register_parameter('frozen', torch.nn.Parameter(torch.ones(300), requires_grad=False))
    model.add_module('lora_A', torch.nn.ModuleDict({
        'default': torch.nn.Linear(2, 1, bias=False), 'old': torch.nn.Linear(2, 1, bias=False)}))
    model.lora_A.old.weight.requires_grad_(False)
    # Match native PEFT's dotted names, including the leading attention module.
    outer = torch.nn.Module()
    outer.add_module('attention', model)
    return H3Policy(outer, H3Devices.cpu())


def step(p, opt):
    opt.zero_grad()
    x = torch.randn(1, 2) + random.random() + float(np.random.rand())
    p.model.attention.lora_A.default(x).sum().backward()
    opt.step()
    p.ema_(.99)


def test_atomic_resume_matches_uninterrupted(tmp_path, monkeypatch):
    from diffusionopsd.video.h3_checkpoint import save_h3_checkpoint, load_h3_checkpoint
    torch.manual_seed(4)
    p = policy()
    opt = torch.optim.AdamW(p.trainable_parameters)
    step(p, opt)
    state = dict(signature=signature(), update=1, prompt_cursor={'offset': 1}, tracker={'A': [1.]},
                 training_references={'A': {'scores': [3.]}}, last_evaluation_update=1,
                 experiment_group='same-group')
    path = tmp_path/'update-000001.pt'
    def forbid_whole_model(*args, **kwargs):
        raise AssertionError('whole model state_dict forbidden')
    monkeypatch.setattr(p.model, 'state_dict', forbid_whole_model)
    save_h3_checkpoint(path, p, opt, state=state, rank_state={'service_sequence': 12},
                       policy_group=None, control_group=None)
    step(p, opt)
    expected = deepcopy(opt.state_dict())
    expected_weights = {n: v.clone() for n, v in p.model.named_parameters()}
    candidate = torch.randn(2, generator=torch.Generator().manual_seed(108))
    restored = policy()
    restored_opt = torch.optim.AdamW(restored.trainable_parameters)
    common, rank = load_h3_checkpoint(tmp_path/'latest.json', restored, restored_opt,
        expected_signature=signature(), policy_group=None, control_group=None)
    assert common == state and rank['service_sequence'] == 12
    step(restored, restored_opt)
    for n, v in restored.model.named_parameters():
        assert torch.equal(v, expected_weights[n])
    for key, values in restored_opt.state_dict()['state'].items():
        for name, value in values.items():
            assert torch.equal(value, expected['state'][key][name])
    assert torch.equal(candidate, torch.randn(2, generator=torch.Generator().manual_seed(108)))
    payload = torch.load(path, weights_only=False)
    assert all('lora_' in n for n in payload['adapters'])
    assert 'frozen' not in str(payload.keys())


@pytest.mark.parametrize('field', ['identity', 'manifest_sha256', 'geometry', 'training', 'topology'])
def test_reject_identity_changes(tmp_path, field):
    from diffusionopsd.video.h3_checkpoint import save_h3_checkpoint, load_h3_checkpoint
    p = policy()
    opt = torch.optim.AdamW(p.trainable_parameters)
    path = tmp_path/'update.pt'
    save_h3_checkpoint(path, p, opt, state={'signature': signature(), 'update': 1}, rank_state={},
                       policy_group=None, control_group=None)
    wrong = signature()
    wrong[field] = {'changed': True}
    with pytest.raises(RuntimeError, match='signature|topology'):
        load_h3_checkpoint(path, p, opt, expected_signature=wrong, policy_group=None, control_group=None)


def test_torn_write_and_stale_pointer(tmp_path):
    from diffusionopsd.video.h3_checkpoint import save_h3_checkpoint, load_h3_checkpoint
    p = policy()
    opt = torch.optim.AdamW(p.trainable_parameters)
    path = tmp_path/'update.pt'
    save_h3_checkpoint(path, p, opt, state={'signature': signature(), 'update': 1}, rank_state={},
                       policy_group=None, control_group=None)
    path.write_bytes(b'torn')
    with pytest.raises(RuntimeError, match='digest|integrity'):
        load_h3_checkpoint(tmp_path/'latest.json', p, opt, expected_signature=signature(),
                           policy_group=None, control_group=None)


@pytest.mark.parametrize('nodes', [1, 2])
@pytest.mark.parametrize('policies', [2, 4])
def test_shared_topology_dictionary(nodes, policies):
    topology = signature(nodes, policies)['topology']
    assert topology['node_role_maps'] == [dict(node_index=node, policy=list(range(policies)), target=[policies])
                                          for node in range(nodes)]
    assert validate_h3_topology(topology) == topology
    for key in ('num_nodes', 'policy_gpus', 'target_gpus', 'policy_ranks', 'reward_ranks',
                'candidates_per_node', 'policy_microbatch_size', 'device_role_map', 'node_role_maps'):
        wrong = deepcopy(topology)
        wrong[key] = None
        with pytest.raises(ValueError, match='topology'):
            validate_h3_topology(wrong)


def test_stale_latest_pointer(tmp_path):
    from diffusionopsd.video.h3_checkpoint import save_h3_checkpoint, load_h3_checkpoint
    p = policy()
    opt = torch.optim.AdamW(p.trainable_parameters)
    path = tmp_path/'update.pt'
    save_h3_checkpoint(path, p, opt, state={'signature': signature(), 'update': 1}, rank_state={},
                       policy_group=None, control_group=None)
    latest = tmp_path/'latest.json'
    pointer = json.loads(latest.read_text())
    pointer['update'] = 0
    latest.write_text(json.dumps(pointer))
    with pytest.raises(RuntimeError, match='stale'):
        load_h3_checkpoint(latest, p, opt, expected_signature=signature(), policy_group=None, control_group=None)


@pytest.mark.parametrize('termination_signal', [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1])
def test_safe_boundary_reserves_publication_and_restores_signals(termination_signal):
    from diffusionopsd.video.h3_checkpoint import H3SafeBoundary
    clock = [0.]
    boundary = H3SafeBoundary(walltime_seconds=100, publication_reserve_seconds=10, clock=lambda: clock[0])
    assert not boundary.should_stop(20, 5, None)
    clock[0] = 55
    assert boundary.should_stop(20, 20, None)
    boundary.walltime_seconds = 0
    previous = signal.getsignal(termination_signal)
    with boundary.signals():
        os.kill(os.getpid(), termination_signal)
        assert boundary.should_stop(0, 0, None)
    assert signal.getsignal(termination_signal) is previous


def test_failed_publication_preserves_previous_latest(tmp_path, monkeypatch):
    from diffusionopsd.video import h3_checkpoint as checkpoint
    p = policy()
    opt = torch.optim.AdamW(p.trainable_parameters)
    def save(update):
        checkpoint.save_h3_checkpoint(tmp_path/f'update-{update}.pt', p, opt,
            state={'signature': signature(), 'update': update}, rank_state={}, policy_group=None, control_group=None)
    save(1)
    pointer = (tmp_path/'latest.json').read_bytes()
    atomic = checkpoint._atomic_bytes
    def fail_pointer(path, data):
        if path.name == 'latest.json':
            raise OSError('injected pointer publication failure')
        return atomic(path, data)
    monkeypatch.setattr(checkpoint, '_atomic_bytes', fail_pointer)
    with pytest.raises(RuntimeError, match='publication failure'):
        save(2)
    assert (tmp_path/'latest.json').read_bytes() == pointer
    state, _ = checkpoint.load_h3_checkpoint(tmp_path/'latest.json', p, opt,
        expected_signature=signature(), policy_group=None, control_group=None)
    assert state['update'] == 1


@pytest.mark.parametrize('latest', [False, True])
def test_first_snapshot_preserves_publication_timing(tmp_path, monkeypatch, latest):
    from diffusionopsd.video import h3_checkpoint as checkpoint
    ticks = iter([10., 12.5])
    monkeypatch.setattr(checkpoint, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    p = policy()
    opt = torch.optim.AdamW(p.trainable_parameters)
    path = tmp_path/'update.pt'
    checkpoint.save_h3_checkpoint(path, p, opt, state={
        'signature': signature(), 'update': 1, 'longest_publication_seconds': 0.},
        rank_state={}, policy_group=None, control_group=None)
    restored, _ = checkpoint.load_h3_checkpoint(tmp_path/'latest.json' if latest else path, p, opt,
        expected_signature=signature(), policy_group=None, control_group=None)
    assert restored['longest_publication_seconds'] == 2.5


def test_rng_accesses_only_owned_cuda_device(monkeypatch):
    from diffusionopsd.video.h3_checkpoint import _rng_state, _restore_rng
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 3)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda device: calls.append(('get', device)) or torch.ones(4))
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda value, device: calls.append(('set', device)))
    def foreign_access():
        raise AssertionError('foreign device RNG access')
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', foreign_access)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', foreign_access)
    _restore_rng(_rng_state())
    assert calls == [('get', 3), ('set', 3)]


def _distributed_worker(rank, count, rendezvous, directory, divergent):
    from pathlib import Path
    from diffusionopsd.video.h3_checkpoint import save_h3_checkpoint, load_h3_checkpoint
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=count+1,
                            timeout=timedelta(seconds=20))
    policy_group = dist.new_group(list(range(count)), timeout=timedelta(seconds=20))
    try:
        torch.manual_seed(9)
        p = policy() if rank < count else None
        devices = H3Devices(rank, 'policy' if rank < count else 'reward', count, (count,), torch.device('cpu'),
            policy_group=policy_group, all_policy_group=policy_group, control_group=dist.group.WORLD,
            policy_index=rank if rank < count else -1)
        if p:
            p.devices = devices
        opt = torch.optim.AdamW(p.trainable_parameters) if p else None
        # Replicated optimization draws must match; caller RNG diverges afterwards.
        if p:
            random.seed(10)
            np.random.seed(10)
            step(p, opt)
        torch.manual_seed(40+rank)
        random.seed(40+rank)
        np.random.seed(40+rank)
        state = dict(signature=signature(policies=count), update=1, prompt_cursor={'offset': 1},
                     tracker={'record': [2.]}, training_references={'record': {'scores': [3.]}},
                     last_evaluation_update=1, experiment_group='group')
        path = Path(directory)/'distributed.pt'
        if divergent and rank == 1:
            with torch.no_grad():
                p.trainable_parameters[0].add_(1.)
        if divergent:
            with pytest.raises(RuntimeError, match='divergent'):
                save_h3_checkpoint(path, p, opt, state=state, rank_state={},
                                   policy_group=policy_group, control_group=dist.group.WORLD)
            return
        save_h3_checkpoint(path, p, opt, state=state, rank_state={'service_sequence': rank+20, 'run_id': 'stable'},
                           policy_group=policy_group, control_group=dist.group.WORLD)
        next_rng = (torch.rand(3), random.random(), float(np.random.rand()))
        from diffusionopsd.video.h3_training import collect_candidates
        class SeedRollout:
            def rollout(self, conditioning, *, seed, geometry, adapter):
                return torch.randn(2, generator=torch.Generator().manual_seed(seed))
        def next_candidates(policy, common):
            attempt = common['prompt_cursor']['offset']
            return collect_candidates(policy, SeedRollout(), dict(conditioning={}, geometry={}, attempt=attempt,
                seed=100+attempt*4), rank=rank, policy_count=count)
        expected_candidates = next_candidates(p, state)
        if p:
            step(p, opt)
            expected_params = {n: v.clone() for n, v in p.model.named_parameters()}
            expected_opt = deepcopy(opt.state_dict())
            # Fresh policy reconstructs the frozen backbone outside checkpoint loading.
            restored = policy()
            restored.devices = devices
            restored_opt = torch.optim.AdamW(restored.trainable_parameters)
        else:
            restored = restored_opt = None
        common, caller = load_h3_checkpoint(path, restored, restored_opt, expected_signature=state['signature'],
                                           policy_group=policy_group, control_group=dist.group.WORLD)
        assert common == state
        assert caller == {'service_sequence': rank+20, 'run_id': 'stable'}
        assert torch.equal(torch.rand(3), next_rng[0])
        assert random.random() == next_rng[1] and float(np.random.rand()) == next_rng[2]
        resumed_candidates = next_candidates(restored, common)
        assert [c.candidate_id for c in resumed_candidates] == [c.candidate_id for c in expected_candidates]
        for resumed, expected in zip(resumed_candidates, expected_candidates):
            assert torch.equal(resumed.query, expected.query)
        if restored:
            step(restored, restored_opt)
            for name, value in restored.model.named_parameters():
                assert torch.equal(value, expected_params[name])
            for key, values in restored_opt.state_dict()['state'].items():
                for name, value in values.items():
                    assert torch.equal(value, expected_opt['state'][key][name])
        # Service rejoins the same readiness/completion exchanges used by evaluation.
        from diffusionopsd.video.h3_training import _phase
        assert _phase('evaluation_ready', lambda: rank, dist.group.WORLD) == rank
        assert _phase('evaluation_complete', lambda: rank, dist.group.WORLD) == rank
        with pytest.raises(RuntimeError, match='signature'):
            load_h3_checkpoint(path, restored, restored_opt, expected_signature=signature(nodes=2, policies=count),
                               policy_group=policy_group, control_group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('count', [2, 4])
@pytest.mark.parametrize('divergent', [False, True])
def test_distributed_exact_resume_with_service(tmp_path, count, divergent):
    context = mp.spawn(_distributed_worker, args=(count, str(tmp_path/'rdzv'), str(tmp_path), divergent),
                       nprocs=count+1, join=False)
    try:
        import time
        deadline = time.monotonic()+90
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('checkpoint workers exceeded 90-second deadline')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)


@pytest.mark.h3_gpu
@pytest.mark.skipif(os.environ.get('H3_RUN_GPU_TESTS') != '1', reason='opt-in torchrun n=3/n=5 CUDA checkpoint')
def test_native_cuda_shards_checkpoint_roundtrip(monkeypatch):
    """Run under torchrun --nproc-per-node=3 or 5 with H3_TEST_CHECKPOINT set."""
    from pathlib import Path
    from test_h3_fsdp import tiny_model
    from diffusionopsd.video.h3_fsdp import init_h3_distributed, load_h3_policy
    from diffusionopsd.video.h3_checkpoint import save_h3_checkpoint, load_h3_checkpoint
    count = int(os.environ['LOCAL_WORLD_SIZE'])-1
    assert count in (2, 4)
    devices = init_h3_distributed(policy_gpus=tuple(range(count)), target_gpus=(count,))
    directory = Path(os.environ['H3_TEST_CHECKPOINT'])/'resume'
    try:
        if devices.rank == 0:
            directory.mkdir(parents=True, exist_ok=True)
            tiny_model().save_pretrained(directory/'transformer')
        dist.barrier(group=devices.control_group)
        torch.manual_seed(12)
        p = load_h3_policy(directory/'transformer', 'test-local', devices, rank=2, alpha=4) if devices.role == 'policy' else None
        opt = torch.optim.AdamW(p.trainable_parameters) if p else None
        def update():
            if p:
                opt.zero_grad()
                sum(value.square().sum() for value in p.trainable_parameters).backward()
                opt.step()
                p.ema_(.99)
        update()
        def forbid_state_dict(*args, **kwargs):
            raise AssertionError('whole frozen model state_dict forbidden')
        if p:
            monkeypatch.setattr(p.model, 'state_dict', forbid_state_dict)
        save_h3_checkpoint(directory/'update-1.pt', p, opt,
            state={'signature': signature(policies=count), 'update': 1}, rank_state={'service_sequence': 9},
            policy_group=devices.all_policy_group, control_group=devices.control_group)
        expected_rng = torch.rand(4, device=devices.device)
        update()
        expected = {n: v.detach().cpu().clone() for n, v in p.model.named_parameters() if '.lora_' in n} if p else None
        if p:
            del p, opt
            p = load_h3_policy(directory/'transformer', 'test-local', devices, rank=2, alpha=4)
            opt = torch.optim.AdamW(p.trainable_parameters)
        _, caller = load_h3_checkpoint(directory/'latest.json', p, opt, expected_signature=signature(policies=count),
            policy_group=devices.all_policy_group, control_group=devices.control_group)
        assert caller['service_sequence'] == 9
        assert torch.equal(torch.rand(4, device=devices.device), expected_rng)
        update()
        if p:
            for name, parameter in p.model.named_parameters():
                if '.lora_' in name:
                    assert torch.equal(parameter.detach().cpu(), expected[name])
        dist.barrier(group=devices.control_group)
    finally:
        dist.destroy_process_group()
