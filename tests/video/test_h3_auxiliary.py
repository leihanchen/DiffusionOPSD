import pytest
import torch

from test_h3_decode import tiny_vae


def make_service():
    from diffusionopsd.video.h3_auxiliary import H3RewardService
    return H3RewardService(tiny_vae(), lambda clip: clip.mean((1, 2, 3, 4)), device='cpu',
        cfg=dict(rho=.1, n_ascent=2, eta=1., reward_frames=3, reward_height=2, reward_width=2))


@pytest.fixture
def service_config_without_checkpoints(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusionopsd.video import estimators, geo_reward, h3_auxiliary, h3_rollout

    vae = tiny_vae()
    model = tmp_path / 'model'
    (model / 'vae').mkdir(parents=True)
    vae.save_config(model / 'vae')
    monkeypatch.setattr(AutoencoderKLMiniMaxH3, 'from_pretrained', lambda *args, **kwargs: vae)
    for name in ('load_depth_anything3', 'load_waft', 'load_dinov2'):
        monkeypatch.setattr(estimators, name, lambda device: None)
    monkeypatch.setattr(geo_reward, 'GeoReward', lambda *args: lambda clip: clip.mean((1, 2, 3, 4)))
    # Preload imports before a test removes repository entries from sys.path.
    assert callable(h3_rollout.vae_descriptor)
    return h3_auxiliary, SimpleNamespace(model=str(model)), SimpleNamespace(
        role='reward', rank=2, device=torch.device('cpu'))


def test_service_finds_source_worker_with_remapped_package_and_unrelated_cwd(
        tmp_path, monkeypatch, service_config_without_checkpoints):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from diffusionopsd.video import cosmos_quality

    auxiliary, cfg, devices = service_config_without_checkpoints
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(auxiliary, '__file__', '/opt/venv/lib/python3.12/site-packages/diffusionopsd/video/h3_auxiliary.py')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', [str(repo), *sys.path])
    monkeypatch.setenv('PYTHONPATH', '/h3-overlay:' + str(repo))
    launches = []

    def start(command, **kwargs):
        launches.append((command, kwargs['env']))
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(cosmos_quality.subprocess, 'Popen', start)
    monkeypatch.setattr(cosmos_quality.IsolatedVideoJudge, '_read', lambda self: {'ready': True})
    service = auxiliary.H3RewardService.from_config(cfg, devices, tmp_path / 'run')
    try:
        command, env = launches[0]
        assert Path(command[2]) == repo / 'scripts/cosmos_quality_worker.py'
        assert Path(command[2]).is_file()
        assert env['PYTHONPATH'] == str(repo)
        assert env['PYTHONNOUSERSITE'] == '1'
        assert command[3:] == ['--device', 'cpu', '--residency-protocol']
        assert service.judge.residency_protocol
    finally:
        service.judge.close()


def test_service_reports_missing_source_worker_before_model_loading(
        tmp_path, monkeypatch, service_config_without_checkpoints):
    import sys
    from diffusers import AutoencoderKLMiniMaxH3

    auxiliary, cfg, devices = service_config_without_checkpoints
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', [str(tmp_path)])

    def unexpected_load(*args, **kwargs):
        pytest.fail('worker discovery must fail before checkpoint loading')

    monkeypatch.setattr(AutoencoderKLMiniMaxH3, 'from_pretrained', unexpected_load)
    with pytest.raises(FileNotFoundError, match='cosmos_quality_worker.py'):
        auxiliary.H3RewardService.from_config(cfg, devices, tmp_path / 'run')


def test_nested_real_service_preserves_enclosing_phase_allocator_peaks(monkeypatch):
    from diffusionopsd.video import h3_auxiliary as auxiliary
    peaks, resets = [0, 0], []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    def reset(device):
        resets.append(str(device))
        peaks[:] = [0, 0]
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', reset)
    monkeypatch.setattr(auxiliary, 'memory_snapshot', lambda device: dict(
        peak_allocated_bytes=peaks[0], peak_reserved_bytes=peaks[1],
        nvml_device_used_bytes=123, nvml_process_bytes={'17': 42}))
    service = make_service()
    service.device = torch.device('cuda:0')  # counters are simulated; no GPU allocation
    service.decode01 = lambda video: video
    def reward(video):
        peaks[:] = [max(peaks[0], 7), max(peaks[1], 17)]
        return torch.ones(1)
    service.reward = reward
    replies = []
    # Same enclosing profiler used by the trainer's phase(...) wrapper.
    with auxiliary.H3PhaseProfiler('cuda:0', 'evaluation') as outer:
        peaks[:] = [99, 199]
        for index in range(2):
            replies.append(service.dispatch('score_endpoint', {'video': torch.ones(1, 3, 3, 2, 2)},
                {'prompt': 'caption', 'request_id': str(index)}))
    assert outer.metrics['peak_allocated_bytes'] == 99
    assert outer.metrics['peak_reserved_bytes'] == 199
    assert resets == ['cuda:0']
    for reply in replies:
        assert reply['metrics']['allocator_peak_scope'] == 'enclosing phase evaluation'
        assert reply['metrics']['nvml_device_peak_used_bytes'] == 123
        assert reply['metrics']['nvml_pid_peak_used_bytes'] == {'17': 42}


def test_reward_service_matches_local_targets(tmp_path):
    from diffusionopsd.video.h3_auxiliary import H3RewardClient
    from diffusionopsd.video.opa_video import opa_tr_step_nd
    service = make_service()
    x = torch.randn(1, 2, 3, 2, 2)
    got = service.construct_targets(x)
    expected = opa_tr_step_nd(x, service.reward_values, .1, 2, 1., 1.)
    torch.testing.assert_close(got.positive, expected)
    assert service.reward_values(got.positive) > service.reward_values(x)
    assert service.reward_values(got.negative) < service.reward_values(x)
    for target in (got.positive, got.negative):
        assert target.dtype == torch.float32 and not target.requires_grad
        assert (target-x).norm() <= .1*x.norm() + 1e-6
    client = H3RewardClient(service=service, run_id='test')
    service.run_dir = tmp_path
    reply = client.request('construct_targets', 'node0/update0/candidate0', {'video': x}, {'accepted': True})
    torch.testing.assert_close(reply['positive'], expected)
    assert reply['metrics']['payload_bytes'] == x.numel()*4
    import json
    phase = json.loads((tmp_path / 'phases.jsonl').read_text().splitlines()[-1])
    assert phase['candidate_id'] == 'node0/update0/candidate0'
    assert phase['request_id'] == reply['request_id'] and phase['latent_geometry'] == list(x.shape)
    assert phase['tiling_mode'] == 'tiled' and phase['service_pid'] > 0
    with pytest.raises(ValueError, match='accepted'):
        client.request('construct_targets', 'rejected', {'video': x}, {})


def test_request_identity_and_timeout():
    from diffusionopsd.video.h3_auxiliary import H3RewardClient
    client = H3RewardClient(service=make_service(), run_id='test', max_payload_bytes=16)
    with pytest.raises(ValueError, match='payload'):
        client.request('score_endpoint', 'a', {'video': torch.ones(1, 2, 3, 2, 2)}, {'prompt': 'p'})
    client = H3RewardClient(service=make_service(), run_id='test', timeout_s=.0000001)
    with pytest.raises(TimeoutError):
        client.request('score_endpoint', 'a', {'video': torch.ones(1, 2, 3, 2, 2)}, {'prompt': 'p'})


def test_service_rejects_concurrent_graph_and_keeps_default_reward_clip():
    service = make_service()
    service.cfg = None
    clip = service.reward_clip(torch.rand(1, 12, 3, 2, 2))
    assert clip.shape == (1, 9, 3, 224, 384)
    service._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='one in-flight'):
            service.dispatch('decode_evaluation', {}, {})
    finally:
        service._lock.release()


def test_legacy_judge_score_protocol(tmp_path):
    from diffusionopsd.video.cosmos_quality import IsolatedVideoJudge
    worker = tmp_path/'fake.py'
    worker.write_text('import json,sys\nprint(json.dumps({"ready":True}),flush=True)\n'
        'for line in sys.stdin:\n r=json.loads(line); print(json.dumps({"score":{"VQ":1}}),flush=True)\n')
    judge = IsolatedVideoJudge('.', tmp_path, 'cpu', worker_path=worker)
    try:
        assert judge.score(torch.ones(3, 3, 2, 2), 'p') == {'VQ': 1}
    finally:
        judge.close()


def test_judge_park_acknowledges_release(tmp_path):
    from diffusionopsd.video.cosmos_quality import IsolatedVideoJudge
    worker = tmp_path/'fake.py'
    worker.write_text('import json,sys\nprint(json.dumps({"ready":True}),flush=True)\n'
        'for line in sys.stdin:\n r=json.loads(line); print(json.dumps(dict(version=1,request_id=r["request_id"],'
        'residency="cpu",allocated_bytes=0,reserved_bytes=0)),flush=True)\n')
    judge = IsolatedVideoJudge('.', tmp_path, 'cpu', residency_protocol=True, worker_path=worker)
    try:
        assert judge.park()['reserved_bytes'] == 0
    finally:
        judge.close()


def test_real_worker_dispatch_legacy_and_park(tmp_path, monkeypatch):
    from scripts.cosmos_quality_worker import dispatch

    def forbidden(*args, **kwargs):
        pytest.fail('CPU park and legacy scoring must not access CUDA cleanup/accounting')

    for name in ('synchronize', 'empty_cache', 'memory_allocated', 'memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch._C, '_cuda_clearCublasWorkspaces', forbidden, raising=False)

    class Judge:
        def __init__(self):
            self.device = 'cpu'

        def to(self, device):
            self.device = device

        def score(self, clip, prompt):
            return dict(VQ=float(clip.mean()), prompt=prompt)

    path = tmp_path / 'clip.pt'
    torch.save(torch.ones(3, 3, 2, 2), path)
    judge = Judge()
    request = dict(clip=str(path), prompt='explicit prompt')
    assert dispatch(judge, request, 'cpu') == {'score': {'VQ': 1., 'prompt': 'explicit prompt'}}
    reply = dispatch(judge, dict(version=1, command='park', request_id='r'), 'cpu', True)
    assert reply == dict(version=1, request_id='r', residency='cpu', allocated_bytes=0, reserved_bytes=0)
    assert judge.device == 'cpu'


def test_real_worker_park_releases_live_blas_workspaces_before_empty_cache(monkeypatch):
    import gc
    from scripts.cosmos_quality_worker import dispatch

    events = []
    active = dict(model=4096, cublas=512, cublaslt=1024)
    inactive = [8192]

    class Judge:
        def to(self, device):
            assert device == 'cpu'
            events.append('cpu')
            inactive[0] += active.pop('model')

    def clear():
        events.append('clear')
        # Both handle/stream workspace maps own live allocator storage until
        # the pinned clear hook destroys their DataPtrs.
        inactive[0] += sum(active.values())
        active.clear()

    def empty_cache():
        events.append('empty_cache')
        inactive[0] = 0  # Active workspace storage deliberately survives.

    def allocated(device):
        events.append('allocated')
        return sum(active.values())

    def reserved(device):
        events.append('reserved')
        return sum(active.values()) + inactive[0]

    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: events.append('sync'))
    monkeypatch.setattr(gc, 'collect', lambda: events.append('gc'))
    monkeypatch.setattr(torch._C, '_cuda_clearCublasWorkspaces', clear, raising=False)
    monkeypatch.setattr(torch.cuda, 'empty_cache', empty_cache)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', allocated)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', reserved)
    reply = dispatch(Judge(), dict(version=1, command='park', request_id='workspace'), 'cuda:0', True)
    assert reply == dict(version=1, request_id='workspace', residency='cpu', allocated_bytes=0, reserved_bytes=0)
    assert events == ['sync', 'cpu', 'gc', 'clear', 'sync', 'empty_cache', 'allocated', 'reserved']
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('capability', ['missing', 'raises'])
def test_real_worker_park_reports_workspace_cleanup_failure_with_request_identity(monkeypatch, capability):
    import gc
    from types import SimpleNamespace
    from scripts.cosmos_quality_worker import dispatch

    events = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: events.append('sync'))
    monkeypatch.setattr(gc, 'collect', lambda: events.append('gc'))

    def forbidden(*args, **kwargs):
        pytest.fail('failed workspace cleanup must not acknowledge or measure release')

    for name in ('empty_cache', 'memory_allocated', 'memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    if capability == 'missing':
        monkeypatch.delattr(torch._C, '_cuda_clearCublasWorkspaces', raising=False)
        expected = '_cuda_clearCublasWorkspaces'
    else:
        def fail():
            raise RuntimeError('workspace release failed')

        monkeypatch.setattr(torch._C, '_cuda_clearCublasWorkspaces', fail, raising=False)
        expected = 'workspace release failed'
    judge = SimpleNamespace(to=lambda device: events.append(device))
    reply = dispatch(judge, dict(version=1, command='park', request_id='failed-workspace'), 'cuda:0', True)
    assert reply['version'] == 1 and reply['request_id'] == 'failed-workspace'
    assert expected in reply['error']
    assert 'residency' not in reply and events == ['sync', 'cpu', 'gc']
    assert not torch.cuda.is_initialized()


def test_external_judge_release_requires_observed_pid_budget():
    from diffusionopsd.video.cosmos_quality import verify_judge_release
    before = dict(nvml_device_uuid='GPU-test', nvml_process_bytes={'17': 4*1024**3})
    after = dict(nvml_device_uuid='GPU-test', nvml_process_bytes={'17': 300*1024**2})
    result = verify_judge_release(before, after, 17, baseline_bytes=0, context_allowance_bytes=512*1024**2)
    assert result['verified'] and result['released_bytes'] > 3*1024**3
    assert result['pid'] == 17 and result['after_bytes'] == 300*1024**2
    assert not verify_judge_release(before, before, 17, 0, 512*1024**2)['verified']
    assert not verify_judge_release(before, {'nvml_unavailable': 'missing'}, 17, 0, 0)['verified']
    assert not verify_judge_release(before, dict(after, nvml_device_uuid='other'), 17, 0, 0)['verified']


@pytest.mark.parametrize('post_bytes', [300*1024**2, 3*1024**3, None])
def test_parent_park_enforces_external_child_evidence(tmp_path, monkeypatch, post_bytes):
    from diffusionopsd.video.cosmos_quality import IsolatedVideoJudge
    worker = tmp_path / 'fake.py'
    worker.write_text('import json,sys\nprint(json.dumps({"ready":True}),flush=True)\n'
        'for line in sys.stdin:\n r=json.loads(line); print(json.dumps(dict(version=1,request_id=r["request_id"],'
        'residency="cpu",allocated_bytes=0,reserved_bytes=0)),flush=True)\n')
    judge = IsolatedVideoJudge('.', tmp_path, 'cpu', residency_protocol=True, worker_path=worker)
    pid = str(judge.process.pid)
    judge.device = torch.device('cuda:0')
    judge.parked_baseline = dict(nvml_device_uuid='GPU-test', nvml_process_bytes={})
    snapshots = iter([dict(nvml_device_uuid='GPU-test', nvml_process_bytes={pid: 4*1024**3}),
        dict(nvml_device_uuid='GPU-test', nvml_process_bytes={pid: post_bytes})
        if post_bytes is not None else {'nvml_unavailable': 'missing'}])
    monkeypatch.setattr(judge, '_nvml_snapshot', lambda: next(snapshots))
    try:
        if post_bytes == 300*1024**2:
            assert judge.park()['external_release']['verified']
        else:
            with pytest.raises(RuntimeError, match='external CUDA release unverified'):
                judge.park()
            assert not judge.last_release_evidence['verified']
        assert judge.last_release_evidence['pid'] == judge.process.pid
    finally:
        judge.close()


def test_profiler_preserves_judge_peak_after_park(monkeypatch):
    import diffusionopsd.video.h3_auxiliary as auxiliary
    snapshots = iter([
        dict(nvml_device_used_bytes=900, nvml_process_bytes={'10': 200, '17': 600}),
        dict(nvml_device_used_bytes=300, nvml_process_bytes={'10': 200, '17': 20})])
    monkeypatch.setattr(auxiliary, 'memory_snapshot', lambda device: next(snapshots))
    profiler = auxiliary.H3PhaseProfiler('cpu', 'score_endpoint', interval_s=100.,
        metadata=dict(service_pid=10, judge_pid=17, candidate_id='candidate0',
                      latent_geometry=[1, 2, 3, 4, 4], tiling_mode='tiled'))
    with profiler:
        import time
        deadline = time.monotonic()+1
        while not profiler.samples and time.monotonic() < deadline:
            time.sleep(.001)
    assert profiler.metrics['nvml_pid_peak_used_bytes'] == {'10': 200, '17': 600}
    assert profiler.metrics['nvml_device_peak_used_bytes'] == 900
    assert profiler.metrics['judge_pid'] == 17 and profiler.metrics['candidate_id'] == 'candidate0'
    assert profiler.metrics['sample_interval_s'] == 100 and profiler.metrics['nvml_samples'] == 2


def test_diagnostic_error_preserves_service_error_and_releases_lock(monkeypatch):
    import diffusionopsd.video.h3_auxiliary as auxiliary
    service = make_service()
    monkeypatch.setattr(auxiliary.H3PhaseProfiler, '__exit__', lambda *args: (_ for _ in ()).throw(OSError('metrics')))
    with pytest.raises(ValueError, match='unknown request'):
        service.dispatch('unknown', {}, {})
    assert not service._lock.locked()


def _gloo_worker(rank, path, stall=False):
    from datetime import timedelta
    import torch.distributed as dist
    from diffusionopsd.video.h3_auxiliary import H3RewardClient
    from diffusionopsd.video.h3_fsdp import H3Devices
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=3,
                            timeout=timedelta(seconds=20))
    try:
        devices = H3Devices(rank, 'reward' if rank == 2 else 'policy', 2, (2,), torch.device('cpu'),
                            node_control_group=dist.group.WORLD)
        if stall:
            import time
            client = H3RewardClient(devices, run_id='gloo-timeout', timeout_s=.2)
            if rank == 0:
                time.sleep(1)
            with pytest.raises((RuntimeError, TimeoutError)):
                client.request('score_endpoint', 'stall', {}, dict(owner_rank=0, candidate_index=0))
            assert client.failed
            return
        # Only the reward process constructs a model. Policy groups are absent.
        service = make_service() if rank == 2 else None
        client = H3RewardClient(devices, service, run_id='gloo-test', timeout_s=15.)
        x = torch.full((1, 2, 3, 2, 2), .2)
        reply = client.request('construct_targets', 'node0/update0/candidate0',
            {'video': x} if rank == 0 else {}, dict(owner_rank=0, candidate_index=0, accepted=True))
        assert reply['candidate_id'] == 'node0/update0/candidate0'
        if rank == 0:
            assert reply['positive'].dtype == torch.float32 and not reply['positive'].requires_grad
            assert reply['positive'].shape == x.shape
            assert (reply['positive']-x).norm() <= .1*x.norm() + 1e-6
        else:
            assert 'positive' not in reply
        # Service rejects an unaccepted candidate and all ranks see the error.
        with pytest.raises(RuntimeError, match='accepted'):
            client.request('construct_targets', 'node0/update0/candidate1',
                {'video': x} if rank == 1 else {}, dict(owner_rank=1, candidate_index=1))
        assert client.failed
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('stall', [False, True])
def test_gloo_targets_and_coordinated_service_error(tmp_path, stall):
    import torch.multiprocessing as mp
    context = mp.start_processes(_gloo_worker, args=(str(tmp_path / 'rendezvous'), stall),
                                 nprocs=3, start_method='spawn', join=False)
    import time
    deadline = time.monotonic() + 50
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError('Gloo integration process deadline')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
