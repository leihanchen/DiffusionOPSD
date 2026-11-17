"""NVML accounting contracts without CUDA initialization or driver calls."""
from ctypes import c_ulonglong
import io
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from diffusionopsd.video.cosmos_quality import nvml_device_snapshot


@pytest.mark.parametrize('device,uuid,visible,child_device', [
    ('cuda:2', 'owned-uuid', 'GPU-owned-uuid', 'cuda:0'),
    ('cuda:2', 'GPU-owned-uuid', 'GPU-owned-uuid', 'cuda:0'),
    ('cpu', None, '', 'cpu'),
])
def test_judge_child_exposes_only_owned_gpu_by_uuid(
        tmp_path, monkeypatch, device, uuid, visible, child_device):
    from diffusionopsd.video import cosmos_quality

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7,3,4')
    launches = []

    def properties(selected):
        assert selected == torch.device('cuda:2')
        assert uuid is not None, 'CPU judge must not inspect CUDA devices'
        return SimpleNamespace(uuid=uuid)

    monkeypatch.setattr(torch.cuda, 'get_device_properties', properties)
    monkeypatch.setattr(cosmos_quality.IsolatedVideoJudge, '_read', lambda self: {'ready': True})

    def start(command, **kwargs):
        launches.append((command, kwargs['env']))
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(cosmos_quality.subprocess, 'Popen', start)
    judge = cosmos_quality.IsolatedVideoJudge('.', tmp_path, device)
    try:
        command, env = launches[0]
        assert env['CUDA_VISIBLE_DEVICES'] == visible
        assert command[-2:] == ['--device', child_device]
        assert judge.device == torch.device(device)
    finally:
        judge.close()


@pytest.fixture
def pinned_nvml(monkeypatch):
    # Model the verified nvidia-ml-py 13.580.82 namespace independently of
    # whether the optional binding is installed. The generic alias is absent.
    pynvml = SimpleNamespace(
        NVML_VALUE_NOT_AVAILABLE_ulonglong=c_ulonglong(18446744073709551615),
        nvmlInit=lambda: None,
        nvmlDeviceGetComputeRunningProcesses=lambda handle: [],
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=0))
    monkeypatch.setitem(sys.modules, 'pynvml', pynvml)
    handle = object()
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda device: SimpleNamespace(uuid='GPU-owned'))

    def by_uuid(identifier):
        assert identifier == 'GPU-owned'
        return handle

    pynvml.nvmlDeviceGetHandleByUUID = by_uuid
    return pynvml, handle


@pytest.mark.parametrize('process_bytes, device_bytes', [(1024, 2048), (0, 0)])
def test_nvml_snapshot_accepts_pinned_namespace_and_zero_memory(monkeypatch, pinned_nvml,
                                                              process_bytes, device_bytes):
    pynvml, handle = pinned_nvml

    def processes(requested):
        assert requested is handle
        return [SimpleNamespace(pid=42, usedGpuMemory=process_bytes)]

    monkeypatch.setattr(pynvml, 'nvmlDeviceGetComputeRunningProcesses', processes)
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetMemoryInfo', lambda requested: SimpleNamespace(used=device_bytes))
    assert nvml_device_snapshot('cuda:1') == dict(nvml_device_uuid='GPU-owned',
        nvml_device_used_bytes=device_bytes, nvml_process_bytes={'42': process_bytes})
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('representation', ['none', 'raw_sentinel'])
def test_nvml_snapshot_rejects_unavailable_process_accounting(monkeypatch, pinned_nvml, representation):
    pynvml, _ = pinned_nvml
    unavailable = None if representation == 'none' else pynvml.NVML_VALUE_NOT_AVAILABLE_ulonglong.value
    monkeypatch.setattr(pynvml, 'nvmlDeviceGetComputeRunningProcesses',
                        lambda handle: [SimpleNamespace(pid=42, usedGpuMemory=unavailable)])
    snapshot = nvml_device_snapshot('cuda:1')
    assert snapshot == {'nvml_unavailable': 'NVML process memory accounting unavailable'}
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('residency, allocated, reserved', [('cpu', 256, 512), ('cpu', 0, 512), ('cuda:0', 0, 0)])
def test_parent_park_failed_ack_includes_only_structured_release_evidence(residency, allocated, reserved):
    from diffusionopsd.video.cosmos_quality import IsolatedVideoJudge

    judge = IsolatedVideoJudge.__new__(IsolatedVideoJudge)
    judge.residency_protocol = True
    judge.device = torch.device('cuda:0')
    judge.last_release_evidence = None
    judge.process = SimpleNamespace(stdin=io.StringIO(), pid=42)
    snapshots = []
    judge._nvml_snapshot = lambda: snapshots.append('before') or {}

    def reply():
        request = json.loads(judge.process.stdin.getvalue())
        return dict(version=1, request_id=request['request_id'], residency=residency,
                    allocated_bytes=allocated, reserved_bytes=reserved,
                    prompt='private prompt', clip='private payload')

    judge._read = reply
    with pytest.raises(RuntimeError, match='judge did not acknowledge released CUDA storage') as error:
        judge.park()
    evidence = json.loads(str(error.value).partition(': ')[2])
    request = json.loads(judge.process.stdin.getvalue())
    assert evidence == dict(version=1, request_id=request['request_id'], residency=residency,
                            allocated_bytes=allocated, reserved_bytes=reserved)
    assert 'private' not in str(error.value)
    assert snapshots == ['before'] and judge.last_release_evidence is None
    assert not torch.cuda.is_initialized()
