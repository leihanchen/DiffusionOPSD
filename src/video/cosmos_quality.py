"""Keep the released Qwen2-VL reward weights on their validated dependencies."""

from __future__ import annotations

import json
import uuid
import os
from pathlib import Path
import selectors
import subprocess
import sys

import torch


def nvml_device_snapshot(device):
    """External allocations by PID on the scheduler-visible CUDA device."""
    if torch.device(device).type != 'cuda':
        return {'nvml_unavailable': 'CPU device; external CUDA release is not applicable'}
    try:
        import pynvml
        pynvml.nvmlInit()
        identifier = str(torch.cuda.get_device_properties(device).uuid)
        handle = pynvml.nvmlDeviceGetHandleByUUID(identifier)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if any(p.usedGpuMemory is None or p.usedGpuMemory == pynvml.NVML_VALUE_NOT_AVAILABLE_ulonglong.value
               for p in processes):
            raise RuntimeError('NVML process memory accounting unavailable')
        return dict(nvml_device_uuid=identifier,
            nvml_device_used_bytes=pynvml.nvmlDeviceGetMemoryInfo(handle).used,
            nvml_process_bytes={str(p.pid): p.usedGpuMemory for p in processes})
    except Exception as exc:
        return {'nvml_unavailable': str(exc)}


def verify_judge_release(before, after, pid, baseline_bytes, context_allowance_bytes):
    """Require child use below its parked baseline plus a bounded CUDA context allowance.

    The allowance covers driver/context memory that survives empty_cache. It is
    an explicit upper bound, not an assertion that context memory was measured.
    Task 8 must qualify this bound on the target hardware.
    """
    evidence = dict(pid=pid, before=before, after=after, baseline_bytes=baseline_bytes,
                    context_allowance_bytes=context_allowance_bytes, verified=False)
    if ('nvml_unavailable' in before or 'nvml_unavailable' in after or baseline_bytes is None
            or not before.get('nvml_device_uuid') or before.get('nvml_device_uuid') != after.get('nvml_device_uuid')):
        return dict(evidence, reason='external process/device evidence unavailable or mismatched')
    pre = before['nvml_process_bytes'].get(str(pid), 0)
    post = after['nvml_process_bytes'].get(str(pid), 0)
    ceiling = baseline_bytes + context_allowance_bytes
    return dict(evidence, before_bytes=pre, after_bytes=post, released_bytes=pre-post,
                ceiling_bytes=ceiling, verified=post <= ceiling,
                reason='within parked/context budget' if post <= ceiling else 'child retains external GPU memory')


class IsolatedVideoJudge:
    def __init__(self, repo, run_dir, device, *, residency_protocol=False, worker_path=None, timeout_s=600,
                 context_allowance_bytes=512*1024**2):
        self.residency_protocol = residency_protocol
        self.timeout_s = timeout_s
        self.device, self.context_allowance_bytes = torch.device(device), context_allowance_bytes
        self.last_release_evidence = None
        self.path = Path(run_dir) / "judge_clip.pt"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo)  # Remove the Cosmos dependency overlay.
        env["PYTHONNOUSERSITE"] = "1"
        child_device = str(self.device)
        if self.device.type == 'cuda':
            # VideoAlign's TrainingArguments selects cuda:0 even when the
            # model starts on CPU. Resolve the scheduler-visible device to its
            # UUID before launch so that child defaults cannot touch policy GPUs.
            identifier = str(torch.cuda.get_device_properties(self.device).uuid)
            env['CUDA_VISIBLE_DEVICES'] = (identifier if identifier.startswith(('GPU-', 'MIG-'))
                                           else 'GPU-' + identifier)
            child_device = 'cuda:0'
        else:
            env['CUDA_VISIBLE_DEVICES'] = ''
        for key in list(env):
            if key.startswith(("MASTER_", "TORCHELASTIC_")) or key in {
                "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"}:
                env.pop(key)
        command = [sys.executable, "-u", str(worker_path or Path(repo) / "scripts/cosmos_quality_worker.py"),
                   "--device", child_device]
        if residency_protocol:
            command += ["--residency-protocol"]
        self.log = open(Path(run_dir) / "quality_worker.log", "a", buffering=1)
        self.process = subprocess.Popen(
            command,
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
        )
        try:
            if self._read() != {"ready": True}:
                raise RuntimeError("VideoReward worker failed to initialize")
            self.parked_baseline = self._nvml_snapshot() if residency_protocol else {}
        except Exception:
            self.close()
            raise

    def _read(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=self.timeout_s):
                raise TimeoutError(f"VideoReward worker timed out; inspect {self.log.name}")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"VideoReward worker exited; inspect {self.log.name}")
        return json.loads(line)

    def score(self, clip, prompt):
        torch.save(clip.detach().cpu(), self.path)
        request = {"clip": str(self.path), "prompt": prompt}
        if self.residency_protocol:
            request.update(version=1, command="score", request_id=uuid.uuid4().hex)
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = self._read()
        if self.residency_protocol:
            self._validate(response, request["request_id"])
        return response["score"]

    def _validate(self, response, identity):
        if response.get("version") != 1 or response.get("request_id") != identity:
            raise RuntimeError("judge response identity/version mismatch")
        if "error" in response:
            raise RuntimeError(response["error"])

    def park(self):
        if not self.residency_protocol:
            raise RuntimeError("park requires opt-in residency protocol")
        identity = uuid.uuid4().hex
        before = self._nvml_snapshot()
        self.process.stdin.write(json.dumps(dict(version=1, command="park", request_id=identity)) + "\n")
        self.process.stdin.flush()
        response = self._read()
        self._validate(response, identity)
        if response.get("residency") != "cpu" or response.get("allocated_bytes") != 0 or response.get("reserved_bytes") != 0:
            acknowledgment = {key: response.get(key) for key in
                              ('version', 'request_id', 'residency', 'allocated_bytes', 'reserved_bytes')}
            raise RuntimeError(f'judge did not acknowledge released CUDA storage: {json.dumps(acknowledgment)}')
        after = self._nvml_snapshot()
        baseline = (self.parked_baseline.get('nvml_process_bytes', {}).get(str(self.process.pid), 0)
                    if 'nvml_device_uuid' in self.parked_baseline else None)
        evidence = verify_judge_release(before, after, self.process.pid, baseline, self.context_allowance_bytes)
        self.last_release_evidence = response['external_release'] = evidence
        if self.device.type == 'cuda' and not evidence['verified']:
            raise RuntimeError(f'judge external CUDA release unverified: {evidence}')
        return response

    def _nvml_snapshot(self):
        return nvml_device_snapshot(self.device)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.log.close()
        self.path.unlink(missing_ok=True)
