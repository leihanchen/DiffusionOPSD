"""Allocation ownership and bounded subprocess execution for curriculum rounds."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import socket
import subprocess


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def file_lock(path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Resource is already owned or locked: {path}') from exc
        try:
            yield stream.fileno()
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def device_claims(directory: Path, gpu_ids: list[str]):
    with ExitStack() as stack:
        descriptors = []
        for gpu_id in sorted(set(gpu_ids)):
            identity = hashlib.sha256(f'{socket.gethostname()}:{gpu_id}'.encode()).hexdigest()
            descriptors.append(stack.enter_context(file_lock(Path(directory) / f'{identity}.lock')))
        yield descriptors


def invoke(command: list[str], *, directory: Path, environment: dict, timeout: float,
           lock_fds: list[int] = ()) -> None:
    if not command or not all(isinstance(value, str) and value for value in command):
        raise ValueError('Subprocess command must be a nonempty argument list')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'stdout.log').open('a') as output, (directory / 'stderr.log').open('a') as errors:
        process = subprocess.Popen(command, stdout=output, stderr=errors,
                                   env={**os.environ, **environment}, start_new_session=True,
                                   pass_fds=tuple(lock_fds))
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if returncode:
        raise RuntimeError(f'Worker exited with code {returncode}; see {directory / "stderr.log"}')


def inventory(command: list[str] | None) -> dict:
    command = command or ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits']
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    devices = []
    for line in result.stdout.splitlines():
        index, uuid, model, memory = [value.strip() for value in line.split(',')]
        devices.append(dict(index=int(index), uuid=uuid, model=model, memory_mb=float(memory)))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        selected = []
        for value in visible.split(','):
            matches = [device for device in devices
                       if str(device['index']) == value or device['uuid'] == value]
            if len(matches) != 1:
                raise ValueError('Cannot resolve allocated CUDA_VISIBLE_DEVICES against measured inventory')
            selected.append(matches[0])
        devices = selected
    if not devices or len({device['uuid'] for device in devices}) != len(devices):
        raise ValueError('Measured allocation must contain distinct GPUs')
    if len({device['model'] for device in devices}) != 1:
        raise ValueError('A single run requires homogeneous GPU models')
    return dict(gpus_per_node=len(devices), gpu_memory_gb=min(device['memory_mb'] for device in devices) / 1024,
                gpu_model=devices[0]['model'], devices=devices, host=socket.gethostname())
