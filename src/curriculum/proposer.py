"""Bounded local proposal calls with durable receipts and host-side lineage gates."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from diffusionopsd.curriculum.records import PromptRecord, validate_manifest


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_candidates(result, request: dict, history: list[PromptRecord]) -> list[PromptRecord]:
    payload = request['payload']
    if not isinstance(result, list) or len(result) != payload['count']:
        raise ValueError('Proposal does not match requested candidate count')
    try:
        records = [PromptRecord(**row) for row in result]
    except (TypeError, KeyError) as exc:
        raise ValueError('Malformed prompt record') from exc
    validate_manifest(records, history)
    if Counter(r.source for r in records) != Counter(payload['sources']):
        raise ValueError('Proposal source quotas differ from request')
    parents = {p['prompt_id']: p for p in payload.get('parents', [])}
    if Counter(r.parent_id for r in records if r.source == 'edited') != Counter(list(parents)):
        raise ValueError('Each allocated parent must produce exactly one controlled edit')
    for r in records:
        if payload.get('id_prefix') and not r.prompt_id.startswith(payload['id_prefix']):
            raise ValueError('Candidate ID must use the requested unique prefix')
        if r.split != payload['split'] or r.round_index != payload['round_index']:
            raise ValueError('Proposal changes requested split/round')
        if r.source == 'edited':
            if r.parent_id not in parents:
                raise ValueError('Proposal uses an unallocated parent')
            if r.attributes.get('edit_axis') not in {'object/scene', 'camera_path', 'viewpoint_range', 'clutter'}:
                raise ValueError('Edit must declare one allowed edit axis')
            if not r.attributes.get('edit', '').strip():
                raise ValueError('Edit must record its intended change')
    return records


def _invoke(command: list[str], request_path: Path, response_path: Path, timeout: float,
            lock_fds: tuple[int, ...] = ()) -> None:
    with (request_path.parent / (request_path.stem + '.stderr')).open('w') as errors:
        process = subprocess.Popen([*command, '--request', str(request_path), '--response', str(response_path)],
                                   stdout=subprocess.DEVNULL, stderr=errors, start_new_session=True, pass_fds=lock_fds)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError('Local VLM request timed out; process group terminated') from exc
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        if returncode and not (returncode == 2 and response_path.is_file()):
            raise RuntimeError(f'Local VLM worker exited with code {returncode}; see saved stderr')


def propose(request: dict, command: list[str], output_dir: Path, timeout_seconds: int,
            *, lock_fds: tuple[int, ...] = ()) -> list[PromptRecord]:
    """At most three attempts. Interrupted in-flight calls require reconciliation.

    `history` is host-only and is never serialized to the VLM process. The visible
    payload contains explicitly permitted training parents and aggregate feedback.
    """
    if not command or isinstance(command, str) or timeout_seconds <= 0 or request.get('mode') != 'propose':
        raise ValueError('Expected a proposer request, executable argument list, and positive timeout')
    from diffusionopsd.curriculum.vlm import _template, validate_request

    def semantic(value):
        if isinstance(value, dict):
            return bool(value.get('observations') or value.get('diagnoses')) or any(semantic(v) for v in value.values())
        return isinstance(value, list) and any(semantic(v) for v in value)

    if semantic(request['payload'].get('feedback', {})):
        from diffusionopsd.curriculum.diagnosis import validate_audit_approval

        if not request.get('diagnosis_audit_path'):
            raise ValueError('Semantic proposal guidance requires an approved diagnosis audit')
        validate_audit_approval(request['diagnosis_audit_path'], request['model_id'], request['model_revision'])
    visible = {k: v for k, v in request.items() if k not in {'history', 'diagnosis_audit_path'}}
    validate_request(visible)
    history = [PromptRecord(**r) for r in request.get('history', [])]
    _, template_hash = _template(visible)
    key = content_hash({'request': request, 'command': command, 'resolved_template_hash': template_hash})
    directory = Path(output_dir).resolve() / key
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / 'receipt.json'
    # Lock creation is atomic; a crash leaves a visible lock for explicit recovery.
    lock = directory / 'request.lock'
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError('Proposal request is active or interrupted; reconcile its lock before retry') from exc
    os.close(lock_fd)
    try:
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['status'] == 'complete':
                return validate_candidates(receipt['result'], request, history)
            raise RuntimeError('Incomplete/failed request requires explicit reconciliation before retry')
        receipt = {'request_hash': key, 'status': 'pending', 'attempts': [], 'result': None}
        repair_error = None
        for attempt in range(3):
            request_path = directory / f'request_{attempt}.json'
            response_path = directory / f'response_{attempt}.json'
            outgoing = dict(visible, seed=visible.get('seed', 42) + attempt, repair_attempt=attempt)
            if repair_error:
                outgoing['repair_error'] = repair_error
            atomic_json(request_path, outgoing)
            row = {'attempt': attempt, 'status': 'in_flight', 'started_unix': time.time()}
            receipt['attempts'].append(row)
            receipt['status'] = 'in_flight'
            atomic_json(receipt_path, receipt)
            started = time.monotonic()
            try:
                _invoke(command, request_path, response_path, timeout_seconds, lock_fds)
                response = json.loads(response_path.read_text())
                row['usage'] = response.get('usage', {})
                row['worker_attempts'] = response.get('attempts', [])
                if response.get('error'):
                    raise ValueError(response.get('detail', response['error']))
                if (response.get('model_id'), response.get('model_revision')) != (
                    request['model_id'], request['model_revision']
                ):
                    raise ValueError('Worker returned a different model identity')
                records = validate_candidates(response['result'], request, history)
            except (ValueError, KeyError, TypeError) as exc:
                repair_error = str(exc)
                row.update(status='invalid', error=repair_error, elapsed_seconds=time.monotonic() - started)
                receipt['status'] = 'failed'
                atomic_json(receipt_path, receipt)
                if attempt == 2:
                    raise ValueError('Proposal failed validation after three attempts') from exc
                continue
            except Exception as exc:
                row.update(status='failed', error=type(exc).__name__, elapsed_seconds=time.monotonic() - started)
                receipt['status'] = 'failed'
                atomic_json(receipt_path, receipt)
                raise
            row.update(status='complete', elapsed_seconds=time.monotonic() - started)
            receipt.update(status='complete', result=[asdict(r) for r in records])
            atomic_json(receipt_path, receipt)
            return records
        raise AssertionError('Unreachable attempt state')
    finally:
        lock.unlink()
