#!/usr/bin/env python3
"""Offline, frozen Gemma/Qwen worker for separate diagnosis/proposal JSON requests."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Support an isolated interpreter without installing the incompatible training dependencies.
if 'diffusionopsd' not in sys.modules:
    import types

    package = types.ModuleType('diffusionopsd')
    package.__path__ = [str(ROOT / 'src')]
    sys.modules['diffusionopsd'] = package

from diffusionopsd.curriculum.proposer import atomic_json, content_hash  # noqa: E402
from diffusionopsd.curriculum.vlm import (  # noqa: E402
    InvalidModelResponse, run_requests, validate_local_checkpoint, validate_request,
)


def load_frozen_model(path: Path, model_id: str, revision: str, device: str):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    path = validate_local_checkpoint(path, model_id, revision)
    parsed = torch.device(device)
    if parsed.type != 'cuda' or not torch.cuda.is_available() or (parsed.index or 0) >= torch.cuda.device_count():
        raise ValueError('VLM inference requires an allocated local CUDA device')
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={'': str(parsed)},
    )
    model.eval().requires_grad_(False)
    return model, processor


def cache_identity(request: dict) -> str:
    evidence = request.get('payload', {}).get('evidence', {})
    images = []
    for frame in evidence.get('frames', []):
        path = Path(frame['path'])
        if not path.is_absolute() or not path.is_file():
            raise ValueError('Evidence requires existing absolute local paths')
        import hashlib

        images.append(hashlib.sha256(path.read_bytes()).hexdigest())
    from diffusionopsd.curriculum.vlm import _template

    _, template_hash = _template(request)
    return content_hash(dict(request=request, frame_hashes=images, template_hash=template_hash, schema_version=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, default=os.environ.get('CURRICULUM_VLM_MODEL_PATH'))
    parser.add_argument('--model-id', default=os.environ.get('CURRICULUM_VLM_MODEL_ID', 'google/gemma-4-12B-it'))
    parser.add_argument('--revision', default=os.environ.get('CURRICULUM_VLM_REVISION'))
    parser.add_argument('--device', default=os.environ.get('CURRICULUM_VLM_DEVICE', 'cuda:0'))
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()
    if args.model_path is None or args.revision is None:
        parser.error('A local model path and pinned revision are required; no downloads or fallback models')
    args.response.parent.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or args.response.parent / '.curriculum_vlm_cache'
    cache.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.request.read_text())
    batch = isinstance(data, list)
    requests = data if batch else [data]
    if not requests:
        raise ValueError('Empty request batch')
    for request in requests:
        validate_request(request)
        if request['model_id'] != args.model_id or request['model_revision'] != args.revision:
            raise ValueError('Request identity differs from selected frozen checkpoint')
    os.environ.setdefault('CURRICULUM_TEMPLATE_DIR', str(ROOT / 'config/curriculum_prompts'))
    model, processor, results = None, None, []
    for request in requests:
        key = cache_identity(request)
        path, lock = cache / f'{key}.json', cache / f'{key}.lock'
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt['status'] in {'complete', 'invalid'}:
                response = receipt['response']
                response['usage'] = dict(response['usage'], cache_hit=True)
                results.append(response)
                continue
            raise RuntimeError('Interrupted/failed VLM request requires explicit reconciliation')
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError('VLM request is already active or interrupted; reconcile before retry') from exc
        os.close(fd)
        atomic_json(path, {'status': 'in_flight', 'request_hash': key, 'started_unix': time.time()})
        attempts = []
        request_started = time.monotonic()
        stage = 'model_load' if model is None else 'inference'
        try:
            load_seconds = 0.
            if model is None:
                started = time.monotonic()
                model, processor = load_frozen_model(args.model_path, args.model_id, args.revision, args.device)
                load_seconds = time.monotonic() - started
            stage = 'inference'
            outgoing = dict(request)
            # Proposer repairs are owned by propose(); direct diagnosis has the same three-attempt bound here.
            for attempt in range(3 if request['mode'] == 'diagnose' else 1):
                try:
                    response = run_requests([outgoing], model, processor)[0]
                    attempts.append(dict(status='complete', usage=response['usage']))
                    break
                except InvalidModelResponse as exc:
                    attempts.append(dict(status='invalid', error=str(exc), usage=exc.usage))
                    response = dict(error='invalid_response', detail=str(exc), usage=exc.usage,
                                    model_id=args.model_id, model_revision=args.revision)
                    outgoing['repair_error'] = str(exc)
            response.update(request_hash=key, model_load_seconds=load_seconds, attempts=attempts,
                            runtime_versions={name: importlib.metadata.version(name)
                                              for name in ('torch', 'transformers')})
            status = 'invalid' if 'error' in response else 'complete'
            atomic_json(path, dict(status=status, request_hash=key, response=response))
            results.append(response)
        except Exception as exc:
            import torch

            peak = None
            if torch.cuda.is_available():
                try:
                    peak = torch.cuda.max_memory_allocated(args.device)
                except (RuntimeError, ValueError):
                    pass
            usage = dict(elapsed_seconds=time.monotonic() - request_started, peak_memory_bytes=peak,
                         input_tokens=None, output_tokens=None, tokens_unknown=True)
            atomic_json(path, dict(status='failed', request_hash=key, error=type(exc).__name__,
                                   error_message=str(exc), stage=stage, usage=usage, attempts=attempts))
            raise
        finally:
            lock.unlink()
    atomic_json(args.response, results if batch else results[0])
    if any('error' in result for result in results):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
