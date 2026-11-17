"""Shared frozen VLM request processing; no cross-request conversation state."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from diffusionopsd.curriculum.diagnosis import select_frames, validate_diagnosis
from diffusionopsd.curriculum.records import PromptRecord, SUBTYPES

MODELS = {'google/gemma-4-12B-it', 'Qwen/Qwen3.8-27B'}


def validate_local_checkpoint(path: Path, model_id: str, revision: str) -> Path:
    path = Path(path).resolve()
    if model_id not in MODELS or re.fullmatch('[a-f0-9]{40}', revision) is None:
        raise ValueError('Approved model ID and immutable 40-character revision required')
    marker = path / 'opsd_revision.txt'
    if path.name != revision and (not marker.is_file() or marker.read_text().strip() != revision):
        raise ValueError('Local checkpoint revision does not match requested identity')
    config = json.loads((path / 'config.json').read_text())
    family = 'gemma4' if model_id.startswith('google/') else 'qwen'
    if not config.get('model_type', '').startswith(family):
        raise ValueError('Local model architecture differs from selected family')
    return path


def _proposal_payload(payload):
    if not isinstance(payload, dict) or set(payload) - {
        'domain', 'round_index', 'split', 'count', 'sources', 'parents', 'feedback', 'id_prefix'
    }:
        raise ValueError('Unsupported proposal fields')
    if payload.get('domain') != 'G1' or payload.get('split') not in {'train', 'development', 'test'}:
        raise ValueError('Only bounded G1 task generation is enabled')
    if 'id_prefix' in payload and re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', payload['id_prefix']) is None:
        raise ValueError('Invalid prompt ID prefix')
    count, index, sources = payload.get('count'), payload.get('round_index'), payload.get('sources')
    if type(count) is not int or not 1 <= count <= 4 or type(index) is not int or index < 0:
        raise ValueError('Proposal requests require 1..4 candidates and a nonnegative round')
    if (not isinstance(sources, dict) or set(sources) - {'initial', 'new', 'edited'}
            or any(type(v) is not int or v <= 0 for v in sources.values()) or sum(sources.values()) != count):
        raise ValueError('Invalid proposal source quotas')
    initial = sources == {'initial': count}
    if (index == 0) != initial or (payload['split'] != 'train' and not initial):
        raise ValueError('Development/test generation must precede training and use initial requests')
    parents = payload.get('parents', [])
    if not isinstance(parents, list):
        raise ValueError('Parents must be a list')
    records = [PromptRecord(**p) for p in parents]
    if (len(records) != sources.get('edited', 0) or len({p.prompt_id for p in records}) != len(records)
            or any(p.split != 'train' or p.round_index >= index for p in records)):
        raise ValueError('Each edit requires one distinct earlier training parent')
    feedback = payload.get('feedback', {})
    if (not isinstance(feedback, dict) or set(feedback) - {'prompts', 'subtypes', 'development', 'diagnoses'}
            or (initial and feedback)):
        raise ValueError('Initial tasks must be authored before feedback is consumed')
    development = feedback.get('development', {})
    if not isinstance(development, dict) or set(development) - SUBTYPES:
        raise ValueError('Development feedback must contain subtype aggregates only')

    def private(value):
        if isinstance(value, dict):
            return bool(set(value) & {'prompt_id', 'text', 'lineage_id', 'seed'}) or any(private(v) for v in value.values())
        return isinstance(value, list) and any(private(v) for v in value)

    if private(development):
        raise ValueError('Private development queries cannot enter proposal feedback')
    return payload


def validate_request(request: dict) -> None:
    if request.get('mode') not in {'diagnose', 'propose'}:
        raise ValueError('Unknown VLM request mode')
    if request.get('model_id') not in MODELS or re.fullmatch('[a-f0-9]{40}', request.get('model_revision', '')) is None:
        raise ValueError('Request must identify the pinned frozen model')
    if type(request.get('seed', 42)) is not int:
        raise ValueError('Request seed must be an integer')
    if request['mode'] == 'propose':
        _proposal_payload(request['payload'])


def _template(request):
    name = ('diagnose' if request['mode'] == 'diagnose' else
            'initial' if 'initial' in request['payload']['sources'] else 'revise')
    root = Path(os.environ.get('CURRICULUM_TEMPLATE_DIR', 'config/curriculum_prompts'))
    text = (root / f'{name}.md').read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if request.get('template_hash', digest) != digest:
        raise ValueError('Request template hash does not match local template')
    return text, digest


def build_messages(request: dict) -> list[dict]:
    """Construct a fresh two-message request with whitelisted role inputs."""
    payload = request['payload']
    if request['mode'] == 'propose':
        _proposal_payload(payload)
    elif request['mode'] != 'diagnose':
        raise ValueError('Unknown request mode')
    template, _ = _template(request)
    if request.get('repair_error'):
        template += '\nPrior response failed schema validation: ' + str(request['repair_error'])[:1000]
    if request['mode'] == 'propose':
        content = [{'type': 'text', 'text': json.dumps(payload, sort_keys=True, allow_nan=False)}]
    else:
        from PIL import Image

        evidence = payload['evidence']
        frames, fps, count = evidence['frames'], evidence['native_fps'], evidence['frame_count']
        requirements = payload['requirements']
        if (not isinstance(payload.get('prompt'), str) or not payload['prompt'].strip()
                or not isinstance(requirements, list) or not requirements
                or any(not isinstance(r, str) or not r for r in requirements)
                or len(set(requirements)) != len(requirements)
                or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0):
            raise ValueError('Diagnosis requires a prompt, requirement IDs, and valid frame timing')
        if [f['frame_id'] for f in frames] != select_frames(count):
            raise ValueError('Diagnosis must use the fixed ordered frame selection including endpoints')
        content = []
        for frame in frames:
            path = Path(frame['path'])
            if not path.is_absolute() or not path.is_file():
                raise ValueError('Only existing absolute local frame paths are accepted')
            if not math.isclose(frame['timestamp'], frame['frame_id'] / fps, abs_tol=1e-6):
                raise ValueError('Frame timestamps differ from source timing')
            with Image.open(path) as source:
                image = source.convert('RGB')
                image.thumbnail((448, 448))
            content.extend([{'type': 'text', 'text': f"Frame {frame['frame_id']} at {frame['timestamp']:.6f}s"},
                            {'type': 'image', 'image': image}])
        content.append({'type': 'text', 'text': json.dumps({
            'prompt': payload['prompt'], 'requirements': requirements,
            'frame_count': count, 'native_fps': fps}, sort_keys=True)})
    return [{'role': 'system', 'content': [{'type': 'text', 'text': template}]},
            {'role': 'user', 'content': content}]


def parse_json_answer(answer: str) -> tuple[object, bool]:
    """Normalize one complete JSON Markdown fence; never extract from prose.

    The real Gemma checkpoint emits a JSON fence despite the no-Markdown prompt.
    This is transport normalization only: all record/evidence gates still run.
    """
    text = answer.strip()
    wrapped = re.fullmatch(r"```json[ \t]*\r?\n(.*?)\r?\n```", text, flags=re.DOTALL)
    normalized = wrapped is not None
    if wrapped:
        text = wrapped.group(1)
    return json.loads(text), normalized


def run_requests(requests: list[dict], model: object, processor: object) -> list[dict]:
    import torch

    from diffusionopsd.curriculum.proposer import validate_candidates

    if not requests:
        raise ValueError('Worker request list must not be empty')
    for request in requests:
        validate_request(request)
    identities = {(r['model_id'], r['model_revision']) for r in requests}
    if len(identities) != 1:
        raise ValueError('Both roles must use one frozen model revision')
    model.eval().requires_grad_(False)
    responses = []
    for request in requests:
        messages = build_messages(request)
        started = time.monotonic()
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              return_dict=True, return_tensors='pt', enable_thinking=False)
        token_count = inputs['input_ids'].shape[-1]
        frame_sizes = [list(item['image'].size) for message in messages for item in message['content']
                       if item.get('type') == 'image']
        modality_ids = inputs.get('mm_token_type_ids')
        image_ids = getattr(processor, 'image_token_ids', [])
        if modality_ids is not None:
            visual_tokens = int((modality_ids == 1).sum())
            visual_source = 'processor_mm_token_type_ids'
        elif image_ids:
            visual_tokens = int(sum((inputs['input_ids'] == i).sum() for i in image_ids))
            visual_source = 'processor_image_token_ids'
        else:
            visual_tokens = 0 if not frame_sizes else None
            visual_source = 'no_images' if not frame_sizes else 'unavailable'
        processor_metadata = {k: inputs[k].tolist() for k in ('image_sizes', 'image_grid_thw')
                              if k in inputs and hasattr(inputs[k], 'tolist')}
        processor_metadata.update({k + '_shape': list(v.shape) for k, v in inputs.items()
                                   if 'pixel_values' in k and hasattr(v, 'shape')})
        if token_count > 8192:
            raise ValueError(f'Input token budget exceeded: {token_count} > 8192; no automatic truncation')
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        # Keep token indices integer while casting floating visual inputs to BF16.
        inputs = {k: v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v
                  for k, v in inputs.items()}
        proposal = request['mode'] == 'propose'
        options = dict(do_sample=proposal, max_new_tokens=2048 if proposal else 1024, max_time=300.)
        if proposal:
            options.update(temperature=.7, top_p=.9)
        devices = [model.device.index or 0] if model.device.type == 'cuda' else []
        if devices:
            torch.cuda.reset_peak_memory_stats(model.device)
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(request.get('seed', 42))
            output = model.generate(**inputs, **options)
        output_tokens = output.shape[-1] - token_count
        usage = dict(input_tokens=token_count, output_tokens=output_tokens,
                     visual_tokens=visual_tokens, visual_token_count_source=visual_source,
                     selected_frame_sizes=frame_sizes, processor_image_metadata=processor_metadata,
                     output_cap_reached=output_tokens >= options['max_new_tokens'],
                     selected_frames=0 if proposal else len(request['payload']['evidence']['frames']),
                     truncated=False, elapsed_seconds=time.monotonic() - started,
                     peak_memory_bytes=torch.cuda.max_memory_allocated(model.device) if devices else 0)
        try:
            text = processor.decode(output[0][token_count:], skip_special_tokens=False)
            parsed = processor.parse_response(text, prefix=inputs['input_ids'][0])
            if not isinstance(parsed, dict) or parsed.get('tool_calls') or not isinstance(parsed.get('content'), str):
                raise ValueError('Model did not return a plain assistant answer')
            result, normalized = parse_json_answer(parsed['content'])
            usage['json_fence_normalized'] = normalized
            if proposal:
                validate_candidates(result, request, [PromptRecord(**r) for r in request['payload'].get('parents', [])])
            else:
                validate_diagnosis(result, [f['frame_id'] for f in request['payload']['evidence']['frames']],
                                   request['payload']['requirements'])
        except (ValueError, TypeError, KeyError) as exc:
            raise InvalidModelResponse(str(exc), usage) from exc
        _, template_hash = _template(request)
        responses.append(dict(result=result, model_id=request['model_id'], model_revision=request['model_revision'],
                              template_hash=template_hash, usage=usage))
    return responses


class InvalidModelResponse(ValueError):
    """Schema failures carry consumed inference costs to bounded repair receipts."""

    def __init__(self, message: str, usage: dict):
        super().__init__(message)
        self.usage = usage
