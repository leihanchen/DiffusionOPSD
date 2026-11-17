"""Native H3 conditioning, locked CPU caches, and allocation-free preflight.

Only preparation loads Qwen3-VL. Cache state preserves the complete text block
output contract consumed by the t2va core: prompt_embeds AND text_token_tags.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import struct
import tempfile
from types import SimpleNamespace

import torch

from diffusionopsd.video.training_data import content_hash, load_training_data

SCHEMA_VERSION = 2
MODEL_ID = 'MiniMaxAI/MiniMax-H3'
MODEL_REVISION = '0919fb684102ef115126612f3fbb84d1027e7317'
RUNTIME = {'torch': '2.14.0+cu130', 'diffusers': '0.42.1', 'transformers': '5.22.1', 'peft': '0.21.2'}
REQUIRED_FOLDERS = ('transformer', 'text_encoder', 'tokenizer', 'processor', 'vae', 'audio_vae',
                    'scheduler', 'audio_scheduler')
PREFETCH_PATTERNS = ['modular_model_index.json', 'model_index.json', *[f'{p}/*' for p in REQUIRED_FOLDERS]]


def validate_h3_topology(cfg) -> dict:
    """Shared preflight/resume signature. Device IDs are explicit node-local IDs."""
    if isinstance(cfg, dict):
        try:
            canonical = validate_h3_topology(SimpleNamespace(
                num_nodes=cfg['num_nodes'], policy_gpus=cfg['policy_gpus'], target_gpus=cfg['target_gpus'],
                train_pool_size=cfg['num_nodes'], workflow='t2va', clips_per_prompt=cfg['candidates_per_node'],
                candidates_per_node=cfg['candidates_per_node'], policy_microbatch_size=cfg['policy_microbatch_size'],
                eval_count=cfg['evaluation_rounds']*cfg['policy_ranks']*cfg['num_nodes']))
            if canonical != cfg:
                raise ValueError('inconsistent canonical fields')
            return canonical
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'invalid H3 topology: {error}') from error
    policy, target = list(cfg.policy_gpus), list(cfg.target_gpus)
    if not isinstance(cfg.num_nodes, int) or cfg.num_nodes < 1:
        raise ValueError('num_nodes must be an integer M>=1')
    if cfg.train_pool_size < cfg.num_nodes:
        raise ValueError('Training pool must contain at least M records')
    if len(policy) not in (2, 4):
        raise ValueError('Production policy count must be P=2 or P=4')
    if len(policy)+len(target) > 8:
        raise ValueError('Production topology permits at most eight assigned GPUs per node')
    if not target or len(set(policy + target)) != len(policy + target):
        raise ValueError('Device role lists overlap or contain duplicate IDs')
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in policy + target):
        raise ValueError('Explicit device role lists require nonnegative integer IDs')
    if policy != list(range(len(policy))) or target != list(range(len(policy), len(policy) + len(target))):
        raise ValueError('Device role map must follow torchrun rank order: policy first, then target')
    if cfg.workflow != 't2va':
        raise ValueError('Only the native t2va workflow is qualified')
    if getattr(cfg, 'video_shift', 12.) != 12. or getattr(cfg, 'audio_shift', 3.) != 3.:
        raise ValueError('Pinned native scheduler shifts must be video=12 and audio=3')
    if cfg.candidates_per_node != cfg.clips_per_prompt:
        raise ValueError('candidates_per_node and clips_per_prompt disagree')
    if cfg.candidates_per_node != 4 or cfg.policy_microbatch_size != 1:
        raise ValueError('Qualified topology requires candidates_per_node=4 and policy_microbatch_size=1')
    return dict(num_nodes=cfg.num_nodes, policy_gpus=policy, target_gpus=target,
                policy_ranks=len(policy), reward_ranks=len(target), candidates_per_node=cfg.candidates_per_node,
                device_role_map=dict(policy=policy, target=target),
                node_role_maps=[dict(node_index=node, policy=list(policy), target=list(target))
                                for node in range(cfg.num_nodes)],
                policy_microbatch_size=cfg.policy_microbatch_size,
                policy_rounds=cfg.clips_per_prompt // len(policy), optimizer_steps_per_update=1,
                evaluation_rounds=math.ceil(cfg.eval_count / (cfg.num_nodes*len(policy))))


def select_records(data, cfg) -> dict:
    if len(data.train) < cfg.train_pool_size or len(data.evaluation) < cfg.eval_count:
        raise ValueError('Pinned dataset has fewer records than requested train/evaluation pool')
    if not 0 < cfg.calibration_prompts <= cfg.train_pool_size:
        raise ValueError('calibration_prompts must fit the training pool')
    order = list(range(len(data.train)))
    random.Random(cfg.seed).shuffle(order)
    def stable_rows(rows, split):
        return [dict(row, sample_id=row.get('sample_id', f'{split}:{i}:' + _json_hash(row)))
                for i, row in enumerate(rows)]
    train_rows = stable_rows(data.train, 'train')
    test_rows = stable_rows(data.evaluation, 'test')
    train = [train_rows[i] for i in order[:cfg.train_pool_size]]
    evaluation = sorted(test_rows, key=lambda r: r['sample_id'])[:cfg.eval_count]
    return dict(train=train, calibration=train[:cfg.calibration_prompts], evaluation=evaluation)


def encode_native_conditioning(components, prompt: str, *, record_key: str) -> tuple[dict, dict]:
    """Invoke the actual pinned text block, with explicit context overflow refusal."""
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    ids = components.tokenizer(prompt, add_special_tokens=False)['input_ids']
    text_config = components.text_encoder.config.text_config
    limits = [int(text_config.max_position_embeddings)]
    tokenizer_limit = getattr(components.tokenizer, 'model_max_length', None)
    if tokenizer_limit is not None and tokenizer_limit < 10**12:
        limits.append(int(tokenizer_limit))
    limit = min(limits)
    if not ids or len(ids) > limit:
        raise ValueError(f'{record_key}: token count {len(ids)} exceeds context limit {limit} or is empty')
    state = PipelineState(values={'prompt': prompt})
    _, state = MiniMaxH3TextEncoderStep()(components, state)
    fields = {name: getattr(state, name).detach().cpu().contiguous()
              for name in ('prompt_embeds', 'text_token_tags')}
    _validate_state(fields)
    return fields, dict(text_tokens=len(ids), context_limit=limit, encoder_context_limit=limits[0],
                        tokenizer_context_limit=tokenizer_limit, add_special_tokens=False, truncation=False,
                        chat_template=False, layer=50, normalized=False,
                        token_ids_sha256=_json_hash(ids))


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_metadata(state) -> dict:
    return {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                      sha256=hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
            for key, value in state.items()}


def _validate_state(state):
    if set(state) != {'prompt_embeds', 'text_token_tags'}:
        raise ValueError('Native conditioning schema mismatch')
    embeds, tags = state['prompt_embeds'], state['text_token_tags']
    if embeds.ndim != 3 or embeds.shape[0] != 1 or tags.shape != (embeds.shape[1],):
        raise ValueError('Native conditioning tensor shape mismatch')
    if embeds.dtype not in (torch.float32, torch.bfloat16, torch.float16) or tags.dtype != torch.int64:
        raise ValueError('Native conditioning dtype mismatch')
    if not torch.isfinite(embeds).all() or any(v.device.type != 'cpu' for v in state.values()):
        raise ValueError('Conditioning must be finite CPU tensors')


def _validate_metadata(metadata, state, identity):
    required = {'text_tokens', 'context_limit', 'encoder_context_limit', 'tokenizer_context_limit',
                'add_special_tokens', 'truncation', 'chat_template', 'layer', 'normalized', 'token_ids_sha256'}
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError('Conditioning metadata schema mismatch')
    for name in ('add_special_tokens', 'truncation', 'chat_template', 'normalized'):
        if metadata[name] is not False:
            raise ValueError(f'Conditioning metadata native setting mismatch: {name}')
    if type(metadata['layer']) is not int or metadata['layer'] != 50:
        raise ValueError('Conditioning metadata requires native layer 50')
    for name in ('text_tokens', 'context_limit', 'encoder_context_limit'):
        if type(metadata[name]) is not int or metadata[name] <= 0:
            raise ValueError(f'Invalid conditioning token/context metadata: {name}')
    tokenizer_limit = metadata['tokenizer_context_limit']
    if tokenizer_limit is not None and (type(tokenizer_limit) is not int or tokenizer_limit <= 0):
        raise ValueError('Invalid tokenizer context metadata')
    limits = [metadata['encoder_context_limit']]
    if tokenizer_limit is not None and tokenizer_limit < 10**12:
        limits.append(tokenizer_limit)
    if metadata['context_limit'] != min(limits):
        raise ValueError('Conditioning context limits disagree')
    count = metadata['text_tokens']
    if count != state['text_token_tags'].numel() or count > metadata['context_limit']:
        raise ValueError('Conditioning token count/tensor/context mismatch')
    digest = metadata['token_ids_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid conditioning token IDs hash metadata')
    for name, value in identity.get('native_context', {}).items():
        if metadata.get(name) != value:
            raise ValueError(f'Conditioning context metadata differs from pinned components: {name}')


def _entry_integrity(record_key, identity, metadata, tensors):
    return _json_hash(dict(record_key=record_key, identity=identity, metadata=metadata, tensors=tensors))


def entry_path(cache_root: Path, record_key: str) -> Path:
    return Path(cache_root) / (hashlib.sha256(record_key.encode()).hexdigest() + '.pt')


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _atomic_write(path, writer):
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def load_conditioning(cache_root: Path, record_key: str, *, expected_identity: dict) -> dict[str, torch.Tensor]:
    path = entry_path(cache_root, record_key)
    if not path.is_file():
        raise FileNotFoundError(f'Missing conditioning entry: {record_key} ({path})')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['identity'] != expected_identity or payload['record_key'] != record_key:
        raise ValueError(f'Conditioning identity mismatch: {record_key}')
    state = payload['state']
    _validate_state(state)
    metadata = payload.get('metadata')
    _validate_metadata(metadata, state, expected_identity)
    if payload.get('integrity_sha256') != _entry_integrity(record_key, expected_identity, metadata, payload['tensors']):
        raise ValueError(f'Conditioning metadata/tensor identity integrity mismatch: {record_key}')
    if _tensor_metadata(state) != payload['tensors']:
        raise ValueError(f'Conditioning tensor dtype/shape/hash mismatch: {record_key}')
    expected_dtype = expected_identity.get('dtype')
    if expected_dtype and str(state['prompt_embeds'].dtype) != expected_dtype:
        raise ValueError(f'Conditioning dtype mismatch: {record_key}')
    return state


def publish_conditioning(cache_root: Path, record_key: str, identity: dict, producer) -> Path:
    """Lock, recheck, compute one record, then durably publish an immutable entry."""
    path = entry_path(cache_root, record_key)
    with _lock(path.with_suffix('.lock')):
        if path.exists():
            load_conditioning(cache_root, record_key, expected_identity=identity)
            return path
        state, metadata = producer()
        _validate_state(state)
        _validate_metadata(metadata, state, identity)
        tensors = _tensor_metadata(state)
        payload = dict(identity=identity, record_key=record_key, state=state,
                       metadata=metadata, tensors=tensors,
                       integrity_sha256=_entry_integrity(record_key, identity, metadata, tensors))
        _atomic_write(path, lambda stream: torch.save(payload, stream))
        load_conditioning(cache_root, record_key, expected_identity=identity)
    return path


def model_root(cfg) -> Path:
    value = os.path.expandvars(str(cfg.model))
    if '$' in value:
        raise FileNotFoundError(f'Unresolved model path {value}; set VIDEO_REWARD_CKPT_PATH')
    return Path(value)


def runtime_identity() -> dict:
    actual = {name: importlib.metadata.version(name) for name in RUNTIME}
    if actual != RUNTIME:
        raise ValueError(f'H3 runtime identity mismatch: expected {RUNTIME}, actual {actual}')
    return actual


def weight_inventory(folder: Path) -> dict:
    """Read safetensors headers and shard index only, never materialize tensors."""
    folder = Path(folder)
    indexes = sorted(folder.glob('*.safetensors.index.json'))
    weight_map = {}
    expected_size = None
    if indexes:
        if len(indexes) != 1:
            raise ValueError(f'Ambiguous safetensors indexes in {folder}')
        index = json.loads(indexes[0].read_text())
        weight_map = index['weight_map']
        expected_size = index.get('metadata', {}).get('total_size')
        files = [folder / name for name in sorted(set(weight_map.values()))]
    else:
        files = sorted(folder.glob('*.safetensors'))
    if not files:
        raise FileNotFoundError(f'Missing safetensors weights/index in {folder}')
    inventory, shards = {}, {}
    tensor_bytes = file_bytes = 0
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f'Missing indexed H3 weight shard: {path}')
        with path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f'Truncated safetensors header: {path}')
            length = struct.unpack('<Q', prefix)[0]
            if length > 100_000_000 or length > path.stat().st_size - 8:
                raise ValueError(f'Invalid safetensors header size: {path}')
            header = json.loads(stream.read(length))
        size = path.stat().st_size
        file_bytes += size
        shard_bytes = shard_count = 0
        for name, info in header.items():
            if name == '__metadata__':
                continue
            start, end = info['data_offsets']
            if start < 0 or end < start or end + 8 + length > size:
                raise ValueError(f'Truncated tensor {name} in {path}')
            if name in inventory or (weight_map and weight_map.get(name) != path.name):
                raise ValueError(f'Weight index/tensor mismatch: {name}')
            inventory[name] = dict(file=path.name, shape=info['shape'], dtype=info['dtype'], bytes=end-start)
            tensor_bytes += end - start
            shard_bytes += end - start
            shard_count += 1
        shards[path.name] = dict(file_bytes=size, tensor_bytes=shard_bytes, tensor_count=shard_count,
                                 header_sha256=_json_hash(header))
    if weight_map and set(inventory) != set(weight_map):
        raise ValueError(f'Weight index tensor names mismatch in {folder}')
    if expected_size is not None and tensor_bytes != expected_size:
        raise ValueError(f'Weight index total_size mismatch in {folder}: {tensor_bytes} != {expected_size}')
    return dict(tensor_bytes=tensor_bytes, file_bytes=file_bytes, tensor_count=len(inventory), tensors=inventory,
                shards=shards, index=(dict(file=indexes[0].name, sha256=_file_hash(indexes[0]),
                                          metadata=index.get('metadata', {}), weight_map=weight_map) if indexes else None))


def reward_asset_identity(root: Path) -> dict:
    """Hash frozen scorer files without importing or allocating reward models."""
    names = ('waft_tar_c_t.pth', 'depth-anything-3-large-v1.1/model.safetensors',
             'dinov2-base/model.safetensors', 'depth-anything-3-large-v1.1/config.json',
             'dinov2-base/config.json')
    missing = [str(root / name) for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing GeoFlow reward assets: ' + ', '.join(missing))
    return {name: dict(sha256=_file_hash(root / name), bytes=(root / name).stat().st_size) for name in names}


def _asset_identity(cfg) -> dict:
    root = model_root(cfg)
    missing = [str(root / 'opsd_revision.txt')] if not (root / 'opsd_revision.txt').is_file() else []
    if not missing and (root / 'opsd_revision.txt').read_text().strip() != cfg.model_revision:
        raise ValueError('H3 model revision marker mismatch')
    for folder in REQUIRED_FOLDERS:
        if not (root / folder).is_dir() or not any((root / folder).iterdir()):
            missing.append(str(root / folder))
    if not (root / 'modular_model_index.json').is_file():
        missing.append(str(root / 'modular_model_index.json'))
    if missing:
        raise FileNotFoundError('Missing H3 assets: ' + ', '.join(missing) +
                                '; run prepare_minimax_h3.py --prefetch on a network-enabled node')
    if cfg.model_id != MODEL_ID or cfg.model_revision != MODEL_REVISION or cfg.workflow != 't2va':
        raise ValueError('H3 model/workflow identity differs from pinned t2va')
    required_files = [f'{name}/config.json' for name in ('transformer', 'text_encoder', 'vae', 'audio_vae')]
    required_files += [f'{name}/scheduler_config.json' for name in ('scheduler', 'audio_scheduler')]
    required_files += ['tokenizer/tokenizer_config.json', 'tokenizer/tokenizer.json']
    missing_files = [str(root / name) for name in required_files if not (root / name).is_file()]
    if not list((root / 'processor').glob('*config.json')):
        missing_files.append(str(root / 'processor' / '*config.json'))
    if missing_files:
        raise FileNotFoundError('Missing H3 component configs/tokenizer assets: ' + ', '.join(missing_files))
    weights = {name: weight_inventory(root / name)
               for name in ('transformer', 'text_encoder', 'vae', 'audio_vae')}
    # The pinned snapshot supplies weight identities; component config/tokenizer files are hashed independently.
    files = {str(p.relative_to(root)): _file_hash(p) for p in sorted(root.rglob('*'))
             if p.is_file() and (p.suffix == '.json' or p.parent.name in ('tokenizer', 'processor'))}
    from diffusionopsd.video.geo_reward import reward_metadata
    if dict(cfg.geometry_reward) != reward_metadata():
        raise ValueError('H3 reward identity mismatch')
    text_config = json.loads((root / 'text_encoder' / 'config.json').read_text())['text_config']
    tokenizer_config = json.loads((root / 'tokenizer' / 'tokenizer_config.json').read_text())
    native_context = dict(encoder_context_limit=int(text_config['max_position_embeddings']),
                          tokenizer_context_limit=tokenizer_config.get('model_max_length'))
    return dict(schema_version=SCHEMA_VERSION, model_id=cfg.model_id, model_revision=cfg.model_revision,
                tokenizer_revision=cfg.model_revision, processor_revision=cfg.model_revision,
                encoder_revision=cfg.model_revision, component_files=files,
                weight_inventory=weights, weight_inventory_sha256=_json_hash(weights), native_context=native_context,
                weight_bytes={name: value['tensor_bytes'] for name, value in weights.items()},
                runtime=runtime_identity(),
                dtype='torch.' + cfg.conditioning_dtype, layer=50, normalized=False,
                workflow='t2va', reward=reward_metadata(),
                reward_assets=reward_asset_identity(Path(os.environ.get('VIDEO_REWARD_CKPT_PATH', root.parent))))


def _preparation_identity(cfg, data, selected, *, base_identity=None):
    identity = dict(base_identity) if base_identity is not None else _asset_identity(cfg)
    identity.pop('weight_inventory', None)
    identity.update(dataset=data.metadata, dataset_fingerprint=data.fingerprint,
                    train_hash=content_hash(selected['train']), eval_hash=content_hash(selected['evaluation']))
    return identity


def _record_identity(identity, row):
    return dict(identity, prompt=row['text'], prompt_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                record_sha256=_json_hash(row))


def prefetch_h3_assets(cfg) -> Path:
    from huggingface_hub import snapshot_download
    root = model_root(cfg)
    snapshot_download(cfg.model_id, revision=cfg.model_revision, local_dir=root,
                      allow_patterns=PREFETCH_PATTERNS, max_workers=1)
    (root / 'opsd_revision.txt').write_text(cfg.model_revision + '\n')
    load_training_data(cfg, offline=False)
    return root


def _load_encoder_components(cfg, device):
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MINIMAX_H3_TEXT_TAG
    root = model_root(cfg)
    dtype = getattr(torch, cfg.conditioning_dtype)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(root / 'text_encoder',
                    dtype=dtype, local_files_only=True).eval().to(device)
    return SimpleNamespace(text_encoder=encoder,
        tokenizer=Qwen2TokenizerFast.from_pretrained(root / 'tokenizer', local_files_only=True),
        processor=Qwen3VLProcessor.from_pretrained(root / 'processor', local_files_only=True),
        text_encoder_layer=50, text_tag=MINIMAX_H3_TEXT_TAG, _execution_device=torch.device(device))


def h3_geometry(cfg) -> dict:
    """Return the generation geometry used by conditioning and checkpoints."""
    return dict(num_frames=cfg.num_frames, height=cfg.height, width=cfg.width, fps=cfg.fps)


def _modality_counts(cfg, text_tokens):
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        video_latent_num_frames, audio_latent_num_frames,
    )
    video = video_latent_num_frames(cfg.num_frames, 17, 5) * (cfg.height // 32) * (cfg.width // 32)
    audio = audio_latent_num_frames(cfg.num_frames) * 2
    return dict(text=text_tokens, video=video, audio=audio, total=text_tokens + video + audio)


def prepare_conditioning(cfg, *, device: str, cache_root: Path) -> Path:
    validate_h3_topology(cfg)
    data = load_training_data(cfg)
    selected = select_records(data, cfg)
    identity = _preparation_identity(cfg, data, selected)
    components = None
    records = {}
    try:
        for row in selected['train'] + selected['evaluation']:
            key = row['sample_id']
            expected = _record_identity(identity, row)
            def produce(row=row, key=key):
                nonlocal components
                if components is None:
                    components = _load_encoder_components(cfg, device)
                state, metadata = encode_native_conditioning(components, row['text'], record_key=key)
                return state, metadata
            path = publish_conditioning(cache_root, key, expected, produce)
            state = load_conditioning(cache_root, key, expected_identity=expected)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            records[key] = dict(identity=expected, file=path.name, integrity_sha256=payload['integrity_sha256'],
                                modality_tokens=_modality_counts(cfg, state['text_token_tags'].numel()))
    finally:
        del components
    manifest = dict(identity=identity, records=records,
                    selections={name: [r['sample_id'] for r in rows] for name, rows in selected.items()},
                    geometry=h3_geometry(cfg),
                    longest_record=max(records, key=lambda key: records[key]['modality_tokens']['text']))
    path = Path(cache_root) / 'manifest.json'
    with _lock(path.with_suffix('.lock')):
        _atomic_write(path, lambda stream: stream.write(json.dumps(manifest, indent=2, sort_keys=True).encode()))
    return path


def validate_h3_assets(cfg, cache_root: Path) -> dict:
    """Offline preflight: no encoder, VAE, CUDA initialization or network access."""
    topology = validate_h3_topology(cfg)
    asset_identity = _asset_identity(cfg)
    path = Path(cache_root) / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing conditioning manifest: {path}; run prepare_minimax_h3.py --condition')
    manifest = json.loads(path.read_text())
    geometry = h3_geometry(cfg)
    if manifest.get('geometry') != geometry:
        raise ValueError(f'Conditioning manifest geometry mismatch: cached {manifest.get("geometry")}, '
                         f'configured {geometry}; run prepare_minimax_h3.py --condition for the new geometry')
    data = load_training_data(cfg, offline=True)
    selected = select_records(data, cfg)
    expected = _preparation_identity(cfg, data, selected, base_identity=asset_identity)
    if manifest['identity'] != expected:
        raise ValueError('Conditioning manifest identity mismatch')
    for name, rows in selected.items():
        if manifest['selections'][name] != [r['sample_id'] for r in rows]:
            raise ValueError(f'Conditioning selection mismatch: {name}')
    missing = []
    for row in selected['train'] + selected['evaluation']:
        key = row['sample_id']
        if key not in manifest['records'] or not entry_path(cache_root, key).is_file():
            missing.append(key)
            continue
        state = load_conditioning(cache_root, key, expected_identity=_record_identity(expected, row))
        payload = torch.load(entry_path(cache_root, key), map_location='cpu', weights_only=True)
        if manifest['records'][key].get('integrity_sha256') != payload['integrity_sha256']:
            raise ValueError(f'Conditioning metadata integrity differs from manifest: {key}')
        if manifest['records'][key]['modality_tokens'] != _modality_counts(cfg, state['text_token_tags'].numel()):
            raise ValueError(f'Conditioning geometry/token counts mismatch: {key}')
    if missing:
        raise FileNotFoundError('Missing conditioning entries: ' + ', '.join(missing))
    return dict(identity=expected, topology=topology, weight_inventory=asset_identity['weight_inventory'],
                manifest_sha256=_file_hash(path),
                manifest=str(path), records=len(manifest['records']), longest_record=manifest['longest_record'])
