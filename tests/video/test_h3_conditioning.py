"""CPU contracts for native H3 preparation; no checkpoint downloads."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from diffusionopsd.video import h3_conditioning as h3
from diffusionopsd.video.training_data import TrainingData


def config():
    spec = importlib.util.spec_from_file_location('h3_config', 'config/minimax_h3_t2va.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def test_defaults_and_topology():
    cfg = config()
    assert (cfg.num_frames, cfg.height, cfg.width, cfg.num_steps) == (124, 544, 960, 50)
    assert (cfg.policy_microbatch_size, cfg.clips_per_prompt, cfg.train_pool_size) == (1, 4, 256)
    two = h3.validate_h3_topology(cfg)
    cfg.policy_gpus = (0, 1, 2, 3)
    cfg.target_gpus = (4,)
    four = h3.validate_h3_topology(cfg)
    assert (two['policy_rounds'], four['policy_rounds']) == (2, 1)
    assert two['candidates_per_node'] == four['candidates_per_node'] == 4
    assert two['optimizer_steps_per_update'] == four['optimizer_steps_per_update'] == 1
    cfg.target_gpus = (3,)
    with pytest.raises(ValueError, match='overlap'):
        h3.validate_h3_topology(cfg)


def test_topology_rejects_more_than_eight_assigned_gpus_without_allocation(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'init', lambda: pytest.fail('preflight initialized CUDA'))
    cfg = config()
    cfg.target_gpus = tuple(range(2, 9))
    with pytest.raises(ValueError, match='eight'):
        h3.validate_h3_topology(cfg)
    cfg.target_gpus = tuple(range(2, 8))
    assert h3.validate_h3_topology(cfg)['reward_ranks'] == 6


def test_selection_reuses_calibration_and_is_stable():
    rows = [dict(sample_id=str(i), text=f' caption {i} ') for i in range(300)]
    data = TrainingData(rows, [dict(sample_id=f'e{i}', text='test') for i in range(12)], {}, 't', 'e')
    first = h3.select_records(data, config())
    assert first == h3.select_records(data, config())
    assert len(first['train']) == 256 and len(first['evaluation']) == 8
    assert first['calibration'] == first['train'][:4]
    assert first['train'][0]['text'].startswith(' ')


class Tokenizer:
    model_max_length = 100
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {'input_ids': list(range(len(text)))}


class Encoder:
    dtype = torch.float32
    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=52, max_position_embeddings=100))
    def model(self, **kwargs):
        ids = kwargs['input_ids']
        assert kwargs['output_hidden_states'] and not kwargs['use_cache']
        return SimpleNamespace(hidden_states=tuple(torch.full((*ids.shape, 3), float(i + 2)) for i in range(53)))


def components():
    return SimpleNamespace(tokenizer=Tokenizer(), text_encoder=Encoder(),
                           processor=SimpleNamespace(create_mm_token_type_ids=lambda ids: [[0] * len(ids[0])]),
                           text_encoder_layer=50, _execution_device=torch.device('cpu'), text_tag=0)


def test_native_block_preserves_layer_tags_and_longest_prompt():
    result, metadata = h3.encode_native_conditioning(components(), '  longest caption  ', record_key='long')
    assert set(result) == {'prompt_embeds', 'text_token_tags'}
    assert torch.equal(result['prompt_embeds'], torch.full((1, 19, 3), 52.0))
    assert torch.equal(result['text_token_tags'], torch.zeros(19, dtype=torch.long))
    assert metadata['text_tokens'] == 19 and metadata['add_special_tokens'] is False
    with pytest.raises(ValueError, match='overflow.*101.*100'):
        h3.encode_native_conditioning(components(), 'x' * 101, record_key='overflow')
    bad = components()
    bad.text_encoder = Encoder()
    bad.text_encoder.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=50, max_position_embeddings=100))
    with pytest.raises(ValueError, match='more than 50'):
        h3.encode_native_conditioning(bad, 'abc', record_key='bad')


def test_cache_atomic_concurrent_identity_dtype_and_hash(tmp_path):
    identity = {'revision': 'a', 'dtype': 'torch.float32'}
    state, metadata = h3.encode_native_conditioning(components(), 'abc', record_key='same')
    calls = []
    def produce():
        calls.append(1)
        return state, metadata
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda _: h3.publish_conditioning(tmp_path, 'same', identity, produce), range(2)))
    assert len(calls) == 1
    loaded = h3.load_conditioning(tmp_path, 'same', expected_identity=identity)
    assert torch.equal(loaded['prompt_embeds'], state['prompt_embeds'])
    with pytest.raises(ValueError, match='identity'):
        h3.publish_conditioning(tmp_path, 'same', {'revision': 'other'}, produce)
    entry = h3.entry_path(tmp_path, 'same')
    payload = torch.load(entry, weights_only=True)
    payload['state']['prompt_embeds'] = payload['state']['prompt_embeds'].double()
    torch.save(payload, entry)
    with pytest.raises(ValueError, match='dtype|hash'):
        h3.load_conditioning(tmp_path, 'same', expected_identity=identity)


def test_interrupted_cache_is_never_visible(tmp_path):
    def fail():
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):
        h3.publish_conditioning(tmp_path, 'broken', {}, fail)
    assert not h3.entry_path(tmp_path, 'broken').exists()
    with pytest.raises(FileNotFoundError, match='broken'):
        h3.load_conditioning(tmp_path, 'broken', expected_identity={})
    assert not list(tmp_path.glob('*.tmp'))


def test_candidate_alias_and_multinode_signature():
    cfg = config()
    assert cfg.candidates_per_node == 4
    cfg.num_nodes = 2
    signature = h3.validate_h3_topology(cfg)
    assert signature['num_nodes'] == 2
    assert signature['device_role_map'] == {'policy': [0, 1], 'target': [2]}
    cfg.clips_per_prompt = 2
    with pytest.raises(ValueError, match='candidates_per_node'):
        h3.validate_h3_topology(cfg)


def test_smoke_defaults():
    spec = importlib.util.spec_from_file_location('h3_smoke', 'config/minimax_h3_t2va_smoke.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.get_config()
    assert (cfg.num_frames, cfg.height, cfg.width, cfg.num_steps, cfg.updates, cfg.eval_count) == (124, 256, 256, 16, 1, 4)
    assert cfg.conditioning_cache != config().conditioning_cache


def test_weight_inventory_detects_missing_indexed_shards(tmp_path):
    import json
    from safetensors.torch import save_file
    torch_state = {'first': torch.ones(2, 3), 'second': torch.ones(3, dtype=torch.bfloat16)}
    save_file(torch_state, tmp_path / 'part.safetensors')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({
        'metadata': {'total_size': 30},
        'weight_map': {'first': 'part.safetensors', 'second': 'part.safetensors'},
    }))
    inventory = h3.weight_inventory(tmp_path)
    assert inventory['tensor_bytes'] == 30 and inventory['tensor_count'] == 2
    assert inventory['index']['file'] == 'model.safetensors.index.json'
    assert inventory['index']['weight_map'] == {'first': 'part.safetensors', 'second': 'part.safetensors'}
    assert inventory['shards']['part.safetensors']['tensor_count'] == 2
    assert inventory['tensors']['second'] == dict(file='part.safetensors', shape=[3], dtype='BF16', bytes=6)
    (tmp_path / 'part.safetensors').unlink()
    with pytest.raises(FileNotFoundError, match='part.safetensors'):
        h3.weight_inventory(tmp_path)


def test_prepare_and_complete_preflight_without_model_allocations(tmp_path, monkeypatch):
    import json
    cfg = config()
    cfg.train_pool_size, cfg.calibration_prompts, cfg.eval_count = 4, 2, 2
    rows = [dict(sample_id=f't{i}', text='x' * (i + 1)) for i in range(4)]
    data = TrainingData(rows, [dict(sample_id=f'e{i}', text='test') for i in range(2)], {}, 't', 'e')
    monkeypatch.setattr(h3, 'load_training_data', lambda *a, **kw: data)
    inventory = {'transformer': {'tensor_count': 1, 'tensor_bytes': 24,
        'tensors': {'weight': dict(file='part.safetensors', shape=[2, 3], dtype='F32', bytes=24)},
        'index': {'file': 'model.safetensors.index.json', 'weight_map': {'weight': 'part.safetensors'}},
        'shards': {'part.safetensors': {'tensor_count': 1}}}}
    monkeypatch.setattr(h3, '_asset_identity', lambda cfg: {'dtype': 'torch.float32', 'weight_inventory': inventory})
    loads = []
    def load(*args):
        loads.append(1)
        return components()
    monkeypatch.setattr(h3, '_load_encoder_components', load)
    manifest = h3.prepare_conditioning(cfg, device='cpu', cache_root=tmp_path)
    assert len(loads) == 1
    assert len(json.loads(manifest.read_text())['records']) == 6
    h3.prepare_conditioning(cfg, device='cpu', cache_root=tmp_path)
    assert len(loads) == 1
    provenance = h3.validate_h3_assets(cfg, tmp_path)
    assert provenance['records'] == 6
    assert provenance['weight_inventory'] == inventory
    entry = h3.entry_path(tmp_path, 'e1')
    payload = torch.load(entry, weights_only=True)
    payload['metadata']['token_ids_sha256'] = '0' * 64
    payload['integrity_sha256'] = h3._entry_integrity('e1', payload['identity'], payload['metadata'], payload['tensors'])
    torch.save(payload, entry)
    with pytest.raises(ValueError, match='metadata integrity differs from manifest'):
        h3.validate_h3_assets(cfg, tmp_path)
    h3.entry_path(tmp_path, 'e1').unlink()
    h3.prepare_conditioning(cfg, device='cpu', cache_root=tmp_path)
    entry = h3.entry_path(tmp_path, 'e0')
    entry.unlink()
    with pytest.raises(FileNotFoundError, match='e0'):
        h3.validate_h3_assets(cfg, tmp_path)


def test_explicit_device_map_matches_torchrun_rank_order():
    cfg = config()
    cfg.policy_gpus = (1, 2)
    cfg.target_gpus = (0,)
    with pytest.raises(ValueError, match='rank order'):
        h3.validate_h3_topology(cfg)


def test_file_record_ids_are_stable():
    cfg = config()
    cfg.train_pool_size, cfg.calibration_prompts, cfg.eval_count = 4, 2, 1
    data = TrainingData([{'text': str(i)} for i in range(4)], [{'text': 'test'}], {}, 't', 'e', is_hf=False)
    selected = h3.select_records(data, cfg)
    assert len({row['sample_id'] for row in selected['train'] + selected['evaluation']}) == 5
    assert selected == h3.select_records(data, cfg)


def test_atomic_serialization_failure_cleans_temporary_file(tmp_path, monkeypatch):
    state, metadata = h3.encode_native_conditioning(components(), 'abc', record_key='partial')
    def partial_save(payload, stream):
        stream.write(b'partial data')
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', partial_save)
    with pytest.raises(OSError, match='disk full'):
        h3.publish_conditioning(tmp_path, 'partial', {}, lambda: (state, metadata))
    assert not h3.entry_path(tmp_path, 'partial').exists()
    assert not list(tmp_path.glob('*.tmp'))


def test_reward_assets_missing_and_content_identity(tmp_path):
    with pytest.raises(FileNotFoundError, match='waft_tar_c_t.pth'):
        h3.reward_asset_identity(tmp_path)
    for name in ('waft_tar_c_t.pth', 'depth-anything-3-large-v1.1/model.safetensors',
                 'dinov2-base/model.safetensors', 'depth-anything-3-large-v1.1/config.json',
                 'dinov2-base/config.json'):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(b'original')
    before = h3.reward_asset_identity(tmp_path)
    (tmp_path / 'waft_tar_c_t.pth').write_bytes(b'changed')
    assert h3.reward_asset_identity(tmp_path) != before


@pytest.mark.parametrize('mutation', ['missing', 'token_hash', 'token_count', 'context', 'encoder_context',
                                     'tokenizer_context', 'layer', 'normalized', 'special', 'truncation', 'chat'])
def test_cache_rejects_token_context_metadata_corruption(tmp_path, mutation):
    state, metadata = h3.encode_native_conditioning(components(), 'abc', record_key='metadata')
    identity = {'revision': 'a', 'dtype': 'torch.float32'}
    path = h3.publish_conditioning(tmp_path, 'metadata', identity, lambda: (state, metadata))
    payload = torch.load(path, weights_only=True)
    changes = {'token_hash': ('token_ids_sha256', '0' * 64), 'token_count': ('text_tokens', 2),
               'context': ('context_limit', 2), 'encoder_context': ('encoder_context_limit', 1),
               'tokenizer_context': ('tokenizer_context_limit', 1), 'layer': ('layer', 49),
               'normalized': ('normalized', True), 'special': ('add_special_tokens', True),
               'truncation': ('truncation', True), 'chat': ('chat_template', True)}
    if mutation == 'missing':
        payload.pop('metadata')
    else:
        field, value = changes[mutation]
        payload['metadata'][field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match='metadata|token|context'):
        h3.load_conditioning(tmp_path, 'metadata', expected_identity=identity)


def test_cache_rejects_self_consistent_but_out_of_context_metadata(tmp_path):
    state, metadata = h3.encode_native_conditioning(components(), 'abc', record_key='context')
    metadata.update(context_limit=2, encoder_context_limit=2, tokenizer_context_limit=2)
    with pytest.raises(ValueError, match='context'):
        h3.publish_conditioning(tmp_path, 'context', {}, lambda: (state, metadata))
    assert not h3.entry_path(tmp_path, 'context').exists()


def test_cli_reports_inventory_audit_with_file_hash(tmp_path, monkeypatch, capsys):
    import hashlib
    import json
    import sys
    spec = importlib.util.spec_from_file_location('prepare_h3', 'scripts/prepare_minimax_h3.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inventory = {'transformer': {'tensor_count': 1, 'tensor_bytes': 24,
        'tensors': {'long_tensor_name': dict(shape=[2, 3], dtype='F32', bytes=24, file='part.safetensors')},
        'shards': {'part.safetensors': {'tensor_count': 1}}, 'index': None}}
    monkeypatch.setattr(h3, 'validate_h3_assets', lambda *a: {'weight_inventory': inventory, 'records': 6})
    monkeypatch.setattr(sys, 'argv', ['prepare_minimax_h3.py', '--check-only', '--cache-root', str(tmp_path)])
    assert module.main() == 0
    output = capsys.readouterr().out
    result = json.loads(output)
    audit = tmp_path / 'preflight-weight-inventory.json'
    assert json.loads(audit.read_text()) == inventory
    assert result['weight_inventory_audit']['path'] == str(audit)
    assert result['weight_inventory_audit']['sha256'] == hashlib.sha256(audit.read_bytes()).hexdigest()
    assert 'long_tensor_name' not in output
