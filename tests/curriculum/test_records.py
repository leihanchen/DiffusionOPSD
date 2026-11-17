from dataclasses import replace
import json

import pytest

from diffusionopsd.curriculum.records import PromptRecord, manifest_hash, read_manifest, validate_manifest
from config.curriculum_cosmos import get_config, execution_profile, validate_execution_profile


def record(i='a', **kw):
    return PromptRecord(**dict(prompt_id=i, lineage_id=i, parent_id=None, split='train',
                              text=f'A rigid statue {i}, camera orbits.', subtype='orbit',
                              attributes={'scene': 'static'}, source='initial', round_index=0) | kw)


def test_lineage_partition_inheritance():
    parent = record()
    validate_manifest([record('b', parent_id='a', lineage_id='a', source='edited', round_index=1)], [parent])
    for changes in ({'split': 'test'}, {'lineage_id': 'other'}, {'round_index': 0}):
        with pytest.raises(ValueError):
            validate_manifest([record('b', **({'parent_id': 'a', 'lineage_id': 'a', 'source': 'edited', 'round_index': 1} | changes))], [parent])


def test_duplicate_prompt_rejected():
    with pytest.raises(ValueError):
        validate_manifest([record(), record('b', text='  A rigid   statue a, CAMERA orbits. ')], [])
    with pytest.raises(ValueError):
        validate_manifest([record('a', text='changed')], [record()])
    with pytest.raises(ValueError):
        validate_manifest([record('b', lineage_id='a', split='test')], [record()])


def test_manifest_hash_stable(tmp_path):
    rows = [record(), record('b')]
    path = tmp_path / 'manifest.json'
    from dataclasses import asdict
    path.write_text(json.dumps([asdict(x) for x in rows], indent=4))
    assert manifest_hash(read_manifest(path)) == manifest_hash(rows)
    assert manifest_hash(rows) != manifest_hash(list(reversed(rows)))
    assert manifest_hash(rows) != manifest_hash([replace(rows[0], text='changed'), rows[1]])


@pytest.mark.parametrize('changes', [{'source': 'bad'}, {'split': 'bad'}, {'subtype': 'gravity'},
                                     {'text': ''}, {'round_index': -1}, {'attributes': {'x': 2}}])
def test_invalid_records_rejected(changes):
    with pytest.raises(ValueError):
        validate_manifest([record(**changes)], [])


def inventory(count=4):
    return {'gpus_per_node': count, 'gpu_memory_gb': 40, 'gpu_model': 'A100'}


def test_unresolved_h200_inventory_rejected():
    p = execution_profile('h200_gemma12_control')
    with pytest.raises(ValueError):
        validate_execution_profile(p, {})
    inv = {'gpus_per_node': 2, 'gpu_memory_gb': 141, 'gpu_model': 'H200'}
    resolved = validate_execution_profile(p, inv)
    assert resolved['gpus_per_node'] == 2
    assert resolved['nodes_per_run'] == 1


def test_device_map_exceeds_allocation_rejected():
    with pytest.raises(ValueError):
        validate_execution_profile(execution_profile('narval_gemma12'), inventory(1))
    p = execution_profile('narval_gemma12')
    p['train_device_map']['policy'] = 4
    with pytest.raises(ValueError):
        validate_execution_profile(p, inventory())


def test_profile_preserves_pilot_budget():
    c = get_config()
    for name in ('narval_gemma12', 'h200_gemma12_control', 'h200_qwen27_comparison'):
        execution_profile(name)
        assert (c['prompts_per_round'], c['clips_per_prompt'], c['rounds']) == (32, 4, 3)
    assert validate_execution_profile(execution_profile('narval_gemma12'), inventory())['vlm_batch_size'] == 1
