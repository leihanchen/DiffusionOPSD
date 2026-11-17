from dataclasses import asdict
import json
import sys

import pytest

from diffusionopsd.curriculum.proposer import propose
from diffusionopsd.curriculum.records import PromptRecord


@pytest.fixture
def request_data():
    parent = PromptRecord('p', 'p', None, 'train', 'A stationary statue; orbit camera.',
                          'orbit', {}, 'initial', 0)
    return {'mode': 'propose', 'model_id': 'google/gemma-4-12B-it', 'model_revision': 'a' * 40,
            'seed': 42, 'history': [asdict(parent)],
            'payload': {'domain': 'G1', 'round_index': 1, 'split': 'train', 'count': 1,
                        'sources': {'edited': 1}, 'parents': [asdict(parent)], 'feedback': {}}}


def executable(tmp_path, kind):
    path = tmp_path / 'worker.py'
    path.write_text('''import argparse, json, pathlib, time
p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--response'); a=p.parse_args()
r=json.loads(pathlib.Path(a.request).read_text())
assert 'history' not in r
seen=pathlib.Path(a.request).parent / 'calls'
seen.write_text(str(int(seen.read_text())+1) if seen.exists() else '1')
row={'prompt_id':'child','lineage_id':'p','parent_id':'p','split':'train',
     'text':'A stationary marble statue, camera orbits with foreground clutter.', 'subtype':'orbit',
     'attributes':{'edit':'Add foreground clutter','edit_axis':'clutter'}, 'source':'edited','round_index':1}
''' + {
        'valid': '', 'malformed': "pathlib.Path(a.response).write_text('oops'); raise SystemExit(0)\n",
        'wrong_parent': "row['parent_id']='missing'\n", 'cross_split': "row['split']='test'\n",
        'duplicate': "row['text']='A stationary statue; orbit camera.'\n",
        'timeout': 'time.sleep(10)\n',
    }[kind] + '''response={'result':[row], 'usage':{'input_tokens':10,'output_tokens':5},
          'model_id':r['model_id'],'model_revision':r['model_revision']}
pathlib.Path(a.response).write_text(json.dumps(response))
''')
    return [sys.executable, str(path)]


def test_valid_proposal_replays_completed_receipt(tmp_path, request_data):
    cmd = executable(tmp_path, 'valid')
    result = propose(request_data, cmd, tmp_path / 'run', 10)
    assert result[0].parent_id == 'p'
    assert propose(request_data, cmd, tmp_path / 'run', 10) == result
    assert list((tmp_path / 'run').rglob('calls'))[0].read_text() == '1'


@pytest.mark.parametrize('kind', ['malformed', 'wrong_parent', 'cross_split', 'duplicate'])
def test_invalid_proposal_has_bounded_repairs(tmp_path, request_data, kind):
    with pytest.raises(ValueError):
        propose(request_data, executable(tmp_path, kind), tmp_path / 'run', 10)
    assert list((tmp_path / 'run').rglob('calls'))[0].read_text() == '3'
    receipt = json.loads(list((tmp_path / 'run').rglob('receipt.json'))[0].read_text())
    assert receipt['status'] == 'failed'
    assert len(receipt['attempts']) == 3


def test_timeout_is_recorded_and_not_silently_retried(tmp_path, request_data):
    with pytest.raises(TimeoutError):
        propose(request_data, executable(tmp_path, 'timeout'), tmp_path / 'run', .1)
    receipt = json.loads(list((tmp_path / 'run').rglob('receipt.json'))[0].read_text())
    assert receipt['status'] == 'failed'
    assert receipt['attempts'][0]['elapsed_seconds'] > 0


def test_interrupted_receipt_requires_reconciliation(tmp_path, request_data):
    from diffusionopsd.curriculum.proposer import content_hash

    cmd = executable(tmp_path, 'valid')
    from diffusionopsd.curriculum.vlm import _template
    _, digest = _template(request_data)
    directory = tmp_path / content_hash({'request': request_data, 'command': cmd, 'resolved_template_hash': digest})
    directory.mkdir()
    (directory / 'receipt.json').write_text(json.dumps({'status': 'in_flight'}))
    with pytest.raises(RuntimeError, match='reconciliation'):
        propose(request_data, cmd, tmp_path, 10)
    assert not (directory / 'calls').exists()


def test_template_change_invalidates_outer_proposal_receipt(tmp_path, request_data, monkeypatch):
    import shutil

    templates = tmp_path / 'templates'
    shutil.copytree('config/curriculum_prompts', templates)
    monkeypatch.setenv('CURRICULUM_TEMPLATE_DIR', str(templates))
    cmd = executable(tmp_path, 'valid')
    propose(request_data, cmd, tmp_path / 'run', 10)
    path = templates / 'revise.md'
    path.write_text(path.read_text() + '\nAdditional checked constraint.\n')
    propose(request_data, cmd, tmp_path / 'run', 10)
    assert len(list((tmp_path / 'run').rglob('receipt.json'))) == 2


def test_schema_failure_exit_keeps_bounded_repairs_and_usage(tmp_path, request_data):
    path = tmp_path / 'invalid_worker.py'
    path.write_text('''import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--response');a=p.parse_args()
pathlib.Path(a.response).write_text(json.dumps({'error':'invalid_response','detail':'JSON expected',
'usage':{'output_tokens':12},'attempts':[]}))
raise SystemExit(2)
''')
    with pytest.raises(ValueError, match='three attempts'):
        propose(request_data, [sys.executable, str(path)], tmp_path / 'run', 10)
    receipt = json.loads(next((tmp_path / 'run').rglob('receipt.json')).read_text())
    assert len(receipt['attempts']) == 3
    assert all(r['usage']['output_tokens'] == 12 for r in receipt['attempts'])
