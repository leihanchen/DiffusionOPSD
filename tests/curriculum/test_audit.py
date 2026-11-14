from dataclasses import asdict
import json
import sys

import pytest

from diffusionopsd.curriculum.diagnosis import diagnosis_audit
from diffusionopsd.curriculum.proposer import propose
from diffusionopsd.curriculum.records import PromptRecord


def audit_rows(count=12):
    from diffusionopsd.curriculum.vlm import _template
    template_hash = _template({'mode': 'diagnose'})[1]
    statuses = ['observed_pass', 'observed_fail', 'unknown']
    return [{'clip_id': str(i), 'reviewer': 'human-1',
             'labels': {'stationarity': statuses[i % 3]},
             'prediction': {'requirements': [{'requirement_id': 'stationarity', 'status': statuses[i % 3],
                                             'frame_ids': [0], 'reason': 'Visible evidence or ambiguous.'}]},
             'frame_ids': [0], 'request_hash': f'test-input-{i}',
             'diagnosis_template_hash': template_hash} for i in range(count)]


def test_audit_reports_disagreement_and_requires_independent_coverage():
    rows = audit_rows()
    rows[0]['prediction']['requirements'][0]['status'] = 'observed_fail'
    result = diagnosis_audit(rows, 'google/gemma-4-12B-it', 'a' * 40)
    assert result['requirements']['stationarity']['disagreements'] == 1
    assert result['requirements']['stationarity']['unknown_rate'] == pytest.approx(4 / 12)
    assert result['approved'] is False
    with pytest.raises(ValueError):
        diagnosis_audit(rows[:-1], 'google/gemma-4-12B-it', 'a' * 40)
    rows[0]['reviewer'] = ''
    with pytest.raises(ValueError):
        diagnosis_audit(rows, 'google/gemma-4-12B-it', 'a' * 40)


def test_semantic_guidance_requires_reviewed_matching_audit(tmp_path):
    p = PromptRecord('p', 'p', None, 'train', 'A static scene', 'orbit', {}, 'initial', 0)
    request = {'mode': 'propose', 'model_id': 'google/gemma-4-12B-it', 'model_revision': 'a' * 40,
               'history': [asdict(p)], 'payload': {'domain': 'G1', 'round_index': 1, 'split': 'train', 'count': 1,
               'sources': {'edited': 1}, 'parents': [asdict(p)],
               'feedback': {'prompts': {'p': {'observations': {'stationarity': {'observed_fail': 1}}}}}}}
    with pytest.raises(ValueError, match='audit'):
        propose(request, [sys.executable, 'missing_worker.py'], tmp_path, 10)
    report = diagnosis_audit(audit_rows(), request['model_id'], request['model_revision'])
    report['approved'] = True
    report['approved_by'] = 'human-reviewer'
    path = tmp_path / 'audit.json'
    path.write_text(json.dumps(report))
    request['diagnosis_audit_path'] = str(path)
    report['model_revision'] = 'b' * 40
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='audit'):
        propose(request, [sys.executable, 'missing_worker.py'], tmp_path, 10)


def test_approved_audit_validates_unchanged_report(tmp_path):
    from diffusionopsd.curriculum.diagnosis import validate_audit_approval

    report = diagnosis_audit(audit_rows(), 'google/gemma-4-12B-it', 'a' * 40)
    report.update(approved=True, approved_by='independent-reviewer')
    path = tmp_path / 'audit.json'
    path.write_text(json.dumps(report))
    assert validate_audit_approval(path, report['model_id'], report['model_revision']) == report
    report['requirements']['stationarity']['disagreements'] = 99
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_audit_approval(path, report['model_id'], report['model_revision'])


def test_explicit_24_clip_audit_remains_available():
    report = diagnosis_audit(audit_rows(24), 'google/gemma-4-12B-it', 'a' * 40, expected_count=24)
    assert report['clip_count'] == report['required_clip_count'] == 24
    assert report['audit_scope'] == 'pilot'


def test_small_audit_reports_coverage_gaps_and_overconfident_errors():
    rows = audit_rows()
    for row in rows:
        row['labels']['stationarity'] = 'unknown'
        row['prediction']['requirements'][0]['status'] = 'observed_pass'
    report = diagnosis_audit(rows, 'google/gemma-4-12B-it', 'a' * 40)
    result = report['requirements']['stationarity']
    assert result['count'] == 12
    assert result['overconfident_on_unknown'] == 12
    assert result['missing_human_classes'] == ['observed_fail', 'observed_pass']
    assert report['coverage_complete'] is False
    assert report['approved'] is False


def test_approval_rejects_changed_active_diagnosis_template(tmp_path, monkeypatch):
    import shutil
    from diffusionopsd.curriculum.diagnosis import validate_audit_approval

    report = diagnosis_audit(audit_rows(), 'google/gemma-4-12B-it', 'a' * 40)
    report.update(approved=True, approved_by='human-reviewer')
    path = tmp_path / 'audit.json'
    path.write_text(json.dumps(report))
    templates = tmp_path / 'templates'
    shutil.copytree('config/curriculum_prompts', templates)
    template = templates / 'diagnose.md'
    template.write_text(template.read_text() + '\nChanged rubric\n')
    monkeypatch.setenv('CURRICULUM_TEMPLATE_DIR', str(templates))
    with pytest.raises(ValueError, match='audit'):
        validate_audit_approval(path, report['model_id'], report['model_revision'])
