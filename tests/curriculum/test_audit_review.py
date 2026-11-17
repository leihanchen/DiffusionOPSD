import pytest

from diffusionopsd.curriculum.audit_review import compare_review


def inputs():
    from diffusionopsd.curriculum.vlm import _template
    template_hash = _template({'mode': 'diagnose'})[1]
    ids = ['camera_motion', 'stationarity', 'identity', 'geometry']
    clips, humans, predictions = [], [], []
    for i in range(12):
        clips.append({'clip_id': str(i), 'request_hash': f'input-{i}', 'frames': [{'frame_id': 0}, {'frame_id': 16}]})
        humans.append({'clip_id': str(i), 'request_hash': f'input-{i}', 'reviewer': 'reviewer-a', 'labels': {k: 'unknown' for k in ids},
                       'human_evidence': {k: {'frame_ids': [], 'reason': 'Insufficient views'} for k in ids}})
        predictions.append({'model_id': 'google/gemma-4-12B-it', 'model_revision': 'a' * 40, 'request_hash': f'input-{i}',
                            'template_hash': template_hash, 'result': {'requirements': [{'requirement_id': k, 'status': 'unknown',
                            'frame_ids': [], 'reason': 'Insufficient views'} for k in ids]}})
    manifest = {'model_id': 'google/gemma-4-12B-it', 'model_revision': 'a' * 40,
                'requirements': ids, 'clips': clips}
    return manifest, humans, predictions


def test_review_reports_disagreements_without_automatic_approval():
    manifest, humans, predictions = inputs()
    predictions[0]['result']['requirements'][0].update(status='observed_pass', frame_ids=[0, 16])
    report, disagreements = compare_review(manifest, humans, predictions)
    assert report['clip_count'] == 12
    assert report['approved'] is False
    assert len(disagreements) == 1
    assert disagreements[0]['clip_id'] == '0'
    assert disagreements[0]['human_label'] == 'unknown'
    assert disagreements[0]['gemma_label'] == 'observed_pass'


@pytest.mark.parametrize('defect', ['missing_labels', 'duplicate_id', 'wrong_model', 'missing_reason', 'bad_frame'])
def test_review_rejects_unreliable_comparison_inputs(defect):
    manifest, humans, predictions = inputs()
    if defect == 'missing_labels':
        humans[0]['labels']['geometry'] = None
    elif defect == 'duplicate_id':
        humans[0]['clip_id'] = '1'
    elif defect == 'wrong_model':
        predictions[0]['model_revision'] = 'b' * 40
    elif defect == 'missing_reason':
        humans[0]['human_evidence']['geometry']['reason'] = ''
    elif defect == 'bad_frame':
        humans[0]['human_evidence']['geometry'].update(frame_ids=[9], reason='Visible')
    with pytest.raises(ValueError):
        compare_review(manifest, humans, predictions)


def test_comparison_cli_writes_unapproved_report_and_disagreement_files(tmp_path, monkeypatch):
    import json
    import scripts.audit_curriculum_diagnosis as cli

    manifest, humans, predictions = inputs()
    for name, value in [('manifest', manifest), ('labels', {'rows': humans}), ('predictions', predictions)]:
        (tmp_path / f'{name}.json').write_text(json.dumps(value))
    output = tmp_path / 'comparison'
    monkeypatch.setattr('sys.argv', ['audit', '--manifest', str(tmp_path / 'manifest.json'),
        '--labels', str(tmp_path / 'labels.json'), '--predictions', str(tmp_path / 'predictions.json'),
        '--output-dir', str(output)])
    cli.main()
    assert json.loads((output / 'agreement_report.json').read_text())['approved'] is False
    assert json.loads((output / 'disagreements.json').read_text()) == []
    assert len((output / 'agreement_summary.csv').read_text().splitlines()) == 5


def test_review_rejects_labels_from_different_packet_request():
    manifest, humans, predictions = inputs()
    manifest['clips'][0]['request_hash'] = 'current-frame-prompt-hash'
    predictions[0]['request_hash'] = 'current-frame-prompt-hash'
    humans[0]['request_hash'] = 'stale-frame-prompt-hash'
    with pytest.raises(ValueError, match='Human'):
        compare_review(manifest, humans, predictions)


def test_report_preserves_prediction_input_and_template_identity():
    manifest, humans, predictions = inputs()
    for i, (clip, human, prediction) in enumerate(zip(manifest['clips'], humans, predictions)):
        clip['request_hash'] = human['request_hash'] = prediction['request_hash'] = f'input-{i}'
    report, _ = compare_review(manifest, humans, predictions)
    changed = dict(predictions[0], request_hash='other')
    with pytest.raises(ValueError):
        compare_review(manifest, humans, [changed, *predictions[1:]])
    from diffusionopsd.curriculum.vlm import _template
    assert report['diagnosis_template_hash'] == _template({'mode': 'diagnose'})[1]
    assert len(report['request_hashes']) == 12
