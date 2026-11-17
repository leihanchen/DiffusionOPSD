import pytest

from diffusionopsd.curriculum.diagnosis import select_frames, validate_diagnosis


def test_endpoint_inclusion_and_short_clip():
    assert select_frames(17) == [0, 2, 5, 7, 9, 11, 14, 16]
    assert select_frames(3) == [0, 1, 2]
    with pytest.raises(ValueError):
        select_frames(0)


def report(status='observed_fail', evidence=None):
    return {'requirements': [{'requirement_id': 'stationary', 'status': status,
                              'frame_ids': [0, 16] if evidence is None else evidence,
                              'reason': 'Object changes position relative to scene.'}]}


def test_valid_evidence_and_unknown():
    assert validate_diagnosis(report(), [0, 8, 16], ['stationary']) == report()
    assert validate_diagnosis(report('unknown', []), [0, 8, 16], ['stationary'])['requirements'][0]['status'] == 'unknown'


@pytest.mark.parametrize('bad', [report(evidence=[]), report(evidence=[7]), report('pass'),
                                {'requirements': []}, {'requirements': [report()['requirements'][0]] * 2}])
def test_invalid_or_absent_evidence(bad):
    with pytest.raises(ValueError):
        validate_diagnosis(bad, [0, 8, 16], ['stationary'])


def test_saved_frames_retain_times_and_bound_resolution(tmp_path):
    import torch
    from PIL import Image
    from diffusionopsd.curriculum.diagnosis import save_frame_evidence

    clip = torch.zeros(17, 3, 20, 40)
    evidence = save_frame_evidence(clip, tmp_path, fps=8., longest_side=16)
    assert evidence['frames'][0]['timestamp'] == 0.
    assert evidence['frames'][-1]['timestamp'] == 2.
    assert len(evidence['frames']) == 8
    assert Image.open(evidence['frames'][0]['path']).size == (16, 8)
