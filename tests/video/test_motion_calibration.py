import torch
import pytest
from diffusionopsd.video.motion_calibration import motion_score, calibration_indices


def test_motion_matches_georeward_average():
    frames = torch.zeros(1, 3, 3, 2, 2)
    fields = [torch.ones(1, 2, 2, 2) * 3, torch.ones(1, 2, 2, 2) * 4]

    def flow(a, b):
        return fields.pop(0)

    score = motion_score(frames, flow)
    assert torch.allclose(score, torch.tensor([(3 + 4) * 2**0.5 / 2]))


def test_calibration_sample_is_reproducible_and_extends_old_sample():
    selected = calibration_indices(200, 50)
    assert len(set(selected)) == 50
    assert selected[:2] == [98, 194]
    assert selected == calibration_indices(200, 50)
    with pytest.raises(ValueError):
        calibration_indices(20, 50)
