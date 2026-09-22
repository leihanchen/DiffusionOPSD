import torch
from diffusionopsd.video.gates import (
    identity_keep, motion_mask, quality_mask, endpoint_weight, effective_weight, percentile_threshold,
)


def test_identity_keep():
    keep = identity_keep(torch.tensor([0.9, 0.5, 0.7]), tau_id=0.7)
    assert keep.tolist() == [True, False, True]


def test_motion_and_quality_masks_are_float_indicators():
    assert motion_mask(torch.tensor([0.1, 2.0]), 0.5).tolist() == [0.0, 1.0]
    assert quality_mask(torch.tensor([0.39, 0.4]), 0.4).tolist() == [0.0, 1.0]


def test_endpoint_weight_matches_repo_r1():
    adv = torch.tensor([-10.0, -5.0, 0.0, 2.5, 5.0, 10.0])
    r1 = endpoint_weight(adv, adv_clip_max=5.0)
    assert torch.allclose(r1, torch.tensor([0.0, 0.0, 0.5, 0.75, 1.0, 1.0]))


def test_effective_weight_is_hard_masked():
    adv = torch.tensor([5.0, 5.0, 5.0])
    w = effective_weight(adv, 5.0, m=torch.tensor([1.0, 0.0, 1.0]), tau_motion=0.5,
                         p_q=torch.tensor([0.9, 0.9, 0.1]), tau_q=0.4)
    assert w.tolist() == [1.0, 0.0, 0.0]


def test_percentile_threshold():
    vals = torch.arange(1.0, 101.0)
    assert abs(percentile_threshold(vals, 0.10) - 10.9) < 0.2
