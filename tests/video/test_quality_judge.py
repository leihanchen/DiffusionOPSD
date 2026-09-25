import importlib
import sys
import types

import pytest
import torch

from diffusionopsd.video.quality_judge import VideoRewardJudge, gap_to_probability, quality_gap


class _MeanScorer:
    """Stand-in reward: VQ and MQ equal the clip mean, so a brighter clip scores higher."""

    def __call__(self, clip, prompt):
        del prompt
        value = float(clip.mean())
        return {"VQ": value, "MQ": value, "TA": 0.0, "Overall": 2.0 * value}


def test_quality_gap_uses_the_worse_of_vq_and_mq():
    rollout = {"VQ": 0.5, "MQ": -0.2}
    reference = {"VQ": 0.0, "MQ": 0.0}
    assert quality_gap(rollout, reference) == -0.2


def test_equal_clips_sit_above_the_mask_threshold():
    assert abs(gap_to_probability(0.0) - 0.5) < 1e-6


def test_p_win_prefers_the_higher_scoring_clip():
    judge = VideoRewardJudge(_MeanScorer(), num_frames=4)
    bright, dark = torch.full((6, 3, 8, 8), 0.9), torch.full((6, 3, 8, 8), 0.1)
    assert judge.p_win(bright, dark, "a cat walking") > 0.6
    assert judge.p_win(dark, bright, "a cat walking") < 0.4
    assert abs(judge.p_win(bright, bright, "a cat walking") - 0.5) < 1e-5


@pytest.mark.parametrize("fail_import", [False, True])
def test_videoalign_imports_restore_existing_utils(tmp_path, monkeypatch, fail_import):
    from diffusionopsd.video.quality_judge import _videoalign_import_scope

    waft_utils = types.ModuleType("utils")
    waft_child = types.ModuleType("utils.utils")
    monkeypatch.setitem(sys.modules, "utils", waft_utils)
    monkeypatch.setitem(sys.modules, "utils.utils", waft_child)
    (tmp_path / "utils.py").write_text("def save_video():\n    return 'videoalign'\n")
    (tmp_path / "inference.py").write_text("from utils import save_video\n")
    original_path = sys.path[:]

    try:
        with _videoalign_import_scope(str(tmp_path)):
            inference = importlib.import_module("inference")
            assert inference.save_video() == "videoalign"
            assert "utils.utils" not in sys.modules
            if fail_import:
                raise RuntimeError("loader failed")
    except RuntimeError:
        assert fail_import

    assert sys.modules["utils"] is waft_utils
    assert sys.modules["utils.utils"] is waft_child
    assert sys.path == original_path
    # References retained by the scorer remain usable after restoring WAFT.
    assert inference.save_video() == "videoalign"
