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
