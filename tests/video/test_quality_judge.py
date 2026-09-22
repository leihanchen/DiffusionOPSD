import torch
from diffusionopsd.video.quality_judge import PairwiseVLMJudge, JUDGE_PROMPT


class _StubProc:
    tokenizer = type("T", (), {"convert_tokens_to_ids": staticmethod(lambda s: {"A": 1, "B": 2}[s])})()

    def build(self, frames_a, frames_b, prompt):
        return {"which_first": "A" if frames_a.mean() > frames_b.mean() else "B"}


class _StubModel:
    def next_token_logits(self, inputs):
        logits = torch.full((10,), -10.0)
        # prefer the brighter clip regardless of ordering
        logits[1 if inputs["which_first"] == "A" else 2] = 5.0
        return logits


def test_p_win_prefers_brighter_clip_and_is_order_symmetric():
    judge = PairwiseVLMJudge(_StubModel(), _StubProc(), device="cpu")
    bright, dark = torch.full((4, 3, 8, 8), 0.9), torch.full((4, 3, 8, 8), 0.1)
    assert judge.p_win(bright, dark, "a cat") > 0.99
    assert judge.p_win(dark, bright, "a cat") < 0.01


def test_prompt_mentions_overall_quality_not_geometry():
    assert "overall quality" in JUDGE_PROMPT and "geometr" not in JUDGE_PROMPT.lower()
