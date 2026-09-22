"""Pairwise VLM quality judge used only as a hard mask on omega and as a held-out metric (spec Sec. 4.4)."""

from __future__ import annotations

import os

import torch

JUDGE_PROMPT = (
    "You are shown two short video clips, A then B, both generated for the prompt: \"{prompt}\". "
    "Judge overall quality: visual fidelity, temporal smoothness, absence of artifacts, and how well the clip "
    "matches the prompt. Answer with a single letter, A or B, for the better clip."
)


class PairwiseVLMJudge:
    def __init__(self, model, processor, device, num_frames: int = 8):
        self.model, self.processor, self.device, self.num_frames = model, processor, device, num_frames
        tok = processor.tokenizer
        self.id_a, self.id_b = tok.convert_tokens_to_ids("A"), tok.convert_tokens_to_ids("B")

    def _subsample(self, clip: torch.Tensor) -> torch.Tensor:
        idx = torch.linspace(0, clip.shape[0] - 1, self.num_frames).round().long()
        return clip[idx]

    @torch.no_grad()
    def _p_first(self, first, second, prompt) -> float:
        inputs = self.processor.build(self._subsample(first), self._subsample(second), JUDGE_PROMPT.format(prompt=prompt))
        logits = self.model.next_token_logits(inputs)
        two = torch.stack([logits[self.id_a], logits[self.id_b]]).float()
        return float(torch.softmax(two, 0)[0])

    def p_win(self, clip_a, clip_b, prompt) -> float:
        return 0.5 * (self._p_first(clip_a, clip_b, prompt) + (1.0 - self._p_first(clip_b, clip_a, prompt)))


class _QwenProcessorAdapter:
    def __init__(self, processor):
        self.processor, self.tokenizer = processor, processor.tokenizer

    def build(self, frames_a, frames_b, text):
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": [f for f in frames_a]}, {"type": "video", "video": [f for f in frames_b]},
            {"type": "text", "text": text}]}]
        chat = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[chat], videos=[frames_a, frames_b], return_tensors="pt")


class _QwenModelAdapter:
    def __init__(self, model, device):
        self.model, self.device = model, device

    def next_token_logits(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(**inputs).logits[0, -1]


def load_qwen25_vl(device) -> PairwiseVLMJudge:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    path = os.path.join(os.environ["VIDEO_REWARD_CKPT_PATH"], "Qwen2.5-VL-7B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(path)
    return PairwiseVLMJudge(_QwenModelAdapter(model, device), _QwenProcessorAdapter(proc), device)
