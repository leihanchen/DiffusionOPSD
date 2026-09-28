"""VideoReward quality judge used only as a hard mask on omega and as a held-out metric.

The score is KlingTeam VideoReward (KwaiVGI/VideoReward): a Qwen2-VL-2B reward with three
logits, visual quality (VQ), motion quality (MQ), and text alignment (TA). The mask uses VQ
and MQ. The model is never differentiated.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch


def quality_gap(rollout: dict, reference: dict) -> float:
    """Worse of the visual-quality and motion-quality gaps, rollout minus reference."""
    return min(rollout["VQ"] - reference["VQ"], rollout["MQ"] - reference["MQ"])


def gap_to_probability(gap: float) -> float:
    """Map a normalized score gap to (0, 1). Equal clips return 0.5, above tau_q=0.4."""
    return float(torch.sigmoid(torch.tensor(float(gap))))


class VideoRewardJudge:
    """`scorer(clip[T,3,H,W] in [0,1], prompt) -> {VQ, MQ, TA, Overall}`."""

    def __init__(self, scorer, num_frames: int = 8):
        self.scorer = scorer
        self.num_frames = num_frames

    def _subsample(self, clip: torch.Tensor) -> torch.Tensor:
        idx = torch.linspace(0, clip.shape[0] - 1, self.num_frames).round().long()
        return clip[idx]

    @torch.no_grad()
    def score(self, clip: torch.Tensor, prompt: str) -> dict:
        return self.scorer(self._subsample(clip), prompt)

    def p_win(self, clip_a: torch.Tensor, clip_b: torch.Tensor, prompt: str) -> float:
        """Probability-shaped score that clip_a is at least as good as clip_b on VQ and MQ."""
        return gap_to_probability(quality_gap(self.score(clip_a, prompt), self.score(clip_b, prompt)))


def _to_pil(clip: torch.Tensor) -> list:
    """clip [T,3,H,W] float [0,1] -> RGB PIL frames. VideoAlign accepts a list of images as a video."""
    from PIL import Image

    frames = (clip.detach().clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    images = []
    for t in range(frames.shape[0]):
        images.append(Image.fromarray(frames[t].permute(1, 2, 0).numpy(), mode="RGB"))
    return images


class _VideoAlignScorer:
    """Calls the official VideoVLMRewardInference forward on in-memory frames."""

    def __init__(self, inferencer, build_prompt, process_vision_info):
        self.inferencer = inferencer
        self.build_prompt = build_prompt
        self.process_vision_info = process_vision_info

    def __call__(self, clip: torch.Tensor, prompt: str) -> dict:
        inf = self.inferencer
        cfg = inf.data_config
        text = self.build_prompt(prompt, cfg.eval_dim, cfg.prompt_template_type)
        video = {
            "type": "video",
            "video": _to_pil(clip),
            "max_pixels": cfg.max_frame_pixels,
            "sample_type": cfg.sample_type,
        }
        # Match VideoAlign.prepare_batch: fps when the checkpoint leaves num_frames unset.
        if cfg.num_frames is None:
            video["fps"] = cfg.fps
        else:
            video["nframes"] = cfg.num_frames
        chat = [[{"role": "user", "content": [video, {"type": "text", "text": text}]}]]
        image_inputs, video_inputs = self.process_vision_info(chat)
        batch = inf.processor(
            text=inf.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = inf._prepare_inputs(batch)
        logits = inf.model(return_dict=True, **batch)["logits"]
        row = logits[0]
        reward = {"VQ": float(row[0]), "MQ": float(row[1]), "TA": float(row[2])}
        reward = inf._norm(reward)
        reward["Overall"] = reward["VQ"] + reward["MQ"] + reward["TA"]
        return reward


@contextmanager
def _videoalign_import_scope(repo: str):
    """Temporarily resolve VideoAlign's flat imports during single-threaded startup.

    WAFT also imports a top-level ``utils`` package. Restore existing modules
    (including their children) even if VideoAlign initialization fails.
    Callers must retain references to any VideoAlign helpers needed afterward.
    """
    local_names = {path.stem for path in Path(repo).glob("*.py")}

    def is_local(name):
        return name.split(".", 1)[0] in local_names

    previous_modules = {name: module for name, module in sys.modules.copy().items() if is_local(name)}
    previous_path = sys.path[:]
    try:
        for name in previous_modules:
            del sys.modules[name]
        sys.path.insert(0, repo)
        yield
    finally:
        for name in list(sys.modules):
            if is_local(name):
                del sys.modules[name]
        sys.modules.update(previous_modules)
        sys.path[:] = previous_path


def load_video_reward(device, num_frames: int = 8) -> VideoRewardJudge:
    """Load KwaiVGI/VideoReward via the VideoAlign repo cloned under VIDEO_REWARD_CKPT_PATH."""
    root = os.environ["VIDEO_REWARD_CKPT_PATH"]
    repo = os.path.join(root, "VideoAlign")
    ckpt = os.path.join(root, "VideoReward")
    with _videoalign_import_scope(repo):
        from inference import VideoVLMRewardInference
        from prompt_template import build_prompt
        from vision_process import process_vision_info

        inferencer = VideoVLMRewardInference(ckpt, device=device, dtype=torch.bfloat16)
    inferencer.model.requires_grad_(False)
    inferencer.model.eval()
    return VideoRewardJudge(
        _VideoAlignScorer(inferencer, build_prompt, process_vision_info), num_frames=num_frames
    )
