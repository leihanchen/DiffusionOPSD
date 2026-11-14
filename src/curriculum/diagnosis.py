"""Ordered frame selection and evidence-linked, explicitly attributed observations."""

from __future__ import annotations


def select_frames(frame_count: int, limit: int = 8) -> list[int]:
    if type(frame_count) is not int or frame_count < 1 or type(limit) is not int or limit < 2:
        raise ValueError('Frame count must be positive; limit must include both endpoints')
    n = min(frame_count, limit)
    if n == 1:
        return [0]
    return [round(i * (frame_count - 1) / (n - 1)) for i in range(n)]


def validate_diagnosis(response: dict, frame_ids: list[int], requirement_ids: list[str]) -> dict:
    if (not isinstance(response, dict) or set(response) != {'requirements'}
            or not isinstance(response['requirements'], list)):
        raise ValueError('Diagnosis requires a requirements array')
    if not frame_ids or len(frame_ids) != len(set(frame_ids)) or frame_ids != sorted(frame_ids):
        raise ValueError('Source frame IDs must be unique and ordered')
    if not requirement_ids or len(set(requirement_ids)) != len(requirement_ids):
        raise ValueError('Requirement IDs must be nonempty and unique')
    seen = set()
    for row in response['requirements']:
        if not isinstance(row, dict) or set(row) != {'requirement_id', 'status', 'frame_ids', 'reason'}:
            raise ValueError('Invalid diagnosis fields')
        req, status, evidence = row['requirement_id'], row['status'], row['frame_ids']
        if req not in requirement_ids or req in seen:
            raise ValueError('Unknown or repeated requirement')
        seen.add(req)
        if status not in {'observed_pass', 'observed_fail', 'unknown'}:
            raise ValueError('Only attributed observations and unknown are supported')
        if (not isinstance(evidence, list) or any(type(i) is not int or i not in frame_ids for i in evidence)
                or len(evidence) != len(set(evidence))):
            raise ValueError('Evidence must reference unique selected frames')
        if status != 'unknown' and not evidence:
            raise ValueError('An observation requires visible frame evidence')
        if not isinstance(row['reason'], str) or not row['reason'].strip():
            raise ValueError('Every judgment needs a reason, including unknown')
    if seen != set(requirement_ids):
        raise ValueError('Missing requirement judgments')
    return response


def save_frame_evidence(clip, output_dir, *, fps: float, longest_side: int = 448, limit: int = 8) -> dict:
    """Persist selected RGB frames from [T,C,H,W] without upscaling."""
    import math
    from pathlib import Path

    import torch
    from PIL import Image

    if clip.ndim != 4 or clip.shape[1] != 3 or not math.isfinite(fps) or fps <= 0 or longest_side < 1:
        raise ValueError('Expected RGB clip, positive FPS and frame-size limit')
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in select_frames(clip.shape[0], limit):
        pixels = (clip[index].detach().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(pixels)
        image.thumbnail((longest_side, longest_side))
        path = output_dir / f'frame_{index:06d}.png'
        image.save(path)
        frames.append(dict(frame_id=index, timestamp=index / fps, path=str(path)))
    return dict(frame_count=clip.shape[0], native_fps=fps, frames=frames)
