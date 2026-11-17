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


def diagnosis_audit(rows: list[dict], model_id: str, model_revision: str, *, expected_count: int = 12) -> dict:
    """Summarize a declared pilot size; approval remains a human action."""
    from collections import Counter, defaultdict

    from diffusionopsd.curriculum.proposer import content_hash

    if expected_count not in {12, 24} or len(rows) != expected_count or len({r['clip_id'] for r in rows}) != expected_count:
        raise ValueError(f'Diagnosis audit requires {expected_count} distinct clips (supported pilot sizes: 12 or 24)')
    templates = {r.get('diagnosis_template_hash') for r in rows}
    if len(templates) != 1 or not next(iter(templates)) or any(not r.get('request_hash') for r in rows):
        raise ValueError('Audit requires frozen request identities and one diagnosis template')
    totals = defaultdict(Counter)
    statuses = {'observed_pass', 'observed_fail', 'unknown'}
    for row in rows:
        if not isinstance(row.get('reviewer'), str) or not row['reviewer'].strip():
            raise ValueError('Independent reviewer identity is required')
        labels = row['labels']
        if not labels or any(v not in statuses for v in labels.values()):
            raise ValueError('Every audited requirement needs an independent label')
        prediction = validate_diagnosis(row['prediction'], row['frame_ids'], list(labels))
        for judgment in prediction['requirements']:
            req, observed = judgment['requirement_id'], judgment['status']
            count = totals[req]
            count['count'] += 1
            count['disagreements'] += observed != labels[req]
            count['unknown'] += observed == 'unknown'
            count['false_pass'] += labels[req] == 'observed_fail' and observed == 'observed_pass'
            count['false_fail'] += labels[req] == 'observed_pass' and observed == 'observed_fail'
            count['overconfident_on_unknown'] += labels[req] == 'unknown' and observed != 'unknown'
            count['unknown_on_decidable'] += labels[req] != 'unknown' and observed == 'unknown'
            count['missing_evidence'] += not judgment['frame_ids']
            count['label_' + labels[req]] += 1
            count[labels[req] + '->' + observed] += 1
    coverage = {k: sorted(status for status in statuses if v['label_' + status] == 0) for k, v in totals.items()}
    report = dict(model_id=model_id, model_revision=model_revision, clip_count=len(rows),
                  required_clip_count=expected_count, audit_scope='pilot',
                  diagnosis_template_hash=next(iter(templates)),
                  request_hashes=[r['request_hash'] for r in rows],
                  coverage_complete=all(not missing for missing in coverage.values()),
                  input_hash=content_hash(rows), requirements={
                      k: dict(v) | {'missing_human_classes': coverage[k],
                                    'agreement_rate': 1 - v['disagreements'] / v['count'],
                                    'unknown_rate': v['unknown'] / v['count'],
                                    'disagreement_rate': v['disagreements'] / v['count']}
                      for k, v in totals.items()})
    report['report_hash'] = content_hash(report)
    return report | {'approved': False, 'approved_by': ''}


def validate_audit_approval(path, model_id: str, revision: str) -> dict:
    import json
    from pathlib import Path

    from diffusionopsd.curriculum.proposer import content_hash

    from diffusionopsd.curriculum.vlm import _template

    report = json.loads(Path(path).read_text())
    active_template_hash = _template({'mode': 'diagnose'})[1]
    core = {k: v for k, v in report.items() if k not in {'report_hash', 'approved', 'approved_by'}}
    if (report.get('model_id') != model_id or report.get('model_revision') != revision
            or report.get('required_clip_count') not in {12, 24}
            or report.get('clip_count') != report.get('required_clip_count')
            or report.get('audit_scope') != 'pilot'
            or report.get('diagnosis_template_hash') != active_template_hash
            or len(report.get('request_hashes', [])) != report.get('clip_count')
            or report.get('approved') is not True
            or not isinstance(report.get('approved_by'), str) or not report['approved_by'].strip()
            or report.get('report_hash') != content_hash(core)):
        raise ValueError('Diagnosis audit requires reviewed approval for this model revision and unchanged report')
    return report
