"""Join independently saved human labels to frozen-model observations."""

from __future__ import annotations

from diffusionopsd.curriculum.diagnosis import diagnosis_audit, validate_diagnosis


def compare_review(manifest: dict, human_rows: list[dict], predictions: list[dict]) -> tuple[dict, list[dict]]:
    """Return an unapproved report and every disagreement; never manufacture labels.

    Predictions follow the ordered worker request batch. Where recorded, the
    request hash also binds each answer to the exact clip/frame/template input.
    """
    clips, requirements = manifest['clips'], manifest['requirements']
    expected = manifest.get('expected_count', 12)
    if (len(clips) != expected or len(predictions) != expected or len(human_rows) != expected
            or len({r['clip_id'] for r in clips}) != expected
            or len({r['clip_id'] for r in human_rows}) != expected):
        raise ValueError('Review requires the declared number of distinct clips, labels, and predictions')
    human = {r['clip_id']: r for r in human_rows}
    if set(human) != {r['clip_id'] for r in clips}:
        raise ValueError('Human clip IDs do not match the review manifest')
    rows, disagreements = [], []
    for clip, prediction in zip(clips, predictions):
        clip_id = clip['clip_id']
        reviewer = human[clip_id]
        frames = [f['frame_id'] for f in clip['frames']]
        if not clip.get('request_hash') or reviewer.get('request_hash') != clip['request_hash']:
            raise ValueError('Human labels do not match the frozen clip/prompt/frame request')
        if (prediction.get('error') or prediction.get('model_id') != manifest['model_id']
                or prediction.get('model_revision') != manifest['model_revision']
                or prediction.get('request_hash') != clip['request_hash']
                or not prediction.get('template_hash')):
            raise ValueError('Model answer does not match the frozen model and clip request')
        if set(reviewer['labels']) != set(requirements) or set(reviewer['human_evidence']) != set(requirements):
            raise ValueError('Human review must label every declared requirement')
        human_judgments = {'requirements': [dict(requirement_id=req, status=reviewer['labels'][req],
            frame_ids=reviewer['human_evidence'][req]['frame_ids'],
            reason=reviewer['human_evidence'][req]['reason']) for req in requirements]}
        validate_diagnosis(human_judgments, frames, requirements)
        judgments = validate_diagnosis(prediction['result'], frames, requirements)['requirements']
        rows.append(dict(clip_id=clip_id, reviewer=reviewer['reviewer'], labels=reviewer['labels'],
                         human_evidence=reviewer['human_evidence'], prediction=prediction['result'], frame_ids=frames,
                         request_hash=clip['request_hash'], diagnosis_template_hash=prediction['template_hash']))
        for judgment in judgments:
            req = judgment['requirement_id']
            if judgment['status'] != reviewer['labels'][req]:
                evidence = reviewer['human_evidence'][req]
                disagreements.append(dict(clip_id=clip_id, requirement=req, human_label=reviewer['labels'][req],
                    gemma_label=judgment['status'], human_reason=evidence['reason'], gemma_reason=judgment['reason'],
                    human_frames=evidence['frame_ids'], gemma_frames=judgment['frame_ids']))
    report = diagnosis_audit(rows, manifest['model_id'], manifest['model_revision'], expected_count=expected)
    return report, disagreements
