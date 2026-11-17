"""Descriptive feedback and a declared deterministic numeric allocation heuristic."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
import statistics

from diffusionopsd.curriculum.records import SUBTYPES


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _summary(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get('valid') is True and _finite(r.get('geo'))]
    values = [r['geo'] for r in valid]
    observations = defaultdict(Counter)
    for r in rows:
        for judgment in r.get('diagnosis', {}).get('requirements', []):
            status = judgment['status']
            if status not in {'observed_pass', 'observed_fail', 'unknown'}:
                raise ValueError('Unsupported observation status')
            observations[judgment['requirement_id']][status] += 1
    return dict(attempts=len(rows), count=len(values),
                mean_geo=statistics.mean(values) if values else None,
                std_geo=statistics.pstdev(values) if values else None,
                identity_rejected=sum(r.get('identity_retained') is False for r in rows),
                motion_masked=sum(r.get('motion_masked') is True for r in rows),
                quality_masked=sum(r.get('quality_masked') is True for r in rows),
                elapsed_seconds=sum(r.get('elapsed_seconds', 0) for r in rows),
                observations={k: dict(v) for k, v in observations.items()})


def summarize_round(rows: list[dict], development_rows: list[dict]) -> dict:
    groups, subtypes, development = defaultdict(list), defaultdict(list), defaultdict(list)
    for collection, split in ((rows, 'train'), (development_rows, 'development')):
        for row in collection:
            if row.get('split') != split or row.get('subtype') not in SUBTYPES:
                raise ValueError('Feedback must use the intended split and enabled G1 subtype')
    for row in rows:
        groups[row['prompt_id']].append(row)
        subtypes[row['subtype']].append(row)
    for row in development_rows:
        development[row['subtype']].append(row)
    return dict(prompts={k: _summary(v) for k, v in groups.items()},
                subtypes={k: _summary(v) for k, v in subtypes.items()},
                development={k: _summary(v) for k, v in development.items()})


def allocate_next_round(prompt_ids: list[str], summary: dict, seed: int, quotas: dict | None = None) -> dict:
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError('Allocation requires unique prompt IDs')
    quotas = quotas if quotas is not None else dict(retained=8, edited=12, rehearsal=4, new=8)
    if (set(quotas) != {'retained', 'edited', 'rehearsal', 'new'}
            or any(type(value) is not int or value < 0 for value in quotas.values())
            or sum(quotas.values()) < 1):
        raise ValueError('Allocation requires four nonnegative integer quotas')
    ids = list(prompt_ids)
    random.Random(seed).shuffle(ids)
    scores = summary['prompts']

    def key(i):
        value = scores.get(i, {}).get('mean_geo')
        return (0, value) if _finite(value) else (1, 0)

    ids.sort(key=key)  # Stable sort preserves seeded tie breaking.
    retained = ids[:quotas['retained']]
    remaining = ids[len(retained):]
    valid_high = [i for i in reversed(remaining) if _finite(scores.get(i, {}).get('mean_geo'))]
    rehearsal = valid_high[:quotas['rehearsal']]
    edit_parents = [i for i in remaining if i not in rehearsal][:quotas['edited']]
    shortfall = sum(quotas.values()) - quotas['new'] - len(retained) - len(edit_parents) - len(rehearsal)
    return dict(retained=retained, edit_parents=edit_parents, rehearsal=rehearsal,
                new_count=quotas['new'] + shortfall, shortfall=shortfall)


def development_contract(records, training_history, *, seed: int, seeds_per_prompt: int = 2,
                         expected_count: int = 8) -> dict:
    """Freeze independent queries before any adaptive feedback is consumed."""
    import hashlib
    import json

    from diffusionopsd.curriculum.records import manifest_hash, validate_manifest

    from diffusionopsd.curriculum.records import normalized_text

    validate_manifest(records, [])
    other_texts = {normalized_text(r.text) for r in training_history if r.split != 'development'}
    if any(normalized_text(r.text) in other_texts for r in records):
        raise ValueError('Development text duplicates another partition')
    if (len(records) != expected_count or seeds_per_prompt < 1
            or any(r.split != 'development' for r in records)):
        raise ValueError('Development manifest size/split or seed count is invalid')
    lineages = {r.lineage_id for r in training_history if r.split != 'development'}
    if lineages & {r.lineage_id for r in records}:
        raise ValueError('Development lineages overlap another partition')
    queries = [dict(prompt_id=r.prompt_id, text=r.text, subtype=r.subtype, split='development',
                    seed=seed + 1000 + i * seeds_per_prompt + j)
               for i, r in enumerate(records) for j in range(seeds_per_prompt)]
    payload = dict(manifest_hash=manifest_hash(records), queries=queries)
    payload['hash'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload
