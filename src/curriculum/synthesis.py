"""Deterministic next-round assembly from bounded, validated proposals."""

from dataclasses import asdict, replace
from pathlib import Path
import time

from diffusionopsd.curriculum.feedback import allocate_next_round
from diffusionopsd.curriculum.proposer import atomic_json, propose
from diffusionopsd.curriculum.records import validate_manifest


def next_manifest(records, history, summary: dict, *, config: dict, round_index: int,
                  directory: Path, diagnoses: list | None = None, timeout_seconds: float | None = None,
                  lock_fds: tuple[int, ...] = ()):
    if config['curriculum_mode'] == 'fixed':
        return [replace(record, source='retained', round_index=round_index) for record in records]
    allocation = allocate_next_round([record.prompt_id for record in records], summary,
                                     config['seed'] + round_index, config['allocation'])
    atomic_json(directory / 'allocation.json', allocation)
    lookup = {record.prompt_id: record for record in records}
    result = [replace(lookup[identity], source=source, round_index=round_index)
              for source in ('retained', 'rehearsal') for identity in allocation[source]]
    parents = [lookup[identity] for identity in allocation['edit_parents']]
    feedback = summary
    if diagnoses is not None:
        feedback = dict(summary, diagnoses=diagnoses)
    batch_index = 0
    deadline = time.monotonic() + (timeout_seconds or config['phase_timeout_seconds'])
    for source, count in (('edited', len(parents)), ('new', allocation['new_count'])):
        for start in range(0, count, 4):
            size = min(4, count - start)
            request = dict(mode='propose', model_id=config['execution_profile']['vlm_model_id'],
                           model_revision=config['environment']['CURRICULUM_VLM_REVISION'],
                           seed=config['seed'] + round_index * 1000 + batch_index,
                           payload=dict(domain='G1', round_index=round_index, split='train',
                                        count=size, sources={source: size}, feedback=feedback,
                                        id_prefix=f'r{round_index}_batch{batch_index}_',
                                        parents=[asdict(parent) for parent in parents[start:start + size]]
                                        if source == 'edited' else []),
                           history=[asdict(record) for record in [*history, *records, *result]])
            if diagnoses is not None:
                request['diagnosis_audit_path'] = config['diagnosis_audit_path']
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Prompt synthesis phase exhausted its time budget')
            result.extend(propose(request, config['proposer_command'], directory / 'proposals',
                                  remaining / 3, lock_fds=lock_fds))
            batch_index += 1
    validate_manifest(result, [*history, *records])
    if len(result) != config['prompts_per_round']:
        raise ValueError('Next manifest does not satisfy the declared round budget')
    return result
