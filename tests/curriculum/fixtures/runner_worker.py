"""Subprocess fixture for runner contracts; not evidence of GPU training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if sys.argv[1] == 'inventory':
        print('0, GPU-fixture-0, NVIDIA A100-SXM4-40GB, 40960')
        print('1, GPU-fixture-1, NVIDIA A100-SXM4-40GB, 40960')
        return
    if sys.argv[1] == 'vlm':
        p = argparse.ArgumentParser()
        p.add_argument('--request')
        p.add_argument('--response')
        a = p.parse_args(sys.argv[2:])
        data = json.loads(Path(a.request).read_text())
        requests = data if isinstance(data, list) else [data]
        responses = []
        for r in requests:
            assert 'history' not in r
            payload = r['payload']
            if r['mode'] == 'propose':
                rows = []
                prefix = payload.get('id_prefix', 'candidate_')
                for source, count in payload['sources'].items():
                    for i in range(count):
                        parent = payload.get('parents', [])[i] if source == 'edited' else None
                        key = prefix + str(i)
                        rows.append(dict(prompt_id=key, lineage_id=parent['lineage_id'] if parent else key,
                            parent_id=parent['prompt_id'] if parent else None, split=payload['split'],
                            text=(parent['text'] + ' Add one foreground block.' if parent else 'A new rigid scene ' + key),
                            subtype=parent['subtype'] if parent else 'dolly', source=source,
                            round_index=payload['round_index'], attributes={'edit_axis': 'clutter', 'edit': 'Add one block'}))
                result = rows
            else:
                result = {'requirements': [dict(requirement_id=k, status='unknown', frame_ids=[],
                                                reason='Fixture: no semantic assertion') for k in payload['requirements']]}
            responses.append(dict(result=result, model_id=r['model_id'], model_revision=r['model_revision'],
                usage={'input_tokens': 10, 'output_tokens': 5, 'elapsed_seconds': .01, 'peak_memory_bytes': 0,
                       'device': os.environ.get('CURRICULUM_VLM_DEVICE')},
                attempts=[], template_hash='fixture'))
        write(a.response, responses if isinstance(data, list) else responses[0])
        return
    p = argparse.ArgumentParser()
    for flag in ('config', 'run-dir', 'curriculum-manifest', 'curriculum-history',
                 'previous-round-checkpoint', 'development-manifest', 'fail-once-path', 'wait-marker',
                 'curriculum-execution-profile'):
        p.add_argument('--' + flag)
    p.add_argument('--curriculum-round', type=int)
    p.add_argument('--development-count', type=int)
    p.add_argument('--development-seeds', type=int)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(sys.argv[2:])
    root = Path(a.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / 'training_state.pt'
    if a.wait_marker:
        Path(a.wait_marker).write_text('running')
        time.sleep(float(os.environ.get('FIXTURE_WAIT_SECONDS', '.5')))
    rows = json.loads(Path(a.curriculum_manifest).read_text())
    # Match production canonical manifest_hash, independently of runner implementation.
    manifest_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    previous = json.loads(Path(a.previous_round_checkpoint).read_text()) if a.previous_round_checkpoint else {}
    state = json.loads(checkpoint.read_text()) if a.resume else dict(update=previous.get('update', 0), cursor=0,
        visits=[], policy=previous.get('policy', 0), ema=previous.get('ema', 0),
        costs=previous.get('costs', {'training_clips': 0, 'development_clips': 0, 'reference_clips': 0}))
    config = json.loads(Path(a.config).read_text())
    feedback = []
    for i, record in enumerate(rows):
        for k in range(4):
            feedback.append(dict(prompt_id=record['prompt_id'], split='train', subtype=record['subtype'],
                geo=(i+1)/10, valid=True, identity_retained=True, motion_masked=False, quality_masked=False,
                sampling_attempt=a.curriculum_round * len(rows) + i, clip_index=k,
                evidence={'frame_count': 1, 'native_fps': 8., 'frames': [{'frame_id': 0, 'timestamp': 0.,
                           'path': str((root / f'frame_{i}_{k}.png').resolve())}]}))
            (root / f'frame_{i}_{k}.png').write_bytes(b'fixture evidence, worker does not decode')
        if i < state['cursor']:
            continue
        state['cursor'] += 1
        state['visits'].append(record['prompt_id'])
        state['costs']['training_clips'] += 4
        if not config.get('reject_all', False):
            state['update'] += 1
            state['policy'] += .1
            state['ema'] += .01
        write(checkpoint, state)
        if a.fail_once_path and not Path(a.fail_once_path).exists():
            Path(a.fail_once_path).write_text('failed-once')
            raise SystemExit(17)
    if not state.get('development_done'):
        state['costs']['development_clips'] += a.development_count * a.development_seeds
        state['development_done'] = True
    write(checkpoint, state)
    dev = json.loads(Path(a.development_manifest).read_text())
    dev_rows = [dict(prompt_id=r['prompt_id'], split='development', subtype=r['subtype'], geo=.4, valid=True)
                for r in dev for _ in range(a.development_seeds)]
    write(root / 'training_feedback.json', feedback)
    write(root / 'development_feedback.json', dev_rows)
    write(root / 'costs.json', state['costs'])
    write(root / 'round_state.json', dict(format_version=1, checkpoint_sha256=digest(checkpoint),
        round_index=a.curriculum_round, manifest_hash=manifest_hash, next_index=len(rows), prompt_count=len(rows),
        completed_updates=state['update'], development_done=True,
        round_stats={'prompt_visits': len(rows), 'attempted_clips': len(rows)*4,
                     'successful_updates': 0 if config.get('reject_all') else len(rows)}, costs=state['costs']))


if __name__ == '__main__':
    main()
