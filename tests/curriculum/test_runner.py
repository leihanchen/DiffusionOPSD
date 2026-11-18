from dataclasses import asdict
import json
from pathlib import Path
import sys
import subprocess
import time

import pytest

from config.curriculum_cosmos import execution_profile, get_runner_config
from diffusionopsd.curriculum.records import PromptRecord, read_manifest
from diffusionopsd.curriculum.runner import run_experiment

WORKER = Path(__file__).parent / 'fixtures/runner_worker.py'


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2))
    return str(Path(path).resolve())


def manifest(tmp_path, split, n=2):
    rows = [PromptRecord(f'{split}-{i}', f'{split}-{i}', None, split, f'A static {split} sculpture {i}. Camera orbits.',
                        'orbit', {}, 'initial', 0) for i in range(n)]
    return save(tmp_path / f'{split}.json', [asdict(r) for r in rows])


@pytest.fixture
def settings(tmp_path):
    cfg = get_runner_config()
    model = tmp_path / ('a'*40)
    model.mkdir()
    cfg.update(initial_manifest=manifest(tmp_path, 'train'), development_manifest=manifest(tmp_path, 'development'),
        test_manifest=manifest(tmp_path, 'test'), training_config=save(tmp_path / 'training.json', {}),
        rounds=2, prompts_per_round=2, development_count=2, development_seeds=1,
        allocation={'retained': 1, 'edited': 1, 'rehearsal': 0, 'new': 0},
        trainer_command=[sys.executable, str(WORKER), 'train'],
        proposer_command=[sys.executable, str(WORKER), 'vlm'],
        diagnosis_command=[sys.executable, str(WORKER), 'vlm'],
        inventory_command=[sys.executable, str(WORKER), 'inventory'],
        execution_profile=execution_profile('narval_gemma12'),
        environment={'CURRICULUM_VLM_MODEL_PATH': str(model), 'CURRICULUM_VLM_PYTHON': sys.executable,
                     'CURRICULUM_VLM_REVISION': 'a'*40},
        budget={'max_wall_seconds': 1200, 'max_gpu_hours': {'A100': 1.},
                'phase_estimate_seconds': {'train_one_pass': 1, 'diagnose_clips': 1, 'prepare_next_manifest': 1}},
        phase_timeout_seconds=30)
    path = tmp_path / 'runner.json'
    save(path, cfg)
    return cfg, path


def result(run_dir):
    return json.loads((run_dir / 'state.json').read_text())


def test_two_round_numeric_loop_creates_edits_and_preserves_visits(settings, tmp_path):
    cfg, path = settings
    root = tmp_path / 'run'
    run_experiment(path, root)
    state = result(root)
    assert state['status'] == 'complete'
    assert state['completed_rounds'] == 2
    first = read_manifest(root / 'round_000/manifest.json')
    second = read_manifest(root / 'round_001/manifest.json')
    assert len(first) == len(second) == 2
    assert {r.source for r in second} == {'retained', 'edited'}
    assert next(r for r in second if r.source == 'edited').parent_id in {r.prompt_id for r in first}
    for i in range(2):
        round_state = json.loads((root / f'round_{i:03}/training/round_state.json').read_text())
        assert round_state['round_stats']['prompt_visits'] == 2
        assert round_state['round_stats']['attempted_clips'] == 8
    before = (root / 'state.json').read_bytes()
    run_experiment(path, root)
    assert (root / 'state.json').read_bytes() == before
    assert state['costs']['gpu_hours']['A100'] > 0


@pytest.mark.parametrize('phase', ['prepare_manifest', 'train_one_pass', 'evaluate_development',
                                  'diagnose_clips', 'summarize', 'prepare_next_manifest'])
def test_resume_after_each_committed_phase_matches_uninterrupted(settings, tmp_path, monkeypatch, phase):
    from diffusionopsd.curriculum.runner import CurriculumRunner

    _, path = settings
    root, full = tmp_path / 'interrupted', tmp_path / 'full'
    run_experiment(path, full)
    original = CurriculumRunner.finish_phase
    crashed = False

    def interrupted(self, name, *args, **kwargs):
        nonlocal crashed
        original(self, name, *args, **kwargs)
        if name == phase and not crashed:
            crashed = True
            raise InterruptedError('Simulated controller interruption after committed phase')

    with monkeypatch.context() as context:
        context.setattr(CurriculumRunner, 'finish_phase', interrupted)
        with pytest.raises(InterruptedError):
            run_experiment(path, root)
    run_experiment(path, root)
    assert result(root)['status'] == 'complete'
    assert (root / 'round_001/manifest.json').read_bytes() == (full / 'round_001/manifest.json').read_bytes()
    for i in range(2):
        actual = json.loads((root / f'round_{i:03}/training/training_state.pt').read_text())
        expected = json.loads((full / f'round_{i:03}/training/training_state.pt').read_text())
        assert actual == expected


def test_changed_input_manifest_and_profile_rejected_on_resume(settings, tmp_path):
    cfg, path = settings
    root = tmp_path / 'run'
    run_experiment(path, root)
    modified = json.loads(Path(cfg['initial_manifest']).read_text())
    modified[0]['text'] = 'Changed'
    save(cfg['initial_manifest'], modified)
    with pytest.raises(ValueError, match='changed|hash|differs'):
        run_experiment(path, root)


def test_unresolved_h200_and_overlapping_gpu_claims(settings, tmp_path):
    from diffusionopsd.curriculum.execution import device_claims

    cfg, path = settings
    cfg['execution_profile'] = execution_profile('h200_gemma12_control')
    save(path, cfg)
    with pytest.raises(ValueError):
        run_experiment(path, tmp_path / 'h200')
    with device_claims(tmp_path / 'locks', ['GPU-fixture-0']):
        with pytest.raises(RuntimeError, match='owned|use|locked'):
            with device_claims(tmp_path / 'locks', ['GPU-fixture-0']):
                pass


def test_training_failure_resumes_committed_checkpoint(settings, tmp_path):
    cfg, path = settings
    cfg['trainer_command'] += ['--fail-once-path', str(tmp_path / 'once')]
    save(path, cfg)
    root = tmp_path / 'run'
    with pytest.raises(RuntimeError):
        run_experiment(path, root)
    run_experiment(path, root)
    assert result(root)['status'] == 'complete'
    checkpoint = json.loads((root / 'round_000/training/training_state.pt').read_text())
    assert checkpoint['visits'] == ['train-0', 'train-1']
    assert checkpoint['costs']['training_clips'] == 8
    phase = result(root)['phases']['000/train_one_pass']
    assert phase['attempts'] == 2
    assert phase['attempt_history'][0]['status'] == 'failed'
    assert '17' in phase['attempt_history'][0]['error']


def test_all_rejected_smoke_stops_without_lowering_gates(settings, tmp_path):
    cfg, path = settings
    save(cfg['training_config'], {'reject_all': True})
    with pytest.raises(RuntimeError, match='update|rejected'):
        run_experiment(path, tmp_path / 'run')


def test_shadow_diagnosis_does_not_change_numeric_curriculum(settings, tmp_path):
    cfg, path = settings
    numeric = tmp_path / 'numeric'
    run_experiment(path, numeric)
    cfg['feedback_mode'] = 'visual_shadow'
    save(path, cfg)
    shadow = tmp_path / 'shadow'
    run_experiment(path, shadow)
    assert (numeric / 'round_001/manifest.json').read_bytes() == (shadow / 'round_001/manifest.json').read_bytes()
    assert (shadow / 'round_000/diagnoses.json').is_file()
    for p in shadow.rglob('request_*.json'):
        r = json.loads(p.read_text())
        if isinstance(r, dict) and r.get('mode') == 'propose':
            assert not r['payload']['feedback'].get('diagnoses')


def test_combined_mode_requires_approved_audit_before_training(settings, tmp_path):
    cfg, path = settings
    cfg['feedback_mode'] = 'numeric_and_visual'
    save(path, cfg)
    with pytest.raises(ValueError, match='audit'):
        run_experiment(path, tmp_path / 'combined')


@pytest.mark.parametrize('target', ['manifest', 'checkpoint', 'snapshot', 'profile'])
def test_resume_rejects_tampered_outputs(settings, tmp_path, target):
    cfg, path = settings
    root = tmp_path / 'run'
    run_experiment(path, root)
    targets = dict(manifest=root / 'round_001/manifest.json',
                   checkpoint=root / 'round_000/training/training_state.pt',
                   snapshot=root / 'test.json', profile=root / 'profile.json')
    targets[target].write_text('{}')
    with pytest.raises(ValueError, match='hash|changed|snapshot'):
        run_experiment(path, root)


def test_changed_profile_rejected(settings, tmp_path):
    cfg, path = settings
    root = tmp_path / 'run'
    run_experiment(path, root)
    cfg['execution_profile']['vlm_device_ids'] = [1]
    save(path, cfg)
    with pytest.raises(ValueError, match='changed'):
        run_experiment(path, root)


def test_single_writer_rejected(settings, tmp_path):
    from diffusionopsd.curriculum.execution import file_lock

    _, path = settings
    root = tmp_path / 'run'
    with file_lock(root / 'run.lock'):
        with pytest.raises(RuntimeError, match='locked'):
            run_experiment(path, root)


def test_budget_stops_before_worker_launch(settings, tmp_path):
    cfg, path = settings
    cfg['budget']['max_gpu_hours']['A100'] = 0.00001
    save(path, cfg)
    root = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='Budget'):
        run_experiment(path, root)
    assert not (root / 'round_000/training/training_state.pt').exists()


def test_fixed_arm_retains_prompt_content(settings, tmp_path):
    cfg, path = settings
    cfg['curriculum_mode'] = 'fixed'
    save(path, cfg)
    root = tmp_path / 'run'
    run_experiment(path, root)
    first = read_manifest(root / 'round_000/manifest.json')
    second = read_manifest(root / 'round_001/manifest.json')
    assert [record.text for record in first] == [record.text for record in second]
    assert all(record.source == 'retained' for record in second)
    assert not list(root.rglob('receipt.json'))


def test_worker_timeout_preserves_failed_phase(settings, tmp_path):
    cfg, path = settings
    cfg['trainer_command'] += ['--wait-marker', str(tmp_path / 'marker')]
    cfg['phase_timeout_seconds'] = 1
    cfg['environment']['FIXTURE_WAIT_SECONDS'] = '5'
    save(path, cfg)
    root = tmp_path / 'run'
    with pytest.raises(subprocess.TimeoutExpired):
        run_experiment(path, root)
    assert result(root)['phases']['000/train_one_pass']['status'] == 'failed'


def test_snapshot_test_partition_never_enters_proposals(settings, tmp_path):
    _, path = settings
    root = tmp_path / 'run'
    run_experiment(path, root)
    for request_path in root.rglob('request_*.json'):
        text = request_path.read_text()
        assert 'test sculpture' not in text
        assert 'development sculpture' not in text


def test_controller_death_keeps_worker_lock_and_recovers(settings, tmp_path):
    from diffusionopsd.curriculum.execution import file_lock

    cfg, path = settings
    marker = tmp_path / 'marker'
    cfg['trainer_command'] += ['--wait-marker', str(marker)]
    cfg['environment']['FIXTURE_WAIT_SECONDS'] = '2'
    save(path, cfg)
    root = tmp_path / 'run'
    script = Path(__file__).resolve().parents[2] / 'scripts/run_curriculum.py'
    process = subprocess.Popen([sys.executable, str(script), '--config', str(path), '--run-dir', str(root)])
    try:
        deadline = time.monotonic() + 30
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert marker.exists()
        process.kill()
        process.wait(timeout=5)
        with pytest.raises(RuntimeError, match='locked'):
            run_experiment(path, root)
        deadline = time.monotonic() + 10
        while True:
            try:
                with file_lock(root / 'run.lock'):
                    break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.05)
        run_experiment(path, root)
        assert result(root)['status'] == 'complete'
        checkpoint = json.loads((root / 'round_000/training/training_state.pt').read_text())
        assert len(checkpoint['visits']) == 2
        assert checkpoint['costs']['development_clips'] == 2
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_vlm_uses_profile_device(settings, tmp_path):
    cfg, path = settings
    cfg['execution_profile']['vlm_device_ids'] = [1]
    save(path, cfg)
    root = tmp_path / 'run'
    run_experiment(path, root)
    response = next(root.rglob('response_0.json'))
    assert json.loads(response.read_text())['usage']['device'] == 'cuda:1'


@pytest.mark.parametrize('override', ['environment', 'command', 'budget'])
def test_vlm_profile_conflicts_rejected_before_training(settings, tmp_path, override):
    cfg, path = settings
    if override == 'environment':
        cfg['environment']['CURRICULUM_VLM_DEVICE'] = 'cuda:1'
    elif override == 'command':
        cfg['proposer_command'] += ['--device=cuda:1']
    else:
        cfg['execution_profile']['input_token_cap'] = 4096
    save(path, cfg)
    root = tmp_path / 'run'
    with pytest.raises(ValueError, match='profile|Profile'):
        run_experiment(path, root)
    assert not (root / 'round_000/training').exists()
