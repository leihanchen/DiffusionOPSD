"""CLI subprocess checks with no production test hooks."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'scripts/train_opsd_video_minimax_h3.py'


@pytest.mark.parametrize('name,value', [('video_shift', 11.), ('audio_shift', 4.)])
def test_allocation_rejects_mismatching_shifts_without_cuda(name, value, monkeypatch):
    import runpy
    import torch
    entry = runpy.run_path(str(ENTRY))
    cfg = entry['load_config']('config/minimax_h3_t2va.py')
    setattr(cfg, name, value)
    monkeypatch.setattr(torch.cuda, 'init', lambda: pytest.fail('preflight allocated CUDA'))
    with pytest.raises(ValueError, match='shift'):
        entry['allocation'](cfg)


def subprocess_code(code, *args):
    return subprocess.run([sys.executable, '-c', code, *map(str, args)], cwd=ROOT,
                          env=os.environ.copy(), text=True, capture_output=True, timeout=90)


@pytest.mark.parametrize('visible,scheduler', [('4,6,7', ''), ('0,1,2', '4,6,7')])
def test_cli_preflight_mapping_without_cuda_or_models(visible, scheduler):
    result = subprocess_code('''
import runpy, torch, os, sys
m = runpy.run_path('scripts/train_opsd_video_minimax_h3.py')
cfg = m['load_config']('config/minimax_h3_t2va.py')
def forbidden(*a, **kw): raise AssertionError('CUDA/model allocation')
torch.cuda.init = forbidden
torch.cuda.device_count = forbidden
import diffusionopsd.video.h3_conditioning as c
c.validate_h3_assets = lambda *a: dict(identity={'runtime': 'pinned'}, weight_inventory={'transformer': {'tensor_bytes': 42}})
m['preflight'].__globals__['runtime_contract'] = lambda: {'test': True}
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
os.environ['SLURM_JOB_GPUS'] = sys.argv[2]
os.environ.pop('SLURM_STEP_GPUS', None)
import json
print(json.dumps(m['preflight'](cfg)))
''', visible, scheduler)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert [row['physical_device'] for row in report['rank_map']] == (['4', '6', '7'] if not scheduler else [None]*3)
    if scheduler:
        assert report['rank_map'][0]['scheduler_allocation_tokens'] == ['4', '6', '7']
        assert [row['cuda_visible_token'] for row in report['rank_map']] == ['0', '1', '2']
    assert [row['role'] for row in report['rank_map']] == ['policy', 'policy', 'reward']
    assert report['weight_inventory']['transformer']['tensor_bytes'] == 42
    assert report['timeouts'] == {'collective_timeout_s': 3600., 'service_timeout_s': 600.}


@pytest.mark.parametrize('preset', ['minimax_h3_t2va', 'minimax_h3_t2va_smoke', 'minimax_h3_t2va_multinode'])
def test_h3_presets_resolve_explicit_timeout_budgets(preset):
    import importlib
    from diffusionopsd.video.h3_fsdp import resolve_h3_timeouts
    cfg = importlib.import_module(f'config.{preset}').get_config()
    assert cfg.collective_timeout_s == 3600.
    assert cfg.service_timeout_s == 600.
    assert resolve_h3_timeouts(cfg) == {'collective_timeout_s': 3600., 'service_timeout_s': 600.}


@pytest.mark.parametrize('explicit', [False, True])
def test_preflight_reports_resolved_legacy_or_custom_timeouts(explicit, monkeypatch):
    import runpy
    import torch
    from diffusionopsd.video import h3_conditioning
    entry = runpy.run_path(str(ENTRY))
    cfg = entry['load_config']('config/minimax_h3_t2va.py')
    if explicit:
        cfg.collective_timeout_s, cfg.service_timeout_s = 4200., 900.
    else:
        del cfg['collective_timeout_s']
        del cfg['service_timeout_s']
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: pytest.fail('CUDA touched'))
    monkeypatch.setattr(h3_conditioning, 'validate_h3_assets', lambda *args: {})
    monkeypatch.setitem(entry['preflight'].__globals__, 'runtime_contract', lambda: {})
    assert entry['preflight'](cfg)['timeouts'] == {
        'collective_timeout_s': 4200. if explicit else 3600.,
        'service_timeout_s': 900. if explicit else 600.}


@pytest.mark.parametrize('name,value', [
    (name, value) for name in ('collective_timeout_s', 'service_timeout_s')
    for value in (float('nan'), float('inf'), 0., -1.)
] + [('collective_timeout_s', 600.), ('collective_timeout_s', 599.),
     ('service_timeout_s', 3600.), ('service_timeout_s', 4000.)])
def test_preflight_rejects_timeout_budgets_before_cuda_or_assets(name, value, monkeypatch):
    import runpy
    import torch
    from diffusionopsd.video import h3_conditioning
    entry = runpy.run_path(str(ENTRY))
    cfg = entry['load_config']('config/minimax_h3_t2va.py')
    setattr(cfg, name, value)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: pytest.fail('CUDA touched'))
    monkeypatch.setattr(h3_conditioning, 'validate_h3_assets', lambda *args: pytest.fail('assets touched'))
    with pytest.raises(ValueError, match='timeout'):
        entry['preflight'](cfg, training=True)


def test_runtime_feature_contract_without_cuda_or_nvml_initialization():
    result = subprocess_code('''
import runpy, torch, pynvml
def forbidden(*args, **kwargs): raise AssertionError('GPU initialization')
torch.cuda.init=forbidden
torch.cuda.device_count=forbidden
pynvml.nvmlInit=forbidden
m=runpy.run_path('scripts/train_opsd_video_minimax_h3.py')
assert m['runtime_contract']()['fsdp_ignored_params']
''')
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('mutation, message', [
    ("cfg.workflow='t2v'", 't2va'),
    ("cfg.num_frames=125", '124'),
    ("os.environ['CUDA_VISIBLE_DEVICES']='0,1'", 'visible'),
    ("os.environ['WORLD_SIZE']='5'", 'WORLD_SIZE'),
    ("cfg.walltime_seconds=float('nan')", 'finite'),
    ("os.environ['SLURM_JOB_GPUS']='4,6'", 'physical'),
])
def test_invalid_configuration_or_allocation_precedes_assets(mutation, message):
    result = subprocess_code('''
import runpy, os
m=runpy.run_path('scripts/train_opsd_video_minimax_h3.py')
cfg=m['load_config']('config/minimax_h3_t2va.py')
os.environ['CUDA_VISIBLE_DEVICES']='4,6,7'
''' + mutation + "\nm['preflight'](cfg)\n")
    assert result.returncode != 0
    assert message in result.stderr


def test_absent_cache_cli_fails_closed(tmp_path):
    config = tmp_path/'cfg.py'
    config.write_text("from config.minimax_h3_t2va import get_config as base\ndef get_config():\n"
                      "    cfg=base()\n    cfg.model='/absent-h3-assets'\n    return cfg\n")
    result = subprocess.run([sys.executable, str(ENTRY), '--config', str(config), '--preflight'],
                            cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert result.returncode != 0
    assert 'Missing' in result.stderr or 'missing' in result.stderr


def test_missing_manifest_cli_with_valid_external_asset_identity(tmp_path):
    result = subprocess_code('''
import runpy, sys
import diffusionopsd.video.h3_conditioning as c
m=runpy.run_path('scripts/train_opsd_video_minimax_h3.py')
cfg=m['load_config']('config/minimax_h3_t2va.py')
cfg.conditioning_cache=sys.argv[1]
c._asset_identity=lambda cfg: {}
m['main'].__globals__['load_config']=lambda path: cfg
m['main'](['--preflight'])
''', tmp_path)
    assert result.returncode != 0
    assert 'Missing conditioning manifest' in result.stderr


def test_injected_tiny_cli_uses_real_runner(tmp_path):
    result = subprocess_code('''
import runpy, pytest, sys
from pathlib import Path
m=runpy.run_path('tests/video/test_h3_training.py')
with pytest.MonkeyPatch.context() as patch:
    m['test_runner_eight_global_evaluation_records_and_checkpoint_boundary'](
        Path(sys.argv[1]), patch, 2, False, 1, entrypoint=True)
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_launcher_survives_usr1_and_workers_handle_warning(tmp_path):
    worker = tmp_path/'worker.py'
    worker.write_text('''
import os, signal, sys, time
from pathlib import Path
root=Path(sys.argv[1]); rank=os.environ['RANK']
stop=False
def flag(*args):
    global stop
    stop=True
signal.signal(signal.SIGUSR1, flag)
(root / ('ready-'+rank)).write_text(str(os.getpid()))
deadline=time.monotonic()+30
while not stop and time.monotonic()<deadline: time.sleep(.02)
assert stop, 'warning did not reach worker'
(root / ('stopped-'+rank)).write_text('clean')
''')
    process = subprocess.Popen([sys.executable, str(ENTRY), '--launch', '--standalone',
                                '--nproc_per_node=2', str(worker), str(tmp_path)], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    try:
        deadline = time.monotonic()+45
        while len(list(tmp_path.glob('ready-*'))) != 2 and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert len(list(tmp_path.glob('ready-*'))) == 2
        # Elastic starts workers in separate sessions. Slurm job-step delivery
        # signals each process; reproduce that rather than a Unix process group.
        os.kill(process.pid, signal.SIGUSR1)
        assert process.poll() is None
        for ready in tmp_path.glob('ready-*'):
            os.kill(int(ready.read_text()), signal.SIGUSR1)
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stdout+stderr
        assert len(list(tmp_path.glob('stopped-*'))) == 2
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=10)


@pytest.mark.parametrize('preflight_exit', [0, 7])
def test_slurm_submission_preflight_precedes_or_prevents_sbatch(tmp_path, preflight_exit):
    commands = tmp_path/'commands'
    commands.mkdir()
    journal = tmp_path/'commands.jsonl'
    runtime = commands/'bash'
    runtime.write_text('''#!/bin/sh
case "$*" in
  *--preflight*)
    echo "preflight:gpu=$H3_USE_GPU:offline=$HF_HUB_OFFLINE" >> "$H3_TEST_JOURNAL"
    exit "$H3_TEST_PREFLIGHT_EXIT" ;;
  *' -c '*) echo topology >> "$H3_TEST_JOURNAL"; echo 3 ;;
  *) echo unexpected-runtime-command >&2; exit 99 ;;
esac
''')
    scheduler = commands/'sbatch'
    scheduler.write_text('#!/bin/sh\necho sbatch >> "$H3_TEST_JOURNAL"\n')
    runtime.chmod(0o755)
    scheduler.chmod(0o755)
    environment = dict(os.environ, PATH=f'{commands}:{os.environ["PATH"]}',
        H3_ACCOUNT='test-account', H3_PARTITION='test-partition', H3_USE_GPU='1',
        HF_HUB_OFFLINE='0', H3_TEST_JOURNAL=str(journal), H3_TEST_PREFLIGHT_EXIT=str(preflight_exit))
    environment.pop('SLURM_JOB_ID', None)
    result = subprocess.run(['/bin/bash', str(ROOT/'scripts/train_opsd_video_minimax_h3.slurm')],
        cwd=ROOT, env=environment, text=True, capture_output=True, timeout=15)
    events = journal.read_text().splitlines()
    assert result.returncode == preflight_exit, result.stdout+result.stderr
    assert 'preflight:gpu=0:offline=1' in events
    if preflight_exit:
        assert 'sbatch' not in events
    else:
        assert events.index('preflight:gpu=0:offline=1') < events.index('sbatch')


@pytest.mark.parametrize('nodes', [1, 2])
def test_node_major_cli_mapping_and_global_world_size(nodes):
    result = subprocess_code('''
import runpy, os, json, sys
m=runpy.run_path('scripts/train_opsd_video_minimax_h3.py')
c=m['load_config']('config/minimax_h3_t2va.py'); c.num_nodes=int(sys.argv[1])
os.environ.update(CUDA_VISIBLE_DEVICES='0,1,2', WORLD_SIZE=str(c.num_nodes*3),
 LOCAL_WORLD_SIZE='3', RANK=str((c.num_nodes-1)*3+1), LOCAL_RANK='1', SLURM_JOB_NUM_NODES=str(c.num_nodes))
os.environ.pop('SLURM_JOB_GPUS', None); os.environ.pop('SLURM_STEP_GPUS', None)
topology, mapping=m['allocation'](c)
assert mapping[0]['rank']==(c.num_nodes-1)*3
assert topology['evaluation_rounds']==__import__('math').ceil(8/(c.num_nodes*2))
print(json.dumps(mapping))
''', nodes)
    assert result.returncode == 0, result.stderr


def test_actual_cli_defaults_to_allocation_deadline_before_preflight():
    result = subprocess_code('''
import runpy, os, time
m=runpy.run_path('scripts/train_opsd_video_minimax_h3.py')
deadline=int(time.time())+600
os.environ['SLURM_JOB_END_TIME']=str(deadline)
def preflight(c, **kw):
 assert 590<c.walltime_seconds<=600
 assert c.walltime_deadline==deadline
 return {}
m['main'].__globals__['preflight']=preflight
m['main'](['--preflight'])
''')
    assert result.returncode == 0, result.stderr


def test_verification_cli_keeps_allocation_deadline_through_preflight():
    result = subprocess_code('''
import runpy, os, time
import scripts.train_opsd_video_minimax_h3 as entry
import scripts.prepare_minimax_h3 as prepare
import diffusionopsd.video.h3_training as training
from config.minimax_h3_t2va import get_config
c=get_config()
deadline=int(time.time())+600
os.environ['SLURM_JOB_END_TIME']=str(deadline)
prepare.load_config=lambda path: c
def preflight(cfg, **kw):
 assert cfg.walltime_deadline==deadline
 cfg.preflight_deadline=cfg.walltime_deadline
entry.preflight=preflight
def run(cfg, **kw):
 assert cfg.walltime_deadline==cfg.preflight_deadline==deadline
 return dict(status='stage_measured', reached_phase='load')
training.run_h3_training=run
runpy.run_path('scripts/check_minimax_h3_update.py')['main'](
 ['--config','config/minimax_h3_t2va.py','--phase','load','--run-dir','unused'])
''')
    assert result.returncode == 0, result.stderr


def test_allocated_launcher_one_agent_per_node_and_finite_default(tmp_path):
    commands = tmp_path/'commands'
    commands.mkdir()
    journal = tmp_path/'args'
    runtime = commands/'bash'
    runtime.write_text('''#!/bin/sh
case "$*" in
 *--preflight*) exit 0 ;;
 *' -c '*) echo '3 2' ;;
 *) exit 99 ;;
esac
''')
    scheduler = commands/'srun'
    scheduler.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$H3_TEST_JOURNAL"\n')
    runtime.chmod(0o755)
    scheduler.chmod(0o755)
    environment = dict(os.environ, PATH=f'{commands}:{os.environ["PATH"]}', SLURM_JOB_ID='123',
        SLURM_JOB_NUM_NODES='2', SLURM_JOB_END_TIME='9999999999', H3_RUN_DIR=str(tmp_path/'run'),
        H3_RDZV_ENDPOINT='host0:29550', H3_TEST_JOURNAL=str(journal))
    environment.pop('H3_WALLTIME_SECONDS', None)
    result = subprocess.run(['/bin/bash', str(ROOT/'scripts/train_opsd_video_minimax_h3.slurm')],
        cwd=ROOT, env=environment, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout+result.stderr
    args = journal.read_text().splitlines()
    assert '--nodes=2' in args and '--ntasks=2' in args and '--ntasks-per-node=1' in args
    assert '--nnodes=2' in args and '--nproc_per_node=3' in args
    assert '--rdzv_endpoint=host0:29550' in args
    assert '--walltime-seconds' not in args, 'default must reach worker deadline resolver'
