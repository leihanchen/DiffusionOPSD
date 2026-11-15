"""CPU audits reject missing actual execution and corrupted evidence."""
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import wave

import pytest
import torch

from diffusionopsd.video.h3_verification import (
    H3Verification, audit_adapter_update, audit_audio, audit_query, host_cgroup, native_audio_layout, resource_evidence,
)
from diffusionopsd.video.h3_checkpoint import _digest, _rng_state


def adapters():
    return {'block.lora_A.default.weight': torch.ones(2), 'block.lora_A.old.weight': torch.ones(2)}


def test_real_changed_adapter_and_ema_required():
    before = adapters()
    after = {name: value.clone() for name, value in before.items()}
    with pytest.raises(ValueError, match='unchanged'):
        audit_adapter_update(before, after, .99)
    after['block.lora_A.default.weight'].add_(1)
    with pytest.raises(ValueError, match='EMA'):
        audit_adapter_update(before, after, .99)
    after['block.lora_A.old.weight'].add_(.01)
    assert audit_adapter_update(before, after, .99)['ema_valid']
    after['block.lora_A.default.weight'][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        audit_adapter_update(before, after, .99)


def test_native_schedule_endpoint_audit():
    query = SimpleNamespace(video_schedule=torch.tensor([1., .6, 0.]),
        audio_schedule=torch.tensor([1., .3, 0.]), video_endpoint=torch.ones(1), audio_endpoint=torch.ones(2),
        query_index=1, sigma_video=.6, sigma_audio=.3)
    assert audit_query(query)['sigma_audio'] == .3
    query.audio_schedule = query.video_schedule
    with pytest.raises(ValueError, match='distinct'):
        audit_query(query)
    query.audio_schedule = torch.tensor([float('nan')])
    with pytest.raises(ValueError, match='nonfinite'):
        audit_query(query)


def test_reward_and_policy_zero_gradients_cannot_pass():
    verifier = H3Verification('target')
    target = SimpleNamespace(positive=torch.ones(1), negative=torch.ones(1), diagnostics={'gradient_norm': [0.]})
    with pytest.raises(ValueError, match='reward gradient'):
        verifier.targets({0: target}, 1)
    with pytest.raises(ValueError, match='accepted'):
        verifier.targets({}, 0)
    target.diagnostics['gradient_norm'] = [1.]
    with pytest.raises(ValueError, match='shape'):
        verifier.targets({0: target}, 1, anchors={0: torch.ones(2)})
    with pytest.raises(ValueError, match='optimizer update'):
        verifier.updated(None, {'optimizer_step': False, 'grad_norm': 0.}, None)


def test_paired_native_audio_and_prequantization_finiteness(tmp_path):
    verifier = H3Verification('update')
    folder = tmp_path/'audio_vae'
    folder.mkdir()
    (folder/'config.json').write_text(json.dumps(dict(sampling_rate=32000,
        encoder_rates=[2, 4, 4, 5, 5], decoder_rates=[5, 5, 2, 2, 2, 2, 2])))
    cfg = SimpleNamespace(num_frames=124, fps=24, model=str(tmp_path))
    rate, length = 32000, 165600  # native 40-Hz grid: round(124/24*40) = 207 tokens
    reply = dict(audio=torch.zeros(1, 2, length), sampling_rate=rate)
    verifier.waveform(reply, cfg)
    for malformed in (dict(reply, sampling_rate=24000), dict(reply, audio=reply['audio'][..., :-1])):
        with pytest.raises(ValueError, match='waveform'):
            verifier.waveform(malformed, cfg)
    reply['audio'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='waveform'):
        verifier.waveform(reply, cfg)
    path = tmp_path/'native.wav'
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(bytes(length*4))
    row = dict(candidate_index=0, prompt='paired', seed=7, sampling_rate=rate, audio_path=str(path))
    verifier.media([row], [dict(row)], cfg)
    evidence = verifier.events[-1]['pairs'][0]['base']
    assert evidence['channels'] == 2 and evidence['audio_tokens'] == 207
    assert evidence['duration_s'] == 5.175
    assert evidence['video_duration_s'] == 124/24
    assert evidence['audio_minus_video_s'] == pytest.approx(5.175-124/24)
    with pytest.raises(ValueError, match='pairing'):
        verifier.media([row], [dict(row, seed=8)], cfg)
    with pytest.raises(ValueError, match='duration'):
        audit_audio(path, rate=rate, samples=length-1, video_duration=124/24)
    with pytest.raises(ValueError, match='native-rate'):
        audit_audio(path, rate=24000, samples=length, video_duration=124/24)


def test_pinned_native_meta_audio_decoder_matches_configured_grid(tmp_path):
    from diffusers import AutoencoderKLMiniMaxH3Audio
    folder = tmp_path/'audio_vae'
    folder.mkdir()
    # Released pinned architecture defaults; meta execution allocates no weights/GPU tensors.
    with torch.device('meta'):
        decoder = AutoencoderKLMiniMaxH3Audio()
        decoder.save_config(folder)
        cfg = SimpleNamespace(model=str(tmp_path), num_frames=124, fps=24)
        layout = native_audio_layout(cfg)
        waveform = decoder.decode(torch.empty(2, decoder.config.latent_channels, layout['audio_tokens']))[0]
    assert layout['rate'] == 32000 and layout['audio_tokens'] == 207
    assert layout['samples_per_token'] == 800 and layout['samples'] == 165600
    assert waveform.device.type == 'meta' and waveform.shape == (2, 1, layout['samples'])


@pytest.mark.parametrize('argv', [[], ['--phase', 'update'],
    ['--phase', 'resume', '--config', 'x', '--run-dir', 'x'],
    ['--phase', 'load', '--config', 'x', '--run-dir', 'x', '--tiny-gpu']])
def test_cli_phase_contract_before_runtime_import(argv):
    if not argv:
        argv = ['--config', 'x']
    with pytest.raises(SystemExit) as error:
        runpy.run_path('scripts/check_minimax_h3_update.py')['main'](argv)
    assert error.value.code == 2


def test_report_never_claims_full_qualification(tmp_path):
    verifier = H3Verification('load')
    result = verifier.write(tmp_path, SimpleNamespace(rank=0, role='policy', device='cpu'), reached='load', phases=[])
    assert result['status'] == 'stage_measured'
    report = json.loads(Path(tmp_path, 'verification-load-rank-0.json').read_text())
    assert report['full_task8_qualification'] is False
    assert report['pending']
    assert 'available' in host_cgroup()


def test_four_seeds_and_native_repeat_are_checked():
    verifier = H3Verification('rollout')
    rows = [dict(index=index, seed=20+index) for index in range(4)]
    devices = SimpleNamespace(num_nodes=1)
    verifier.reproducibility([(rows[:2], rows[:2]), (rows[2:], rows[2:])], devices)
    with pytest.raises(ValueError, match='repeat'):
        verifier.reproducibility([(rows, [dict(rows[0], seed=99)]+rows[1:])], devices)
    with pytest.raises(ValueError, match='four canonical'):
        verifier.reproducibility([(rows[:3], rows[:3])], devices)


def test_resume_checks_actual_rng_adapter_optimizer_cursor(tmp_path):
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.lora_A = torch.nn.ModuleDict({name: torch.nn.Linear(1, 1, bias=False) for name in ('default', 'old')})
    policy = SimpleNamespace(model=model)
    optimizer = torch.optim.AdamW(model.block.lora_A['default'].parameters())
    state = dict(update=1, prompt_cursor={'cursor': 2})
    control = dict(service_sequence=3)
    from diffusionopsd.video.h3_checkpoint import _adapter_parameters, _cpu
    payload = dict(common=state, ranks=[dict(rng=_rng_state(), control=control)],
        adapters=_cpu(_adapter_parameters(policy)), optimizer=_cpu(optimizer.state_dict()))
    path = tmp_path/'update.pt'
    torch.save(payload, path)
    verifier = H3Verification('resume')
    before_rng = _digest(_rng_state())
    verifier.restored(path, state, control, policy, optimizer, SimpleNamespace(rank=0))
    assert _digest(_rng_state()) == before_rng
    torch.rand(1)
    with pytest.raises(ValueError, match='owned RNG'):
        verifier.restored(path, state, control, policy, optimizer, SimpleNamespace(rank=0))


def test_resume_without_checkpoint_rejected_before_distributed_initialization(tmp_path):
    from diffusionopsd.video.h3_training import run_h3_training
    with pytest.raises(ValueError, match='--resume'):
        run_h3_training(None, run_dir=tmp_path, verification=H3Verification('resume'))


def test_empty_pid_peak_map_is_missing_and_preserves_nvml_error_reason():
    row = dict(phase='evaluation', peak_allocated_bytes=1, peak_reserved_bytes=1,
        nvml_pid_peak_used_bytes={}, host_peak_rss_bytes=1, elapsed_s=1,
        nvml_unavailable='NVML permission denied')
    evidence = resource_evidence([row], 'cpu')
    assert 'nvml_pid_peak_used_bytes' in evidence['phase_missing']['evaluation']
    assert evidence['phase_unavailable_reasons']['evaluation']['nvml'] == 'NVML permission denied'
    assert evidence['gpu_fit_qualified'] is False
