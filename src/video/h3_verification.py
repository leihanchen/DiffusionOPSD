"""Explicit staged audits of the production H3 runner, never a qualification proxy."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import wave

import numpy as np
import torch

from diffusionopsd.video.h3_checkpoint import _adapter_parameters, _cpu, _digest, _rng_state


PHASES = ('load', 'rollout', 'target', 'update', 'resume')


def audit_query(query):
    result = {}
    for name in ('video_schedule', 'audio_schedule', 'video_endpoint', 'audio_endpoint'):
        tensor = getattr(query, name)
        if not tensor.numel() or not torch.isfinite(tensor).all():
            raise ValueError(f'nonfinite/empty native {name}')
        result[name] = dict(shape=list(tensor.shape), sha256=_digest(tensor))
        if name.endswith('schedule'):
            result[name]['values'] = tensor.detach().cpu().tolist()
    if torch.equal(query.video_schedule, query.audio_schedule):
        raise ValueError('native video and audio schedules must remain distinct')
    result.update(query_index=query.query_index, sigma_video=query.sigma_video, sigma_audio=query.sigma_audio)
    return result


def audit_adapter_update(before, after, decay):
    if before.keys() != after.keys() or not before:
        raise ValueError('missing or changed adapter layout')
    changed = False
    for name, tensor in after.items():
        if not torch.isfinite(tensor).all():
            raise ValueError('nonfinite adapter')
        if '.default.' in name:
            changed |= not torch.equal(before[name], tensor)
        elif '.old.' in name:
            current = name.replace('.old.', '.default.')
            expected = before[name].lerp(after[current].to(before[name]), 1-decay)
            if not torch.allclose(tensor, expected, atol=1e-6, rtol=1e-5):
                raise ValueError('invalid EMA update')
    if not changed or not any('.old.' in name for name in after):
        raise ValueError('default unchanged or EMA absent')
    return dict(default_changed=True, ema_valid=True, adapter_sha256=_digest(after))


def native_audio_layout(cfg):
    """Native rounded token grid and decoder expansion from the pinned local config."""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import audio_latent_num_frames
    root = Path(os.path.expandvars(str(cfg.model)))
    config = json.loads((root/'audio_vae/config.json').read_text())
    rate = config['sampling_rate']
    encoder_stride = math.prod(config['encoder_rates'])
    expansion = math.prod(config['decoder_rates'])
    if rate <= 0 or encoder_stride <= 0 or expansion <= 0:
        raise ValueError('invalid native audio configuration')
    tokens = audio_latent_num_frames(cfg.num_frames, fps=cfg.fps, latents_per_second=rate/encoder_stride)
    return dict(rate=rate, samples=tokens*expansion, audio_tokens=tokens,
                samples_per_token=expansion, video_duration=cfg.num_frames/cfg.fps)


def audit_audio(path, *, rate, samples, video_duration, **layout):
    with wave.open(str(path), 'rb') as stream:
        if stream.getnchannels() != 2 or stream.getframerate() != rate or stream.getsampwidth() != 2:
            raise ValueError('expected native-rate stereo PCM16 audio')
        frames = stream.getnframes()
        seconds = frames/rate
        waveform = np.frombuffer(stream.readframes(frames), dtype='<i2')
    if not waveform.size or not np.isfinite(waveform).all() or frames != samples:
        raise ValueError('invalid audio duration or waveform')
    return dict(path=str(path), sampling_rate=rate, channels=2, duration_s=seconds,
                video_duration_s=video_duration, audio_minus_video_s=seconds-video_duration, **layout,
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                qualitative_audio_preservation='pending human audiovisual inspection')


class H3Verification:
    """Rank-local evidence; runner readiness wraps every potentially failing audit."""

    def __init__(self, phase):
        if phase not in PHASES:
            raise ValueError('unknown verification phase')
        self.phase = phase
        self.events = []
        self.before = None
        self.resume_state = None

    def restored(self, resume, state, rank_state, policy, optimizer, devices):
        if resume is None or state['update'] < 1:
            raise ValueError('resume verification requires a previously updated checkpoint')
        self.resume_state = _cpu(state)
        source = Path(resume)
        if source.name == 'latest.json':
            source = source.parent/json.loads(source.read_text())['file']
        payload = torch.load(source, map_location='cpu', weights_only=False)
        saved_rank = payload['ranks'][devices.rank]
        if _digest(_rng_state()) != _digest(saved_rank['rng']):
            raise ValueError('restored owned RNG differs from checkpoint')
        if _digest(rank_state) != _digest(saved_rank['control']):
            raise ValueError('restored service sequence/identity differs from checkpoint')
        if state['prompt_cursor'] != payload['common']['prompt_cursor']:
            raise ValueError('restored prompt cursor differs from checkpoint')
        if policy:
            self.before = _cpu(_adapter_parameters(policy))
            if (_digest(self.before) != _digest(payload['adapters'])
                    or _digest(optimizer.state_dict()) != _digest(payload['optimizer'])):
                raise ValueError('restored adapter/optimizer differs from checkpoint')
        self.events.append(dict(stage='restore', checkpoint=str(resume), update=state['update'],
            cursor=state['prompt_cursor'], service_sequence=rank_state['service_sequence'],
            restored_rank_state_sha256=_digest(rank_state),
            owned_rng_matches_checkpoint=True, cursor_matches_checkpoint=True,
            adapter_optimizer_matches_checkpoint=policy is not None))

    def loaded(self, policy, selected, cfg, preflight):
        if policy:
            self.before = _cpu(_adapter_parameters(policy))
            if not self.before:
                raise ValueError('verification requires actual default/old adapters')
        self.events.append(dict(stage='load', provenance=preflight,
            geometry={key: getattr(cfg, key) for key in ('num_frames', 'height', 'width', 'fps')},
            longest_selected={key: max(records, key=lambda record: len(record['text']))
                              for key, records in selected.items() if records}))

    def rollout(self, candidates, request):
        rows = [dict(index=item.candidate_index, seed=request['seed']+item.candidate_index % 4,
                     **audit_query(item.query)) for item in candidates]
        self.events.append(dict(stage='rollout', prompt=request['record'], attempt=request['attempt'], candidates=rows))
        return rows

    def reproducibility(self, evidence, devices):
        first = sorted([row for pair in evidence for row in pair[0]], key=lambda row: row['index'])
        second = sorted([row for pair in evidence for row in pair[1]], key=lambda row: row['index'])
        if first != second:
            raise ValueError('native same-seed repeat differs')
        if [row['index'] for row in first] != list(range(4*devices.num_nodes)):
            raise ValueError('verification requires exactly four canonical candidates per node')
        for node in range(devices.num_nodes):
            rows = first[node*4:(node+1)*4]
            if len({row['seed'] for row in rows}) != 4:
                raise ValueError('four distinct candidate seeds required')
        self.events.append(dict(stage='reproducibility', candidates=first, native_repeat_identical=True))

    def targets(self, targets, retained, anchors=None):
        for index, target in targets.items():
            norms = target.diagnostics['gradient_norm']
            if not norms or not all(math.isfinite(float(value)) and value > 0 for value in norms):
                raise ValueError('reward gradient must be finite and nonzero')
            for tensor in (target.positive, target.negative):
                if tensor.requires_grad or not torch.isfinite(tensor).all():
                    raise ValueError('invalid detached target')
                if anchors is not None and tensor.shape != anchors[index].shape:
                    raise ValueError('target shape does not match native anchor')
        if retained <= 0:
            raise ValueError('target verification requires accepted candidates')
        self.events.append(dict(stage='target', global_accepted=retained,
            local_targets={index: item.diagnostics for index, item in targets.items()}))

    def updated(self, policy, metrics, cfg):
        if not metrics['optimizer_step'] or not math.isfinite(metrics['grad_norm']) or metrics['grad_norm'] <= 0:
            raise ValueError('verification requires a real finite nonzero optimizer update')
        result = audit_adapter_update(self.before, _cpu(_adapter_parameters(policy)), cfg.ema_decay)
        self.events.append(dict(stage='update', **metrics, **result))

    def waveform(self, reply, cfg):
        audio = reply['audio']
        rate = reply['sampling_rate']
        layout = native_audio_layout(cfg)
        if (audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] != 2
                or not torch.isfinite(audio).all() or rate != layout['rate']
                or audio.shape[-1] != layout['samples']):
            raise ValueError('invalid finite native stereo waveform/rate/duration')

    def media(self, base, current, cfg):
        if not base or len(base) != len(current):
            raise ValueError('paired base/current evaluation media required')
        pairs = []
        for left, right in zip(base, current):
            for key in ('candidate_index', 'prompt', 'seed', 'sampling_rate'):
                if left[key] != right[key]:
                    raise ValueError('base/current prompt/seed/rate pairing mismatch')
            pairs.append(dict(prompt=left['prompt'], seed=left['seed'], **{
                label: audit_audio(row['audio_path'], **native_audio_layout(cfg))
                for label, row in (('base', left), ('current', right))}))
        self.events.append(dict(stage='media', pairs=pairs))

    def checkpoint(self, path, state, devices):
        if self.resume_state is not None:
            if state['update'] != self.resume_state['update']+1:
                raise ValueError('resume must complete exactly one additional optimizer update')
            if state['prompt_cursor']['cursor'] <= self.resume_state['prompt_cursor']['cursor']:
                raise ValueError('resume prompt cursor did not advance')
        if devices.rank != 0:
            return
        manifest = json.loads(Path(str(path)+'.json').read_text())
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['sha256']:
            raise ValueError('published checkpoint hash mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['common']['update'] != state['update'] or not payload['optimizer']['state']:
            raise ValueError('missing checkpoint optimizer/update state')
        if any('rng' not in rank or 'control' not in rank for rank in payload['ranks']):
            raise ValueError('missing checkpoint rank RNG/service state')
        self.events.append(dict(stage='checkpoint', path=str(path), sha256=manifest['sha256'],
            update=state['update'], cursor=state['prompt_cursor'],
            adapter_optimizer_rank_agreement='enforced by production save_h3_checkpoint',
            ranks=len(payload['ranks']), optimizer_sha256=_digest(payload['optimizer'])))

    def write(self, run_dir, devices, *, reached, phases, state=None):
        report = dict(requested_phase=self.phase, reached_phase=reached, rank=devices.rank,
            role=devices.role, events=self.events, phases=phases,
            update=None if state is None else state['update'],
            prompt_cursor=None if state is None else state['prompt_cursor'],
            calibration=None if state is None else state.get('calibration'),
            gate_decisions_path=f'gate-decisions-rank-{devices.rank}.jsonl',
            full_task8_qualification=False,
            pending=['full H100/full checkpoint resource and correctness qualification',
                     'longest selected train/eval prompt memory execution',
                     'five successful updates and 50-update pilot', 'held-out quality and audiovisual inspection'])
        report['host_cgroup'] = host_cgroup()
        report['job_id'] = os.environ.get('SLURM_JOB_ID')
        report['resource_qualification'] = 'pending explicit peak/capacity and allocation comparison'
        report['resource_evidence'] = resource_evidence(phases, devices.device)
        Path(run_dir, f'verification-{self.phase}-rank-{devices.rank}.json').write_text(json.dumps(report, indent=2))
        return dict(status='stage_measured' if reached == self.phase else reached, **report)


def host_cgroup():
    """Keep scope explicit: cgroup peak is allocation lifetime, RSS is per process."""
    result = dict(scope='cgroup lifetime; not attributable to a single phase', available=False)
    try:
        relative = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                        if line.startswith('0::'))
        root = Path('/sys/fs/cgroup')/relative.lstrip('/')
        for name in ('memory.current', 'memory.peak', 'memory.max'):
            value = (root/name).read_text().strip()
            result[name] = int(value) if value != 'max' else value
        result['available'] = True
    except (OSError, StopIteration, ValueError) as error:
        result['unavailable_reason'] = str(error)
    return result


def resource_evidence(phases, device):
    keys = ('peak_allocated_bytes', 'peak_reserved_bytes', 'nvml_device_peak_used_bytes',
            'nvml_pid_peak_used_bytes', 'host_peak_rss_bytes', 'elapsed_s')
    result = dict(phase_missing={row['phase']: [key for key in keys if key not in row
                    or (key == 'nvml_pid_peak_used_bytes' and not row[key])]
                    for row in phases}, gpu_fit_qualified=False)
    result['phase_unavailable_reasons'] = {
        row['phase']: {'nvml': row.get('nvml_unavailable') or 'no per-PID NVML samples'}
        for row in phases if row.get('nvml_unavailable') or not row.get('nvml_pid_peak_used_bytes')}
    if torch.device(device).type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        limit = min(70*1024**3, .90*props.total_memory)
        result.update(model=props.name, capacity_bytes=props.total_memory, acceptance_limit_bytes=limit)
        result['measured_phase_peaks_below_limit'] = bool(phases) and all(
            not result['phase_missing'][row['phase']]
            and max(row['peak_reserved_bytes'], row['nvml_device_peak_used_bytes']) < limit for row in phases)
    else:
        result['unavailable_reason'] = 'CPU execution is not GPU memory qualification'
    return result
