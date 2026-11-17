"""Native H3 reward owner and bounded, node-local CPU/Gloo request protocol.

All local ranks call request in the same order, passing owner_rank in metadata.
Only that owner supplies payload tensors; only the selected reward rank loads a
service. No policy process group is used here. Candidate ids include node/update.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import uuid

import torch
import torch.distributed as dist
import torch.nn.functional as F

from diffusionopsd.video.geo_reward import reward_metadata
from diffusionopsd.video.opa_video import opa_tr_step_nd

VERSION = 1
KINDS = {'score_endpoint', 'score_anchor', 'construct_targets', 'decode_evaluation'}
DTYPES = {str(dtype): dtype for dtype in (torch.float32, torch.bfloat16, torch.float16)}


def _get(cfg, name, default):
    return cfg.get(name, default) if cfg is not None else default


def memory_snapshot(device):
    """Allocator figures plus NVML device/process use (including the judge)."""
    result = {'host_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if torch.device(device).type != 'cuda':
        return result
    result.update(allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    from diffusionopsd.video.cosmos_quality import nvml_device_snapshot
    result.update(nvml_device_snapshot(device))
    return result


class H3PhaseProfiler:
    """Sample external device allocations while a synchronous service phase runs."""

    _active = {}
    _active_lock = threading.Lock()

    def __init__(self, device, phase, interval_s=.05, metadata=None):
        self.device, self.phase, self.interval_s = torch.device(device), phase, interval_s
        self.metadata = metadata or {}
        self.done = threading.Event()
        self.samples = []

    def __enter__(self):
        self.allocator_scope = None
        if self.device.type == 'cuda':
            if self.device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
            torch.cuda.synchronize(self.device)
            with self._active_lock:
                stack = self._active.setdefault(self.device, [])
                if not stack:
                    torch.cuda.reset_peak_memory_stats(self.device)
                self.allocator_scope = f'enclosing phase {stack[0].phase}' if stack else 'this phase'
                stack.append(self)
        self.started = time.monotonic()

        def sample():
            while not self.done.is_set():
                self.samples.append(memory_snapshot(self.device))
                self.done.wait(self.interval_s)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        try:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
        finally:
            self.done.set()
            self.thread.join()
            if self.device.type == 'cuda':
                with self._active_lock:
                    stack = self._active[self.device]
                    stack.remove(self)
                    if not stack:
                        del self._active[self.device]
        self.metrics = dict(phase=self.phase, elapsed_s=time.monotonic()-self.started,
                            sample_interval_s=self.interval_s, **self.metadata, **memory_snapshot(self.device))
        if self.allocator_scope is not None:
            # Children share the owner's counters; only external telemetry is request-local.
            self.metrics['allocator_peak_scope'] = self.allocator_scope
        samples = self.samples + [self.metrics]
        external = [sample['nvml_device_used_bytes'] for sample in samples if 'nvml_device_used_bytes' in sample]
        if external:
            self.metrics['nvml_device_peak_used_bytes'] = max(external)
        self.metrics['nvml_samples'] = len(external)
        peaks = {}
        for sample in samples:
            for pid, value in sample.get('nvml_process_bytes', {}).items():
                peaks[pid] = max(value, peaks.get(pid, 0))
        self.metrics['nvml_pid_peak_used_bytes'] = peaks
        own_pids = [str(self.metadata[name]) for name in ('service_pid', 'judge_pid') if self.metadata.get(name)]
        if own_pids and external:
            self.metrics['nvml_service_judge_peak_used_bytes'] = max(
                sum(sample.get('nvml_process_bytes', {}).get(pid, 0) for pid in own_pids)
                for sample in samples)


@dataclass
class H3Targets:
    positive: torch.Tensor
    negative: torch.Tensor
    diagnostics: dict


class H3RewardService:
    def __init__(self, vae, reward, *, device, cfg=None, judge=None, audio_loader=None, run_dir=None):
        self.device, self.cfg = torch.device(device), cfg
        self.vae = vae.to(self.device).eval().requires_grad_(False)
        self.vae.enable_gradient_checkpointing()
        self.vae.enable_tiling()
        self.reward, self.judge, self.audio_loader = reward, judge, audio_loader
        self.run_dir = Path(run_dir) if run_dir else None
        self._lock = threading.Lock()
        channels = self.vae.config.latent_channels
        if len(self.vae.config.latents_mean) != channels or len(self.vae.config.latents_std) != channels:
            raise ValueError('VAE normalization/config channel mismatch')
        if any(value <= 0 for value in self.vae.config.latents_std):
            raise ValueError('VAE latent standard deviations must be positive')

    @classmethod
    def from_config(cls, cfg, devices, run_dir, *, with_judge=True):
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
        from diffusionopsd.video.estimators import load_depth_anything3, load_waft, load_dinov2
        from diffusionopsd.video.geo_reward import GeoReward
        from diffusionopsd.video.cosmos_quality import IsolatedVideoJudge
        from diffusionopsd.video.h3_rollout import vae_descriptor
        if devices.role != 'reward':
            raise ValueError('only reward ranks may load service models')
        if with_judge:
            # The runtime bind-mounts this package into site-packages; only the
            # source repository on sys.path contains the standalone judge worker.
            for entry in sys.path:
                repo = Path(entry)
                if repo.is_absolute() and (repo / 'scripts/cosmos_quality_worker.py').is_file():
                    repo = repo.resolve()
                    break
            else:
                raise FileNotFoundError(
                    'Cannot find scripts/cosmos_quality_worker.py; add the source repository '
                    'as an absolute path on PYTHONPATH before loading the H3 reward service')
        root = Path(os.path.expandvars(str(cfg.model)))
        folder = Path(run_dir) / f'reward-rank-{devices.rank}'
        folder.mkdir(parents=True, exist_ok=True)
        vae = AutoencoderKLMiniMaxH3.from_pretrained(root / 'vae', local_files_only=True,
                                                  torch_dtype=torch.bfloat16)
        descriptor = vae_descriptor(root / 'vae/config.json')
        for name in ('spatial_compression_ratio', 'temporal_compression_ratio', 'frame_pre_padding',
                     'tokens_chunk_size', 'token_overlap', 'frame_overlap'):
            if getattr(vae, name) != getattr(descriptor, name):
                raise ValueError(f'VAE descriptor mismatch: {name}')
        reward = GeoReward(load_depth_anything3(devices.device), load_waft(devices.device),
                           load_dinov2(devices.device))
        judge = IsolatedVideoJudge(repo, folder, str(devices.device),
            residency_protocol=True, worker_path=repo / 'scripts/cosmos_quality_worker.py') if with_judge else None
        return cls(vae, reward, device=devices.device, cfg=cfg, judge=judge, run_dir=folder,
            audio_loader=lambda: AutoencoderKLMiniMaxH3Audio.from_pretrained(root / 'audio_vae',
                local_files_only=True, torch_dtype=torch.bfloat16).to(devices.device).eval().requires_grad_(False))

    def decode01(self, video_latents):
        if video_latents.ndim != 5 or video_latents.shape[1] != self.vae.config.latent_channels:
            raise ValueError('video latents must be unpacked [B,C,T,H,W] matching VAE channels')
        x = video_latents.to(self.device).float()
        mean = x.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = x.new_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        raw = self.vae.decode(x * std + mean, return_dict=False)[0].float()
        mean = raw.new_tensor((.485, .456, .406)).view(1, 3, 1, 1, 1)
        std = raw.new_tensor((.229, .224, .225)).view(1, 3, 1, 1, 1)
        return (raw * std + mean).clamp(0, 1).permute(0, 2, 1, 3, 4)

    def reward_clip(self, video):
        indices = torch.linspace(0, video.shape[1]-1, _get(self.cfg, 'reward_frames', 9),
                                 device=video.device).round().long()
        clip = video[:, indices]
        height, width = _get(self.cfg, 'reward_height', 224), _get(self.cfg, 'reward_width', 384)
        return F.interpolate(clip.flatten(0, 1), size=(height, width), mode='bilinear',
            align_corners=False, antialias=True).reshape(video.shape[0], len(indices), 3, height, width)

    def reward_values(self, latents):
        result = self.reward(self.reward_clip(self.decode01(latents)))
        return result.geo if hasattr(result, 'geo') else result

    def construct_targets(self, anchor, cfg=None):
        cfg = self.cfg if cfg is None else cfg
        if self.judge is not None:
            self.judge.park()
        x = anchor.detach().to(self.device).float().requires_grad_(True)
        gradient, = torch.autograd.grad(self.reward_values(x).sum(), x)
        if not torch.isfinite(gradient).all() or not (gradient.flatten(1).norm(dim=1) > 0).all():
            raise RuntimeError('nonfinite or zero initial reward gradient')
        rho, steps, eta = _get(cfg, 'rho', .1), _get(cfg, 'n_ascent', 2), _get(cfg, 'eta', 1.)
        if not 0 < rho <= .1 or steps < 1 or eta <= 0:
            raise ValueError('invalid H3 trust region configuration')
        targets = [opa_tr_step_nd(x, self.reward_values, rho, steps, eta, direction, first_grad=gradient.detach())
                   for direction in (1., -1.)]
        budget = rho * x.detach().flatten(1).norm(dim=1)
        norms = [(target-x.detach()).flatten(1).norm(dim=1) for target in targets]
        for target, norm in zip(targets, norms):
            if not torch.isfinite(target).all() or not (norm <= budget + 1e-5 * budget.clamp_min(1)).all():
                raise RuntimeError('target violated finite/trust budget assertion')
        diagnostics = dict(gradient_norm=gradient.flatten(1).norm(dim=1).detach().cpu().tolist(),
            trust_budget=budget.detach().cpu().tolist(), positive_distance=norms[0].cpu().tolist(),
            negative_distance=norms[1].cpu().tolist(), reward=reward_metadata())
        return H3Targets(*(target.detach().float() for target in targets), diagnostics)

    def dispatch(self, kind, payload, metadata):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('one in-flight candidate graph per worker')
        start = time.monotonic()
        profiler = None
        try:
            profiler = H3PhaseProfiler(self.device, kind, metadata=dict(service_pid=os.getpid(),
                judge_pid=self.judge.process.pid if self.judge else None,
                candidate_id=metadata.get('candidate_id'), request_id=metadata.get('request_id'),
                latent_geometry=list(payload['video'].shape) if 'video' in payload else None,
                tiling_mode='tiled' if self.vae.use_tiling else 'untiled'))
            profiler.__enter__()
            if metadata.get('reward_version', reward_metadata()['version']) != reward_metadata()['version']:
                raise ValueError('reward version mismatch')
            if kind == 'construct_targets':
                if metadata.get('accepted') is not True:
                    raise ValueError('construct_targets requires accepted candidate')
                targets = self.construct_targets(payload['video'])
                result = dict(positive=targets.positive.cpu(), negative=targets.negative.cpu(),
                              diagnostics=targets.diagnostics)
            elif kind in ('score_endpoint', 'score_anchor'):
                if not isinstance(metadata.get('prompt'), str):
                    raise ValueError('score requests require explicit prompt metadata')
                with torch.no_grad():
                    video = self.decode01(payload['video'])
                    score = self.reward(self.reward_clip(video))
                    result = {key: getattr(score, key).detach().float().cpu().tolist()
                        for key in ('geo', 's_id', 'motion')} if hasattr(score, 'geo') else {'geo': score.cpu().tolist()}
                    if self.judge:
                        judge_clip = video[0].cpu()
                        del video, score
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        try:
                            result['quality'] = self.judge.score(judge_clip, metadata['prompt'])
                        finally:
                            score_error = sys.exc_info()[1]
                            try:
                                self.judge.park()
                            except Exception as park_error:
                                if score_error is None:
                                    raise
                                score_error.add_note(f'Judge park also failed: {park_error}')
            elif kind == 'decode_evaluation':
                with torch.no_grad():
                    result = {'video': self.decode01(payload['video']).cpu(), 'metadata': metadata}
                    if 'audio' in payload:
                        if self.audio_loader is None:
                            raise ValueError('audio decoder is unavailable')
                        vae = self.audio_loader()
                        try:
                            x = payload['audio'].to(self.device).float()
                            x = x * x.new_tensor(vae.config.latents_std).view(1, -1, 1) + x.new_tensor(
                                vae.config.latents_mean).view(1, -1, 1)
                            result['audio'] = vae.decode(x, return_dict=False)[0].float().permute(1, 0, 2).cpu()
                            result['sampling_rate'] = vae.config.sampling_rate
                        finally:
                            del vae
                            if self.device.type == 'cuda':
                                torch.cuda.empty_cache()
            else:
                raise ValueError(f'unknown request kind {kind}')
            result['metrics'] = dict(service_s=time.monotonic()-start, queue_wait_s=0., **memory_snapshot(self.device))
            return result
        finally:
            primary = sys.exc_info()[1]
            try:
                if hasattr(profiler, 'thread'):
                    profiler.__exit__()
                if self.judge:
                    profiler.metrics['judge_release_evidence'] = self.judge.last_release_evidence
                if self.run_dir:
                    with (self.run_dir / 'phases.jsonl').open('a') as stream:
                        stream.write(json.dumps(profiler.metrics) + '\n')
                if 'result' in locals():
                    result.setdefault('metrics', {}).update(profiler.metrics)
            except Exception as diagnostic_error:
                if primary is not None:
                    primary.add_note(f'Optional phase diagnostics failed: {diagnostic_error}')
                elif 'result' in locals():
                    result.setdefault('metrics', {})['diagnostic_error'] = str(diagnostic_error)
            finally:
                if profiler is not None:
                    profiler.done.set()
                if hasattr(profiler, 'thread'):
                    profiler.thread.join()
                self._lock.release()


class H3RewardClient:
    """SPMD: every node rank enters; owner_rank and candidate_index are explicit.

    The node control group must be Gloo. Returns detached result only on owner;
    other ranks receive completion identity/metrics. A failed request poisons the
    client on every rank: the trainer must terminate before replay/optimizer work.
    """
    def __init__(self, devices=None, service=None, *, run_id=None, timeout_s=600., max_payload_bytes=1024**3):
        self.devices, self.service = devices, service
        self.run_id = run_id or os.environ.get('TORCHELASTIC_RUN_ID')
        if devices is not None and not self.run_id:
            raise ValueError('distributed clients require a shared globally unique run_id')
        self.run_id = self.run_id or uuid.uuid4().hex
        self.timeout_s, self.max_payload_bytes = timeout_s, max_payload_bytes
        self.sequence, self.failed = 0, False
        self._lock = threading.Lock()

    def _wait(self, work, deadline):
        remaining = deadline-time.monotonic()
        if remaining <= 0 or not work.wait(timeout=timedelta(seconds=max(remaining, .001))):
            raise TimeoutError('H3 reward request deadline exceeded')

    def _header(self, header, source, deadline):
        group, rank = self.devices.node_control_group, self.devices.rank
        encoded = json.dumps(header).encode() if rank == source else b''
        if len(encoded) > 65532:
            raise ValueError('control metadata exceeds 64 KiB')
        buffer = torch.zeros(65536, dtype=torch.uint8)
        if rank == source:
            buffer[:4] = torch.tensor(list(len(encoded).to_bytes(4, 'little')), dtype=torch.uint8)
            buffer[4:4+len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8)
        self._wait(dist.broadcast(buffer, src=source, group=group, async_op=True), deadline)
        size = int.from_bytes(bytes(buffer[:4].tolist()), 'little')
        return json.loads(bytes(buffer[4:4+size].tolist()))

    def _schema(self, payload):
        total, schema = 0, {}
        for key, tensor in payload.items():
            if not isinstance(tensor, torch.Tensor) or str(tensor.dtype) not in DTYPES:
                raise ValueError('payload must contain floating tensors')
            if tensor.ndim > 5 or tensor.numel() == 0 or not torch.isfinite(tensor).all():
                raise ValueError('invalid payload tensor shape/values')
            total += tensor.numel()*tensor.element_size()
            schema[key] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype))
        if total > self.max_payload_bytes:
            raise ValueError('payload exceeds byte cap')
        return schema, total

    def _transfer(self, payload, schema, source, destination, deadline):
        rank, group = self.devices.rank, self.devices.node_control_group
        received = {}
        for key, spec in schema.items():
            if rank == source:
                tensor = payload[key].detach().cpu().contiguous()
                self._wait(dist.isend(tensor, dst=destination, group=group), deadline)
            elif rank == destination:
                tensor = torch.empty(spec['shape'], dtype=DTYPES[spec['dtype']])
                self._wait(dist.irecv(tensor, src=source, group=group), deadline)
                received[key] = tensor
        return received

    def request(self, kind, candidate_id, payload, metadata):
        if self.failed:
            raise RuntimeError('reward client poisoned by previous error; terminate before replay')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('one in-flight request per client')
        started = time.monotonic()
        deadline = started + self.timeout_s
        self.sequence += 1
        node = f'node-{self.devices.node_index}/' if self.devices is not None else ''
        identity = f'{self.run_id}/{node}{self.sequence}/{candidate_id}'
        metadata = dict(metadata, candidate_id=candidate_id, request_id=identity)
        try:
            if self.devices is None:
                if kind not in KINDS or not candidate_id:
                    raise ValueError('invalid kind/candidate identity')
                _, size = self._schema(payload)
                copy_started = time.monotonic()
                copied = {k: v.detach().cpu().contiguous() for k, v in payload.items()}
                transfer = time.monotonic()-copy_started
                result = self.service.dispatch(kind, copied, metadata)
                if time.monotonic() > deadline:
                    raise TimeoutError('H3 reward request deadline exceeded')
            else:
                owner = metadata['owner_rank']
                node_start = self.devices.node_index * (self.devices.policy_count + len(self.devices.reward_ranks))
                if owner in self.devices.reward_ranks or not node_start <= owner < node_start + self.devices.policy_count + len(self.devices.reward_ranks):
                    raise ValueError('request owner must be a node-local policy rank')
                worker = self.devices.reward_ranks[metadata.get('candidate_index', self.sequence-1) % len(self.devices.reward_ranks)]
                header = None
                if self.devices.rank == owner:
                    try:
                        if kind not in KINDS or not candidate_id:
                            raise ValueError('invalid kind/candidate identity')
                        schema, size = self._schema(payload)
                        header = dict(version=VERSION, identity=identity, kind=kind, owner=owner, worker=worker,
                                      schema=schema, bytes=size, metadata=metadata)
                        if len(json.dumps(header).encode()) > 65532:
                            raise ValueError('control metadata exceeds 64 KiB')
                    except Exception as exc:
                        header = dict(error=str(exc), identity=identity)
                header = self._header(header, owner, deadline)
                if header.get('identity') != identity:
                    raise RuntimeError('request identity mismatch')
                if 'error' in header:
                    raise RuntimeError(header['error'])
                if (header['version'], header['owner'], header['worker'], header['kind']) != (VERSION, owner, worker, kind):
                    raise RuntimeError('request routing/version mismatch')
                copy_started = time.monotonic()
                incoming = self._transfer(payload, header['schema'], owner, worker, deadline)
                transfer = time.monotonic()-copy_started
                response, tensors = None, {}
                if self.devices.rank == worker:
                    try:
                        result = self.service.dispatch(kind, incoming, header['metadata'])
                        tensors = {k: v for k, v in result.items() if isinstance(v, torch.Tensor)}
                        schema, _ = self._schema(tensors)
                        response = dict(identity=identity, schema=schema,
                            result={k: v for k, v in result.items() if k not in tensors})
                        if len(json.dumps(response).encode()) > 65532:
                            raise ValueError('response control metadata exceeds 64 KiB')
                    except Exception as exc:
                        response = dict(identity=identity, error=f'{type(exc).__name__}: {exc}')
                response = self._header(response, worker, deadline)
                if response['identity'] != identity:
                    raise RuntimeError('response identity mismatch')
                if 'error' in response:
                    raise RuntimeError(response['error'])
                return_started = time.monotonic()
                returned = self._transfer(tensors, response['schema'], worker, owner, deadline)
                transfer += time.monotonic()-return_started
                result = response['result'] if self.devices.rank == owner else {}
                result.update(returned)
                size = header['bytes']
            result.update(version=VERSION, request_id=identity, candidate_id=candidate_id)
            result.setdefault('metrics', {}).update(payload_bytes=size, host_transfer_s=transfer,
                                                   elapsed_s=time.monotonic()-started)
            return result
        except Exception:
            self.failed = True
            raise
        finally:
            self._lock.release()
