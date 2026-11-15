"""Native joint H3 sampling with collective-ordered, globally normalized OPSD."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
import wave

import torch
import torch.distributed as dist

from diffusionopsd.stat_tracking import PerPromptStatTracker
from diffusionopsd.video.branch_loss import branch_loss
from diffusionopsd.video.cosmos_training import PromptCursor
from diffusionopsd.video.gates import effective_weight, percentile_threshold
from diffusionopsd.video.h3_auxiliary import H3PhaseProfiler, H3RewardClient, H3RewardService, H3Targets
from diffusionopsd.video.h3_conditioning import (
    load_conditioning, model_root, select_records, validate_h3_assets,
)
from diffusionopsd.video.h3_checkpoint import H3SafeBoundary
from diffusionopsd.video.h3_fsdp import (
    init_h3_distributed, load_h3_policy, reduce_adapter_gradients_, resolve_h3_timeouts,
)
from diffusionopsd.video.h3_rollout import H3Rollout, build_h3_core
from diffusionopsd.video.quality_judge import gap_to_probability, quality_gap
from diffusionopsd.video.training_data import load_training_data
from diffusionopsd.video.training_eval import evaluation_due, save_video
from diffusionopsd.video.training_logging import VideoTrainingLogger, evaluation_metrics


@dataclass
class H3Candidate:
    candidate_index: int
    owner_rank: int
    query: object
    candidate_id: str
    dummy: bool = False


def h3_candidate_assignments(attempt: int, node_rank: int, node_count: int, policy_rank: int,
                             policy_count: int, record_order: list[int]) -> list[tuple[int, int]]:
    """Four local candidates for node j's next record in one fixed permutation."""
    if (attempt < 0 or node_count < 1 or not 0 <= node_rank < node_count
            or policy_count not in (1, 2, 4) or not 0 <= policy_rank < policy_count or not record_order):
        raise ValueError('invalid H3 candidate topology, attempt or record order')
    record = record_order[(attempt*node_count+node_rank) % len(record_order)]
    return [(record, candidate) for candidate in range(policy_rank, 4, policy_count)]


def h3_evaluation_assignments(node_rank, node_count, policy_rank, policy_count, record_count=8):
    """Equal native calls on every policy member, with explicit dummy slots."""
    if (node_count < 1 or policy_count < 1 or record_count < 1
            or not 0 <= node_rank < node_count or not 0 <= policy_rank < policy_count):
        raise ValueError('invalid H3 evaluation topology')
    width = node_count*policy_count
    return [index if index < record_count else None
            for index in range(node_rank*policy_count+policy_rank,
                               math.ceil(record_count/width)*width, width)]


def _node_positions(attempt, nodes, pool_size):
    return [dict(node_rank=node, next_record=(attempt*nodes+node) % pool_size)
            for node in range(nodes)]


def _validate_node_count(devices, configured):
    if devices.num_nodes != configured:
        raise ValueError('runtime M node count differs from configured topology')


def _exchange(value, group):
    if group is None:
        return [value]
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def _ready(phase, group, error=None):
    statuses = _exchange(dict(phase=phase, error=error), group)
    if any(status['phase'] != phase for status in statuses):
        raise RuntimeError(f'H3 collective phase mismatch before {phase}')
    failures = [status['error'] for status in statuses if status['error']]
    if failures:
        raise RuntimeError(f'H3 {phase} failed: ' + '; '.join(failures))


def _phase(name, operation, group):
    """Exchange readiness and completion; group deadlines bound a failed peer."""
    _ready(name + ':ready', group)
    rank = dist.get_rank() if dist.is_initialized() else 0

    def emit(event, **details):
        # Keep emergency diagnostics CPU-only: CUDA work may already be stalled.
        print(json.dumps(dict(kind='h3_phase', rank=rank, phase=name, event=event,
                              time_unix_s=time.time(), **details)), file=sys.stderr, flush=True)

    emit('start')
    result, error = None, None
    try:
        result = operation()
    except Exception as exception:
        error = f'{type(exception).__name__}: {exception}'
        # Completion can block behind a peer's failed FSDP operation. Preserve
        # the original exception locally before entering that collective.
        emit('error', error=error, traceback=traceback.format_exc())
    else:
        emit('returned')
    _ready(name + ':complete', group, error)
    return result


def collect_candidates(policy, rollout, request: dict, *, rank: int, policy_count: int) -> list[H3Candidate]:
    if policy is None:
        return []
    if policy_count not in (1, 2, 4):
        raise ValueError('candidate layout requires P=2 or P=4 (P=1 is CPU reference only)')
    devices = policy.devices
    group = devices.all_policy_group
    local_index = devices.policy_index if devices.policy_count == policy_count else rank % policy_count
    result = []
    assignments = h3_candidate_assignments(request['attempt'], devices.node_index, devices.num_nodes,
                                            local_index, policy_count, [0])
    for round_index, (_, index) in enumerate(assignments):
        global_index = devices.node_index * 4 + index
        query = _phase(f'rollout:{round_index}', lambda: rollout.rollout(request['conditioning'],
            seed=int(request['seed']) + index, geometry=request['geometry'], adapter=request.get('adapter', 'old')),
            group)
        result.append(H3Candidate(global_index, rank, query,
            f"node-{devices.node_index}/record-{request.get('record_key', 'unknown')}/"
            f"attempt-{request['attempt']}/candidate-{index}", index >= 4))
    return result


def fit_h3_update(policy, rollout, candidates, *, targets, weights, global_kept, optimizer, cfg) -> dict:
    """Replay equal slots, SUM all gradients, divide once by global accepted N."""
    group = policy.devices.all_policy_group
    parameters = policy.trainable_parameters

    def validate():
        if len(candidates) != math.ceil(4 / policy.devices.policy_count):
            raise ValueError('unequal candidate replay slot count')
        if global_kept < 0:
            raise ValueError('global accepted count must be nonnegative')
        for candidate in candidates:
            if candidate.dummy:
                continue
            target = targets[candidate.candidate_index]
            anchor = candidate.query.y0
            if not 0 <= weights[candidate.candidate_index] <= 1:
                raise ValueError('OPSD weight must be in [0,1]')
            for tensor in (target.positive, target.negative):
                if tensor.requires_grad or tensor.shape != anchor.shape or not torch.isfinite(tensor).all():
                    raise ValueError('H3 targets must be detached, finite and match the anchor')
    _phase('fit_layout', validate, group)
    optimizer.zero_grad(set_to_none=True)
    if global_kept == 0:
        return dict(retained=0, loss=0., grad_norm=0., optimizer_step=False, replay_slots=0)
    loss_sum = 0.
    try:
        for slot, candidate in enumerate(candidates):
            def replay():
                prediction = rollout.prediction(candidate.query)
                if candidate.dummy:
                    loss = prediction.float().sum() * 0
                else:
                    anchor = candidate.query.y0.detach().to(prediction.device).float()
                    target = targets[candidate.candidate_index]
                    loss = branch_loss(prediction.float(), anchor, target.positive.to(prediction.device),
                        target.negative.to(prediction.device), anchor.new_tensor([weights[candidate.candidate_index]]),
                        cfg.beta).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite H3 replay loss')
                return loss
            loss = _phase(f'replay:{slot}', replay, group)
            _phase(f'backward:{slot}', lambda: (loss * cfg.loss_scale).backward(), group)
            loss_sum += float(loss.detach())
        _phase('gradient_sum', lambda: reduce_adapter_gradients_(parameters, group, global_kept), group)

        def clip():
            norm = torch.nn.utils.clip_grad_norm_(parameters, min(float(cfg.max_grad_norm), 1.))
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError('H3 policy gradient must be finite and nonzero')
            return float(norm)
        norm = _phase('gradient_clip', clip, group)
        _ready('optimizer:ready', group)
        optimizer.step()
        policy.ema_(cfg.ema_decay)
        _ready('optimizer:complete', group)
    finally:
        optimizer.zero_grad(set_to_none=True)
    totals = _exchange(loss_sum, group)
    return dict(retained=global_kept, loss=sum(totals) / global_kept, grad_norm=norm,
                optimizer_step=True, replay_slots=len(candidates))


def endpoint_decisions(rows, tracker, cfg):
    """Gate anchors before updating accepted endpoint history by record key."""
    if not rows:
        return {}, {}
    masks = {row['candidate_index']: row['anchor_s_id'] >= cfg.tau_id for row in rows}
    accepted = [row for row in rows if masks[row['candidate_index']]]
    if not accepted:
        return masks, {}
    advantages = tracker.update([row['record_key'] for row in accepted], [row['geo'] for row in accepted])
    weights = {}
    for row, advantage in zip(accepted, advantages):
        index = row['candidate_index']
        probability = gap_to_probability(quality_gap(row['quality'], row['reference_quality']))
        weights[index] = float(effective_weight(torch.tensor(float(advantage)),
            float(getattr(cfg, 'adv_clip_max', 5.)), torch.tensor(row['motion']), cfg.tau_motion,
            torch.tensor(probability), cfg.tau_quality))
    return masks, weights


def _broadcast(value, devices):
    if devices.control_group is None:
        return value
    values = [value if devices.rank == 0 else None]
    dist.broadcast_object_list(values, src=0, group=devices.control_group)
    return values[0]


def _gather_rows(rows, devices):
    return sorted([row for rank_rows in _exchange(rows, devices.control_group) for row in rank_rows],
                  key=lambda row: row['candidate_index'])


def _score_row(reply, *, index, record, seed):
    result = dict(candidate_index=index, sample_id=record['sample_id'], prompt=record['text'], seed=seed)
    for name in ('geo', 's_id', 'motion'):
        value = reply[name]
        if len(value) != 1 or not math.isfinite(float(value[0])):
            raise ValueError('endpoint scores must contain one finite scalar per candidate')
        result[name] = float(value[0])
    result['quality'] = {name: float(reply['quality'][name]) for name in ('VQ', 'MQ', 'TA', 'Overall')}
    if not all(math.isfinite(value) for value in result['quality'].values()):
        raise ValueError('quality scores must be finite')
    return result


def _conditioning(cache, manifest, record):
    return load_conditioning(cache, record['sample_id'],
                             expected_identity=manifest['records'][record['sample_id']]['identity'])


def _training_reference(request, references, *, cfg, devices, policy, rollout, client, manifest):
    """Warm one common scalar cache using canonical owners on the assigned nodes.

    Every policy rank executes one base pass when any node needs a reference;
    each distinct record is scored once by its lowest assigned node's service.
    Cache hits never regenerate latents or transfer payloads.
    """
    record_key, record = request['record_key'], request['record']
    identity = dict(record_key=record_key, prompt=record['text'],
        seed=cfg.seed, adapter=None, geometry=request['geometry'], preparation=manifest['identity'])
    # File statistics/reference keys intentionally follow the legacy caption key.
    # Individual file IDs remain transport provenance, while HF records stay isolated.
    if request.get('record_source', 'hf') == 'hf':
        identity['sample_id'] = record['sample_id']
    node_size = devices.policy_count+len(devices.reward_ranks)
    owner = devices.node_index*node_size
    descriptors = _exchange((record_key, identity, devices.node_index) if devices.rank == owner else None,
                            devices.control_group)
    canonical = {}
    for descriptor in descriptors:
        if descriptor is not None:
            key, record_identity, node = descriptor
            if key in canonical and canonical[key][0] != record_identity:
                raise ValueError('conflicting training reference identities')
            canonical.setdefault(key, (record_identity, node))
    missing = _broadcast([key for key in canonical if key not in references]
                         if devices.rank == 0 else None, devices)

    def validate():
        if [key for key in canonical if key not in references] != missing:
            raise ValueError('training reference cache differs across ranks')
        for key, (record_identity, _) in canonical.items():
            if key in references and references[key]['identity'] != record_identity:
                raise ValueError('fixed training reference identity mismatch')
    _phase('training_reference:cache', validate, devices.control_group)
    if missing:
        query = _phase('training_reference:generate', lambda: rollout.rollout(request['conditioning'],
            seed=cfg.seed, geometry=request['geometry'], adapter=None) if policy else None, devices.control_group)

        def score():
            if record_key not in missing or canonical[record_key][1] != devices.node_index:
                return None
            metadata = dict(owner_rank=owner, candidate_index=devices.node_index*4, prompt=record['text'],
                sample_id=record['sample_id'], record_key=record_key, seed=cfg.seed, adapter=None,
                node_slot=devices.node_index, attempt=request.get('attempt', 'calibration'))
            reply = client.request('score_endpoint', f'training-reference/{record["sample_id"]}',
                {'video': query.video_endpoint.detach()} if devices.rank == owner else {}, metadata)
            if devices.rank == owner:
                return dict(identity=identity, scores=_score_row(reply, index=0, record=record, seed=cfg.seed))
            return None
        reference = _phase('training_reference:score', score, devices.control_group)
        for entry in _exchange(reference, devices.control_group):
            if entry is not None:
                references[entry['identity']['record_key']] = entry
    return references[record_key]


def _save_media(reply, path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_video(reply['video'][0], path.with_suffix('.mp4'), cfg.fps)
    audio = reply['audio']
    if audio.ndim != 3 or audio.shape[0] != 1:
        raise ValueError('decoded audio must be [1,channels,samples]')
    samples = (audio[0].clamp(-1, 1).T.contiguous()*32767).round().to(torch.int16).numpy()
    with wave.open(str(path.with_suffix('.wav')), 'wb') as stream:
        stream.setnchannels(samples.shape[1])
        stream.setsampwidth(2)
        stream.setframerate(reply['sampling_rate'])
        stream.writeframes(samples.tobytes())
    return dict(path=str(path.with_suffix('.mp4')), audio_path=str(path.with_suffix('.wav')))


def _evaluate_records(records, *, adapter, label, cfg, devices, policy, rollout, client,
                      cache, manifest, run_dir, verification=None):
    """Global held-out layout: each real record is generated exactly once."""
    width = devices.policy_count * devices.num_nodes
    rows, media = [], []
    node_size = devices.policy_count + len(devices.reward_ranks)
    geometry = {name: getattr(cfg, name) for name in ('num_frames', 'height', 'width', 'fps')}
    slots = h3_evaluation_assignments(devices.node_index, devices.num_nodes, max(devices.policy_index, 0),
                                      devices.policy_count, len(records))
    for round_index, assigned in enumerate(slots):
        index = assigned if assigned is not None else len(records)-1
        record = records[min(index, len(records)-1)]
        conditioning = _phase(f'{label}:conditioning:{round_index}',
            lambda: _conditioning(cache, manifest, record) if policy else None, devices.control_group)
        query = _phase(f'{label}:generate:{round_index}', lambda: rollout.rollout(conditioning,
            seed=cfg.seed+100_000+min(index, len(records)-1), geometry=geometry, adapter=adapter) if policy else None,
            devices.control_group)
        def score_node():
            for local_index in range(devices.policy_count):
                item_index = round_index*width + devices.node_index*devices.policy_count + local_index
                if item_index >= len(records):
                    continue
                owner = devices.node_index*node_size + local_index
                item_record = records[item_index]
                metadata = dict(owner_rank=owner, candidate_index=item_index, prompt=item_record['text'],
                                sample_id=item_record['sample_id'], seed=cfg.seed+100_000+item_index,
                                adapter=adapter, baseline_identity=manifest['identity'],
                                node_slot=devices.node_index, record_key=item_record['sample_id'], attempt=label)
                payload = {'video': query.video_endpoint} if devices.rank == owner else {}
                reply = client.request('score_endpoint', f'{label}/{item_index}', payload, metadata)
                if devices.rank == owner:
                    rows.append(_score_row(reply, index=item_index, record=item_record, seed=metadata['seed']))
                if cfg.save_videos:
                    payload = dict(video=query.video_endpoint, audio=query.audio_endpoint) if devices.rank == owner else {}
                    decoded = client.request('decode_evaluation', f'{label}/media/{item_index}', payload, metadata)
                    def save():
                        if devices.rank == owner:
                            if verification is not None:
                                verification.waveform(decoded, cfg)
                            entry = _save_media(decoded, run_dir/'eval'/label/f'prompt-{item_index:03d}', cfg)
                            if verification is not None:
                                entry['sampling_rate'] = decoded['sampling_rate']
                            media.append(dict(entry, prompt=item_record['text'], seed=metadata['seed'],
                                              candidate_index=item_index))
                    _phase(f'{label}:media:{round_index}:{local_index}', save, devices.node_control_group)
        _phase(f'{label}:node_scores:{round_index}', score_node, devices.control_group)
    rows = _gather_rows(rows, devices)
    videos = _gather_rows(media, devices)
    if len(rows) != len(records):
        raise RuntimeError('evaluation did not produce exactly the requested global records')
    return rows, videos


def run_h3_training(cfg, *, run_dir: Path, resume: Path | None = None, verification=None,
                    num_nodes=None, policy_count=None, reward_count=None) -> dict:
    """Install termination flags before startup, restoring prior handlers on exit."""
    if verification is not None and verification.phase == 'resume' and resume is None:
        raise ValueError('resume verification requires --resume')
    for actual, expected, name in ((getattr(cfg, 'num_nodes', 1), num_nodes, 'M'),
            (len(cfg.policy_gpus), policy_count, 'P'), (len(cfg.target_gpus), reward_count, 'R')):
        if expected is not None and actual != expected:
            raise ValueError(f'H3 explicit {name} topology disagrees with configuration')
    boundary = H3SafeBoundary(walltime_seconds=getattr(cfg, 'walltime_seconds', 0),
        publication_reserve_seconds=getattr(cfg, 'checkpoint_publication_reserve_seconds', 60),
        deadline=getattr(cfg, 'walltime_deadline', None))
    with boundary.signals():
        return _run_h3_training(cfg, run_dir=run_dir, resume=resume, boundary=boundary, verification=verification)


def _run_h3_training(cfg, *, run_dir, resume, boundary, verification=None):
    """Execute native cached t2va sampling, service targets, fitting and persistence.

    Restore precedes generation; every successful update publishes an atomic
    adapter snapshot at a collective service-idle boundary.
    """
    timeouts = resolve_h3_timeouts(cfg)
    devices = init_h3_distributed(policy_gpus=tuple(cfg.policy_gpus), target_gpus=tuple(cfg.target_gpus),
                                  timeout_s=timeouts['collective_timeout_s'])
    group = devices.control_group
    _phase('topology', lambda: _validate_node_count(devices, getattr(cfg, 'num_nodes', 1)), group)
    run_dir = Path(run_dir)
    phases = []

    def phase(name, operation):
        profiler = H3PhaseProfiler(devices.device, name, metadata={'rank': devices.rank})
        def measured():
            with profiler:
                return operation()
        try:
            return _phase(name, measured, group)
        finally:
            measurement = dict(getattr(profiler, 'metrics', {}), phase=name, rank=devices.rank)
            if verification is not None:
                from diffusionopsd.video.h3_verification import host_cgroup
                measurement['host_cgroup'] = host_cgroup()
            phases.append(measurement)

    def checkpoint_api():
        from diffusionopsd.video.h3_checkpoint import load_h3_checkpoint, save_h3_checkpoint
        return load_h3_checkpoint, save_h3_checkpoint
    load_checkpoint, save_checkpoint = phase('checkpoint_dependency', checkpoint_api)
    cache = Path(os.path.expandvars(str(cfg.conditioning_cache)))
    preflight = phase('preparation_validation', lambda: validate_h3_assets(cfg, cache))
    manifest = phase('manifest', lambda: json.loads(Path(preflight['manifest']).read_text()))
    data = phase('dataset', lambda: load_training_data(cfg, offline=True))
    selected = phase('record_selection', lambda: select_records(data, cfg))
    phase('output_directory', lambda: run_dir.mkdir(parents=True, exist_ok=True))
    policy = phase('policy_loading', lambda: load_h3_policy(model_root(cfg), cfg.model_revision, devices,
        rank=cfg.lora_rank, alpha=cfg.lora_alpha, seed=cfg.seed) if devices.role == 'policy' else None)
    service = phase('service_loading', lambda: H3RewardService.from_config(cfg, devices, run_dir)
                    if devices.role == 'reward' else None)
    rollout = phase('native_core', lambda: H3Rollout(build_h3_core(model_root(cfg), policy),
        query_sigma=cfg.query_sigma, num_sigma_points=cfg.num_steps) if policy else None)
    optimizer = phase('optimizer_loading', lambda: torch.optim.AdamW(policy.trainable_parameters,
        lr=cfg.learning_rate) if policy else None)
    if verification is not None:
        try:
            phase('verification_load', lambda: verification.loaded(policy, selected, cfg, preflight))
        except Exception:
            (run_dir/f'policy-phases-rank-{devices.rank}.json').write_text(json.dumps(phases, indent=2))
            if service is not None and service.judge is not None:
                service.judge.close()
            raise
        if verification.phase == 'load':
            try:
                return verification.write(run_dir, devices, reached='load', phases=phases)
            finally:
                (run_dir/f'policy-phases-rank-{devices.rank}.json').write_text(json.dumps(phases, indent=2))
                if service is not None and service.judge is not None:
                    service.judge.close()
    run_id = _broadcast(uuid.uuid4().hex if devices.rank == 0 else None, devices)
    client = H3RewardClient(devices, service, run_id=run_id,
                           timeout_s=timeouts['service_timeout_s'])
    tracker = PerPromptStatTracker()
    cursor = PromptCursor(len(selected['train']))
    configuration = cfg.to_dict() if hasattr(cfg, 'to_dict') else vars(cfg).copy()
    configuration['h3_provenance'] = preflight
    configuration['h3_execution'] = dict(rank=devices.rank, role=devices.role, owned_device=str(devices.device),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        local_world_size=os.environ.get('LOCAL_WORLD_SIZE'), world_size=os.environ.get('WORLD_SIZE'))
    training_keys = ('beta', 'rho', 'n_ascent', 'eta', 'loss_scale', 'max_grad_norm', 'ema_decay',
        'learning_rate', 'lora_rank', 'lora_alpha', 'seed', 'updates', 'query_sigma', 'num_steps',
        'video_shift', 'audio_shift', 'tau_id', 'tau_motion', 'tau_quality', 'adv_clip_max',
        'reward_frames', 'reward_height', 'reward_width', 'train_pool_size', 'calibration_prompts', 'eval_count')
    signature = dict(identity=preflight['identity'], topology=preflight['topology'],
                     manifest_sha256=preflight['manifest_sha256'],
                     geometry=manifest['geometry'],
                     training={key: configuration[key] for key in training_keys if key in configuration})
    state = dict(update=0, signature=signature, prompt_cursor=cursor.state_dict(), tracker={},
                 baselines={}, training_references={}, calibration=None, experiment_group=None,
                 last_evaluation_update=None)
    if resume is not None:
        state, rank_state = phase('restore', lambda: load_checkpoint(resume, policy, optimizer,
            expected_signature=signature, all_policy_group=devices.all_policy_group,
            node_policy_group=devices.policy_group, topology=signature['topology'], control_group=group))
        cursor = PromptCursor(**state['prompt_cursor'])
        tracker.stats = state['tracker']
        tracker.history_prompts = {hash(prompt) for prompt in tracker.stats}
        client.sequence = rank_state['service_sequence']
        client.run_id = rank_state['service_run_id']
        expected_positions = _node_positions(cursor.cursor, devices.num_nodes, cursor.num_prompts)
        if state.get('node_positions', expected_positions) != expected_positions:
            raise ValueError('checkpoint node sampling positions disagree with global attempt')
    if verification is not None and verification.phase == 'resume':
        try:
            phase('verification_resume_input', lambda: verification.restored(resume, state, rank_state, policy,
                                                                           optimizer, devices))
        except Exception:
            (run_dir/f'policy-phases-rank-{devices.rank}.json').write_text(json.dumps(phases, indent=2))
            if service is not None and service.judge is not None:
                service.judge.close()
            raise
    logger = VideoTrainingLogger(run_dir, configuration, 'minimax-h3', rank=devices.rank,
        start_update=state['update'], resume_checkpoint=resume, experiment_group=state['experiment_group'],
        owned_cuda_device=devices.device if devices.device.type == 'cuda' else None)
    phase('logger_start', logger.start)
    state['experiment_group'] = _broadcast(logger.experiment_group, devices)
    starting_update = state['update']
    local_budget = int(getattr(cfg, 'max_updates_this_job', 0))
    if local_budget < 0:
        raise ValueError('max_updates_this_job must be nonnegative; zero runs to the total budget')
    stopping_update = min(cfg.updates, starting_update+local_budget) if local_budget else cfg.updates
    if verification is not None:
        stopping_update = min(cfg.updates, starting_update+1)
    longest_update = state.get('longest_update_seconds', 0.)
    longest_publication = state.get('longest_publication_seconds', 0.)
    last_saved_cursor = None

    def publish():
        nonlocal longest_publication, last_saved_cursor
        state.update(prompt_cursor=cursor.state_dict(),
            node_positions=_node_positions(cursor.cursor, devices.num_nodes, cursor.num_prompts),
            tracker=_broadcast(tracker.stats if devices.rank == 0 else None, devices),
            longest_update_seconds=longest_update, longest_publication_seconds=longest_publication)
        destination = run_dir/f'update-{state["update"]:06d}.pt'
        # A previously published snapshot is immutable, including after rejected attempts.
        exists = _broadcast(destination.exists() if devices.rank == 0 else None, devices)
        if exists:
            destination = Path(_broadcast(str(run_dir/f'update-{state["update"]:06d}-{uuid.uuid4().hex}.pt')
                if devices.rank == 0 else None, devices))
        publication_started = time.monotonic()
        phase('checkpoint', lambda: save_checkpoint(destination, policy, optimizer, state=state,
            rank_state=dict(service_sequence=client.sequence, service_run_id=client.run_id),
            all_policy_group=devices.all_policy_group, node_policy_group=devices.policy_group,
            topology=signature['topology'], control_group=group))
        if verification is not None and state['update'] > starting_update:
            phase('verification_checkpoint', lambda: verification.checkpoint(destination, state, devices))
        longest_publication = max(longest_publication, *(_exchange(time.monotonic()-publication_started, group)))
        state['longest_publication_seconds'] = longest_publication
        last_saved_cursor = cursor.state_dict()

    try:
        if verification is not None and starting_update >= stopping_update:
            raise ValueError('verification requires an available optimizer update budget')
        if state['update'] < stopping_update and boundary.should_stop(longest_update, longest_publication, group):
            publish()
            phase('logger_finish', logger.finish)
            if verification is not None:
                return verification.write(run_dir, devices, reached='incomplete', phases=phases, state=state)
            return dict(update=state['update'], prompt_cursor=cursor.state_dict(), phases=phases,
                        run_dir=str(run_dir), status='checkpointed')
        if state['calibration'] is None:
            def calibrate():
                rows = []
                for record in selected['calibration']:
                    conditioning = _phase('calibration:conditioning', lambda: _conditioning(cache, manifest,
                        record) if policy else None, group)
                    request = dict(record=record, record_source='hf' if data.is_hf else 'file',
                        record_key=record['sample_id'] if data.is_hf else record['text'], conditioning=conditioning,
                        geometry={name: getattr(cfg, name) for name in ('num_frames', 'height', 'width', 'fps')})
                    reference = _training_reference(request, state['training_references'], cfg=cfg,
                        devices=devices, policy=policy, rollout=rollout, client=client, manifest=manifest)
                    rows.append(reference['scores'])
                return rows
            calibration = phase('calibration', calibrate)
            state['calibration'] = dict(rows=calibration,
                tau_id=cfg.tau_id if cfg.tau_id >= 0 else percentile_threshold(
                    torch.tensor([row['s_id'] for row in calibration]), .10),
                tau_motion=cfg.tau_motion if cfg.tau_motion >= 0 else percentile_threshold(
                    torch.tensor([row['motion'] for row in calibration]), .10))
        cfg.tau_id, cfg.tau_motion = state['calibration']['tau_id'], state['calibration']['tau_motion']

        def evaluate(update):
            if 'evaluation' not in state['baselines'] or verification is not None:
                base, base_media = _evaluate_records(selected['evaluation'], adapter=None, label='base', cfg=cfg,
                    devices=devices, policy=policy, rollout=rollout, client=client, cache=cache,
                    manifest=manifest, run_dir=run_dir, verification=verification)
                state['baselines']['evaluation'] = base
            rows, videos = _evaluate_records(selected['evaluation'], adapter='default', label=f'update-{update:06d}',
                cfg=cfg, devices=devices, policy=policy, rollout=rollout, client=client, cache=cache,
                manifest=manifest, run_dir=run_dir, verification=verification)
            def numeric(row):
                return {key: row[key] for key in ('geo', 's_id', 'motion')} | row['quality']
            metrics = evaluation_metrics([numeric(row) for row in rows],
                [numeric(row) for row in state['baselines']['evaluation']])
            if devices.rank == 0:
                (run_dir/'eval'/f'update-{update:06d}.json').parent.mkdir(parents=True, exist_ok=True)
                (run_dir/'eval'/f'update-{update:06d}.json').write_text(json.dumps(dict(rows=rows,
                    baselines=state['baselines']['evaluation'], metrics=metrics, media=videos), indent=2))
                logger.log_eval(update, metrics, videos)
            if verification is not None:
                # Already inside the evaluation profiler: nested profilers reset
                # global CUDA peak counters and would discard generation peaks.
                _phase('verification_media', lambda: verification.media(base_media, videos, cfg)
                       if devices.rank == 0 else None, group)
            state['last_evaluation_update'] = update

        if state['last_evaluation_update'] != state['update'] and evaluation_due(
                state['update'], cfg.updates, cfg.eval_every):
            phase('evaluation', lambda: evaluate(state['update']))
        while state['update'] < stopping_update:
            if boundary.should_stop(longest_update, longest_publication, group):
                if last_saved_cursor != cursor.state_dict():
                    publish()
                break
            started = time.monotonic()
            def request_layout():
                _, attempt = cursor.take()
                requests = []
                # select_records already provides the single deterministic seed permutation.
                order = list(range(cursor.num_prompts))
                for node in range(devices.num_nodes):
                    prompt_index = h3_candidate_assignments(attempt, node, devices.num_nodes,
                                                            0, devices.policy_count, order)[0][0]
                    record = selected['train'][prompt_index]
                    requests.append(dict(record=record, record_source='hf' if data.is_hf else 'file',
                        record_key=record['sample_id'] if data.is_hf else record['text'],
                        attempt=attempt, node_slot=node, seed=cfg.seed+(attempt*devices.num_nodes+node)*4,
                        geometry={name: getattr(cfg, name) for name in ('num_frames', 'height', 'width', 'fps')}))
                return requests
            requests = phase('request_broadcast', lambda: _broadcast(request_layout() if devices.rank == 0 else None, devices))
            request = requests[devices.node_index]
            if devices.rank != 0:
                cursor.take()
            request['conditioning'] = phase('request_conditioning', lambda: _conditioning(cache, manifest,
                request['record']) if policy else None)
            reference = phase('training_reference', lambda: _training_reference(request, state['training_references'],
                cfg=cfg, devices=devices, policy=policy, rollout=rollout, client=client, manifest=manifest))
            candidates = phase('behavior_rollout', lambda: collect_candidates(policy, rollout, request,
                rank=devices.rank, policy_count=devices.policy_count))
            if verification is not None:
                first = phase('verification_rollout', lambda: verification.rollout(candidates, request))
                del candidates
                candidates = phase('verification_native_repeat', lambda: collect_candidates(policy, rollout, request,
                    rank=devices.rank, policy_count=devices.policy_count))
                second = phase('verification_repeat_audit', lambda: verification.rollout(candidates, request))
                evidence = phase('verification_candidate_gather', lambda: _exchange((first, second), group))
                phase('verification_reproducibility', lambda: verification.reproducibility(evidence, devices))
                if verification.phase == 'rollout':
                    phase('logger_finish', logger.finish)
                    return verification.write(run_dir, devices, reached='rollout', phases=phases, state=state)
            by_index = {candidate.candidate_index: candidate for candidate in candidates}
            endpoint_rows = []
            node_size = devices.policy_count + len(devices.reward_ranks)
            def score_node():
                for local_index in range(4):
                    index = devices.node_index*4+local_index
                    owner = devices.node_index*node_size + local_index % devices.policy_count
                    metadata = dict(owner_rank=owner, candidate_index=index, prompt=request['record']['text'],
                                    record_key=request['record_key'], sample_id=request['record']['sample_id'],
                                    node_slot=devices.node_index, attempt=request['attempt'], local_candidate_index=local_index)
                    reply = client.request('score_endpoint', f'endpoint/{request["attempt"]}/{index}',
                        {'video': by_index[index].query.video_endpoint} if devices.rank == owner else {}, metadata)
                    anchor_reply = client.request('score_anchor', f'anchor/{request["attempt"]}/{index}',
                        {'video': by_index[index].query.y0.detach()} if devices.rank == owner else {}, metadata)
                    def make_row():
                        if devices.rank == owner:
                            row = _score_row(reply, index=index, record=request['record'], seed=request['seed']+local_index)
                            anchor_identity = anchor_reply['s_id']
                            if len(anchor_identity) != 1 or not math.isfinite(float(anchor_identity[0])):
                                raise ValueError('anchor identity must contain one finite scalar')
                            row.update(anchor_s_id=float(anchor_identity[0]), record_key=request['record_key'],
                                       reference_quality=reference['scores']['quality'])
                            endpoint_rows.append(row)
                    _phase(f'endpoint:{index}', make_row, devices.node_control_group)
            phase('node_endpoint_scores', score_node)
            endpoint_rows = phase('endpoint_gather', lambda: _gather_rows(endpoint_rows, devices))
            masks, weights = phase('tracker_decisions', lambda: _broadcast(endpoint_decisions(
                endpoint_rows, tracker, cfg) if devices.rank == 0 else None, devices))
            retained = sum(masks.values())
            if verification is not None:
                def gate_evidence():
                    # Append compact scalar evidence before the terminal full-pool gate can raise.
                    row = dict(attempt=request['attempt'], update=state['update'],
                        calibration=state['calibration'], global_accepted=retained,
                        candidates=[dict(candidate_index=item['candidate_index'], seed=item['seed'],
                            record_key=item['record_key'], sample_id=item['sample_id'],
                            anchor_s_id=item['anchor_s_id'], endpoint_s_id=item['s_id'],
                            accepted=masks[item['candidate_index']], margin=item['anchor_s_id']-cfg.tau_id)
                            for item in endpoint_rows])
                    with (run_dir/f'gate-decisions-rank-{devices.rank}.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                phase('verification_gate_evidence', gate_evidence)
            phase('prompt_gate', lambda: cursor.record_batch(retained))
            state['prompt_cursor'] = cursor.state_dict()
            targets = {}
            def target_node():
                for local_index in range(4):
                    index = devices.node_index*4+local_index
                    if not masks[index]:
                        continue
                    owner = devices.node_index*node_size + local_index % devices.policy_count
                    metadata = dict(owner_rank=owner, candidate_index=index, accepted=True,
                        node_slot=devices.node_index, record_key=request['record_key'],
                        sample_id=request['record']['sample_id'], attempt=request['attempt'], local_candidate_index=local_index)
                    reply = client.request('construct_targets', f'target/{request["attempt"]}/{index}',
                        {'video': by_index[index].query.y0.detach()} if devices.rank == owner else {}, metadata)
                    if devices.rank == owner:
                        targets[index] = H3Targets(reply['positive'], reply['negative'], reply['diagnostics'])
            phase('node_targets', target_node)
            for candidate in candidates:
                candidate.dummy = not masks[candidate.candidate_index]
            if verification is not None and retained:
                phase('verification_target', lambda: verification.targets(targets, retained,
                    anchors={index: by_index[index].query.y0 for index in targets}))
                if verification.phase == 'target':
                    phase('logger_finish', logger.finish)
                    return verification.write(run_dir, devices, reached='target', phases=phases, state=state)
            metrics = phase('fit', lambda: fit_h3_update(policy, rollout, candidates, targets=targets,
                weights=weights, global_kept=retained, optimizer=optimizer, cfg=cfg) if policy else None)
            if verification is not None and retained:
                phase('verification_update', lambda: verification.updated(policy, metrics, cfg) if policy else None)
            if retained:
                state['update'] += 1
                phase('train_log', lambda: logger.log_train(state['update'], metrics) if devices.rank == 0 else None)
                if verification is not None or evaluation_due(state['update'], cfg.updates, cfg.eval_every):
                    phase('evaluation', lambda: evaluate(state['update']))
            longest_update = max(longest_update, *(_exchange(time.monotonic()-started, group)))
            state.update(prompt_cursor=cursor.state_dict(), tracker=_broadcast(tracker.stats if devices.rank == 0 else None,
                devices), longest_update_seconds=longest_update)
            if retained:
                publish()
        phase('logger_finish', logger.finish)
        if verification is not None:
            reached = verification.phase if state['update'] > starting_update else 'incomplete'
            return verification.write(run_dir, devices, reached=reached, phases=phases, state=state)
        return dict(update=state['update'], prompt_cursor=cursor.state_dict(), phases=phases,
                    run_dir=str(run_dir), status='finished' if state['update'] == cfg.updates else 'checkpointed')
    except Exception as error:
        state['prompt_cursor'] = cursor.state_dict()
        logger.finish('failed')
        if verification is not None:
            try:
                verification.write(run_dir, devices, reached='failed', phases=phases, state=state)
            except Exception as diagnostic_error:
                error.add_note(f'Verification failure report unavailable: {diagnostic_error}')
        raise
    finally:
        (run_dir/f'policy-phases-rank-{devices.rank}.json').write_text(json.dumps(phases, indent=2))
        if service is not None and service.judge is not None:
            service.judge.close()
