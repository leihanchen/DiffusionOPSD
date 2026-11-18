"""Matched target and round diagnostics through a saved original continuation."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import math
import random

from peft import get_peft_model_state_dict, set_peft_model_state_dict
import torch

from diffusionopsd.video.cosmos_training import preserve_rng


def select_probe_prompts(records, count, seed):
    """Select prompt IDs before sampling, round-robin across available subtypes."""
    if type(count) is not int or not 0 <= count <= len(records):
        raise ValueError('Probe count must be an integer within the manifest size')
    groups = {}
    for record in sorted(records, key=lambda r: r.prompt_id):
        groups.setdefault(record.subtype, []).append(record.prompt_id)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        for subtype in sorted(groups):
            if groups[subtype] and len(selected) < count:
                selected.append(groups[subtype].pop())
    return selected


def adapter_snapshot(model, name):
    return {key: value.detach().cpu().clone()
            for key, value in get_peft_model_state_dict(model, adapter_name=name).items()}


@contextmanager
def restored_adapters(model):
    """Restore both adapters and all training flags even after an audit failure."""
    snapshots = {name: adapter_snapshot(model, name) for name in ('default', 'old')}
    active = model.get_active_adapters() if hasattr(model, 'get_active_adapters') else model.active_adapters
    active = copy.deepcopy(active() if callable(active) else active)
    flags = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    with preserve_rng():
        try:
            yield
        finally:
            for name, values in snapshots.items():
                set_peft_model_state_dict(model, values, adapter_name=name)
            model.set_adapter(active[0] if len(active) == 1 else active)
            for parameter, flag in flags:
                parameter.requires_grad_(flag)
            for module, mode in modes:
                module.training = mode


def query_to(record, device):
    """Clone tensors, captured inputs and native solver history onto one device."""
    def transfer(value):
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=device, copy=True)
        if isinstance(value, dict):
            return type(value)({key: transfer(item) for key, item in value.items()})
        if isinstance(value, list):
            return [transfer(item) for item in value]
        if isinstance(value, tuple):
            return tuple(transfer(item) for item in value)
        return copy.deepcopy(value)

    result = copy.copy(record)
    for key, value in vars(record).items():
        if key == 'scheduler':
            scheduler = copy.copy(value)
            scheduler.__dict__.update(transfer(vars(value)))
            setattr(result, key, scheduler)
        else:
            setattr(result, key, transfer(value))
    return result


@torch.no_grad()
def score_candidates(record, candidates, continue_fn, reward_fn, *, on_endpoint=None):
    """Reset RNG per candidate and retain nonfinite scores as explicit missingness."""
    scores = {}
    for name, target in candidates.items():
        with preserve_rng():
            endpoint = continue_fn(record, target)
            value = float(reward_fn(endpoint).mean())
            scores[name] = value if math.isfinite(value) else None
            if on_endpoint is not None:
                on_endpoint(name, endpoint)
    return scores


def target_gains(scores):
    def difference(after, before):
        a, b = scores.get(after), scores.get(before)
        return a - b if a is not None and b is not None else None
    return {'g_plus': difference('positive', 'baseline'),
            'g_minus': difference('negative', 'baseline'),
            'g_reflected': difference('reflected', 'baseline'),
            'delta_round': difference('student_after', 'student_before')}


def latent_parity(actual, expected, tolerance):
    expected = expected.to(actual.device).float()
    error = actual.float() - expected
    relative = float(error.norm() / expected.norm().clamp_min(1e-12))
    return dict(relative_l2=relative, max_abs=float(error.abs().max()),
                tolerance=tolerance, passed=relative <= tolerance)


@torch.no_grad()
def audit_query(entry, *, rollout, inference, round_start, round_end, reward_fn, device,
               on_endpoint=None, parity_tolerance=1e-3):
    """Use each query's original EMA for every suffix, with a fixed solver state."""
    model = inference.model
    record = entry['record']
    # Native generation has no outer autocast. Match its model and VAE execution
    # even when a training caller still has mixed precision enabled.
    with restored_adapters(model), torch.autocast(device_type=torch.device(device).type, enabled=False):
        candidates = {'baseline': record.y0, **entry['targets']}
        candidates['reflected'] = 2 * record.y0 - candidates['negative']
        for name, snapshot in [('student_before', round_start), ('student_after', round_end)]:
            set_peft_model_state_dict(model, snapshot, adapter_name='default')
            inference.select('default')
            with inference.active_on(device):
                query = query_to(record, device)
                candidates[name] = rollout.prediction(query).detach().cpu()
        # These immutable original weights also generated the captured solver prefix.
        set_peft_model_state_dict(model, entry['original_behavior'], adapter_name='old')
        inference.select('old')
        with inference.active_on(device):
            replayed = rollout.prediction(query_to(record, device))
        prefix_parity = latent_parity(replayed, record.y0, parity_tolerance)
        local = score_candidates(record, candidates, lambda rec, y: y.to(device), reward_fn)
        parity = {}

        def continue_original(rec, target):
            with inference.active_on(device):
                query = query_to(rec, device)
                endpoint = rollout.continue_from(query, target.to(device))
            return endpoint

        def observe(name, endpoint):
            if name == 'baseline':
                parity.update(latent_parity(endpoint, record.x0, parity_tolerance))
            if on_endpoint is not None:
                on_endpoint(name, endpoint)

        final = score_candidates(record, candidates, continue_original, reward_fn, on_endpoint=observe)
        # Keep a failed parity check visible; its gains cannot support conclusions.
        return {'metadata': entry['metadata'], 'effective_weight': entry['effective_weight'],
                'local_scores': local, 'final_scores': final,
                'local_gains': target_gains(local), 'final_gains': target_gains(final),
                'baseline_parity': parity, 'prefix_parity': prefix_parity,
                'missing_scores': sum(v is None for v in [*local.values(), *final.values()]),
                'valid': prefix_parity['passed'] and parity['passed'] and all(v is not None for v in final.values())}
