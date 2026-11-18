"""Paired diagnostics must isolate candidate changes from continuation state."""
from types import SimpleNamespace

import pytest
import torch


def test_candidate_gains_and_nonfinite_missingness():
    from diffusionopsd.video.curriculum_probe import score_candidates, target_gains
    rec = SimpleNamespace()
    result = score_candidates(rec, {"baseline": torch.tensor(1.), "positive": torch.tensor(2.),
                                   "negative": torch.tensor(0.), "reflected": torch.tensor(3.),
                                   "student_before": torch.tensor(4.), "student_after": torch.tensor(4.),
                                   "bad": torch.tensor(float('nan'))},
                              lambda record, y: y * 2, lambda x: x)
    assert result['positive'] == 4.
    assert result['bad'] is None
    gains = target_gains(result)
    assert gains == {'g_plus': 2., 'g_minus': -2., 'g_reflected': 4., 'delta_round': 0.}
    # A before/after difference must not be confused with the student/behavior gap.
    assert result['student_before'] - result['baseline'] == 6.


def test_candidate_scoring_resets_rng_for_every_candidate_and_restores_caller():
    from diffusionopsd.video.curriculum_probe import score_candidates
    torch.manual_seed(5)
    before = torch.get_rng_state().clone()
    def continuation(record, y):
        return y + torch.rand(())
    result = score_candidates(None, {'a': torch.tensor(0.), 'b': torch.tensor(0.)}, continuation, lambda x: x)
    assert result['a'] == result['b']
    assert torch.equal(torch.get_rng_state(), before)


def test_query_transfer_does_not_alias_solver_history():
    from diffusionopsd.video.curriculum_probe import query_to
    rec = SimpleNamespace(z_q=torch.tensor([1.]),
                          scheduler=SimpleNamespace(history=[torch.tensor([2.])]),
                          forward_inputs=[{'tokens': torch.tensor([3.])}])
    saved = query_to(rec, 'cpu')
    rec.z_q.add_(5)
    rec.scheduler.history[0].add_(5)
    rec.forward_inputs[0]['tokens'].add_(5)
    assert saved.z_q.item() == 1.
    assert saved.scheduler.history[0].item() == 2.
    assert saved.forward_inputs[0]['tokens'].item() == 3.


def test_adapter_context_restores_weights_active_flags_modes_rng_on_failure():
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from diffusionopsd.video.curriculum_probe import adapter_snapshot, restored_adapters
    model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(2, 2)), LoraConfig(r=1, target_modules=['0']))
    model.add_adapter('old', LoraConfig(r=1, target_modules=['0']))
    model.set_adapter('old')
    model.eval().requires_grad_(False)
    original = adapter_snapshot(model, 'old')
    modes = {n: m.training for n, m in model.named_modules()}
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='injected'):
        with restored_adapters(model):
            model.set_adapter('default')
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if 'lora' in name:
                        p.add_(1)
            torch.rand(())
            raise RuntimeError('injected')
    assert model.active_adapters == ['old']
    assert {n: m.training for n, m in model.named_modules()} == modes
    assert {n: p.requires_grad for n, p in model.named_parameters()} == flags
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(v, get_peft_model_state_dict(model, adapter_name='old')[k])
               for k, v in original.items())


def test_original_behavior_continuation_parity_and_zero_student_change():
    """A changed live EMA must not change saved-query comparisons."""
    from peft import LoraConfig, get_peft_model
    from diffusionopsd.video.cosmos_rollout import CosmosRollout
    from diffusionopsd.video.curriculum_probe import adapter_snapshot, audit_query, query_to
    from diffusionopsd.video.device_layout import InferenceReplica
    from test_cosmos_rollout import TinyPipeline, TinyPackedTransformer

    class AdapterTransformer(TinyPackedTransformer):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2, 2, bias=False)
        def forward(self, vision_tokens, **kwargs):
            prediction = super().forward(vision_tokens=vision_tokens, **kwargs)[0][0]
            return [prediction + self.projection(vision_tokens[0]) * .01], None, None

    torch.manual_seed(4)
    pipe = TinyPipeline()
    model = get_peft_model(AdapterTransformer(), LoraConfig(r=1, target_modules=['projection']))
    model.add_adapter('old', LoraConfig(r=1, target_modules=['projection']))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_A' in name:
                parameter.fill_(.2)
            elif 'lora_B.default' in name:
                parameter.fill_(.6)
            elif 'lora_B.old' in name:
                parameter.fill_(.1)
    pipe.transformer = model
    model.set_adapter('old')
    roll = CosmosRollout(pipe, 8, 6.)
    rec = roll.rollout('prompt', generator=torch.Generator().manual_seed(42), sigma_star=.4)
    entry = {'record': query_to(rec, 'cpu'), 'original_behavior': adapter_snapshot(model, 'old'),
             'targets': {'positive': rec.y0 + .01, 'negative': rec.y0 - .01},
             'metadata': {'prompt_id': 'x'}, 'effective_weight': .5}
    start = adapter_snapshot(model, 'default')
    # Current EMA differs from the EMA that produced the stored solver history.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B.old' in name:
                parameter.fill_(.9)
    replica = InferenceReplica(model, device='cpu')
    pipe.transformer = replica.model
    current_old = adapter_snapshot(replica.model, 'old')
    result = audit_query(entry, rollout=roll, inference=replica, round_start=start, round_end=start,
                         reward_fn=lambda y: -y.square().mean(), device='cpu', parity_tolerance=1e-5)
    assert result['valid']
    assert result['baseline_parity']['relative_l2'] < 1e-5
    assert result['final_gains']['delta_round'] == 0.
    assert abs(result['final_scores']['student_before'] - result['final_scores']['baseline']) > 1e-6
    assert all(torch.equal(v, adapter_snapshot(replica.model, 'old')[k]) for k, v in current_old.items())


def test_prompt_selection_is_balanced_and_input_order_independent():
    from diffusionopsd.video.curriculum_probe import select_probe_prompts
    records = [SimpleNamespace(prompt_id=f'{s}{i}', subtype=s) for s in ['orbit', 'dolly', 'lateral'] for i in range(3)]
    chosen = select_probe_prompts(records, 4, 42)
    assert chosen == select_probe_prompts(list(reversed(records)), 4, 42)
    assert len(set(chosen)) == 4
    assert {s for s in ['orbit', 'dolly', 'lateral'] if any(p.startswith(s) for p in chosen)} == {
        'orbit', 'dolly', 'lateral'}
    with pytest.raises(ValueError, match='count'):
        select_probe_prompts(records, 10, 42)


def test_adapter_context_supports_diffusers_callable_active_adapters():
    from diffusers import ModelMixin
    from diffusers.loaders import PeftAdapterMixin
    from peft import LoraConfig
    from diffusionopsd.video.curriculum_probe import restored_adapters
    class Model(ModelMixin, PeftAdapterMixin):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2, 2)
    model = Model()
    model.add_adapter(LoraConfig(r=1, target_modules=['projection']), adapter_name='default')
    model.add_adapter(LoraConfig(r=1, target_modules=['projection']), adapter_name='old')
    model.set_adapter('old')
    with restored_adapters(model):
        model.set_adapter('default')
    assert model.active_adapters() == ['old']


def test_audit_matches_native_precision_inside_an_autocast_caller():
    """An unchanged native prediction must stay unchanged during diagnostic replay."""
    from contextlib import contextmanager
    from peft import LoraConfig, get_peft_model
    from diffusionopsd.video.curriculum_probe import adapter_snapshot, audit_query
    torch.manual_seed(17)
    model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(2, 2)),
                          LoraConfig(r=1, target_modules=['0']))
    model.add_adapter('old', LoraConfig(r=1, target_modules=['0']))
    model.set_adapter('old')
    inputs = torch.tensor([[.12345, .67891]])
    native = model(inputs).detach()
    record = SimpleNamespace(y0=native, x0=native, inputs=inputs)
    class Replica:
        def __init__(self):
            self.model = model
        def select(self, name):
            self.model.set_adapter(name)
            self.model.eval().requires_grad_(False)
        @contextmanager
        def active_on(self, device):
            yield self.model
    class Rollout:
        def prediction(self, rec):
            return model(rec.inputs)
        def continue_from(self, rec, y):
            return y
    entry = {'record': record, 'original_behavior': adapter_snapshot(model, 'old'),
             'targets': {'positive': native+.01, 'negative': native-.01},
             'metadata': {}, 'effective_weight': .5}
    start = adapter_snapshot(model, 'default')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = audit_query(entry, rollout=Rollout(), inference=Replica(), round_start=start, round_end=start,
                             reward_fn=lambda x: x.float().square().mean(), device='cpu')
    assert result['local_scores']['student_before'] == result['local_scores']['baseline']
    assert result['final_gains']['delta_round'] == 0.
