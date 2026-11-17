"""Native H3 joint trajectory contracts on a small CPU transformer."""
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from diffusionopsd.video.h3_rollout import H3Rollout, build_h3_core, vae_descriptor


@pytest.mark.parametrize('incoming_device', ['cpu', 'cuda:0'])
def test_rollout_keeps_native_layout_tags_on_cpu_and_embeddings_on_policy_device(monkeypatch, incoming_device):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    from torch._subclasses.fake_tensor import FakeTensorMode, unset_fake_temporarily

    class LayoutReached(Exception):
        pass

    tags = torch.tensor([1, 0], dtype=torch.int32)
    embeds = torch.randn(1, 2, 5)
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    if incoming_device != 'cpu':
        with mode:
            tags = tags.to(incoming_device)
    transformer = torch.nn.Identity()

    class LayoutBoundary:
        fps = 24
        _execution_device = torch.device('cuda:1')
        policy = SimpleNamespace(generation_adapter=lambda adapter: nullcontext())

        def __call__(self, *, prompt_embeds, text_token_tags, **kwargs):
            # Run the real CPU layout outside the ambient fake mode: otherwise
            # its float64 position-grid scalar reads are data-dependent fakes.
            with unset_fake_temporarily():
                packed = MiniMaxH3PrepareLayoutStep.build_packed_sequence(text_token_tags,
                    num_latent_frames=2, latent_height=4, latent_width=4, num_audio_latents=3,
                    patch_size=(1, 2, 2), audio_channels=2, audio_tag=2, video_tag=0)
                assert prompt_embeds.device == torch.device('cuda:1')
                assert prompt_embeds.shape == (1, 2, 5) and prompt_embeds.dtype == embeds.dtype
                assert text_token_tags.device.type == 'cpu'
                assert text_token_tags.shape == (2,) and text_token_tags.dtype == torch.int32
                assert packed[1].device.type == 'cpu' and packed[1].dtype == torch.long
                if incoming_device == 'cpu':
                    torch.testing.assert_close(text_token_tags, torch.tensor([1, 0], dtype=torch.int32))
                    torch.testing.assert_close(packed[1], torch.tensor([1, 0, 2, 2, 2, 2, 2, 2,
                                                                      0, 0, 0, 0, 0, 0, 0, 0]))
            raise LayoutReached

    core = LayoutBoundary()
    core.transformer = transformer
    # Generator construction is the only CUDA runtime operation before core(...).
    # Replace it with a real CPU generator; fake tensors model device transfers.
    generator = torch.Generator
    monkeypatch.setattr(torch, 'Generator', lambda *, device: generator(device='cpu'))
    with mode, pytest.raises(LayoutReached):
        H3Rollout(core, query_sigma=.278, num_sigma_points=3).rollout(
            dict(prompt_embeds=embeds, text_token_tags=tags), seed=14,
            geometry=dict(height=32, width=32, num_frames=124, fps=24))
    assert not transformer._forward_hooks
    assert not torch.cuda.is_initialized()


def make_core(tmp_path):
    from diffusers import MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel
    from diffusionopsd.video.h3_fsdp import H3Devices, H3Policy, attention_lora_targets
    model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=8,
        hidden_size=16, num_layers=1, num_refiner_layers=1, ffn_dim=24, in_channels=2,
        audio_in_channels=3, patch_size=(1, 2, 2), text_dim=5, freq_dim=8,
        time_embed_hidden_dim=16, time_embed_dim=8, rope_freq_dim=1)
    from peft import LoraConfig
    model.requires_grad_(False)
    cfg = LoraConfig(r=2, lora_alpha=4, target_modules=list(attention_lora_targets(model)))
    model.add_adapter(cfg, adapter_name='default')
    model.add_adapter(cfg, adapter_name='old')
    model.set_adapter('default')
    policy = H3Policy(model, H3Devices.cpu())
    vcfg = dict(latent_channels=2, clip_length=17, token_drop=3,
        spatial_downsample_factors=[2, 2, 2, 2, 1, 1], temporal_downsample_factors=[1, 2, 2, 1, 1, 1],
        latents_mean=[0., 0.], latents_std=[1., 1.])
    acfg = dict(latent_channels=3, sampling_rate=32000, encoder_rates=[2, 4, 4, 5, 5])
    for name, cfg in [('vae', vcfg), ('audio_vae', acfg)]:
        folder = tmp_path / name
        folder.mkdir(exist_ok=True)
        (folder / 'config.json').write_text(json.dumps(cfg))
    return build_h3_core(tmp_path, policy, scheduler=MiniMaxH3Scheduler(shift=12),
                         audio_scheduler=MiniMaxH3Scheduler(shift=3))


def sample(core, seed=14):
    rollout = H3Rollout(core, query_sigma=.278, num_sigma_points=50)
    conditioning = dict(prompt_embeds=torch.randn(1, 2, 5, generator=torch.Generator().manual_seed(123)), text_token_tags=torch.ones(2, dtype=torch.long))
    record = rollout.rollout(conditioning, seed=seed,
                             geometry=dict(height=32, width=32, num_frames=124, fps=24))
    return rollout, record


def test_native_joint_replay_and_suffix(tmp_path):
    core = make_core(tmp_path)
    calls = []
    hook = core.transformer.register_forward_hook(lambda *args: calls.append(1))
    rollout, record = sample(core)
    assert len(calls) == 49
    assert record.query_index == int((record.video_schedule[:-1] - .278).abs().argmin())
    assert record.sigma_video != record.sigma_audio
    assert record.geometry['num_latent_frames'] == 37
    assert record.y0.shape == (1, 2, 37, 2, 2)
    assert record.query_index < 48  # Suffix must include a later joint recomputation.
    inputs = record.forward_inputs
    manual = inputs['hidden_states'][0].float() + record.sigma_video * record.video_velocity[0].float()
    torch.testing.assert_close(record.y0, manual.reshape(37, 1, 1, 2, 1, 2, 2).permute(
        3, 0, 4, 1, 5, 2, 6).reshape(1, 2, 37, 2, 2))
    time_by_row = inputs['timestep'][inputs['timestep_indices']]
    torch.testing.assert_close(time_by_row[inputs['video_indices']],
        torch.full((37,), 1 - record.sigma_video))
    assert not inputs['audio_hidden_states'].requires_grad
    prediction = rollout.prediction(record)
    torch.testing.assert_close(prediction, record.y0, rtol=1e-6, atol=1e-6)
    prediction.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in core.policy.trainable_parameters)
    video, audio = rollout.continue_from(record, record.y0)
    torch.testing.assert_close(video, record.video_endpoint, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(audio, record.audio_endpoint, rtol=1e-6, atol=1e-6)
    changed, changed_audio = rollout.continue_from(record, record.y0 + 1)
    assert not torch.equal(changed, video)
    assert changed_audio.shape == audio.shape
    assert not torch.equal(changed_audio, audio)
    hook.remove()
    core.policy.select_adapter('old')
    state = core(prompt_embeds=torch.randn(1, 2, 5, generator=torch.Generator().manual_seed(123)),
                 text_token_tags=torch.ones(2, dtype=torch.long), height=32, width=32, num_frames=124,
                 generator=torch.Generator().manual_seed(14), num_inference_steps=50)
    assert torch.equal(state.latents, record.video_endpoint)
    assert torch.equal(state.audio_latents, record.audio_endpoint)


def test_capture_clones_and_hooks_restore(tmp_path):
    core = make_core(tmp_path)
    forward = core.transformer.forward
    rollout, record = sample(core)
    assert core.transformer.forward == forward
    hook_count = len(core.transformer._forward_hooks)
    saved_audio = record.forward_inputs['audio_hidden_states'].clone()
    saved = record.forward_inputs['hidden_states'].clone()
    core.scheduler.sigmas.zero_()
    record.video_endpoint.zero_()
    assert torch.equal(saved, record.forward_inputs['hidden_states'])
    core.transformer.forward = lambda **kw: (_ for _ in ()).throw(RuntimeError('probe'))
    broken = core.transformer.forward
    with pytest.raises(RuntimeError, match='probe'):
        sample(core)
    assert core.transformer.forward == broken
    assert len(core.transformer._forward_hooks) == hook_count
    assert torch.equal(saved_audio, record.forward_inputs['audio_hidden_states'])


def test_suffix_failure_restores_both_scheduler_dicts_and_subsequent_use(tmp_path, monkeypatch):
    from diffusionopsd.video.h3_checkpoint import _digest
    core = make_core(tmp_path)
    rollout, record = sample(core)
    # Give entry state a different completed trajectory than the saved query.
    sample(core, seed=15)
    before = tuple(_digest(scheduler.__dict__) for scheduler in (core.scheduler, core.audio_scheduler))
    original = core.transformer.forward
    calls = []
    def fail_later(**kwargs):
        calls.append(1)
        assert core.scheduler._step_index > record.query_index
        assert core.audio_scheduler._step_index > record.query_index
        raise RuntimeError('suffix joint forward failed')
    with monkeypatch.context() as patch:
        patch.setattr(core.transformer, 'forward', fail_later)
        with pytest.raises(RuntimeError, match='suffix joint forward'):
            rollout.continue_from(record, record.y0)
    assert len(calls) == 1 and core.transformer.forward == original
    assert before == tuple(_digest(scheduler.__dict__) for scheduler in (core.scheduler, core.audio_scheduler))
    video, audio = rollout.continue_from(record, record.y0)
    torch.testing.assert_close(video, record.video_endpoint)
    torch.testing.assert_close(audio, record.audio_endpoint)
    assert before == tuple(_digest(scheduler.__dict__) for scheduler in (core.scheduler, core.audio_scheduler))


def test_descriptor_from_config_has_no_weights(tmp_path, monkeypatch):
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    def forbidden(*args, **kwargs):
        raise AssertionError('policy factory must never construct a VAE')

    monkeypatch.setattr(AutoencoderKLMiniMaxH3, '__init__', forbidden)
    monkeypatch.setattr(AutoencoderKLMiniMaxH3Audio, '__init__', forbidden)
    core = make_core(tmp_path)
    desc = vae_descriptor(tmp_path / 'vae' / 'config.json')
    assert desc.spatial_compression_ratio == 16
    assert desc.temporal_compression_ratio == 4
    assert desc.tokens_chunk_size == 5
    assert desc.token_overlap == 2
    assert desc.frame_overlap == 5
    assert not isinstance(desc, torch.nn.Module)
    assert set(core._component_specs) == {'transformer', 'scheduler', 'audio_scheduler'}


@pytest.mark.parametrize('name', ['scheduler', 'audio_scheduler'])
def test_loaded_scheduler_rejects_mismatching_shifts(tmp_path, name):
    from diffusers import MiniMaxH3Scheduler
    core = make_core(tmp_path)
    schedulers = dict(scheduler=core.scheduler, audio_scheduler=core.audio_scheduler)
    schedulers[name] = MiniMaxH3Scheduler(shift=7)
    with pytest.raises(ValueError, match='shift'):
        build_h3_core(tmp_path, core.policy, **schedulers)


def test_missing_old_adapter_is_refused(tmp_path):
    core = make_core(tmp_path)
    core.transformer.delete_adapters('old')
    with pytest.raises(ValueError, match='old'):
        sample(core)


def test_generation_adapter_selection_changes_native_joint_outputs(tmp_path):
    core = make_core(tmp_path)
    rollout = H3Rollout(core, query_sigma=.278, num_sigma_points=3)
    conditioning = dict(prompt_embeds=torch.randn(1, 2, 5), text_token_tags=torch.ones(2, dtype=torch.long))
    geometry = dict(height=32, width=32, num_frames=124, fps=24)
    with torch.no_grad():
        for name, parameter in core.transformer.named_parameters():
            if '.lora_B.default.' in name:
                parameter.fill_(.3)
    baseline = rollout.rollout(conditioning, seed=14, geometry=geometry, adapter=None)
    current = rollout.rollout(conditioning, seed=14, geometry=geometry, adapter='default')
    behavior = rollout.rollout(conditioning, seed=14, geometry=geometry)
    torch.testing.assert_close(baseline.video_endpoint, behavior.video_endpoint)
    assert not torch.equal(current.video_endpoint, behavior.video_endpoint)
    assert not torch.equal(current.audio_endpoint, behavior.audio_endpoint)
    assert all(parameter.requires_grad == core.policy._mask[name]
               for name, parameter in core.transformer.named_parameters())


def test_descriptors_match_native_vae_constructor(tmp_path):
    from diffusers import AutoencoderKLMiniMaxH3
    core = make_core(tmp_path)
    cfg = vars(core.vae.config)
    with torch.device('meta'):
        native = AutoencoderKLMiniMaxH3(**cfg, block_out_channels=(32, 32, 32, 32, 32, 32),
            decoder_num_layers=1, decoder_num_attention_heads=2, decoder_attention_head_dim=8)
    for name in ('spatial_compression_ratio', 'temporal_compression_ratio', 'frame_pre_padding',
                 'tokens_chunk_size', 'token_overlap', 'frame_overlap'):
        assert getattr(core.vae, name) == getattr(native, name)
    assert native.config.latents_mean == core.vae.config.latents_mean
    assert native.config.latents_std == core.vae.config.latents_std
    assert all(p.device.type == 'meta' for p in native.parameters())


def test_native_scheduler_mutation_does_not_alias_capture(tmp_path):
    core = make_core(tmp_path)
    saved = []
    original = core.scheduler.step
    audio_original = core.audio_scheduler.step

    def mutate(model_output, timestep, sample, return_dict=True):
        result = original(model_output, timestep, sample, return_dict=False)
        sample.copy_(result[0])
        return (sample,)

    def mutate_audio(model_output, timestep, sample, return_dict=True):
        result = audio_original(model_output, timestep, sample, return_dict=False)
        sample.copy_(result[0])
        return (sample,)

    def reference(module, args, kwargs, output):
        q = int((core.scheduler.sigmas[:-1] - .278).abs().argmin())
        if core.scheduler._step_index == q:
            saved.append((kwargs['hidden_states'].clone(), kwargs['audio_hidden_states'].clone()))

    hook = core.transformer.register_forward_hook(reference, with_kwargs=True)
    core.scheduler.step, core.audio_scheduler.step = mutate, mutate_audio
    _, record = sample(core)
    hook.remove()
    assert len(saved) == 1
    assert torch.equal(record.forward_inputs['hidden_states'], saved[0][0])
    assert torch.equal(record.forward_inputs['audio_hidden_states'], saved[0][1])
