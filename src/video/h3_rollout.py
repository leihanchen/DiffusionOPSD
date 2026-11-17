"""Capture native joint H3 trajectories and replay only their video clean anchor."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone(item) for item in value)
    return deepcopy(value)


def vae_descriptor(config_path):
    """Reproduce constructor-derived VAE geometry without allocating a module."""
    cfg = json.loads(Path(config_path).read_text())
    temporal = math.prod(cfg.get('temporal_downsample_factors', [1]))
    result = SimpleNamespace(config=SimpleNamespace(**cfg))
    result.spatial_compression_ratio = math.prod(cfg.get('spatial_downsample_factors', [1]))
    result.temporal_compression_ratio = temporal
    if 'clip_length' in cfg:
        result.frame_pre_padding = (-cfg['clip_length']) % temporal
        result.tokens_chunk_size = math.ceil(cfg['clip_length'] / temporal)
        result.token_overlap = (-cfg['token_drop']) % result.tokens_chunk_size
        result.frame_overlap = max(result.token_overlap * temporal - result.frame_pre_padding, 0)
    return result


def build_h3_core(model_path, policy, *, scheduler=None, audio_scheduler=None):
    """Build the pinned t2va workflow, retaining native setup/loop/unpacking only.

    The sharded transformer is injected as-is. Only local schedulers are loaded;
    no encoder or decoder is constructed, and no pipeline dtype/device cast runs.
    """
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
    from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks

    root = Path(model_path)
    workflow = MiniMaxH3Blocks().get_workflow('t2va')
    blocks = SequentialPipelineBlocks.from_blocks_dict({
        name: block for name, block in workflow.sub_blocks.items()
        if name != 'text_encoder' and not name.startswith('decode.')
    })
    core = blocks.init_pipeline()
    core.update_components(transformer=policy.model,
        scheduler=scheduler if scheduler is not None else MiniMaxH3Scheduler.from_pretrained(
            root / 'scheduler', local_files_only=True),
        audio_scheduler=audio_scheduler if audio_scheduler is not None else MiniMaxH3Scheduler.from_pretrained(
            root / 'audio_scheduler', local_files_only=True))
    for name, expected in (('scheduler', 12.), ('audio_scheduler', 3.)):
        loaded = getattr(core, name)
        if (not isinstance(loaded, MiniMaxH3Scheduler) or loaded.config.shift != expected
                or loaded.shift != expected):
            raise ValueError(f'Pinned native {name} shift must be {expected}')
    # These are descriptors, deliberately not model components: no loader can
    # interpret them as checkpoint requests or recursively move their storage.
    core.vae = vae_descriptor(root / 'vae' / 'config.json')
    core.audio_vae = vae_descriptor(root / 'audio_vae' / 'config.json')
    # Keep serialization/loading identities local, including injected FSDP
    # models whose native loader metadata may otherwise still name the Hub.
    from diffusionopsd.video.h3_conditioning import MODEL_REVISION
    for name, spec in core._component_specs.items():
        spec.pretrained_model_name_or_path = str(root)
        spec.subfolder = name
        spec.revision = MODEL_REVISION
        spec.repo = str(root)
    core.policy = policy
    return core


def _unpack(rows, geometry):
    pt, ph, pw = geometry['patch_size']
    channels = geometry['latent_channels']
    frames, height, width = (geometry[key] for key in ('num_latent_frames', 'latent_height', 'latent_width'))
    return rows.reshape(-1, frames // pt, height // ph, width // pw, channels, pt, ph, pw).permute(
        0, 4, 1, 5, 2, 6, 3, 7).reshape(-1, channels, frames, height, width).contiguous()


def _pack(video, patch):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    return patchify_video_latents(video, patch)


@dataclass
class H3Query:
    forward_inputs: dict
    generated_video_rows: slice
    generated_audio_rows: slice
    sigma_video: float
    sigma_audio: float
    query_index: int
    video_schedule: torch.Tensor
    audio_schedule: torch.Tensor
    scheduler_states: tuple[dict, dict]
    video_velocity: torch.Tensor
    audio_velocity: torch.Tensor
    video_endpoint: torch.Tensor
    audio_endpoint: torch.Tensor
    geometry: dict
    row_timestep_plan: list

    @property
    def y0(self):
        rows = self.forward_inputs['hidden_states'][0, self.generated_video_rows].float()
        return _unpack(rows + self.sigma_video * self.video_velocity[0, self.generated_video_rows].float(),
                       self.geometry)


class H3Rollout:
    def __init__(self, core, *, query_sigma: float, num_sigma_points: int):
        if not 0 < query_sigma <= 1 or num_sigma_points < 2:
            raise ValueError('query_sigma must be in (0,1] and num_sigma_points >=2')
        self.core, self.query_sigma, self.num_sigma_points = core, query_sigma, num_sigma_points

    @torch.no_grad()
    def rollout(self, conditioning: dict, *, seed: int, geometry: dict, adapter: str | None = 'old') -> H3Query:
        core = self.core
        if geometry.get('fps', 24) != core.fps:
            raise ValueError('Native H3 requires 24 fps')
        captured = {}
        counter = 0

        def capture(module, args, kwargs, output):
            nonlocal counter
            video, audio = core.scheduler.sigmas, core.audio_scheduler.sigmas
            if video.numel() != audio.numel():
                raise ValueError('Joint H3 schedules have different lengths')
            q = int((video[:-1] - self.query_sigma).abs().argmin())
            if counter == q:
                captured.update(forward_inputs=_clone(kwargs), query_index=q,
                    sigma_video=float(video[q]), sigma_audio=float(audio[q]),
                    video_schedule=_clone(video), audio_schedule=_clone(audio),
                    scheduler_states=(_clone(core.scheduler.__dict__), _clone(core.audio_scheduler.__dict__)),
                    video_velocity=_clone(output[0]), audio_velocity=_clone(output[1]))
            counter += 1

        handle = core.transformer.register_forward_hook(capture, with_kwargs=True)
        try:
            device = core._execution_device
            with core.policy.generation_adapter(adapter):
                # Native layout construction consumes CPU text tags before
                # moving its completed packed metadata to the policy device.
                state = core(**{key: value.to('cpu' if key == 'text_token_tags' else device)
                                for key, value in conditioning.items()},
                    **{key: value for key, value in geometry.items() if key != 'fps'},
                    generator=torch.Generator(device=device).manual_seed(seed), num_inference_steps=self.num_sigma_points)
        finally:
            handle.remove()
        resolved = dict(geometry, **{key: getattr(state, key) for key in
                        ('num_latent_frames', 'latent_height', 'latent_width', 'num_audio_latents')},
                        patch_size=core.patch_size, latent_channels=core.vae_latent_channels)
        resolved.update(num_frames=state.num_frames, height=state.height, width=state.width)
        resolved['normalization'] = {
            name: {field: deepcopy(getattr(descriptor.config, field))
                   for field in ('latents_mean', 'latents_std') if hasattr(descriptor.config, field)}
            for name, descriptor in (('video', core.vae), ('audio', core.audio_vae))
        }
        return H3Query(**captured, generated_video_rows=slice(state.num_condition_video_rows, None),
            generated_audio_rows=slice(state.num_condition_audio_rows, None),
            video_endpoint=_clone(state.latents), audio_endpoint=_clone(state.audio_latents), geometry=resolved,
            row_timestep_plan=_clone(state.row_timestep_plan))

    def prediction(self, record: H3Query) -> torch.Tensor:
        self.core.policy.select_adapter('default')
        video, _ = self.core.transformer(**_clone(record.forward_inputs))
        rows = record.forward_inputs['hidden_states'][0, record.generated_video_rows].float()
        return _unpack(rows + record.sigma_video * video[0, record.generated_video_rows].float(), record.geometry)

    @torch.no_grad()
    def continue_from(self, record: H3Query, forced_video_y: torch.Tensor):
        from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopSchedulerStep
        from diffusers.modular_pipelines.modular_pipeline import BlockState

        core = self.core
        core.policy.select_adapter('old')
        previous = (_clone(core.scheduler.__dict__), _clone(core.audio_scheduler.__dict__))
        try:
            for scheduler, saved in zip((core.scheduler, core.audio_scheduler), record.scheduler_states):
                scheduler.__dict__.clear()
                scheduler.__dict__.update(_clone(saved))
            inputs = _clone(record.forward_inputs)
            video, audio = inputs['hidden_states'][0], inputs['audio_hidden_states'][0]
            noise = record.video_velocity.clone()
            noise[0, record.generated_video_rows] = (
                _pack(forced_video_y.float(), core.patch_size) - video[record.generated_video_rows]
            ) / record.sigma_video
            update = MiniMaxH3LoopSchedulerStep()
            for i in range(record.query_index, len(record.video_schedule) - 1):
                if i == record.query_index:
                    audio_noise = record.audio_velocity.clone()
                else:
                    inputs.update(hidden_states=video[None], audio_hidden_states=audio[None],
                                  timestep=record.row_timestep_plan[i][0],
                                  timestep_indices=record.row_timestep_plan[i][1])
                    noise, audio_noise = core.transformer(**inputs)
                state = BlockState(latents=video, audio_latents=audio, noise_pred=noise,
                    audio_noise_pred=audio_noise, audio_timesteps=core.audio_scheduler.timesteps,
                    num_condition_video_rows=record.generated_video_rows.start,
                    num_condition_audio_rows=record.generated_audio_rows.start)
                update(core, state, i, core.scheduler.timesteps[i])
            audio_rows = audio[record.generated_audio_rows]
            return _unpack(video[record.generated_video_rows], record.geometry), audio_rows.reshape(
                core.audio_channels, record.geometry['num_audio_latents'], -1).permute(0, 2, 1).contiguous()
        finally:
            for scheduler, saved in zip((core.scheduler, core.audio_scheduler), previous):
                scheduler.__dict__.clear()
                scheduler.__dict__.update(saved)
