#!/usr/bin/env python3
"""One real-backbone OPSD update; no dataset training or quality-gate calibration."""
# ruff: noqa: E402
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from diffusionopsd.video.cosmos_training import fit_queries
from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
from diffusionopsd.video.geo_reward import GeoReward, reward_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=['wan', 'cosmos'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from diffusionopsd.video.device_layout import InferenceReplica, encode_wan_prompt, validate_two_gpu_layout
    validate_two_gpu_layout('cuda:0', ['cuda:1'], visible_count=torch.cuda.device_count())
    torch.manual_seed(42)
    started = time.perf_counter()
    weights = Path(os.environ['VIDEO_REWARD_CKPT_PATH'])
    prompt = 'The camera slowly moves forward through a sunlit room with wooden furniture.'
    if args.backbone == 'wan':
        from config.wan22_ti2v import get_config
        from diffusers import WanPipeline
        from diffusionopsd.video.wan_clean_output import WanRollout, clean_output
        from diffusionopsd.video.wan_geometry import latent_shape_from_pipe, require_expand_timesteps
        from diffusionopsd.video.wan_policy import attach_lora
        cfg = get_config()
        model = weights / 'Wan2.2-TI2V-5B-Diffusers'
        pipe = WanPipeline.from_pretrained(model, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.text_encoder.to('cuda:1').requires_grad_(False)
        pipe.vae.to('cuda:1').requires_grad_(False)
        pipe.transformer.to('cuda:0').requires_grad_(False)
        require_expand_timesteps(pipe, str(model))
        pipe.transformer = attach_lora(pipe.transformer, cfg.train.lora_path)
        policy = pipe.transformer
        policy.enable_gradient_checkpointing()
        roll = WanRollout(pipe, cfg.sample.num_steps, cfg.sample.guidance_scale, offload_vae_activations=True)
        pe, ne = encode_wan_prompt(pipe, prompt, negative_prompt='', do_classifier_free_guidance=True, device='cuda:1')[:2]
        shape = latent_shape_from_pipe(pipe, 17, 256, 384)
        inference = InferenceReplica(policy, device='cuda:1')
        inference.select('old')
        pipe.transformer = inference.model
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            rec = roll.rollout(pe, ne, torch.randn(shape, device='cuda:1'), cfg.opa.query_sigma)
        y0 = clean_output(rec.z_q, rec.v_old_q, torch.tensor([rec.sigma_q], device='cuda:0')).float()

        def prediction(_):
            velocity = roll.velocity(rec.z_q, torch.tensor(rec.sigma_q, device='cuda:0'), pe, ne, transformer=policy)
            return clean_output(rec.z_q, velocity, torch.tensor([rec.sigma_q], device='cuda:0')).float()

        fitting_roll = SimpleNamespace(prediction=prediction)
        query = SimpleNamespace(y0=y0)
    else:
        from config.cosmos_nano import get_config
        from diffusers import Cosmos3OmniPipeline, UniPCMultistepScheduler
        from diffusionopsd.video.cosmos_rollout import CosmosRollout
        cfg = get_config()
        pipe = Cosmos3OmniPipeline.from_pretrained(
            weights / 'Cosmos3-Nano', torch_dtype=torch.bfloat16, local_files_only=True,
            sound_tokenizer=None, safety_checker=None, enable_safety_checker=False,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=cfg.flow_shift, use_karras_sigmas=False,
        )
        pipe.transformer.to('cuda:0').eval().requires_grad_(False)
        pipe.vae.to('cuda:1').eval().requires_grad_(False)
        policy = pipe.transformer
        lora = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_rank * 2, target_modules=list(cfg.lora_targets))
        policy.add_adapter(lora, adapter_name='default')
        policy.add_adapter(lora, adapter_name='old')
        set_peft_model_state_dict(policy, get_peft_model_state_dict(policy), adapter_name='old')
        policy.enable_gradient_checkpointing()
        roll = CosmosRollout(pipe, cfg.num_steps, cfg.guidance_scale, offload_vae_activations=True)
        inference = InferenceReplica(policy, device='cuda:1')
        inference.select('old')
        pipe.transformer = inference.model
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            query = roll.rollout(json.dumps({'description': prompt}),
                                 generator=torch.Generator('cuda:1').manual_seed(42),
                                 sigma_star=cfg.query_sigma, num_frames=17, height=256, width=384, fps=cfg.fps)
        fitting_roll = SimpleNamespace(prediction=lambda rec: roll.prediction(rec, transformer=policy))
    policy.set_adapter('default')
    policy.eval()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    for param in trainable:
        param.data = param.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    reward = GeoReward(load_depth_anything3('cuda:1'), load_waft('cuda:1'), load_dinov2('cuda:1'))

    def score(latents):
        clip = roll.decode01(latents)
        indices = torch.linspace(0, clip.shape[1] - 1, 9, device=clip.device).round().long()
        clip = clip[:, indices].to('cuda:1')
        b, t, c, h, w = clip.shape
        clip = F.interpolate(clip.flatten(0, 1), (224, 384), mode='bilinear', align_corners=False)
        return reward(clip.reshape(b, t, c, 224, 384)).geo.to(latents.device)

    before = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(policy).items()}
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    target_started = time.perf_counter()
    fit_cfg = SimpleNamespace(rho=.1, n_ascent=2, eta=1., beta=1., loss_scale=5., max_grad_norm=1.)

    def progress(stage, index):
        print(json.dumps({'stage': stage, 'backbone': args.backbone, 'seconds': time.perf_counter()-target_started}),
              flush=True)

    with torch.autograd.graph.save_on_cpu(), torch.autocast('cuda', dtype=torch.bfloat16):
        metrics = fit_queries(fitting_roll, [query], [.75], score, optimizer, fit_cfg, progress=progress)
    changed = any(not torch.equal(before[k], v.detach().cpu())
                  for k, v in get_peft_model_state_dict(policy).items())
    if not changed:
        raise RuntimeError('Optimizer did not change LoRA weights')
    for module in (pipe.vae, reward):
        if any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise RuntimeError('Frozen estimator or VAE received parameter gradients')
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
    result = dict(backbone=args.backbone, policy_changed=changed, reward=reward_metadata(),
                  frames=17, height=256, width=384, reward_frames=9, reward_height=224, reward_width=384,
                  fitting_seconds=time.perf_counter()-target_started, elapsed_seconds=time.perf_counter()-started,
                  peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30
                                      for i in range(torch.cuda.device_count())], **metrics)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
