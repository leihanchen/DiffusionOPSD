# DiffusionOPSD on Wan2.2-TI2V-5B

Date: 2026-09-22
Status: draft for review
Extends: `docs/superpowers/specs/2026-09-22-geometry-consistent-opsd-video-design.md`

## 1. Goal

The fork keeps every trainer it has now, and adds DiffusionOPSD post-training of **Wan2.2-TI2V-5B** (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`) on the same geometry reward, gates, and branch loss.

Wan2.2-TI2V-5B is a 5B dense text-and-image-to-video model. It is not a text-to-image model. This spec trains it as **text-to-video**: one prompt per line, noise latents, no source image. That is the official default when no image is passed. Image-to-video (a first-frame condition) is out of scope.

One launch trains one backbone. "At the same time" means the repository contains all of the trainers below, and a config file chooses which one runs.

| Trainer | Checkpoint | How this change treats it |
|---|---|---|
| `scripts/train_opsd_ri_sd3.py` | SD3.5-Medium | Untouched |
| `scripts/train_opsd_zimage.py` | Z-Image-Turbo | Untouched |
| `scripts/train_opsd_video_wan.py` with `config/wan_video.py` | Wan2.1-T2V-1.3B | Same full-parameter fp32 run as today |
| `scripts/train_opsd_video_wan.py` with `config/wan22_ti2v.py` | Wan2.2-TI2V-5B | New. Same script, LoRA policy |

## 2. Why one video stack

Three layouts were considered.

1. A copied trainer and a copied rollout for TI2V-5B. The OPSD loop would be free to drift away from Wan 2.1.
2. Replacing the Wan 2.1 default with TI2V-5B. That removes a model the branch already trains.
3. **Chosen.** One rollout and one video trainer. A small geometry helper reads the loaded VAE and transformer. The config file selects the checkpoint and whether the policy is full-finetune or LoRA.

SD3.5-M and Z-Image stay in their own scripts. Do not edit `scripts/train_opsd_ri_sd3.py`, `scripts/train_opsd_zimage.py`, or `src/rewards.py`.

## 3. What is identical to the Wan 2.1 video design

These stay as specified in the geometry design:

- Reward \(R_{geo}\), identity gate, motion floor, VideoReward quality mask, and \(\omega_{eff}\).
- Prompt files `data/video_motion/train.txt` and `data/video_motion/test.txt`.
- Pixel grid **17 × 480 × 832**. Both sizes are legal for both VAEs (see Sec. 4).
- Query \(\sigma^\* = 0.278\), trust region, branch coefficient \(\beta = 1\), group size \(K = 4\).
- Euler rollout and the clean-output map \(y = z - \sigma v\). Do not switch the loop to UniPC `scheduler.step`.
- Frozen Depth Anything 3, WAFT, DINOv2, and VideoReward. VideoReward still subsamples 8 frames.
- The fixed-suffix probe remains the go/no-go before a long run.

## 4. Latent geometry

The trainer stops hardcoding `// 4` and `// 8`. The shape comes from the loaded modules.

| | Wan2.1-T2V-1.3B | Wan2.2-TI2V-5B |
|---|---|---|
| VAE temporal scale | 4 (class default; absent from the 2.1 config) | 4 |
| VAE spatial scale | 8 (class default) | 16 |
| Latent channels | 16 | 48 |
| Patch size | `[1, 2, 2]` | `[1, 2, 2]` |
| Pixel rule | frames \(= 4k+1\); height and width multiples of 16 | frames \(= 4k+1\); height and width multiples of 32 |
| Latent at 17×480×832 | `[1, 16, 5, 60, 104]` | `[1, 48, 5, 30, 52]` |

`scale_factor_temporal` and `scale_factor_spatial` default to 4 and 8 when the VAE config omits them, which is the Wan 2.1 file. After a LoRA wrap, channel and patch sizes are read from the base transformer, not from the PEFT config.

An illegal frame count or an illegal height or width raises `ValueError` before noise is sampled. The official TI2V demo size (121 frames, 704×1280) satisfies the same rule and is not the training default. 121 frames at 720p would multiply the cost of Depth Anything 3 and WAFT; the training grid stays 17×480×832 so the reward stack matches Wan 2.1.

Batch size stays 1.

## 5. Timestep tensor

Wan 2.1 passes a length-`B` timestep, `sigma * 1000`, in the latent dtype. That path stays.

TI2V-5B sets `pipeline.config.expand_timesteps = True`. Text-only generation uses an all-ones mask, so every patch token gets the same timestep. The tensor shape is

\[
[B,\; T_{\mathrm{lat}} \cdot (H_{\mathrm{lat}}/2) \cdot (W_{\mathrm{lat}}/2)].
\]

At the training grid that is `[1, 1950]` (5 × 15 × 26). The value is still `sigma * 1000`. Missing `expand_timesteps` means false, so a Wan 2.1 pipeline is unchanged.

There is one transformer. `boundary_ratio` is null. No high-noise / low-noise expert switch.

## 6. Policy parameters

Wan 2.1 keeps today's full-parameter update: the transformer is cast to fp32, the behavior policy is a frozen deepcopy, and AdamW steps the fp32 weights. `config/wan_video.py` sets `use_lora = False` so the base default (`use_lora = True`) cannot silently change that run.

TI2V-5B uses the dual-adapter LoRA pattern already used by `scripts/train_opsd_ri_sd3.py`:

- Rank 32, alpha 64, `init_lora_weights="gaussian"`, dropout 0.
- Target modules whose names end with `to_q`, `to_k`, `to_v`, `to_out.0`, `add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out`.
- Adapter `"default"` is the trainable policy. Adapter `"old"` is the frozen behavior policy.
- The base transformer stays bf16 and frozen. LoRA weights stay fp32. AdamW sees only `requires_grad` parameters. This is the same reason Wan 2.1 uses an fp32 master: AdamW on bf16 weights underflows, without keeping a second fp32 copy of the 5B base.
- Rollouts call `set_adapter("old")`. The fitting step calls `set_adapter("default")`.
- After each optimizer step, the `"old"` adapter is an EMA of `"default"` with decay 0.99, matching the in-place behavior update on the Wan 2.1 path.
- Checkpoints store `get_peft_model_state_dict` for `"default"` only.

`config.train.lora_path` empty means a fresh adapter. A path there loads that adapter into `"default"` and still adds a fresh `"old"` adapter, then the EMA takes over.

## 7. Scripts

`scripts/train_opsd_video_wan.py`, `scripts/eval_video_consistency.py`, and `scripts/probe_fixed_suffix_video.py` all call the shared latent helper. Eval and probe read `expand_timesteps` through `WanRollout`, so they do not grow a second timestep implementation.

The probe does not wrap LoRA. It scores the base TI2V transformer. It leaves that transformer in bf16 under the existing autocast. The fp32 cast stays on the Wan 2.1 probe, where the module is the 1.3B model. The probe still restores parameters between clips.

Eval of a TI2V run loads the base pipeline in bf16, applies the saved `"default"` adapter, and generates with that adapter active. Eval of a Wan 2.1 run keeps the full state-dict load it has now.

`scripts/download_video_reward_weights.sh` gains one line:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir "$TARGET/Wan2.2-TI2V-5B-Diffusers"
```

The checkpoint is not in the OpenResearch demo bundle. A training, probe, or eval run needs that download. The geometry reward weights are unchanged.

`diffusers>=0.36` stays the pin. The TI2V loader requires `WanPipeline.config.expand_timesteps` on this checkpoint. A build that loads the repo and leaves the flag missing fails in the latent/timestep unit of the setup, with a message that names the flag.

## 8. Out of scope

- Image-to-video training, first-frame datasets, and a zero-timestep mask on the first latent frame.
- Wan2.2-T2V-A14B and its high-noise / low-noise pair.
- Changing the geometry reward, the VideoReward mask, the prompt files, or the 17×480×832 training grid.
- Editing the SD3.5-M or Z-Image trainers, `src/rewards.py`, or the image reward stack.
- Downloading weights as part of writing or unit-testing this design.

## 9. Success criteria

Unit tests, with no checkpoint and no GPU:

- Wan 2.1 numbers: 17×480×832, scales 4 and 8, 16 channels → `[1, 16, 5, 60, 104]`.
- TI2V numbers: scales 4 and 16, 48 channels → `[1, 48, 5, 30, 52]`.
- A frame count that is not \(4k+1\), or a side that is not a multiple of `spatial_scale * 2`, raises `ValueError`.
- `expand_timesteps=False` returns a length-`B` timestep. `True` on a `[1, 48, 5, 30, 52]` latent returns `[1, 1950]`, filled with `sigma * 1000`.
- `config/wan_video.py` has `use_lora is False` and the Wan 2.1 path. `config/wan22_ti2v.py` has `use_lora is True`, the TI2V path, and the same prompt file and pixel grid.

A later GPU run, after the weights are downloaded, is the fixed-suffix probe on TI2V-5B with the same pass rule as the geometry spec. That run is not part of the unit-test bar.
