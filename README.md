*Note: This is an external implementation of the algorithm in the following paper:*


<div align="center">

# ✨ DiffusionOPSD: On-Policy Self-Distillation in Diffusion Models ✨

**Reward-guided diffusion post-training through explicit, continually refreshed intermediate targets**

<p align="center">
  Wei Zhou<sup>1,2</sup>, Xiongwei Zhu<sup>1</sup>, Lingdong Kong<sup>2</sup>, Bo Chen<sup>1</sup>, Lei Zhang<sup>3</sup>, Yongyuan Liang<sup>4</sup>, Xiaoxia Hou<sup>1</sup>, Tian Ye<sup>5</sup>,<br>
  Xian Sun<sup>6</sup>, Yingshuo Wang<sup>7</sup>, Linfeng Li<sup>2</sup>, Shengqiong Wu<sup>8</sup>, Leigang Qu<sup>2</sup>, Feng Li<sup>9</sup>,<br> Wei Liu<sup>1,†</sup>, Julian McAuley<sup>3</sup>, Tat-Seng Chua<sup>2</sup>
</p>
<p align="center">
  <a href="https://seed.bytedance.com/"><img src="https://www.google.com/s2/favicons?domain=seed.bytedance.com&sz=64" width="16" height="16" alt="ByteDance Seed icon"> <sup>1</sup> ByteDance Seed</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://www.nus.edu.sg/"><img src="https://www.google.com/s2/favicons?domain=nus.edu.sg&sz=64" width="16" height="16" alt="NUS icon"> <sup>2</sup> NUS</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://ucsd.edu/"><img src="https://www.google.com/s2/favicons?domain=ucsd.edu&sz=64" width="16" height="16" alt="UC San Diego icon"> <sup>3</sup> UC San Diego</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://umd.edu/"><img src="https://www.google.com/s2/favicons?domain=umd.edu&sz=64" width="16" height="16" alt="UMD icon"> <sup>4</sup> UMD</a><br>
  <a href="https://www.hkust-gz.edu.cn/"><img src="https://www.google.com/s2/favicons?domain=hkust-gz.edu.cn&sz=64" width="16" height="16" alt="HKUST Guangzhou icon"> <sup>5</sup> HKUST (Guangzhou)</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://www.duke.edu/"><img src="https://www.google.com/s2/favicons?domain=duke.edu&sz=64" width="16" height="16" alt="Duke icon"> <sup>6</sup> Duke</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://www.berkeley.edu/"><img src="https://www.google.com/s2/favicons?domain=berkeley.edu&sz=64" width="16" height="16" alt="UC Berkeley icon"> <sup>7</sup> UC Berkeley</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://www.ox.ac.uk/"><img src="https://www.google.com/s2/favicons?domain=ox.ac.uk&sz=64" width="16" height="16" alt="Oxford icon"> <sup>8</sup> Oxford</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://hkust.edu.hk/"><img src="https://www.google.com/s2/favicons?domain=hkust.edu.hk&sz=64" width="16" height="16" alt="HKUST icon"> <sup>9</sup> HKUST</a>
</p>
<p align="center"><sub><sup>†</sup> Corresponding Author</sub></p>
<p align="center">
  <a href="https://arxiv.org/abs/2608.24646">
    <img src="https://img.shields.io/badge/arXiv-2608.24646-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper">
  </a>
  <a href="https://diffusionopsd.github.io/">
    <img src="https://img.shields.io/badge/Project-Page-3B82F6?style=for-the-badge&logo=githubpages&logoColor=white" alt="Project Page">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-16A34A?style=for-the-badge&logo=apache&logoColor=white" alt="Apache 2.0 License">
  </a>
  <a href="https://diffusionopsd.github.io/blog/v2/">
    <img src="https://img.shields.io/badge/V2-Blog-B0448C?style=for-the-badge&logo=rss&logoColor=white" alt="V2 Blog">
  </a>
</p>

<br>

<a href="https://diffusionopsd.github.io/blog/v2/"><img src="assets/opsd_v2.png" width="100%" alt="DiffusionOPSD v2 training overview"></a>

</div>

🔥 DiffusionOPSD turns image-level reward guidance into explicit targets for clean-output predictions at sampled queries. In v2, we keep that principle and make the training recipe more efficient: **one reward-ascent step for target construction, broader prompt coverage, and rollouts that stop at the query.** The controlled experiment preserves held-out performance while reducing training GPU time by approximately **30.5%**. A separate measurement of one complete update with the old and new recipes reduces update time by **39.7%**.

<div align="center">

<img src="assets/qualitative_gallery.png" width="100%" alt="DiffusionOPSD qualitative gallery">

</div>

---

## 📌 Abstract

Diffusion reward optimization observes an image-level score only after a multi-step denoising trajectory, leaving a supervision gap at intermediate predictions. **DiffusionOPSD** closes this gap with an on-policy self-distillation loop. A frozen behavior policy collects low-noise query states and clean-output anchors; differentiable reward gradients construct bounded positive and negative targets around each anchor; the trainable policy then fits these detached targets under a finite update budget. An EMA refreshes the behavior policy before the next round of trajectories and targets.

This separation makes **target construction** and **finite realization** independently observable. Across SD3.5-M and Z-Image-Turbo, DiffusionOPSD achieves the best final held-out score in **19 of 20** reward-matched settings and reduces training GPU-hours relative to DiffusionNFT by **40%** and **63%**, respectively.

---

## 🌟 Highlights

- **On-policy query collection.** Supervision is built on states visited by the current behavior policy rather than offline or forward-noised substitutes.
- **Explicit reward-guided targets.** Normalized reward ascent and descent construct bounded positive and negative clean-output targets.
- **Detached finite fitting.** Reward/decoder graphs are discarded before policy fitting, separating target quality from model realization.
- **Continually refreshed supervision.** An EMA behavior policy regenerates trajectories, anchors, and targets after each outer update.
- **Two distinct backbones.** The release supports SD3.5-M at 512² and the native few-step Z-Image-Turbo regime at 1024².
- **Single- and mixed-reward training.** Public presets cover all seven open-weight evaluators and arbitrary positive weighted sums; the paper example uses PickScore/26 + CLIPScore + HPSv2.1.

<img src="assets/training_curves.png" width="92%" alt="Training and held-out quality curves">

---

## 🧠 Method

At query $s=(c,z_q,\sigma_q)$, the frozen behavior policy defines a clean-output anchor

$$
y_0 = z_q - \sigma_q v_{\mathrm{old}}(z_q,c,\sigma_q).
$$

DiffusionOPSD applies normalized reward-gradient steps inside a relative trust region,

$$
y_+ \leftarrow y_+ + h\frac{\nabla_y \widetilde R(y_+,c)}{\|\nabla_y \widetilde R(y_+,c)\|_2+\epsilon},
\qquad
y_- \leftarrow y_- - h\frac{\nabla_y \widetilde R(y_-,c)}{\|\nabla_y \widetilde R(y_-,c)\|_2+\epsilon},
$$

then fits the detached targets through positive and negative branches:

$$
\mathcal L_{\mathrm{OPSD}}
= \omega\frac{\|y_\theta^+-\bar y_+\|_2^2}{\gamma_+}
+ (1-\omega)\frac{\|y_\theta^- -\bar y_-\|_2^2}{\gamma_-}.
$$

The implementation uses detached mean-absolute residual normalizers, elementwise-mean squared residuals, and optimizes $c_{\mathrm{adv}}\mathcal L_{\mathrm{OPSD}}$ with $c_{\mathrm{adv}}=5$.

<p align="center">
  <a href="https://diffusionopsd.github.io/"><img src="assets/method_overview.gif" width="100%" alt="Animated DiffusionOPSD method overview"></a>
</p>

Minimal pseudocode:

```python
trajectory = rollout(behavior_policy, prompt)
query = select_low_noise_state(trajectory)
anchor = behavior_policy.clean_output(query).detach()
weight = group_normalized_endpoint_weight(trajectory.reward)

positive = bounded_reward_ascent(anchor, reward).detach()
negative = bounded_reward_descent(anchor, reward).detach()

prediction = trainable_policy.clean_output(query.detach())
loss = detached_target_loss(prediction, anchor, positive, negative, weight)
loss.backward()
optimizer.step()
update_behavior_policy_ema()
```

---

## 📊 Main Results

Reward-specific checkpoints are evaluated on their matched held-out objective. Higher is better.

### SD3.5-M

| Method | Pick | CLIP | HPSv2.1 | Aes | ImgR | HPSv3 | DeQA | AltCLIP | Point | Pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ReFL | <u>23.92</u> | 0.308 | <u>0.358</u> | **12.09** | 1.28 | 6.18 | <u>4.85</u> | 0.408 | 0.193 | 0.290 |
| DiffusionNFT | 23.43 | 0.298 | 0.336 | 9.11 | 1.46 | 13.51 | 4.76 | 0.412 | 0.199 | 0.323 |
| **DiffusionOPSD** | **24.94** | **0.340** | **0.390** | <u>12.08</u> | **1.76** | **15.14** | **4.94** | <u>0.450</u> | <u>0.214</u> | **0.465** |
| **DiffusionOPSD\*** | **24.94** | <u>0.327</u> | **0.390** | **12.09** | <u>1.73</u> | <u>14.79</u> | 4.81 | **0.458** | **0.216** | <u>0.463</u> |

### Z-Image-Turbo

| Method | Pick | CLIP | HPSv2.1 | Aes | ImgR | HPSv3 | DeQA | AltCLIP | Point | Pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FlowGRPO | 22.96 | 0.275 | 0.305 | 5.46 | 1.01 | 11.58 | 4.51 | 0.394 | 0.217 | 0.420 |
| ReFL | 24.54 | 0.313 | 0.380 | 9.79 | 1.37 | 11.32 | 4.60 | 0.441 | 0.227 | 0.481 |
| DiffusionNFT | 22.28 | 0.280 | 0.277 | 6.07 | 0.58 | 11.01 | 3.37 | 0.363 | 0.166 | 0.357 |
| **DiffusionOPSD** | <u>25.15</u> | **0.320** | **0.390** | <u>10.74</u> | **1.79** | **15.59** | **4.78** | <u>0.451</u> | <u>0.243</u> | **0.551** |
| **DiffusionOPSD\*** | **25.28** | <u>0.314</u> | <u>0.386</u> | **10.80** | <u>1.76</u> | <u>15.51</u> | <u>4.77</u> | **0.459** | **0.249** | <u>0.549</u> |

### Training efficiency

Eight-GPU profiles; initialization, calibration, and evaluation are excluded.

| Backbone | Method | Seconds / update | Peak VRAM | GPU-h / 100 updates | Relative to NFT |
|---|---|---:|---:|---:|---:|
| SD3.5-M | DiffusionNFT | 212.4 | 47.8 GB | 47.2 | 1.00× |
| SD3.5-M | **DiffusionOPSD** | **126.9** | 50.0 GB | **28.2** | **0.60×** |
| Z-Image-Turbo | DiffusionNFT | 1826.2 | 49.9 GB | 405.8 | 1.00× |
| Z-Image-Turbo | **DiffusionOPSD** | **674.0** | 61.5 GB | **149.8** | **0.37×** |

<p align="center">
  <img src="assets/robustness.png" width="100%" alt="Shared-reference gain, terminal stability, and multi-reward composite">
</p>

<table>
  <tr>
    <td width="50%" valign="top"><img src="assets/native_training_rewards.png" width="100%" alt="Native training rewards across two backbones"></td>
    <td width="50%" valign="top"><img src="assets/heldout_quality_compute.png" width="100%" alt="Held-out quality versus cumulative GPU-hours"></td>
  </tr>
  <tr>
    <td align="center"><b>Native training rewards across ten objectives and two backbones.</b></td>
    <td align="center"><b>Held-out quality versus cumulative GPU-hours.</b></td>
  </tr>
</table>

---

## 🔬 Ablation Studies

The paper isolates target direction, implementation sensitivity, and train/evaluation CFG dependence. Reward-gradient targets outperform random, no-op, and rollout-residual controls, while the canonical settings remain stable across the tested implementation variants.

<p align="center">
  <img src="assets/ablation_controls.png" width="100%" alt="DiffusionOPSD target, implementation, and CFG ablations">
</p>

---

## ⚙️ Installation

A recent Linux environment with CUDA GPUs is required for training. Python 3.10–3.11 is recommended.

```bash
git clone https://github.com/worldbench/DiffusionOPSD.git
cd DiffusionOPSD

conda create -n diffusionopsd python=3.11 -y
conda activate diffusionopsd

# Install the CUDA-matched PyTorch build first, then the standard reward stack.
pip install -e ".[rewards]"
# ImageReward's package metadata pins an obsolete timm; its inference code is
# compatible with the validated stack, so install the package without deps.
pip install --no-deps 'image-reward==1.5'

# Download the HPSv2.1 and Aesthetic checkpoint files used by the public presets.
export REWARD_CKPT_PATH="$PWD/reward_ckpts"
bash scripts/download_reward_weights.sh

# Materialize the exact 25,415-prompt Pick-a-Pic paper manifest.
# The training launchers also do this automatically.
python scripts/prepare_pickapic_prompts.py
```

Z-Image requires a Diffusers build containing `ZImagePipeline`. If it is absent from the installed release, install Diffusers from its official source checkout:

```bash
git clone https://github.com/huggingface/diffusers.git
pip install -e "./diffusers[torch]"
```

Optional environment variables:

```bash
export HF_HOME=/path/to/huggingface-cache
export WANDB_MODE=offline                    # default in train_public.sh
export REWARD_CKPT_PATH=/path/to/reward_ckpts
```

SD3.5-M is gated on Hugging Face; accept its model license and authenticate before the first download.

### Reward-model setup

DiffusionOPSD exposes all seven public evaluators used in the paper. Their model weights are resolved from the upstream projects and are not redistributed here.

#### Standard rewards

HPSv2.1, CLIPScore, PickScore, Aesthetic, and ImageReward share the standard environment:

```bash
pip install -e ".[rewards]"
# Keep the validated timm>=1.0 stack. ImageReward's runtime is compatible;
# only its old package metadata still pins timm==0.6.13.
pip install --no-deps 'image-reward==1.5'
export REWARD_CKPT_PATH="$PWD/reward_ckpts"
bash scripts/download_reward_weights.sh
```

CLIPScore, PickScore, and ImageReward download their remaining Hugging Face weights on first use. Because ImageReward declares its obsolete `timm==0.6.13` pin in package metadata, `pip check` reports that one intentional mismatch; `scripts/smoke_reward_gradient.py` validates the differentiable runtime path used by this release.

#### Heavy rewards: HPSv3 and DeQA

[HPSv3](https://github.com/MizzenAI/HPSv3) and [DeQA](https://github.com/zhiyuanyou/DeQA-Score) use 7B/8B evaluator stacks. Use a separate heavy-reward environment. Do not install HPSv3 with dependency resolution: its package metadata pins an older Transformers release, while the joint SD3.5/Z-Image path uses the validated Transformers 4.51 stack.

```bash
# In a fresh Python 3.10/3.11 environment, install a CUDA-matched PyTorch
# build first, then install the project without the standard reward extra.
pip install -e .
pip install --no-deps 'hpsv3==1.0.0'
pip install 'qwen-vl-utils>=0.0.8' omegaconf safetensors \
  einops 'timm>=1.0' fire 'trl==0.15.2' matplotlib tensorboard requests \
  packaging pyyaml sentencepiece icecream

python scripts/check_reward_setup.py --backbone sd35 --reward hpsv3
python scripts/check_reward_setup.py --backbone sd35 --reward deqa
python scripts/smoke_reward_gradient.py --reward hpsv3
python scripts/smoke_reward_gradient.py --reward deqa
```

The no-dependency HPSv3 install intentionally bypasses its training-oriented `transformers==4.45.2` and Deepspeed metadata. This release uses neither Deepspeed nor the HPSv3 training entry point; it validates reward inference and image gradients against Transformers 4.51.

Optional FlashAttention optimization, matched to the installed CUDA/PyTorch build:

```bash
pip install 'flash-attn==2.7.4.post1' --no-build-isolation
```

Weight resolution:

- HPSv3 downloads the pinned `MizzenAI/HPSv3` reward checkpoint and the Qwen2-VL-7B base model. Set `HPSV3_CHECKPOINT=/path/to/HPSv3.safetensors` for an offline checkpoint, `HPSV3_CONFIG=/path/to/HPSv3_7B.yaml` for an offline base-model path, or `HPSV3_REVISION` to test a different upstream revision.
- DeQA downloads a pinned revision of [`zhiyuanyou/DeQA-Score-Mix3`](https://huggingface.co/zhiyuanyou/DeQA-Score-Mix3). Set `DEQA_MODEL_PATH=/path/to/DeQA-Score-Mix3` for an offline copy, or `DEQA_MODEL_REVISION` to test another revision.

#### Heavy-reward launch topology

All seven SD3.5-M rewards use eight colocated policy ranks. The DeQA differentiable target micro-batch is one; other presets retain their paper settings:

```bash
bash scripts/train_public.sh sd35 hpsv3
bash scripts/train_public.sh sd35 deqa
```

For Z-Image-Turbo, HPSv3 and DeQA use six policy ranks plus one differentiable reward-server rank (`NPROC=7`). The server returns both scalar rewards and image gradients:

```bash
bash scripts/train_public.sh zimage hpsv3
bash scripts/train_public.sh zimage deqa
```

The Z-Image DiffusionNFT and FlowGRPO launchers use the same six-policy-plus-one-server layout for HPSv3 and DeQA, but only request scalar rewards:

```bash
bash scripts/train_baseline.sh nft zimage hpsv3
bash scripts/train_baseline.sh flowgrpo zimage deqa
```

Custom DiffusionOPSD launchers must set the same topology:

```bash
export ZIMAGE_HEAVY_DIFF_BRIDGE=1
export PUBLIC_POLICY_WORLD_SIZE=6
export PUBLIC_LAUNCH_WORLD_SIZE=7
python -m torch.distributed.run --standalone --nproc_per_node=7 \
  scripts/train_opsd_zimage.py --config config/public.py:zimage_hpsv3
```

Run `scripts/check_reward_setup.py` before a long job. It checks interfaces, required local assets, Transformers compatibility, and policy/server topology without loading multi-gigabyte weights. Run `scripts/smoke_reward_gradient.py` once per reward to load the actual checkpoint and require a finite, nonzero image-space gradient.

---

## Public Runnable Matrix

The public presets use downloadable base models and reward models.

| Backbone | Public model | Preset | Public rewards | Default regime |
|---|---|---|---|---|
| SD3.5-M | `stabilityai/stable-diffusion-3.5-medium` | `config/public.py:sd35_<reward>` | HPSv2.1, CLIPScore, PickScore, Aesthetic, ImageReward, HPSv3, DeQA | 512², 10-step CFG-free rollout; 8 policy ranks |
| SD3.5-M | same | `config/public.py:sd35_open3` | PickScore/26 + CLIPScore + HPSv2.1 | joint Open3 objective |
| SD3.5-M | same | `config/mixed.py:sd35_<opsd\|nft>` | any positive weighted sum of two or more public rewards | configurable mixed-reward objective |
| Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` | `config/public.py:zimage_<reward>` | same seven evaluators | 1024², native few-step rollout; HPSv3/DeQA use 6 policy + 1 reward-server rank |

Paper-matched baseline presets:

| Method | Backbone | Preset | Public rewards | Budget |
|---|---|---|---|---|
| DiffusionNFT | SD3.5-M | `config/nft.py:sd3_<reward>` | all seven | 100 optimizer updates |
| FlowGRPO | SD3.5-M | `config/flowgrpo.py:sd35_clipscore` | CLIPScore control used for dynamics/efficiency analysis | 50 rollout rounds = 100 optimizer updates |
| DiffusionNFT | Z-Image-Turbo | `config/zimage.py:zimg_nft_<reward>` | all seven | 100 optimizer updates |
| FlowGRPO | Z-Image-Turbo | `config/zimage.py:zimg_flowgrpo_<reward>` | all seven | 50 rollout rounds = 100 optimizer updates |
| ReFL | both | `config/refl.py:<backbone>_<reward>` | all seven | 100 optimizer updates |

### Short end-to-end check

This is a real model/reward/gradient run and downloads the selected weights on first use. `SMOKE_TEST=1` keeps the paper's trajectories-per-prompt grouping but uses one group-complete rollout batch instead of the full 48-group update; FlowGRPO uses two batches to exercise both PPO optimizer windows.

```bash
SMOKE_TEST=1 NPROC=8 UPDATES=1 bash scripts/train_public.sh sd35 hpsv2
SMOKE_TEST=1 NPROC=8 UPDATES=1 bash scripts/train_public.sh zimage hpsv2

# Heavy reward path: NPROC=7 is selected automatically for Z-Image.
SMOKE_TEST=1 UPDATES=1 bash scripts/train_public.sh zimage hpsv3
SMOKE_TEST=1 UPDATES=1 bash scripts/train_public.sh zimage deqa

# Baselines. FlowGRPO performs two optimizer updates in one smoke rollout.
SMOKE_TEST=1 UPDATES=1 bash scripts/train_baseline.sh nft zimage hpsv2
SMOKE_TEST=1 UPDATES=2 bash scripts/train_baseline.sh flowgrpo sd35 clipscore
SMOKE_TEST=1 UPDATES=2 bash scripts/train_baseline.sh flowgrpo zimage hpsv2
```

Before loading a diffusion backbone, verify each evaluator's exact target-construction path with one image:

```bash
python scripts/smoke_reward_gradient.py --reward hpsv2
# Repeat for: clipscore, pickscore, aesthetic, imagereward, hpsv3, deqa
```

### Full reward-specific training

The training scripts enable the accelerated DiffusionOPSD v2 recipe by default.

```bash
NPROC=8 UPDATES=100 OUTPUT_DIR=outputs/sd35_hpsv2 \
  bash scripts/train_public.sh sd35 hpsv2

NPROC=8 UPDATES=100 OUTPUT_DIR=outputs/zimage_hpsv2 \
  bash scripts/train_public.sh zimage hpsv2
```

### Baseline training

```bash
# The matched SD3.5-M FlowGRPO CLIPScore control.
NPROC=8 UPDATES=100 bash scripts/train_baseline.sh flowgrpo sd35 clipscore

# Reward-specific Z-Image DiffusionNFT and FlowGRPO runs.
for reward in hpsv2 clipscore pickscore aesthetic imagereward hpsv3 deqa; do
  UPDATES=100 bash scripts/train_baseline.sh nft zimage "$reward"
  UPDATES=100 bash scripts/train_baseline.sh flowgrpo zimage "$reward"
done
```

Public trainers save every 10 optimizer updates and at the final update by default. FlowGRPO uses the optimizer-step counter rather than rollout-round count, and smoke runs disable checkpoint writing automatically. Set `--config.save_freq=1` only when every intermediate curve point is required.

### Mixed-reward training

The mixed-reward launcher accepts any number of distinct public rewards and an optional positive weight for each one. Both DiffusionOPSD and DiffusionNFT use exactly the same weighted endpoint objective; DiffusionOPSD also uses the same weighted sum for its differentiable target-construction gradient.

The script defaults to the paper's three-reward objective. PickScore is normalized by `/26` inside its scorer, so unit weights implement `PickScore/26 + CLIPScore + HPSv2.1`:

```bash
NPROC=8 UPDATES=300 OUTPUT_DIR=outputs/mixed_opsd \
  bash scripts/train_mixed_reward.sh opsd

NPROC=8 UPDATES=300 OUTPUT_DIR=outputs/mixed_nft \
  bash scripts/train_mixed_reward.sh nft
```

To select any other combination, pass `MIXED_REWARDS` as `reward[=weight],...`:

```bash
MIXED_REWARDS='clipscore=1,hpsv2=0.5,aesthetic=0.1,imagereward=0.25' \
NPROC=8 UPDATES=300 OUTPUT_DIR=outputs/mixed_custom \
  bash scripts/train_mixed_reward.sh opsd
```

Supported names are `hpsv2`, `clipscore`, `pickscore`, `aesthetic`, `imagereward`, `hpsv3`, and `deqa`. Terms without `=weight` default to `1`. Weights act on each evaluator's native output scale; there is no automatic cross-reward normalization beyond the scorer's paper-matched PickScore `/26`. Duplicate rewards, non-positive weights, unknown names, and fewer than two rewards fail before `torchrun`. Combinations containing HPSv3 or DeQA retain their heavier environment and memory requirements. The fixed `bash scripts/train_public.sh sd35 open3` command remains a backward-compatible alias for the paper objective.

### Use your own prompt split

Expected format: one prompt per line in `train.txt` and `test.txt`.

```text
/path/to/prompts/
├── train.txt
└── test.txt
```

```bash
NPROC=8 UPDATES=100 bash scripts/train_public.sh sd35 clipscore \
  --config.dataset=/path/to/prompts
```

The default is already the paper split: a pinned text-only Hugging Face dataset plus the bundled reconstruction recipe creates the exact 25,415-prompt manifest and verifies SHA-256 `39d94f…aa7100c`.

---

## 🤗 Released Checkpoints

Three LoRA checkpoints are available from [`WeiChow/DiffusionOPSD`](https://huggingface.co/WeiChow/DiffusionOPSD):

| Checkpoint | Backbone | Training reward |
|---|---|---|
| `sd35-m-hpsv3` | SD3.5-M | HPSv3 |
| `z-image-turbo-hpsv3` | Z-Image-Turbo | HPSv3 |
| `z-image-turbo-pointwise` | Z-Image-Turbo | DiffusionOPSD checkpoint |

Download them into the relative path used below:

```bash
hf download WeiChow/DiffusionOPSD --local-dir checkpoints/diffusionopsd
```

Generate the complete paper held-out set and run the matched HPSv3 benchmark for SD3.5-M. The generation batch size is part of the fixed seed schedule, so keep it at `16` when reproducing the reported protocol:

```bash
python scripts/cross_eval.py \
  --ckpt checkpoints/diffusionopsd/sd35-m-hpsv3 \
  --config_file config/public.py --config sd35_hpsv3 \
  --prompts data/drawbench/test.txt \
  --prompt_set_name drawbench --protocol_name table1_drawbench_flow40 \
  --sampler flow --num_steps 40 --guidance_scale 1.0 \
  --seed 42 --batch_size 16 --score_batch_size 1 \
  --mixed_precision fp16 --rewards hpsv3 \
  --images_dir outputs/sd35-m-hpsv3/images \
  --out outputs/sd35-m-hpsv3/result.json
```

Use the complete native 9-step, 1024² Z-Image held-out protocol for either Z-Image adapter. Keep generation batches at `8` to preserve the reported per-batch seeds:

```bash
python scripts/native_eval.py \
  --pipeline zimage --model Tongyi-MAI/Z-Image-Turbo \
  --lora checkpoints/diffusionopsd/z-image-turbo-hpsv3 \
  --prompts data/drawbench/test.txt \
  --prompt_set_name drawbench --protocol_name native_zimage_drawbench \
  --resolution 1024 --num_steps 9 --guidance_scale 0.0 \
  --seed 42 --dtype bf16 --batch_size 8 --score_batch_size 8 \
  --rewards hpsv3 \
  --images_dir outputs/z-image-turbo-hpsv3/images \
  --out outputs/z-image-turbo-hpsv3/result.json
```

For a short but seed-exact generation probe, append `--n_prompts 16` to the SD3.5-M command or `--n_prompts 8` to the Z-Image command; changing `--batch_size` changes the canonical seed grouping.

Replace the LoRA path with `z-image-turbo-pointwise` to test its image-generation path. The corresponding paper evaluator is not included in this repository.

---

## 🚀 Direct Launch

The helper above is equivalent to:

```bash
export PUBLIC_POLICY_WORLD_SIZE=8
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_opsd_ri_sd3.py \
  --config config/public.py:sd35_hpsv2 \
  --config.num_epochs=100 \
  --config.save_dir=outputs/sd35_hpsv2
```

Resume from a saved checkpoint:

```bash
NPROC=8 UPDATES=100 bash scripts/train_public.sh sd35 hpsv2 \
  --config.resume_from=outputs/sd35_hpsv2/checkpoints/checkpoint-50
```

---

## 📏 Evaluation

The SD3.5 evaluator regenerates a fixed prompt set once, then scores the same images with the selected reward models:

```bash
python scripts/cross_eval.py \
  --ckpt outputs/sd35_hpsv2/checkpoints/checkpoint-100/lora \
  --config sd35_hpsv2 \
  --config_file config/public.py \
  --prompts data/drawbench/test.txt \
  --prompt_set_name drawbench \
  --protocol_name drawbench_flow40_cfgfree \
  --num_steps 40 \
  --guidance_scale 1.0 \
  --rewards hpsv2,clipscore,pickscore,aesthetic \
  --pickscore_scale raw \
  --out outputs/sd35_hpsv2/drawbench.json
```

Use `scripts/native_eval.py` for native Z-Image generation and `scripts/eval_diversity.py` for matched-prompt reward/diversity analysis.

---

## 🧩 Supported Backbones

| Backbone | Trainer | Sampler | Student update |
|---|---|---|---|
| SD3.5-M | `scripts/train_opsd_ri_sd3.py` | deterministic DPM2 rollout; flow-40 evaluation | PEFT LoRA |
| Z-Image-Turbo | `scripts/train_opsd_zimage.py` | native FlowMatchEuler few-step rollout | PEFT LoRA |

Canonical DiffusionOPSD defaults live in `config/opsd_defaults.py`:

| Setting | SD3.5-M | Z-Image-Turbo |
|---|---:|---:|
| query noise `σ*` | 0.278 | 0.273 |
| target radius `ρ` | 0.10 | 0.10 |
| target steps | 2 | 2 |
| branch coefficient `β` | 1.0 (0.1 for Open3) | 1.0 |
| positive / negative targets | both | both |
| certification | off | off |
| policy-reference MSE | off | off |

---

## 📁 Repository Layout

```text
.
├── assets/                  # README figures and qualitative gallery
├── config/
│   ├── public.py            # public runnable presets
│   ├── mixed.py             # arbitrary weighted mixed-reward presets
│   ├── flowgrpo.py          # matched SD3.5-M FlowGRPO control
│   ├── refl.py              # public ReFL presets
│   ├── opsd_defaults.py     # canonical method defaults
│   ├── nft.py               # SD3.5 base configuration
│   └── zimage.py            # Z-Image base configuration
├── data/
│   ├── pickapic_recipe.json # exact paper-split reconstruction recipe
│   └── drawbench/           # fixed 1,000-prompt held-out manifest
├── src/
│   ├── diffusers_patch/     # samplers, log-prob utilities, Z-Image rollout
│   ├── rewards.py           # reward registry
│   ├── metrics.py           # shared WandB → metrics.jsonl instrumentation
│   ├── experiment_io.py     # checkpoint/resume/provenance helpers
│   └── *_scorer.py          # differentiable reward adapters
├── scripts/
│   ├── train_public.sh      # public launcher
│   ├── train_mixed_reward.sh # mixed DiffusionOPSD / DiffusionNFT launcher
│   ├── train_baseline.sh    # DiffusionNFT / FlowGRPO launcher
│   ├── train_refl.sh        # public ReFL launcher
│   ├── prepare_pickapic_prompts.py
│   ├── download_reward_weights.sh
│   ├── train_opsd_ri_sd3.py
│   ├── train_opsd_zimage.py
│   ├── cross_eval.py
│   └── native_eval.py
├── opd/                     # DanceOPD, DiffusionOPD, FlowOPD
├── LICENSE
└── pyproject.toml
```

## 🧪 Baselines

The launcher above covers DiffusionNFT and FlowGRPO. ReFL and the teacher-distillation baselines use their dedicated launchers:

```bash
# ReFL: 48 prompt groups × K trajectories/update (K=24 SD3.5, K=12 Z-Image).
bash scripts/train_refl.sh sd35 hpsv2
bash scripts/train_refl.sh zimage hpsv2

# OPD: train the three 100-update specialists, then run a short full-flow pilot.
for reward in pickscore clipscore hpsv2; do
  bash scripts/train_public.sh sd35 "$reward"
done
METHODS="danceopd diffusionopd flowopd" bash opd/benchmark_opd.sh
METHOD=danceopd bash opd/launch_opd.sh
```

See [`opd/README.md`](opd/README.md) for DanceOPD, DiffusionOPD, and FlowOPD. Their second stage uses the same three frozen specialists, 300 updates, and the paper's calibrated 2,688/528/544 samples per update.

---

## 📝 Citation

```bibtex
@article{zhou2026diffusionopsd,
  title   = {On-Policy Self-Distillation in Diffusion Models},
  author  = {Zhou, Wei and Zhu, Xiongwei and Kong, Lingdong and Chen, Bo and Zhang, Lei and Liang, Yongyuan and Hou, Xiaoxia and Tian, Ye and Sun, Xian and Wang, Yingshuo and Li, Linfeng and Wu, Shengqiong and Qu, Leigang and Li, Feng and Liu, Wei and McAuley, Julian and Chua, Tat-Seng},
  year    = {2026}
}
```

---

## 🙏 Acknowledgements

This release builds on Hugging Face [Diffusers](https://github.com/huggingface/diffusers) and [PEFT](https://github.com/huggingface/peft), [Flow-GRPO](https://github.com/yifan123/flow_grpo), [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT), [ReFL/ImageReward](https://github.com/zai-org/ImageReward), and [DanceOPD](https://github.com/worldbench/DanceOPD). We thank the maintainers of the base models, public reward models, and evaluation suites used in this project.

## 📄 License

Released under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for third-party attribution and weight-license boundaries.

## Video training logs

All video OPSD trainers (Wan single-worker/distributed and Cosmos) automatically create offline W&B runs in the
training directory's `wandb/` folder. Logging requires no API key or network access. Default destination is
`cvis_tmu/diffusionopsd`; override it with `WANDB_ENTITY` and `WANDB_PROJECT`. There is no automatic upload.
Failures initializing or writing logging records stop training; existing JSONL, journals, and MP4 files remain available.
Abrupt termination (node loss, SIGKILL, or hard OOM) cannot guarantee SDK finalization.

Each job execution gets a fresh run named `job-JOB_ID-BACKBONE-opsd-training-CLUSTER`; `WANDB_RUN_PURPOSE` overrides the
purpose. Cluster comes from `WANDB_CLUSTER_NAME` or `SLURM_CLUSTER_NAME` (required for Slurm; local runs use `local`).
Resumed jobs share an experiment group and log only their own activity at absolute `optimizer_updates`. Repeated events
at one optimizer update use separate W&B event indices. Logging metadata is stored in `logging/run-*.json`, durable local
events in `logging/events-*.jsonl`, and the experiment group in `logging/experiment.json`. Existing checkpoint formats
remain loadable; new resumable checkpoints also record the group.

Logged data includes configuration/provenance, training metrics, evaluation scores, existing timing/memory measurements,
and all evaluation videos enabled by the experiment configuration. View videos in W&B Workspace media panels using
`eval/prompt_0/video`, `eval/prompt_1/video`, etc.; captions identify prompt, seed, and optimizer update. These are held-out
evaluation outputs. Disabling video saving disables media logging without disabling evaluation scores.

To upload a completed offline run explicitly (use the directory recorded as `offline_directory` in its logging manifest):

```bash
wandb sync --entity cvis_tmu --project diffusionopsd /path/to/run/wandb/offline-run-TIMESTAMP-ID
```

Future video training entry points must use `VideoTrainingLogger` for the entire model-loading/training lifecycle;
the video test suite checks this contract. Image trainers and non-OPSD baselines retain their existing logging behavior.

## Cosmos3-Nano video OPSD

The Cosmos trainer supports `nvidia/Cosmos3-Nano` text-to-video with native UniPC sampling, intermediate-query capture,
rank-16 LoRA, the geometry reward, identity/motion/VideoReward gates, and an EMA behavior adapter. It saves both adapters,
optimizer state, reward statistics, calibration, and RNG for resume. Wan trainers keep their existing sampling behavior.
The update budget counts successful optimizer steps. Prompts whose clips all fail the identity gate are skipped using a
persisted sampling cursor; a full pass of consecutive rejected prompts stops with an error instead of looping indefinitely.

Use the existing `containers/diffusionopsd.sif` and local video reward weights. The Cosmos dependency versions are isolated
in `logs/cosmos-runtime/python`; VideoReward runs in a subprocess on the container's original Transformers stack.
Do not install these versions over the shared Wan environment.

```bash
# Login node: create the dependency overlay and download the pinned Nano checkpoint.
bash scripts/setup_cosmos_nano.sh
bash scripts/setup_cosmos_nano.sh --check-only

# CPU tests, including a tiny real Cosmos transformer/pipeline integration test.
OMP_NUM_THREADS=1 bash scripts/run_cosmos_python.sh -m pytest tests/video/test_cosmos* -q

# First run: one real optimizer update plus held-out generation, on two A100s.
sbatch --partition=gpubase_bygpu_b1 --time=03:00:00 \
  --export=ALL,COSMOS_MAX_UPDATES=1,COSMOS_RUN_DIR="$PWD/logs/cosmos-nano-first-run" \
  scripts/train_opsd_video_cosmos.slurm

# After that job succeeds: resume the same configured 50-update training run.
sbatch --export=ALL,COSMOS_RESUME=1,COSMOS_RUN_DIR="$PWD/logs/cosmos-nano-first-run" \
  scripts/train_opsd_video_cosmos.slurm
```

Defaults are in `config/cosmos_nano.py`: 17 frames at 480×832, 35 UniPC steps, guidance 6, flow shift 10, and four clips per
prompt. Geometry is scored on nine frames resized to 224×384 to control reward memory. The VAE keeps native full-frame decoding; overlapping tiles retain too many saved tensors for target gradients with CPU offload. These are initial training settings;
the short clips and resolution are not a claim about the model's native-quality recipe. Prompts are wrapped deterministically
as JSON descriptions; no external prompt-upsample service is used. Audio, action, image-conditioned, quantized, and few-step
distilled modes are outside this trainer's scope. The native continuation helper restores UniPC history for same-query probes;
the frozen behavior adapter must remain selected throughout such comparisons.

The implementation plan and validation ledger are in
[`docs/superpowers/plans/2026-10-30-cosmos-nano-opsd.md`](docs/superpowers/plans/2026-10-30-cosmos-nano-opsd.md).
