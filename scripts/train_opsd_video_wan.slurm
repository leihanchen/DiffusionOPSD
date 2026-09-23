#!/bin/bash
# TamIA compute nodes have no outbound internet. This job trains Wan2.2-TI2V-5B
# with the geometry reward and the VideoReward mask. Hugging Face, Transformers,
# datasets, and W&B stay offline. Weights live in this repo, written by
# scripts/prefetch_wan22_login.sh on a login node:
#   video_reward_ckpts/   Wan, VideoReward, Qwen2-VL, DA3, WAFT, DINOv2
#   hf/                   Hugging Face cache
#   data/video_motion/    train.txt and test.txt, already in git
# The image is $SCRATCH/diffusionopsd.sif. Job temp and W&B files stay on scratch.
#
# The cluster account is aip-wangcs. gpubase_bynode_b2 is the 12-hour H200
# partition, so an 8-hour whole-node request fits. H200 nodes are 8 GPUs,
# 64 CPUs, and 1000000M RAM. The trainer uses cuda:0 for the policy and
# cuda:1 for the geometry reward and VideoReward.
#SBATCH --job-name=opsd-wan22
#SBATCH --account=aip-wangcs
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --gpus-per-node=h200:8
#SBATCH --constraint=h200
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-${SCRATCH}/DiffusionOPSD}"
REPO="$(pwd)"
mkdir -p logs

# Offline compute node. Do not load httpproxy.
# Model files are the checkout's video_reward_ckpts/ and hf/. Scratch holds
# the image, W&B, and node temp only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export WANDB_DIR="${SCRATCH}/wandb"
export HF_HOME="${REPO}/hf"
export TORCH_HOME="${REPO}/video_reward_ckpts"
export VIDEO_REWARD_CKPT_PATH="${REPO}/video_reward_ckpts"
export XDG_CACHE_HOME="${REPO}/.cache"
export APPTAINER_CACHEDIR="${SCRATCH}/apptainer/cache"
export APPTAINER_TMPDIR="${SCRATCH}/apptainer/tmp"
export TMPDIR="${SLURM_TMPDIR:-${SCRATCH}/tmp}"
mkdir -p \
  "${HF_HOME}" \
  "${WANDB_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${APPTAINER_CACHEDIR}" \
  "${APPTAINER_TMPDIR}" \
  "${TMPDIR}" \
  "${VIDEO_REWARD_CKPT_PATH}"

if [[ -s "${SCRATCH}/diffusionopsd.sif" ]]; then
  SIF="${SCRATCH}/diffusionopsd.sif"
elif [[ -s "${REPO}/containers/diffusionopsd.sif" ]]; then
  SIF="${REPO}/containers/diffusionopsd.sif"
elif [[ -s "${REPO}/diffusionopsd.sif" ]]; then
  SIF="${REPO}/diffusionopsd.sif"
else
  SIF="${SCRATCH}/diffusionopsd.sif"
fi
if [[ "${REPO}" == "${SCRATCH}" || "${REPO}" == "${SCRATCH}"/* ]]; then
  APPTAINER_BINDS=(--bind "${SCRATCH}")
else
  APPTAINER_BINDS=(--bind "${REPO}" --bind "${SCRATCH}")
fi

# No network here. Confirm the login-node prefetch, then point VideoReward at
# the local Qwen checkout and SDPA if that rewrite is still pending.
bash scripts/prefetch_wan22_login.sh --check-only

# Batch shells do not inherit login modules.
# shellcheck disable=SC1091
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load apptainer

echo "Starting Wan OPSD on $(hostname) at $(date -Is)"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE} WANDB_MODE=${WANDB_MODE}"
echo "SIF=${SIF}"
echo "VIDEO_REWARD_CKPT_PATH=${VIDEO_REWARD_CKPT_PATH}"
echo "HF_HOME=${HF_HOME}"

apptainer exec --nv "${APPTAINER_BINDS[@]}" --pwd "${PWD}" \
  --env "HF_HOME=${HF_HOME}" \
  --env "VIDEO_REWARD_CKPT_PATH=${VIDEO_REWARD_CKPT_PATH}" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env HF_HUB_DISABLE_TELEMETRY=1 \
  --env WANDB_MODE=offline \
  --env "WANDB_DIR=${WANDB_DIR}" \
  --env "TORCH_HOME=${TORCH_HOME}" \
  --env "XDG_CACHE_HOME=${XDG_CACHE_HOME}" \
  --env GIT_TERMINAL_PROMPT=0 \
  "${SIF}" \
  python scripts/train_opsd_video_wan.py --config config/wan22_ti2v.py
