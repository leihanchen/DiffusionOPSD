#!/bin/bash
# TamIA compute nodes have no outbound internet. This job is configured to run
# entirely from local scratch: Hugging Face, Transformers, datasets, and W&B
# are forced offline, and caches stay under $SCRATCH.
#
# The cluster account is aip-wangcs. gpubase_bynode_b2 is the 12-hour H200
# partition, so an 8-hour whole-node request fits. H200 nodes are 8 GPUs,
# 64 CPUs, and 1000000M RAM.
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
mkdir -p logs

# Offline compute node. Do not load httpproxy.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export WANDB_DIR="${SCRATCH}/wandb"
export HF_HOME="${SCRATCH}/hf"
export TORCH_HOME="${SCRATCH}/video_reward_ckpts"
export VIDEO_REWARD_CKPT_PATH="${SCRATCH}/video_reward_ckpts"
export XDG_CACHE_HOME="${SCRATCH}/.cache"
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

if [[ ! -e "${SCRATCH}/diffusionopsd.sif" && -e "${SCRATCH}/DiffusionOPSD/containers/diffusionopsd.sif" ]]; then
  ln -sfn "${SCRATCH}/DiffusionOPSD/containers/diffusionopsd.sif" "${SCRATCH}/diffusionopsd.sif"
fi

if [[ ! -s "${SCRATCH}/diffusionopsd.sif" && ! -L "${SCRATCH}/diffusionopsd.sif" ]]; then
  echo "Missing ${SCRATCH}/diffusionopsd.sif" >&2
  exit 1
fi
if [[ ! -d "${VIDEO_REWARD_CKPT_PATH}/Wan2.2-TI2V-5B-Diffusers" ]]; then
  echo "Offline node cannot download weights. Missing ${VIDEO_REWARD_CKPT_PATH}/Wan2.2-TI2V-5B-Diffusers" >&2
  echo "Run scripts/download_video_reward_weights.sh on a login node first." >&2
  exit 1
fi

# Batch shells do not inherit login modules.
# shellcheck disable=SC1091
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load apptainer

echo "Starting Wan OPSD on $(hostname) at $(date -Is)"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE} WANDB_MODE=${WANDB_MODE}"
echo "SIF=${SCRATCH}/diffusionopsd.sif"
echo "VIDEO_REWARD_CKPT_PATH=${VIDEO_REWARD_CKPT_PATH}"

apptainer exec --nv --bind "${SCRATCH}" \
  --env HF_HOME="${SCRATCH}/hf",VIDEO_REWARD_CKPT_PATH="${SCRATCH}/video_reward_ckpts",HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1,HF_HUB_DISABLE_TELEMETRY=1,WANDB_MODE=offline,WANDB_DIR="${SCRATCH}/wandb",TORCH_HOME="${SCRATCH}/video_reward_ckpts",XDG_CACHE_HOME="${SCRATCH}/.cache" \
  "${SCRATCH}/diffusionopsd.sif" \
  python scripts/train_opsd_video_wan.py --config config/wan22_ti2v.py
