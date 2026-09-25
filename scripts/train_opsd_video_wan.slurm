#!/bin/bash
# Narval (Alliance). This job trains Wan2.2-TI2V-5B with the geometry reward
# and the VideoReward mask. Hugging Face, Transformers, datasets, and W&B stay
# offline even though Narval compute nodes can reach the internet. Weights live
# in this repo, written by scripts/prefetch_wan22_login.sh on a login node:
#   video_reward_ckpts/   Wan, VideoReward, Qwen2-VL, DA3, WAFT, DINOv2
#   hf/                   Hugging Face cache
#   data/video_motion/    train.txt and test.txt, already in git
# The image is containers/diffusionopsd.sif in this checkout. Do not use a
# $SCRATCH/diffusionopsd.sif symlink. Job temp and W&B files stay on scratch.
#
# Submit from the repo root on a Narval login node:
#   sbatch scripts/train_opsd_video_wan.slurm
# One-hour debugging run: two calibration prompts, one epoch, then save.
#
# Narval GPUs are A100-40GB, not H100. The trainer uses cuda:0 for the policy
# and cuda:1 for the rewards, so this asks for two A100s on the 3-hour by-GPU
# partition rather than a whole four-GPU node. 24 CPUs and 240G match Narval's
# 12 CPU / ~124G per A100 share. Do not name a node.
#SBATCH --job-name=wan22-opsd
#SBATCH --account=def-wangcs
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --gpus-per-node=a100:2
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "Submit from the repo root: sbatch scripts/train_opsd_video_wan.slurm" >&2
  exit 1
fi
cd "${SLURM_SUBMIT_DIR}"
REPO="$(pwd)"
mkdir -p logs

# Stay offline. Do not load httpproxy. Narval nodes can reach the network,
# but this job must use the prefetched tree, not the hub.
# Model files are the checkout's video_reward_ckpts/ and hf/. The image is
# containers/diffusionopsd.sif. Scratch holds W&B and node temp only.
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

SIF="${REPO}/containers/diffusionopsd.sif"
if [[ -L "${SIF}" ]]; then
  echo "Refusing symlink ${SIF}. Use the regular file in containers/." >&2
  exit 1
fi
if [[ ! -f "${SIF}" ]]; then
  echo "Missing regular file ${SIF}." >&2
  exit 1
fi
if [[ "${REPO}" == "${SCRATCH}" || "${REPO}" == "${SCRATCH}"/* ]]; then
  APPTAINER_BINDS=(--bind "${SCRATCH}")
else
  APPTAINER_BINDS=(--bind "${REPO}" --bind "${SCRATCH}")
fi
# The image pip-installed a copy of src. Overlay the checkout so offline
# loader fixes are what this job runs, without rebuilding the SIF.
APPTAINER_BINDS+=(--bind "${REPO}/src:/opt/venv/lib/python3.12/site-packages/diffusionopsd")

# No network here. Confirm the login-node prefetch, then point VideoReward at
# the local Qwen checkout and SDPA if that rewrite is still pending.
bash scripts/prefetch_wan22_login.sh --check-only

# Batch shells do not inherit login modules.
# Cluster initialization reads optional variables that may be unset.
set +u
# shellcheck disable=SC1091
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load apptainer
set -u

echo "Starting Wan OPSD on $(hostname) at $(date -Is)"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE} WANDB_MODE=${WANDB_MODE}"
echo "SIF=${SIF}"
echo "VIDEO_REWARD_CKPT_PATH=${VIDEO_REWARD_CKPT_PATH}"
echo "HF_HOME=${HF_HOME}"

# Keep host ~/.local packages from overriding the container's Python stack.
apptainer exec --nv "${APPTAINER_BINDS[@]}" --pwd "${PWD}" \
  --env PYTHONNOUSERSITE=1 \
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
  python scripts/train_opsd_video_wan.py --config config/wan22_ti2v_debug.py \
    --config.logdir="${REPO}/logs/wan22-debug-${SLURM_JOB_ID}"
