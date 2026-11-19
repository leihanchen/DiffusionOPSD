#!/bin/bash
#SBATCH --job-name=cosmos-quality
#SBATCH --account=def-wangcs_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export COSMOS_USE_GPU=1 WANDB_MODE=offline HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
: "${COSMOS_SIF:?Set the staged Cosmos container}"
: "${COSMOS_RUNTIME:?Set the staged Cosmos Python runtime}"
: "${VIDEO_REWARD_CKPT_PATH:?Set the staged model path}"
srun --ntasks=1 --gpus-per-task=a100:1 --gpu-bind=per_task:1 --kill-on-bad-exit=1 \
    bash scripts/run_cosmos_python.sh scripts/compare_cosmos_video_quality.py \
    --probe-report "${COSMOS_QUALITY_PROBE_REPORT:-logs/curriculum-step6/replay-native/training/round_probe.json}" \
    --run-dir "${COSMOS_QUALITY_RUN_DIR:-logs/cosmos-quality/${SLURM_JOB_ID}}"
