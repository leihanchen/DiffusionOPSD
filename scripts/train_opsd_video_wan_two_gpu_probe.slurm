#!/bin/bash
# EAS-7: four workers, two GPUs per node, two real updates, no successor.
#SBATCH --job-name=wan22-two-gpu-probe
#SBATCH --account=def-wangcs
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=448G
#SBATCH --gpus-per-node=a100:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export WAN_EPOCH_CONFIG=config/wan22_ti2v_epoch_two_gpu.py
export WAN_EPOCH_RUN_PREFIX=wan22-two-gpu-probe
export WAN_EPOCH_CHAIN_JOBS=0
export WAN_EPOCH_MAX_UPDATES=2
export WAN_EPOCH_JOB_BUDGET_HOURS=3
exec bash scripts/train_opsd_video_wan_epoch.slurm
