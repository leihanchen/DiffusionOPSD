#!/bin/bash
# Submit from the repository root after setup_cosmos_nano.sh --check-only passes.
#SBATCH --job-name=cosmos-nano-opsd
#SBATCH --account=def-wangcs
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --gpus-per-node=a100:2
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
export WANDB_ENTITY="${WANDB_ENTITY:-cvis_tmu}" WANDB_PROJECT="${WANDB_PROJECT:-diffusionopsd}"
export WANDB_CLUSTER_NAME="${WANDB_CLUSTER_NAME:-${SLURM_CLUSTER_NAME:-}}"
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export COSMOS_USE_GPU=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 WANDB_MODE=offline
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
RUN_DIR="${COSMOS_RUN_DIR:-$PWD/logs/cosmos-nano-${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"
ARGS=(--config "${COSMOS_CONFIG:-config/cosmos_nano.py}" --run-dir "$RUN_DIR"
      --max-updates-this-job "${COSMOS_MAX_UPDATES:-0}")
[[ "${COSMOS_RESUME:-0}" == 1 ]] && ARGS+=(--resume)
COSMOS_USE_GPU=0 bash scripts/run_cosmos_python.sh scripts/prefetch_video_dataset.py \
  --config "${COSMOS_CONFIG:-config/cosmos_nano.py}" --check-only
srun --ntasks=1 --gpus-per-task=a100:2 --gpu-bind=per_task:2 --kill-on-bad-exit=1 \
  bash scripts/run_cosmos_python.sh scripts/train_opsd_video_cosmos.py "${ARGS[@]}"
