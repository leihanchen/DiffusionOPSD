#!/bin/bash
# Submit from the repository root after setup_cosmos_nano.sh --check-only passes.
#SBATCH --job-name=cosmos-nano-opsd
#SBATCH --account=def-wangcs
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export COSMOS_USE_GPU=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 WANDB_MODE=offline
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
RUN_DIR="${COSMOS_RUN_DIR:-$PWD/logs/cosmos-nano-${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"
ARGS=(--config "${COSMOS_CONFIG:-config/cosmos_nano.py}" --run-dir "$RUN_DIR"
      --max-updates-this-job "${COSMOS_MAX_UPDATES:-0}")
[[ "${COSMOS_RESUME:-0}" == 1 ]] && ARGS+=(--resume)
bash scripts/run_cosmos_python.sh scripts/train_opsd_video_cosmos.py "${ARGS[@]}"
