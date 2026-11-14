#!/bin/bash
#SBATCH --job-name=geoflow-update-check
#SBATCH --account=def-wangcs
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
set +u
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load apptainer
set -u
TASK_ROOT=/lustre06/project/6062393/leihan
TASK_WORKTREE="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
TASK_BASE="${GEOFLOW_ASSET_ROOT:-$TASK_ROOT/DiffusionOPSD}"
TASK_REFERENCE="$TASK_ROOT/GeoFlow-eas14-reference"
cd "$TASK_WORKTREE"
export VIDEO_REWARD_CKPT_PATH="$TASK_BASE/video_reward_ckpts"
export HF_HOME="$TASK_BASE/hf" TORCH_HOME="$VIDEO_REWARD_CKPT_PATH"
export GEOFLOW_WAFT_DIR="$TASK_REFERENCE/third_party/WAFT"
export GEOFLOW_DA3_DIR="$TASK_REFERENCE/third_party/Depth-Anything-3"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
TASK_OUTPUT="$TASK_WORKTREE/logs/geoflow-update-${SLURM_JOB_ID}"
mkdir -p "$TASK_OUTPUT"
for TASK_BACKBONE in wan cosmos; do
  TASK_PYTHONPATH="$TASK_WORKTREE"
  if [[ "$TASK_BACKBONE" == cosmos ]]; then
    TASK_PYTHONPATH="$TASK_BASE/logs/cosmos-runtime/python:$TASK_WORKTREE"
  fi
  apptainer exec --nv --bind "$TASK_ROOT" \
    --bind "$TASK_WORKTREE/src:/opt/venv/lib/python3.12/site-packages/diffusionopsd" \
    --env PYTHONNOUSERSITE=1 --env "PYTHONPATH=$TASK_PYTHONPATH" \
    "$TASK_BASE/containers/diffusionopsd.sif" \
    python scripts/check_video_reward_update.py --backbone "$TASK_BACKBONE" \
      --output "$TASK_OUTPUT/$TASK_BACKBONE.json"
done
