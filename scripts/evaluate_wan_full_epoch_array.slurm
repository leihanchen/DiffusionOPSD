#!/bin/bash
# Eight shards of eight prompts; at most two shards run concurrently.
#SBATCH --job-name=wan22-epoch-eval-array
#SBATCH --account=def-wangcs_gpu
#SBATCH --array=0-7%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus-per-node=a100:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
task="${SLURM_ARRAY_TASK_ID:?Requires a Slurm array}"
[[ "$task" =~ ^[0-7]$ ]] || { echo "Invalid array task: $task" >&2; exit 1; }
root="${WAN_EVAL_ARRAY_ROOT:-$PWD/logs/wan22-full-epoch-array-${SLURM_ARRAY_JOB_ID:?}}"
mkdir -p "$root"
export PROMPT_START=$((task * 8)) NUM_PROMPTS=8
export POLICY_DEVICE=cuda:0 VAE_DEVICE=cuda:0 REWARD_DEVICE=cuda:1
export INFERENCE_OUTPUT_DIR="$root/task_$task"
export WANDB_RUN_GROUP="job-${SLURM_ARRAY_JOB_ID}-base-step7-step50-eval-narval"
export WANDB_RUN_NAME="job-${SLURM_ARRAY_JOB_ID}_${task}-base-step7-step50-prompts-${PROMPT_START}-narval"
export WANDB_ENTITY=cvis_tmu WANDB_PROJECT=diffusionopsd WANDB_MODE=offline
bash scripts/evaluate_wan_full_epoch.slurm
# Every successful shard checks; the last one combines all results under a lock.
set +u
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load apptainer
set -u
apptainer exec --nv --bind "$PWD" --bind "$SCRATCH" \
  --bind "$PWD/src:/opt/venv/lib/python3.12/site-packages/diffusionopsd" --pwd "$PWD" \
  --env PYTHONNOUSERSITE=1 --env WANDB_MODE=offline \
  "$PWD/containers/diffusionopsd.sif" python scripts/aggregate_wan_evaluation.py \
  --array-root "$root" --if-ready --wandb
