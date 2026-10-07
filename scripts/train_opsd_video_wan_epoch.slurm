#!/bin/bash
# One 24-hour segment of the 200-prompt Wan2.2 OPSD pass.
# Submit once from the repository root; successful incomplete segments submit
# a single afterok successor using the same WAN_EPOCH_RUN_ID.
#SBATCH --job-name=wan22-epoch
#SBATCH --account=def-wangcs
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=448G
#SBATCH --gpus-per-node=a100:3
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
REPO="$PWD"
RUN_ID="${WAN_EPOCH_RUN_ID:-${SLURM_JOB_ID}}"
RUN_DIR="${REPO}/logs/wan22-epoch-${RUN_ID}"
JOB_BUDGET_HOURS="${WAN_EPOCH_JOB_BUDGET_HOURS:-20}"
CHAIN_JOBS="${WAN_EPOCH_CHAIN_JOBS:-1}"
SIF="${REPO}/containers/diffusionopsd.sif"
[[ -f "$SIF" && ! -L "$SIF" ]] || { echo "Missing regular container: $SIF" >&2; exit 1; }
mkdir -p "$RUN_DIR"
export WANDB_MODE=offline WANDB_DIR="$RUN_DIR" PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="${REPO}/hf" TORCH_HOME="${REPO}/video_reward_ckpts"
export VIDEO_REWARD_CKPT_PATH="${REPO}/video_reward_ckpts"
export XDG_CACHE_HOME="${REPO}/.cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export APPTAINER_CACHEDIR="${SCRATCH}/apptainer/cache"
export APPTAINER_TMPDIR="${SCRATCH}/apptainer/tmp"
export TMPDIR="${SLURM_TMPDIR:-${SCRATCH}/tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$TMPDIR" "$XDG_CACHE_HOME"
bash scripts/prefetch_wan22_login.sh --check-only
set +u
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load apptainer
set -u
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
export WORLD_SIZE=4
if [[ "$REPO" == "$SCRATCH" || "$REPO" == "$SCRATCH"/* ]]; then
  BINDS=(--bind "$SCRATCH")
else
  BINDS=(--bind "$REPO" --bind "$SCRATCH")
fi
BINDS+=(--bind "$REPO/src:/opt/venv/lib/python3.12/site-packages/diffusionopsd")
echo "Wan2.2 epoch run=$RUN_ID job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST master=$MASTER_ADDR budget_hours=$JOB_BUDGET_HOURS chain_jobs=$CHAIN_JOBS"
srun --label --ntasks=4 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  apptainer exec --nv "${BINDS[@]}" --pwd "$REPO" \
    --env PYTHONNOUSERSITE=1 --env WANDB_MODE=offline --env "WANDB_DIR=$WANDB_DIR" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 \
    --env "HF_HOME=$HF_HOME" --env "TORCH_HOME=$TORCH_HOME" \
    --env "VIDEO_REWARD_CKPT_PATH=$VIDEO_REWARD_CKPT_PATH" --env "XDG_CACHE_HOME=$XDG_CACHE_HOME" \
    --env "MASTER_ADDR=$MASTER_ADDR" --env "MASTER_PORT=$MASTER_PORT" \
    "$SIF" python scripts/train_opsd_video_wan_epoch.py \
      --config config/wan22_ti2v_epoch.py \
      --config.logdir="$RUN_DIR" --job_budget_hours="$JOB_BUDGET_HOURS" \
      --distributed_timeout_minutes="${WAN_EPOCH_DISTRIBUTED_TIMEOUT_MINUTES:-180}" \
      --max_updates_this_job="${WAN_EPOCH_MAX_UPDATES:-0}"
RESULT="$RUN_DIR/job_${SLURM_JOB_ID}.json"
[[ -f "$RESULT" ]] || { echo "Missing completed-job result: $RESULT" >&2; exit 1; }
STATUS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RESULT")"
if [[ "$STATUS" == continue && "$CHAIN_JOBS" == 1 ]]; then
  NEXT="$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
    --export="ALL,WAN_EPOCH_RUN_ID=${RUN_ID}" scripts/train_opsd_video_wan_epoch.slurm)"
  printf '%s -> %s\n' "$SLURM_JOB_ID" "$NEXT" >> "$RUN_DIR/job_chain.txt"
  echo "Submitted successor $NEXT after job $SLURM_JOB_ID"
elif [[ "$STATUS" == continue && "$CHAIN_JOBS" == 0 ]]; then
  echo "Probe stopped after update $(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["completed_updates"])' "$RESULT"); no successor submitted"
elif [[ "$STATUS" == complete ]]; then
  echo "Full pass completed at update 50"
else
  echo "Unexpected status: $STATUS" >&2
  exit 1
fi
