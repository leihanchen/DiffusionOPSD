#!/bin/bash
# Allocation is supplied by submit_opsd_video_wan_scaled.py.
#SBATCH --job-name=wan22-scaled
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
REPO="$PWD"
RUN_DIR="${WAN_EPOCH_RUN_DIR:?Use the scaled submission wrapper}"
CONFIG="${WAN_EPOCH_CONFIG:?}"
SIF="$REPO/containers/diffusionopsd.sif"
[[ -f "$SIF" && ! -L "$SIF" ]] || { echo "Missing regular container: $SIF" >&2; exit 1; }
[[ "$SLURM_NTASKS" == "$WAN_SCALE_WORLD_SIZE" ]] || { echo "Task count differs from requested layout" >&2; exit 1; }
[[ "$SLURM_JOB_NUM_NODES" == "$WAN_SCALE_NODES" ]] || { echo "Node count differs from requested layout" >&2; exit 1; }
mkdir -p "$RUN_DIR"
export WANDB_MODE=offline WANDB_DIR="$RUN_DIR" PYTHONNOUSERSITE=1
# Capture transport setup failures before the first application collective.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
# Slurm-isolated peers cannot import CUDA device/host handles in this container.
# Verified on Narval: POSIX host shared memory supports gathers and gradient sums.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$REPO/hf" TORCH_HOME="$REPO/video_reward_ckpts" VIDEO_REWARD_CKPT_PATH="$REPO/video_reward_ckpts"
export XDG_CACHE_HOME="$REPO/.cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export APPTAINER_CACHEDIR="$SCRATCH/apptainer/cache" APPTAINER_TMPDIR="$SCRATCH/apptainer/tmp"
export TMPDIR="${SLURM_TMPDIR:-$SCRATCH/tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$TMPDIR" "$XDG_CACHE_HOME"
bash scripts/prefetch_wan22_login.sh --check-only
set +u
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load apptainer
set -u
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))" WORLD_SIZE="$SLURM_NTASKS"
if [[ "$REPO" == "$SCRATCH" || "$REPO" == "$SCRATCH"/* ]]; then
  BINDS=(--bind "$SCRATCH")
else
  BINDS=(--bind "$REPO" --bind "$SCRATCH")
fi
BINDS+=(--bind "$REPO/src:/opt/venv/lib/python3.12/site-packages/diffusionopsd")
export WAN_SCALE_SIF="$SIF" WAN_SCALE_REPO="$REPO"
# Expansion of CUDA_VISIBLE_DEVICES happens inside the Slurm task after binding.
srun --label --ntasks="$SLURM_NTASKS" --ntasks-per-node="$WAN_SCALE_WORKERS_PER_NODE" \
  --gpus-per-task=a100:2 --gpu-bind=per_task:2 --kill-on-bad-exit=1 \
  bash -c 'exec apptainer exec --nv "$@" --pwd "$WAN_SCALE_REPO" \
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    --env "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}" \
    --env "NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE:-0}" \
    --env PYTHONNOUSERSITE=1 --env WANDB_MODE=offline --env "WANDB_DIR=$WANDB_DIR" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 \
    --env "HF_HOME=$HF_HOME" --env "TORCH_HOME=$TORCH_HOME" \
    --env "VIDEO_REWARD_CKPT_PATH=$VIDEO_REWARD_CKPT_PATH" --env "XDG_CACHE_HOME=$XDG_CACHE_HOME" \
    --env "MASTER_ADDR=$MASTER_ADDR" --env "MASTER_PORT=$MASTER_PORT" \
    "$WAN_SCALE_SIF" python scripts/train_opsd_video_wan_epoch.py \
      --config "$WAN_EPOCH_CONFIG" --config.logdir="$WAN_EPOCH_RUN_DIR" \
      --job_budget_hours="$WAN_EPOCH_JOB_BUDGET_HOURS" \
      --distributed_timeout_minutes="${WAN_EPOCH_DISTRIBUTED_TIMEOUT_MINUTES:-180}" \
      --max_updates_this_job="$WAN_EPOCH_MAX_UPDATES" \
      --monitor_eval_freq="$WAN_EPOCH_MONITOR_EVAL_FREQ" \
      --evaluate_on_resume="$WAN_EPOCH_EVALUATE_ON_RESUME" \
      --expected_resume_update="$WAN_EPOCH_EXPECTED_RESUME_UPDATE" \
      --allow_topology_change="$WAN_EPOCH_ALLOW_TOPOLOGY_CHANGE" --scalable_two_gpu=true \
      --single_node_smoke="${WAN_SCALE_SINGLE_NODE_SMOKE:-false}" \
      ${WAN_EPOCH_RESUME_FROM:+--resume_from="$WAN_EPOCH_RESUME_FROM"}' scaled-worker "${BINDS[@]}"
RESULT="$RUN_DIR/job_${SLURM_JOB_ID}.json"
[[ -f "$RESULT" ]] || { echo "Missing completed-job result: $RESULT" >&2; exit 1; }
STATUS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RESULT")"
if [[ "$STATUS" == continue && "$WAN_EPOCH_CHAIN_JOBS" == 1 ]]; then
  NEXT="$(python scripts/submit_opsd_video_wan_scaled.py --resubmit "$WAN_SCALE_OPTIONS" --dependency "afterok:$SLURM_JOB_ID")"
  printf '%s -> %s\n' "$SLURM_JOB_ID" "$NEXT" >> "$RUN_DIR/job_chain.txt"
  echo "Submitted successor: $NEXT"
elif [[ "$STATUS" == continue && "$WAN_EPOCH_CHAIN_JOBS" == 0 ]]; then
  echo "Probe completed; no successor submitted"
elif [[ "$STATUS" == complete ]]; then
  echo "Full pass completed at update 50"
else
  echo "Unexpected status: $STATUS" >&2
  exit 1
fi
