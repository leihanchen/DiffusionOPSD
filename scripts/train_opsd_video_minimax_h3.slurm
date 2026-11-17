#!/bin/bash
# Run with bash to submit; no H100 partition has been verified on Narval.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO"
CONFIG="${H3_CONFIG:-config/minimax_h3_t2va.py}"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  : "${H3_ACCOUNT:?Set a verified site submission account}"
  : "${H3_PARTITION:?Set a verified partition supporting the requested GPU type}"
  mkdir -p logs
  # Reject missing/incompatible runtime, assets and conditioning before paying
  # for a GPU allocation. The allocated branch repeats physical-map validation.
  H3_USE_GPU=0 HF_HUB_OFFLINE=1 bash scripts/run_minimax_h3_python.sh \
    scripts/train_opsd_video_minimax_h3.py --config "$CONFIG" --preflight
  # Derive P+R from config rather than allowing a second topology to drift.
  H3_TOPOLOGY="$(H3_USE_GPU=0 bash scripts/run_minimax_h3_python.sh -c \
    'import sys; from scripts.prepare_minimax_h3 import load_config; from diffusionopsd.video.h3_conditioning import validate_h3_topology; c=load_config(sys.argv[1]); validate_h3_topology(c); print(len(c.policy_gpus)+len(c.target_gpus), c.num_nodes)' "$CONFIG")"
  read -r H3_TOTAL_GPUS H3_NUM_NODES <<< "$H3_TOPOLOGY"
  H3_NUM_NODES="${H3_NUM_NODES:-1}"
  export H3_TOTAL_GPUS H3_NUM_NODES
  exec sbatch --job-name=minimax-h3-opsd --account="$H3_ACCOUNT" --partition="$H3_PARTITION" \
    --nodes="$H3_NUM_NODES" --ntasks="$H3_NUM_NODES" --ntasks-per-node=1 \
    --cpus-per-task="${H3_CPUS:-32}" --mem="${H3_MEMORY:-384G}" \
    --gres="gpu:${H3_GPU_TYPE:-h100}:$H3_TOTAL_GPUS" --time="${H3_TIME:-03:00:00}" \
    --signal=USR1@"${H3_SIGNAL_SECONDS:-300}" --output='logs/%x-%j.out' --error='logs/%x-%j.err' "$0"
fi
export H3_USE_GPU=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 WANDB_MODE=offline
export WANDB_ENTITY="${WANDB_ENTITY:-cvis_tmu}" WANDB_PROJECT="${WANDB_PROJECT:-diffusionopsd}"
export WANDB_CLUSTER_NAME="${WANDB_CLUSTER_NAME:-${SLURM_CLUSTER_NAME:-}}"
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1
RUN_DIR="${H3_RUN_DIR:-$REPO/logs/minimax-h3-${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"
H3_TOPOLOGY="$(H3_USE_GPU=0 bash scripts/run_minimax_h3_python.sh -c \
  'import sys; from scripts.prepare_minimax_h3 import load_config; from diffusionopsd.video.h3_conditioning import validate_h3_topology; c=load_config(sys.argv[1]); validate_h3_topology(c); print(len(c.policy_gpus)+len(c.target_gpus), c.num_nodes)' "$CONFIG")"
read -r H3_TOTAL_GPUS H3_NUM_NODES <<< "$H3_TOPOLOGY"
H3_NUM_NODES="${H3_NUM_NODES:-1}"
export H3_TOTAL_GPUS H3_NUM_NODES
# Header/cache validation never creates CUDA contexts. Training repeats validation
# with actual visible-device and torchrun world sizes before any model is loaded.
H3_USE_GPU=0 bash scripts/run_minimax_h3_python.sh scripts/train_opsd_video_minimax_h3.py \
  --config "$CONFIG" --preflight > "$RUN_DIR/preflight.json"
ARGS=(--config "$CONFIG" --run-dir "$RUN_DIR" --max-updates-this-job "${H3_MAX_UPDATES:-0}"
      --checkpoint-publication-reserve-seconds "${H3_PUBLICATION_RESERVE_SECONDS:-60}")
[[ -n "${H3_WALLTIME_SECONDS:-}" ]] && ARGS+=(--walltime-seconds "$H3_WALLTIME_SECONDS")
[[ -n "${H3_RESUME:-}" ]] && ARGS+=(--resume "$H3_RESUME")
# srun puts the elastic agent and workers in a signalable job step. --launch
# ignores USR1 in the agent before invoking torchrun; workers override that
# disposition with H3's flags. TERM/INT retain normal elastic supervision.
if [[ -z "${H3_RDZV_ENDPOINT:-}" ]]; then
  H3_RDZV_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
  H3_RDZV_ENDPOINT="$H3_RDZV_HOST:$((20000 + SLURM_JOB_ID % 30000))"
fi
exec srun --nodes="$H3_NUM_NODES" --ntasks="$H3_NUM_NODES" --ntasks-per-node=1 \
  bash scripts/run_minimax_h3_python.sh scripts/train_opsd_video_minimax_h3.py --launch \
  --nnodes="$H3_NUM_NODES" --rdzv_backend=c10d --rdzv_endpoint="$H3_RDZV_ENDPOINT" --rdzv_id="$SLURM_JOB_ID" \
  --nproc_per_node="$H3_TOTAL_GPUS" scripts/train_opsd_video_minimax_h3.py "${ARGS[@]}"
