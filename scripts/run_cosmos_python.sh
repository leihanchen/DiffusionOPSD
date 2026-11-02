#!/bin/bash
# Execute the isolated Cosmos stack while retaining the existing CUDA/reward image.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO"
if ! command -v apptainer >/dev/null; then
  set +u
  source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  module load apptainer
  set -u
fi
SIF="${COSMOS_SIF:-$REPO/containers/diffusionopsd.sif}"
RUNTIME="${COSMOS_RUNTIME:-$REPO/logs/cosmos-runtime/python}"
export PYTHONNOUSERSITE=1
export VIDEO_REWARD_CKPT_PATH="${VIDEO_REWARD_CKPT_PATH:-$REPO/video_reward_ckpts}"
export HF_HOME="${HF_HOME:-$REPO/hf}" TORCH_HOME="${TORCH_HOME:-$VIDEO_REWARD_CKPT_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
BINDS=(--bind "$REPO" --bind "$REPO/src:/opt/venv/lib/python3.12/site-packages/diffusionopsd")
if [[ -n "${SCRATCH:-}" && "$REPO" != "$SCRATCH" && "$REPO" != "$SCRATCH"/* ]]; then
  BINDS+=(--bind "$SCRATCH")
fi
EXTRA=()
[[ "${COSMOS_USE_GPU:-0}" == 1 ]] && EXTRA+=(--nv)
PYTHON_PATH="$REPO"
[[ "${COSMOS_USE_BASE:-0}" == 1 ]] || PYTHON_PATH="$RUNTIME:$REPO"
exec apptainer exec "${EXTRA[@]}" "${BINDS[@]}" --pwd "$REPO" \
  --env "WANDB_ENTITY=${WANDB_ENTITY:-cvis_tmu}" --env "WANDB_PROJECT=${WANDB_PROJECT:-diffusionopsd}" \
  --env "WANDB_CLUSTER_NAME=${WANDB_CLUSTER_NAME:-${SLURM_CLUSTER_NAME:-}}" \
  --env "WANDB_RUN_PURPOSE=${WANDB_RUN_PURPOSE:-}" \
  --env PYTHONNOUSERSITE=1 --env "PYTHONPATH=$PYTHON_PATH" \
  --env "OMP_NUM_THREADS=$OMP_NUM_THREADS" --env "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS" \
  --env "VIDEO_REWARD_CKPT_PATH=$VIDEO_REWARD_CKPT_PATH" --env "TORCH_HOME=$TORCH_HOME" \
  --env "HF_HOME=$HF_HOME" --env "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}" \
  --env "TRANSFORMERS_OFFLINE=${HF_HUB_OFFLINE:-1}" --env HF_HUB_DISABLE_XET=1 \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  --env PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
  "$SIF" python "$@"
