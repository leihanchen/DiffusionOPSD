#!/usr/bin/env bash
# Use a dedicated VLM interpreter; never modify Cosmos or VideoReward runtimes.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
: "${CURRICULUM_VLM_PYTHON:?Set CURRICULUM_VLM_PYTHON to the isolated VLM Python executable}"
: "${CURRICULUM_VLM_MODEL_PATH:?Set CURRICULUM_VLM_MODEL_PATH to local pinned model weights}"
: "${CURRICULUM_VLM_REVISION:?Set CURRICULUM_VLM_REVISION to the immutable model commit}"
if [[ ! -x "$CURRICULUM_VLM_PYTHON" || ! -d "$CURRICULUM_VLM_MODEL_PATH" ]]; then
  echo 'VLM interpreter or local checkpoint is unavailable' >&2
  exit 1
fi
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CURRICULUM_TEMPLATE_DIR="$REPO/config/curriculum_prompts"
# Avoid inheriting the training overlay through PYTHONPATH.
unset PYTHONPATH
cd "$REPO"
exec "$CURRICULUM_VLM_PYTHON" -s "$REPO/scripts/curriculum_vlm_worker.py" "$@"
