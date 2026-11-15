#!/bin/bash
# Setup/download on an online login node; compute nodes remain offline.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO"
export H3_RUNTIME="${H3_RUNTIME:-$REPO/logs/minimax-h3-runtime/python}"
if [[ "${1:-}" != --check-only ]]; then
  mkdir -p "$H3_RUNTIME"
  H3_USE_BASE=1 HF_HUB_OFFLINE=0 bash scripts/run_minimax_h3_python.sh \
    -m pip install --target "$H3_RUNTIME" --no-deps -r containers/minimax-h3-requirements.txt
fi
bash scripts/run_minimax_h3_python.sh scripts/check_minimax_h3_update.py --runtime-only
if [[ "${1:-}" == --download ]]; then
  HF_HUB_OFFLINE=0 bash scripts/run_minimax_h3_python.sh scripts/prepare_minimax_h3.py --config config/minimax_h3_t2va.py --prefetch
fi
