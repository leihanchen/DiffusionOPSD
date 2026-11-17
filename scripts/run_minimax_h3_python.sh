#!/bin/bash
# H3 uses a dedicated overlay; never write to the Cosmos runtime.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export COSMOS_RUNTIME="${H3_RUNTIME:-$REPO/logs/minimax-h3-runtime/python}"
export COSMOS_SIF="${H3_SIF:-$REPO/containers/diffusionopsd.sif}"
export COSMOS_USE_GPU="${H3_USE_GPU:-0}"
export COSMOS_USE_BASE="${H3_USE_BASE:-0}"
if [[ "${H3_USE_BASE:-0}" == 1 ]]; then
  export COSMOS_USE_BASE=1
fi
if [[ "${H3_USE_BASE:-0}" != 1 && ! -f "$COSMOS_RUNTIME/diffusers/__init__.py" ]]; then
  echo "H3 overlay missing: run scripts/setup_minimax_h3.sh (or set H3_RUNTIME)." >&2
  exit 1
fi
exec bash "$REPO/scripts/run_cosmos_python.sh" "$@"
