#!/bin/bash
# Run on a login node; downloads are forbidden on the offline training nodes.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO"
RUNTIME="${COSMOS_RUNTIME:-$REPO/logs/cosmos-runtime/python}"
if [[ "${1:-}" != --check-only ]]; then
  mkdir -p "$RUNTIME"
  COSMOS_USE_BASE=1 HF_HUB_OFFLINE=0 bash scripts/run_cosmos_python.sh \
    -m pip install --target "$RUNTIME" --no-deps -r containers/cosmos-nano-requirements.txt
  HF_HUB_OFFLINE=0 bash scripts/run_cosmos_python.sh - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
revision = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
target = Path(os.environ["VIDEO_REWARD_CKPT_PATH"]) / "Cosmos3-Nano"
snapshot_download(
    "nvidia/Cosmos3-Nano", revision=revision, local_dir=target, max_workers=1,
    allow_patterns=["model_index.json", "transformer/*", "vae/*", "text_tokenizer/*", "scheduler/*"],
)
(target / "opsd_revision.txt").write_text(revision + "\n")
PY
fi
bash scripts/run_cosmos_python.sh scripts/train_opsd_video_cosmos.py --preflight
