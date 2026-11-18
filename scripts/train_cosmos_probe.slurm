#!/bin/bash
#SBATCH --job-name=cosmos-step6-probe
#SBATCH --account=def-wangcs_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --gpus-per-node=a100:2
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
PROBE_ROOT="${COSMOS_PROBE_DIR:?Set the prepared probe directory}"
export COSMOS_USE_GPU=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 WANDB_MODE=offline
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
: "${COSMOS_SIF:?Set the staged container}"
: "${COSMOS_RUNTIME:?Set the isolated runtime}"
: "${VIDEO_REWARD_CKPT_PATH:?Set staged assets}"
: "${GEOFLOW_DA3_DIR:?Set pinned DA3 source}"
: "${GEOFLOW_WAFT_DIR:?Set pinned WAFT source}"
python - "$PROBE_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
contract = json.loads((root / 'inputs.json').read_text())
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != contract['source_commit']:
    raise ValueError('Probe source commit changed')
if subprocess.check_output(['git', 'diff', 'HEAD', '--', 'src', 'scripts', 'config'], text=True):
    raise ValueError('Probe source has uncommitted modifications')
for name, expected in contract['sha256'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Probe input changed: {name}')
PY
mkdir -p "$PROBE_ROOT/training"
ARGS=(--config config/cosmos_probe_smoke.py --run-dir "$PROBE_ROOT/training"
      --curriculum-manifest "$PROBE_ROOT/train.json" --curriculum-round 0
      --development-manifest "$PROBE_ROOT/development.json"
      --development-count 3 --development-seeds 1)
[[ -f "$PROBE_ROOT/training/training_state.pt" ]] && ARGS+=(--resume)
[[ "${COSMOS_RECOMPUTE_PROBES:-0}" == 1 ]] && ARGS+=(--recompute-probes)
srun --ntasks=1 --gpus-per-task=a100:2 --gpu-bind=per_task:2 --kill-on-bad-exit=1 \
    bash scripts/run_cosmos_python.sh scripts/train_opsd_video_cosmos.py "${ARGS[@]}" \
    >> "$PROBE_ROOT/training/stdout.log" 2>> "$PROBE_ROOT/training/stderr.log"
