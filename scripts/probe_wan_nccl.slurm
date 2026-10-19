#!/bin/bash
#SBATCH --account=def-wangcs
#SBATCH --job-name=wan-nccl-probe
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=a100:4
#SBATCH --gpu-bind=per_task:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export NCCL_PROBE_REPO="$PWD" NCCL_PROBE_OUTPUT="$PWD/logs/nccl-probe-$SLURM_JOB_ID"
mkdir -p "$NCCL_PROBE_OUTPUT"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set +u
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load apptainer
set -u
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
srun --label --ntasks=2 --ntasks-per-node=2 --cpus-per-task=2 \
  --gpus-per-task=a100:2 --gpu-bind=per_task:2 --kill-on-bad-exit=1 \
  bash -c 'exec apptainer exec --nv --bind "$NCCL_PROBE_REPO" --bind "$SCRATCH" \
    --bind "$NCCL_PROBE_REPO/src:/opt/venv/lib/python3.12/site-packages/diffusionopsd" \
    --pwd "$NCCL_PROBE_REPO" --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    --env PYTHONNOUSERSITE=1 --env NCCL_DEBUG=INFO \
    --env "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE" --env "NCCL_CUMEM_HOST_ENABLE=$NCCL_CUMEM_HOST_ENABLE" \
    "$NCCL_PROBE_REPO/containers/diffusionopsd.sif" python scripts/probe_wan_nccl.py \
    --gpus-per-worker=2 --output-dir "$NCCL_PROBE_OUTPUT"'
python - "$NCCL_PROBE_OUTPUT" <<'PY'
import json
from pathlib import Path
import sys
output = Path(sys.argv[1])
records = [json.loads((output / f"rank_{rank}.json").read_text()) for rank in range(2)]
assert all(record["status"] == "PASSED" for record in records), records
summary = {"status": "PASSED", "workers": 2, "gpus_per_worker": 2, "records": records}
(output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print("NCCL probe passed: two exclusive GPU pairs; balanced and zero-retention gradients match serial reference.")
PY
