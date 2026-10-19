#!/bin/bash
# Resume EAS-7 at step 2; evaluate steps 2, 4, 6; no successor.
#SBATCH --job-name=wan22-two-gpu-monitor
#SBATCH --account=def-wangcs
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=448G
#SBATCH --gpus-per-node=a100:2
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export WAN_EPOCH_CONFIG=config/wan22_ti2v_epoch_two_gpu.py
export WAN_EPOCH_RUN_PREFIX=wan22-two-gpu-probe
export WAN_EPOCH_RUN_ID=4962523
export WAN_EPOCH_CHAIN_JOBS=0
export WAN_EPOCH_MAX_UPDATES="${WAN_EPOCH_MAX_UPDATES:-4}"
export WAN_EPOCH_JOB_BUDGET_HOURS=5
export WAN_EPOCH_MONITOR_EVAL_FREQ=2
export WAN_EPOCH_EVALUATE_ON_RESUME=true
export WAN_EPOCH_EXPECTED_RESUME_UPDATE="${WAN_EPOCH_EXPECTED_RESUME_UPDATE:-2}"
[[ -f logs/wan22-two-gpu-probe-4962523/training_state.pt ]] || {
  echo "Missing step-2 checkpoint for EAS-7 monitoring" >&2
  exit 1
}
exec bash scripts/train_opsd_video_wan_epoch.slurm
