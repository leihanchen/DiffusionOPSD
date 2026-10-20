#!/bin/bash
# 64 held-out prompts x one seed x base/step7/step50 = 192 videos.
#SBATCH --job-name=wan22-full-epoch-eval
#SBATCH --account=aip-wangcs
#SBATCH --partition=gpubase_h100_b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gpus-per-node=h100:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export CHECKPOINT_MANIFEST=config/wan22_full_epoch_evaluation_checkpoints.json
export CONFIG_PATH=config/wan22_inference_debug.json
unset EXPECTED_UPDATE
export CHECKPOINT_FORMAT=training
export SAMPLER=native NUM_PROMPTS="${NUM_PROMPTS:-64}" SEED_OFFSETS=0 FPS=24
export COMPARE_CHECKPOINTS=1 TABLE_LOGGING=1 REPORT_PROMPT_STRATA=1 VERIFY_VIDEOS=1
export EXPECTED_FLOW_SHIFT=5 MIN_GPU_MEMORY_GIB=79
export POLICY_DEVICE=cuda:0 VAE_DEVICE=cuda:0 REWARD_DEVICE="${REWARD_DEVICE:-cuda:1}"
if [[ "${SLURM_GPUS_ON_NODE:-}" == 1 ]]; then
  export REWARD_DEVICE=cuda:0
fi
export WANDB_ENTITY=cvis_tmu WANDB_PROJECT=diffusionopsd WANDB_MODE=offline
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-job-${SLURM_JOB_ID}-base-step7-step50-64prompt-eval-killarney}"
exec bash scripts/investigate_wan_inference.slurm
