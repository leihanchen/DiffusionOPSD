#!/bin/bash
# Submit from the repository root: sbatch scripts/evaluate_wan_step4.slurm
# 16 prompts x 2 seeds x (base, trained), using the shared inference launcher.
#SBATCH --job-name=wan22-step4-eval
#SBATCH --account=def-wangcs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus-per-node=a100:2
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export POLICY_PATH="$PWD/logs/wan22-two-gpu-probe-4962523/training_state_step4.pt"
export CONFIG_PATH="$PWD/config/wan22_step4_evaluation.json"
export CHECKPOINT_FORMAT=training EXPECTED_UPDATE=4 SAMPLER=euler
export NUM_PROMPTS=16 SEED_OFFSETS="0 1000"
export WANDB_MODE=offline WANDB_ENTITY=cvis_tmu WANDB_PROJECT=diffusionopsd
exec bash scripts/investigate_wan_inference.slurm
