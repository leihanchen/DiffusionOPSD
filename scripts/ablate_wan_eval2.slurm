#!/bin/bash
#SBATCH --job-name=wan22-eval2-ablation
#SBATCH --account=aip-wangcs
#SBATCH --partition=gpubase_h100_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gpus-per-node=h100:2
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export PROMPT_INDEX=1
export ABLATION_CONFIG=config/wan22_eval2_duration_ablation.json
export WANDB_ENTITY=cvis_tmu WANDB_PROJECT=diffusionopsd WANDB_MODE=offline
exec bash scripts/ablate_wan_inference.slurm
