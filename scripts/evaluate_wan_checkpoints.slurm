#!/bin/bash
#SBATCH --job-name=wan22-small-eval
#SBATCH --account=aip-wangcs
#SBATCH --partition=gpubase_h100_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export CHECKPOINT_MANIFEST=config/wan22_small_checkpoint_comparison.json
export CONFIG_PATH=config/wan22_step4_evaluation.json
export SAMPLER=euler NUM_PROMPTS=2 SEED_OFFSETS=0
export POLICY_DEVICE=cuda:0 REWARD_DEVICE=cuda:0 VAE_DEVICE=cuda:0 MIN_GPU_MEMORY_GIB=79
export WANDB_MODE=offline WANDB_ENTITY=cvis_tmu WANDB_PROJECT=diffusionopsd
exec bash scripts/investigate_wan_inference.slurm
