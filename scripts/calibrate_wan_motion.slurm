#!/bin/bash
#SBATCH --job-name=wan22-motion-calibration
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
REPO="$PWD"
export HF_HOME="$REPO/hf" TORCH_HOME="$REPO/video_reward_ckpts"
export VIDEO_REWARD_CKPT_PATH="$REPO/video_reward_ckpts" XDG_CACHE_HOME="$REPO/.cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline WANDB_ENTITY=cvis_tmu WANDB_PROJECT=diffusionopsd
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export APPTAINER_CACHEDIR="$SCRATCH/apptainer/cache" APPTAINER_TMPDIR="$SCRATCH/apptainer/tmp"
export TMPDIR="${SLURM_TMPDIR:-$SCRATCH/tmp}"
for input in containers/diffusionopsd.sif data/video_motion/train.txt \
    "$VIDEO_REWARD_CKPT_PATH/Wan2.2-TI2V-5B-Diffusers/model_index.json" \
    "$VIDEO_REWARD_CKPT_PATH/waft_tar_c_t.pth"; do
    [[ -f "$input" ]] || { echo "Missing input: $input" >&2; exit 1; }
done
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$TMPDIR" "$XDG_CACHE_HOME"
set +u
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load apptainer
set -u
apptainer exec --nv --bind "$REPO" --bind "$SCRATCH" \
    --bind "$REPO/src:/opt/venv/lib/python3.12/site-packages/diffusionopsd" --pwd "$REPO" \
    --env PYTHONNOUSERSITE=1 --env WANDB_MODE=offline \
    containers/diffusionopsd.sif python scripts/calibrate_wan_motion.py \
    --config wan22_ti2v --num-prompts 50 --output-dir "$REPO/logs/wan22-motion-calibration-$SLURM_JOB_ID"
