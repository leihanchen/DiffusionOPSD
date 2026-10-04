#!/bin/bash
# From repository root: mkdir -p logs && sbatch scripts/ablate_wan_inference.slurm
#SBATCH --job-name=wan22-ablation
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
# Let the cluster select a partition compatible with this shorter allocation.
# Sequential inference: policy/text/VAE share GPU 0; scorers use GPU 1.
# No backward pass or optimizer state requires a dedicated VAE GPU.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_MAX_THREADS="$OMP_NUM_THREADS"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
REPO="$PWD"
SIF="${REPO}/containers/diffusionopsd.sif"
OUTPUT="${REPO}/logs/wan22-ablation-${SLURM_JOB_ID}"
export HF_HOME="${REPO}/hf"
export TORCH_HOME="${REPO}/video_reward_ckpts"
export VIDEO_REWARD_CKPT_PATH="${REPO}/video_reward_ckpts"
export XDG_CACHE_HOME="${REPO}/.cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline PYTHONNOUSERSITE=1
# Do not pre-create OUTPUT: the Python command owns its empty-directory check.
export WANDB_DIR="${OUTPUT}"
export APPTAINER_CACHEDIR="${SCRATCH}/apptainer/cache"
export APPTAINER_TMPDIR="${SCRATCH}/apptainer/tmp"
export TMPDIR="${SLURM_TMPDIR:-${SCRATCH}/tmp}"
[[ -f "$SIF" && ! -L "$SIF" ]] || { echo "Missing regular container: $SIF" >&2; exit 1; }
for input in "config/wan22_inference_ablation.json" "data/video_motion/test.txt" \
  "${VIDEO_REWARD_CKPT_PATH}/Wan2.2-TI2V-5B-Diffusers/model_index.json" \
  "${VIDEO_REWARD_CKPT_PATH}/VideoReward/model_config.json" \
  "${VIDEO_REWARD_CKPT_PATH}/waft_tar_c_t.pth"; do
  [[ -f "$input" ]] || { echo "Missing local input: $input" >&2; exit 1; }
done
for model in depth-anything-3-large-v1.1 VideoAlign VideoReward Qwen2-VL-2B-Instruct dinov2-base WAFT; do
  [[ -d "${VIDEO_REWARD_CKPT_PATH}/${model}" ]] || { echo "Missing scorer directory: $model" >&2; exit 1; }
done
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$TMPDIR" "$XDG_CACHE_HOME"
set +u
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load apptainer
set -u
apptainer exec --nv --bind "$REPO" --bind "$SCRATCH" \
  --bind "$REPO/src:/opt/venv/lib/python3.12/site-packages/diffusionopsd" --pwd "$REPO" \
  --env PYTHONNOUSERSITE=1 --env WANDB_MODE=offline \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 \
  "$SIF" python scripts/ablate_wan_inference.py \
    --config config/wan22_inference_ablation.json \
    --output-dir "$OUTPUT" \
    --policy-device cuda:0 --reward-device cuda:1 --vae-device cuda:0
