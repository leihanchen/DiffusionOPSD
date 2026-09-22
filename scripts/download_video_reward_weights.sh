#!/usr/bin/env bash
set -euo pipefail
TARGET=${VIDEO_REWARD_CKPT_PATH:-"$(pwd)/video_reward_ckpts"}
mkdir -p "$TARGET"; export VIDEO_REWARD_CKPT_PATH="$TARGET"
command -v hf >/dev/null 2>&1 || { echo "Missing 'hf' CLI" >&2; exit 2; }

echo "Depth Anything 3 Large v1.1"
hf download depth-anything/DA3LARGE-1.1 --local-dir "$TARGET/depth-anything-3-large-v1.1"
pip install -q "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"

echo "WAFT"
[ -d "$TARGET/WAFT" ] || git clone -q https://github.com/princeton-vl/WAFT.git "$TARGET/WAFT"
hf download princeton-vl/WAFT waft_tar_c_t.pth --local-dir "$TARGET"

echo "DINOv2 (torch.hub cache warm-up)"
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"

echo "Qwen2.5-VL-7B-Instruct"
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "$TARGET/Qwen2.5-VL-7B-Instruct"

echo "Wan 2.1 T2V-1.3B (diffusers layout)"
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir "$TARGET/Wan2.1-T2V-1.3B-Diffusers"

echo "export VIDEO_REWARD_CKPT_PATH='$TARGET'"
