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
TORCH_HOME="$TARGET" python -c "import torch, os; torch.hub.set_dir(os.path.join('$TARGET','torch_hub')); torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"

echo "VideoReward (KwaiVGI) and the VideoAlign loader"
# model_config.json and checkpoint-*/model.pth must stay in $TARGET/VideoReward (the loader reads that directory).
[ -d "$TARGET/VideoAlign" ] || git clone -q --depth 1 https://github.com/KwaiVGI/VideoAlign.git "$TARGET/VideoAlign"
hf download KwaiVGI/VideoReward --local-dir "$TARGET/VideoReward"
pip install -q peft pandas 'qwen-vl-utils>=0.0.8'

echo "Wan 2.1 T2V-1.3B (diffusers layout)"
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir "$TARGET/Wan2.1-T2V-1.3B-Diffusers"

echo "export VIDEO_REWARD_CKPT_PATH='$TARGET'"
