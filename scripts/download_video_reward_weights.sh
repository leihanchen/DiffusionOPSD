#!/usr/bin/env bash
# Workstation helper. On Compute Canada, use scripts/prefetch_wan22_login.sh
# instead. This script installs into the host Python and does not fetch
# Qwen2-VL-2B-Instruct, which VideoReward loads on an offline node.
set -euo pipefail
TARGET=${VIDEO_REWARD_CKPT_PATH:-"$(pwd)/video_reward_ckpts"}
mkdir -p "$TARGET"; export VIDEO_REWARD_CKPT_PATH="$TARGET"
command -v hf >/dev/null 2>&1 || { echo "Missing 'hf' CLI" >&2; exit 2; }

echo "Depth Anything 3 Large v1.1"
hf download depth-anything/DA3-LARGE-1.1 --local-dir "$TARGET/depth-anything-3-large-v1.1"
pip install -q "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"

echo "WAFT"
[ -d "$TARGET/WAFT" ] || git clone -q https://github.com/princeton-vl/WAFT.git "$TARGET/WAFT"
# Official a1 model zoo: https://drive.google.com/file/d/1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC
if [[ ! -s "$TARGET/waft_tar_c_t.pth" ]]; then
  curl -L --fail --retry 5 --retry-delay 2 \
    -o "$TARGET/waft_tar_c_t.pth.partial" \
    "https://drive.usercontent.google.com/download?id=1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC&export=download&confirm=t"
  mv "$TARGET/waft_tar_c_t.pth.partial" "$TARGET/waft_tar_c_t.pth"
fi

echo "DINOv2 (torch.hub cache warm-up)"
TORCH_HOME="$TARGET" python -c "import torch, os; torch.hub.set_dir(os.path.join('$TARGET','torch_hub')); torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"

echo "VideoReward (KwaiVGI) and the VideoAlign loader"
# model_config.json and checkpoint-*/model.pth must stay in $TARGET/VideoReward (the loader reads that directory).
[ -d "$TARGET/VideoAlign" ] || git clone -q --depth 1 https://github.com/KwaiVGI/VideoAlign.git "$TARGET/VideoAlign"
hf download KwaiVGI/VideoReward --local-dir "$TARGET/VideoReward"
pip install -q peft pandas 'qwen-vl-utils>=0.0.8'

echo "Wan 2.1 T2V-1.3B (diffusers layout)"
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir "$TARGET/Wan2.1-T2V-1.3B-Diffusers"

echo "Wan 2.2 TI2V-5B (diffusers layout)"
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir "$TARGET/Wan2.2-TI2V-5B-Diffusers"

echo "export VIDEO_REWARD_CKPT_PATH='$TARGET'"
