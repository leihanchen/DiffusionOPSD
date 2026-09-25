#!/usr/bin/env bash
# Stage every file the Wan 2.2 + VideoReward job reads, then stop.
#
# Run this on a Compute Canada / TamIA LOGIN node, from the repo you will
# submit:
#
#   bash scripts/prefetch_wan22_login.sh
#
# The matching job is scripts/train_opsd_video_wan.slurm. That job runs on a
# compute node with no outbound network, so this script is the only place that
# contacts Hugging Face or GitHub. Downloads land in this repo, next to the
# code the job runs:
#   video_reward_ckpts/   model and loader files
#   hf/                   Hugging Face cache created by those downloads
# The Apptainer image stays at $SCRATCH/diffusionopsd.sif. It is built, not
# downloaded, and the job still reads it from there.
#
# What this training run consumes:
#   Wan-AI/Wan2.2-TI2V-5B-Diffusers
#   KwaiVGI/VideoReward          (model_config.json + checkpoint-*/model.pth)
#   KwaiVGI/VideoAlign           (loader; patched below to use SDPA)
#   Qwen/Qwen2-VL-2B-Instruct    (VideoReward's base; the saved config names this hub id)
#   depth-anything/DA3-LARGE-1.1
#   princeton-vl/WAFT            (repo + official a1 tar-c-t.pth)
#   facebook/dinov2-base         (Transformers snapshot, loaded offline)
#   data/video_motion/train.txt and test.txt
# The prompt files ship in this repo. There is no video-clip dataset to download.
#
# The container has no flash-attn wheel. After cloning VideoAlign, this script
# sets inference.py's disable_flash_attn2 so the judge uses PyTorch SDPA.
# It also rewrites VideoReward's model_name_or_path to the local Qwen directory
# so the offline node never resolves Qwen/Qwen2-VL-2B-Instruct on the hub.
#
# Build the image once on this login node before the first prefetch:
#   module load apptainer
#   export APPTAINER_CACHEDIR=$SCRATCH/apptainer/cache
#   export APPTAINER_TMPDIR=$SCRATCH/apptainer/tmp
#   mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
#   apptainer build $SCRATCH/diffusionopsd.sif containers/diffusionopsd.def
#
# If a download fails with a proxy error, run `module load httpproxy` in this
# login shell and rerun. Do not load httpproxy in the SLURM job.
set -euo pipefail

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
elif [[ -n "${1:-}" ]]; then
  echo "Usage: bash scripts/prefetch_wan22_login.sh [--check-only]" >&2
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${REPO}/video_reward_ckpts"
HF_HOME="${REPO}/hf"
QWEN="${TARGET}/Qwen2-VL-2B-Instruct"

if [[ -n "${VIDEO_REWARD_CKPT_PATH:-}" && "${VIDEO_REWARD_CKPT_PATH}" != "${TARGET}" ]]; then
  echo "The SLURM job reads ${TARGET}." >&2
  echo "Unset VIDEO_REWARD_CKPT_PATH or set it to that path." >&2
  exit 1
fi
export VIDEO_REWARD_CKPT_PATH="${TARGET}"
export HF_HOME

# The image is built, not downloaded. Prefer the scratch SIF from
# containers/diffusionopsd.def, then a copy sitting in this repo.
SIF=""
if [[ -n "${SCRATCH:-}" && ! -e "${SCRATCH}/diffusionopsd.sif" && -e "${SCRATCH}/DiffusionOPSD/containers/diffusionopsd.sif" ]]; then
  ln -sfn "${SCRATCH}/DiffusionOPSD/containers/diffusionopsd.sif" "${SCRATCH}/diffusionopsd.sif"
fi
# Prefer the image next to the repo. $SCRATCH/diffusionopsd.sif is only a
# convenience symlink, and it disappeared mid-download on this login node.
if [[ -s "${REPO}/containers/diffusionopsd.sif" ]]; then
  SIF="${REPO}/containers/diffusionopsd.sif"
elif [[ -n "${SCRATCH:-}" && -s "${SCRATCH}/diffusionopsd.sif" ]]; then
  SIF="${SCRATCH}/diffusionopsd.sif"
elif [[ -s "${REPO}/diffusionopsd.sif" ]]; then
  SIF="${REPO}/diffusionopsd.sif"
else
  SIF="${SCRATCH:-${REPO}}/diffusionopsd.sif"
fi

mkdir -p "${TARGET}/torch_hub/checkpoints" "${HF_HOME}" "${REPO}/logs"

# Bind the repo. Also bind scratch when the image lives there and the repo
# is not already inside it. Binding both when one contains the other makes
# Apptainer reject the mount.
apptainer_binds=()
if [[ -n "${SCRATCH:-}" && "${REPO}" != "${SCRATCH}" && "${REPO}" != "${SCRATCH}"/* ]]; then
  apptainer_binds+=(--bind "${REPO}" --bind "${SCRATCH}")
elif [[ -n "${SCRATCH:-}" && "${SIF}" == "${SCRATCH}"/* ]]; then
  apptainer_binds+=(--bind "${SCRATCH}")
else
  apptainer_binds+=(--bind "${REPO}")
fi

# TamIA login nodes cap this user cgroup at 4 GiB and SIGKILL anything over it.
# hf download defaults to 8 workers, and hf_xet (installed in the image) keeps
# large reconstruction buffers. One HTTP worker stays under that cap. Partial
# .incomplete files from the killed run are still resumed.
hf_download() {
  apptainer exec "${apptainer_binds[@]}" \
    --env "HF_HOME=${HF_HOME}" \
    --env HF_HUB_OFFLINE=0 \
    --env TRANSFORMERS_OFFLINE=0 \
    --env HF_DATASETS_OFFLINE=0 \
    --env HF_HUB_DISABLE_TELEMETRY=1 \
    --env HF_HUB_DISABLE_XET=1 \
    "${SIF}" hf download --max-workers 1 "$@"
}

clone_repo() {
  local url="$1" dest="$2" sentinel="$3"
  if [[ -f "${dest}/${sentinel}" ]]; then
    return 0
  fi
  rm -rf "${dest}"
  GIT_TERMINAL_PROMPT=0 git clone --depth 1 "${url}" "${dest}"
}

patch_local_configs() {
  python3 - "${TARGET}" "${QWEN}" <<'PY'
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
qwen = Path(sys.argv[2])
cfg_path = target / "VideoReward" / "model_config.json"
if cfg_path.is_file() and qwen.is_dir():
    cfg = json.loads(cfg_path.read_text())
    block = cfg.get("model_config")
    if isinstance(block, dict) and "model_name_or_path" in block:
        block["model_name_or_path"] = str(qwen)
        # A local directory has no hub revision. "main" makes from_pretrained
        # look up the repo id again.
        block["model_revision"] = None
    elif "model_name_or_path" in cfg:
        cfg["model_name_or_path"] = str(qwen)
        cfg["model_revision"] = None
    else:
        raise SystemExit(f"{cfg_path} has no model_name_or_path")
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"VideoReward base model -> {qwen}")

infer = target / "VideoAlign" / "inference.py"
if infer.is_file():
    text = infer.read_text()
    old = "disable_flash_attn2=False,"
    new = "disable_flash_attn2=True,  # SDPA: diffusionopsd.sif has no flash-attn"
    if old in text:
        infer.write_text(text.replace(old, new, 1))
        print(f"VideoAlign attention -> SDPA ({infer})")
    elif "disable_flash_attn2=True" not in text:
        raise SystemExit(
            "VideoAlign inference.py has no disable_flash_attn2 flag; "
            "the clone does not match the loader this job expects."
        )
PY
}

require_file() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    return 0
  fi
  if [[ ! -s "${path}" ]]; then
    MISSING+=("${path}")
  fi
}

require_weights() {
  local dir="$1"
  local found
  if [[ ! -d "${dir}" ]]; then
    MISSING+=("${dir}")
    return
  fi
  found="$(find "${dir}" \( -type f -o -type l \) \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pth' -o -name '*.pt' \) -print -quit)"
  if [[ -z "${found}" || ! -s "${found}" ]]; then
    MISSING+=("${dir} (no weight file)")
  fi
}

verify_offline_tree() {
  MISSING=()
  if [[ ! -s "${SIF}" ]]; then
    MISSING+=("${SIF}")
  fi
  require_file "${TARGET}/Wan2.2-TI2V-5B-Diffusers/model_index.json"
  require_weights "${TARGET}/Wan2.2-TI2V-5B-Diffusers/transformer"
  require_weights "${TARGET}/Wan2.2-TI2V-5B-Diffusers/vae"
  require_weights "${TARGET}/Wan2.2-TI2V-5B-Diffusers/text_encoder"
  require_file "${TARGET}/Wan2.2-TI2V-5B-Diffusers/tokenizer"
  require_file "${TARGET}/VideoReward/model_config.json"
  require_weights "${TARGET}/VideoReward"
  require_file "${TARGET}/VideoAlign/inference.py"
  require_file "${QWEN}/config.json"
  require_weights "${QWEN}"
  require_file "${TARGET}/depth-anything-3-large-v1.1/config.json"
  require_weights "${TARGET}/depth-anything-3-large-v1.1"
  require_file "${TARGET}/WAFT/config/tar-c-t.json"
  require_file "${TARGET}/WAFT/model/vitwarp_v8.py"
  require_file "${TARGET}/WAFT/depth-anything-ckpts/depth_anything_v2_vits.pth"
  require_file "${TARGET}/waft_tar_c_t.pth"
  require_file "${TARGET}/dinov2-base/config.json"
  require_weights "${TARGET}/dinov2-base"
  require_file "${REPO}/data/video_motion/train.txt"
  require_file "${REPO}/data/video_motion/test.txt"

  if [[ -f "${TARGET}/VideoReward/model_config.json" ]] \
    && grep -q 'Qwen/Qwen2-VL-2B-Instruct' "${TARGET}/VideoReward/model_config.json"; then
    MISSING+=("${TARGET}/VideoReward/model_config.json still names Qwen/Qwen2-VL-2B-Instruct")
  fi
  if [[ -f "${TARGET}/VideoReward/model_config.json" && -d "${QWEN}" ]] \
    && ! grep -q "\"model_name_or_path\": \"${QWEN}\"" "${TARGET}/VideoReward/model_config.json"; then
    MISSING+=("${TARGET}/VideoReward/model_config.json does not point at ${QWEN}")
  fi
  if [[ -f "${TARGET}/VideoAlign/inference.py" ]] \
    && ! grep -q 'disable_flash_attn2=True' "${TARGET}/VideoAlign/inference.py"; then
    MISSING+=("${TARGET}/VideoAlign/inference.py still requests flash_attention_2")
  fi

  if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "Wan 2.2 VideoReward offline tree is incomplete:" >&2
    local item
    for item in "${MISSING[@]}"; do
      echo "  ${item}" >&2
    done
    echo "On a login node, from this repo: bash scripts/prefetch_wan22_login.sh" >&2
    exit 1
  fi
}

if [[ "${CHECK_ONLY}" -eq 0 ]]; then
  if [[ ! -s "${SIF}" ]]; then
    echo "Missing ${SIF}." >&2
    echo "Build it on this login node, then rerun this script:" >&2
    echo "  module load apptainer" >&2
    echo "  export APPTAINER_CACHEDIR=\${SCRATCH}/apptainer/cache" >&2
    echo "  export APPTAINER_TMPDIR=\${SCRATCH}/apptainer/tmp" >&2
    echo "  mkdir -p \"\$APPTAINER_CACHEDIR\" \"\$APPTAINER_TMPDIR\"" >&2
    echo "  cd ${REPO}" >&2
    echo "  apptainer build ${SIF} containers/diffusionopsd.def" >&2
    exit 1
  fi
  if [[ -z "${SCRATCH:-}" ]]; then
    echo "SCRATCH is unset. Build and run Apptainer from a Compute Canada login node." >&2
    exit 1
  fi
  if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  fi
  module load apptainer
  export APPTAINER_CACHEDIR="${SCRATCH}/apptainer/cache"
  export APPTAINER_TMPDIR="${SCRATCH}/apptainer/tmp"
  mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
  # A login shell may have inherited the compute-node offline flags.
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true

  # cgroup v2 charges written page cache to this 4 GiB login slice. A multi-GB
  # shard will OOM unless those pages are dropped while the download runs.
  python3 - "${TARGET}" "${HF_HOME}" "$$" <<'PY' &
import ctypes, ctypes.util, os, pathlib, sys, time
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
DONTNEED = 4
roots = [pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])]
parent = int(sys.argv[3])

def drop():
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                if path.stat().st_size < 8 * 1024 * 1024 and not path.name.endswith(".incomplete"):
                    continue
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                libc.posix_fadvise(fd, 0, 0, DONTNEED)
            finally:
                os.close(fd)

while True:
    try:
        os.kill(parent, 0)
    except OSError:
        break
    drop()
    time.sleep(2)
PY
  CACHE_DROPPER_PID=$!
  trap 'kill "${CACHE_DROPPER_PID}" 2>/dev/null || true' EXIT

  echo "Downloading Wan2.2-TI2V-5B-Diffusers"
  hf_download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir "${TARGET}/Wan2.2-TI2V-5B-Diffusers"

  echo "Downloading VideoReward"
  hf_download KwaiVGI/VideoReward --local-dir "${TARGET}/VideoReward"

  echo "Downloading Qwen2-VL-2B-Instruct (VideoReward base)"
  hf_download Qwen/Qwen2-VL-2B-Instruct --local-dir "${QWEN}"

  echo "Downloading Depth Anything 3 Large v1.1"
  hf_download depth-anything/DA3-LARGE-1.1 --local-dir "${TARGET}/depth-anything-3-large-v1.1"

  echo "Downloading WAFT checkpoint (official a1 tar-c-t.pth)"
  WAFT_CKPT="${TARGET}/waft_tar_c_t.pth"
  if [[ ! -s "${WAFT_CKPT}" ]]; then
    python3 - "${WAFT_CKPT}" <<'PY'
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

dest = Path(sys.argv[1])
partial = dest.with_name(dest.name + ".partial")
page = "https://drive.usercontent.google.com/download?id=1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC&export=download"
req = urllib.request.Request(page, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=60) as response:
    html = response.read().decode("utf-8", "replace")
action = re.search(r'action="([^"]+)"', html).group(1)
fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', html))
url = action + "?" + urllib.parse.urlencode(fields)
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=60) as response, partial.open("wb") as out:
    if "text/html" in (response.headers.get("Content-Type") or ""):
        raise SystemExit("Google Drive returned HTML instead of tar-c-t.pth")
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            break
        out.write(chunk)
partial.replace(dest)
print(dest, dest.stat().st_size)
PY
  fi

  echo "Cloning VideoAlign and WAFT"
  clone_repo https://github.com/KwaiVGI/VideoAlign.git "${TARGET}/VideoAlign" inference.py
  # a1 commit matches waft_tar_c_t.pth. The default branch is waftv2 and has a different API.
  if [[ ! -f "${TARGET}/WAFT/model/vitwarp_v8.py" ]]; then
    rm -rf "${TARGET}/WAFT"
    git clone --filter=blob:none --no-checkout https://github.com/princeton-vl/WAFT.git "${TARGET}/WAFT"
    git -C "${TARGET}/WAFT" fetch --depth 1 origin 8dd41723f5
    git -C "${TARGET}/WAFT" checkout --detach 8dd41723f5
  fi

  DA2_CKPT="${TARGET}/WAFT/depth-anything-ckpts/depth_anything_v2_vits.pth"
  if [[ ! -s "${DA2_CKPT}" ]]; then
    echo "Downloading Depth Anything V2 Small (WAFT backbone)"
    mkdir -p "${TARGET}/WAFT/depth-anything-ckpts"
    curl -L --fail --retry 5 --retry-delay 2 -o "${DA2_CKPT}.partial" \
      https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
    mv "${DA2_CKPT}.partial" "${DA2_CKPT}"
  fi

  echo "Downloading DINOv2 base (Transformers snapshot)"
  hf_download facebook/dinov2-base --local-dir "${TARGET}/dinov2-base"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to point VideoReward at the local Qwen checkout." >&2
  exit 1
fi
patch_local_configs
verify_offline_tree

echo "Offline tree is ready under ${TARGET}"
echo "Hugging Face cache: ${HF_HOME}"
echo "Prompt dataset: ${REPO}/data/video_motion/train.txt and test.txt"
echo "Submit from ${REPO}: sbatch scripts/train_opsd_video_wan.slurm"
