"""Repository-relative artifact paths for movable Wan training directories."""

from __future__ import annotations

import json
import os
from pathlib import Path


def resolve_repo_path(value: str | Path, root: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def portable_path(value: str | Path, root: Path) -> str:
    path = resolve_repo_path(value, root)
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        # Explicit artifacts outside the copied repository remain external.
        return str(path)


def portable_reference_files(refs: dict, root: Path) -> dict:
    return {index: portable_path(value, root) for index, value in refs.items()}


def resolve_reference_files(refs: dict, root: Path, source_root: Path | None = None) -> dict:
    resolved = {}
    for index, value in refs.items():
        path = Path(value)
        if path.is_absolute() and path.parent.name == "calibration" and path.parent.parent.parent.name == "logs":
            # Legacy checkpoints stored absolute paths. Prefer the relocated copy
            # even when the original repository still exists.
            copied = root / path.relative_to(path.parents[3])
            if copied.is_file():
                path = copied
        path = resolve_repo_path(path, root)
        if not path.is_file() and not Path(value).is_absolute() and source_root is not None:
            # Explicit external checkpoints may still use their source repository.
            path = resolve_repo_path(value, source_root)
        if not path.is_file():
            raise FileNotFoundError(f"Missing calibration reference: {path}")
        resolved[index] = str(path)
    return resolved


def normalize_video_reward_config(root: Path) -> bool:
    config_path = root / "video_reward_ckpts/VideoReward/model_config.json"
    qwen = root / "video_reward_ckpts/Qwen2-VL-2B-Instruct"
    if not config_path.is_file() or not qwen.is_dir():
        return False
    config = json.loads(config_path.read_text())
    block = config.get("model_config")
    if not isinstance(block, dict) or "model_name_or_path" not in block:
        block = config
    if "model_name_or_path" not in block:
        raise ValueError(f"{config_path} has no model_name_or_path")
    desired = str(qwen.relative_to(root))
    if block["model_name_or_path"] == desired and block.get("model_revision") is None:
        return False
    block["model_name_or_path"] = desired
    block["model_revision"] = None
    temporary = config_path.with_name(config_path.name + f".{os.getpid()}.portable.tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(config_path)
    return True


if __name__ == "__main__":
    normalize_video_reward_config(Path(__file__).resolve().parents[1])
