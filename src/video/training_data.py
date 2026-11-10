"""Pinned video prompt records, offline HF caching, and deterministic scheduling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random


DATASET_ID = "cvis-tmu/VideoGPA-VideoREPA"
DATASET_REVISION = "b5c9967cb7801ceec0b3581839d065131eefc5be"
FIELDS = ("text", "video_source", "sample_id", "source_dataset", "source_subsets", "source_group_id", "media_type")


def default_dataset_config() -> dict:
    return dict(source="hf", repo_id=DATASET_ID, name="combined", revision=DATASET_REVISION,
                train_split="train", eval_split="test", seed=42)


def content_hash(rows: list[dict]) -> str:
    data = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass
class TrainingData:
    train: list[dict]
    evaluation: list[dict]
    metadata: dict
    train_hash: str
    eval_hash: str
    is_hf: bool = True

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def key(self, index: int) -> str:
        return self.train[index]["sample_id"] if self.is_hf else self.train[index]["text"]

    def order(self, updates: int, batch_size: int = 4) -> list[int]:
        return training_order(len(self.train), updates, self.metadata.get("seed", 42), batch_size)


def training_order(count: int, updates: int, seed: int, batch_size: int = 4) -> list[int]:
    """Cycle a seeded permutation; each batch has distinct record indices."""
    if count < batch_size or updates <= 0 or batch_size <= 0:
        raise ValueError("Training needs positive updates and at least batch_size records")
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return [order[position % count] for position in range(updates * batch_size)]


def file_records(path: str | Path, *, strip: bool = True) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [{"text": line.strip() if strip else line} for line in lines if line.strip()]


def _validate_records(rows: list[dict], expected: dict, split: str) -> None:
    ids = set()
    for row in rows:
        if set(row) != set(FIELDS) or any(not isinstance(row[key], str) or not row[key].strip()
                                        for key in FIELDS if key != "source_subsets"):
            raise ValueError(f"Invalid dataset fields in {split}")
        if not isinstance(row["source_subsets"], list) or not row["source_subsets"] or any(
            not isinstance(value, str) or not value for value in row["source_subsets"]
        ):
            raise ValueError("source_subsets must be a nonempty string list")
        if row["sample_id"] in ids:
            raise ValueError(f"Duplicate sample_id in {split}")
        ids.add(row["sample_id"])
    if not rows or len(rows) != expected["rows"] or content_hash(rows) != expected["content_sha256"]:
        raise ValueError(f"Dataset {split} content hash or count mismatch")


def load_training_data(cfg, *, offline: bool | None = None, strip_files: bool = True) -> TrainingData:
    settings = getattr(cfg, "dataset", {"source": "files"})
    if settings["source"] == "files":
        train_path = getattr(cfg, "train_prompts", None) or cfg.prompt_fn_kwargs["path"]
        eval_path = cfg.eval_prompts
        train, evaluation = file_records(train_path, strip=strip_files), file_records(eval_path, strip=strip_files)
        if not train or not evaluation:
            raise ValueError("Training and evaluation prompt files must be nonempty")
        hashes = [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (train_path, eval_path)]
        return TrainingData(train, evaluation, dict(source="files", train_hash=hashes[0], eval_hash=hashes[1]),
                            *hashes, is_hf=False)
    if settings["source"] != "hf" or settings["name"] != "combined":
        raise ValueError("Select dataset.source=hf with name=combined, or dataset.source=files")
    if (settings["train_split"], settings["eval_split"]) != ("train", "test"):
        raise ValueError("The published combined configuration requires train/test splits")
    revision = settings["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Pin dataset.revision to a complete HF commit SHA")
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    files = ("data/combined.parquet", "data/combined_test.parquet", "audit.json", "sources.json")
    if offline is None:
        offline = os.environ.get("HF_HUB_OFFLINE", "0").upper() in ("1", "TRUE", "YES", "ON")
    try:
        root = Path(snapshot_download(repo_id=settings["repo_id"], repo_type="dataset", revision=revision,
                                      allow_patterns=list(files), local_files_only=offline, max_workers=1))
        missing = [name for name in files if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Snapshot is missing {missing}")
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "Pinned video dataset cache unavailable. On a network-enabled login node run "
            "HF_HUB_OFFLINE=0 bash scripts/run_cosmos_python.sh scripts/prefetch_video_dataset.py "
            f"--repo-id {settings['repo_id']} --revision {revision} "
            "using the same HF_HOME as training."
        ) from exc
    audit = json.loads((root / "audit.json").read_text())
    for name in files:
        if name == "audit.json":
            continue
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != audit["files"][name]:
            raise ValueError(f"Pinned dataset file hash mismatch: {name}")
    data = load_dataset("parquet", data_files={"train": str(root / files[0]), "test": str(root / files[1])})
    train, evaluation = data["train"].to_list(), data["test"].to_list()
    _validate_records(train, audit["configurations"]["combined"], "train")
    _validate_records(evaluation, audit["combined_test"], "test")
    if {r["sample_id"] for r in train} & {r["sample_id"] for r in evaluation}:
        raise ValueError("Training/test sample_id overlap")
    if {r["source_group_id"] for r in train} & {r["source_group_id"] for r in evaluation}:
        raise ValueError("Training/test source_group_id overlap")
    if {" ".join(r["text"].split()).casefold() for r in train} & {
        " ".join(r["text"].split()).casefold() for r in evaluation
    }:
        raise ValueError("Training/test caption overlap")
    train_hash, eval_hash = content_hash(train), content_hash(evaluation)
    metadata = dict(source="hf", repo_id=settings["repo_id"], name=settings["name"], revision=revision,
                    train_split="train", eval_split="test", seed=int(settings["seed"]),
                    sampling_policy="seeded-permutation-cycle-v1", train_rows=len(train), eval_rows=len(evaluation),
                    train_hash=train_hash, eval_hash=eval_hash,
                    audit_sha256=hashlib.sha256((root / "audit.json").read_bytes()).hexdigest())
    return TrainingData(train, evaluation, metadata, train_hash, eval_hash)


def evaluation_records(cfg) -> list[dict]:
    if getattr(cfg, "dataset", {"source": "files"})["source"] == "hf":
        return load_training_data(cfg).evaluation
    return file_records(cfg.eval_prompts)


def validate_dataset_cli(default_config: str) -> None:
    """Dispatch before training imports, CUDA checks, or model initialization."""
    import argparse
    import importlib.util

    parser = argparse.ArgumentParser(description="Validate the training/evaluation dataset without models")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--validate-dataset-only", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("dataset_config", args.config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.get_config()
    data = load_training_data(cfg, strip_files="cosmos" not in default_config)
    if data.is_hf and default_config == "config/wan_video.py":
        data.metadata["sampling_policy"] = "seeded-random-subset-v1"
    budget = cfg.updates if hasattr(cfg, "updates") else cfg.num_epochs
    batch = 1 if hasattr(cfg, "updates") else cfg.sample.num_batches_per_epoch
    if data.is_hf:
        data.order(budget, batch)
    print(json.dumps({"dataset": data.metadata, "fingerprint": data.fingerprint,
                      "train_count": len(data.train), "test_count": len(data.evaluation),
                      "update_budget": budget}, indent=2))
