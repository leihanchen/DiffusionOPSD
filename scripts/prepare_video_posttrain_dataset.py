#!/usr/bin/env python3
"""Collect exact VideoGPA/VideoREPA selections and publish caption/source-path records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = ROOT / "data/video_posttrain_sources.json"
CARD_PATH = ROOT / "data/video_posttrain/README.md"
SOURCE_CONFIGS = ("videogpa", "videorepa_32k", "videorepa_64k")
CONFIGS = (*SOURCE_CONFIGS, "combined")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def row_hash(rows: list[dict]) -> str:
    """Hash ordered UTF-8 records independently of Parquet/library serialization."""
    return sha256("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode())


def relative_source(prefix: str, path: str) -> str:
    if (
        not isinstance(path, str) or not path or "\\" in path or ":" in path
        or any(ord(c) < 32 for c in path) or PurePosixPath(path).is_absolute()
        or any(part in ("", ".", "..") for part in path.split("/"))
    ):
        raise ValueError(f"Expected canonical relative source path, got {path!r}")
    return f"{prefix}/{path}"


def parse_source(config: str, data: bytes) -> list[dict]:
    if config == "videogpa":
        mapping = json.loads(data)
        if not isinstance(mapping, dict):
            raise ValueError("VideoGPA caption source must be a JSON object")
        pairs = mapping.items()
        prefix = "DL3DV"
    elif config in ("videorepa_32k", "videorepa_64k"):
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        if reader.fieldnames != ["video", "caption", "seconds"]:
            raise ValueError(f"Unexpected VideoREPA columns: {reader.fieldnames!r}")
        pairs = [(row["video"], row["caption"]) for row in reader]
        prefix = "OpenVid"
    else:
        raise ValueError(f"Unknown source configuration {config!r}")
    rows = []
    for path, text in pairs:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid caption for {path!r}")
        rows.append({"text": text, "video_source": relative_source(prefix, path)})
    return rows


def source_bytes(source: dict, cache_dir: Path, *, offline: bool) -> bytes:
    target = cache_dir / Path(source["path"]).name
    if target.is_file():
        data = target.read_bytes()
    else:
        if offline:
            raise FileNotFoundError(f"Missing offline source {target}; rerun without --offline")
        with urllib.request.urlopen(source["url"], timeout=60) as response:
            data = response.read()
    actual = sha256(data)
    if actual != source["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {target}: {actual}; remove corrupt cache and retry")
    if not target.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, target)
    return data


def collect(recipe: dict, cache_dir: Path, *, offline: bool = False) -> dict[str, list[dict]]:
    if recipe.get("format_version") != 1:
        raise ValueError("Unsupported source recipe format_version")
    configs = {name: [] for name in SOURCE_CONFIGS}
    for source in recipe["sources"]:
        rows = parse_source(source["config"], source_bytes(source, cache_dir, offline=offline))
        if len(rows) != source["rows"]:
            raise ValueError(f"Source row count mismatch for {source['path']}")
        configs[source["config"]].extend(rows)
    combined = {}
    for rows in configs.values():
        for row in rows:
            combined.setdefault((row["text"], row["video_source"]), row)
    configs["combined"] = list(combined.values())
    for name, rows in configs.items():
        if len(rows) != recipe["expected_rows"][name]:
            raise ValueError(f"Configuration {name} row count mismatch: {len(rows)}")
    shared = len(
        {(r["text"], r["video_source"]) for r in configs["videorepa_32k"]}
        & {(r["text"], r["video_source"]) for r in configs["videorepa_64k"]}
    )
    if shared != recipe["expected_shared_videorepa_rows"]:
        raise ValueError(f"VideoREPA overlap mismatch: {shared}")
    return configs


def collect_test(recipe: dict, configs: dict, cache_dir: Path, *, offline: bool) -> list[dict]:
    source = recipe.get("test_source")
    if source is None:
        return []
    rows = parse_source("videogpa", source_bytes(source, cache_dir, offline=offline))
    if len(rows) != source["rows"]:
        raise ValueError(f"Test source row count mismatch: {len(rows)}")
    train_text = {" ".join(row["text"].split()).casefold() for row in configs["combined"]}
    test_text = {" ".join(row["text"].split()).casefold() for row in rows}
    train_scenes = {row["video_source"].split("/")[2] for row in configs["videogpa"]}
    test_scenes = {row["video_source"].split("/")[2] for row in rows}
    if train_text & test_text or train_scenes & test_scenes:
        raise ValueError("Training/test caption or DL3DV scene overlap")
    return rows


def materialize(recipe_path: Path, output: Path, cache_dir: Path, *, offline: bool = False) -> dict:
    from datasets import Dataset, Features, Value

    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    configs = collect(recipe, cache_dir, offline=offline)
    test_rows = collect_test(recipe, configs, cache_dir, offline=offline)
    features = Features({"text": Value("string"), "video_source": Value("string")})
    output = output.resolve()
    # An existing release is immutable. A caller can choose a new output directory to rebuild.
    if output.exists():
        report = verify_release(output)
        if report["recipe_sha256"] != sha256(recipe_bytes):
            raise ValueError(f"Existing {output} uses another recipe; choose a new --output")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".video-posttrain-", dir=output.parent) as directory:
        stage = Path(directory) / "release"
        (stage / "data").mkdir(parents=True)
        report = {
            "format_version": 1,
            "recipe_sha256": sha256(recipe_bytes),
            "shared_videorepa_rows": recipe["expected_shared_videorepa_rows"],
            "configurations": {},
            "files": {},
        }
        for name, rows in configs.items():
            dataset = Dataset.from_list(rows, features=features)
            dataset.to_parquet(str(stage / "data" / f"{name}.parquet"))
            report["configurations"][name] = {"rows": len(rows), "content_sha256": row_hash(rows)}
        if test_rows:
            Dataset.from_list(test_rows, features=features).to_parquet(str(stage / "data/videogpa_test.parquet"))
            report["test"] = {"rows": len(test_rows), "content_sha256": row_hash(test_rows)}
        shutil.copyfile(CARD_PATH, stage / "README.md")
        (stage / "sources.json").write_bytes(recipe_bytes)
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                report["files"][path.relative_to(stage).as_posix()] = sha256(path.read_bytes())
        (stage / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        verify_release(stage)
        os.rename(stage, output)
    return report


def verify_release(output: Path) -> dict:
    from datasets import Dataset, Features, Value

    report = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    expected_files = {"README.md", "sources.json", *(f"data/{name}.parquet" for name in CONFIGS)}
    if "test" in report:
        expected_files.add("data/videogpa_test.parquet")
    if set(report["files"]) != expected_files:
        raise ValueError("Unexpected release file inventory")
    for name, expected_hash in report["files"].items():
        if sha256((output / name).read_bytes()) != expected_hash:
            raise ValueError(f"SHA-256 mismatch for output {name}")
    features = Features({"text": Value("string"), "video_source": Value("string")})
    for name in CONFIGS:
        data = Dataset.from_parquet(str(output / "data" / f"{name}.parquet"))
        expected = report["configurations"][name]
        if data.features != features or len(data) != expected["rows"] or row_hash(data.to_list()) != expected["content_sha256"]:
            raise ValueError(f"Schema, row count or content mismatch for {name}")
    if "test" in report:
        data = Dataset.from_parquet(str(output / "data/videogpa_test.parquet"))
        expected = report["test"]
        if data.features != features or len(data) != expected["rows"] or row_hash(data.to_list()) != expected["content_sha256"]:
            raise ValueError("Schema, row count or content mismatch for test")
    return report


def verify_hub(output: Path, repo_id: str, revision: str) -> dict:
    from datasets import load_dataset

    report = verify_release(output)
    for name in CONFIGS:
        data = load_dataset(repo_id, name, split="train", revision=revision)
        expected = report["configurations"][name]
        if list(data.features) != ["text", "video_source"] or any(f.dtype != "string" for f in data.features.values()):
            raise ValueError(f"Hub schema mismatch for {name}")
        if len(data) != expected["rows"] or row_hash(data.to_list()) != expected["content_sha256"]:
            raise ValueError(f"Hub content mismatch for {name}")
    default = load_dataset(repo_id, split="train", revision=revision)
    if row_hash(default.to_list()) != report["configurations"]["combined"]["content_sha256"]:
        raise ValueError("Hub default configuration differs from combined")
    if "test" in report:
        for config in ["combined", "videogpa", None]:
            data = load_dataset(repo_id, config, split="test", revision=revision)
            if data.features != default.features or len(data) != report["test"]["rows"]:
                raise ValueError(f"Hub test schema/count mismatch for {config}")
            if row_hash(data.to_list()) != report["test"]["content_sha256"]:
                raise ValueError(f"Hub test content mismatch for {config}")
    return {"repo_id": repo_id, "revision": revision, "verified_configurations": list(CONFIGS),
            "verified_test_configurations": ["combined", "videogpa"] if "test" in report else []}


def publish(output: Path, repo_id: str, *, expected_revision: str | None = None) -> dict:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    report = verify_release(output)
    api = HfApi()
    user = api.whoami()
    namespaces = {user["name"], *(org["name"] for org in user.get("orgs", []))}
    if repo_id.split("/")[0] not in namespaces:
        raise ValueError(f"Authenticated account has no visible membership in {repo_id.split('/')[0]}")
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="dataset")
    existing = {file.rfilename for file in info.siblings} - {".gitattributes"}
    if existing:
        if not existing.issubset({*report["files"], "audit.json"}) or "audit.json" not in existing:
            raise ValueError("Target HF repository has other content; refusing to overwrite it")
        remote = Path(hf_hub_download(repo_id, "audit.json", repo_type="dataset", revision=info.sha))
        if remote.read_bytes() == (output / "audit.json").read_bytes():
            return verify_hub(output, repo_id, info.sha)
        if expected_revision is None or info.sha != expected_revision:
            raise ValueError("Different Hub release; pass --expected-hub-revision with the current SHA to update")
        previous = json.loads(remote.read_bytes())
        if previous["configurations"] != report["configurations"]:
            raise ValueError("Test-split update would alter training data; refusing to overwrite it")
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(output / name))
        for name in sorted({*report["files"], "audit.json"})
    ]
    commit = api.create_commit(
        repo_id, repo_type="dataset", operations=operations, parent_commit=info.sha,
        commit_message="Publish exact VideoGPA/VideoREPA selections and combined captions",
    )
    return verify_hub(output, repo_id, commit.oid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/video_posttrain/sources")
    parser.add_argument("--output", type=Path, default=ROOT / "data/video_posttrain/generated")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--repo-id", default="cvis-tmu/VideoGPA-VideoREPA")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--expected-hub-revision", help="Explicit parent SHA when updating an existing release")
    parser.add_argument("--verify-hub-revision", help="Reload an already published commit without uploading")
    args = parser.parse_args()
    report = materialize(args.recipe, args.output, args.cache_dir, offline=args.offline)
    print(json.dumps(report, indent=2))
    if args.push_to_hub:
        print(json.dumps(publish(args.output, args.repo_id, expected_revision=args.expected_hub_revision), indent=2))
    elif args.verify_hub_revision:
        print(json.dumps(verify_hub(args.output, args.repo_id, args.verify_hub_revision), indent=2))


if __name__ == "__main__":
    main()
