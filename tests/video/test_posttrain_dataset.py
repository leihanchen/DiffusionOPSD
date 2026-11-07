"""Catch source-membership loss, caption rewriting and unsafe publication artifacts."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_video_posttrain_dataset.py"


@pytest.fixture
def dataset_script():
    assert SCRIPT.is_file(), "dataset preparation script has not been implemented"
    spec = importlib.util.spec_from_file_location("prepare_video_posttrain_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sources(tmp_path):
    """Two OpenVid subsets overlap once; identical captions on other videos must survive."""
    inputs = [
        ("videogpa", "captions.json", '{"8K/scene/images_8": "  A scene.\\n"}', 1),
        ("videorepa_32k", "small.csv", 'video,caption,seconds\na.mp4,"A scene.",5\nb.mp4,Shared,4\n', 2),
        ("videorepa_64k", "large.csv", 'video,caption,seconds\nb.mp4,Shared,4\nc.mp4,A scene.,6\n', 2),
    ]
    recipe = {"format_version": 1, "sources": [], "expected_rows": {
        "videogpa": 1, "videorepa_32k": 2, "videorepa_64k": 2, "combined": 4,
    }, "expected_shared_videorepa_rows": 1}
    for config, name, content, count in inputs:
        data = content.encode()
        (tmp_path / name).write_bytes(data)
        recipe["sources"].append({
            "config": config, "path": name, "url": "https://example.invalid/" + name,
            "sha256": hashlib.sha256(data).hexdigest(), "rows": count,
        })
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    return path, tmp_path


def test_preserves_exact_membership_text_and_distinct_video_paths(dataset_script, sources):
    recipe_path, cache = sources
    configs = dataset_script.collect(json.loads(recipe_path.read_text()), cache, offline=True)
    assert configs["videogpa"] == [{"text": "  A scene.\n", "video_source": "DL3DV/8K/scene/images_8"}]
    assert configs["videorepa_32k"] == [
        {"text": "A scene.", "video_source": "OpenVid/a.mp4"},
        {"text": "Shared", "video_source": "OpenVid/b.mp4"},
    ]
    assert configs["videorepa_64k"] == [
        {"text": "Shared", "video_source": "OpenVid/b.mp4"},
        {"text": "A scene.", "video_source": "OpenVid/c.mp4"},
    ]
    assert configs["combined"] == [
        {"text": "  A scene.\n", "video_source": "DL3DV/8K/scene/images_8"},
        {"text": "A scene.", "video_source": "OpenVid/a.mp4"},
        {"text": "Shared", "video_source": "OpenVid/b.mp4"},
        {"text": "A scene.", "video_source": "OpenVid/c.mp4"},
    ]


@pytest.mark.parametrize("path", ["../a.mp4", "/a.mp4", "a/../../b", "C:/a.mp4", "a\\b", "a//b", "./a"])
def test_rejects_noncanonical_or_escaping_source_paths(dataset_script, path):
    with pytest.raises(ValueError, match="relative"):
        dataset_script.relative_source("OpenVid", path)


@pytest.mark.parametrize("payload", ['{"8K/scene/images_8": 7}', '{"8K/scene/images_8": " "}'])
def test_rejects_invalid_caption_schema(dataset_script, payload):
    with pytest.raises(ValueError, match="caption"):
        dataset_script.parse_source("videogpa", payload.encode())


def test_rejects_csv_missing_required_column(dataset_script):
    with pytest.raises(ValueError, match="columns"):
        dataset_script.parse_source("videorepa_32k", b"video,seconds\na.mp4,3\n")


def test_tampered_cache_fails_before_export(dataset_script, sources, tmp_path):
    recipe, cache = sources
    (cache / "small.csv").write_text("corrupted")
    with pytest.raises(ValueError, match="SHA-256"):
        dataset_script.materialize(recipe, tmp_path / "release", cache, offline=True)
    assert not (tmp_path / "release").exists()


def test_offline_missing_source_does_not_download(dataset_script, sources):
    recipe, cache = sources
    (cache / "small.csv").unlink()
    with pytest.raises(FileNotFoundError, match="offline"):
        dataset_script.collect(json.loads(recipe.read_text()), cache, offline=True)


def test_parquet_roundtrip_configs_hashes_and_offline_rebuild(dataset_script, sources, tmp_path):
    from datasets import load_dataset

    recipe, cache = sources
    output = tmp_path / "release"
    report = dataset_script.materialize(recipe, output, cache, offline=True)
    second = dataset_script.materialize(recipe, tmp_path / "rebuilt", cache, offline=True)
    assert second == report
    assert report["shared_videorepa_rows"] == 1
    expected_counts = {"videogpa": 1, "videorepa_32k": 2, "videorepa_64k": 2, "combined": 4}
    for config, count in expected_counts.items():
        data = load_dataset(str(output), config, split="train")
        assert len(data) == count
        assert data.features.to_dict() == {
            "text": {"dtype": "string", "_type": "Value"},
            "video_source": {"dtype": "string", "_type": "Value"},
        }
        assert data.to_list() == dataset_script.collect(json.loads(recipe.read_text()), cache, offline=True)[config]
    assert load_dataset(str(output), split="train").to_list() == load_dataset(
        str(output), "combined", split="train"
    ).to_list()
    parquet = output / "data" / "combined.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        dataset_script.verify_release(output)


def test_refuses_to_overwrite_unrelated_hub_data(dataset_script, sources, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub

    recipe, cache = sources
    output = tmp_path / "release"
    dataset_script.materialize(recipe, output, cache, offline=True)

    class ExistingRepoApi:
        def whoami(self):
            return {"name": "tester", "orgs": [{"name": "cvis-tmu"}]}

        def create_repo(self, *args, **kwargs):
            pass

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="existing", siblings=[SimpleNamespace(rfilename="other.parquet")])

        def create_commit(self, *args, **kwargs):
            pytest.fail("Unrelated Hub data must be rejected before committing")

    monkeypatch.setattr(huggingface_hub, "HfApi", ExistingRepoApi)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        dataset_script.publish(output, "cvis-tmu/VideoGPA-VideoREPA")


def test_preserves_valid_release_when_new_source_validation_fails(dataset_script, sources, tmp_path):
    recipe, cache = sources
    output = tmp_path / "release"
    before = dataset_script.materialize(recipe, output, cache, offline=True)
    (cache / "large.csv").write_text("interrupted download")
    with pytest.raises(ValueError, match="SHA-256"):
        dataset_script.materialize(recipe, output, cache, offline=True)
    assert dataset_script.verify_release(output) == before
