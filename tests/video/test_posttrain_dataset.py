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
    test_bytes = b'{"1K/test_scene/images_8": "  A held-out scene.\\n"}'
    (tmp_path / "test.json").write_bytes(test_bytes)
    recipe["test_source"] = {
        "config": "videogpa", "path": "test.json", "url": "https://example.invalid/test.json",
        "sha256": hashlib.sha256(test_bytes).hexdigest(), "rows": 1,
    }
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


def test_original_videogpa_evaluation_is_test_in_combined_and_gpa(dataset_script, sources, tmp_path):
    from datasets import load_dataset

    recipe, cache = sources
    output = tmp_path / "release"
    report = dataset_script.materialize(recipe, output, cache, offline=True)
    expected = [{"text": "  A held-out scene.\n", "video_source": "DL3DV/1K/test_scene/images_8"}]
    assert report["test"]["rows"] == 1
    for config in ["combined", "videogpa"]:
        assert load_dataset(str(output), config, split="test").to_list() == expected
    assert load_dataset(str(output), split="test").to_list() == expected
    assert set(load_dataset(str(output), "videorepa_32k")) == {"train"}


@pytest.mark.parametrize("key,text", [
    ("1K/scene/images_8", "Different description"),
    ("1K/test_scene/images_8", " a SCENE. "),
])
def test_rejects_train_test_scene_or_caption_leakage(dataset_script, sources, tmp_path, key, text):
    recipe_path, cache = sources
    recipe = json.loads(recipe_path.read_text())
    data = json.dumps({key: text}).encode()
    (cache / "test.json").write_bytes(data)
    recipe["test_source"]["sha256"] = hashlib.sha256(data).hexdigest()
    recipe_path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="overlap"):
        dataset_script.materialize(recipe_path, tmp_path / "release", cache, offline=True)


@pytest.mark.parametrize("expected_revision,altered_training,match", [
    (None, False, "expected-hub-revision"),
    ("stale", False, "expected-hub-revision"),
    ("current", True, "alter training data"),
])
def test_hub_update_requires_current_parent_and_matching_training(
    dataset_script, sources, tmp_path, monkeypatch, expected_revision, altered_training, match
):
    from types import SimpleNamespace

    import huggingface_hub

    recipe, cache = sources
    output = tmp_path / "release"
    previous = dataset_script.materialize(recipe, output, cache, offline=True)
    del previous["test"]
    del previous["files"]["data/videogpa_test.parquet"]
    if altered_training:
        previous["configurations"]["combined"]["rows"] = 99
    remote = tmp_path / "remote-audit.json"
    remote.write_text(json.dumps(previous))

    class PreviousReleaseApi:
        def whoami(self):
            return {"name": "tester", "orgs": [{"name": "cvis-tmu"}]}

        def create_repo(self, *args, **kwargs):
            pass

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="current", siblings=[
                SimpleNamespace(rfilename=name) for name in [*previous["files"], "audit.json"]
            ])

        def create_commit(self, *args, **kwargs):
            pytest.fail("Unapproved/stale/destructive update must not reach commit")

    monkeypatch.setattr(huggingface_hub, "HfApi", PreviousReleaseApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(remote))
    with pytest.raises(ValueError, match=match):
        dataset_script.publish(output, "cvis-tmu/VideoGPA-VideoREPA", expected_revision=expected_revision)


def test_combined_metadata_preserves_all_source_memberships_and_identity(dataset_script):
    gpa = {"text": "  scene\n", "video_source": "DL3DV/8K/hash/images_8"}
    video = {"text": "Action", "video_source": "OpenVid/original_id_2_10to50.mp4"}
    configs = {"videogpa": [gpa], "videorepa_32k": [video], "videorepa_64k": [video]}
    rows = dataset_script.enrich_combined([gpa, video], configs)
    assert rows[0] == {**gpa, "sample_id": hashlib.sha256(
        b'["  scene\\n","DL3DV/8K/hash/images_8"]'
    ).hexdigest(), "source_dataset": "videogpa", "source_subsets": ["videogpa_8k"],
        "source_group_id": "DL3DV/hash", "media_type": "frame_directory"}
    assert rows[1]["source_subsets"] == ["videorepa_32k", "videorepa_64k"]
    assert rows[1]["source_group_id"] == "OpenVid/original_id"
    assert rows[1]["media_type"] == "video"
    assert rows[1]["source_dataset"] == "videorepa"
    assert rows[0]["sample_id"] != rows[1]["sample_id"]
    assert dataset_script.enrich_combined([video], configs)[0] == rows[1]
    assert configs["videogpa"][0] == gpa


def test_combined_test_metadata_and_source_views_have_correct_schemas(dataset_script, sources, tmp_path):
    from datasets import load_dataset

    recipe, cache = sources
    data = json.loads(recipe.read_text())
    data["combined_metadata_version"] = 1
    recipe.write_text(json.dumps(data))
    # Replace the abbreviated fixture filenames with real OpenVid suffix conventions.
    for name in ["small.csv", "large.csv"]:
        raw = (cache / name).read_bytes()
        for char in [b"a", b"b", b"c"]:
            raw = raw.replace(char + b".mp4", char + b"_0_0to10.mp4")
        (cache / name).write_bytes(raw)
        for source in data["sources"]:
            if source["path"] == name:
                source["sha256"] = hashlib.sha256(raw).hexdigest()
    recipe.write_text(json.dumps(data))
    output = tmp_path / "release"
    report = dataset_script.materialize(recipe, output, cache, offline=True)
    combined = load_dataset(str(output), "combined")
    columns = ["text", "video_source", "sample_id", "source_dataset", "source_subsets", "source_group_id", "media_type"]
    assert combined["train"].column_names == columns
    assert combined["test"].column_names == columns
    assert combined["test"][0]["source_subsets"] == ["videogpa_1k"]
    assert len(set(combined["train"]["sample_id"])) == 4
    assert set(combined["train"]["sample_id"]).isdisjoint(combined["test"]["sample_id"])
    assert load_dataset(str(output), "videogpa", split="test").column_names == ["text", "video_source"]
    assert report["configurations"]["combined"]["base_content_sha256"]
    assert dataset_script.verify_release(output) == report
