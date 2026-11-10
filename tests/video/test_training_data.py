"""Training uses record identity and pinned splits, even for duplicate captions."""

import hashlib
import json
from types import SimpleNamespace

import pytest


def test_cyclic_schedule_supports_uneven_datasets_without_duplicate_batches():
    from diffusionopsd.video.training_data import training_order

    order = training_order(5, 7, 42)
    assert len(order) == 28
    assert order[:5] == [3, 1, 2, 4, 0]
    assert order == training_order(5, 7, 42)
    assert all(len(set(order[i:i + 4])) == 4 for i in range(0, 28, 4))
    assert order[12:] == training_order(5, 7, 42)[12:]
    with pytest.raises(ValueError):
        training_order(3, 1, 42)


@pytest.fixture
def snapshot(tmp_path):
    from datasets import Dataset

    def record(i, text="same caption", split="train"):
        return dict(text=text, video_source=f"DL3DV/{split}/{i}/images_8", sample_id=f"{split}-{i}",
                    source_dataset="videogpa", source_subsets=["videogpa_8k"],
                    source_group_id=f"DL3DV/{split}-{i}", media_type="frame_directory")

    train = [record(i) for i in range(5)]
    test = [record(0, "different test caption", "test")]
    (tmp_path / "data").mkdir()
    files = {}
    for split, rows in [("train", train), ("test", test)]:
        name = "combined.parquet" if split == "train" else "combined_test.parquet"
        path = tmp_path / "data" / name
        Dataset.from_list(rows).to_parquet(str(path))
        files[f"data/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "sources.json").write_text("{}")
    files["sources.json"] = hashlib.sha256(b"{}").hexdigest()
    def content(rows):
        return hashlib.sha256("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode()).hexdigest()
    (tmp_path / "audit.json").write_text(json.dumps({"files": files,
        "configurations": {"combined": {"rows": 5, "content_sha256": content(train)}},
        "combined_test": {"rows": 1, "content_sha256": content(test)}}))
    return tmp_path


def test_loader_preserves_rows_and_fingerprints_metadata(snapshot, monkeypatch):
    import huggingface_hub
    from diffusionopsd.video.training_data import load_training_data

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(snapshot))
    cfg = SimpleNamespace(dataset=dict(source="hf", repo_id="org/data", name="combined", revision="a" * 40,
                                      train_split="train", eval_split="test", seed=42))
    data = load_training_data(cfg)
    assert len(data.train) == 5 and len(data.evaluation) == 1
    assert data.train[0]["text"] == data.train[1]["text"]
    assert data.key(0) != data.key(1)
    assert data.metadata["revision"] == "a" * 40
    before = data.fingerprint
    cfg.dataset["seed"] = 43
    assert load_training_data(cfg).fingerprint != before
    (snapshot / "data/combined.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_training_data(cfg)


def test_offline_missing_cache_has_recovery_command(monkeypatch):
    import huggingface_hub
    from diffusionopsd.video.training_data import load_training_data, default_dataset_config

    def missing(**kwargs):
        raise FileNotFoundError("cache missing")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(RuntimeError, match="prefetch_video_dataset.py"):
        load_training_data(SimpleNamespace(dataset=default_dataset_config()), offline=True)


def test_explicit_legacy_files_keep_original_hash_and_text(tmp_path):
    from diffusionopsd.video.training_data import load_training_data

    train, test = tmp_path / "train.txt", tmp_path / "test.txt"
    train.write_text(" first \nsecond\n")
    test.write_text(" held out \n")
    cfg = SimpleNamespace(dataset={"source": "files"}, prompt_fn_kwargs={"path": str(train)}, eval_prompts=str(test))
    data = load_training_data(cfg)
    assert [r["text"] for r in data.train] == ["first", "second"]
    assert data.train_hash == hashlib.sha256(train.read_bytes()).hexdigest()
    assert data.key(0) == "first"
    assert load_training_data(cfg, strip_files=False).train[0]["text"] == " first "


@pytest.mark.parametrize("script", ["train_opsd_video_wan.py", "train_opsd_video_wan_epoch.py", "train_opsd_video_cosmos.py"])
def test_training_entrypoint_validates_hf_before_model_imports(snapshot, monkeypatch, capsys, script):
    import runpy
    import sys
    from pathlib import Path
    from types import ModuleType

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(snapshot))
    config = snapshot / "config.py"
    config.write_text('''from ml_collections import ConfigDict
def get_config():
    return ConfigDict(dict(dataset=dict(source="hf", repo_id="org/data", name="combined", revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", train_split="train", eval_split="test", seed=42), updates=2, num_epochs=2, sample=dict(num_batches_per_epoch=4)))
''')
    path = Path(__file__).resolve().parents[2] / "scripts" / script
    monkeypatch.setattr(sys, "argv", [str(path), "--config", str(config), "--validate-dataset-only"])
    # Training-only imports must never be reached by this command.
    monkeypatch.setitem(sys.modules, "peft", ModuleType("blocked_peft"))
    monkeypatch.setitem(sys.modules, "diffusers", ModuleType("blocked_diffusers"))
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(path), run_name="__main__")
    assert stopped.value.code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["train_count"] == 5 and output["test_count"] == 1
    assert output["update_budget"] == 2


@pytest.mark.parametrize("mutation,match", [("duplicate_id", "Duplicate"), ("group_overlap", "source_group"),
                                           ("caption_overlap", "caption overlap"), ("missing_field", "fields")])
def test_invalid_hf_records_fail_before_training(snapshot, monkeypatch, mutation, match):
    import huggingface_hub
    from datasets import Dataset
    from diffusionopsd.video.training_data import content_hash, load_training_data

    path = snapshot / "data/combined_test.parquet"
    test = Dataset.from_parquet(str(path)).to_list()
    audit = json.loads((snapshot / "audit.json").read_text())
    if mutation == "duplicate_id":
        path = snapshot / "data/combined.parquet"
        train = Dataset.from_parquet(str(path)).to_list()
        train[1]["sample_id"] = train[0]["sample_id"]
        Dataset.from_list(train).to_parquet(str(path))
        audit["configurations"]["combined"]["content_sha256"] = content_hash(train)
    else:
        if mutation == "group_overlap":
            test[0]["source_group_id"] = "DL3DV/train-0"
        elif mutation == "caption_overlap":
            test[0]["text"] = " SAME caption "
        else:
            del test[0]["media_type"]
        Dataset.from_list(test).to_parquet(str(path))
        audit["combined_test"]["content_sha256"] = content_hash(test)
    audit["files"]["data/" + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (snapshot / "audit.json").write_text(json.dumps(audit))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(snapshot))
    cfg = SimpleNamespace(dataset=dict(source="hf", repo_id="org/data", name="combined", revision="a" * 40,
                                      train_split="train", eval_split="test", seed=42))
    with pytest.raises(ValueError, match=match):
        load_training_data(cfg)


@pytest.mark.parametrize("workers", [4, 8, 16])
def test_cyclic_record_schedule_preserves_distributed_ownership(workers):
    from diffusionopsd.video.training_data import training_order
    from diffusionopsd.video.distributed_epoch import assign_rollouts, canonical_samples

    order = training_order(5, 7, 42)
    for update in range(7):
        gathered = []
        for rank in range(workers):
            index, rollouts = assign_rollouts(order, update, rank, workers)
            gathered.append([dict(prompt_index=index, rollout_index=r, prompt="repeated caption",
                                  record_key=f"record-{index}") for r in rollouts])
        samples = canonical_samples(gathered, order, update, workers)
        assert len(samples) == 16
        assert len({s["record_key"] for s in samples}) == 4


def test_published_defaults_keep_existing_budgets():
    from config.wan_video import get_config as wan
    from config.wan22_ti2v_epoch import get_config as epoch
    from config.cosmos_nano import get_config as cosmos
    from diffusionopsd.video.training_data import DATASET_ID, DATASET_REVISION

    for cfg in (wan(), epoch(), cosmos()):
        assert cfg.dataset.repo_id == DATASET_ID
        assert cfg.dataset.revision == DATASET_REVISION
        assert cfg.dataset.seed == 42
    assert epoch().num_epochs == 50
    assert cosmos().updates == 50


def test_resume_rejects_different_dataset_fingerprint(snapshot, monkeypatch, tmp_path):
    import huggingface_hub
    from diffusionopsd.video.training_data import load_training_data
    from diffusionopsd.video.distributed_epoch import atomic_checkpoint, load_checkpoint

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(snapshot))
    cfg = SimpleNamespace(dataset=dict(source="hf", repo_id="org/data", name="combined", revision="a" * 40,
                                      train_split="train", eval_split="test", seed=42))
    data = load_training_data(cfg)
    path = tmp_path / "state.pt"
    atomic_checkpoint(path, dict(prompt_hash=data.fingerprint, config_hash="config", order=data.order(3)))
    assert load_checkpoint(path, data.fingerprint, "config")["order"] == data.order(3)
    cfg.dataset["seed"] = 43
    with pytest.raises(ValueError, match="prompt_hash"):
        load_checkpoint(path, load_training_data(cfg).fingerprint, "config")
