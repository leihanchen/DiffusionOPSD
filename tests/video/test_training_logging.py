import json
from pathlib import Path
import struct
import zlib

import pytest

from diffusionopsd.video.training_logging import VideoTrainingLogger


class FakeRun:
    def __init__(self):
        self.summary = {}
        self.records = []
        self.axes = []
        self.finished = []
        self.fail_log = False

    def define_metric(self, *args, **kwargs):
        self.axes.append((args, kwargs))

    def log(self, data):
        if self.fail_log:
            raise OSError("disk full")
        self.records.append(data)

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


class FakeWandb:
    def __init__(self):
        self.run = FakeRun()
        self.options = None

    def init(self, **kwargs):
        self.options = kwargs
        return self.run

    def Video(self, path, **kwargs):
        return {"path": path, **kwargs}


@pytest.fixture
def env(monkeypatch):
    for key in ["WANDB_ENTITY", "WANDB_PROJECT", "WANDB_CLUSTER_NAME", "WANDB_RUN_PURPOSE",
                "SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "SLURM_RESTART_COUNT"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "narval")


def test_offline_identity_events_and_media(tmp_path, env):
    sdk = FakeWandb()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"stub")
    with VideoTrainingLogger(tmp_path, {"lr": 1e-5}, "cosmos-nano", wandb_module=sdk) as logger:
        logger.log_event(0, "calibration", {"tau_id": .8})
        logger.log_train(1, {"loss": .1, "grad_norm": .2})
        logger.log_eval(1, {"eval/geo": -.2}, [{"path": str(clip), "prompt": "orbit", "seed": 42}])
        assert sdk.options["mode"] == "offline"
        assert sdk.options["entity"] == "cvis_tmu"
        assert sdk.options["project"] == "diffusionopsd"
        assert sdk.options["name"] == "job-123-cosmos-nano-opsd-training-narval"
        assert [r["optimizer_updates"] for r in sdk.run.records] == [0, 0, 1, 1]
        assert "seed=42" in sdk.run.records[-1]["eval/prompt_0/video"]["caption"]
        assert sdk.run.records[-1]["eval/prompt_0/video"]["format"] == "mp4"
    assert sdk.run.finished == [0]
    manifest = json.loads(logger.manifest_path.read_text())
    assert manifest["status"] == "finished"
    assert len(logger.journal_path.read_text().splitlines()) == 4


def test_resume_group_and_new_id(tmp_path, env):
    first = VideoTrainingLogger(tmp_path, {}, "wan22", wandb_module=FakeWandb()).start()
    first.finish()
    second = VideoTrainingLogger(tmp_path, {}, "wan22", start_update=8,
                                 resume_checkpoint=tmp_path / "training_state.pt", wandb_module=FakeWandb()).start()
    assert first.experiment_group == second.experiment_group
    assert first.run_id != second.run_id
    assert second.metadata["starting_update"] == 8
    second.finish()
    migrated = VideoTrainingLogger(tmp_path / "new", {}, "wan22", start_update=8,
                                   resume_checkpoint=tmp_path / "training_state.pt",
                                   experiment_group=first.experiment_group, wandb_module=FakeWandb()).start()
    assert migrated.experiment_group == first.experiment_group
    migrated.finish()


def test_nonzero_rank_does_nothing(tmp_path, env):
    sdk = FakeWandb()
    with VideoTrainingLogger(tmp_path / "peer", {}, "wan22", rank=1, wandb_module=sdk) as logger:
        logger.log_train(1, {"loss": .1})
        logger.log_eval(1, {}, [{"path": "missing.mp4", "prompt": "x", "seed": 1}])
    assert sdk.options is None
    assert not (tmp_path / "peer").exists()


def test_unknown_cluster_is_fatal_before_sdk_init(tmp_path, env, monkeypatch):
    monkeypatch.delenv("SLURM_CLUSTER_NAME")
    sdk = FakeWandb()
    with pytest.raises(ValueError, match="cluster"):
        VideoTrainingLogger(tmp_path, {}, "wan22", wandb_module=sdk).start()
    assert sdk.options is None


def test_writes_fail_strictly_and_cleanup_preserves_training_error(tmp_path, env):
    sdk = FakeWandb()
    with pytest.raises(OSError, match="disk full"):
        with VideoTrainingLogger(tmp_path, {}, "cosmos", wandb_module=sdk) as logger:
            sdk.run.fail_log = True
            logger.log_train(1, {"loss": .1})
    assert sdk.run.finished == [1]
    sdk = FakeWandb()
    with pytest.raises(ValueError, match="training failed"):
        with VideoTrainingLogger(tmp_path / "error", {}, "cosmos", wandb_module=sdk):
            sdk.run.fail_log = True
            raise ValueError("training failed")
    assert sdk.run.finished == [1]


def test_missing_video_is_fatal(tmp_path, env):
    with VideoTrainingLogger(tmp_path, {}, "wan", wandb_module=FakeWandb()) as logger:
        with pytest.raises(FileNotFoundError):
            logger.log_eval(0, {}, [{"path": "missing.mp4", "prompt": "x", "seed": 1}])


def test_journal_failure_is_fatal(tmp_path, env):
    with VideoTrainingLogger(tmp_path, {}, "wan", wandb_module=FakeWandb()) as logger:
        logger.journal_path.unlink()
        logger.journal_path.mkdir()
        with pytest.raises(OSError):
            logger.log_train(1, {"loss": .1})


def test_initialization_and_finish_errors_are_fatal(tmp_path, env):
    sdk = FakeWandb()

    def fail(**kwargs):
        raise OSError("sdk failed")

    sdk.init = fail
    with pytest.raises(OSError, match="sdk failed"):
        VideoTrainingLogger(tmp_path, {}, "wan", wandb_module=sdk).start()
    sdk = FakeWandb()
    with pytest.raises(OSError, match="sdk failed"):
        with VideoTrainingLogger(tmp_path / "finish", {}, "wan", wandb_module=sdk):
            sdk.run.finish = fail


def test_environment_overrides_and_local_name(tmp_path, env, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID")
    monkeypatch.setenv("WANDB_ENTITY", "test-team")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_CLUSTER_NAME", "test-cluster")
    monkeypatch.setenv("WANDB_RUN_PURPOSE", "wan22-validation")
    sdk = FakeWandb()
    with VideoTrainingLogger(tmp_path, {}, "wan", wandb_module=sdk):
        assert sdk.options["name"] == "job-local-wan22-validation-test-cluster"
        assert sdk.options["entity"] == "test-team"
        assert sdk.options["project"] == "test-project"


def test_all_video_training_entrypoints_use_shared_logger():
    root = Path(__file__).resolve().parents[2]
    scripts = sorted((root / "scripts").glob("train_opsd_video_*.py"))
    assert len(scripts) >= 3
    for script in scripts:
        source = script.read_text()
        assert "VideoTrainingLogger" in source, script.name
        assert "wandb.init(" not in source, script.name
        assert "install_wandb_jsonl_tee" not in source, script.name
        assert "with logger:" in source or "with VideoTrainingLogger(" in source, script.name
        assert "logger.log_record(" in source or ("logger.log_train(" in source and "logger.log_eval(" in source)


def test_real_offline_wandb_has_video_history_and_no_rng_changes(tmp_path, env, monkeypatch):
    import random
    import numpy as np
    import torch
    from diffusionopsd.video.training_eval import save_video

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "disabled")  # Contract overrides unrelated parent settings.
    clip = tmp_path / "eval.mp4"
    save_video(torch.zeros(5, 3, 32, 32), clip, 8)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with VideoTrainingLogger(tmp_path, {}, "cosmos") as logger:
        logger.log_eval(0, {"eval/geo": .1}, [{"path": str(clip), "prompt": "base", "seed": 3}])
        logger.log_train(1, {"loss": .2})
        logger.log_eval(1, {"eval/geo": .3}, [{"path": str(clip), "prompt": "final", "seed": 3}])
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert list((tmp_path / "wandb").glob("offline-run-*/run-*.wandb"))
    assert len(list((tmp_path / "wandb").glob("offline-run-*/files/media/videos/**/*.mp4"))) == 2
    from wandb.proto import wandb_internal_pb2
    history = []
    # W&B 0.30 removed its Python DataStore reader; read the LevelDB-log framing
    # directly so this check verifies actual persisted protobuf history in both
    # container SDK versions, without importing SDK-private implementation paths.
    path = next((tmp_path / "wandb").glob("offline-run-*/run-*.wandb"))
    with path.open("rb") as stream:
        assert stream.read(7)[:4] == b":W&B"
        fragments = b""
        while True:
            remaining = 32768 - stream.tell() % 32768
            if remaining < 7:
                assert stream.read(remaining) == b"\0" * remaining
            header = stream.read(7)
            if not header:
                break
            checksum, length, kind = struct.unpack("<IHB", header)
            chunk = stream.read(length)
            assert zlib.crc32(chunk, zlib.crc32(bytes([kind]))) & 0xFFFFFFFF == checksum
            fragments += chunk
            if kind not in (1, 4):
                continue
            record = wandb_internal_pb2.Record()
            record.ParseFromString(fragments)
            fragments = b""
            if record.HasField("history"):
                row = {}
                for item in record.history.item:
                    keys = [item.key] if item.key else list(item.nested_key)
                    target = row
                    for key in keys[:-1]:
                        target = target.setdefault(key, {})
                    target[keys[-1]] = json.loads(item.value_json)
                history.append(row)
    assert [row["optimizer_updates"] for row in history] == [0, 0, 1, 1]
    assert [row["event"] for row in history] == ["startup", "evaluation", "training", "evaluation"]
    assert history[-1]["eval/prompt_0/video"]["_type"] == "video-file"
    assert "seed=3 | update=1" in history[-1]["eval/prompt_0/video"]["caption"]
    assert list(Path(json.loads(logger.manifest_path.read_text())["offline_directory"]).glob("run-*.wandb"))


def test_media_sdk_error_is_fatal(tmp_path, env):
    sdk = FakeWandb()
    path = tmp_path / "video.mp4"
    path.write_bytes(b"stub")

    def fail(*args, **kwargs):
        raise OSError("media failed")

    sdk.Video = fail
    with pytest.raises(OSError, match="media failed"):
        with VideoTrainingLogger(tmp_path, {}, "wan", wandb_module=sdk) as logger:
            logger.log_eval(0, {}, [{"path": str(path), "prompt": "test", "seed": 1}])
    assert sdk.run.finished == [1]


def test_sdk_cannot_change_initialized_cuda_rng(tmp_path, env, monkeypatch):
    import torch

    state = [torch.tensor([1, 2], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [s.clone() for s in state])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda saved: state.__setitem__(slice(None), saved))
    sdk = FakeWandb()
    original = sdk.init

    def init(**kwargs):
        state[0].fill_(9)
        return original(**kwargs)

    sdk.init = init
    with VideoTrainingLogger(tmp_path, {}, "wan", wandb_module=sdk):
        assert state[0].tolist() == [1, 2]


def test_evaluation_aggregates_and_per_prompt_deltas():
    from diffusionopsd.video.training_logging import evaluation_metrics

    rows = [{"geo": 2., "identity": .9, "quality_Overall": 1.},
            {"geo": 4., "identity": .8, "quality_Overall": 3.}]
    baseline = [{"geo": 1., "identity": .8, "quality_Overall": 2.},
                {"geo": 2., "identity": .8, "quality_Overall": 1.}]
    metrics = evaluation_metrics(rows, baseline)
    assert metrics["eval/geo"] == 3.
    assert metrics["eval/geo_delta"] == 1.5
    assert metrics["eval/num_prompts"] == 2
    assert metrics["eval/prompt_0/quality_Overall_delta"] == -1.
    assert metrics["eval/prompt_1/quality_Overall_delta"] == 2.
    assert metrics["eval/quality_Overall_delta"] == .5
