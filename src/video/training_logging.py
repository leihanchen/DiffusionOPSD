"""Strict, offline experiment logging shared by every video OPSD backbone.

Only rank zero owns files or the SDK. W&B's event index is deliberately separate
from optimizer_updates: several events can describe the same optimizer state.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from numbers import Integral, Real
import os
from pathlib import Path
import random
import re
import socket
import sys
import time
import uuid

import numpy as np
import torch


@contextmanager
def _preserve_rng():
    # SDK imports/initialization must not perturb the trainer's sampling streams.
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _atomic_json(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _scalar_record(data):
    result = {}
    for key, value in data.items():
        if isinstance(value, (str, bool)) or value is None:
            result[key] = value
        elif isinstance(value, Integral):
            result[key] = int(value)
        elif isinstance(value, Real):
            result[key] = float(value)
    # Reject nonfinite metrics before either sink sees them.
    json.dumps(result, allow_nan=False)
    return result


def _slug(value):
    return re.sub(r"[^a-z0-9_-]+", "-", str(value).lower()).strip("-")


def evaluation_metrics(rows, baselines):
    """Aggregate matched scalar score rows without assigning metric semantics."""
    if not rows or len(rows) != len(baselines):
        raise ValueError("Evaluation requires nonempty matched score and baseline rows")
    result = {"eval/num_prompts": len(rows)}
    for key in rows[0]:
        values = [float(row[key]) for row in rows]
        base_values = [float(row[key]) for row in baselines]
        result[f"eval/{key}"] = sum(values) / len(rows)
        result[f"eval/{key}_delta"] = sum(v - b for v, b in zip(values, base_values)) / len(rows)
        for index, (value, base) in enumerate(zip(values, base_values)):
            result[f"eval/prompt_{index}/{key}"] = value
            result[f"eval/prompt_{index}/{key}_delta"] = value - base
    return result


class VideoTrainingLogger:
    """One offline run per execution, grouped across checkpoint resumes.

    ``wandb_module`` allows deterministic SDK failure tests. No global SDK methods
    are patched. All paths are rooted in the training directory, not WANDB_DIR.
    """

    def __init__(self, run_dir, config, backbone, *, purpose=None, rank=0, start_update=0,
                 resume_checkpoint=None, experiment_group=None, wandb_module=None):
        self.root = Path(run_dir).resolve()
        self.config, self.backbone, self.rank = config, backbone, rank
        self.purpose = purpose
        self.start_update = int(start_update)
        self.last_update = self.start_update
        self.resume_checkpoint = Path(resume_checkpoint).resolve() if resume_checkpoint else None
        self.experiment_group = experiment_group
        self.sdk, self.run = wandb_module, None
        self.run_id = uuid.uuid4().hex[:12]
        self.manifest_path = self.root / "logging" / f"run-{self.run_id}.json"
        self.journal_path = self.root / "logging" / f"events-{self.run_id}.jsonl"
        self.metadata = {}
        self._finished = False

    def start(self):
        if self.rank != 0 or self.run is not None:
            return self
        cluster = os.environ.get("WANDB_CLUSTER_NAME") or os.environ.get("SLURM_CLUSTER_NAME")
        job = os.environ.get("SLURM_JOB_ID", "local")
        if job != "local" and not cluster:
            raise ValueError("Slurm logging requires SLURM_CLUSTER_NAME or WANDB_CLUSTER_NAME for the cluster")
        cluster = cluster or "local"
        purpose = os.environ.get("WANDB_RUN_PURPOSE") or self.purpose or f"{self.backbone}-opsd-training"
        name = f"job-{job}-{_slug(purpose)}-{_slug(cluster)}"
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        group_path = self.root / "logging" / "experiment.json"
        if self.experiment_group is None:
            source = self.resume_checkpoint.parent if self.resume_checkpoint else self.root
            source_manifest = source / "logging" / "experiment.json"
            if source_manifest.is_file():
                self.experiment_group = json.loads(source_manifest.read_text())["experiment_group"]
            else:
                digest = hashlib.sha256(str(source).encode()).hexdigest()[:12]
                self.experiment_group = f"video-opsd-{digest}"
        if group_path.exists() and json.loads(group_path.read_text())["experiment_group"] != self.experiment_group:
            raise ValueError("Logging experiment group differs from the destination run directory")
        _atomic_json(group_path, {"experiment_group": self.experiment_group})
        versions = {}
        for package in ("torch", "diffusers", "transformers", "peft", "wandb"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                continue
        self.metadata = {
            "run_id": self.run_id, "name": name, "experiment_group": self.experiment_group,
            "entity": os.environ.get("WANDB_ENTITY") or "cvis_tmu",
            "project": os.environ.get("WANDB_PROJECT") or "diffusionopsd",
            "mode": "offline", "backbone": self.backbone, "cluster": cluster, "job_id": job,
            "restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"), "host": socket.gethostname(),
            "starting_update": self.start_update,
            "resume_checkpoint": str(self.resume_checkpoint) if self.resume_checkpoint else None,
            "versions": versions, "status": "initializing", "started_at": time.time(),
            "allocation": {key: os.environ[key] for key in
                           ("SLURM_NTASKS", "SLURM_CPUS_PER_TASK", "SLURM_JOB_NUM_NODES", "SLURM_JOB_GPUS")
                           if key in os.environ},
        }
        _atomic_json(self.manifest_path, self.metadata)
        try:
            with _preserve_rng():
                if self.sdk is None:
                    import wandb
                    self.sdk = wandb
                self.run = self.sdk.init(
                    entity=self.metadata["entity"], project=self.metadata["project"],
                    name=name, id=self.run_id, group=self.experiment_group, dir=str(self.root),
                    mode="offline", config={**self.config, "video_logging": dict(self.metadata)},
                )
                if self.run is None or getattr(self.run, "disabled", False):
                    raise RuntimeError("W&B did not create an offline run")
                self.run.define_metric("optimizer_updates")
                self.run.define_metric("*", step_metric="optimizer_updates")
                self.metadata["offline_directory"] = str(Path(getattr(self.run, "dir", self.root / "wandb")).parent)
                self.metadata["status"] = "running"
                _atomic_json(self.manifest_path, self.metadata)
                self.log_event(self.start_update, "startup", {"backbone": self.backbone})
        except BaseException:
            try:
                self.finish("failed")
            except Exception:
                pass
            raise
        return self

    def _log(self, update, event, data, videos=()):
        if self.rank != 0:
            return
        if self.run is None or self._finished:
            raise RuntimeError("Video logger is not running")
        record = {**_scalar_record(data), "optimizer_updates": int(update), "event": event}
        self.last_update = int(update)
        media = []
        with _preserve_rng():
            payload = dict(record)
            for index, video in enumerate(videos):
                path = Path(video["path"]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                prompt, seed = video["prompt"], int(video["seed"])
                prompt_index = video.get("index", index)
                payload[f"eval/prompt_{prompt_index}/video"] = self.sdk.Video(
                    str(path), format="mp4", caption=f"{prompt} | seed={seed} | update={update}",
                )
                media.append({"path": str(path), "prompt": prompt, "seed": seed, "index": prompt_index})
            # Durable journal is an additional recovery source, not a claim of
            # transactional/exactly-once delivery into the SDK's asynchronous sink.
            with self.journal_path.open("a") as stream:
                stream.write(json.dumps({**record, "time": time.time(), "videos": media}, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.run.log(payload)

    def log_train(self, update, metrics):
        self._log(update, "training", metrics)

    def log_eval(self, update, metrics, videos=()):
        self._log(update, "evaluation", metrics, videos)

    def log_event(self, update, event, details=None):
        self._log(update, event, details or {})

    def log_record(self, record, videos=()):
        """Compatibility sink for Wan's existing combined metric/JSONL schema."""
        if self.rank != 0:
            return
        update = record["optimizer_updates"]
        scalars = _scalar_record(record)
        with (self.root / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({**scalars, "_step": update}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        payload = dict(record)
        train = self.config.get("train", {})
        if "train/rl_distillation_loss" in record:
            payload["learning_rate"] = train.get("learning_rate")
            payload["loss_scale"] = train.get("adv_clip_max")
        self._log(update, "evaluation" if videos or any(k.startswith("eval/") for k in record) else "training",
                  payload, videos)

    def finish(self, status="finished"):
        if self.rank != 0 or self._finished:
            return
        self._finished = True
        self.metadata.update(status=status, finished_at=time.time(), final_update=self.last_update)
        try:
            with _preserve_rng():
                if self.run is not None:
                    self.run.summary.update({"training_status": status, "job_id": self.metadata["job_id"],
                                             "final_optimizer_updates": self.last_update})
                    self.run.finish(exit_code=0 if status == "finished" else 1)
        except BaseException:
            self.metadata["status"] = "logging_failed"
            raise
        finally:
            if self.manifest_path.parent.exists():
                _atomic_json(self.manifest_path, self.metadata)

    def __enter__(self):
        return self.start()

    def __exit__(self, exception_type, exception, traceback):
        if exception_type is None:
            self.finish()
        else:
            try:
                self.log_event(self.last_update, "failure", {"exception_type": exception_type.__name__})
            except Exception:
                print("Video logger could not record the training failure; inspect its local journal", file=sys.stderr)
            try:
                self.finish("failed")
            except Exception:
                print("Video logger cleanup failed; preserving the training exception", file=sys.stderr)
        return False
