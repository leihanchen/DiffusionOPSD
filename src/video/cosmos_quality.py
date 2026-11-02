"""Keep the released Qwen2-VL reward weights on their validated dependencies."""

from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import subprocess
import sys

import torch


class IsolatedVideoJudge:
    def __init__(self, repo, run_dir, device):
        self.path = Path(run_dir) / "judge_clip.pt"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo)  # Remove the Cosmos dependency overlay.
        env["PYTHONNOUSERSITE"] = "1"
        self.log = open(Path(run_dir) / "quality_worker.log", "a", buffering=1)
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(Path(repo) / "scripts/cosmos_quality_worker.py"), "--device", device],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
        )
        try:
            if self._read() != {"ready": True}:
                raise RuntimeError("VideoReward worker failed to initialize")
        except Exception:
            self.close()
            raise

    def _read(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=600):
                raise TimeoutError(f"VideoReward worker timed out; inspect {self.log.name}")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"VideoReward worker exited; inspect {self.log.name}")
        return json.loads(line)

    def score(self, clip, prompt):
        torch.save(clip.detach().cpu(), self.path)
        self.process.stdin.write(json.dumps({"clip": str(self.path), "prompt": prompt}) + "\n")
        self.process.stdin.flush()
        return self._read()["score"]

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.log.close()
        self.path.unlink(missing_ok=True)
