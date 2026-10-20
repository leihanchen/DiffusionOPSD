import json
from pathlib import Path

import pytest

from diffusionopsd.video.evaluation_shards import CONDITIONS, collect_shards
from diffusionopsd.video.inference_investigation import SCORES
from scripts.aggregate_wan_evaluation import aggregate


@pytest.fixture
def shards(tmp_path):
    for task in range(8):
        directory = tmp_path / f"task_{task}"
        directory.mkdir()
        indices = list(range(task * 8, task * 8 + 8))
        args = dict(
            prompt_start=task * 8,
            num_prompts=8,
            seed_offsets=[0],
            config_sha256="config",
            sampler="native",
            height=704,
            width=1280,
            num_frames=121,
            num_steps=50,
            guidance_scale=5,
            fps=24,
        )
        manifest = dict(
            status="complete",
            conditions=list(CONDITIONS),
            prompt_indices=indices,
            completed_videos=24,
            videos_verified=True,
            arguments=args,
            checkpoint_sha256="checkpoints",
            checkpoint={"step7": "hash7", "step50": "hash50"},
            prompt_file_sha256="prompts",
            scheduler={"flow_shift": 5},
            git_sha="same-code",
            model_config_hashes={"model": "hash"},
            prompts=[f"prompt {i}" for i in indices],
            wandb={"id": f"run{task}", "entity": "test", "project": "test"},
        )
        (directory / "manifest.json").write_text(json.dumps(manifest))
        rows = []
        for index in indices:
            for condition in CONDITIONS:
                video = f"prompt_{index:03d}_seed_{index}_{condition}.mp4"
                (directory / video).write_bytes(b"test video")
                rows.append(
                    dict(
                        prompt_index=index,
                        seed=index,
                        condition=condition,
                        prompt=f"prompt {index}",
                        video=video,
                        video_verified=True,
                        initial_latent_hash=f"noise{index}",
                        prompt_embedding_hash=f"prompt{index}",
                        negative_embedding_hash="negative",
                        compute_seconds=1,
                        duration_seconds=2,
                        **{key: args[key] for key in ("height", "width", "num_frames", "num_steps", "fps")},
                        **{key: float(index + CONDITIONS.index(condition)) for key in SCORES},
                    )
                )
        (directory / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path


def test_shards_cover_all_prompts_and_match_serial_summary(shards):
    from diffusionopsd.video.evaluation_shards import COMPARISONS
    from diffusionopsd.video.inference_investigation import summarize

    rows, result, _ = collect_shards(shards)
    assert len(rows) == 192 and len(result["pairs"]) == 64
    assert sum(len(pair["training"]) for pair in result["pairs"]) == 192
    assert {row["prompt_index"] for row in rows} == set(range(64))
    assert all(row["seed"] == row["prompt_index"] for row in rows)
    assert result["means"] == summarize(rows, CONDITIONS, COMPARISONS)["means"]
    assert result["differences"]["step50-base"]["geo"] == 2
    assert [s["num_prompts"] for s in result["prompt_strata"].values()] == [2, 14, 48]


@pytest.mark.parametrize("problem", ["duplicate", "nan", "wrong_seed", "different_noise", "missing_video"])
def test_invalid_shard_records_rejected(shards, problem):
    path = shards / "task_3/samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if problem == "duplicate":
        rows[-1] = rows[0]
    elif problem == "nan":
        rows[0]["geo"] = float("nan")
    elif problem == "wrong_seed":
        rows[0]["seed"] = 0
    elif problem == "different_noise":
        rows[0]["initial_latent_hash"] = "different"
    else:
        (path.parent / rows[0]["video"]).unlink()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        collect_shards(shards)


def test_incomplete_or_mismatched_manifests(shards):
    path = shards / "task_7/manifest.json"
    manifest = json.loads(path.read_text())
    path.write_text("{")
    assert collect_shards(shards, if_ready=True) is None
    path.write_text(json.dumps({**manifest, "status": "running"}))
    assert collect_shards(shards, if_ready=True) is None
    with pytest.raises(ValueError, match="incomplete"):
        collect_shards(shards)
    path.write_text(json.dumps({**manifest, "checkpoint_sha256": "wrong"}))
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        collect_shards(shards)


def test_combined_exports_and_idempotent_finalization(shards):
    output = aggregate(shards)
    assert len(list(output.glob("*.mp4"))) == 192
    assert len(list((output / "review").glob("*.mp4"))) == 192
    assert json.loads((output / "manifest.json").read_text())["completed_pairs"] == 64
    assert aggregate(shards) == output
    assert all((shards / f"task_{i}/samples.jsonl").exists() for i in range(8))


def test_array_resources_and_ranges():
    text = Path("scripts/evaluate_wan_full_epoch_array.slurm").read_text()
    for value in (
        "--array=0-7%2",
        "--time=02:00:00",
        "--partition=gpubase_h100_b1",
        "--gpus-per-node=h100:1",
        "--cpus-per-task=4",
        "--mem=128G",
        "task * 8",
        "NUM_PROMPTS=8",
    ):
        assert value in text
    assert "--if-ready --wandb" in text
    assert "POLICY_DEVICE=cuda:0 VAE_DEVICE=cuda:0 REWARD_DEVICE=cuda:0" in text
    assert 'REWARD_DEVICE="${REWARD_DEVICE:-cuda:1}"' in Path("scripts/evaluate_wan_full_epoch.slurm").read_text()


@pytest.mark.parametrize("gpus,override,expected", [(1, None, "cuda:0"), (2, None, "cuda:1"), (2, "cuda:0", "cuda:0")])
def test_shared_scorer_device_from_allocation(gpus, override, expected):
    import os
    import subprocess

    script = (
        Path("scripts/evaluate_wan_full_epoch.slurm")
        .read_text()
        .replace("exec bash scripts/investigate_wan_inference.slurm", 'printf "%s\\n" "$REWARD_DEVICE"')
    )
    env = {**os.environ, "SLURM_SUBMIT_DIR": str(Path.cwd()), "SLURM_JOB_ID": "test", "SLURM_GPUS_ON_NODE": str(gpus)}
    env.pop("REWARD_DEVICE", None)
    if override is not None:
        env["REWARD_DEVICE"] = override
    assert subprocess.check_output(["bash", "-c", script], env=env, text=True).strip() == expected


def test_combined_offline_wandb_media_and_tables(shards, monkeypatch):
    import shutil
    import subprocess
    import sys
    import torch
    from diffusionopsd.video.training_eval import save_video

    template = shards / "template.mp4"
    save_video(torch.zeros(5, 3, 32, 32), template, 24)
    for task in range(8):
        for path in (shards / f"task_{task}").glob("*.mp4"):
            shutil.copyfile(template, path)
    for key, value in {
        "WANDB_ENTITY": "test",
        "WANDB_PROJECT": "diffusionopsd-test",
        "WANDB_RUN_GROUP": "array-test",
        "SLURM_ARRAY_JOB_ID": "123",
    }.items():
        monkeypatch.setenv(key, value)
    # Match the batch process boundary: SDK table files finish draining at exit.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from scripts.aggregate_wan_evaluation import aggregate; "
            "import sys; aggregate(Path(sys.argv[1]), log_wandb=True)",
            str(shards),
        ],
        check=True,
        timeout=120,
    )
    output = shards / "combined"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["wandb"]["project"] == "diffusionopsd-test"
    files = next((shards / "wandb").glob("offline-run-*/files"))
    assert (files / "manifest.json").is_file() and not (files / "manifest.json").is_symlink()
    assert len(list((files / "media/videos").rglob("*.mp4"))) == 192
    tables = list((files / "media/table").rglob("*.table.json"))
    pairs = next(json.loads(p.read_text()) for p in tables if "pairs_" in p.name)
    assert len(pairs["data"]) == 192
    assert "not_frozen" in pairs["columns"]
    strata = next(json.loads(p.read_text()) for p in tables if "prompt_strata_" in p.name)
    assert len(strata["data"]) == 9
