# Two-GPU worker scaling

The scalable launcher derives process counts from the allocation. Each worker sees two exclusive GPUs: local `cuda:0` hosts the Wan policy and text encoder; local `cuda:1` hosts the VAE and scorers. The existing three-GPU launch scripts remain available.

| Nodes | A100s per node | Workers per node | Total workers | Rollouts per worker |
| --- | --- | --- | --- | --- |
| 4 | 2 | 1 | 4 | 4 |
| 4 | 4 | 2 | 8 | 2 |
| 8 | 4 | 2 | 16 | 1 |

Every global update still contains four distinct prompts and four videos per prompt. Workers within a prompt group split rollout indices 0–3. All four scores are used together for prompt-level advantage calculation. Gradients are summed across workers and divided by the global identity-retained count, including ranks that retain no videos. The prompt schedule still takes 50 updates to cover 200 prompts.

## Submit a monitored probe

Run from the repository root. Preview the exact allocation without submitting:

```bash
python scripts/submit_opsd_video_wan_scaled.py \
  --nodes 4 --gpus-per-node 4 \
  --resume-from logs/wan22-two-gpu-probe-4962523/training_state_step4.pt \
  --allow-topology-change --expected-resume-update 4 \
  --eval-freq 2 --dry-run
```

Remove `--dry-run` to submit. Defaults are four hours, two new optimizer updates, 12 CPU cores per worker, and no successors. Per-node host RAM defaults to 448 GiB for one worker and 768 GiB for two workers; the latter selects Narval's larger-memory nodes. Allocation does not guarantee an immediate start. The partition is omitted by default so Narval can route the memory/GPU request; `--partition` is available as an override. Job allocation uses GPUs per node, and the launch step uses GPUs per task, because the site rejects combining both allocation options.

The wrapper creates a distinct `logs/wan22-scaled-<run-id>/` directory. Optional `--run-id` sets the suffix. Without `--resume-from`, a fresh epoch is initialized. Scalable workers use the existing two-GPU configuration; generation, target construction, learning rate, gates, and batch semantics are unchanged.

## Resume and topology changes

Checkpoints are trusted local PyTorch training state files; never pass an untrusted pickle. An explicit `--resume-from` requires a new destination directory and leaves its source untouched. Changing world size or workers per node requires `--allow-topology-change`.

Version-2 checkpoints record training configuration, execution topology, adapters, optimizer, prompt order, calibration references/thresholds, evaluation baseline, tracker history, and per-worker RNG state. Output-directory location is excluded from semantic configuration comparison; all training settings are checked. Version-1 checkpoints are accepted only after verifying their original full configuration hash using the source run directory.

The same topology restores saved RNG states. A changed topology reseeds workers from the saved nonce, update number, and new rank, and resets the update-duration estimate. Initial rollout noise seeds retain their prompt/rollout identities; topology migration does not promise bitwise-equivalent targets or optimizer updates. Keep calibration reference files accessible at their original paths.

## Continue an epoch across segments

After validating the probe, a separate full-epoch submission can use `--time 24:00:00 --max-updates 0 --chain`. Evaluation defaults to every 25 updates; model export remains every 50, with durable training checkpoints after every update. This is an example interface, not an instruction to start the full epoch automatically.

Successful incomplete chained jobs submit one `afterok` successor from saved launch options. The successor retains allocation, configuration, evaluation and budget settings and resumes its destination's latest checkpoint; the initial external checkpoint is not loaded again. Failed jobs do not submit successors.

## Evidence and limitations

Rank 0 records topology, sample counts, adapter consistency, losses, masks, timing, GPU peaks and host RSS in offline W&B and durable JSONL. Each completed update checks exact adapter fingerprints across all workers. Startup records GPU UUIDs and rejects device overlap on the same host before model loading.

A successful probe must reach updates 5 and 6, save `training_state.pt` with `next_update=6`, report 16 rollouts per update and identical adapters, and exit without OOM, SIGSEGV or distributed failure. Compare measured update time and allocated GPU-hours with the earlier probe; more workers alone do not guarantee acceleration. Host offload remains substantial and does not scale down proportionally with the rollout count.

## One-node smoke test

Use an explicit smoke mode when testing two workers on one four-GPU node:

```bash
python scripts/submit_opsd_video_wan_scaled.py \
  --nodes 1 --gpus-per-node 4 --single-node-smoke --max-updates 1 \
  --resume-from logs/wan22-two-gpu-probe-4962523/training_state_step4.pt \
  --allow-topology-change --expected-resume-update 4 --eval-freq 2
```

This keeps all four prompts and 16 videos in the global update. Each worker generates rollout indices 0–1 or 2–3 for each of the four prompts: eight sequential rollouts per worker. The global advantage computation and retained-count gradient normalization are unchanged. The source checkpoint remains untouched; the destination checkpoint and W&B configuration identify the single-node smoke topology. Exactly one new update is permitted, and chaining is rejected.

The one-node test requests 480 GiB host RAM and a four-hour limit. Previous peaks of approximately 227.2 GiB per worker imply about 454.4 GiB for two workers, leaving roughly 25.6 GiB of requested headroom. The request fits standard Narval nodes (510000 MiB advertised); concurrent two-worker peak memory still needs measurement. Production two-worker-per-node defaults remain 768 GiB, and `--mem` explicitly overrides either default. It tests device isolation, concurrent target construction, and one synchronized optimizer update. It is not a timing benchmark of the eight-worker layout. With evaluation every two updates, a step-4-to-step-5 test does not trigger held-out evaluation.
