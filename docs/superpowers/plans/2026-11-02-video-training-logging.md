# Shared offline W&B logging for video OPSD

**Status:** Complete. Shared logging is implemented, reviewed, unit tested in both environments, and verified in four
successful GPU jobs, including checkpoint resume. Work is committed locally on `minimax_h3`; no automatic sync is enabled.

## Accepted behavior
All three video OPSD entry points (Cosmos, Wan, distributed Wan) must use a shared model-independent logger.
Offline only; default destination cvis_tmu/diffusionopsd; no automatic sync or credential requirement.
One run per job execution, grouped across resume; job-purpose-cluster display names. Rank zero owns all writes.
Logging failure is fatal, with original training exceptions preserved during cleanup. Log existing scalar metrics,
configuration/provenance, calibration/rejection/checkpoint events, and every saved evaluation MP4 with prompt/seed/update.
Use absolute optimizer_updates as plot axis and independent W&B event indices. Preserve legacy JSONL formats and checkpoints.
Keep local event journal and logging manifests. Forward container metadata explicitly. Future video trainers must satisfy
the shared integration contract. Image/non-OPSD trainers, automatic upload and workspace changes are excluded.

## Execution
- [x] Add logger with tested offline lifecycle, provenance, grouping, media, strict errors and no-op nonzero ranks.
- [x] Integrate three trainers; preserve training math and legacy JSONL; add checkpoint experiment-group metadata.
- [x] Update launchers and documentation; test integration discovery and real offline media.
- [x] Run full authored suite, Ruff and shell checks; whole-change independent review.
- [x] Submit Cosmos/Wan one-update and two-worker Wan smoke checks; inspect actual offline records and resume metadata.

## Rulings and evidence
- Ruling: work in the existing minimax_h3 feature checkout so ignored container/model/runtime assets remain accessible;
  no merge, push or online publishing is part of this task. Existing untracked SIF is left untouched.
- Ruling: legacy checkpoints get experiment grouping from their source run directory; new checkpoints persist it explicitly.
  Each execution uses a new W&B ID, so resumed jobs never append conflicting internal steps to old histories.
- Red/green evidence: missing shared logger failed collection; additional regressions demonstrated missing CUDA RNG
  restoration and missing evaluation aggregation before implementation. SDK 0.30 removed the private DataStore reader;
  tests now validate persisted protobuf history through its log framing, including nested video metadata.
- Full authored suite: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 bash scripts/run_cosmos_python.sh -m pytest tests -q`
  **237 passed**, two known warnings, 236.79s. Log: `logs/cosmos-runtime/shared-logging-project-verified.log`.
- Original Wan runtime: `COSMOS_USE_BASE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 bash scripts/run_cosmos_python.sh
  -m pytest tests/video/test_training_logging.py tests/video/test_training_eval.py -q` — **21 passed**, one known warning,
  34.17s. Log: `logs/cosmos-runtime/shared-logging-base-confirmed.log`. Test MP4 decoding limits FFmpeg to one thread to avoid
  oversubscribing the login node during parallel environment verification.
- Changed-file Ruff, Python compilation, shell syntax and whitespace checks pass. Both model asset preflights pass.
- Whole-change reviewer found missing Cosmos aggregate/baseline-delta metrics; fixed with tested matched evaluation rows.
  Reviewer recheck confirms this and initialized CUDA RNG restoration resolved, with no further concrete issues.
- Ruling: smoke presets use five frames, 224x384, eight denoising steps to test logging on actual models economically.
  OPSD mathematics, candidate counts and gates are preserved; production configurations are unchanged.
- GPU smoke acceptance: one offline run per job; finite nonzero optimizer gradient; checkpoint and experiment group;
  four persisted media entries spanning baseline/update 1 with prompt/seed/update captions. Distributed peers must not
  create duplicate runs. The audit helper and submissions are retained under ignored `logs/video-logging-smoke/`.
- Implementation commit: `7f833ac`. Submitted Cosmos smoke **7704612** (2 A100s, 192 GiB, 3h) and single-worker Wan
  **7704619** (same allocation). Both were initially PENDING; no GPU result is inferred from submission.
- Four-GPU/two-worker submission with an explicit by-GPU partition was rejected before allocation. Removed that partition
  override as instructed by Slurm; successful job **7704703** (4 A100s, 480 GiB, 3h, no chaining) immediately RUNNING on
  ng30902. Rank zero created exactly one offline binary run with name
  `job-7704703-wan22-two-worker-offline-logging-smoke-narval`, ID `b41c47598d88`, before loading model weights;
  rank one created no duplicate run. Successful optimizer/media/checkpoint verification is still pending runtime.

## Final GPU evidence (supersedes initial submission states above)

| Job | Workload | Slurm result | Elapsed | Offline run ID | Verified videos |
| --- | --- | --- | --- | --- | --- |
| 7704612 | Cosmos, update 0 to 1 | COMPLETED / 0:0 | 00:05:40 | c6b08353f186 | 4 |
| 7704619 | Single-worker Wan, update 0 to 1 | COMPLETED / 0:0 | 00:07:26 | 33c312f2de50 | 4 |
| 7704703 | Two-worker Wan, update 0 to 1 | COMPLETED / 0:0 | 00:07:29 | b41c47598d88 | 4 |
| 7705712 | Cosmos resume, update 1 to 2 | COMPLETED / 0:0 | 00:04:04 | dc080bfdbdb4 | 2 |

- Audited actual SDK binary/protobuf history, not just SDK return codes: each job has one offline run, positive finite
  optimizer gradient, checkpoint events, absolute optimizer updates, and the expected video entries with prompt/seed/update
  captions. Original MP4s and SDK media copies both exist. Distributed workers produced exactly one run for their job.
- Cosmos and distributed Wan checkpoint audits confirm optimizer step 1 and persisted experiment-group metadata matching
  their logging manifests. Single-worker Wan saved its existing policy checkpoint. These are logging smoke tests using the
  documented reduced video settings, not claims about training quality.
- Resume job 7705712 started at optimizer update 1 and ended at 2; its run ID differs from 7704612, but both share group
  `video-opsd-f0e6f5007d1d`. Its history contains exactly one new training record (update 2) and two evaluation videos
  (update 2), with no predecessor history copied. Its source checkpoint is recorded in the manifest.
- Final Cosmos resume checkpoint audit: optimizer steps exactly `[2.0]`, all adapter tensors finite, saved logging group
  matches the resumed run. Resumed SDK history also contains 26 system-statistics records.
- Evidence paths: `logs/video-logging-smoke/cosmos-20261102/` (initial/resume logging and checkpoint audit JSON files),
  `logs/video-logging-smoke/wan-single-7704619/logging_audit.json`, and
  `logs/wan22-scaled-offline-logging-20261102/logging_audit.json`. Slurm output uses `logs/logging-cosmos-7704612.*`,
  `logs/logging-wan-single-7704619.*`, `logs/wan22-scaled-7704703.*`, and `logs/logging-cosmos-resume-7705712.*`.
- All logs remain offline under their recorded `offline_directory`; destination metadata is cvis_tmu/diffusionopsd.
  No W&B upload, GitHub push, PR modification, or merge was performed for this implementation task.
