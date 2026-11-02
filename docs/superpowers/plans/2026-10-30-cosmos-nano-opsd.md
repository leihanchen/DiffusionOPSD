# Cosmos3-Nano OPSD Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Execution and defaults are delegated by the user; no approval pause.

**Goal:** Add, verify, and launch OPSD training on nvidia/Cosmos3-Nano.

**Status:** Complete. Job 7635933 finished all 50 optimizer updates; final authored-project suite passed 223 tests. Earlier execution-ledger entries describe historical states, superseded by the completion evidence below.

**Architecture:** Wrap native Cosmos UniPC generation to capture query inputs and scheduler history. Replay the query with gradients and reuse video rewards and OPSD targets/loss; preserve Wan trainers.

**Tech Stack:** PyTorch, Diffusers Cosmos3OmniPipeline/UniPC, PEFT LoRA, existing geometry rewards, Apptainer, Slurm.

**Spec:** docs/superpowers/specs/2026-10-30-cosmos-nano-opsd-design.md

## Global Constraints
- Text-to-video only; no audio/actions/conditioning images, inference caches, or quantization in the initial trainer.
- Native sampler and FP32 clean-output arithmetic; y=z-sigma*v.
- Save checkpoints and logs outside Git; use existing checkpoint/cache environment variables.
- Work in place; do not alter the existing untracked SIF or Wan dependency stack.

## Review Focus
- CFG hook ordering and raw versus BF16-rounded query latents: test replay and endpoint parity.
- UniPC historical state: test baseline continuation equals the native endpoint.
- Gradient suppression and stale caches: test differentiable query and decoder paths.
- Failure during generation: test hooks and scheduler methods are restored.
- Resume/config mismatch and empty/nonfinite updates: test validation and checkpoint state.

### Task 1: Native rollout adapter
**Files:** Create src/video/cosmos_rollout.py; tests/video/test_cosmos_rollout.py.
**Interfaces:** CosmosRollout(pipeline, num_steps, guidance_scale), rollout(prompt, generator, **geometry) -> CosmosQuery; prediction(record) -> clean latent; continue_from(record, forced_y) -> endpoint; decode01(latents) -> [B,T,3,H,W].
- [x] Write failing tests using a tiny packed transformer and real UniPC for parity, CFG gradients, continuation and cleanup.
- [x] Run tests and observe missing adapter failure.
- [x] Implement query capture with detached forward kwargs and saved scheduler state; keep native step unchanged.
- [x] Implement differentiable replay, scheduler-history continuation and normalized VAE decoding.
- [x] Run focused tests and commit.

### Task 2: OPSD training and configuration
**Files:** Create src/video/cosmos_training.py, config/cosmos_nano.py, scripts/train_opsd_video_cosmos.py, tests/video/test_cosmos_training.py.
**Interfaces:** train_update(...) returns measured losses/rewards/gradients; checkpoints include both adapters, optimizer, RNG, configuration and completed update count.
- [x] Write failing tiny-model tests covering a real branch-loss update, EMA, invalid updates, checkpoint/resume.
- [x] Implement configurable policy targeting and training loop using existing target/reward/gate modules.
- [x] Add local prompt formatting, initial calibration, quality reference generation, periodic native evaluation and atomic resume.
- [x] Run tests and commit.

### Task 3: Reproducible runtime and launch
**Files:** Create scripts/setup_cosmos_nano.sh, scripts/train_opsd_video_cosmos.slurm; update README.md.
- [x] Inspect actual package requirements and checkpoint metadata; pin the verified environment and model revision.
- [x] Prepare isolated runtime and download required local model files; verify imports/assets.
- [x] Run all video tests, changed-file Ruff, shell syntax checks and a fresh whole-change review.
- [x] Submit a one-update GPU integration run, inspect finite gradients/checkpoint/evaluation, then submit the configured training run.
- [x] Record job IDs, statuses, limitations and validation results in this plan. Commit implementation and docs without artifacts.

## Execution ledger
- Plan created before implementation. Ruling: implement inline in current checkout because existing container/cache paths are shared and user delegated all defaults; do not create another dependency stack inside the base image.

- Task 1 committed as c06363d; later native-source inspection identified the additional solver axis and split-device encoding requirement. Both are exercised by the real tiny Cosmos pipeline test. Baseline continuation restores pre-query UniPC history and matches the native endpoint.
- Task 2 tests prove reward-derived branch-loss updates, nonfinite/empty-update rejection, checkpoint/RNG/config roundtrip, actual Cosmos LoRA gradients and EMA arithmetic.
- Ruling: geometry reward uses nine frames at 224x384 rather than the full decoded clip to bound GPU memory; it retains the existing reward implementation and is applied consistently to targets, calibration and endpoints.
- Ruling: VideoReward runs under the original container dependencies in a subprocess because modern Diffusers requires a newer Hub/Transformers stack. Both legacy judge imports and modern geometry imports were verified.
- Runtime: Diffusers 0.42.1, Transformers 5.22.1, PEFT 0.21.2, container PyTorch 2.14.0+cu130. Required Nano weights downloaded at revision e59a53c25979a090fa8706c9acc0c254a6e89b92. Preflight validates the revision marker, components and all transformer shards.
- Independent review found one P2: bounded job segments could finish between checkpoint intervals. Fixed by treating the per-job final update as an evaluation/checkpoint boundary with the existing tested evaluation_due schedule.
- Verification before GPU submission: 15 focused Cosmos tests passed under the modern runtime; the prior container video suite passed 186 tests. Full modern video suite passed 189 tests (two known warnings). Reviewer verified the checkpoint fix and native decoder/device-transfer paths; no critical or important findings remain. Changed-file Ruff, compileall and shell syntax checks passed.

- Implementation committed as fc0daa3. Slurm integration job 7537710 submitted on 2026-10-30 (three A100s, 256 GiB host RAM, three-hour limit, one real optimizer update and held-out evaluation). Last verified state: PENDING / Priority. Run directory: logs/cosmos-nano-opsd-20261030. Full-size GPU gradient/checkpoint verification and submission of the remaining training updates are pending this job.
- Additional checkpoint compatibility check: meta-device construction matches all 814 published tensor names/shapes, with zero missing, extra or mismatched tensors (logs/cosmos-runtime/checkpoint-layout.log).

- Ruling: queue the remaining training as an afterok-dependent resume while the integration job waits for resources. Its eligibility is gated by successful completion of the real update, finite/nonzero gradients, changed adapter weights, held-out generation, and checkpoint save in the integration job. This avoids requiring an interactive submission when that gate passes.
- Remaining training submitted as Slurm job 7537901, dependency afterok:7537710, COSMOS_RESUME=1, same run directory, configured total 50 updates and 24-hour limit. Last verified state: PENDING / Dependency. Full-size GPU validation is still pending job 7537710; neither pending submission is evidence of a successful GPU update.

- Ruling (queue/resource audit): the original three-GPU integration job is still pending and Slurm estimates 2026-11-01T15:40:22. Use two GPUs by placing the VAE and both reward scorers on cuda:1, retaining the same model, 17-frame geometry, four-rollout update and 50-update goal. The saved-activation offload/reward resizing already in the implementation make this plausible; actual GPU memory is still an integration check. Add tested configuration-based device validation rather than hardcoding a three-GPU requirement, and replace only still-pending allocations.

- Shared-device validator regression: observed missing-function failure, implemented configured-index validation, then ran the full modern suite: 190 passed (two known warnings). Changed-file Ruff and shell syntax passed. Independent review found no correctness issue in the two-GPU layout; separate parent/judge caching allocator peaks remain a runtime memory observation for the GPU check.

- Replaced only still-pending allocations after verifying state: jobs 7537710 and 7537901 are CANCELLED without any GPU runtime. New integration job 7538161 requests two A100s/192 GiB for three hours. New remaining-training job 7538162 requests the same layout for 24 hours and depends on afterok:7538161. Run directory remains logs/cosmos-nano-opsd-20261030; no checkpoint existed and no experiment settings were changed other than device placement. Implementation commit d23b60f. Initial verified states: PENDING/Priority and PENDING/Dependency respectively.

- Completion/resource audit: full-size GPU integration remains unproved because Slurm has not allocated any node to job 7538161 (PENDING/Priority, zero runtime, no application logs). The same allocation blocker persisted across at least three consecutive goal turns after all independent implementation/validation work. Scheduler estimate remains 2026-11-01T15:40:22. Goal is blocked on external GPU allocation, not marked complete; both live queued jobs remain submitted, and job 7538162 retains its afterok dependency. Resume the audit against these exact job IDs once allocation or runtime state changes.

- Resumed audit on 2026-10-31: integration job 7538161 did receive GPUs and reached calibration, two held-out native evaluations, four retained rollouts, then fitting. It ended OUT_OF_MEMORY after 12m45s; batch MaxRSS=201325344 KiB at the 192 GiB host-memory limit. No CUDA OOM traceback. Dependent job 7538162 is CANCELLED and never ran. No checkpoint was produced.
- Root-cause experiment (actual full-shape BF16 VAE on meta, no weight allocation): untiled decoder saves 100170240576 activation bytes + 5524995584 parameter bytes (~98.4 GiB); forced overlapping tiles save 167060030272 + 82875210240 (~232.8 GiB). The latter exceeds the entire job allocation before other fitting graphs. Profiles are cumulative saved-tensor accounting, not a measured successful process peak. Logs: logs/cosmos-runtime/vae-saved-tensors-profile.log and vae-tiled-profile.log. CPU diagnostic job 7548814 was cancelled while pending after local profiling completed.
- Ruling: remove forced VAE tiling and retain the official full-frame decoder default with existing CPU saved-tensor offload, without reducing frames, resolution, rollout count or target steps. Add per-sample target/policy stage logs with host RSS peak and allocated/reserved GPU memory for the retry. Independent review found no critical/important issues in this fix; full-size backward/peak memory still requires the retry.

- Memory-fix verification: full modern video suite 190 passed, two known warnings, in132.06s; changed-file Ruff passed; independent review clean. This verifies code/gradient regression behavior, not successful full-size backward. Next GPU execution is recovery attempt1 (second actual integration execution); use a fresh run directory to preserve the failed run artifacts.

- Recovery submission: integration job7548926 (two A100s/192GiB,3h,one actual optimizer update+held-out evaluation); remaining training job7548927 with afterok:7548926 and resume toward50updates. Fresh run directory logs/cosmos-nano-opsd-20261031-recovery1. Fix commit1d6595c. Initial states PENDING/Priority and PENDING/Dependency.

- Recovery allocation audit (2026-10-31 05:07 UTC): job7548926 remains PENDING/Priority, zero runtime/no node/no logs; job7548927 remains PENDING/Dependency. Explicit TZ=UTC scheduler estimate is2026-10-31T05:44:50 (cluster default output is EDT, not UTC). Allocation has remained the same external blocker across three consecutive recovery goal turns. All independent code/testing/review work is done; full-size backward is still unverified. Goal blocked on allocation with queued jobs preserved; resume against these exact handles once runtime state changes.

- Resumed after timeout: job7548926 ended TIMEOUT at3h (batch TotalCPU2h56m, MaxRSS55491672KiB); its last application event was calibration, with no denoising progress or checkpoint. Dependent7548927 was CANCELLED without allocation. This does not demonstrate successful untiled backward, nor prove an OOM. Added an ignored diagnostic wrapper using faulthandler dumps every120s around the unchanged full one-update path. Short diagnostic GPU job7589336 (2A100,192GiB,20min) submitted to locate the exact stalled call before another fix; diagnostic artifacts in logs/cosmos-runtime/cosmos_stage_probe.py and logs/cosmos-nano-diagnostic-7589336.*.

- Diagnostic allocation audit: job7589336 remains PENDING/ReqNodeNotAvail with no assigned node, zero runtime and no stdout/stderr artifacts across three consecutive diagnostic goal turns. No explicit node constraint or reservation was requested. Exact timeout call stack and a successful full-size fitting/checkpoint remain unproved; further diagnosis requires this queued GPU diagnostic. Goal blocked on external node availability with job preserved, not marked complete.

- Resource-conscious diagnostic split: full-path diagnostic7589336 remains pending for two GPUs. Added independent one-GPU VAE probe7589584 (64GiB host,15min) to exercise the exact17x480x832 untiled encoder and decoder shapes with60s stack dumps, without loading Cosmos transformer/reward models. This tests the preprocessing hypothesis and does not replace full-model training validation. Full-path diagnostic stays queued until evidence determines whether it is needed. Probe source/output: logs/cosmos-runtime/cosmos_vae_gpu_probe.py and vae-gpu-probe-7589584.*.

- Diagnostic results (2026-11-01 audit): isolated VAE probe7589584 COMPLETED onng10802; untiled encoder4.37s, decoder1.04s, output17x480x832, peak reservedGPU7.88GiB. This falsifies the hypothesis that untiled preprocessing is inherently multi-hour. Full-path diagnostic7589336 FAILED139 after10m17s onng10705, after successful calibration and both held-out native evaluations, during rollout scoring. Final timed stack dump stops at DA3 make_sincos_pos_embed/head_utils.py:137, with no complete fatal stack; whether periodic asynchronous stack dumping contributed is unproved. Neither result validates backward.
- Next controlled execution: unchanged untiled/model/reward settings, ordinary trainer (no periodic dump wrapper), component enter/exit logging with CUDA synchronization enabled through COSMOS_TRACE_STAGES=1, and PYTHONFAULTHANDLER=1 for real fatal signals in both processes. This is observability/isolation, not a claim that the segfault root cause is fixed.

- Tracing change verification: focused Cosmos suite16 passed (one expected PEFT adapter warning), changed-file Ruff and Python/shell syntax passed. Container environment check confirms COSMOS_TRACE_STAGES=1 and fatal-signal faulthandler activation propagate. Model, sampler, dimensions, target construction and loss are unchanged; this is a controlled normal-trainer retry, not a claimed fix for the unconfirmed segmentation fault.

- Normal-trainer controlled retry submitted: validation7619889 (2A100/192GiB,3h,one optimizer update), dependent remaining training7619891 (afterok:7619889,resume toward50updates,24h). COSMOS_TRACE_STAGES=1; no periodic stack-dump wrapper. Run directory logs/cosmos-nano-opsd-20261101-recovery2. Code commit1a40258. Initial states PENDING/Priority and PENDING/Dependency. No successful full-size backward or checkpoint is claimed yet.

## Runtime recovery: empty identity-filtered batches
- [x] Add a deterministic PromptCursor with persisted prompt/seed position and a full-pass rejection bound; test a rejected prompt followed by a real accepted branch-loss update and checkpoint/resume.
- [x] Integrate cursor into training sampling; leave identity threshold, optimizer-step budget and EMA unchanged for rejected batches. Preserve compatibility with the existing update8 checkpoint.
- [x] Run video suite, focused Ruff and independent review; resume the existing run toward50 actual updates, verifying skip behavior and a new checkpoint.

- Full-size GPU validation7619889 COMPLETED after23m41s. It retained four clips, produced finite target gradient0.110156 and policy gradient0.863883, changed the policy, evaluated held-out generation and saved update1. Host application peak during fitting was103.44GiB. Thus the untiled backward/memory gate is now verified.
- Dependent training7619891 resumed successfully and saved updates2 through8, then FAILED1 after1h34m because all clips for the next prompt failed the identity gate. This is a caller control-flow bug: rejection should skip a sampling batch, not abort training.
- Checkpoint audit: update8;288 tensors in each adapter; all finite; optimizer step8; behavior and trainable adapters differ. Preserved a backup at logs/cosmos-nano-opsd-20261101-recovery2/training_state_before_empty_gate_fix.pt before changes.

- Empty-batch recovery verified: observed two new regression tests fail for missing cursor, then full video suite192 passed (two known warnings,130.42s), changed-file Ruff/syntax passed, independent review clean. Read-only compatibility audit confirms current config/prompt hashes equal update8 checkpoint; legacy next attempt is(8,8), and rejection advances to(9,9) without changing completed updates8.

- Submitted resume job7635933 for the remaining updates (24h,2A100/192GiB,trace enabled,no dependency on failed job). It resumes the audited update8 state in logs/cosmos-nano-opsd-20261101-recovery2, with the original checkpoint backed up. Fix commit a54554c. Initial state PENDING/Priority. Next runtime gate: observe the rejected batch being skipped, a new successful update and checkpoint including the sampling cursor.

## Completion evidence — 2026-11-02

- Plan/design were committed before implementation (2db9081). Native Cosmos sampling/query replay, reward-derived branch fitting, LoRA/EMA, checkpoint/RNG resume, isolated reward runtime and Slurm entry points are implemented and committed.
- Full-size validation job **7619889** completed successfully. Resume job **7635933** then completed with Slurm **COMPLETED / 0:0**, elapsed **04:07:58**, on ng31005. It resumed update 8 and saved updates 9–50.
- Final checkpoint audit confirms **50 optimizer steps**, 50 consecutive metric records, finite weights/metrics and nonzero target/policy gradients for every update. Both adapters contain 288 tensors; behavior differs from the trainable policy. Three identity-rejected batches were skipped without consuming optimizer updates. Persisted cursor: num_prompts=200, cursor=53, consecutive_empty=0.
- Local artifacts (excluded from Git): `logs/cosmos-nano-opsd-20261101-recovery2/training_state.pt`, `metrics.jsonl`, `completion_audit.json`, and `eval_50_0.mp4` / `eval_50_1.mp4`. ffprobe verifies both final videos are 832×480, 17 frames, 24 fps.
- Checkpoint size: 215203726 bytes. SHA-256: `be0e965024002dcc2d803deb738e9c8cb81b4d45c0ad5f7fe71d7e5c923a709b`. The update-8 backup is preserved separately.
- Latest implementation video suite: 192 passed, two known warnings. Final authored-project verification: `OMP_NUM_THREADS=1 bash scripts/run_cosmos_python.sh -m pytest tests -q` — **223 passed, two known warnings**, in 239.54s; log: `logs/cosmos-runtime/final-project-tests.log`. Final changed-file Ruff, shell syntax and whitespace checks pass. Independent review of the cursor recovery found no critical or important issues.
- Training remains native UniPC, BF16 Cosmos3-Nano with rank-16 LoRA, 17×480×832 video, 35 steps, CFG 6, geometry/identity/motion/quality guards, and the original 50-successful-update budget. No quality-improvement conclusion is inferred from the small held-out sample.
- Branch `minimax_h3` and the current checkout are preserved. The pre-existing untracked container, model weights, caches, and generated artifacts are not committed.
