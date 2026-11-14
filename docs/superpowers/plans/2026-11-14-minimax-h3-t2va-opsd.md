# MiniMax H3 t2va OPSD implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task. Steps use checkbox syntax. The user delegated defaults and requested no questions. Resolve routine choices from the design and recorded verification. This document describes future implementation work.

**Goal:** Add reproducible MiniMax H3 `t2va` video-reward LoRA post-training and verify it on one eight-H100 node.

**Architecture:** Four policy processes shard one frozen backbone with FSDP2 and keep synchronized FP32 LoRA replicas. Each process owns one additional GPU for differentiable VAE/reward computation. Native joint audio/video rollout feeds detached video targets and one global OPSD update.

**Tech Stack:** PyTorch FSDP2, Diffusers modular H3, PEFT, safetensors, existing GeoFlow/VideoReward, Apptainer, torchrun, Slurm, offline W&B.

**Spec:** `docs/superpowers/specs/2026-11-14-minimax-h3-t2va-opsd-design.md`.

## Global constraints

- Eight 80 GB H100s on one node, four policy ranks, auxiliary index `rank + 4`.
- Request 64 CPU cores and 768 GiB host RAM for initial verification.
- Pinned model revision `0919fb684102ef115126612f3fbb84d1027e7317`, `t2va` workflow.
- Native deterministic Euler with both schedules, shifts 12/3, one joint forward per step.
- Production pilot geometry `124x544x960` at 24 fps and 50 sigma grid points.
- Four candidates per prompt, one candidate per rank, video-only clean-output targets.
- Rank-16/alpha-32 zero-dropout attention LoRA in the 50 main transformer blocks.
- One frozen backbone, replicated `default` and `old` FP32 adapters, explicit SUM/N gradient normalization.
- Existing video reward/branch math, historical prompt statistics, gates, dataset provenance, and offline logging.
- Fifty successful updates in the pilot, selected 256 training records, eight evaluation records, four calibration prompts.
- Checkpoint every update and at safe wall-time boundaries, evaluate every ten updates and at completion.
- CPU tests and native tiny-model probes precede full-checkpoint GPU tests.
- Each memory claim must identify geometry, phase, allocated/reserved counters, and external-process device sampling.

## Review focus

1. FSDP frozen parameters must remain usable during backward through ignored trainable LoRA modules. Test this before adopting the wrapper in the trainer, in Task 1.
2. Rejected, empty, nonfinite, and failed candidates must preserve collective order and optimizer/EMA consistency, in Task 5.
3. In-place native scheduler updates must not corrupt saved joint query states or either modality's schedule, in Task 3.
4. Peak loading, decoder backward, and external judge allocations must fit the device and host budgets, in Tasks 1, 4, and 8.
5. Interrupted checkpoint/cache writes and resume must preserve the accepted update, RNG, cursor, and all adapter replicas, in Tasks 2 and 6.

## Files and responsibilities

| Files to create | Responsibility |
| --- | --- |
| `src/video/h3_fsdp.py` | Sharded loading, adapter invariants, synchronization, memory sampling |
| `src/video/h3_conditioning.py` | Cached native conditioning and reproducible input identity |
| `src/video/h3_rollout.py` | Native t2va query capture, replay, suffix diagnostic, differentiable decoding |
| `src/video/h3_training.py` | Four-rank sampling, target, masked fitting, evaluation protocol |
| `src/video/h3_checkpoint.py` | Atomic replicated-adapter checkpoint and rank state restoration |
| `config/minimax_h3_t2va.py`, `config/minimax_h3_t2va_smoke.py` | Pilot and reduced-canvas full-model smoke settings |
| `containers/minimax-h3-requirements.txt` | Verified isolated dependency versions |
| `scripts/setup_minimax_h3.sh`, `scripts/run_minimax_h3_python.sh` | Runtime preparation and container execution |
| `scripts/prepare_minimax_h3.py` | Model prefetch, streaming conditioning cache, asset preflight |
| `scripts/train_opsd_video_minimax_h3.py`, `scripts/train_opsd_video_minimax_h3.slurm` | Training entry point and eight-GPU allocation |
| `scripts/check_minimax_h3_update.py` | Stage-level full-checkpoint GPU verification and audit output |
| `tests/video/test_h3_{fsdp,conditioning,rollout,decode,training,checkpoint,entrypoint}.py` | Focused contracts for the corresponding units |
| `docs/minimax_h3_training.md` | Setup, commands, resources, resume, measured results |

Reuse the existing target builder, branch loss, gates, dataset loader, logger,
reward, and isolated judge. Give each judge a separate rank directory. Add the H3
usage link to `README.md` in Task 7. Add shared changes only if a concrete H3
caller needs them and the existing callers have regression coverage.

## Task 1 — Prove the sharded policy and runtime contract

**Files:** Create `h3_fsdp.py`, the two runtime scripts, dependency requirements,
`test_h3_fsdp.py`, and the initial verification script.

**Interfaces:**

- `init_h3_distributed() -> H3Devices`, with `rank`, `world_size`, `policy`, `auxiliary`, `policy_group`, and `control_group`.
- `load_h3_policy(model_path: Path, revision: str, devices: H3Devices, *, rank: int, alpha: int) -> H3Policy`.
- `H3Policy.model` is callable, `trainable_parameters` is a deterministic tuple, `select_adapter(name)` preserves the trainability mask, and `ema_(decay)` updates replicated `old` tensors.
- `reduce_adapter_gradients_(parameters, group, kept_count: int) -> None` performs SUM/N reduction and finite validation. `fit_h3_update` owns preclip-norm measurement and clipping after reduction.

- [ ] Write `test_frozen_fsdp_backbone_propagates_ignored_lora_gradients`, `test_streamed_loading_matches_unsharded_weights`, and `test_adapter_switch_preserves_trainability`. Use a tiny real H3 transformer with two modalities. Assert nonzero LoRA gradients, frozen base tensors, native output parity, and identical adapter replicas after update.
- [ ] Run the CPU serialization and module-selection cases and observe the expected missing-interface failure. Run four-process NCCL cases under a GPU allocation. Mark GPU cases with `pytest.mark.h3_gpu` and an explicit environment opt-in so ordinary CPU suites skip them cleanly.
- [ ] Implement bottom-up frozen-weight FSDP2 with both adapters in `ignored_params`, original checkpoint dtypes, explicit FP32 adapter placement/broadcast, checkpointing, and `reshard_after_forward=True`. Avoid whole-model GPU loading. Stream checkpoint slices matching DTensor offsets and validate sharded padding on uneven dimensions. Test FP32 projection leaves and the root's remaining parameters.
- [ ] Reconstruct nonpersistent/constructor-derived buffers after materialization, including FP32 rotary frequencies. Compare buffers and loaded tiny-model outputs against ordinary `from_pretrained`. Assert frozen base DTensors have no gradients and adapters remain ordinary replicated tensors with non-reentrant checkpointing enabled.
- [ ] Probe the installed Torch 2.14 container first. Confirm that its `fully_shard` actually exposes `ignored_params`, and that gradients through the frozen sharded blocks work in this arrangement. Pin a compatible H3 Diffusers/Transformers/PEFT stack in a separate runtime. If a newer Torch is technically necessary, prepare and verify a dedicated H3 image and record the reason and exact versions.
- [ ] Verify one tiny forward/backward/update with four independent candidates against an unsharded four-candidate reference. Test that ignored gradients are synchronized exactly once. Test repeated behavior/default switching and a forward after evaluation.
- [ ] Run `bash scripts/run_minimax_h3_python.sh -m pytest tests/video/test_h3_fsdp.py -q` and the same test file with four-process GPU opt-in. Assert zero missing/unexpected checkpoint tensors and no full-model GPU allocation during loading.

Suggested commit `feat(video): add sharded H3 policy runtime`.

## Task 2 — Prepare reproducible t2va inputs

**Files:** Create `h3_conditioning.py`, `prepare_minimax_h3.py`, both configs, and
`test_h3_conditioning.py`. Extend setup to prefetch only required subfolders.

**Interfaces:**

- `prepare_conditioning(cfg, *, device: str, cache_root: Path) -> Path` returns a cache manifest.
- `load_conditioning(cache_root: Path, record_key: str, *, expected_identity: dict) -> dict[str, torch.Tensor]` returns CPU state fields consumed by native H3's core workflow.
- `validate_h3_assets(cfg, cache_root: Path) -> dict` verifies model/runtime/cache/reward identity and returns provenance.

- [ ] Write tests for layer-50 unnormalized embeddings, preserved text token tags, deterministic record selection, dtype/hash validation, interrupted writes, and simultaneous writers. A second writer reuses a complete matching entry and rejects mismatched identity.
- [ ] Verify failures before implementing cache creation and loading.
- [ ] Use the actual pinned native text-encoding block to produce the complete core input schema. Stream one record at a time and publish locked temporary entries atomically. Record tokenizer, processor, model, layer, dataset, prompt and schema identity.
- [ ] Record native tokenization/truncation settings and encoder context limits. Add cache tests for the longest selected prompt and rejected out-of-context input. Preserve prompt content and record all modality token counts for memory verification.
- [ ] Select the first 256 records from the existing pinned dataset's seeded train permutation, plus four calibration and eight test records by stable IDs. Use the same selected training pool for calibration without duplicating cache entries. Load the encoder once in preparation and release its process before torchrun training.
- [ ] Set the design's exact pilot defaults. Set smoke geometry to `124x256x256`, 16 sigma grid points, four candidates, one update, and four evaluation records so all ranks participate. Keep the native duration and log the actual query sigma for each preset.
- [ ] Run `bash scripts/run_minimax_h3_python.sh -m pytest tests/video/test_h3_conditioning.py -q`. Check `prepare_minimax_h3.py --config config/minimax_h3_t2va.py --check-only` verifies a complete cache and reports missing items precisely before allocation-consuming training begins.

Suggested commit `feat(video): cache native H3 text conditioning`.

## Task 3 — Capture and replay native joint trajectories

**Files:** Create `h3_rollout.py` and `test_h3_rollout.py`.

**Interfaces:**

- `H3Query` has detached packed forward inputs, generated-row slices, `sigma_video`, `sigma_audio`, `query_index`, both schedules/scheduler states, `video_endpoint`, `audio_endpoint`, and resolved geometry. Its `y0` property returns unpacked FP32 video clean output.
- `H3Rollout(core, *, query_sigma: float, num_sigma_points: int)` uses an initialized cached-conditioning native t2va core.
- `rollout(conditioning: dict, *, seed: int, geometry: dict) -> H3Query`.
- `prediction(record: H3Query) -> torch.Tensor` returns `[1,24,T,H,W]` with gradients.
- `continue_from(record: H3Query, forced_video_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` returns the two endpoint latents using fixed behavior weights.

- [ ] Write tests using tiny native H3 modules and native modular layout/timestep blocks. Assert endpoint parity, correct `z + sigma*u` anchor, actual nearest nonterminal sigma, different video/audio times, generated-row selection, and one joint forward per native step.
- [ ] Include input-clone tests that mutate scheduler output and verify saved query states remain unchanged. Test hook restoration after a thrown error. Run and confirm missing-adapter failures.
- [ ] Initialize `MiniMaxH3Blocks` via `workflow="t2va"`, then split off cached text encoding and final decode blocks using their supported workflow APIs. Inspect the pinned signatures and wire exact core-state fields. Preserve H3's loading specs and native denoising blocks.
- [ ] Add scoped capture at pre-step q for both modalities, then direct differentiable transformer replay through `model(...)`. Keep all structural fields, native normalization, and batch dimensions. Preserve the native mixed-precision forward and compute anchors in FP32.
- [ ] Implement the suffix diagnostic restoring both states/schedulers. Force video clean output at q, use the saved old audio output at q, and jointly recompute subsequent steps.
- [ ] Run `bash scripts/run_minimax_h3_python.sh -m pytest tests/video/test_h3_rollout.py -q`. Require exact native scheduler outputs with identical seeds/precision and explicit tolerances for replay arithmetic. Test baseline suffix parity and a changed-video-target suffix.

Suggested commit `feat(video): replay native H3 joint queries for OPSD`.

## Task 4 — Verify differentiable video target construction

**Files:** Extend `h3_rollout.py`, create `test_h3_decode.py`, extend the GPU check script.

**Interfaces:** `H3Rollout.decode01(video_latents: torch.Tensor) -> torch.Tensor`
returns `[B,T,3,H,W]`. Existing `opa_tr_step_nd` and `branch_loss` are consumed directly.

- [ ] Write tiny native H3 VAE tests for unpacking, normalization inversion, native output parity, finite nonzero latent gradients, detached trust-region targets, and `[B,T,3,H,W]` range/shape.
- [ ] Run them and verify the new decode interface fails before implementation.
- [ ] Call the native VAE directly outside modular inference decorators. Enable native decoder gradient checkpointing and tiling. Apply the existing nine-frame reward selection, resize to `224x384`, and preserve the current `reward_metadata()` contract.
- [ ] Ensure target tensors are FP32, decoded/reward graphs are freed before fitting, and only accepted candidates construct targets. Keep the existing derivative policy explicit in manifests.
- [ ] Run `bash scripts/run_minimax_h3_python.sh -m pytest tests/video/test_h3_decode.py tests/video/test_opa_video.py tests/video/test_train_step_math.py -q`. Check ascent/descent direction and `||target-anchor|| <= 0.10*||anchor||` with numerical tolerance.
- [ ] Add full-size target-phase profiling for the native VAE, including parent and quality subprocess GPU allocations. Test native tiled and untiled modes separately before selecting any memory fallback. Record host overhead if CPU saved-tensor offload is enabled.

Suggested commit `feat(video): construct reward targets through H3 video decoder`.

## Task 5 — Implement collective-safe global OPSD updates

**Files:** Create `h3_training.py` and `test_h3_training.py`, extend policy tests.

**Interfaces:**

- `collect_candidates(policy: H3Policy, rollout: H3Rollout, request: dict, *, rank: int) -> H3Candidate`, with one owned candidate and scalar score/keep metadata.
- `fit_h3_update(policy: H3Policy, rollout: H3Rollout, candidate: H3Candidate, *, weight: float, keep: bool, global_kept: int, reward_fn, optimizer, cfg) -> dict`.
- `run_h3_training(cfg, *, run_dir: Path, resume: Path | None) -> dict` orchestrates preparation validation, sampling, fitting, evaluation, and checkpointing.

- [ ] Write distributed tests against a single-process summed-loss reference for keep masks `[1,1,1,1]`, `[1,0,1,0]`, `[0,0,0,1]`, and `[0,0,0,0]`. Assert identical parameter updates across ranks and reference parity after clipping.
- [ ] Add tests for unequal endpoint rewards, repeated prompts and historical tracking, globally rejected prompts, nonfinite target on one rank, zero gradients, and evaluation with eight prompts. Bound each multiprocess test with a timeout. Verify expected failures before implementing.
- [ ] Broadcast one prompt request per attempt and assign candidates by rank. Gather scalars, update the existing rank-zero tracker, broadcast masks/weights, and advance one global prompt cursor. All-rejected prompts skip the optimizer and EMA consistently.
- [ ] Add readiness exchanges before each policy-bearing phase, including staged model loading, request/layout preparation, rollout, baseline/evaluation, fitting, and restore. Inject one-rank failures before a forward and during a forward and verify bounded group termination without a published partial update.
- [ ] Construct local targets and synchronize status before replay. For N accepted candidates, every rank executes one replay/backward, using `loss_scale*branch_loss` or graph-connected zero. Reduce adapter gradients in deterministic order with SUM and divide once by N, check finite/nonzero, clip norm to 1, step, and EMA 0.99. Do not call `cosmos_training.fit_queries` here because its optimizer and normalization are local.
- [ ] Place VAE and geometry estimators on the paired auxiliary device. Reuse the isolated judge with a unique rank path and sanitized distributed environment. Differentiate local rewards across device copies and move only scalar scores between policy ranks.
- [ ] Implement collective-ordered calibration and base references with disabled adapters. Evaluate all ranks in equal rounds with stable prompt/seed assignments, and aggregate metrics/media through rank zero. Persist fixed baseline identities and generate audio output samples during evaluation.
- [ ] Run CPU tests and opt-in four-rank tests. Assert rejected ranks execute the same number of policy calls and that base parameters remain frozen. Record complete phase metrics rather than last-snapshot memory values.

Suggested commit `feat(video): train H3 with synchronized video OPSD updates`.

## Task 6 — Add atomic checkpoint and resume

**Files:** Create `h3_checkpoint.py`, `test_h3_checkpoint.py`, integrate trainer.

**Interfaces:**

- `save_h3_checkpoint(path: Path, policy: H3Policy, optimizer, *, state: dict, rank_state: dict, group) -> None` publishes a rank-zero CPU checkpoint and `latest.json` after all ranks confirm state agreement.
- `load_h3_checkpoint(path: Path, policy: H3Policy, optimizer, *, expected_signature: dict, group) -> tuple[dict, dict]` restores common and caller-rank state.

- [ ] Write an uninterrupted-two-updates versus save/restart-between-updates test. Compare adapters, optimizer moments, EMA, prompt cursor, tracker, RNG draws on both devices, and next candidate identities.
- [ ] Add tests for a torn write, stale latest pointer, mismatched model/cache/reward/geometry/runtime identity, divergent rank adapters, and changed rank topology. Confirm failures before implementing.
- [ ] Save only intentionally replicated adapter and optimizer tensors plus gathered per-rank RNG/control state. Reconstruct frozen DTensors from pinned safetensors. Validate hashes across ranks and publish the complete checkpoint before updating the latest pointer.
- [ ] Restore and broadcast common state before any baseline or policy generation. Preserve the offline logging experiment group while creating one new run per execution.
- [ ] Connect wall-time/termination flags to a collective safe boundary, and reserve time using the longest measured update duration. Terminate on a rank failure during collectives and recover from the last published update.
- [ ] Run `bash scripts/run_minimax_h3_python.sh -m pytest tests/video/test_h3_checkpoint.py -q` and its opt-in four-rank cases. Verify no base-model tensors appear in saved checkpoints.

Suggested commit `feat(video): resume sharded H3 adapter training`.

## Task 7 — Expose the normal training and Slurm entry points

**Files:** Create the H3 training Python/Slurm entry points,
`test_h3_entrypoint.py`, and `docs/minimax_h3_training.md`. Extend `README.md`.

**Interfaces:** CLI flags `--config`, `--run-dir`, `--resume PATH`,
`--preflight`, and `--max-updates-this-job`. The preparation CLI accepts
`--config`, `--cache-root`, `--prefetch`, `--condition`, and `--check-only`.

- [ ] Write subprocess CLI tests for preflight without GPU/model allocation, invalid workflow/geometry, absent cache, and injected tiny-model training through the entry point. Assert one logger owner and correct physical-to-visible GPU mapping.
- [ ] Implement `--preflight` to inspect versions, safetensors/index completeness, cache identity, rewards, topology, and the FSDP feature contract before loading. Report exact weight bytes and intended rank map.
- [ ] Request one node, eight H100s, 64 CPUs, 768 GiB RAM, and a configurable wall limit. Launch one Slurm task invoking `torchrun --standalone --nproc_per_node=4` with all eight assigned GPUs visible. Discover the valid site partition/account at execution time and record them in the job manifest.
- [ ] Add one offline W&B run per job execution through `VideoTrainingLogger`, canonical optimizer-update axes, rank-local diagnostic journals, eval media, and complete model/cache/reward/runtime provenance. Disable automatic online sync by default as in existing video trainers.
- [ ] Document prepare/preflight/train/resume commands and the four-policy/four-auxiliary topology. Clearly label estimated resources until Task 8 records real measurements.
- [ ] Run changed-file Ruff, shell syntax checks, the H3 CPU suite, and `python -m pytest tests/video -q` in the H3 runtime. Also run existing Wan/Cosmos focused tests in their validated runtime if shared modules changed.

Suggested commit `feat(video): launch eight-GPU H3 OPSD training`.

## Task 8 — Verify full-checkpoint training and report resources

**Files:** Finish `check_minimax_h3_update.py`, extend the operations document,
and retain measurement/audit artifacts under ignored `logs/`.

**Interfaces:** Verification CLI accepts `--config`, `--run-dir`, and
`--phase {load,rollout,target,update,resume}` and runs with four policy processes.

- [ ] Submit a short eight-H100 topology test. Verify peer collectives, policy/auxiliary ownership, frozen-backbone sharded loading, native inference, and per-phase profiler output. Confirm four candidate seeds and both schedules remain distinct and reproducible.
- [ ] Run one complete optimizer update with the actual H3 checkpoint at smoke geometry. Audit finite nonzero reward and policy gradients, changed `default`, valid EMA, accepted candidate count, identical rank adapter hashes, evaluation media, and published checkpoint.
- [ ] Resume that checkpoint for one additional update. Compare deterministic tiny-case restart tests and verify real-checkpoint update/RNG/cursor continuity.
- [ ] Run the same checks at `124x544x960`, 50 sigma grid points. Acceptance targets are measured device peaks below 70 GiB per card and host use within the 768 GiB allocation. Save both framework and NVML device peaks and timing per phase.
- [ ] Include the longest selected training and evaluation prompts in memory verification. Save paired base/current audio with identical prompt/seed, verify finite native-rate stereo waveforms and duration, and inspect audiovisual synchronization. Record audio preservation as a qualitative result.
- [ ] If measured peaks exceed the targets, profile allocations and adjust supported checkpointing, prefetch, or target activation offload. Repeat the same correctness tests and report the exact change. If activations still require sequence/tensor parallelism, record the failing phase and amend the design with the measured requirement before implementing that extension.
- [ ] Run five successful updates with checkpointing and held-out evaluation, then run the configured 50-update pilot. Save matched base/trained video rewards, identity/motion/quality metrics, and audio samples. Completion requires 50 optimizer updates, resume verification, and a final checkpoint audit. A positive quality result requires reporting the actual held-out comparison.
- [ ] Update `docs/minimax_h3_training.md` with job IDs, actual GPU models/capacity, runtime/model revisions, full geometry, selected sigmas, per-device peaks, host peak, seconds/update, candidates/update, and any changed defaults. Link audit artifacts through their local run directory.

Suggested commit `docs(video): record H3 OPSD GPU validation`.

## Plan verification and execution order

Execute Tasks 1–8 sequentially. Task 1 settles the FSDP/LoRA execution dependency
before building the trainer. Tasks 2–4 establish the native data/model/decoder
interfaces. Tasks 5–7 integrate global updates, restart, and the ordinary launch
path. Task 8 measures the full-scale behavior.

The design's mathematical contract maps to Tasks 3–5, its memory/loading contract
to Tasks 1, 4, and 8, its collective protocol to Tasks 5–7, and its reproducibility
contract to Tasks 2, 6, and 8. The planning deliverables are complete when the
design's builder check and this plan's interface/coverage self-review are resolved.
