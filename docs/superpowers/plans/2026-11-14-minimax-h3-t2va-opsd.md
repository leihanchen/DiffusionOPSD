# MiniMax H3 t2va OPSD implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task. Steps use checkbox syntax. The user delegated defaults and requested no questions. Resolve routine choices from the design and recorded verification. This document describes future implementation work.

**Goal:** Add reproducible MiniMax H3 `t2va` video-reward LoRA post-training with two policy GPUs and one target GPU per node initially, four candidates per node, and a qualified multi-node extension.

**Architecture:** P policy processes per node shard that node's frozen backbone with FSDP2 and keep synchronized FP32 LoRA replicas. R node-local target workers construct detached targets through VAE and reward models. Default P=2/R=1 executes four candidates in two microbatch-one rounds before one optimizer update. P=4 is the measured expansion. The multi-node extension synchronizes LoRA gradients across nodes and attempts four candidates per node per global update.

**Tech Stack:** PyTorch FSDP2, Diffusers modular H3, PEFT, safetensors, existing GeoFlow/VideoReward, Apptainer, torchrun, Slurm, offline W&B.

**Spec:** `docs/superpowers/specs/2026-11-14-minimax-h3-t2va-opsd-design.md`.

## Global constraints

- Default `num_nodes=1`, `policy_gpus=(0,1)`, `target_gpus=(2,)`, `candidates_per_node=4`, `policy_microbatch_size=1`. Request n=P+R H100 80 GB GPUs per node. Qualify P=2/R=1 first, compare P=4/R=1 for policy capacity/latency or P=2/R=2 for target throughput. Production presets use P in {2,4}, with at most eight GPUs per node.
- Start profiling with 32 CPU cores and 384 GiB host RAM, then size the training request from measured use. Verify partial-node scheduling and billing.
- Pinned model revision `0919fb684102ef115126612f3fbb84d1027e7317`, `t2va` workflow.
- Native deterministic Euler with both schedules, shifts 12/3, one joint forward per step.
- Production pilot geometry `124x544x960` at 24 fps and 50 sigma grid points.
- Four candidates for one prompt per node independent of P. P=2 runs two microbatch-one rounds and P=4 runs one. Accumulate all rounds before one optimizer/EMA step. At M nodes, the global attempted candidate batch is 4*M and gradients divide once by the global accepted count.
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
| `src/video/h3_rollout.py` | Native t2va query capture, replay, and suffix diagnostic |
| `src/video/h3_auxiliary.py` | Explicitly owned target workers, local differentiable decoding/targets, bounded tensor transport |
| `src/video/h3_training.py` | P-policy-rank sampling, masked multi-round fitting, evaluation protocol |
| `src/video/h3_checkpoint.py` | Atomic replicated-adapter checkpoint and rank state restoration |
| `config/minimax_h3_t2va.py`, `config/minimax_h3_t2va_smoke.py` | Pilot and reduced-canvas full-model smoke settings |
| `config/minimax_h3_t2va_multinode.py` | Two-node qualification with the same four candidates per node |
| `containers/minimax-h3-requirements.txt` | Verified isolated dependency versions |
| `scripts/setup_minimax_h3.sh`, `scripts/run_minimax_h3_python.sh` | Runtime preparation and container execution |
| `scripts/prepare_minimax_h3.py` | Model prefetch, streaming conditioning cache, asset preflight |
| `scripts/train_opsd_video_minimax_h3.py`, `scripts/train_opsd_video_minimax_h3.slurm` | Training entry point and configurable n-GPU allocation |
| `scripts/check_minimax_h3_update.py` | Stage-level full-checkpoint GPU verification and audit output |
| `tests/video/test_h3_{fsdp,conditioning,rollout,decode,auxiliary,training,checkpoint,entrypoint,multinode}.py` | Focused contracts for the corresponding units |
| `docs/minimax_h3_training.md` | Setup, commands, resources, resume, measured results |

Reuse the existing target builder, branch loss, gates, dataset loader, logger,
reward, and isolated judge. Extend `src/video/cosmos_quality.py` and
`scripts/cosmos_quality_worker.py` with an opt-in versioned CPU-residency protocol
for H3, with regression coverage for existing score requests. Give each judge a
separate reward-service directory. Add the H3
usage link to `README.md` in Task 7. Add shared changes only if a concrete H3
caller needs them and the existing callers have regression coverage.

## Task 1 — Prove the sharded policy and runtime contract

**Files:** Create `h3_fsdp.py`, the two runtime scripts, dependency requirements,
`test_h3_fsdp.py`, and the initial verification script.

**Interfaces:**

- `init_h3_distributed(*, policy_gpus: tuple[int, ...], target_gpus: tuple[int, ...]) -> H3Devices`, with `rank`, `role`, `policy_count`, `reward_ranks`, `device`, `policy_group`, and `control_group`. Validate disjoint scheduler-visible indices and n=P+R. All n ranks create groups in a common order. Only P policy ranks participate in FSDP/adapter collectives. All ranks share Gloo control and CPU tensor transport. Initial execution uses R=1.
- `load_h3_policy(model_path: Path, revision: str, devices: H3Devices, *, rank: int, alpha: int) -> H3Policy`.
- `H3Policy.model` is callable, `trainable_parameters` is a deterministic tuple, `select_adapter(name)` preserves the trainability mask, and `ema_(decay)` updates replicated `old` tensors.
- `reduce_adapter_gradients_(parameters, group, kept_count: int) -> None` performs SUM/N reduction and finite validation. `fit_h3_update` owns preclip-norm measurement and clipping after reduction.

- [ ] Write `test_frozen_fsdp_backbone_propagates_ignored_lora_gradients`, `test_streamed_loading_matches_unsharded_weights`, and `test_adapter_switch_preserves_trainability`. Use a tiny real H3 transformer with two modalities. Assert nonzero LoRA gradients, frozen base tensors, native output parity, and identical adapter replicas after update.
- [ ] Run the CPU serialization and module-selection cases and observe the expected missing-interface failure. Run n=3 and n=5 distributed cases under GPU allocations, with two/four policy ranks and one reward rank. Mark GPU cases with `pytest.mark.h3_gpu` and explicit opt-in so ordinary CPU suites skip them cleanly.
- [ ] Implement bottom-up frozen-weight FSDP2 with both adapters in `ignored_params`, original checkpoint dtypes, explicit FP32 adapter placement/broadcast, checkpointing, and `reshard_after_forward=True`. Avoid whole-model GPU loading. Stream checkpoint slices matching DTensor offsets and validate sharded padding on uneven dimensions. Test FP32 projection leaves and the root's remaining parameters.
- [ ] Reconstruct nonpersistent/constructor-derived buffers after materialization, including FP32 rotary frequencies. Compare buffers and loaded tiny-model outputs against ordinary `from_pretrained`. Assert frozen base DTensors have no gradients and adapters remain ordinary replicated tensors with non-reentrant checkpointing enabled.
- [ ] Probe the installed Torch 2.14 container first. Confirm that its `fully_shard` actually exposes `ignored_params`, and that gradients through the frozen sharded blocks work in this arrangement. Pin a compatible H3 Diffusers/Transformers/PEFT stack in a separate runtime. If a newer Torch is technically necessary, prepare and verify a dedicated H3 image and record the reason and exact versions.
- [ ] Verify one tiny forward/backward/update with four independent candidates against an unsharded four-candidate reference at P=2 and P=4. Test ignored gradients are synchronized exactly once, service rank exclusion, repeated behavior/default switching, and a forward after evaluation. P=2 must accumulate candidates {0,2} and {1,3} on the respective ranks before one optimizer step.
- [ ] Run `bash scripts/run_minimax_h3_python.sh -m pytest tests/video/test_h3_fsdp.py -q` and its n=3/n=5 GPU cases. Assert zero missing/unexpected checkpoint tensors and no full-model GPU allocation during loading.

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
- [ ] Set the design's exact pilot defaults, including `policy_microbatch_size=1` and four candidates per node. Set smoke geometry to `124x256x256`, 16 sigma grid points, one update, and four evaluation records over P policy ranks. Test that changing P from 2 to 4 changes rounds from 2 to 1 and preserves the local logical batch and optimizer-step count.
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
- [ ] Supply configuration-only VAE attributes for native policy-side layout/geometry setup. Test that policy workflow construction loads no VAE weights and that its descriptor matches the actual reward-rank VAE config. Preserve native compression, chunking, normalization, and generated-row layout semantics.
- [ ] Add scoped capture at pre-step q for both modalities, then direct differentiable transformer replay through `model(...)`. Keep all structural fields, native normalization, and batch dimensions. Preserve the native mixed-precision forward and compute anchors in FP32.
- [ ] Implement the suffix diagnostic restoring both states/schedulers. Force video clean output at q, use the saved old audio output at q, and jointly recompute subsequent steps.
- [ ] Run `bash scripts/run_minimax_h3_python.sh -m pytest tests/video/test_h3_rollout.py -q`. Require exact native scheduler outputs with identical seeds/precision and explicit tolerances for replay arithmetic. Test baseline suffix parity and a changed-video-target suffix.

Suggested commit `feat(video): replay native H3 joint queries for OPSD`.

## Task 4 — Verify differentiable video target construction

**Files:** Extend `h3_rollout.py`, create `h3_auxiliary.py`,
`test_h3_decode.py`, and `test_h3_auxiliary.py`, extend the GPU check script and
the isolated judge/worker's opt-in protocol.

**Interfaces:** `H3RewardService.decode01(video_latents: torch.Tensor) -> torch.Tensor`
returns `[B,T,3,H,W]`. `H3RewardService.construct_targets(anchor, cfg) -> H3Targets`
uses `opa_tr_step_nd` and returns detached FP32 positive/negative targets plus
gradient/trust-region diagnostics. `H3RewardClient.request(kind, candidate_id,
payload, metadata) -> dict` provides versioned, bounded CPU/Gloo transport for
endpoint/anchor scoring, targets, and evaluation decoding. One reward process
owns the GPU and models. `park()` on the isolated judge acknowledges CPU residency
and released GPU reservations before conflicting service work.

- [ ] Write tiny native H3 VAE tests for unpacking, normalization inversion, native output parity, finite nonzero latent gradients, detached trust-region targets, and `[B,T,3,H,W]` range/shape.
- [ ] Run them and verify the new decode interface fails before implementation.
- [ ] Call the native VAE directly outside modular inference decorators. Enable native decoder gradient checkpointing and tiling. Apply the existing nine-frame reward selection, resize to `224x384`, and preserve the current `reward_metadata()` contract.
- [ ] Ensure target tensors are FP32, local decoder/reward graphs are freed after each service request, and only accepted candidates construct targets. Transport detached CPU latents and targets. Keep all VAE/reward differentiation within the reward process using a fresh latent leaf. Record the existing derivative policy in manifests.
- [ ] Test `test_reward_service_matches_local_targets`, `test_request_identity_and_timeout`, `test_judge_park_acknowledges_release`, and `test_legacy_judge_score_protocol`. Assert one service model copy, one candidate graph at a time, bounded queue memory, exact candidate-response matching, and correct returned target shape/dtype.
- [ ] Keep reward components resident if the measured service GPU budget permits. Start the isolated judge parked until scoring is needed. If components conflict, park at graph-free phase boundaries, synchronize work, release model/output storage and subprocess allocator reservations, and verify release through NVML.
- [ ] Record payload bytes, host-transfer time, service queue wait, and per-phase device residency. Inject a service error before policy replay and require an error response and coordinated termination before any optimizer step.
- [ ] Run `bash scripts/run_minimax_h3_python.sh -m pytest tests/video/test_h3_decode.py tests/video/test_opa_video.py tests/video/test_train_step_math.py -q`. Check ascent/descent direction and `||target-anchor|| <= 0.10*||anchor||` with numerical tolerance.
- [ ] Add full-size target-phase profiling for the native VAE, including parent and quality subprocess GPU allocations. Test native tiled and untiled modes separately before selecting any memory fallback. Record host overhead if CPU saved-tensor offload is enabled.

Suggested commit `feat(video): construct reward targets through H3 video decoder`.

## Task 5 — Implement collective-safe global OPSD updates

**Files:** Create `h3_training.py` and `test_h3_training.py`, extend policy tests.

**Interfaces:**

- `collect_candidates(policy: H3Policy, rollout: H3Rollout, request: dict, *, rank: int, policy_count: int) -> list[H3Candidate]`, with canonical slot/candidate identity and explicit dummy markers.
- `fit_h3_update(policy: H3Policy, rollout: H3Rollout, candidates: list[H3Candidate], *, targets: dict[int, H3Targets], weights: dict[int, float], global_kept: int, optimizer, cfg) -> dict` accumulates equal slot counts across policy ranks. Target construction has completed on the service.
- `run_h3_training(cfg, *, run_dir: Path, resume: Path | None) -> dict` orchestrates preparation validation, sampling, fitting, evaluation, and checkpointing.

- [ ] Write distributed tests against a single-process summed-loss reference for keep masks `[1,1,1,1]`, `[1,0,1,0]`, `[0,0,0,1]`, and `[0,0,0,0]`. Test P=2/4 scheduling, unique ownership of four real candidates, equal collective counts per policy rank, and reference update parity after clipping. Assert accepted count normalization occurs once after both local rounds at P=2.
- [ ] Add tests for unequal endpoint rewards, repeated prompts and historical tracking, globally rejected prompts, nonfinite target on one rank, zero gradients, and evaluation with eight prompts. Bound each multiprocess test with a timeout. Verify expected failures before implementing.
- [ ] Broadcast one prompt request per attempt and assign candidate `round*P+node_local_policy_rank` for K/P rounds. Gather scalar service responses, update the existing rank-zero tracker, broadcast masks/weights, and advance one global prompt cursor. All-rejected prompts skip optimizer and EMA consistently. The P=2/4 production presets have no out-of-range slots, while rejected slots use zero-loss replay.
- [ ] Add readiness exchanges before each policy-bearing phase, including staged model loading, request/layout preparation, rollout, baseline/evaluation, fitting, and restore. Inject one-rank failures before a forward and during a forward and verify bounded group termination without a published partial update.
- [ ] Route accepted detached anchors to the shared service in canonical order and return detached targets to their owner. All n ranks exchange completion status before fitting. Every policy rank executes ceil(4/P) replay/backward slots, using `loss_scale*branch_loss` or graph-connected zero. After all slots, SUM adapter gradients over the policy group, divide once by N, check finite/nonzero, clip to 1, step, and EMA 0.99. The service never enters those collectives.
- [ ] Keep VAE, geometry estimators, and isolated judge on reward rank P's GPU with its own directory and sanitized subprocess environment. Test the service's local VAE/reward gradient against direct computation and confirm transported targets need no cross-process autograd. Bound pending requests and intermediate graphs to one candidate.
- [ ] Implement collective-ordered calibration and base references with disabled adapters. Evaluate all ranks in equal rounds with stable prompt/seed assignments, and aggregate metrics/media through rank zero. Persist fixed baseline identities and generate audio output samples during evaluation.
- [ ] Run CPU tests and opt-in n=3/n=5 tests. Assert rejected/padded policy ranks execute the same number of policy calls, service rank executes no transformer calls, and base parameters remain frozen. Record complete phase metrics rather than last-snapshot memory values.

Suggested commit `feat(video): train H3 with synchronized video OPSD updates`.

## Task 6 — Add atomic checkpoint and resume

**Files:** Create `h3_checkpoint.py`, `test_h3_checkpoint.py`, integrate trainer.

**Interfaces:**

- `save_h3_checkpoint(path: Path, policy: H3Policy | None, optimizer, *, state: dict, rank_state: dict, policy_group, control_group) -> None` publishes a rank-zero CPU checkpoint and `latest.json`. The service passes `policy=None, optimizer=None`. Policy hashes use the policy group, and rank-state gathering/publication status use all-rank control.
- `load_h3_checkpoint(path: Path, policy: H3Policy | None, optimizer, *, expected_signature: dict, policy_group, control_group) -> tuple[dict, dict]` restores common and caller-rank state. Adapter/optimizer broadcasts exclude the service, which restores only service/control state.

- [ ] Write an uninterrupted-two-updates versus save/restart-between-updates test. Compare policy adapters, optimizer moments, EMA, prompt cursor, tracker, per-rank RNG, service RNG/control state, and next candidate identities. Use explicit per-candidate sampling generators and test P=2/4 resume with the same topology.
- [ ] Add tests for a torn write, stale latest pointer, mismatched model/cache/reward/geometry/runtime identity, divergent rank adapters, and changed rank topology. Confirm failures before implementing.
- [ ] Share topology validation between preflight and checkpoint restore. Require matching M/P/R, explicit per-node GPU-role lists, candidates per node, and policy microbatch size for exact resume. Exercise rejected changes to each field alongside valid M=1 and M=2 signatures.
- [ ] Test the service rejoins all-rank checkpoint and evaluation readiness/completion exchanges with no policy or optimizer objects, and never enters a policy-group collective or policy backward.
- [ ] Save intentionally replicated adapter/optimizer tensors after hash agreement among policy ranks. Gather all ranks' RNG/control state, including the service. Reconstruct frozen DTensors from pinned safetensors and publish the complete checkpoint before updating the latest pointer.
- [ ] Restore and broadcast common state before any baseline or policy generation. Preserve the offline logging experiment group while creating one new run per execution.
- [ ] Connect wall-time/termination flags to a collective safe boundary, and reserve time using the longest measured update duration. Terminate on a rank failure during collectives and recover from the last published update.
- [ ] Run `bash scripts/run_minimax_h3_python.sh -m pytest tests/video/test_h3_checkpoint.py -q` and its opt-in n=3/n=5 cases. Verify no base-model tensors appear in saved checkpoints.

Suggested commit `feat(video): resume sharded H3 adapter training`.

## Task 7 — Expose the normal training and Slurm entry points

**Files:** Create the H3 training Python/Slurm entry points,
`test_h3_entrypoint.py`, and `docs/minimax_h3_training.md`. Extend `README.md`.

**Interfaces:** CLI flags `--config`, `--run-dir`, `--resume PATH`,
`--preflight`, and `--max-updates-this-job`. The preparation CLI accepts
`--config`, `--cache-root`, `--prefetch`, `--condition`, and `--check-only`.

- [ ] Write subprocess CLI tests for preflight without GPU/model allocation, invalid workflow/geometry, absent cache, and injected tiny-model training through the entry point. Assert one logger owner and correct physical-to-visible GPU mapping.
- [ ] Implement `--preflight` to inspect versions, safetensors/index completeness, cache identity, rewards, topology, and the FSDP feature contract before loading. Report exact weight bytes and intended rank map.
- [ ] Default qualification to one node, three H100s, 32 CPUs, 384 GiB RAM, and a configurable wall limit. Read explicit `policy_gpus` and `target_gpus` from config, validate P+R against scheduler-visible resources, and launch n processes with `torchrun --standalone --nproc_per_node="$H3_TOTAL_GPUS"`. Target workers occupy the final R ranks and policy ranks form their own group. Record valid site partition/account and partial-node billing. Reject an allocation/config mismatch before model loading.
- [ ] Add one offline W&B run per job execution through `VideoTrainingLogger`, canonical optimizer-update axes, rank-local diagnostic journals, eval media, and complete model/cache/reward/runtime provenance. Disable automatic online sync by default as in existing video trainers.
- [ ] Document prepare/preflight/train/resume commands and explicit P/R placement. Show P=2/R=1 and P=4/R=1 examples. Explain scaling target workers independently from policy sharding. Clearly label estimated resources until Task 8 records measurements. Report whole-node charges for partial-node requests when applicable.
- [ ] Run changed-file Ruff, shell syntax checks, the H3 CPU suite, and `python -m pytest tests/video -q` in the H3 runtime. Also run existing Wan/Cosmos focused tests in their validated runtime if shared modules changed.

Suggested commit `feat(video): launch resource-qualified H3 OPSD training`.

## Task 8 — Verify full-checkpoint training and report resources

**Files:** Finish `check_minimax_h3_update.py`, extend the operations document,
and retain measurement/audit artifacts under ignored `logs/`.

**Interfaces:** Verification CLI accepts `--config`, `--run-dir`, and
`--phase {load,rollout,target,update,resume}` and runs with n processes including
the dedicated reward rank.

- [ ] Submit a short three-H100 test with two policy ranks and one reward rank. Verify peer collectives, service exclusion from FSDP, bounded request/response transport, frozen-backbone sharded loading, native inference, and per-phase profiler output. Confirm four candidate seeds and both schedules remain distinct and reproducible.
- [ ] Run one complete optimizer update with the actual H3 checkpoint at smoke geometry. Audit finite nonzero reward and policy gradients, changed `default`, valid EMA, accepted candidate count, identical rank adapter hashes, evaluation media, and published checkpoint.
- [ ] Resume that checkpoint for one additional update. Compare deterministic tiny-case restart tests and verify real-checkpoint update/RNG/cursor continuity.
- [ ] Run the same checks at `124x544x960`, 50 sigma grid points. Acceptance targets are combined per-device peaks below `min(70 GiB, 0.90 * device capacity)` and host use within the allocation. Save both framework and NVML device peaks and timing per phase. Size subsequent CPU/RAM requests from measured cgroup/process use.
- [ ] Include the longest selected training and evaluation prompts in memory verification. Save paired base/current audio with identical prompt/seed, verify finite native-rate stereo waveforms and duration, and inspect audiovisual synchronization. Record audio preservation as a qualitative result.
- [ ] If measured peaks exceed the targets, profile allocations and adjust supported checkpointing, prefetch, or target activation offload. Repeat the same correctness tests and report the exact change. If activations still require sequence/tensor parallelism, record the failing phase and amend the design with the measured requirement before implementing that extension.
- [ ] Benchmark n=5 when n=3 policy memory or measured runtime warrants it. Hold the four-candidate experiment fixed. Report complete-update seconds and active/billed GPU-hours including rejection, request transport, queue wait, evaluation, and checkpoints. Five-card use improves allocated GPU-hours only if update time falls below 3/5 of the three-card time. Record whether any selection instead prioritizes wall time.
- [ ] Measure reward-service utilization and serialized decoder/reward critical-path time before expanding policy allocation from two to four. Keep the production presets at most four policy GPUs per node for K=4. Use node-local additional batches through Task 9 for multi-node scaling. Reward batching or asynchronous overlap requires separate correctness and throughput tests.
- [ ] If targets dominate, compare P=2/R=2 against P=2/R=1 at fixed four-candidate workload. Route candidate i deterministically to worker i%R, use bounded per-worker channels, and serialize each worker's local graph. Verify target/update parity, request failures, per-worker checkpoint/RNG state, and all-rank readiness before fitting. Keep the simpler R=1 implementation as the initial qualification baseline.
- [ ] Report a topology comparison with policy seconds, target seconds, queue/transport time, per-role memory, wall time/update, and billed GPU-hours/update. Increase only the pool that limits measured capacity or throughput. Treat multiple independent differentiable reward gradients as a weighted sum before OPSD normalization, with a focused equivalence test if that feature is added.
- [ ] Run five successful updates with checkpointing and held-out evaluation, then run the configured 50-update pilot. Save matched base/trained video rewards, identity/motion/quality metrics, and audio samples. Completion requires 50 optimizer updates, resume verification, and a final checkpoint audit. A positive quality result requires reporting the actual held-out comparison.
- [ ] Update `docs/minimax_h3_training.md` with job IDs, actual GPU models/capacity, runtime/model revisions, full geometry, selected sigmas, per-device peaks, host peak, seconds/update, candidates/update, and any changed defaults. Link audit artifacts through their local run directory.

Suggested commit `docs(video): record H3 OPSD GPU validation`.

## Task 9 — Scale node-local batches across nodes

**Files:** Extend `h3_fsdp.py`, `h3_training.py`, `h3_auxiliary.py`,
`h3_checkpoint.py`, both training entry points, and the operations document.
Create `config/minimax_h3_t2va_multinode.py` and `test_h3_multinode.py`.

**Interfaces:**

- Extend `H3Devices` with `node_rank`, `node_count`, `node_policy_rank`, `node_policy_group`, `all_policy_group`, and `node_control_group`. Preserve `policy_group` as the node-local FSDP group. `all_policy_group` contains M*P policy ranks and excludes every target rank. `control_group` spans all ranks for global readiness/publication status.
- `h3_candidate_assignments(attempt: int, node_rank: int, node_count: int, policy_rank: int, policy_count: int, record_order: list[int]) -> list[tuple[int,int]]` returns stable `(record_index,candidate_index)` pairs. Each node owns one prompt and candidates 0..3.
- Extend `run_h3_training` and checkpoint APIs with explicit M/P/R topology and separate node-local versus global groups. Default M=1 retains the single-node execution path.

- [ ] Write pure scheduling tests for M=1/2/4 with P=2/4. Assert four unique candidates per node, distinct node prompts when the pool has at least M records, two local rounds for P=2, one for P=4, and global candidate count 4*M.
- [ ] Write a two-node-group tiny-model test against a single-process reference using the same fixed candidate queries, targets, weights, and global loss. Test unequal accepted counts, one entirely rejected node, and a globally empty batch. Require SUM/global-N normalization, identical adapter/optimizer/EMA state on all policy ranks, and exactly one global optimizer step.
- [ ] Create FSDP meshes within each node and an all-policy gradient group across nodes. Generate and score each node's candidates with its own target workers. Accumulate unnormalized local-round gradients, reduce once over the all-policy group, divide by global N, then clip/step/EMA. A node with zero accepted candidates participates in zero-loss replay if global N is positive.
- [ ] Have global rank zero assign node j record `(attempt*M+j) % pool_size` from one deterministic permutation and own the existing tracker/calibration/baselines. Gather scalar records across nodes before computing weights. Candidate requests carry node slot, record ID, attempt, and candidate index so node-local workers cannot collide. Use bounded node-local transport and globally coordinated status before fitting.
- [ ] Add interrupted-two-node-update recovery and a single-node service-failure test. Persist every rank's RNG, each node's sampling position, target state, global tracker, and topology. Verify adapter/optimizer hashes over all policy ranks and publish one complete rank-zero checkpoint. Initial exact resume requires the same M/P/R.
- [ ] Distribute eight global evaluation records over M*P policy ranks using ceil(8/(M*P)) equal rounds and dummy forwards for out-of-range slots. Baseline/current assignments match. Aggregate each real record exactly once and give every media path a unique global evaluation index. Test M=2/P=2 and M=4/P=4, requiring eight real evaluation records in both cases.
- [ ] Launch one torchrun agent per Slurm node with explicit shared rendezvous, unique node rank, and P+R processes per node. Add CLI tests for M=1 and M=2 mappings and invalid resource/topology combinations. Preserve rank-zero logging and collect node-local phase metrics and evaluator outputs without file collisions.
- [ ] Run opt-in two-node tiny-model tests, then a full-checkpoint two-node smoke and resume after single-node qualification. Use P=2/R=1 per node initially, giving eight global candidates. Report global batch, accepted count, per-node memory, wall time, and billed GPU-hours. Keep the default 50-update pilot at M=1 and use the multi-node preset only after its smoke passes.

Suggested commit `feat(video): synchronize H3 OPSD batches across nodes`.

## Plan verification and execution order

Execute Tasks 1–8 sequentially for the default single-node pilot, then Task 9
qualifies the multi-node extension. Task 1 settles the FSDP/LoRA execution dependency
before building the trainer. Tasks 2–4 establish the native data/model/decoder
interfaces. Tasks 5–7 integrate global updates, restart, and the ordinary launch
path. Task 8 measures the full-scale behavior.

The design's mathematical contract maps to Tasks 3–5, its memory/loading contract
to Tasks 1, 4, and 8, its collective protocol to Tasks 5–7, and its reproducibility
contract to Tasks 2, 6, and 8. Node-local batch ownership, cross-node reductions,
and distributed launch/recovery map to Task 9. The planning deliverables are complete when the
design's builder check and this plan's interface/coverage self-review are resolved.
