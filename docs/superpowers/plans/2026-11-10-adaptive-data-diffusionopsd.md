# Feedback-Guided Training Data Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or, when selected, `superpowers:subagent-driven-development` to implement this plan task-by-task. Track steps with checkboxes.

**Goal:** Build a small, resumable three-round video-training pilot in which one frozen Gemma 4 12B-it model diagnoses clips and revises a mixture of retained, edited, rehearsal, and new prompts after each complete prompt pass.

**Architecture:** Reuse the Cosmos trainer, geometry reward, branch loss, EMA, logging, and native continuation. Add validated prompt manifests, round-level score summaries, a shared JSON-file VLM worker with diagnosis and proposal modes, and a sequential runner. Establish the basic loop first; add budgeted target diagnostics as a separate experimental arm.

**Tech Stack:** Python, dataclasses, JSON/JSONL, PyTorch, PEFT, pytest, existing Cosmos container/runtime, and a dedicated local Transformers VLM worker. No new serving framework is required by this plan.

**Spec:** [Research proposal](../../research/2026-11-10-adaptive-data-diffusionopsd-proposal.md).

Revised November 14, 2026 after repository and resource inspection. This replaces the earlier 12-task roadmap with a bounded first implementation. It incorporates the subsequent discussion about one prompt pass per round, mixed prompt sources, and avoiding unsupported semantic pass labels.

## Global Constraints

- Support Narval nodes with **4 × A100 40 GB per node**, and a second environment with **8 H200 nodes**. H200 GPUs per node, GPU memory, interconnect, and scheduler entitlement must be discovered before allocation; eight nodes must not be interpreted as eight GPUs per node.
- Select and record an explicit execution profile; never sum GPU memory as though it were one device or silently change model, precision, or effective training batch between environments.
- Hold prompts fixed within an inner training round.
- Generate fresh query states and targets after changing a prompt.
- Keep curriculum-development feedback separate from final-test results.
- Preserve the current positive/negative fitting objective and gates in the initial experiment.
- Split scene/event lineages before augmentation; descendants inherit their parent's partition.
- Count all generation, reference, evaluation, diagnostic, and proposer costs, including rejected attempts.
- Store generated prompts, videos, checkpoints, and logs under ignored run directories.
- A geometry score or DINO identity proxy must not be reported as certified camera motion, object stationarity, identity, or physical accuracy.
- Run target comparisons through the original frozen behavior snapshot, with saved scheduler history and matched randomness.

## Review Focus

1. Resume must not skip or double-apply a training update; checkpoints own committed progress (Tasks 2 and 5).
2. An all-rejected batch advances prompt visitation without an optimizer or EMA update (Task 2).
3. Prompt edits must not cross development/test lineages or introduce duplicate scenes (Tasks 1 and 4).
4. A changing EMA must not invalidate paired continuation measurements (Task 6).
5. Sparse, missing, or conflicting scores must remain uncertain rather than becoming semantic failure claims (Tasks 3 and 4).

## Summary and proposed pilot settings

| Item | First implementation |
| --- | --- |
| Backbone | Cosmos3-Nano with existing native-UniPC settings |
| Domain | G1: static rigid objects/scenes with camera motion |
| Prompt coverage | Camera orbit, dolly, and lateral translation variants; these are subtypes of G1 |
| Training manifest | 32 distinct prompts per round |
| Sampling | Four fresh videos per prompt, one shuffled pass |
| Training rounds | Three, after a two-round minimal smoke test |
| Fitting | One existing batch update per visited prompt if any clips survive |
| Round budget | 128 training clips attempted; at most 32 successful optimizer updates |
| Round-zero diagnostic development | 8 fixed prompts, 2 paired seeds each |
| Final test | 16 fixed prompts, 2 seeds each; evaluate only after choices are locked |
| Next-round mix | 8 retained unresolved, 12 controlled edits, 4 rehearsal, 8 new |
| Shared VLM | `google/gemma-4-12B-it`, frozen; separate diagnosis and proposal requests |
| Larger-resource comparison | `Qwen/Qwen3.8-27B` on a verified H200 allocation; separate audited arm, not an automatic replacement |
| Default execution | Narval: 4 × A100 40 GB/node; Cosmos uses 2 GPUs per worker, then Gemma uses 1 released GPU |
| Alternate execution | 8-node H200 environment; discover GPUs/memory per node before selecting placement and concurrency |
| VLM GPU target | Narval: 1 × A100 40 GB; H200: 1 verified device if memory smoke passes; BF16, batch size one, bounded inputs |
| Visual input | 8 ordered frames/clip including endpoints, longest side at most 448 pixels; timestamps retained |
| Diagnosis | Greedy, max 1,024 output tokens; visible evidence and unknown judgments |
| Proposal | Temperature 0.7/top-p 0.9; seeded, 4 candidates/request, max 2,048 output tokens |
| Initial feedback | Numeric metrics plus VLM observations after a 24-clip independent diagnosis audit; numeric-only control |
| Optional diagnostic arm | Four audited training queries per round |

These counts are proposed engineering defaults, not statistically validated sample sizes. This is an operational pilot, not evidence sufficient for all proposal hypotheses. Report uncertainty; do not declare mastery from four successful generations. Runtime is measured in the smoke test before assigning a full GPU-hour cap.

Within a round, visit every prompt exactly once. Generate its clips immediately before fitting; refresh the EMA after each successful update. An all-rejected prompt is logged and skipped, not silently replaced with more training attempts. Thus the sampling budget is fixed while successful updates can vary. Keep the existing legacy trainer behavior when curriculum mode is absent.

The development suite measures round-to-round progress on other scene/event lineages. The same frozen VLM in text mode may author it in separate requests before training. Freeze prompts, rubrics, and paired seeds, and pass only aggregate development statistics to curriculum synthesis. A final test is independent of these adaptive decisions.

## Execution environments and alternatives

The hardware inventory below is user-supplied. Availability does not imply that all nodes can be allocated concurrently. Current code inspection establishes a two-device Cosmos layout, not distributed Cosmos training across nodes.

| Profile | Hardware constraint | Training placement | Frozen diagnosis/proposal model | Intended use |
| --- | --- | --- | --- | --- |
| `narval_gemma12` (first implementation) | 4 × A100 40 GB per node | One Cosmos worker: policy on local GPU 0; VAE/reward/judge on local GPU 1 | `google/gemma-4-12B-it`, BF16, batch 1, on one GPU after training/evaluation exits | Baseline, smoke, and first complete pilot; use only 2 allocated GPUs when sequential phases suffice |
| `narval_parallel_arms` (optional) | Same four-device node; requires exclusive, non-overlapping device assignments | Two independent two-GPU workers, one per experimental arm; each writes its own run directory | Same Gemma 12B; VLM phases wait until their assigned devices are released | Run control/adaptive arms or seeds concurrently; neither worker is a distributed replica of the other |
| `h200_gemma12_control` | 8 H200 nodes available in the environment; per-node inventory and allocation quota unconfirmed | Start with the same two-device layout on one node if at least two GPUs/node are verified | Same pinned Gemma 12B and input/decoding limits as Narval | Hardware transfer smoke and matched-model comparisons; extra nodes run independent seeds/arms |
| `h200_qwen27_comparison` (optional) | Same H200 environment; verify per-device memory and runtime first | Preserve the validated Cosmos layout and sequential phase boundary | `Qwen/Qwen3.8-27B`, BF16, batch 1, initially one device only if measured memory permits | Separate model-capacity comparison after its own 24-clip diagnosis audit |

**Recommended progression:** run `narval_gemma12` first, transfer that exact configuration to `h200_gemma12_control`, then consider the Qwen comparison. The H200 configuration is an alternative execution target, not a prerequisite for implementing the pilot. If only one GPU per H200 node is available, the existing two-device Cosmos layout is not directly runnable there: validate a separately configured single-device layout before use, or keep training on Narval. Cross-node tensor/model parallelism and distributed Cosmos optimization require a separate implementation plan.

**Memory and input alternatives.** Keep the initial VLM budget identical across hardware: 8 ordered frames, longest side 448, input-token cap 8,192, diagnosis output cap 1,024 and proposal output cap 2,048. Gemma's published approximate 26.7 GB BF16 inference estimate is not a measured peak for this workload. Qwen's roughly 54–56 GB of BF16 weights alone rules out a single A100 40 GB without changing precision/placement; the four A100s do not form one 160 GB device. On H200, measure full model/vision/KV-cache peak and available headroom before declaring fit. A reduced-context Gemma profile or quantized/model-sharded Qwen profile is a separately recorded alternative requiring runtime support, memory smoke, and diagnosis re-audit; none is the default or assumed implemented. Increase frames, context, or request batch only in a separately named ablation after the matched-budget comparison.

**Parallelism alternatives.** Initially scale across independent runs, not within one training run. On H200, assign one isolated run per allocated node when its local device count supports the layout; eight nodes is an upper inventory bound, not a promise of eight simultaneous jobs. Keep training, development evaluation, and VLM processing sequential inside each run. A dedicated VLM node/service or overlapping VLM requests from multiple runs would require explicit request ownership, scheduling, and recovery support beyond the initial JSON-file worker; defer it. The existing distributed Wan trainer does not establish distributed support for this Cosmos pilot.

**Comparison constraints.** Across matched profiles retain 32 prompts/round, 4 clips/prompt, 3 rounds, one manifest pass, at most 32 optimizer updates/round, adapter/EMA settings, initialization, seeds, and diagnostic/test partitions. Additional nodes increase independent replication, not prompt count or effective optimization batch. Record unavoidable device-dependent numerical differences; do not promise bitwise GPU parity. Changing model, quantization, visual budget, global batch, or gradient accumulation defines a new experimental arm. Within an arm, both VLM roles use the same frozen revision throughout all rounds; no mid-run model switching.

### Execution-profile contract and preflight

Store named profiles in `config/curriculum_cosmos.py`; resolve environment-specific values into an immutable run receipt before starting Task 5. Required fields are `profile_name`, `cluster`, `nodes_per_run`, `gpus_per_node`, `gpu_memory_gb`, `train_device_map`, `vlm_device_ids`, `vlm_model_id`, `vlm_revision`, `vlm_dtype`, `vlm_batch_size`, `frame_limit`, `frame_longest_side`, `input_token_cap`, `diagnosis_output_cap`, `proposal_output_cap`, `runtime_paths`, and `asset_paths`. Keep scheduling concurrency (`max_concurrent_runs`) separate from GPUs within one run. Unknown inventory may be represented in an unselected profile, but execution rejects an unresolved selected profile.

Before allocating the full pilot:

1. Inspect a real allocation on each chosen environment: GPU model/count/visible memory, local device IDs, driver/CUDA/runtime compatibility, CPU/RAM, scratch/shared storage, and scheduler account/partition/time limits. Record interconnect only if a later placement depends on it.
2. Validate that each per-phase device map fits the allocated local devices, that simultaneous independent runs have disjoint assignments, and that model loading matches actual per-device memory. Never launch cross-node Cosmos training just because eight nodes exist.
3. Map immutable model/reward assets and dedicated runtimes to paths visible on that environment. The Narval `/lustre06/...` paths below are not portable H200 defaults; stage pinned assets or configure equivalent shared paths.
4. Complete training/evaluation and both VLM-role GPU smokes on each selected profile. Save peak memory, latency, model/runtime versions, device inventory, and the resolved profile hash. No such GPU execution has yet been verified in this worktree.
5. Set a measured cost cap and concurrency limit. Report GPU-hours separately for A100 and H200, along with wall time, attempts, and utilization; equal GPU-hours across GPU types are not equal compute or cost. Stop at phase boundaries when the cap is reached.

## Readiness audit

### Verified resources and constraints

Inspected on November 14, 2026:

| Resource | Observation | Consequence |
| --- | --- | --- |
| Training modules | `cosmos_rollout.py`, `cosmos_training.py`, `opa_video.py`, `branch_loss.py`, gates, checkpoints, and VideoTrainingLogger exist | Reuse the inner learning loop |
| Training schedule | Cosmos uses a successful-update budget and a cycling PromptCursor | Add one-pass curriculum mode; current behavior is not a full-manifest epoch |
| Local worktree runtime | No `containers/diffusionopsd.sif` or `logs/cosmos-runtime/python` here | Point wrappers to shared absolute paths |
| Shared container/runtime | Present under `/lustre06/project/6062393/leihan/DiffusionOPSD/` | Run dependency/preflight checks through that runtime |
| Shared assets | Cosmos3-Nano, DA3, WAFT, DINOv2, VideoReward, and VideoAlign directories exist | Presence is verified; live loading and GPU execution need separate evidence |
| Cosmos preflight | Passed with shared paths: torch 2.14.0+cu130, diffusers 0.42.1, transformers 5.22.1, peft 0.21.2 | Pipeline imports, revision marker, and listed model files/shards passed checks |
| Focused existing CPU tests | Cosmos rollout, Cosmos training, and probe statistics: 18 passed in 46.31 seconds | Existing tested primitives are available; the new curriculum modules are not implemented |
| Host environment | PyTorch 2.6.0; no importable pytest; Ruff available | Use container pytest rather than the host environment |
| GPU access | CUDA unavailable in current process; `nvidia-smi -L` cannot communicate with driver | GPU smoke requires a Slurm allocation |
| Scheduler | `srun` and `sbatch` available | Job defaults still need validation against the chosen execution cluster |
| Shared VLM | Gemma 4 12B-it selected for available 40 GB A100s; weights/runtime not downloaded or tested | Implement isolated local worker, pin revision, then validate both modes |
| Semantic evaluators | No calibrated camera/object-specific pass/fail evaluator | Initial feedback uses descriptive continuous scores only |
| Packaging | `pyproject.toml` explicitly lists installed packages | Add `diffusionopsd.curriculum` when creating the package |

The existing `Qwen2-VL-2B-Instruct` and VideoReward assets do not establish that a usable prompt proposer is configured. No credentials were inspected. No installation, model download, external LLM invocation, or training job is part of this readiness review.

### Resource paths for execution

```bash
export COSMOS_SIF=/lustre06/project/6062393/leihan/DiffusionOPSD/containers/diffusionopsd.sif
export COSMOS_RUNTIME=/lustre06/project/6062393/leihan/DiffusionOPSD/logs/cosmos-runtime/python
export VIDEO_REWARD_CKPT_PATH=/lustre06/project/6062393/leihan/DiffusionOPSD/video_reward_ckpts
export HF_HOME=/lustre06/project/6062393/leihan/DiffusionOPSD/hf
```

Run commands below from this worktree after exporting those paths. The wrapper loads Apptainer through the cluster module system if it is absent from PATH. Keep generated experiment outputs in this worktree's ignored `logs/` directory. If a GPU job cannot see these absolute paths, provide explicit binds or stage the same pinned assets before executing it.

### Decisions needed before real execution

1. Resolve and download an immutable `google/gemma-4-12B-it` checkpoint revision on an online node, pin a separate VLM runtime, and verify both modes offline. Set `CURRICULUM_VLM_MODEL_PATH` and the worker runtime; never silently fall back to another model. This plan update performs no download or installation.
2. Select `narval_gemma12` or a fully resolved H200 profile, a valid GPU partition/account, and an initial single-node smoke allocation. Narval needs two GPUs for the sequential default. Discover the H200 per-node inventory first; existing Slurm settings are historical defaults, not verified current entitlement.
3. Record a total experiment cost cap from measured smoke throughput. Stop at phase boundaries; never describe an unmeasured cap as sufficient.
4. Complete the 24-clip diagnosis audit before VLM-based curriculum selection. Physics extensions still require physically grounded evaluators beyond VLM judgments. Numeric-only feedback remains an experimental control.

## File map

| File | Responsibility |
| --- | --- |
| `src/curriculum/__init__.py`, `records.py` | Validated records, lineage, partitions, stable hashes |
| `config/curriculum_cosmos.py` | Pilot settings, named hardware profiles, and required execution-resource inputs |
| `src/curriculum/feedback.py` | Round summaries and deterministic source allocations |
| `src/curriculum/proposer.py` | Proposal requests, lineage validation, receipts |
| `src/curriculum/diagnosis.py` | Ordered-frame selection, diagnosis schema, evidence validation |
| `scripts/curriculum_vlm_worker.py` | One locally loaded frozen checkpoint, two request modes |
| `scripts/run_curriculum_vlm.sh` | Isolated VLM runtime; offline paths and sequential GPU use |
| `config/curriculum_prompts/initial.md`, `revise.md`, `diagnose.md` | Domain-constrained request templates |
| `src/curriculum/runner.py`, `scripts/run_curriculum.py` | Durable round transitions and process execution |
| `src/video/curriculum_probe.py` | Optional paired continuation audit |
| `scripts/train_opsd_video_cosmos.py`, `src/video/cosmos_training.py` | Manifest-bound one-pass training and resume |
| `tests/curriculum/` and targeted video tests | Small fixtures, recovery, lineage, measurement invariants |
| `pyproject.toml` | Include the new package in installation |

## Stage 1 — One complete manifest pass

### Task 1: Records, domain specification, and deterministic manifests

**Files:** Create the package, `records.py`, `config/curriculum_cosmos.py`, and `tests/curriculum/test_records.py`; modify `pyproject.toml`.

**Interfaces:** `PromptRecord(prompt_id: str, lineage_id: str, parent_id: str | None, split: str, text: str, subtype: str, attributes: dict[str, str], source: str, round_index: int)`; `validate_manifest(records: list[PromptRecord], history: list[PromptRecord]) -> None`; `manifest_hash(records: list[PromptRecord]) -> str`.

- [x] Write `test_lineage_partition_inheritance`, `test_duplicate_prompt_rejected`, and `test_manifest_hash_stable`. Assert cross-partition parents and normalized duplicate text raise ValueError; repeated serialization preserves hashes.
- [x] Run `bash scripts/run_cosmos_python.sh -m pytest tests/curriculum/test_records.py -q`; expect missing-package or missing-interface failure before implementation.
- [x] Implement records, strict source/split validation, history lookup, and canonical hashes. Sources are initial, retained, edited, rehearsal, and new. Splits are train, development, and test. Only G1 subtypes orbit, dolly, and lateral are enabled.
- [x] Add named execution profiles and `validate_execution_profile(profile: dict, inventory: dict) -> dict` in `config/curriculum_cosmos.py`. In `tests/curriculum/test_records.py`, add `test_unresolved_h200_inventory_rejected`, `test_device_map_exceeds_allocation_rejected`, and `test_profile_preserves_pilot_budget`; assert eight nodes does not imply eight local devices, out-of-range device maps fail, and selecting hardware alone leaves 32/4/3 and update semantics unchanged.
- [x] Add 32/4/3 pilot counts and seed 42 to configuration. Calibration uses a fixed initial training subset, not development/test. Required proposer and allocation inputs remain unset until supplied; execution preflight rejects missing ones.
- [x] Run the focused tests to PASS and commit `feat(curriculum): define G1 prompt manifests`.

**Deliverable:** The pilot's data contract, with no network or GPU dependency.

### Task 2: One-pass training with safe resume

**Files:** Modify `src/video/cosmos_training.py` and `scripts/train_opsd_video_cosmos.py`; create `tests/curriculum/test_round_training.py`.

**Interfaces:** `RoundCursor(order: list[int], next_index: int = 0)` with `take() -> int | None`; CLI `--curriculum-manifest PATH`, `--curriculum-round INT`, and `--previous-round-checkpoint PATH`. Resume within a round uses the existing `--resume` flag. The previous-round path is an explicit transition, distinct from same-round resume.

- [x] Write `test_complete_one_pass`, `test_rejected_batch_advances_without_update`, and `test_resume_matches_uninterrupted`. Assert 32 prompt visits, 128 attempted clips, and at most 32 updates with four clips/visit; no optimizer or EMA change for empty batches.
- [x] Run the new test file; expect missing-cursor failure before implementation.
- [x] Implement deterministic shuffle by round seed and atomic checkpoint extras: round ID, manifest hash, cursor, successful updates, costs, and reference/calibration state. Checkpoint after each processed prompt, including rejected visits; recover from the last committed checkpoint after interruption.
- [x] Validate round transitions by comparing invariant training settings separately from the replaceable manifest identity. Preserve student, EMA, optimizer, RNG, fixed calibration, and fixed-base reference cache. Reset only the round cursor and per-round statistics. Reject changed configuration during same-round resume.
- [x] Test a legacy text-file run, a manifest hash mismatch, and a transition to a different 32-prompt set. Run `bash scripts/run_cosmos_python.sh -m pytest tests/curriculum/test_round_training.py tests/video/test_cosmos_training.py -q`; expect PASS. Commit `feat(video): add one-pass curriculum training rounds`.

**Deliverable:** A complete prompt pass using the existing reward and fitting mechanism.

## Stage 2 — Round feedback and new prompt content

### Task 3: Frame/evidence contracts, round feedback, and development evaluation

**Files:** Create `src/curriculum/feedback.py`, `src/curriculum/diagnosis.py`, `tests/curriculum/test_feedback.py`, `tests/curriculum/test_diagnosis.py`; modify Cosmos evaluation/report emission in `scripts/train_opsd_video_cosmos.py`.

**Interfaces:** `select_frames(frame_count: int, limit: int = 8) -> list[int]` includes endpoints without duplicate indices; `validate_diagnosis(response: dict, frame_ids: list[int], requirement_ids: list[str]) -> dict` checks statuses and evidence references. `summarize_round(rows: list[dict], development_rows: list[dict]) -> dict`; `allocate_next_round(prompt_ids: list[str], summary: dict, seed: int) -> dict[str, list[str]]` returns retained, edit-parent, and rehearsal IDs plus the new-prompt count.

- [x] Write `test_uncertain_scores_stay_unknown`, `test_test_rows_excluded`, and `test_allocation_budget_and_fallback`. Assert numerical scores cannot invent semantic pass labels and VLM labels remain explicitly attributed observations; test-partition rows raise ValueError; allocations total 32.
- [x] Run the tests to confirm missing functions, then implement per-prompt and G1-subtype summaries: means, dispersion, resolved counts, identity rejection, motion/quality masks, coverage, and elapsed cost. Development input to the proposer contains subtype aggregates only.
- [x] Add tests for endpoint inclusion, monotonic frame order, short-clip handling, absent evidence, invalid frame references, and unknown requirements. Store source times and selected-frame IDs. Aggregate VLM observations separately from numerical scores, with no assumed label accuracy.
- [x] Implement the initial 8 retained / 12 edits / 4 rehearsal / 8 new allocation. For prioritization use lower mean geometry among valid prompts; use fixed-seed tie breaking, enforce non-overlapping retained/rehearsal IDs, and cap exact repeats to those two groups. This is a declared heuristic, not proof of learnability. Reallocate unavailable parent slots to new prompts and report the shortfall.
- [x] Persist fixed development prompt/seed hashes and round-zero scores; evaluate the same student adapter at round boundaries on 8 prompts × 2 fixed seeds. Existing two-prompt evaluation defaults are insufficient for this protocol. Unknown or sparse cases retain sample counts and do not trigger mastery claims.
- [x] Run focused feedback and existing training-evaluation tests to PASS; commit `feat(curriculum): summarize round feedback and prompt allocations`.

**Deliverable:** Explicit end-of-round feedback with independent numerical and VLM evidence fields. Task 4 supplies the real diagnosis worker; run these tasks with fixtures first. The deterministic numeric allocation remains the baseline; validated VLM failure reasons guide edits without claiming calibrated task mastery.

### Task 4: Shared frozen VLM for diagnosis and proposal

**Files:** Create `src/curriculum/proposer.py`, the three prompt templates, `scripts/curriculum_vlm_worker.py`, `scripts/run_curriculum_vlm.sh`, `tests/curriculum/test_proposer.py`, and `tests/curriculum/test_vlm_worker.py`.

**Interfaces:** Worker requests include `mode: diagnose | propose`; `run_requests(requests: list[dict], model: object, processor: object) -> list[dict]` handles both with one model load and no conversational state between requests. `propose(request: dict, command: list[str], output_dir: Path, timeout_seconds: int) -> list[PromptRecord]`. The executable receives `--request PATH --response PATH`; the request includes schema, model/template identity, domain, allowed edits, allocations, and permitted feedback. It emits strict JSON with candidate records and usage metadata.

- [ ] Write tests with a fixture executable for malformed JSON, timeout, duplicate text, wrong parent, cross-split edits, and replay from a completed receipt.
- [ ] Run the focused tests and confirm failure before implementation.
- [ ] Implement subprocess arguments without shell interpolation; validate every response with Task 1. Bound repairs to two additional attempts. Stop visibly if the manifest remains incomplete; do not relax gates silently.
- [ ] Implement a local Transformers worker using one frozen `google/gemma-4-12B-it` checkpoint/processor. Use BF16, batch size one, eval/no-grad, a pinned revision, local-only loading, and Gemma-specific non-thinking chat-template controls. Both roles use the same checkpoint. Do not upgrade the shared Cosmos or VideoReward environments. Default to `narval_gemma12`: target one A100 40 GB in a separate phase after releasing diffusion and reward processes. Resolve model and device placement from the selected execution profile. The H200 Gemma control uses the same checkpoint and budgets; Qwen is an explicitly selected separate arm with its own model-specific processor/template and audit, not a fallback. Google's approximate 26.7 GB BF16 figure does not guarantee fit for our visual-token/context workload. Measure actual peak memory and schema validity. If OOM occurs, stop and record the failure; test a separately recorded smaller visual/context budget rather than silently changing model or precision.
- [ ] Diagnosis sees one clip's selected frames/timestamps and requirements, without numerical scores or future edit suggestions. Proposal sees saved reports and quotas in a fresh text-only request. Validate separate schemas, request-mode dispatch, no cross-request history, and identical model revision for both roles. Use the decoding and frame limits in the pilot table, timeout 300 seconds/request, and at most two repair attempts.
- [ ] Audit diagnosis on 24 clips balanced across independently labeled G1 requirement outcomes; include ambiguous examples. Store reviewer labels, per-requirement disagreement, missing evidence, and unknown rates. Review this report before enabling semantic edits; no physics certification follows from agreement. If accuracy is inadequate, keep numeric-only selection and record the failed VLM arm.
- [ ] Compose G1 tasks by changing one of object/scene, camera path, viewpoint range, or clutter. Preserve static-scene assumptions. Record the intended edit; VLM self-validation is a judgment, not certified scene feasibility. Freeze development/test requests before feedback-driven synthesis.
- [ ] Cache completed responses by request/template/model hashes. Atomically record attempts and costs; ambiguous interrupted external calls remain unknown and are reconciled before retry.
- [ ] Run `bash scripts/run_cosmos_python.sh -m pytest tests/curriculum/test_proposer.py tests/curriculum/test_diagnosis.py tests/curriculum/test_vlm_worker.py -q`; expect fixture-backed tests to PASS. Run real worker smoke tests in the separate VLM runtime. Commit `feat(curriculum): add shared visual diagnosis and prompt generation worker`.

**Deliverable:** One local frozen checkpoint supplies evidence-linked visual reports and controlled prompts. Verify a text-only proposal and a multimodal diagnosis on GPU; record peak memory, versions, immutable model revision, and latency. Merely importing the class is insufficient.

## Stage 3 — Operational pilot

### Task 5: Durable orchestration, smoke, and minimal comparisons

**Files:** Create `src/curriculum/runner.py`, `scripts/run_curriculum.py`, `tests/curriculum/test_runner.py`, and a run-local readiness/results report.

**Interfaces:** `run_experiment(config_path: Path, run_dir: Path) -> None`; phases are prepare_manifest, train_one_pass, evaluate_development, diagnose_clips, summarize, and prepare_next_manifest. Run diagnosis only after videos exist. Stop the diffusion process before launching the VLM phase; process all role requests sequentially with one VLM load where possible. Run state references immutable manifests and committed trainer checkpoints.

- [ ] Write a two-round fixture test that interrupts each phase, resumes, and asserts identical manifests, prompt visit counts, and committed update counts. Add stale-hash and concurrent-writer tests.
- [ ] Run tests to fail, then implement a single-writer lock, atomic state writes, subprocess commands, phase receipts, and cumulative cost reporting. Permit externally prepared manifests for offline training nodes.
- [ ] Run all `tests/curriculum`, relevant Cosmos tests, and focused Ruff. Run the full video suite once the shared-change checks pass; disclose skipped integration dependencies.
- [ ] Resolve and persist the selected execution profile and measured allocation inventory; test unresolved H200 topology, conflicting concurrent device ownership, and profile-hash mismatch on resume in `tests/curriculum/test_runner.py`. Refuse unresolved or changed placements during same-round resume; a hardware migration requires an explicit recorded transition and new smoke.
- [ ] Validate cluster allocation and assets for each selected profile; run a two-round smoke with 2 prompts × 4 clips each round and 2 development prompts × 1 seed. Require at least one retained update per round; a fully rejected round is a failed smoke, not permission to lower gates. Verify task edits, adapter/EMA changes, logs, and checkpoint recovery.
- [ ] Measure reference/evaluation/training and both VLM-role throughput and peak memory, then record a per-profile pilot allocation cap and independent-run concurrency limit. Keep A100/H200 GPU-hours separate; compare Gemma across hardware before introducing the Qwen arm. Log visual-token and generated-token counts, selected frames, truncations, and schema-repair calls. Run three rounds of 32 prompts only if that measured budget fits.
- [ ] Compare fixed synthetic prompts, VLM-diagnosis-driven mixed prompts, and numeric-only feedback, using the same initialization and resource accounting. Final test is 16 prompts × 2 seeds after selection is locked. This pilot establishes functionality, not all H1–H5 claims.
- [ ] Commit `feat(curriculum): orchestrate resumable round-level prompt evolution` and record failures as well as gains.

**Deliverable:** The smallest complete feedback-driven data/model loop.

## Stage 4 — Optional target-aware research arm

### Task 6: Add paired diagnostics after the basic loop works

**Files:** Create `src/video/curriculum_probe.py`, `tests/video/test_curriculum_probe.py`; extend round reporting and optional training callbacks.

**Interfaces:** `score_candidates(record: object, candidates: dict[str, torch.Tensor], continue_fn: Callable, reward_fn: Callable) -> dict[str, float]`; `target_gains(scores: dict[str, float]) -> dict[str, float]`. Store a pre-round `default` adapter snapshot and each audited query's original `old` adapter snapshot. Query tensors may be serialized only in trusted run-local artifacts with explicit storage cost.

- [ ] Write analytic tests for gain signs, nonfinite outcomes, baseline continuation parity, and a zero-change student with a nonzero student–behavior gap. Assert the measured before/after difference is zero in the last case.
- [ ] Confirm test failure; implement no-grad candidate scoring with scheduler restoration and exception-safe adapter/RNG/mode restoration. Immutable original behavior weights are used for every candidate of one query.
- [ ] Audit four deterministic query selections per round. Use that query's original sigma and state; evaluate the round-start and round-end student predictions through its original behavior continuation. Report `delta_round`, not isolated per-sample learning attribution. Include g_plus, g_minus, sample counts, and missingness.
- [ ] Verify prefix/suffix parity and freeze-snapshot restoration in tiny Cosmos integration tests. Test an exception while the current EMA differs from the saved original snapshot.
- [ ] Run focused tests and a budgeted GPU audit. Extend feedback with the measured vector; keep numeric-only feedback as a comparison. Do not derive semantic failure causes from gains.
- [ ] Commit `feat(curriculum): audit paired target and round effects`.

**Deliverable:** A separately costed test of the proposal's diagnostic hypothesis. If it adds no predictive or training value, retain the simpler feedback loop.

## Deferred research extensions

These require separate bounded plans when evidence and resources justify them:

- Calibrated camera-motion, object-stationarity, object-specific identity, and physics evaluators, with pass/fail/unknown labels and independent calibration data.
- G2 occlusion and P1–P3 motion/gravity/collision families with suitable supervision. The current rigid-scene reward is insufficient.
- A larger calibration study for semantic judgments; shared-model diagnosis is included in the pilot, but is not a trusted standalone evaluator.
- Learned curriculum allocation, statistically supported mastery/repeatability rules, many training seeds, and additional backbones.
- Full comparison matrix from the proposal: continual synthesis, adaptive sampling, curated prompts, memory/feedback ablations, and independent human or simulator-backed evaluation.

## Verification commands and completion evidence

After exporting the shared resource paths:

```bash
bash scripts/run_cosmos_python.sh scripts/train_opsd_video_cosmos.py --preflight
OMP_NUM_THREADS=1 bash scripts/run_cosmos_python.sh -m pytest tests/video/test_cosmos_rollout.py tests/video/test_cosmos_training.py tests/video/test_probe_stats.py -q
```

A passing preflight checks imports, revision marker, configured files, and listed shard presence; it is not a full weight-integrity audit or GPU-load test. A CPU test pass establishes tested software behavior, not real-model reward effectiveness. Record those distinctions in every readiness report.

Both commands above were executed successfully during this plan revision using the shared paths. No GPU model-loading or real proposer invocation was performed. Structural checks also confirmed the six-task layout, balanced code fences, and a resolvable proposal link; `git diff --check` reported no issues.

## Implementation readiness conclusion

There is enough detail to begin fixture-backed implementation of Tasks 1–4 and runner development. The recommended shared VLM is now named, but real synthesis and visual diagnosis require its weights, isolated runtime, working worker, and GPU smoke. End-to-end video training requires a GPU allocation, current environment checks, and a measured budget. Calibrated semantic pass/fail and broader physics claims remain deferred. The pilot records attributed VLM observations and checks them against an independent 24-clip audit before using them to guide task edits.

This plan was self-reviewed against the research proposal and subsequent discussion. The broad proposal remains the research roadmap; this document deliberately defines only the implementable first pilot plus one optional diagnostic arm.

## Model recommendation source and readiness boundary

**Primary: `google/gemma-4-12B-it`.** The user selected Gemma first because currently available A100 GPUs have 40 GB memory. Use this same frozen checkpoint for both diagnosis and prompt generation. Google documents text/image/video support and approximately 26.7 GB BF16 inference memory, with runtime-dependent memory. Sources: [Gemma checkpoint](https://huggingface.co/google/gemma-4-12B-it), [Google memory guidance](https://ai.google.dev/gemma/docs/core), and [release history](https://ai.google.dev/gemma/docs/releases), consulted November 14, 2026.

The bounded starting request remains eight frames at at most 448 pixels on the longest side, batch size one, and the output limits in the pilot table. Count processed visual tokens and enforce a total input-token cap of 8,192 for the smoke test; reject oversized requests and chunk proposal reports. The processor's actual resized dimensions and tokens must be recorded. These are engineering limits to validate, not evidence of successful deployment. Keep reserved memory headroom and measure peak GPU allocation before the real pilot.

**Deferred comparison: `Qwen/Qwen3.8-27B`.** Its BF16 weights alone exceed a single 40 GB device. Larger-memory or explicitly validated multi-GPU execution belongs to a separate future arm; no Qwen allocation is required for the Gemma pilot. [Official model card](https://huggingface.co/Qwen/Qwen3.8-27B).

Neither model has been downloaded or benchmarked for this project. Pin the Gemma checkpoint revision and isolated dependencies, run text-only and visual GPU smoke tests, and complete the diagnosis audit. The existing Cosmos preflight and 18 passing tests predate VLM integration and do not establish Gemma readiness.
