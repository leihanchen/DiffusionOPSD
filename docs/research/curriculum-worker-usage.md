# Curriculum implementation: Tasks 1–4

The code implements manifests, one-pass Cosmos training, feedback/evidence, and
an offline frozen-VLM worker. Durable multi-round orchestration is Task 5 and is
not implemented here.

Resource preflight staged Gemma revision `43c9d014b231474fa7212dbb48a762566c2b255d`.
GPU job `9366379` passed text proposal and eight-frame diagnosis on one A100 40GB:
11.94s/22.74s generation, with a maximum 25.22GB allocated GPU memory. Model loading
was 26.10s. These measurements cover only the smoke workload. Independent accuracy
audit is not complete, so semantic curriculum edits remain gated. See the implementation
plan for full versions, token counts, and resource paths.

## Runtime and model preflight

Use the existing Cosmos container for training. Create a separate VLM environment
outside the shared Cosmos/VideoReward runtimes, with CUDA-compatible PyTorch,
Transformers supporting `gemma4_unified` and `AutoModelForMultimodalLM`, Pillow,
and Accelerate. Record its exact package versions. The inspected Cosmos overlay
has Transformers 5.22.1; this is not proof that an independent VLM runtime exists.
Do not install the repository's training dependencies into the VLM environment:
its Transformers constraint intentionally targets another reward stack.

Download the official `google/gemma-4-12B-it` snapshot on an authorized online node,
using an immutable 40-character model commit. Keep processor, tokenizer, chat
and response templates from that same snapshot. Use the HF snapshot directory
named by that commit, or write its verified commit into `opsd_revision.txt` in a
staged copy. The marker is provenance supplied by staging, not a weight-integrity
check. Never replace it to bypass a revision mismatch.

```bash
export CURRICULUM_VLM_PYTHON=/absolute/path/to/vlm-env/bin/python
export CURRICULUM_VLM_MODEL_PATH=/absolute/path/to/snapshots/COMMIT_SHA
export CURRICULUM_VLM_REVISION=COMMIT_SHA
export CURRICULUM_VLM_DEVICE=cuda:0
bash scripts/run_curriculum_vlm.sh --request /absolute/path/request.json --response /absolute/path/response.json
```

Run only after diffusion/reward processes release their devices. Default is
one GPU, BF16, no weight sharding, no automatic precision/model fallback.
The wrapper clears the inherited training overlay and forces offline loading.
Real worker smoke must generate both a text proposal and multimodal diagnosis;
inspect schema, peak memory, tokens, elapsed time, and saved runtime versions.
OOM or oversize input stops the request and requires an explicitly recorded
alternative configuration. No automatic truncation is performed.

The official [Gemma model card](https://huggingface.co/google/gemma-4-12B-it)
and [thinking guide](https://ai.google.dev/gemma/docs/capabilities/thinking)
specify `AutoModelForMultimodalLM`, `enable_thinking=False`, and
`processor.parse_response(..., prefix=...)`. The worker preserves output special
tokens for that parser and validates only its final answer. It normalizes a single
complete JSON Markdown fence (observed in real Gemma output), records the action,
and rejects surrounding prose, multiple documents, and invalid schema types.

## Request contracts

Common keys are `mode`, `model_id`, `model_revision`, `seed`, `template_hash`, and
`payload`. Obtain the template hash from the exact template bytes; it is checked
before inference. The CLI accepts either one request object (one response object)
or an ordered request array (one response array), loading the model once across
both roles. Each request builds fresh messages; no conversational history survives.

A proposal payload has exactly:

```json
{
  "domain": "G1",
  "round_index": 0,
  "split": "train",
  "count": 4,
  "sources": {"initial": 4},
  "parents": [],
  "feedback": {}
}
```

At round zero generate 32 initial training prompts in eight requests, with domain
coverage across orbit/dolly/lateral. Independently author eight development and
16 test prompts before training, with disjoint normalized text and scene lineages.
The model supplies distinct records; validate the combined manifest against all
previous records. A per-request limit of four is not a complete manifest validator.
The three task groups are subtypes within static rigid-scene G1, not three physics
domains. No curated prompt/video pairs are required, but pretrained weights,
rewards, human domain definitions, and independent audit labels are required.

For later rounds, request only `edited`/`new` candidates. Each edit has one distinct
allocated training parent, inherited lineage/split, a new ID, the next round index,
and string attributes `edit_axis` (`object/scene`, `camera_path`, `viewpoint_range`,
or `clutter`) and `edit` (intended change). The declared axis is a model assertion,
not proof of single-axis semantic change. Retained/rehearsal records are copied by
the host, not regenerated. The numeric allocation remains an explicit heuristic:
8 retained, 12 edit parents, 4 rehearsal, 8 new; deficits become new slots.

`propose(request, command, output_dir, timeout_seconds)` adds host-only `history`
for complete lineage/duplicate checks. `command` is an argument list, e.g.
`["bash", "scripts/run_curriculum_vlm.sh"]`. The history never enters the worker.
Only permitted training parents and saved feedback enter proposal messages;
development feedback contains subtype aggregates, never individual held-out queries.
Responses carry result records, model identity, token/time/memory usage, and attempts.
At most two repair calls follow an invalid result. Timeouts kill the local worker
process group and stop. Receipts/cache entries count failed attempts; interrupted
in-flight calls and stale locks require inspection before deliberate recovery.
Never delete a lock while its process is alive. Copying a failed run to a new output
directory is not a free retry: include its acquisition costs in experiment accounting.

A diagnosis payload has `prompt`, `requirements` (unique string IDs), and `evidence`
from `save_frame_evidence`: frame count, native FPS, eight ordered endpoint-inclusive
frames (or all frames for short clips), absolute local PNG paths and timestamps.
Examples of requirement IDs: `camera_motion`, `stationarity`, `identity`, `geometry`.
Only frames, times, prompt, and requirements enter diagnosis messages; numeric scores
and suggested edits do not. Outputs use `observed_pass`, `observed_fail`, `unknown`,
with reasons and visible frame IDs. These are fallible observations, not certificates.

## Diagnosis audit gate

Prepare 12 distinct G1 clips with independently assigned per-requirement labels,
seeking observed pass, fail, and ambiguous examples where available; the report flags missing classes without inventing labels.
Each audit row has `clip_id`, `reviewer`, `labels` (requirement ID → independent
label), `prediction` (worker diagnosis result), selected `frame_ids`, the frozen
`request_hash`, and `diagnosis_template_hash`. Use the comparison tool to assemble
these rows from the exported human labels and ordered worker responses.

Call `diagnosis_audit(rows, model_id, model_revision)` and save its report. It reports
confusion counts, disagreement, unknown rate, and missing evidence. It always returns
`approved: false`. Review the actual report and annotations; only a human reviewer
may set `approved: true` and nonempty `approved_by`. No invented numerical accuracy
threshold automatically approves it. Gemma has diagnosed all 12 candidates successfully in job 9375184, but human labels
are still missing. The blind review page and downloadable bundle are under ignored
`logs/curriculum-preflight/review12/review.html` and `review12.zip`. The existing
12 candidates form the pilot after eligibility review; no additional 12 are required.

Pass the absolute `diagnosis_audit_path` to `propose()` when feedback includes VLM
observations/diagnoses. The host checks model identity, 12 clips, unchanged report,
and explicit approval before launching a worker. Numeric-only feedback remains
available without an audit. Inadequate diagnosis quality means preserving that
numeric-only control and recording the failed semantic-feedback arm.

## One training round

Write manifests as JSON arrays of PromptRecord dictionaries. For later rounds,
provide full history and the previous completed checkpoint. Same-round resume
uses the current run directory plus `--resume`; these two paths cannot be combined.

```bash
bash scripts/run_cosmos_python.sh scripts/train_opsd_video_cosmos.py \
  --config config/cosmos_nano.py --run-dir logs/curriculum/round0 \
  --curriculum-manifest /absolute/path/round0.json --curriculum-round 0 \
  --development-manifest /absolute/path/development.json
```

Use the environment-specific container, runtime, and asset paths from the plan.
Default development is eight prompts × two paired seeds. Smoke overrides are
`--development-count` and `--development-seeds`; changing them invalidates resume.
Each visited prompt attempts four fresh clips and performs at most one update.
All-rejected visits commit the cursor without optimizer/EMA updates. Calibration,
references, optimizer, student/EMA, RNG, rows, costs, and baseline development scores
are checkpoint-owned. Manifest changes require the explicit next-round transition.

Exports after a completed round: `training_feedback.json`, `development_feedback.json`,
`development_baseline.json`, `development_contract.json`, `round_feedback.json`,
`costs.json`, per-clip frame evidence and optional videos. Nonfinite observations
export as JSON null. Checkpoint state, not partially written logs, owns resume.
The round summary is descriptive; it does not classify task mastery or certify physics.

## Hardware profiles

`config/curriculum_cosmos.py` defines Narval Gemma-first and H200 Gemma/Qwen
alternatives. Validate real local GPU inventory with `validate_execution_profile`.
Four A100 40GB devices are not a single 160GB device. Eight H200 nodes does not
specify GPUs per node. The pilot stays 32 prompts × four clips × three rounds across
matched hardware comparisons. Scale independent arms/seeds first; distributed
Cosmos and a persistent shared VLM service remain outside Tasks 1–4.

For the human labeling workflow and report interpretation, see [12-clip review guide](diagnosis-pilot-review.md).
