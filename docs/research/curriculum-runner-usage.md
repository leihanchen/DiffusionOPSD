# Round-level curriculum runner

Step 5 connects the existing one-pass Cosmos trainer, numerical feedback, and frozen
Gemma prompt proposer. A round visits each manifest prompt once with four fresh clips.
The trainer completes its fixed development evaluation before the controller builds
the next manifest.

## Configuration and launch

Create a JSON configuration from `config.curriculum_cosmos.get_runner_config()`.
Supply initial training, development, and final-test manifests, a Cosmos Python
configuration, executable argument lists, a named execution profile, local VLM
paths/revision, and wall/GPU budget caps. All manifest paths must exist before launch.
The initial tasks can be prepared with the Step 4 proposer on an online host and then
copied to an offline training node. Acquisition costs for externally prepared inputs
are reported as unmeasured.

```python
import json
from config.curriculum_cosmos import execution_profile, get_runner_config

config = get_runner_config()
config.update(
    initial_manifest="/absolute/path/train.json",
    development_manifest="/absolute/path/development.json",
    test_manifest="/absolute/path/test.json",
    training_config="/absolute/path/config/cosmos_nano.py",
    trainer_command=["bash", "/absolute/path/scripts/run_cosmos_python.sh",
                     "/absolute/path/scripts/train_opsd_video_cosmos.py"],
    proposer_command=["bash", "/absolute/path/scripts/run_curriculum_vlm.sh"],
    execution_profile=execution_profile("narval_gemma12"),
    environment={
        "COSMOS_USE_GPU": "1",
        "COSMOS_SIF": "/absolute/path/diffusionopsd.sif",
        "COSMOS_RUNTIME": "/absolute/path/cosmos-runtime/python",
        "VIDEO_REWARD_CKPT_PATH": "/absolute/path/video_reward_ckpts",
        "HF_HOME": "/absolute/path/hf",
        "GEOFLOW_DA3_DIR": "/absolute/path/GeoFlow/third_party/Depth-Anything-3",
        "GEOFLOW_WAFT_DIR": "/absolute/path/GeoFlow/third_party/WAFT",
        "CURRICULUM_VLM_USE_GPU": "1",
        "CURRICULUM_VLM_PYTHON": "/absolute/path/isolated-vlm-python",
        "CURRICULUM_VLM_MODEL_PATH": "/absolute/path/pinned-gemma-snapshot",
        "CURRICULUM_VLM_REVISION": "43c9d014b231474fa7212dbb48a762566c2b255d",
    },
)
with open("runner.json", "w") as stream:
    json.dump(config, stream, indent=2)
```

```bash
python scripts/run_curriculum.py \
  --config /absolute/path/runner.json --run-dir /absolute/path/run
```

Use the same command to resume. The default experiment has three rounds, 32 prompts
per round, four clips per prompt, and a fixed 8 × 2 development set. A bounded smoke
uses two rounds, two prompts per round, four clips per prompt, two development
prompts with one seed each, and quotas of one retained prompt and one edited prompt.
Select `config/cosmos_logging_smoke.py` for the existing five-frame, 224 × 384,
eight-step smoke workload. All rejection gates remain those of that configuration.

Run the lightweight controller with a host Python interpreter that can import the
editable package. Its worker commands launch the separate Cosmos and VLM containers.
Launching the controller inside the Cosmos container would require nested container
support and access to the host module environment.

Check paths embedded in the released VideoReward `model_config.json` before an
offline launch. The current checkpoint names its base model with the relative path
`video_reward_ckpts/Qwen2-VL-2B-Instruct`. In a separate worktree, provide the ignored
`video_reward_ckpts` symlink to the staged checkpoint directory so that path resolves
from the repository root. Setting `VIDEO_REWARD_CKPT_PATH` alone does not rewrite
paths inside the checkpoint configuration. Verify the base processor loads with
`local_files_only=True` in the isolated reward environment before allocating GPUs.

The trainer checks its seed, four-clip budget, and device placement against the
resolved runner profile before loading models. Set `CUDA_VISIBLE_DEVICES` through
the scheduler. The controller measures the allocation and resolves local ordinal
devices against GPU UUIDs. Narval uses two A100 40 GB devices for one Cosmos worker,
followed by Gemma on the first device. H200 profiles require measured local inventory
and separate model fit validation. Independent jobs must receive disjoint device
allocations. Cross-node Cosmos training requires additional implementation.

## Feedback modes

- `numeric_only` is the default. Numerical training scores rank retained prompts,
  edit parents, and rehearsal cases. Gemma receives numerical training feedback and
  development subtype aggregates when proposing new text.
- `visual_shadow` also diagnoses saved training frames after the trainer exits.
  These reports are recorded alongside numerical feedback. Prompt synthesis consumes
  the numerical report.
- `numeric_and_visual` passes diagnosis reports to the same frozen proposer after
  validating `diagnosis_audit_path`. This mode requires reviewed approval for the
  pinned model and active diagnosis template. The allocator continues to use numerical
  scores.

`curriculum_mode="fixed"` retains all prompt texts across rounds and supplies the
fixed-synthetic control. Keep initialization, seeds, training settings, and allocation
budgets matched across experimental arms.

## Artifacts and recovery

`state.json` records phase attempts, completion receipts, hashes, and costs.
`resolved_config.json`, `profile.json`, `inventory.json`, and `input_hashes.json`
capture the execution contract. Initial, development, test, and training configuration
snapshots are immutable. Each round contains its manifest/history, trainer checkpoint,
numerical feedback, optional diagnoses, summary, allocation, and proposal receipts.
`results.json` collects round checkpoint receipts and all available VLM usage records.

The phases are `prepare_manifest`, `train_one_pass`, `evaluate_development`,
`diagnose_clips`, `summarize`, and `prepare_next_manifest`. Development generation
occurs inside the trainer. The separate controller phase verifies its exported
membership/count before proceeding. Run and GPU locks prevent overlapping writers
and GPU ownership. Worker processes inherit these locks, retaining ownership if
the controller dies while a worker continues.

Completed phases replay after checking their output hashes. A failed trainer resumes
from its committed cursor, student, EMA, optimizer, RNG, and development state.
The checkpoint receipt binds the checkpoint bytes to the manifest and round counters.
Uncommitted work can be regenerated after a crash. Its acquisition count is reported
as unknown beyond the last saved worker/checkpoint receipt.

Interrupted VLM requests use the Step 4 explicit reconciliation rule. Inspect the
saved request, response, worker cache, stderr, and receipt before retrying an ambiguous
request. The runner preserves those receipts and propagates the failure. A phase has
at most three controller attempts. Proposal schema repair has its existing three-call
bound. Do not remove locks while a worker is live.

Wall costs include failed attempts and conservative interrupted-phase elapsed time.
GPU-hours charge all allocated devices for phase elapsed time, including VLM loading
and phase transitions. A100 and H200 totals have separate keys. Individual worker
receipts retain generated/visual tokens, cache hits, repair attempts, peak memory,
and measured inference times. Failed partial VLM batches retain their per-request
cache receipts. Budget checks reserve configured phase estimates before launch and
bound subprocess duration by remaining wall/GPU budget. Use measured estimates for
production allocations.

A fully rejected round fails the pilot before prompt evolution. Review its diagnostics
with the existing gates preserved. Final-test prompts remain frozen during adaptive
rounds. Run the planned 16 × 2 final evaluation after experimental choices are locked.
The two-round smoke validates execution and checkpoint continuity. Curriculum
effectiveness needs matched experimental arms and independent final evaluation.

## Implementation verification

The Step 5 CPU suite runs with the isolated Cosmos dependencies:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 bash scripts/run_cosmos_python.sh \
  -m pytest tests/curriculum tests/video tests/test_opsd_v2.py -q
```

Result: **427 passed, 10 skipped, 6 warnings**. The skips cover eight optional
GeoFlow-reference parity cases and two tests requiring an isolated two-GPU
allocation. The 28 runner tests include phase interruption/replay, committed
checkpoint recovery, preserved failed-attempt history, child-held locks after
controller death, budget limits, immutable inputs/outputs, profile enforcement,
fixed prompts, and visual-shadow isolation.

A fresh AI reviewer identified one placement defect: the selected VLM device was
recorded without enforcing the worker's actual device. Four regression tests
reproduced the defect. The runner now derives worker model/device settings from
the resolved profile and rejects conflicting arguments, environment values, and
unsupported worker budgets. These tests and the full suite pass after the fix.

The real Narval smoke uses two A100 40 GB GPUs, the five-frame smoke configuration,
and the pinned Gemma revision shown above. Round 0 visited both prompts, attempted
eight training clips, and committed two updates. Its references used four clips
and 80.13 seconds; development used four clips and 126.35 seconds; training used
304.43 seconds. These phase counters exclude process startup. Peak allocated
training memory was 28.71 GiB on GPU 0 and 32.58 GiB on GPU 1.

Gemma proposed one valid camera-path edit on its first attempt: the sculpture
prompt changed from dolly to orbit. It used 1,001 input tokens, 144 output tokens,
10.32 seconds of inference, and 22.73 GiB peak allocated memory. The complete
proposal subprocess took 82.64 seconds including startup. No truncation or schema
repair occurred. The next manifest retained one prompt and included the edited
prompt with its original lineage.

Setup failures are part of the validation record: job 9386807 failed on nested
container startup, 9387058 on an unset pinned DA3 source path, and 9387518 on the
relative VideoReward base-model path. The corrected job is 9387807. None of these
failures justified changing the training gates. No H200 runtime result is claimed.

Job **9387807 completed with exit code 0 in 20 minutes 52 seconds**. Both rounds
visited two prompts, attempted eight training clips, completed development, and
committed two updates each (four total). A CPU inspection of the trusted checkpoints
verified their SHA256 receipts, exhausted cursors, feedback counts, lineage, and
changes to both student and EMA adapters in each round. Student maximum absolute
changes were approximately `2.00e-5` and `2.01e-5`; EMA changes were `6.10e-5` in
each round. The controller recorded 1,234.26 phase seconds and 0.686 allocated
A100 GPU-hours for the successful run; this excludes the earlier failed jobs.

The final cumulative trainer counters report 16 training clips, five reference
clips, and six development clips; timing totals are 614.04, 108.84, and 167.70
seconds respectively. These counters carry across round transitions. The two fixed
development geometry scores changed from `-0.18736` and `-0.24014` initially to
`-0.14875` and `-0.16538` after Round 1. This two-prompt smoke is too small to
establish curriculum effectiveness or physical correctness. Real fault injection
and H200 experiments were not run; checkpoint recovery was verified in automated
tests, and the GPU run verified checkpoint transfer between rounds.
