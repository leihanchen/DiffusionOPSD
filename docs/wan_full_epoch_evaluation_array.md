# Full-epoch checkpoint evaluation array

Submit from the repository root:

```bash
mkdir -p logs
sbatch --test-only scripts/evaluate_wan_full_epoch_array.slurm
sbatch scripts/evaluate_wan_full_epoch_array.slurm
```

Eight tasks evaluate eight held-out prompts each. Task 0 covers indices 0–7;
task 7 covers 56–63. Seeds equal the original zero-based prompt index.
Each task compares base, step 7 and step 50 with identical paired inputs.
The original Wan2.2 inference preset and training configuration are preserved.

Each task requests two 40 GiB A100 GPUs, eight CPU cores, 128 GiB RAM and four
hours on Narval under `def-wangcs_gpu`. Partition selection is left to Narval;
the current four-hour request routes to `gpubase_bygpu_b2`. Policy, text encoder
and VAE use `cuda:0`; scorers use `cuda:1`. The array limits concurrency to two
tasks (four GPUs total). The non-array entry point also requests two A100s,
with a 24-hour limit and automatic partition selection.
The policy GPU capacity check defaults to 39 GiB; set `MIN_GPU_MEMORY_GIB=0`
at submission to disable it. Recorded 121-frame runs peaked at approximately
32 GiB for generation and 15 GiB for scorers on separate GPUs.
Each task generates 24 videos. Narval job 4537528 used two A100s with the same
704x1280, 121-frame, 50-step preset; native-sampler videos took 427–473 seconds
each including scoring and export. This projects to approximately three hours
per shard. The four-hour limit adds margin for loading, verification and W&B
logging; full-shard runtime on Narval has not yet been measured.

Results live in `logs/wan22-full-epoch-array-ARRAY_ID/task_0` through `task_7`.
Each successful task checks whether all eight completed. The last task validates
192 finite, uniquely identified records, 64 matched triplets, consistent model
and generation provenance, verified video flags and matching paired inputs.
It creates `combined/` with merged JSONL/CSV, prompt-weighted summaries, prompt
strata and blinded review files. An exclusive lock prevents duplicate finalization.
Incomplete runs never produce a completed combined manifest; task records remain.

Each shard and the combined result have separate offline W&B runs under
`cvis_tmu/diffusionopsd`, grouped by array ID. Names include job/task ID, purpose
and `narval`. The combined run includes all videos and 192 comparison rows.
Sync these runs after verifying completion; no cluster network connection is needed.

If a task fails, preserve and move its partial `task_N` directory before retrying
that index. For example, after preserving `task_3`, submit:

```bash
WAN_EVAL_ARRAY_ROOT="$PWD/logs/wan22-full-epoch-array-ORIGINAL_ARRAY_ID" \
  sbatch --array=3 scripts/evaluate_wan_full_epoch_array.slurm
```

Automatic resume of partially generated shards is not implemented. To retry
finalization after resolving a failure, use the same
container and environment as the array entry point to run:

```bash
python scripts/aggregate_wan_evaluation.py \
  --array-root logs/wan22-full-epoch-array-ARRAY_ID --wandb
```

For W&B finalization, set `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_RUN_GROUP`, and
`SLURM_ARRAY_JOB_ID` as in the array entry point. Omit `--wandb` for local exports.
