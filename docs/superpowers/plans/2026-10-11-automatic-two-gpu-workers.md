# Automatic two-GPU worker scaling implementation plan

Goal: derive 4/8/16 workers from nodes × GPUs per node / 2, preserving four prompts × four videos per update and 50 updates per epoch.

Architecture: isolate two GPUs per Slurm task; separate logical prompt scheduling from process count; collect canonical sample identities before weight routing; retain global retained-count gradient normalization. Preserve existing three-GPU launchers.

## Tasks

- [x] Scheduling and validation: test and implement allocation validation, prompt/rollout assignment, canonical score collection and weight routing, and per-host device isolation.
- [x] Resume compatibility: test and implement versioned training-config validation, explicit trusted checkpoint migration, exact RNG restoration for unchanged topology, deterministic reseeding and duration reset for changed topology. Preserve source checkpoints.
- [x] Trainer integration: use scheduling identities and dynamic world size, unchanged OPSD settings and global normalization, rank-zero evaluation/logging, topology telemetry and state.
- [x] Launch: add a submission wrapper and separate Slurm entry point, two GPUs per task, consistent task counts and binding, successor option preservation, and dry-run tests.
- [x] Verification: focused tests, full video suite, formatting/lint and shell syntax; independent review of changes.
- [ ] Probe: four nodes × four A100 GPUs, eight tasks, 12 CPUs per task, 768 GiB per node, four hours, two new updates from trusted step-4 checkpoint, evaluation every two updates, no successors. Verify GPUs, updates 5/6, checkpoint and metrics; report timings and resource use. Do not submit full epoch.

## Invariants and acceptance

Same 16 unique samples per update and stable nonce/prompt/global-rollout seeds for 4/8/16 workers. Prompt-level advantages computed across all four samples in canonical order. All ranks participate in gradient reduction, including zero-retention ranks. Reject unsupported allocations, conflicting GPU UUIDs, incompatible checkpoints, or topology changes without explicit permission. Legacy config hashes verified with source run directory; no bypass. New checkpoints retain adapters, optimizer, schedule, calibration, tracker, baseline and RNG state. No bitwise-equivalence claim across topology migration.

Host memory: default two-worker nodes request 768 GiB, selecting 1 TiB Narval nodes; old worker peak was approximately 227 GiB. Existing trainer mitigation, local formatting commit and untracked SIF are preserved.

Implementation adjustment: Narval accepts one GPU allocation form per job. Request GPUs per node in sbatch and assign two per task in srun, with per_task:2 binding. Omit partition by default so site routing selects a suitable memory/GPU partition. Slurm test-only accepted the eight-worker, 768 GiB/node request. GPU probe acceptance remains pending execution.
