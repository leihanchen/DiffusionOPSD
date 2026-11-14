# Two-GPU video training workers

Goal: each worker sees exactly two GPUs. The trainable policy/optimizer lives on cuda:0;
frozen inference replica, text encoder, VAE, geometry and quality models execute on cuda:1.
Cosmos stages its idle inference replica on CPU between generation calls for activation headroom.
LoRA's frozen base parameters remain part of the trainable policy replica; they cannot
be separated from its forward pass. All sampling and evaluation use a frozen replica.
Launchers request 2*n GPUs per node for n isolated workers and bind two GPUs per task.

- [x] Add shared placement validation, frozen replica synchronization and prompt transfers with regression tests.
- [x] Integrate all Wan/Cosmos trainers, configs, and training Slurm wrappers; preserve sampler/loss and checkpoint formats.
- [x] Test invalid placement/allocation, frozen gradient isolation, policy gradients, evaluation state and replica synchronization.
- [x] Run full tests, static checks, independent review and GPU smoke verification.

Work is isolated on codex/video-two-gpu-contract. Existing other worktrees and unmerged
Cosmos epoch branch remain untouched. No production jobs, remote pushes or merges are implied.
Inference-only analysis/evaluation jobs can still use their own allocations; this policy
applies to training workers and launchers. Legacy three-device training presets are replaced.

## Verification ledger

- Red/green: initial placement test collection failed for missing device_layout; implementations then passed 53 focused tests.
- Initial full suite exposed a stateless timestep-parity stub without model-device metadata and an outdated source assertion
  for selecting the old adapter. Added explicit input-device fallback for stateless callables (actual modules still use
  parameter/buffer placement), tested the fallback, and updated the assertion to the frozen inference wrapper.
- Regression rerun: 33 passed, 1 CUDA-only test skipped. Full authored suite after fixes: **315 passed, 9 skipped**, two
  known warnings, 180.90s. Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 bash scripts/run_cosmos_python.sh
  -m pytest tests -q`. Log: `logs/cosmos-runtime/two-gpu-project-fixed.log`.
- Independent whole-change review found no remaining production blocker; original stateless test issue is fixed. GPU1 memory
  peak with replica + frozen models remains an actual-model runtime gate. Ruff, shell syntax and whitespace checks pass.
- Original Wan environment: **84 passed, 1 skipped** in 45.74s across placement, Wan geometry/policy/rollout, distributed
  topology, launcher and worker-scaling tests. Log: `logs/cosmos-runtime/two-gpu-wan-base.log`.
- GPU transfer test **9338210 COMPLETED / 0:0**, 33s, exactly two GPUs. It verifies Cosmos branch fitting and Wan query replay
  across GPU1→GPU0→GPU1, nonzero policy gradients, frozen replica gradient isolation, and post-update synchronization.
- First model checks **9338221** (Cosmos) and **9338222** (Wan) stopped at GeoFlow's pinned-source guard: the container's old
  DA3 install is unverified. This is an environment failure before training, not a placement or numerical result.
- Verified pinned DA3 and WAFT source revisions inside the container with explicit source env vars and bind. Submitted
  corrected Cosmos **9338627**, Wan **9338628**, and two-worker/four-GPU Wan **9338640**. Initial statuses PENDING with no
  allocated node and no runtime evidence. Submissions under `logs/two-gpu-contract/`; no automatic job chaining or upload.
- Initial implementation and test-regression fixes verified before model smoke runs.
- Cosmos 9338627 subsequently FAILED during replica construction: normal deepcopy cloned the 28.26-GiB transformer on
  GPU0 before transferring it, exhausting the 39.49-GiB device. Fixed by pre-seeding deepcopy's memo with independent
  destination-device parameter/buffer copies; full-finetune Wan uses this path too. A parameter whose source deepcopy
  raises reproduced the issue before the fix; focused rerun **26 passed, 1 skipped**. Reviewer recheck found no blocker.
- Submitted Cosmos replica-memory retry **9339220**, same two-GPU allocation and algorithm. Wan 9338628 is running;
  two-worker Wan 9338640 remains queued. Actual-model completion remains unverified until these handles finish.
- Direct-copy full regression suite: **316 passed, 9 skipped**, two known warnings, 197.35s;
  `logs/cosmos-runtime/two-gpu-project-direct-copy.log`. Original Wan runtime focused rerun **26 passed, 1 skipped**,
  17.07s; `logs/cosmos-runtime/two-gpu-direct-copy-wan-base.log`. No unresolved unit-test failures.
- Wan **9338628 COMPLETED / 0:0** in 10m04s. Real two-GPU update retained 15/16 rollouts, branch loss 0.11577646,
  policy gradient norm 0.50453109, saved `policy_1.pt` and baseline/final evaluation MP4s under
  `logs/two-gpu-contract/wan-9338628/`. Cosmos 9339220 is now RUNNING; distributed 9338640 is still PENDING.
- Cosmos **9339220 FAILED / 1:0** during DA3 calibration scoring after successful native generation. GPU1 held the
  28-GiB idle inference replica, geometry models and a separate 5.41-GiB quality-judge process; a 238-MiB DA3 convolution
  allocation exhausted its 39.49 GiB. This is a distinct activation-memory limit, not the source-copy initialization bug.
- Ruling: stage the idle Cosmos inference replica in CPU memory between generation calls, with generation still on GPU1
  and trainable replay on GPU0. All frozen computation remains on GPU1; permanent simultaneous GPU residency is relaxed
  because the actual-model run proves it cannot fit. Cost: host memory and PCIe transfers per generation. Sampling,
  reward settings, precision, thresholds, number of GPUs and distillation math are unchanged.
- Added exception-safe replica activation context and CPU/GPU regression coverage; CPU test first failed with the missing
  `active_on` method. Focused suite: **41 passed, 2 GPU-only skipped**, 21.05s. Independent review found no blocker:
  captured queries are detached, policy replay stays on GPU0, adapter sync targets the CPU replica, and all frozen
  computation stays on GPU1. Any future fixed-suffix continuation must also activate the staged replica explicitly.
- Submitted Cosmos **9339924** with the same model smoke settings and allocation; two real-CUDA replay/staging tests run
  before training in that allocation. Submission: `logs/two-gpu-contract/cosmos_staging_submission.json`. Currently queued
  for node availability; two-worker Wan **9338640** remains queued for priority. Full regression rerun is in progress.
- Staging full regression: **317 passed, 10 skipped**, two known warnings, 197.53s;
  `logs/cosmos-runtime/two-gpu-staging-full.log`. Changed-file Ruff and whitespace checks pass. Job **9339924** started;
  its GPU validation step completed **2 passed**, 24 deselected, 86.29s, exit 0:0. This verifies actual CPU↔GPU1 staging
  and restoration on generation failure as well as GPU1→GPU0 policy replay gradients. Actual Cosmos training is running;
  distributed Wan **9338640** remains queued. No unresolved unit-test failure.
- Cosmos **9339924 COMPLETED / 0:0**, 13m45s including GPU unit checks and preflight. Actual model update retained 4/4
  samples on its second prompt (first batch rejected by unchanged identity gate), loss **0.13095574**, pre-clip policy
  gradient norm **10.89924908**, target gradient norm **0.04453537**, policy weights changed. Saved `training_state.pt`,
  four baseline/final evaluation MP4s and one offline W&B run. Audit: `logs/two-gpu-contract/cosmos-v4-20261114/two_gpu_audit.json`.
  Idle GPU1 use fell to ~4 GiB after staging; calibration, generation, geometry scoring, target/policy backward,
  adapter synchronization, final evaluation and checkpoint all completed.
- Remaining runtime gate: two-worker Wan **9338640** is still PENDING; no distributed completion claim yet.
- Distributed Wan **9338640 COMPLETED / 0:0**, 8m44s. Two workers on one four-GPU node each saw exactly two distinct
  GPU UUIDs with no overlap. Both initialized NCCL, completed target construction and gradient reduction, and produced
  identical default/old adapter hashes after optimizer and EMA synchronization. Update retained **10/16** rollouts,
  branch loss **0.13596949**, policy gradient norm **0.68623310**. Baseline/final evaluation saved four MP4s;
  final `eval/all_pass=true`. The requested one-update smoke ended without submitting a successor.
- Final artifact audit loaded the distributed checkpoint and verified update 1, two-worker topology, two RNG states,
  default/old adapter fingerprints matching both workers, four nonempty evaluation MP4s and one offline W&B run.
  Audit: `logs/wan22-scaled-two-gpu-contract-20261114/two_gpu_audit.json`.

## Completion audit

- All video training entry points (single Wan, epoch/scaled Wan, Cosmos, and reward-update diagnostic) validate exactly
  two visible GPUs and put trainable replay on GPU0 and frozen computation on GPU1. All eight video training presets
  share those device assignments; unit checks reject invalid placements and visible counts.
- Single-worker launchers request two GPUs per node; epoch launchers request two per one-worker node; the scaled
  submitter derives workers as GPUs/2 and binds two GPUs per task. The real two-worker/four-GPU run proves isolation
  through distinct GPU UUIDs, numerical gradient reduction and matching post-update adapter state.
- Full current implementation suite: 317 passed, 10 skipped, two known warnings. The GPU allocation separately passed
  both cross-device replay and replica staging tests. Static checks and independent reviews have no unresolved blocker.
- Real Wan and Cosmos updates, evaluation videos, offline logging and checkpoints are verified. These are 5-frame,
  224x384 smoke runs; production-resolution memory and training convergence are not claimed by these checks.
- Implementation is retained on `codex/video-two-gpu-contract`. The follow-up publishing task squashes the implementation
  into one commit and creates a PR and Linear issue. No merge, branch cleanup, artifact upload or production run is included.
  The pre-existing container is untouched.
