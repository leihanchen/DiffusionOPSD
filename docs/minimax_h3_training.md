# Native MiniMax-H3 t2va OPSD

The normal entry point runs the native joint video/audio policy with cached text
conditioning and separate reward workers. The default pilot is **P=2 policy GPUs,
R=1 target GPU**, one node, three H100s, 32 CPUs and 384 GiB RAM. These are estimated
resources, not measured production fit. Task 8 must still qualify full H3 weights,
H100 memory/time, exact hardware resume, P=4 and the fifty-update pilot. CPU/tiny
tests establish implementation behavior only. Native WAVs are logged separately
from silent MP4 previews; the MP4 does not contain a muxed audio track.

The isolated quality judge sees only its assigned reward GPU, resolved by UUID
from the parent's scheduler-visible device. Inside the child it is `cuda:0`;
parent profiling retains the original device mapping. This contains dependency
code that selects the default CUDA device, including during CPU model loading.
CPU judge children have no visible GPUs.

Use the dedicated H3 runtime, not the shared Cosmos overlay:

```bash
export H3_SIF=/path/to/containers/diffusionopsd.sif
export VIDEO_REWARD_CKPT_PATH=/shared/checkpoints
bash scripts/setup_minimax_h3.sh
HF_HUB_OFFLINE=0 bash scripts/run_minimax_h3_python.sh scripts/prepare_minimax_h3.py \
  --config config/minimax_h3_t2va.py --prefetch
bash scripts/run_minimax_h3_python.sh scripts/prepare_minimax_h3.py \
  --config config/minimax_h3_t2va.py --cache-root /shared/checkpoints/minimax-h3-conditioning --condition
bash scripts/run_minimax_h3_python.sh scripts/prepare_minimax_h3.py \
  --config config/minimax_h3_t2va.py --check-only
bash scripts/run_minimax_h3_python.sh scripts/train_opsd_video_minimax_h3.py \
  --config config/minimax_h3_t2va.py --preflight
```

Prefetch needs network access. Conditioning and preflight use the local assets.
The base video test suite skips native H3 tests when the pinned runtime is
absent. After the H3 runtime setup, run the native H3 CPU tests with:

```bash
bash scripts/run_minimax_h3_python.sh -m pytest -q tests/video/test_h3_*.py
```

Preparation alone loads the text encoder. Preflight reads safetensors headers and
shard indexes, component/reward identities, full conditioning cache integrity,
pinned runtime versions, NVML Python bindings and the FSDP2 ignored-parameter
feature. It prints exact tensor bytes and the intended physical-to-visible rank
map without CUDA initialization or model allocation. With no allocation, physical
IDs are unknown (`null`); repeat preflight inside the actual allocation.

The pilot, smoke, and multinode presets explicitly use `collective_timeout_s=3600.`
and `service_timeout_s=600.`; custom configs omitting these fields resolve the same
defaults. Preflight prints the resolved values under `timeouts`. Both budgets must
be finite and positive, and the collective budget must strictly exceed the service
budget and the existing 600-second judge-readiness wait. Inconsistent budgets fail
before group/CUDA initialization or checkpoint access. The legacy Cosmos judge
readiness timeout is unchanged.

The one-hour budget applies to existing Gloo and NCCL groups, including idle peers
waiting through another role's loading, generation, target work, or fitting. It is
a conservative configurable bound: across nodes, four sequential 600-second
requests can precede a matching phase completion exchange. It is not a measured
full-model latency or memory-fit result. Startup also includes sequential loaders
outside the judge's readiness window, and neither startup nor production work is
proved to finish within an hour. A missing or hung peer can take up to 60 minutes
to detect, consuming allocation time; successful operations incur no added delay.
The unchanged judge/service limits can fail their own paths sooner. CPU tests
exercise timeout validation, wiring, and delayed/failing Gloo peers; a real run
with complete rollout/update timings remains pending. The renewed A100 smoke
below measured successful full-model loading under this budget.

Config device IDs are **visible local indices**, never host physical GPU numbers.
For `CUDA_VISIBLE_DEVICES=4,6,7`, policy ranks 0 and 1 own physical devices 4 and 6;
target rank 2 owns physical device 7. P and R must partition all allocated visible
devices, policy first and target last. Training rejects extra/missing GPUs or
torchrun ranks before loading models. Target ranks do not join policy FSDP groups.
When Slurm cgroups renumber host devices, the report preserves scheduler allocation
tokens and CUDA-visible tokens separately and leaves host physical correspondence
unknown; allocation order alone does not prove that correspondence.
Target worker count can change independently of policy sharding;
changing P/R changes the immutable topology and requires a fresh run/cache identity.

On an already allocated, verified three-H100 node:

```bash
H3_USE_GPU=1 bash scripts/run_minimax_h3_python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=3 scripts/train_opsd_video_minimax_h3.py \
  --config config/minimax_h3_t2va.py --run-dir logs/h3-pilot \
  --max-updates-this-job 1 --walltime-seconds 10200 \
  --checkpoint-publication-reserve-seconds 60
```

For P=4/R=1 create a Python config based on the pilot, set
`cfg.policy_gpus=(0,1,2,3)` and `cfg.target_gpus=(4,)`, prepare its conditioning,
then use `--nproc_per_node=5`. This requires five GPUs on one node and is pending
hardware qualification. The default config uses P=2/R=1. R can increase separately
by appending visible indices to `target_gpus`; every allocation must match P+R.

For Slurm, set a site account and partition verified to support the requested GPU
type. The script derives P+R from config and submits one torchrun process per GPU:

```bash
H3_ACCOUNT=verified_account H3_PARTITION=verified_h100_partition \
H3_GPU_TYPE=h100 H3_TIME=03:00:00 H3_MAX_UPDATES=1 \
H3_WALLTIME_SECONDS=10200 bash scripts/train_opsd_video_minimax_h3.slurm
```

`H3_CONFIG`, `H3_RUN_DIR`, `H3_RESUME`, `H3_CPUS`, `H3_MEMORY`,
`H3_SIGNAL_SECONDS` (default 300), and `H3_PUBLICATION_RESERVE_SECONDS` are
configurable. The wall limit is a submission setting; `--walltime-seconds` is the
runner's elapsed-time budget (0 means unlimited). Set it below allocation expiry.
Workers handle SIGUSR1, SIGTERM and SIGINT through flag-only handlers and stop
collectively at idle iteration boundaries, budgeting longest observed update plus
checkpoint publication plus reserve. Signals cannot interrupt a hung CUDA kernel.
The Slurm entry point uses `srun` for job-step delivery and invokes torchrun through
`--launch`, which ignores USR1 in the elastic agent while workers install their
own handlers. CPU subprocess signal delivery is tested; delivery on Slurm remains
pending actual allocation qualification.
Torchrun terminates peers after fatal collective failure; recover from the last
published checkpoint. Every successful update is persisted regardless of legacy
`checkpoint_every`.

Resume the same immutable experiment with an independent execution cap:

```bash
H3_USE_GPU=1 bash scripts/run_minimax_h3_python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=3 scripts/train_opsd_video_minimax_h3.py \
  --config config/minimax_h3_t2va.py --run-dir logs/h3-pilot \
  --resume logs/h3-pilot/latest.json --max-updates-this-job 2
```

The cap counts successful updates in this execution and never changes total
`cfg.updates=50` or the exact-resume signature. Walltime/reserve controls also stay
outside that signature. Resume restores adapters, optimizer, RNG, service identity,
prompt history and fixed references before generation.

Only global rank zero creates one offline W&B run per execution, grouped across
resumes, with `optimizer_updates` as the canonical axis. Automatic online sync is
disabled. Logging records config plus pinned model/cache/reward/runtime provenance,
evaluation MP4/WAV media and durable scalar journals. Per-rank policy phase JSON
and target service phase JSONL preserve local diagnostics. They are measurements
of those executions, not full-model resource qualification.

## Current site evidence and billing

As of 2026-11-15, the available site is **Narval**, which exposes A100/MIG devices
and no H100 inventory. No H100 partition or H100 partial-node billing has been
verified. Do not treat legacy repository H100 partition names as evidence or
substitute A100 for the pending H100 qualification.

Earlier tiny job 9398935 verified native forward/backward/update on three A100s.
Job 9402491 separately verified tiny native frozen-shard reconstruction and
adapter/AdamW/EMA/owned-RNG checkpoint replay on three A100-SXM4-40GB devices
(2026-11-15T08:23:37Z–08:24:13Z, 36 seconds, node ng10906, driver 580.178.04).
Each rank passed the checkpoint harness; that harness uses a synthetic parameter
loss, so its result is checkpoint evidence, not a native forward memory estimate.
Neither run establishes full H3/H100 fit, P=4 or fifty-update pilot feasibility.

Verified tiny-test partition `gpubase_bygpu_b1` has a three-hour maximum,
`OverSubscribe=NO`, `ExclusiveUser=NO`, and billing weights
`CPU=333.33,Mem=32.26G,GRES/gpu:a100=4000` (plus MIG entries). Job 9398935 requested
three A100s, 8 CPUs and 32 GiB; its allocation recorded `billing=12000` and three
GPU equivalents. Submitted account was `def-wangcs`; effective accounting account
was `def-wangcs_gpu`. This is evidence for that tiny request only, not the proposed
32-CPU/384-GiB request or H100 billing. On a site that bills a whole node for a
partial-node request, budget the whole node's GPU equivalents; obtain actual site
evidence before submitting. Audit timestamps with `TZ=UTC sacct ...` or explicitly
label the source timezone (Narval host default is EST, UTC-05:00).

## Staged production verification (not yet hardware qualification)

The verification entry point uses the same `run_h3_training` policy loader,
native joint sampler, reward client/service, fitter, collectives, evaluation and
checkpoint APIs as training. It accepts `--config`, `--run-dir` and
`--phase {load,rollout,target,update,resume}`. These modes require the prepared full
assets/cache and the exact P+R allocation; imports or runtime probes cannot pass
them. Existing `--runtime-only`, `--tiny-gpu`, and `--profile-targets LATENTS_PT`
modes remain separate prerequisite evidence.

On the planned three-H100 allocation, run each phase in its own torchrun launch:

```bash
H3_USE_GPU=1 bash scripts/run_minimax_h3_python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=3 scripts/check_minimax_h3_update.py \
  --config config/minimax_h3_t2va.py --run-dir logs/h3-verify --phase load
```

Repeat with `--phase rollout`, `--phase target`, then `--phase update`.
`load` stops after the actual policy/service/native-core/optimizer loads.
`rollout` executes the normal calibration/evaluation prerequisites and four native
candidate trajectories, repeats the same seeds, compares both native schedules
and video/audio endpoint hashes, and stops before candidate endpoint selection.
The first candidate tensors are released before the repeat, retaining four
candidates at a time. `target` additionally runs actual endpoint/anchor gates and
accepted-candidate differentiable reward targets, then stops before fitting.
An entirely rejected batch advances the prompt cursor and follows the normal
bounded retry path without an optimizer/EMA update or a target/update success
assertion. A full selected pool of consecutive rejections fails; accepted targets
still require finite nonzero reward gradients. Walltime exits before an accepted
update report an incomplete stage. Verification reports include calibration rows
and thresholds plus the live cursor, including on failure; append-only
`gate-decisions-rank-RANK.jsonl` records compact candidate scores, decisions,
threshold margins, attempt numbers, seeds and record identities before the
full-pool gate can abort.
These explicit checks add work: verification timing is not normal-training
throughput evidence.

`update` performs exactly one accepted optimizer update, checks finite nonzero
policy gradients, changed default weights and the EMA equation, forces paired
base/current held-out evaluation with identical prompts/seeds, and publishes the
production atomic checkpoint. Verification enables evaluation media without
changing the immutable optimization signature. Decode audits reject nonfinite
audio before PCM conversion and require stereo, the pinned decoder sampling rate,
and the exact decoded length from the rounded native latent grid and configured
decoder expansion. At 124 frames/24 fps the native 40-Hz grid rounds to 207 tokens;
the pinned 32000-Hz decoder emits 800 samples/token, hence 165600 samples (5.175 s).
Samples are preserved. Reports record video duration (124/24 s), audio duration,
and their numerical difference separately; that difference does not measure
audiovisual synchronization. Saved WAV metadata is also checked. Listening and
audiovisual synchronization inspection remain human gates.

```bash
H3_USE_GPU=1 bash scripts/run_minimax_h3_python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=3 scripts/check_minimax_h3_update.py \
  --config config/minimax_h3_t2va.py --run-dir logs/h3-verify --phase resume \
  --resume logs/h3-verify/latest.json
```

`resume` requires a previously updated checkpoint and enough of the unchanged
50-update total budget for one more update. Before generation, it compares actual
restored adapters, optimizer, owned-device/CPU/Python/NumPy RNG, cursor and service
sequence/identity with the checkpoint. It then requires one additional optimizer
update, cursor advancement, paired evaluation and a new audited checkpoint.
Probing RNG does not draw random numbers or enumerate foreign CUDA devices.
Production checkpoint publication verifies identical policy adapter/optimizer
hashes and common state across ranks; verification checks published bytes/hash,
update, optimizer state and per-rank RNG/service records.

Artifacts remain under the chosen ignored run directory:

- Slurm stderr includes flushed `kind=h3_phase` JSON records with rank, phase,
  replay/backward slot and operation start/return time. A caught operation error
  includes its original traceback before the completion collective, so a failed
  peer exchange cannot hide the local error. These records make no CUDA
  synchronization calls; `returned` records Python return, not GPU completion.
- `verification-PHASE-rank-RANK.json`: reached stage, role, provenance, geometry,
  selected longest prompt metadata, four seeds, selected sigmas/full schedules,
  endpoint hashes, target gradient diagnostics, adapter/EMA audits and checkpoint
  evidence. `full_task8_qualification` deliberately remains false.
- `policy-phases-rank-RANK.json` and reward-service phase JSONL: phase seconds,
  allocator allocated/reserved peaks, sampled NVML device/per-PID peaks, process
  peak RSS and service queue/transport/judge release evidence. Per-phase cgroup
  snapshots explicitly describe allocation-lifetime current/peak/max scope.
- `eval/base/`, `eval/update-XXXXXX/`, evaluation JSON, atomic adapter checkpoint,
  hash manifest and `latest.json`: paired silent MP4s/native WAVs and numeric
  held-out results. Reports never imply a positive quality change.

Missing telemetry is listed explicitly and cannot qualify GPU fit. The report
records the actual GPU model/capacity and compares available combined
reserved/NVML device peaks against `min(70 GiB, 0.90 * device capacity)`; it does
not convert that comparison into overall qualification. Sampling can miss short
external allocation spikes. Cgroup lifetime high-water marks do not establish
phase-local host peaks. Size later CPU/RAM requests only from actual allocation
and process measurements, retaining these limitations.

As of 2026-11-15, pinned H3 revision
`0919fb684102ef115126612f3fbb84d1027e7317` is locally available (61 model files
and four dataset files). Native header/index, runtime and reward identity checks
passed; all 61 Hub download metadata revisions match the pin. This is not
independent full-file checksum verification. CPU preparation job 9407603 completed
0:0 from 10:56:06Z to 13:35:11Z (9545 s), requesting 32 CPUs/192 GiB and no GPU;
MaxRSS was 66426296 KiB (63.35 GiB). Preparation and check-only both passed for
264 pilot records (256 train/8 eval) and 260 smoke records (256 train/4 eval).
Manifest SHA256 values are
`23303c5474ab7dcc50a21aeba1721c5359b8923c6726dd9f349020c0fc6c3db2` (pilot) and
`df0fe353beac845da1f1c4f7c2e8a4d216c22ff4e5a613b79ba06dfee05bbd57` (smoke).
The longest train prompt's actual BF16 native embedding, `[1,244,5120]`, was
validated on CPU; this does not establish GPU fit. See
`logs/h3-conditioning-9407603-results.json` and `logs/h3-worker-fix-preflight.json`.

Full-checkpoint diagnostics attempted `124x256x256`, 16 sigma grid points, K=4,
microbatch 1, P2/R1 on one node with three A100 40-GB GPUs, 16 CPUs/128 GiB and a
three-hour limit. Intended phases were load/rollout/target/update/resume; these
original three attempts achieved no completed load report or optimizer update.
All three jobs failed 1:0: 9411528 lacked the pinned DA3 source environment,
9412305 resolved the wrong
worker path from a remapped library `__file__`, and 9413179 timed out after 180 s
at the `service_loading` completion collective while the auxiliary rank was not
ready. The final job ran 14:12:49Z–14:18:18Z (329 s), node ng30708; step MaxRSS was
105596712 KiB (100.70 GiB), and allocated GPU-hours were `3*329/3600 = 0.27417`.
Host RSS and allocated GPU-hours measure neither GPU memory nor utilization.

Pinned DA3/WAFT source exports and source-worker discovery independent of the
installed module path are corrected; the isolated child retains base dependencies.
Runtime requires `GEOFLOW_DA3_DIR`/`GEOFLOW_WAFT_DIR` to point to pinned source
checkouts. Existing `VideoReward/model_config.json` uses the relative base path
`video_reward_ckpts/Qwen2-VL-2B-Instruct`; the frozen runtime snapshot supplied a
`video_reward_ckpts` symlink to the shared checkpoint root, and a config/processor
probe passed without loading weights. The final job loaded Qwen shards and selected
VideoReward checkpoint 11352 but never reached READY. The policy control-group
timeout in that submitted configuration was 180 s while judge readiness permitted
600 s. The original shared submission budget was **3/3 exhausted**; its historical
diagnosis is preserved in `logs/h3-full-smoke-final-diagnosis.md`.
The runtime worker fix passed the full video suite: 432 passed, 12 skipped,
13 known warnings in 305.86 s (`logs/h3-worker-path-video-tests.txt`).

Under a separately renewed three-submission budget, job **9418109 (1/3)** ran
the unchanged full native smoke workload on three A100 40-GB devices and passed
the full `load` stage on all three ranks. Policy loading took 86.284 s and
service loading took 180.555 s with the 3600-second collective budget. Reference/
calibration rollout then failed in native CPU layout construction because the
wrapper had moved text-token tags to CUDA. This was a device mismatch, with no
observed OOM and zero optimizer updates. Sampled device peaks were
33579/33157/8241 MiB; phase NVML process accounting was unavailable because of a
pinned Python binding namespace incompatibility. These are failed-attempt/load
measurements, not successful rollout or update peaks. See
`logs/h3-renewed-attempt1-evidence.json` and `logs/h3-full-smoke-renewed-9418109.err`.
The conditioning-placement and NVML repairs were included in the next attempt.

Renewed job **9419746 (2/3)** again passed full loading and completed native
generation (15/15 denoising iterations). Reference scoring then failed the strict
judge parking acknowledgment before backpropagation or any optimizer update.
No OOM was observed. External device samples at five-second intervals peaked at
35907/35485/13769 MiB; sampling can miss shorter peaks. The failed acknowledgment
omitted actual allocated/reserved bytes, so the residual amount and owner are
unknown. The pinned backend retains live cuBLAS/cuBLASLt workspaces independently
of model tensors; parking now explicitly releases those workspaces before
emptying inactive allocator blocks. This repairs a confirmed cleanup omission,
not a proven identification of that job's residual storage. Failed acknowledgments
now include their actual residency and allocator counters. The strict zero-byte
and external NVML checks remain required. The final authorized attempt is pending;
rollout/target/update/resume qualification gates remain open. See
`logs/h3-full-smoke-renewed-9419746.err` and that run's `gpu-memory.csv`.

Narval has no available H100 or five-GPU single-node P4/R1 inventory. Jobs 9398935
and 9402491 above remain distinct tiny prerequisites. No successful full-model
rollout/update peak, throughput, accepted update, audio preservation or held-out
quality result is claimed; the remaining qualification gates below remain open.

The outstanding qualification is unchanged: three H100s with P2/R1, actual full
H3 checkpoint at smoke geometry, one update plus real resume, then the same checks
at `124x544x960` with 50 sigma grid points and the longest selected train/eval
prompts. Longest-prompt metadata in the report is inventory, not proof that those
prompts were executed. Require both framework and external PID/device NVML peaks,
host use within the allocation and per-phase timing. If memory exceeds target,
profile and adjust supported checkpointing/prefetch/activation offload, repeat
correctness, and amend the design with measured evidence before adding sequence
or tensor parallelism.

Measure reward-service utilization and serialized decode/reward critical path
before expanding P2 to P4. If warranted, compare n=3 with n=5 at fixed K=4;
five-card allocated GPU-hours improve only below 3/5 of three-card update time.
Report policy/target/queue/transport/wall seconds, per-role memory and active and
billed GPU-hours including rejection, queue, evaluation and publication; label
wall-time-prioritized choices. Keep at most four policy GPUs per node and defer
additional node-local batches to Task 9. If targets dominate, compare P2/R2 with
P2/R1 using deterministic i%R routing, bounded worker channels, serialized local
graphs, target/update parity, failure tests, per-worker checkpoint/RNG and all-rank
readiness. R1 remains the baseline. Reward batching/overlap and any weighted sum
of independent differentiable rewards need separate correctness/throughput tests.
Finally require five successful updates with checkpointing/held-out evaluation,
then all 50 configured optimizer updates, resume verification and final checkpoint
audit. Save actual matched base/trained rewards, identity/motion/quality metrics
and audio; a positive quality claim requires the measured held-out comparison and
audio preservation requires listening/synchronization inspection.

After the prerequisite full-checkpoint correctness/resource gates pass, the
normal-training segment and pilot commands are:

```bash
H3_USE_GPU=1 bash scripts/run_minimax_h3_python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=3 scripts/train_opsd_video_minimax_h3.py \
  --config config/minimax_h3_t2va.py --run-dir logs/h3-pilot --max-updates-this-job 5
H3_USE_GPU=1 bash scripts/run_minimax_h3_python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=3 scripts/train_opsd_video_minimax_h3.py \
  --config config/minimax_h3_t2va.py --run-dir logs/h3-pilot \
  --resume logs/h3-pilot/latest.json --max-updates-this-job 0
```

The second command continues to the configured total of 50, subject to actual
allocation walltime and safe checkpoint boundaries; it is not a report that the
pilot has run. Root controls allocations and submission budgets.

## Multi-node extension (qualification pending)

The default 50-update pilot remains M=1. The opt-in
`config/minimax_h3_t2va_multinode.py` requests M=2, P=2/R=1 per node and one smoke
update. Use it only after the full-checkpoint single-node gates above pass, then
require a full-checkpoint two-node smoke and exact resume before a multi-node pilot.
CPU and tiny CUDA results do not qualify full H3 memory, throughput or audio.

Rank zero selects node j's prompt at `(attempt*M+j) % pool_size` in the prepared
seed permutation. Each node attempts K=4 independent candidates, using two local
rounds at P=2 or one at P=4. Rewards and target tensors stay on that node's bounded
Gloo service channels. Only scalar scores/cache entries enter the global control
group. Rank zero owns tracker decisions and common calibration/baselines. Policy
ranks accumulate unnormalized gradients and SUM over all M*P adapters once,
divide by the global accepted count, then clip/step/EMA exactly once. An entirely
rejected node still replays real native queries with zero loss when global N>0;
a globally empty batch performs no optimizer or EMA step.

FSDP groups remain node-local; `all_policy_group` excludes reward ranks. Eight
evaluation records use `ceil(8/(M*P))` equal calls with dummy forwards and globally
unique media indices. Checkpoints persist the global attempt, explicit next
record per node, all-rank owned RNG/service sequence/run identity, common scalar
cache/tracker and global adapter/optimizer/EMA hashes. Exact resume requires the
same M/P/R and pinned preparation identities.

The Slurm wrapper derives M and P+R from config and launches one elastic agent per
node, each with P+R workers, a common c10d endpoint/job rendezvous ID and node rank
from `SLURM_PROCID`. Override the endpoint with `H3_RDZV_ENDPOINT=host:port` when
needed. Rank-zero logging and global evaluation indices avoid shared-file collisions;
phase/service records remain named by global rank. At worker startup both normal
and staged verification CLIs resolve `SLURM_JOB_END_TIME` into an absolute deadline,
before preflight or loading. The monotonic safe boundary retains time already
spent in startup. Malformed/expired deadlines fail; `H3_WALLTIME_SECONDS` explicitly
overrides the normal launch budget (zero means unlimited), and non-Slurm runs
default to unlimited. Warning delivery and unmeasured first-update reserves remain
hardware/operations gates.

An independent opt-in tiny native two-node CUDA harness is available for reviewed
allocations with two nodes and three GPUs each. Set `H3_RUN_GPU_TESTS=1` and
`H3_TEST_CHECKPOINT` to a fresh shared directory, then run one agent on each node:

```bash
H3_USE_GPU=1 bash scripts/run_minimax_h3_python.sh -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=3 --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d --rdzv_endpoint="$RENDEZVOUS_HOST:29550" --rdzv_id="$RUN_ID" \
  -m pytest tests/video/test_h3_multinode.py -q -k two_node_native_cuda_update_resume
```

Use distinct `NODE_RANK=0/1` and common endpoint/ID. This harness loads actual tiny
native H3 safetensors, uses node-local frozen FSDP native forwards/backwards and
bounded detached target transport, rejects an entire node, and verifies production
SUM/global-N fitting, AdamW/EMA agreement and continuous-versus-restored second
update state. Target arithmetic is synthetic; full reward models are absent.
`task9-native-two-node/result.json` records eight attempted/three accepted global
candidates, per-rank/node memory and wall time. Worker GPU-hours exclude allocation
startup/teardown; the controller must add scheduler accounting for billed GPU-hours.
Tiny native job 9405207 passed on all six ranks (all Slurm steps COMPLETED 0:0)
on 2026-11-15 from 09:41:37Z to 09:42:21Z (44 s), using two nodes with six A100
40-GB GPUs total, P2/R1 per node. Its eight global candidates/three accepted
candidates exercised the native fitter and reconstructed second update with
exact adapter, AdamW, EMA and owned CUDA RNG agreement. Evidence is in
`logs/h3-task9-cuda-9405207/task9-native-two-node/result.json`. Allocated GPU-hours
were 0.0733333; the worker timed region used 0.0148696 GPU-hours. These are time
accounting figures, not billing credits. Targets were synthetic and the native
model tiny; H100, full reward and all full-checkpoint qualification gates remain open.

Nested service profiling resets CUDA allocator peak counters only at the outer
phase. A nested request labels its allocator peaks as belonging to that enclosing
phase; its sampled external device/per-PID telemetry remains request-local.
