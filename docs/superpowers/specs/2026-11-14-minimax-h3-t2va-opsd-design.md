# MiniMax H3 t2va OPSD design

## Intent and decisions

Extend the existing video DiffusionOPSD implementation to MiniMax H3 through
Diffusers' `t2va` modular workflow. Train attention LoRA adapters using the existing
video geometry reward, detached positive/negative targets, branch loss, and
behavior-policy EMA. Preserve the joint video/audio trajectory and assess held-out
video and audio outputs.

The user described nodes containing eight H100 GPUs and accepted two policy
GPUs per node as the initial configuration, with four as a measured expansion.
Configure P policy GPUs and R target-worker GPUs per node explicitly, with n=P+R.
Choose P from measured policy memory and R from measured target throughput.
Start qualification at P=2/R=1, with two-way policy FSDP2 and one reward GPU.
P=4/R=1 gives four concurrent policy candidates. P=2/R=2 instead increases
target throughput with the same policy memory budget. All configurations
preserve four logical candidates. Start profiling with
32 CPU cores and 384 GiB host RAM and size later requests from measurements.
Actual GPU capacity, interconnect, partial-node scheduling, and billing must be
recorded during preflight. Memory budgets are estimates pending a full-checkpoint
update. Production presets use P in {2,4} and positive R, with at most eight
assigned GPUs per node. Other sharding degrees require a separately qualified
configuration. A running job uses a fixed verified mapping. Two policy GPUs are
sufficient to schedule four candidates in two rounds, while full-checkpoint
memory fit remains unverified.

Default configuration values are `num_nodes=1`, `policy_gpus=(0,1)`,
`target_gpus=(2,)`, `candidates_per_node=4`, and `policy_microbatch_size=1`.
The four-policy expansion uses `policy_gpus=(0,1,2,3)` and `target_gpus=(4,)`.
GPU count does not implicitly change candidate count or microbatch size.

The deliverables for this planning task are this design and its implementation
plan. Implementation includes CPU tests, distributed tests, runtime setup, and
staged GPU verification when the plan is executed.

## Existing implementation and mathematical contract

- `src/video/opa_video.py` implements normalized ascent/descent and a relative
  latent L2 trust region. `src/video/branch_loss.py` implements paper equations
  11–13 over all nonbatch dimensions.
- `src/video/cosmos_training.py::fit_queries` separates target construction from
  policy fitting. Its local-count averaging, gradient clipping, and immediate
  optimizer step require a separate distributed orchestration path for H3.
- `src/video/cosmos_rollout.py` captures native query inputs and scheduler state,
  replays a differentiable prediction, and decodes normalized latents.
- `training_data.py`, `training_logging.py`, `gates.py`, `geo_reward.py`,
  `quality_judge.py`, and the isolated judge provide reusable dataset, logging,
  scoring, and gate behavior. The existing GeoFlow derivative is a partial-gradient
  surrogate recorded by `reward_metadata()`.

At H3 step q, the query contains video and audio states and their separate times.
For the video output, use float32 arithmetic

`y_video = z_video + sigma_video * u_video_theta(joint_query)`.

H3 predicts data-ward velocity. Setting `v = -u` gives the paper's
`y = z - sigma * v`. Model times follow `t_video = 1 - sigma_video` and
`t_audio = 1 - sigma_audio`. Preserve native per-row time tables and modality
indices. Construct targets and the loss over generated video rows only, unpacked
to `[B, C, T, H, W]`. Audio latents at the query remain detached inputs to replay.
The trust-region norm and loss never include text, audio, or conditioning rows.

Sampling, anchors, endpoint weights, and target tensors are detached. Reward
target construction differentiates the frozen video VAE and reward with respect
to a temporary clean video latent. Fitting differentiates a single transformer
query. During a phase, behavior weights remain fixed. Update the behavior adapter
only after a globally valid optimizer step.

## Approaches considered

| Approach | Benefit | Cost and decision |
| --- | --- | --- |
| Eight independent BF16 policy replicas | Simple data parallel sampling | Each transformer consumes about 61.7 GB before activations and auxiliary models. Headroom is uncertain. |
| P FSDP2 policy ranks and R complete target workers | Separately controls policy capacity and target throughput | Selected design. Start at P=2/R=1 and expand the measured bottleneck. |
| Policy and auxiliary work co-located on each rank | Uses every device for policy work | Replicates auxiliary weights and requires residency coordination. Retain as a later measured alternative. |
| Four-way FSDP2 with four paired auxiliary GPUs | Separate decoder/reward capacity for each candidate | Replicates all auxiliary models four times. Additional GPUs require measured throughput benefit. |

Four dedicated auxiliary GPUs replicate VAE/reward/judge work for concurrent
candidates. The algorithm does not require one extra GPU per candidate. Current
target construction completes before policy fitting, and the planned rollout
contains many transformer steps while auxiliary GPUs would largely wait. Separate
devices provide memory isolation but have no specified pipeline overlap.

Candidate count is independent of policy rank count. For K=4 candidates per node,
execute K/P synchronized policy rounds with candidate index
`round*P+node_local_policy_rank`.
Out-of-range slots perform a matching native forward and zero-weight replay.
Sum gradients across rounds and ranks before dividing once by global accepted N.
Increasing FSDP degree reduces persistent weights. Each candidate's activations
still reside on its policy rank. With P=2, local ranks own candidates {0,2} and
{1,3}, respectively. Each rank accumulates its two microbatch-one gradients
before one optimizer step. With P=4, each rank owns one candidate in one round.
Fitting averages over accepted candidates and skips a globally empty batch.

Tensor/context parallelism can reduce activation memory or latency, but its
composition with LoRA, native packed layouts, and replay needs additional tests.
If four-way FSDP still exceeds the activation budget, profile saved activations
and then design that extension against the measured bottleneck. Eight cards
alone do not imply automatic pooling of activation memory.

## Configurable policy group and target workers

Launch n=P+R processes, one per allocated GPU. Policy ranks `0..P-1` form a dedicated
FSDP/NCCL group. Target ranks `P..P+R-1` each own one device and model process.
All ranks join a separate Gloo group for control and CPU tensor transport. Target
ranks do not join policy all-gathers, adapter reductions, or policy-group barriers.
The first integration uses R=1. The configuration and transport identify target
workers explicitly so a second worker can be measured with identical policy
sharding and candidate count.

| Allocated n | Policy ranks P | Target workers R | Policy rounds for K=4 | Purpose |
| --- | --- | --- | --- | --- |
| 3 | 2 | 1 | 2 | Initial capacity and timing measurement |
| 4 | 2 | 2 | 2 | Compare when target construction dominates |
| 5 | 4 | 1 | 1 | Compare when policy capacity or latency dominates |
| 6 | 4 | 2 | 1 | Combine expansions after individual measurements |

Four policy GPUs per node are the practical concurrency ceiling for these
four-candidate presets. Further scaling uses additional node-local batches.

Each policy rank executes the complete model on its candidate, gathering one
block's weights from the policy group as needed. Persistent frozen weight shards
are approximately 1/P of the backbone. Adapters stay replicated across policy
ranks. Rank zero owns the global update, schedule, and checkpoint publication.

1. An offline preparation command loads the frozen Qwen3-VL conditioner on one
   GPU, computes selected training/calibration/evaluation conditioning, and writes
   atomic cache entries. Release it before policy processes start.
2. Launch `torchrun --standalone --nproc_per_node=$H3_TOTAL_GPUS` with the resolved
   scheduler GPU count. Rank index maps to its visible GPU. Create policy/control
   groups in identical global order. Never reinterpret physical GPU IDs outside
   the scheduler's visibility map.
3. Policy ranks stream frozen weights into FSDP shards. Each target rank loads one
   video VAE, DA3, WAFT, DINOv2, and isolated quality judge. Keep model weights
   resident when the measured memory budget permits it. Park the judge or other
   conflicting auxiliary components on CPU at phase boundaries if necessary.
   Load the audio decoder on demand for evaluation.
4. Execute complete native `t2va` trajectories and capture the query. Each rank
   keeps its own scheduler instances and generator, avoiding cross-rank RNG draws.
5. Send detached video endpoints/anchors to the reward rank in canonical candidate
   order. The assigned worker returns scalar endpoint/gate scores and then detached positive and
   negative video targets for accepted candidates. Release reward/decoder graphs
   after each request. Keep only bounded queued payloads.
6. Synchronize the adapter update, EMA, evaluation schedule, and checkpoint boundary.

Each target service implements `score_endpoint`, `score_anchor`, `construct_targets`, and
`decode_evaluation` requests carrying schema version, request ID, candidate ID,
prompt/reference identity, dtype, shape, and expected reward version. A single
owner serializes requests, initially one candidate graph at a time. Transfer
contiguous detached CPU tensors through Gloo, then move them to the reward GPU.
For target construction create a fresh differentiable video latent, run the
existing `opa_tr_step_nd` locally through VAE and reward, and return detached
targets plus gradient/trust-region diagnostics. Policy parameters never occur
in that graph. Scalar reward replies alone cannot construct the OPSD targets.

The isolated judge needs a versioned residency command that parks its model on
CPU after scoring. It synchronizes pending work, releases temporary GPU tensors,
moves model weights to CPU, releases unused subprocess allocator reservations,
and then returns an acknowledgement. Verify the release through NVML, since the
parent's allocator operations cannot reclaim the subprocess's reserved memory.
The service waits for that acknowledgement before conflicting target work. Keep
legacy judge behavior as the default for existing callers. A new H3 auxiliary
controller manages this protocol and logs load/evict/score durations. Record tensor
transport time and service queue wait when comparing GPU-hours.

Each isolated quality judge uses `run_dir/reward-rank-{service_rank}` paths. Its subprocess
must clear inherited torchrun rendezvous/rank variables and retain the same
scheduler GPU visibility map. It uses the validated legacy reward runtime.
Policy collectives use only policy devices. Reward/control requests have bounded
timeouts and explicit success/error replies. At a request boundary the controller
broadcasts source/destination/phase metadata, transfers that one candidate, then
broadcasts completion status. All ranks finish the service round before the next
policy-bearing phase. With R>1, route candidate i to target worker `i % R`,
preserving order within each worker and using separate bounded channels.
Independent workers process disjoint candidate graphs concurrently. Collect all
target replies before fitting, and qualify multi-worker execution against R=1
target/update equivalence. Record topology and negotiated transport.
Use platform defaults initially and verify peer communication on the H100 node.

## Allocation selection and reward ownership

The manifest records `num_nodes`, `policy_gpus`, `target_gpus`, `candidates_per_node`,
model/reward revisions, and measured peak memory per role. Explicit GPU indices
map within scheduler-visible devices, have no overlaps, and cover the selected
allocation. The launcher validates P+R against its GPU request and reports
whole-node billing when relevant. Candidate count and optimizer batch remain
independent of P and R.

One target worker owns the complete VAE/GeoFlow graph. DA3 produces detached
geometry, and WAFT/DINOv2 contribute input derivatives inside the same composite
reward. The inference-only quality judge serves gates and evaluation in a
separate phase, sharing that worker's GPU if its measured memory permits. Prefer
one candidate graph at a time and resident weights. Park conflicting components
only at graph-free boundaries. This controls concurrency and preserves local
intermediate video/geometry data.

If independent differentiable rewards are introduced, evaluate their latent
gradients sequentially, release each graph, and sum the configured weighted
gradients before the one normalized OPSD target step. Preserve each reward's
forward convention and recompute at every target location. Normalizing each
reward gradient separately would change the weighted-sum reward objective.
Quality-gate scores remain scalar inputs to fitting weights.

Profile one complete update at P=2/R=1 first. Increase P if policy allocations
exceed the device budget or policy time justifies more concurrent candidates.
Increase R if serialized target construction dominates and two workers improve
end-to-end time enough to justify their allocation. Compare successful updates
per billed GPU-hour at fixed prompts, seeds, K, and optimizer settings, and
report wall-clock latency separately. Eight available GPUs are a capacity ceiling.

## Multi-node extension

Qualify the single-node trainer first, then extend to M identically configured
nodes. Each node has its own backbone copy sharded across its P local policy
GPUs, its R local target workers, and four candidates for one assigned prompt.
One synchronized optimizer update therefore attempts 4*M candidates. The
microbatch remains one per policy GPU, and the number of local rounds remains
two for P=2 or one for P=4.

FSDP's mesh and frozen-weight collectives stay within each node. A separate
all-policy group spanning M*P ranks broadcasts initial replicated adapters and
reduces their accumulated gradients once per optimizer update. Target workers
participate in node-local request transport and global control status, with no
membership in policy/FSDP collectives. Construct all groups in deterministic
global order. Use node-local control/transport groups and one all-rank control
group with global-rank-unique request identities.

Global rank zero owns the deterministic prompt permutation and assigns node j
the record at `(attempt*M+j) % selected_pool_size`. Require at least M records
in the selected pool and include node slot, record identity, attempt, and candidate
index in explicit generator seeds. Preserve one shared historical tracker,
calibration state, and matched baseline identities. Collect real candidate scores
and masks across nodes before computing and distributing weights. The current
per-prompt normalization convention remains the same.

Every policy rank accumulates unnormalized candidate-loss gradients over its
local rounds. SUM these gradients over all M*P policy ranks, divide once by the
global accepted count, then clip, step, and update EMA identically everywhere.
Use that global reduction directly rather than averaging local-node means. A node
with no accepted candidates executes zero-loss replay in each slot when another
node has accepted work. Skip optimizer and EMA only when the global accepted
count is zero. Advance the global attempt even on rejection and persist the
complete per-node sampling position.

All nodes finish target construction before global fitting, retaining the
behavior-policy snapshot throughout the attempt. Propagate a failure on any
node to the complete job and resume from a globally published checkpoint.
Save RNG/service state for every rank and verify replicated adapter/optimizer
hashes across all policy ranks before rank zero publishes. Exact resume keeps
M, P, R and the selected data/configuration identity unchanged.

Expose `num_nodes`, `candidates_per_node`, and per-node GPU lists explicitly.
Distribute the same eight global evaluation records over M*P policy ranks in
ceil(8/(M*P)) rounds, with a matched dummy forward for each out-of-range slot.
Baseline/current passes use identical assignments. Local target workers score
real records only, and rank zero aggregates each evaluation index once and
publishes uniquely named media. The real held-out count stays eight across M.
Use one torchrun agent per Slurm node with a common rendezvous endpoint, unique
node rank, and P+R processes per node. Test M=1 first and M=2 before wider
scaling. Report the global candidate batch 4*M prominently because increasing
nodes changes the optimization batch. The 50-update pilot uses M=1 by default.

## Sharding, adapters, precision, and loading

Freeze one pretrained transformer. Attach `default` and `old` adapters before
wrapping. Apply rank-16, alpha-32, zero-dropout LoRA to exactly
`transformer_blocks.<index>.attn.{to_q,to_k,to_v,to_out.0}`. Keep the token refiner
frozen. Adapter parameters are FP32 replicas and are passed as `ignored_params`
to every applicable FSDP2 wrapper. They are explicitly moved to the policy GPU,
broadcast after initialization, and reduced explicitly after backward.

Use bottom-up `fully_shard` on transformer and refiner blocks, precision-sensitive
leaf modules where necessary, then the root. Preserve checkpoint BF16/FP32
parameter dtypes with `param_dtype=None`. Preserve H3's FP32 projections, timestep
modules, and output arithmetic. Partition a group's mixed-dtype parameters into
compatible wrapping groups if the pinned FSDP version requires it. Use
`reshard_after_forward=True`, including the root, and bounded prefetch.

The optimizer is created after wrapping and contains only replicated `default`
adapter parameters. PEFT adapter selection must restore an invariant trainability
mask, because `set_adapter` can toggle `requires_grad`. `default` stays trainable,
`old` and the backbone stay frozen, including during no-grad behavior sampling.
Assert that invariant after each switch. EMA uses identical replicated FP32
adapter tensors and requires no full-backbone state dict.
Initialize `old` by copying the broadcast `default` values before the first
rollout. Rank zero's adapter initialization defines both replicas on every policy rank.

Loading builds on meta, wraps bottom-up, materializes only local shards, and
streams matching safetensors slices to each DTensor shard. Validate names,
padding, offsets, dtype, and hashes against a tiny checkpoint before the full
model. Small replicated parameters and ignored adapters use explicit replicated
loading. No process materializes a full transformer on GPU. Never convert all
frozen weights to FP32 to construct the optimizer.

Reconstruct constructor-derived and nonpersistent buffers after meta
materialization, including FP32 rotary `inv_freq`. Checkpoint tensor loading alone
does not restore them. Compare every buffer and loaded tiny-model output with
ordinary `from_pretrained`. Require two- and four-policy-rank update equivalence with frozen base
DTensors, ordinary replicated adapter tensors, adapter switching, non-reentrant
gradient checkpointing, and rejected-rank backward before full-model execution.

Prefer PyTorch's efficient SDPA backend with verified backward support on H100.
Enable native transformer and VAE gradient checkpointing. The documented Hub
FlashAttention backend is optional after output/gradient parity verification.
Quantization and inference caches require their own later validation.

## Memory budgets and profiling

The Diffusers page reports about 61.7 GB for one transformer and 62.1 GB for the
conditioner. Measure exact tensor bytes in preflight because published units and
mixed-precision components affect these totals. Two-way sharding yields roughly
31–33 GiB of persistent frozen weights per policy GPU and four-way roughly
15–17 GiB. Gathering one block,
activations, NCCL buffers, and temporary projections add to this floor.

The selected attention targets have approximately 40.14M adapter parameters at
rank 16 for the published 50-block dimensions. Two FP32 adapters, one gradient,
and two Adam moments total approximately 0.75 GiB per policy GPU. Verify the exact
count from loaded module shapes. Candidate/query storage is additional.

| Resource | Initial budget or action |
| --- | --- |
| Policy GPUs | Target measured use below `min(70 GiB, 0.90 * device capacity)` |
| Reward GPU | Apply the same peak target to combined service and judge memory |
| Host RAM | Start profiling at 384 GiB, measure cgroup and per-process peaks, then size the training request |
| CPU | Start at 32 cores, bound PyTorch/BLAS and video encoder threads |
| Prompt cache | Stream CPU entries from disk, retain only the current request |
| Checkpoint | Store adapters/optimizer/control state, reconstruct pinned backbone |

Reset and read `max_memory_allocated` and `max_memory_reserved` around loading,
rollout, endpoint scoring, target forward/backward, policy forward/backward, EMA,
evaluation, and checkpointing. Add NVML per-device sampling to include external
judge processes and CUDA allocations. Record elapsed seconds, host RSS/cgroup
peak, token counts, frame geometry, and both query sigmas.

Auxiliary memory is controlled through native VAE tiling/chunking, decoder
checkpointing, one candidate graph on the service at a time, and sequential target steps. Use CPU saved
activation offload only as a profiled fallback with host-memory measurement.
Cosmos's 17-frame memory figures do not predict H3's 124-frame decoder graphs.
Preserve native tiling initially and compare tiled/untiled derivative correctness
and memory before changing it.

Qualify n=3 first at smoke and pilot geometry with the longest selected prompts.
Compare n=5 if policy memory or measured latency warrants it. Hold prompts, seeds,
four candidates, target steps, and optimizer settings constant. Report wall time,
attempts, accepted samples, and GPU-hours per successful update, including rejects,
evaluation, tensor transfer, queue wait, and checkpoint amortization. For separately
allocatable cards, five GPUs improve GPU-hours over three only when their update
time is below 3/5 of the three-card time.

Measure service utilization and its complete per-update critical path. With two
target steps, each accepted candidate needs the initial gradient and separate
second positive/negative gradients, in addition to endpoint/gate scoring. More
policy ranks cannot remove serialized decoder/reward time. Start with sequential
requests and add overlap or batching only after numerical/EMA consistency and
queue measurements justify the complexity.

Check whole-node scheduling and billing before presenting allocation savings.
If the site charges eight cards for a partial-node process layout, report both
active and billed GPU-hours. Larger logical batches are a separate experiment,
recorded explicitly if needed to use more policy ranks effectively.

## Native workflow and query adapter

Construct the pinned `MiniMaxH3Blocks` workflow with `workflow="t2va"`. Prune its
text-encoder and decode portions when initializing the cached-conditioning core,
using the documented block splitting pattern. The preparation stage supplies the
exact state fields the core consumes, including prompt embeddings and text tags.
Select only `transformer/` and shared required component subfolders.
Provide a configuration-only descriptor for VAE latent channels, compression
ratios, chunk geometry, and normalization where native policy-side setup reads
those attributes. Policy workflow construction must not instantiate VAE weights.
The reward rank alone loads the actual VAE and validates its config against the
descriptor. Preserve native layout/geometry calculations using that descriptor.

Wrap native transformer forward and scheduler steps with temporary scoped hooks
that clone the query before scheduler in-place mutation. Capture both velocity
outputs and pre-step video/audio states at the same step. Preserve complete
per-row inputs and normalization metadata. Restore hooks/methods in `finally`.

Select q from actual nonterminal video sigmas nearest requested `0.278` and log
the selected sigma and index. `num_inference_steps=50` specifies sigma grid
points including terminal zero, yielding 49 model evaluations for the ordinary
grid. Preserve native schedules with video/audio shifts 12/3 and deterministic
Euler sampling. Validate equal schedule lengths before synchronized generation.
Guidance is distilled and each step uses one joint transformer call.

`H3Query` retains detached packed inputs, generated-row slices, both query sigmas,
both full schedules and pre-query scheduler state, the endpoint video latent,
endpoint audio latent, and geometry. `H3Rollout.prediction(record)` calls the
wrapped transformer with gradients and returns the unpacked video clean output.
The reward service's differentiable decoder reverses native latent normalization,
calls the VAE directly, and returns `[B,T,3,H,W]` in `[0,1]` for the existing
reward interface.

Fixed-suffix diagnostics restore both schedulers and both modality states.
For a forced video target use `u_video=(forced_y-z_video)/sigma_video` at q and
the saved old audio prediction for that step. Later suffix steps recompute both
modalities with fixed behavior weights. This establishes a joint-trajectory
diagnostic that changes only the query's video clean prediction.

## Global fitting protocol

For the single-node qualification, one attempted prompt produces four candidates
in K/P policy rounds. The multi-node extension applies the same local rounds and
one global reduction over all policy ranks. Rank zero
gathers scalar endpoint rewards/identity/motion/quality and derives keep masks and
weights with the existing historical per-prompt tracker and gates. Broadcast the
result and durable sampling cursor. A prompt with no accepted candidates advances
the cursor and leaves optimizer/EMA/update counters unchanged. Abort after a full
pass through the selected prompt pool is rejected, matching existing behavior.
For M>1, the multi-node section replaces the single-prompt rejection decision
and node-local gradient reduction with global accepted-count handling and the
all-policy reduction. A rejected node still performs its zero-loss replay when
another node has accepted work.

Targets are constructed on the reward rank for each accepted detached video
anchor using `opa_tr_step_nd`. Target construction contains no policy collectives.
Return targets to the owning policy rank and broadcast target-status completion
before policy replay. A nonfinite target or service error fails the attempt
consistently before any optimizer action. Service weights stay frozen and
temporary latent gradients are discarded after detached targets are returned.

Every policy rank performs exactly one policy replay/backward per fitting slot when the
global accepted count N is positive. A rejected candidate or padded slot uses its query
and the graph-connected scalar `0 * prediction.float().sum()`. For an accepted
candidate, backpropagate `loss_scale * branch_loss` as an unnormalized local sum.
After accumulating all local fitting rounds, all-reduce replicated adapter
gradients over the policy group with SUM and divide once by N. Apply one
global finite/nonzero-gradient check, clip the identical replicated gradients,
then step and apply EMA on every policy rank. Missing adapter gradients become zeros
in deterministic parameter order. FSDP never also reduces ignored adapters.

Keep rank-local fitting slot counts identical if gradient accumulation is later
configured. Calibration, base-reference generation, evaluation, and resume use
the same collective call order on all ranks. Evaluation has eight fixed prompts,
assigned in ceil(8/P) rounds, with matched dummy slots. Baseline and trainable
passes occur in the same order. Rank zero owns logs, all policy ranks execute
policy forwards, and the reward rank handles scoring/decoding requests.

## Default experiment and reproducibility

- Model `MiniMaxAI/MiniMax-H3`, revision
  `0919fb684102ef115126612f3fbb84d1027e7317`, inspected during planning. Verify that
  the pinned revision contains every expected workflow/component before use.
- Existing pinned HF dataset, deterministic first 256 records of its seeded train
  permutation for the initial 50-successful-update pilot, and eight fixed test
  records. Record full dataset identity plus selected-record hashes. A configured
  larger pool is prepared with the same streaming cache interface.
- `124` frames, `24` fps, height `544`, width `960`, `50` sigma grid points.
  The requested duration is about 5.17 seconds. Resolve geometry natively.
- Four candidates per prompt, replay microbatch one per policy slot, learning rate
  `1e-5`, rank `16`, alpha `32`, beta `1`, rho `0.10`, two target steps, eta `1`,
  loss scale `5`, max gradient norm `1`, behavior EMA `0.99`, seed `42`.
- Nine uniformly selected reward frames resized to `224x384`, existing GeoFlow
  metadata, four calibration prompts, calibrated identity/motion thresholds,
  quality threshold `0.4`. Record sample frame indices and effective temporal gap.
- Checkpoint every successful update and at wall-time boundaries. Evaluate every
  ten updates and at the end. Save muted video and audio samples for held-out
  comparisons. Audio preservation is evaluated separately from video reward.

Cached conditioning includes exact tokenizer/processor/encoder revisions, layer
50 unnormalized hidden-state convention, prompt content, dtype, text tags,
dataset identity, and schema version. Cache entries are written atomically and
rejected on any metadata mismatch. Cache preparation runs separately and computes
only missing entries. Concurrent writers use a lock and recheck before publishing.

Use a dedicated H3 dependency overlay/container pinned after import and tiny-model
verification. Begin compatibility checks with the installed Torch 2.14 runtime.
Select a Diffusers commit exposing the inspected workflow, then pin its exact SHA
and compatible Transformers/PEFT versions. Record the full resolved stack and
avoid inheriting assumed compatibility from the Cosmos overlay.

## Checkpoints, failure handling, and completion

Preparation records the exact native tokenization and truncation behavior and
checks the encoder's context limit. Retain native token content. Profile the
longest selected train and evaluation requests and report text/video/audio token
counts alongside memory. An over-budget request fails with its stable record ID
and actual shape, allowing an explicit recorded configuration revision.

Audio evaluation saves paired base/current samples for identical prompts and
seeds, checks finite waveforms, native sampling rate, stereo shape, and expected
duration, and includes audiovisual synchronization inspection. These are
qualitative preservation checks until a quantitative audio metric is selected.

The replicated adapters and Adam state can use an intentional rank-zero CPU
checkpoint. Frozen FSDP DTensors are reconstructed from the pinned model and are
excluded. Before publishing, confirm adapter/optimizer hashes match across policy ranks,
gather each rank's Python/NumPy/Torch and distinct used-device CUDA RNG states, and save
cursor, successful updates, tracker, calibration, reference/cache identities,
reward version, geometry, topology, runtime, and logging group. Record P, reward
rank list, R, and the visible-device map, and retain every target worker's RNG/control state.
Seed native sampling with explicit per-candidate generators. A changed policy
rank count requires a controlled replay qualification before exact-resume claims.

Publish a temporary checkpoint atomically, then update `latest.json`. Resume
validates the signature and broadcasts restored adapters/optimizer within the
policy group before any generation. Exact resume preserves M, P, R and device-role mapping. An
incompatible topology produces an explicit error before model execution.
Use one topology validator in preflight and checkpoint restoration. The signature
includes M/P/R, explicit device-role lists, candidates per node, and policy
microbatch size, alongside the existing model/data/reward/runtime identities.
Checkpoint APIs accept absent policy/optimizer objects on the service rank.
Adapter hashes and broadcasts use the policy group. RNG/service-state gathering
and publication readiness/completion use the all-rank Gloo control group.

Use bounded process-group and judge timeouts, torchrun failure propagation, and
Gloo status exchanges at phase boundaries. A rank error during a policy collective
terminates the job and resumes from the last completed checkpoint. A wall-time
signal sets a flag and all ranks checkpoint at the next safe update boundary.
Measure longest update time to reserve enough time to reach that boundary.
Exchange readiness status before each collective-bearing phase, including model
loading stages, native layout creation, rollout rounds, baseline/evaluation rounds,
fitting, and checkpoint restore. A local preparation failure propagates before
peers enter policy collectives. Failures inside an active collective terminate
through bounded process-group timeouts and torchrun supervision.

Success requires native query/endpoint/suffix parity on tiny modules, actual H3
VAE-to-latent gradients, distributed masked-update equivalence, a full-checkpoint
optimizer update on the selected H100 allocation, verified checkpoint resume, and a
short multi-update run. Report memory and throughput with geometry and selected
sigmas. Reward gains require a matched held-out comparison after those checks.

## Sources

- [DiffusionOPSD paper, Section 3 and Algorithm 1](https://arxiv.org/pdf/2608.24646)
- [H3 workflow and memory documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3)
- [H3 transformer implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_minimax_h3.py)
- [H3 scheduler implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_minimax_h3.py)
- [PyTorch 2.14 FSDP2 API, including ignored_params](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html)
- [FSDP2 execution and initialization](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
