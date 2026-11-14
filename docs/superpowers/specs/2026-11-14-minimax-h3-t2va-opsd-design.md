# MiniMax H3 t2va OPSD design

## Intent and decisions

Extend the existing video DiffusionOPSD implementation to MiniMax H3 through
Diffusers' `t2va` modular workflow. Train attention LoRA adapters using the existing
video geometry reward, detached positive/negative targets, branch loss, and
behavior-policy EMA. Preserve the joint video/audio trajectory and assess held-out
video and audio outputs.

The user requested a plan for nodes containing eight H100 GPUs and delegated all
choices. The defaults below are design recommendations. Hardware assumes eight
80 GB H100s on one node with NVLink/NVSwitch, 64 CPU cores, and 768 GiB host RAM.
Actual GPU capacity and interconnect must be recorded during preflight. Memory
budgets are estimates pending a full-checkpoint update.

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
| Eight-way FSDP2 with auxiliary models sharing policy GPUs | About one eighth of persistent backbone weights per GPU | Couples target-decoder memory, reward memory, and policy collectives on every card. Retain as a possible measured follow-up. |
| Four-way FSDP2 backbone with four paired auxiliary GPUs | About one quarter of backbone weights per policy GPU and separate decoder/reward capacity | Requires synchronized policy calls and explicit adapter gradient reduction. Selected default. |

Tensor/context parallelism can reduce activation memory or latency, but its
composition with LoRA, native packed layouts, and replay needs additional tests.
If four-way FSDP still exceeds the activation budget, profile saved activations
and then design that extension against the measured bottleneck. Eight cards
alone do not imply automatic pooling of activation memory.

## Eight-GPU topology and lifecycle

Four processes participate in the policy process group. Each process can access
its paired auxiliary GPU for differentiable device transfers.

| Process rank | Policy GPU | Auxiliary GPU | Candidate ownership |
| --- | --- | --- | --- |
| 0 | 0 | 4 | candidate 0 |
| 1 | 1 | 5 | candidate 1 |
| 2 | 2 | 6 | candidate 2 |
| 3 | 3 | 7 | candidate 3 |

Each rank runs the full computation on a distinct candidate, using a sharded
backbone whose block weights are gathered as needed. There are four concurrent
candidates and one globally synchronized adapter update. Four auxiliary GPUs
execute VAE/reward work. They carry no trainable policy parameters.

1. An offline preparation command loads the frozen Qwen3-VL conditioner on one
   GPU, computes selected training/calibration/evaluation conditioning, and writes
   atomic cache entries. Release it before policy processes start.
2. Launch `torchrun --standalone --nproc_per_node=4` inside an allocation exposing
   all eight scheduler-assigned GPUs. `LOCAL_RANK` maps to the local policy index,
   and `LOCAL_RANK + 4` maps to its auxiliary index. Never reinterpret physical GPU
   IDs outside the scheduler's visibility map.
3. Load frozen base weights directly into FSDP shards. Load a video VAE, geometry
   estimators, and isolated quality judge per auxiliary GPU. The audio decoder is
   loaded on demand for evaluation, with conflicting auxiliary components moved
   to CPU if profiling requires it.
4. Execute complete native `t2va` trajectories and capture the query. Each rank
   keeps its own scheduler instances and generator, avoiding cross-rank RNG draws.
5. Release endpoint decoding graphs before target construction and release all
   target graphs before policy fitting. Gather scalar scores and gate flags only.
6. Synchronize the adapter update, EMA, evaluation schedule, and checkpoint boundary.

The isolated quality judge must use `run_dir/rank-{rank}` paths. Its subprocess
must clear inherited torchrun rendezvous/rank variables and retain the same
scheduler GPU visibility map. It uses the validated legacy reward runtime.
All distributed collectives use the policy devices, with a separate Gloo control
group for bounded status messages. Record topology and negotiated transport.
Use platform defaults initially and verify peer communication on the H100 node.

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
rollout. Rank zero's adapter initialization defines both replicas on every rank.

Loading builds on meta, wraps bottom-up, materializes only local shards, and
streams matching safetensors slices to each DTensor shard. Validate names,
padding, offsets, dtype, and hashes against a tiny checkpoint before the full
model. Small replicated parameters and ignored adapters use explicit replicated
loading. No process materializes a full transformer on GPU. Never convert all
frozen weights to FP32 to construct the optimizer.

Reconstruct constructor-derived and nonpersistent buffers after meta
materialization, including FP32 rotary `inv_freq`. Checkpoint tensor loading alone
does not restore them. Compare every buffer and loaded tiny-model output with
ordinary `from_pretrained`. Require four-rank update equivalence with frozen base
DTensors, ordinary replicated adapter tensors, adapter switching, non-reentrant
gradient checkpointing, and rejected-rank backward before full-model execution.

Prefer PyTorch's efficient SDPA backend with verified backward support on H100.
Enable native transformer and VAE gradient checkpointing. The documented Hub
FlashAttention backend is optional after output/gradient parity verification.
Quantization and inference caches require their own later validation.

## Memory budgets and profiling

The Diffusers page reports about 61.7 GB for one transformer and 62.1 GB for the
conditioner. Measure exact tensor bytes in preflight because published units and
mixed-precision components affect these totals. Four-way sharding yields roughly
15–17 GiB of persistent frozen weights per policy GPU. Gathering one block,
activations, NCCL buffers, and temporary projections add to this floor.

The selected attention targets have approximately 40.14M adapter parameters at
rank 16 for the published 50-block dimensions. Two FP32 adapters, one gradient,
and two Adam moments total approximately 0.75 GiB per policy GPU. Verify the exact
count from loaded module shapes. Candidate/query storage is additional.

| Resource | Initial budget or action |
| --- | --- |
| Policy GPU | Target measured peak device use below 70 GiB on an 80 GB card |
| Auxiliary GPU | Target measured combined parent/judge peak below 70 GiB |
| Host RAM | Request 768 GiB initially, measure cgroup and per-process peaks |
| CPU | Request 64 cores, bound PyTorch/BLAS and video encoder threads |
| Prompt cache | Stream CPU entries from disk, retain only the current request |
| Checkpoint | Store adapters/optimizer/control state, reconstruct pinned backbone |

Reset and read `max_memory_allocated` and `max_memory_reserved` around loading,
rollout, endpoint scoring, target forward/backward, policy forward/backward, EMA,
evaluation, and checkpointing. Add NVML per-device sampling to include external
judge processes and CUDA allocations. Record elapsed seconds, host RSS/cgroup
peak, token counts, frame geometry, and both query sigmas.

Auxiliary memory is controlled through native VAE tiling/chunking, decoder
checkpointing, one candidate per rank, and sequential target steps. Use CPU saved
activation offload only as a profiled fallback with host-memory measurement.
Cosmos's 17-frame memory figures do not predict H3's 124-frame decoder graphs.
Preserve native tiling initially and compare tiled/untiled derivative correctness
and memory before changing it.

## Native workflow and query adapter

Construct the pinned `MiniMaxH3Blocks` workflow with `workflow="t2va"`. Prune its
text-encoder and decode portions when initializing the cached-conditioning core,
using the documented block splitting pattern. The preparation stage supplies the
exact state fields the core consumes, including prompt embeddings and text tags.
Select only `transformer/` and shared required component subfolders.

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
Its differentiable decoder reverses native latent normalization, calls the VAE
directly, and returns `[B,T,3,H,W]` in `[0,1]` for the existing reward interface.

Fixed-suffix diagnostics restore both schedulers and both modality states.
For a forced video target use `u_video=(forced_y-z_video)/sigma_video` at q and
the saved old audio prediction for that step. Later suffix steps recompute both
modalities with fixed behavior weights. This establishes a joint-trajectory
diagnostic that changes only the query's video clean prediction.

## Global fitting protocol

One attempted prompt produces four candidates, one per policy rank. Rank zero
gathers scalar endpoint rewards/identity/motion/quality and derives keep masks and
weights with the existing historical per-prompt tracker and gates. Broadcast the
result and durable sampling cursor. A prompt with no accepted candidates advances
the cursor and leaves optimizer/EMA/update counters unchanged. Abort after a full
pass through the selected prompt pool is rejected, matching existing behavior.

Targets are constructed on each accepted local video anchor using the existing
`opa_tr_step_nd`. Target construction contains no policy collectives. All ranks
exchange a target-status flag before policy replay. A nonfinite target or runtime
error fails the attempt consistently before any optimizer action.

Every rank performs exactly one policy replay/backward per fitting slot when the
global accepted count N is positive. A rejected candidate uses its ordinary query
and the graph-connected scalar `0 * prediction.float().sum()`. For an accepted
candidate, backpropagate `loss_scale * branch_loss` as an unnormalized local sum.
All-reduce replicated adapter gradients with SUM and divide once by N. Apply one
global finite/nonzero-gradient check, clip the identical replicated gradients,
then step and apply EMA on every rank. Missing adapter gradients become zeros
in deterministic parameter order. FSDP never also reduces ignored adapters.

Keep rank-local fitting slot counts identical if gradient accumulation is later
configured. Calibration, base-reference generation, evaluation, and resume use
the same collective call order on all ranks. Evaluation has eight fixed prompts,
assigned one per rank in two rounds. Baseline and trainable passes occur in the
same order. Rank zero owns logs, but all ranks execute policy forwards.

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
- Four candidates per prompt, replay microbatch one per rank, learning rate
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
excluded. Before publishing, confirm adapter/optimizer hashes match across ranks,
gather each rank's Python/NumPy/Torch and paired-device CUDA RNG states, and save
cursor, successful updates, tracker, calibration, reference/cache identities,
reward version, geometry, topology, runtime, and logging group.

Publish a temporary checkpoint atomically, then update `latest.json`. Resume
validates the signature and broadcasts restored adapters/optimizer before any
generation. Initial supported resume topology has four policy ranks. An
incompatible topology produces an explicit error before model execution.

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
optimizer update on the eight-H100 allocation, verified checkpoint resume, and a
short multi-update run. Report memory and throughput with geometry and selected
sigmas. Reward gains require a matched held-out comparison after those checks.

## Sources

- [DiffusionOPSD paper, Section 3 and Algorithm 1](https://arxiv.org/pdf/2608.24646)
- [H3 workflow and memory documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3)
- [H3 transformer implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_minimax_h3.py)
- [H3 scheduler implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_minimax_h3.py)
- [PyTorch 2.14 FSDP2 API, including ignored_params](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html)
- [FSDP2 execution and initialization](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
