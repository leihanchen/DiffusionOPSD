# Step 6 paired continuation diagnostics

Step 6 measures whether the actual reward targets and the student's round-level change improve completed Cosmos videos on saved training queries. It is enabled with `paired_probe_count` in the Cosmos trainer configuration. The diagnostic report is `round_probe.json` alongside the training feedback.

## Measurements

For each audited query, let `C_q(y)` continue the original native solver trajectory after substituting prediction `y` at the captured query. Every candidate uses the query's original EMA adapter, captured conditioning, sigma, and pre-query UniPC history. Let `R` be the existing geometry reward on the decoded final video.

- `g_plus = R(C_q(y_plus)) - R(C_q(y0))`.
- `g_minus = R(C_q(y_minus)) - R(C_q(y0))`.
- `g_reflected = R(C_q(2*y0-y_minus)) - R(C_q(y0))`.
- `delta_round = R(C_q(y_student_end)) - R(C_q(y_student_start))`.

The round-start and round-end student predictions are evaluated on exactly the same query. This measures their change through the frozen continuation. The report also records local decoded-prediction rewards, candidate identity/motion scores, effective training weight, source prompt/seed, original behavior update and round-start/end updates. For the current beta=1 objective, the reflected candidate is relevant when the negative branch receives weight.

Baseline continuation is compared against the saved native endpoint with relative latent L2 error and maximum absolute error. `valid` requires relative L2 error at most 0.001 for both original-EMA prediction replay and native endpoint continuation, plus finite final candidate scores. The original query velocity is preserved exactly and candidate prediction deltas are applied relative to it. Replay explicitly disables outer autocast to match native generation, including VAE decoding. Failed parity remains visible in the report and invalidates interpretation of that query's gains. The real-job verifier checks this independently of Slurm exit status.

## Capture, restoration and recovery

Prompt IDs are selected deterministically before sampling, round-robin across available G1 subtypes. The first identity-retained clip of each selected prompt is captured. A selected prompt with no retained clips is reported as missing. These observations are conditional on identity retention.

The fitting observer saves the actual detached positive and negative tensors used in the loss. The trusted local checkpoint stores CPU copies of the query and scheduler history, its original EMA adapter, the round-start student adapter, and completed diagnostic results. Original query snapshots survive resume. Each completed query is checkpointed separately, so a later diagnostic failure can resume without retraining the round or recomputing committed queries. An enabled unfinished diagnostic blocks transition to the next round.

The inference replica runs on GPU1 only during prediction or continuation and returns to CPU before reward evaluation. Adapter weights, active adapter, training modes, parameter gradient flags and RNG are restored on success and exceptions. The optimizer-owned policy remains separate on GPU0.

Six candidate videos per audited query are saved under `probe/` and logged to the existing offline W&B run. Diagnostic time is recorded separately in `round_probe.json`; allocated diagnostic GPU-hours for the two-GPU worker are `2 * seconds / 3600`. CPU snapshots increase checkpoint storage, which the job audit reports.

## Stacked PR scope

The Step6 PR is based on PR#8's Steps1–5 branch. It carries the saved-query diagnostics, RNG preservation, actual-target observer, native replay fixes, tests and probe job entry point. Its smoke preset inherits `config/cosmos_logging_smoke.py` and uses the existing prefix-based calibration.

The historical GPU jobs below ran on the original Step6 development branch, where the smoke preset also enabled balanced motion calibration with one training prompt per subtype crossed with seeds42/43/44. The exact diagnostic module and native continuation implementation are retained in this PR. The calibration experiment's query selector, presets, jobs, diagnostic additions and reports are excluded from the PR. Fresh CPU integration and repository tests validate the trimmed branch with444passed,10skipped and12warnings;47focused tests also passed. Ruff, Bash syntax and diff checks passed. A fresh review found no actionable correctness findings. A new end-to-end GPU run of its prefix-calibration smoke preset has not been performed.

## Bounded GPU test

`config/cosmos_probe_smoke.py` and `scripts/train_cosmos_probe.slurm` define a four-prompt smoke test on two A10040GB GPUs, 12CPUs, 192GB host RAM, with a two-hour limit. The test copies four existing frozen training records covering orbit/dolly/lateral and three development records, one per subtype. It attempts four clips per training prompt and evaluates three fixed development queries before/after. The historical jobs enabled balanced motion calibration with one training prompt per subtype crossed with seeds42/43/44. The current PR preset uses the existing prefix-based calibration. Training seed42 and five frames at224x384 are retained. The corrected follow-up uses35 native denoising steps and requested query sigma0.60, with actual sigma and remaining model steps recorded.

The expected output is up to four audited training queries with six candidate continuations each, real optimizer updates, a final checkpoint, and the matched development results. Final-test prompts are excluded. Results will establish whether gains occur in this bounded diagnostic workload; broader quality claims require independent prompts, seeds and longer clips.

## Verification

The tests cover signed gains and nonfinite missingness, per-candidate RNG isolation, independent CPU query copies, native continuation parity, original-EMA restoration after the live EMA changes, a zero-change student with a nonzero student-behavior gap, PEFT and Diffusers adapter interfaces, actual-target observation, round resume, mid-diagnostic failure recovery, rejected-query missingness and transition completeness.

### Initial GPU run9653348

The initial run used8 denoising steps and requested query sigma0.278. It completed all steps0:0 in21:26 on Narval ng10804, with4 actual updates,16 attempted clips and24 candidate continuations. Checkpoint reload/SHA and all4 native endpoint continuation checks passed. Diagnostic time was283.70seconds (0.15761 allocated A100 GPU-hours); the whole allocation used0.71444 A100 GPU-hours. Peak allocated GPU memory was28.71/32.58GiB and Slurm step MaxRSS was61.38GiB.

Initial probe geometry gains were +0.016054 for positive targets (3/4) and +0.009715 for round-end versus round-start student (4/4). The three fixed development queries declined by0.009628 on average (0/3 improved). Interpretation of these initial probe gains was invalidated because the subsequent audit identified two limitations. All queries landed on the final denoising step (actual sigma0.590166), so they exercised no later model evaluation. The audit also inherited training autocast while native generation ran without it. On the first query, identical original EMA and round-start student adapters produced different predictions under that precision mismatch. A CPU regression test reproduced the issue.

The implementation now matches native precision, verifies original-EMA prediction parity and records remaining model steps. The second GPU run uses35 steps and an earlier query. The sampler/query change provides coverage of later model evaluations. Training-quality comparisons across these two runs also depend on those changed settings.

### Native-precision run9656720 and saved-query replay

Job9656720 completed all steps0:0 in18:09 with35denoising steps. The selected query was index30 at sigma0.626635, with4later model calls. Three prompts produced retained audit queries; the lateral prompt had no identity-retained clips. The round made3updates from16attempted clips. Original-EMA prediction replay matched exactly for all3queries. Two endpoint continuations matched at relative L2 around1e-8; the dolly query diverged by0.040806 and was excluded from valid gain aggregates. On the2valid queries, mean positive-target gain was-0.014012 and mean student round gain was-0.016901. Development geometry improved by0.005353 on the3fixed development queries. These small and different query sets cannot establish an association between diagnostic gains and development gains.

The failing no-op continuation exposed numerical cancellation when reconstructing velocity from `(z_q-y0)/sigma_q`. A regression test reproduces loss of the recorded velocity despite an unchanged prediction. The continuation now uses `v_old + (y0-forced_y)/sigma_q`, preserving the original velocity exactly for a no-op while retaining the derivative with respect to the substituted prediction.

`--resume --recompute-probes` rescores a completed round's saved queries using the same student/EMA snapshots and targets. It rejects an unfinished training/development round, archives prior diagnostic result JSON and writes new videos to a unique directory. A checkpoint replay validates the numerical correction without repeating the optimizer updates. The measured replay results follow below.

### Validated checkpoint replay job9659456

Job9659456 completed all Slurm steps0:0 in8:59 on Narval ng31204. It loaded job9656720's trained checkpoint and recomputed18 candidate videos for the3saved queries. All3 original-EMA predictions and native endpoint continuations matched exactly, with zero maximum absolute and relative L2 error. For the first query, identical original EMA and round-start student snapshots also reproduced identical local and final rewards. All3queries have four subsequent model evaluations at actual sigma0.626635 and are valid. One selected lateral prompt remains missing because all its clips failed the identity gate.

| Geometry change | Mean local gain | Local improvements | Mean completed-video gain | Completed-video improvements |
|---|---:|---:|---:|---:|
| Positive target versus original prediction | +0.003285 | 2/3 | -0.001779 | 1/3 |
| Negative target versus original prediction | -0.096624 | 0/3 | +0.009458 | 3/3 |
| Reflected negative target versus original prediction | -0.014400 | 1/3 | -0.002150 | 1/3 |
| Round-end versus round-start student | +0.027665 | 3/3 | -0.012185 | 0/3 |

The student update improved the decoded prediction at the saved query in all3cases, while the completed-video reward declined in all3cases. The positive targets improved final reward in one case and reduced it in two. Even the negative targets, which reduced local reward in all3cases, improved the final reward in all3cases. This pilot identifies a mismatch between local reward optimization and the completed-video reward at these saved queries.

The full student generation on the3fixed development queries improved by0.005353 on average in the source35-step training run. That evaluation changes student predictions throughout the trajectory, while this diagnostic substitutes a prediction at one saved query and continues with the original EMA. The different measurements address different effects. With only3identity-retained training queries and one training seed, the observed sign reversal motivates a wider continuation study across query times and seeds.

The replay's diagnostic phase took223.86seconds, corresponding to0.12437allocated A100 GPU-hours with the two-GPU worker. Its whole allocation used0.29944A100 GPU-hours. The three actual allocations9653348/9656720/9659456 used0.71444/0.60500/0.29944A100 GPU-hours respectively, totaling1.61889. One intervening direct-partition submission was rejected before allocation. Each actual job requested two A10040GB GPUs,12CPUs and192GB host RAM; the replay reused3already-completed optimizer updates from the source checkpoint.

The real checkpoint audit reloads the native scheduler and original EMA snapshots and compares adapters, optimizer, RNG, update count, targets and saved query tensors against the source checkpoint. Results and online logging verification are kept in `logs/curriculum-step6/`.

The original GPU development branch passed457tests with10skips and14warnings. Post-trim PR test results are reported in the PR description. Ruff, Bash syntax and diff checks passed. A fresh reviewer checked the implementation and both numerical corrections. The existing controller-death fixture's startup deadline was expanded from10to30seconds after repeated shared-filesystem startup timeouts; its lock and recovery assertions remain active.

The saved query, target and student comparisons use the existing geometry scorer. The final replay W&B run contains18candidate videos, preserved original run ID and job-scoped display name. [Replay metrics and videos](https://wandb.ai/cvis_tmu/diffusionopsd/runs/fd4035916cc6).

Detailed artifacts are in `logs/curriculum-step6/replay-native/training/round_probe.json`, `logs/curriculum-step6/replay-native-verified_result.json` and `logs/curriculum-step6/replay-checkpoint-audit.log`. Initial outputs remain in the separate `smoke/` and `smoke-native/` directories. The training round's16final-test prompts remain unevaluated.
