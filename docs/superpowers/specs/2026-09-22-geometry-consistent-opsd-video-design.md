# Geometry-Consistent DiffusionOPSD for Video

Date: 2026-09-22
Status: approved (2026-09-22). Implementation plan: `docs/superpowers/plans/2026-09-22-geometry-consistent-opsd-video.md`

## 1. Goal

Adapt DiffusionOPSD ([arXiv 2608.24646](https://www.alphaxiv.org/abs/2608.24646)) from
image reward optimization to video, with the reward built for 3D spatial and temporal
consistency rather than prompt alignment or aesthetics. The output is a post-trained
rectified-flow video model whose clips have lower depth/flow reprojection error and
stable object identity, while standard video quality stays within an agreed margin.

Non-goals: autoregressive or unified generators; learning a new reward model; long-horizon
(minute-scale) rollouts; physical plausibility beyond rigid-scene geometry.

## 2. Background: what DiffusionOPSD needs from a reward

DiffusionOPSD uses three qualified rewards (paper Sec. 3.1, App. B.4):

| Name | Evaluated on | Needs gradient | Role |
|---|---|---|---|
| Local reward \(\tilde R(y,c)=R(D(y),c)\) | decoded clean-output prediction at a low-noise query | yes | builds positive and negative targets \(\bar y_\pm\) (Eq. 6–9) |
| Endpoint reward \(r_k=R(D(x_0),c)\) | fully denoised rollout endpoint | no | group-normalized fitting weight \(\omega\in[0,1]\) (Eq. 5) |
| Fixed-suffix reward \(F_q(y)\) | endpoint reached by inserting \(y\) at the query and finishing with the frozen behavior policy | no | diagnostic only; separates construction gain from fitting gap (Eq. 19) |

The paper uses one reward model for both training roles. It optimizes one reward per run.
\(\omega\) is a mild lever: at the ideal minimizer it cancels when the positive and negative
targets are symmetric (Eq. 15), so it only re-balances the two branches.

Video changes three things: \(y\) is a clip latent, \(D\) is a video decoder, and the
reward must see several frames at once to say anything about consistency.

## 3. Decisions taken during brainstorming

| Question | Decision |
|---|---|
| Where does the consistency signal enter? | Both the local target and the endpoint weight. |
| What does the reward evaluate? | A decoded video clip, never a single frame. |
| Which error dominates the target gradient? | Geometry. Temporal identity acts as a gate, not a gradient. |
| What counts as success? | Automatic metrics on a held-out prompt set: geometry up, temporal identity above a floor, standard video quality within an agreed margin. |
| Reward source | GeoFlow-style depth + flow reprojection score ([arXiv 2605.18365](https://www.alphaxiv.org/abs/2605.18365)), not a 4D-reconstruction critic and not a VLM judge. |
| Role of a VLM judge | Quality guard only: open VLM pairwise win-probability used as a hard mask on \(\omega\) and as a held-out success criterion. Not a gradient source. |

## 4. Reward design

### 4.1 Geometry reward \(R_{geo}\) (local reward and endpoint reward)

Follow GeoFlow's primary configuration. For each consecutive decoded frame pair
\((I_t, I_{t+1})\):

1. Predict metric depth, intrinsics, and relative camera pose for the pair with
   Depth Anything 3 Large v1.1 (frozen).
2. Predict optical flow \(f_t\) with WAFT (frozen).
3. Rigid flow \(\hat f_t\) is the flow induced by depth + camera motion. The geometry term is
   the confidence-weighted negative reprojection residual
   \(R_{rigid,t} = -\,\mathrm{mean}_p\, w_p\,\|f_t(p)-\hat f_t(p)\|_1\), with \(w_p\) the
   Depth Anything 3 confidence map.
4. Warp DINOv2-base patch features from \(I_{t+1}\) back along \(f_t\) and score cosine
   similarity to \(I_t\): \(R_{dino,t}\).
5. \(R_{geo} = \mathrm{mean}_t\,[\,0.5\,R_{rigid,t} + 0.5\,R_{dino,t}\,]\).

All three vision models are frozen and differentiable with respect to the decoded frames, so
\(\nabla_y \tilde R_{geo}\) exists through the video decoder. GeoFlow uses this score only as a
scalar in a policy-gradient update; using its gradient for target construction is new to this
design and is the first thing the fixed-suffix probe (Sec. 6) must confirm.

The same \(R_{geo}\) is the endpoint reward. \(\omega\) is computed with the paper's Eq. 5
unchanged (per-prompt centering, global batch std, clip to \([-1,1]\), map to \([0,1]\)).

### 4.2 Temporal identity gate

Define the clip identity score \(s_{id} = \mathrm{mean}_t R_{dino,t}\) on the decoded
behavior-policy clip at the query (the anchor \(y_0\)). If \(s_{id} < \tau_{id}\), the query
is dropped: no targets are built and the tuple is not added to \(D_i\). Rationale: a clip
whose objects already change identity has unreliable flow tracks, so its geometry gradient
is noise. The gate uses the anchor, not the targets, so it cannot be gamed by the ascent step.

### 4.3 Motion floor (anti-freeze)

Stream4D ([arXiv 2608.19556](https://www.alphaxiv.org/abs/2608.19556)) documents that
rigid-reconstruction rewards are maximized by freezing the video. \(R_{rigid}\) has the same
failure: a static clip has zero flow and zero rigid residual. Define
\(m = \mathrm{mean}_{t,p}\,\|f_t(p)\|_2\) on the decoded endpoint clip. If \(m < \tau_{motion}\),
set \(\omega = 0\) for that trajectory so it contributes only to the repulsive negative branch.
The prompt set (Sec. 5.3) is also restricted to prompts that call for camera or object motion.

### 4.4 Quality mask

For each prompt, generate one reference clip with the frozen base model at the same seed
before training starts and keep it fixed (mirrors the paper's fixed-reference VLM-Pairwise
protocol, with the base model in place of Seedream 5.0 Pro). At each outer iteration, an open
video-capable VLM (default Qwen2.5-VL-7B, prompted for pairwise preference on overall
quality) returns \(p_q = P(\text{rollout} \succ \text{reference})\) for the decoded endpoint.
If \(p_q < \tau_q\), set \(\omega = 0\). This is a hard, asymmetric use of \(\omega\) by design,
because the soft weight alone has little effect (Sec. 2). The VLM is never differentiated and
runs only once per rollout endpoint.

### 4.5 Effective fitting weight

\[
\omega_{eff} = \omega_{Eq.5}\cdot \mathbb{1}[m \ge \tau_{motion}]\cdot \mathbb{1}[p_q \ge \tau_q]
\]

The branch loss is the paper's Eq. 13 with \(\omega_{eff}\) in place of \(\omega\). Nothing
else in the loss changes.

## 5. Integration into the training loop

### 5.1 Base model and assets

- Base policy: an open rectified-flow text-to-video model with a public latent video decoder.
  Default: Wan 2.1 T2V-1.3B. This is an assumption to confirm at plan time; any rectified-flow
  video model with the \(y = z - \sigma v\) clean-output map works.
- Frozen reward assets to download: Depth Anything 3 Large v1.1, WAFT, DINOv2-base,
  Qwen2.5-VL-7B. None of these, and no video model weights, are in this project's bundled
  evidence; they must be downloaded before any run.

### 5.2 Algorithm (deltas from paper Algorithm 1)

1. Roll out the frozen behavior policy for \(K\) clips per prompt. Decode endpoints. Compute
   \(r_k = R_{geo}\), motion \(m\), and \(p_q\). Compute \(\omega_{eff}\).
2. Select the query nearest \(\sigma^\star\). Compute the anchor \(y_0\), decode it, compute
   \(s_{id}\). Apply the identity gate.
3. Build \(\bar y_\pm\) with Eq. 6–9 using \(\nabla_y \tilde R_{geo}\). Trust-region radius
   \(\rho\) and steps \(M_{tgt}\) start at the paper's defaults.
4. Store \((c, z_q, \sigma_q, \omega_{eff}, \bar y_+, \bar y_-)\). Fit with Eq. 13,
   \(M_{fit}=1\). EMA-update the behavior policy.

Memory: target construction backpropagates through the video decoder and three vision models
on a full clip. Start with target-construction microbatch 1 and a short clip (e.g., 16–25
frames at the model's native resolution). Run the reward stack as a separate process on
dedicated GPUs, as the paper does for its VLM rows.

### 5.3 Prompts

Training prompts: motion-explicit text prompts (camera pans, orbits, dollies; objects
translating through the scene). Held-out prompts: a disjoint set of the same kind, drawn
from the camera-control and 3D-consistency subsets of the WorldScore and VBench-2.0 prompt
suites identified in the earlier literature review, so the held-out numbers are comparable
with published baselines.

## 6. Diagnostics before training: fixed-suffix probe

Before any multi-day run, reproduce the paper's same-query probe (Sec. 4.4–4.5) on 64–128
held-out clips with the geometry reward:

- \(G_{construct} = F_q(\bar y_+) - F_q(y_0)\) must be positive on the majority of clips, and
  the alignment between \(\nabla_y \tilde R_{geo}\) and \(\nabla_y F_q\) must be positive. If
  not, the local geometry gradient does not point toward endpoint consistency and the
  local-target interface is wrong for this reward; stop and revisit Sec. 4.1.
- \(G_{realized}\) after one fresh optimizer update, and the fitting gap \(G_{fit}\) from
  Eq. 19. Report the ordering-reversal rate against a matched-radius random target as the
  paper does.

Pass criterion: median \(G_{construct} > 0\), positive alignment on \(\ge 60\%\) of clips,
and \(G_{realized} > 0\) in aggregate. Parameters are restored between clips.

## 7. Success criteria (held-out set, automatic)

Compared with the frozen base model on the same prompts and seeds:

| Criterion | Metric | Threshold |
|---|---|---|
| Geometry up | mean \(R_{geo}\); additionally the 3D/camera-consistency dimensions of WorldScore and VBench-2.0 | \(R_{geo}\) improves; external dimensions do not regress |
| Identity above floor | mean \(s_{id}\) | \(\ge\) base-model mean minus 0.02 |
| Quality within margin | VLM pairwise \(p_q\) vs base; VBench imaging-quality and aesthetic-quality | \(p_q \ge 0.45\); VBench dimensions within 2% relative of base |
| Not frozen | mean motion \(m\) | \(\ge 0.9\times\) base-model mean |

Thresholds \(\tau_{id}, \tau_{motion}, \tau_q\) for training are set from base-model
statistics on the training prompts before the first outer iteration: \(\tau_{id}\) and
\(\tau_{motion}\) at the base model's 10th percentile, \(\tau_q = 0.4\).

## 8. Risks

| Risk | Mitigation |
|---|---|
| Local geometry gradient does not improve endpoint geometry | Sec. 6 probe is a hard go/no-go gate before training. |
| Static or near-static clips win on \(R_{rigid}\) | Motion floor on \(\omega\), motion-explicit prompts, "not frozen" success criterion. |
| Depth/flow estimator errors become the target | Confidence weighting from Depth Anything 3; GeoFlow's ablation shows the score survives swapping to RAFT / Pi3X, so estimator choice can be varied if needed. |
| Geometry gain paid for with quality | Quality mask on \(\omega\) plus held-out quality margin. |
| \(\omega\) too weak to guard anything | Guards are hard masks (\(\omega=0\)), not soft weights. |
| Video latent dimensionality inflates the fitting gap | Measured directly by \(G_{fit}\) in the probe; start at paper defaults for \(\rho, M_{tgt}, \beta\) and tune only if the probe shows under-realization. |
| Compute | Short clips, microbatch 1, reward stack on separate GPUs, VLM evaluated once per endpoint only. |

## 9. Deliverables for the implementation plan

1. Reward module: differentiable \(R_{geo}\), \(s_{id}\), \(m\) on decoded clips; frozen
   Depth Anything 3 / WAFT / DINOv2-base wrappers.
2. Quality service: fixed reference clips per prompt, VLM pairwise scorer returning \(p_q\).
3. OPSD video loop: clean-output map for the chosen video model, query selection, target
   construction, \(\omega_{eff}\), branch loss, EMA.
4. Fixed-suffix probe script and report (Sec. 6).
5. Held-out evaluation script producing the Sec. 7 table.
