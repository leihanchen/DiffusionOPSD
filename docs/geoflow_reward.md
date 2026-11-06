# GeoFlow reward compatibility

The video trainers use the default training-server score from GeoFlow commit
`fac2426820e303eb1c0ffd1cc43e58585f99f87b`. The authoritative implementation is
[reward_server.py](https://github.com/geometryflow/GeoFlow/blob/fac2426820e303eb1c0ffd1cc43e58585f99f87b/geoflow/reward_server.py).
Related work is tracked in [EAS-14](https://linear.app/cvis-tmu/issue/EAS-14/reward-inconsistency-with-geoflow).

## Scoring contract

Inputs are `[B,T,3,H,W]` RGB in `[0,1]`, with at least two selected frames.
Round RGB to 8-bit values, run DA3 with process resolution 504 and
`upper_bound_resize`, and score at DA3's output dimensions rounded down to
multiples of eight. Image resizing is antialiased. Intrinsics are per frame.
The server accepts already-selected frames and its `fps` argument performs no
subsampling. Existing training frame selection occurs before this interface.

Adjacent geometry uses L2 flow disagreement divided by the sum of flow
magnitudes plus **5**, multiplied by clipped relative depth consistency.
Depth is forward projected with a nearest-pixel minimum-z buffer. Semantic
consistency compares source DINO features against features extracted from the
target image sampled at source coordinates plus forward flow. Invalid patches
are masked. Empty valid support scores −1. Confidence, adjacent occlusion,
dynamic masking, and the motion penalty are disabled by the pinned defaults.

Long pairs use gap 8 and stride 4, rigid flow, depth visibility, and a 4-pixel
flow threshold for semantic support. Their score weights geometry by 0.25 and
semantics by 0.75. Empty semantic support gives a neutral long-pair semantic
score. The final score is 0.7 times the mean of all pair scores plus 0.3 times
the minimum adjacent score.

`geo` is this final score. `rigid` and `dino` summarize adjacent components,
matching the server diagnostics. `s_id = 1 + mean(adjacent r_dino)` retains the
identity gate's cosine scale. `motion` is mean adjacent WAFT magnitude in
processed-resolution pixels. Recalibrate gates when switching reward versions.
The separate historical motion-only calibration utility operates in its input
resolution and its old thresholds require recalibration through the trainer.

The released code differs from the paper's simpler adjacent-pair description,
including the constant 5, long pairs, worst-pair aggregation, and target-to-source
image warp. This implementation follows the selected released code.

## Gradient contract

DA3 runs once per reward call under `no_grad`. Geometry is refreshed for each
OPSD target evaluation and stays detached during that evaluation's backward
pass. Frozen WAFT and DINO propagate input gradients through image sampling
and the VAE. Pair computations use activation checkpointing.

RGB8 rounding uses a straight-through estimator. Its forward values match
rounded RGB8 and its backward uses the clipped continuous input derivative.
Hard masks retain their piecewise derivatives. Vector norms use the zero
subgradient at zero, avoiding `sqrt(0)` backward NaNs without changing the
score. WAFT's own iterative-flow stop-gradients remain as implemented upstream.
Consequently this is an explicitly partial/surrogate gradient of the scoring
pipeline. Finite-difference checks hold DA3 predictions and masks fixed and
exercise continuous scoring before quantization.

## Pinned model code and setup

Use dedicated source checkouts. Loaders verify the source revision and fail
with an actionable message if it differs.

```bash
git clone https://github.com/geometryflow/GeoFlow.git /path/to/GeoFlow
git -C /path/to/GeoFlow checkout fac2426820e303eb1c0ffd1cc43e58585f99f87b
git -C /path/to/GeoFlow submodule update --init --recursive third_party/WAFT third_party/Depth-Anything-3
export GEOFLOW_WAFT_DIR=/path/to/GeoFlow/third_party/WAFT
export GEOFLOW_DA3_DIR=/path/to/GeoFlow/third_party/Depth-Anything-3
pip install -e "$GEOFLOW_DA3_DIR"
```

The WAFT revision is `b152ff1cad1af8c185ee7b141997c48ff3334c87`, using the a1
`tar-c-t.json` config, a1 `waft_tar_c_t.pth`, and the official inference wrapper
with scale 0, tiling disabled, and training-size padding disabled. Place the
Depth Anything V2 Small backbone checkpoint in the pinned WAFT checkout's
`depth-anything-ckpts` directory, following GeoFlow's setup instructions.
DA3 revision is `2c21ea849ceec7b469a3e62ea0c0e270afc3281a`.

The existing `VIDEO_REWARD_CKPT_PATH` supplies `depth-anything-3-large-v1.1`,
`dinov2-base`, and `waft_tar_c_t.pth`. Model identities are
`depth-anything/DA3-LARGE-1.1` and `facebook/dinov2-base`.

## Verification and provenance

```bash
GEOFLOW_REFERENCE=/path/to/GeoFlow python -m pytest tests/video -q
python scripts/check_video_reward_setup.py --device cuda --frames 17 --reference /path/to/GeoFlow
```

The CPU parity tests execute pure functions and the complete original
`score_video` method with identical stub estimator outputs, checking components
and final aggregation at `atol=1e-5, rtol=1e-4`. The real-estimator command uses
shared loaded backbones, compares against the original server, checks gradients
and both target directions, and reports elapsed time and peak GPU memory.
Use `--clip-tensor /path/to/clip.pt` for a fixed decoded RGB clip.

Wan and Cosmos config/checkpoint metadata includes `geometry_reward.version`
and `gradient_policy`. Evaluation outputs include the same provenance. Missing
reward versions identify legacy results. Cross-version success comparisons
raise an error and require rescoring. Resume config validation prevents mixing
old reward calibration and new reward training state.

Real-model numerical parity and Wan/Cosmos GPU updates must be verified in the
training environment before treating this change as validated for production.
