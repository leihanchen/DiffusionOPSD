# Wan inference investigation

This standalone command compares four conditions without editing OPSD training:

| ID | Weights | Generation |
|---|---|---|
| A | Pretrained, LoRA disabled | Existing custom Euler rollout |
| B | Pretrained, LoRA disabled | Native WanPipeline with checkpoint UniPC scheduler |
| C | Trained default LoRA | Existing custom Euler rollout |
| D | Trained default LoRA | Native WanPipeline with checkpoint UniPC scheduler |

The Slurm script selects `config/wan22_inference_debug.json` and generates four
videos: the first held-out chessboard prompt, seed 0, and all four conditions.
The independent inference preset sets width×height to **1280×704**, **121 frames**,
**50 denoising steps**, and **guidance 5**, matching those parameters in the
bundled Wan2.2 model example. It does not import or change training configs.

Every pair receives clones of the same FP32 noise and text embeddings. Both paths
retain the existing BF16 VAE decoder, empty negative prompt, and 8 FPS playback.
This is not a complete official-reference reproduction: that example uses an FP32
VAE, a populated negative prompt, and 24 FPS. The 121 frames here play for about
15 seconds. Each condition gets a fresh scheduler.
Euler uses the existing sigma-to-timestep calculation and autocast; native Wan
retains its own input casting and timestep handling. Differences describe the
whole generation path, not an isolated solver effect.

## Submit on Narval

From the repository root:

```bash
mkdir -p logs
sbatch scripts/investigate_wan_inference.slurm
```

The default policy is `logs/wan22-debug-4175442/policy_4.pt`. Override it with:

```bash
POLICY_PATH=/absolute/path/policy.pt sbatch scripts/investigate_wan_inference.slurm
```

The script requests two A100 GPUs, 8 CPUs, 128G host RAM, and one hour.
Policy/text encoder and VAE share cuda:0; geometry and quality scorers use
cuda:1. CPU math-library threads are capped at the allocated CPU count, and
the cluster selects a compatible partition. It uses the repository container and local weights, with Hugging Face
and W&B offline. It does not download weights or sync W&B automatically.
Create `logs/` before submission so Slurm can open its output files.

Job 4422081 completed 16 videos at the previous 832×480, 17-frame, 30-step
settings in 10 minutes 36 seconds, using 42.7 GiB peak host RSS. This does not
validate the allocation for the larger preset: it has about 14 times as many
latent tokens per video and more denoising steps. Keep the allocation for the
initial four-video attempt and inspect runtime and peak memory. An OOM or other
failure exits nonzero; the command never silently reduces dimensions, frames,
or scoring coverage. No backward pass or optimizer state is needed.

The Python interface, inside the training environment, is:

```bash
python scripts/investigate_wan_inference.py \
  --config config/wan22_inference_debug.json \
  --policy logs/wan22-debug-4175442/policy_4.pt \
  --output-dir logs/wan-inference-manual \
  --num-prompts 1 --seed-offsets 0 \
  --policy-device cuda:0 --reward-device cuda:1 --vae-device cuda:0
```

Use `--help` for model, prompts, dimensions, steps, guidance, and device options.
Configuration precedence is legacy defaults → JSON preset → explicit CLI options,
regardless of argument order. The only accepted JSON keys are `height`, `width`,
`num_frames`, `num_steps`, and `guidance_scale`; unknown keys and invalid values
are rejected. For example, add `--num-steps 30` to override the preset's 50 steps.
Without `--config`, the CLI retains its legacy defaults: 832×480, 17 frames,
30 steps, guidance 5, two prompts, and seed offsets 0 and 1000 (16 videos).
The output directory must be new or empty. Resuming and automatic retries are
not implemented; choose a fresh directory for each attempt. This tool currently
supports the Wan2.2 TI2V geometry and the repository's rank-32 LoRA checkpoint format.

## Evidence and interpretation

Outputs under `logs/wan22-inference-JOBID/` include:

- `manifest.json`: resolved arguments including the inference config's absolute
  path and SHA-256 (`arguments.config` and `arguments.config_sha256`), software/Git identifiers, checkpoint hash,
  model config hashes, weight-file size/mtime identifiers (not weight hashes),
  prompt text, seeds, and saved input latent hashes.
- `samples.jsonl` and `samples.csv`: all per-video scores, condition, pairing
  identifiers, input hashes, timing, and peak allocated/reserved CUDA memory.
- Per-condition JSON: actual scheduler timesteps, model timesteps, sigmas, and
  precision choices. MP4s, first/middle/last PNGs, and final latent tensors sit
  beside these. Initial tensors are saved once per prompt/seed pair.
- `summary.json`: condition means, B−A, C−A, D−B, and (D−B)−(C−A) contrasts.
  Seeds are averaged within each prompt before averaging prompts. Training
  `p_q` and success flags compare C with A and D with B; per-pair results are
  retained. Scores include geometry, rigid and DINO terms, identity, motion,
  VQ, MQ, TA, and Overall. No statistical significance is claimed.
- Offline W&B: per-condition score series, final scalar comparisons, and all
  captioned videos. Run config includes resolved parameters, `config`, and
  `config_sha256`. The logging step is the completed video index, not a
  training update.
- `review/`: anonymous randomized video copies, a prompt manifest, blank
  `ratings.csv`, and randomized comparison pairs in `preferences.csv`. Review
  the full videos for clarity, prompt adherence, motion, and temporal artifacts.
  Enter the preferred clip ID or `tie` and a reason. Keep the separately saved
  `review_condition_map.json` hidden until review is complete. Repeating the
  same experiment order reproduces the review randomization (seed 2026).

Timing synchronizes all selected CUDA devices. `compute_seconds` covers
sampling, decoding, and scoring; `duration_seconds` also includes video/PNG
encoding and latent saving, but excludes W&B logging. Peak memory includes
resident model allocations; reserved memory can reflect prior allocator use.

A failure produces `failure.json`, preserves completed samples, exits nonzero,
and does not leave a complete summary. A successful run means all requested
conditions and artifacts completed, even if image quality is poor. It does not
by itself establish that more training is appropriate. Inspect paired results
before expanding the experiment or changing the RL scheme.
