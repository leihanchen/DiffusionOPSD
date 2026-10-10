# Step-4 paired inference evaluation

## Experiment

Compare base Wan2.2-TI2V-5B (adapter disabled) with `policy_default` from
`logs/wan22-two-gpu-probe-4962523/training_state_step4.pt`. Require `next_update=4`.
Use the first 16 held-out prompts, with seeds `prompt_index` and `prompt_index+1000`:
32 matched pairs / 64 videos. Initial noise and embeddings are identical within each pair.

Settings: custom Euler, 30 denoising steps, 1280x704, 17 frames, guidance 5.
MP4 export uses 8 FPS, matching training evaluation; FPS does not change generated
frames or tensor-based reward scores. Training presets and the training loop are unchanged.

## Run

From the repository root, submit `sbatch scripts/evaluate_wan_step4.slurm`.
The entry point reuses `scripts/investigate_wan_inference.slurm` and
`scripts/investigate_wan_inference.py`, requesting one node, two A100 GPUs,
eight CPU cores, 128 GB RAM, and two hours. GPU 0 hosts policy/text/VAE;
GPU 1 hosts scorers. There is no backward pass or OPSD target construction.

The reusable CLI accepts `--sampler both|euler|native` (default `both`) and
`--checkpoint-format adapter|training` (default `adapter`). Full training checkpoints
require trusted local pickle input; `--expected-update` enforces the intended update.
The existing four-condition investigation remains the default.

## Outputs and interpretation

Outputs are saved under `logs/wan22-inference-JOB_ID/`: manifest with checkpoint hash
and update number, 64 MP4s, first/middle/last frame images, latent tensors,
per-video JSON/CSV/JSONL scores, a summary, and blinded review files.
Conditions A/C mean base/trained Euler. Metrics reuse GeoReward and VideoReward:
geo, rigid, dino, s_id, motion, VQ, MQ, TA, Overall. Differences are trained minus base.
Average seeds within each prompt, then average prompts equally. Existing p_q and
success checks are retained; they are descriptive checks, not statistical evidence.

W&B runs offline in `cvis_tmu/diffusionopsd`. Each sample explicitly commits history;
the final `investigation/pairs` table contains both videos and score differences for
all 32 pairs. Verify a complete manifest, 64 finite score records and decodable videos,
and 32 complete pairs before syncing the offline run. Confirm online media after sync.
Completed local sample files remain available if the job fails.

Review prompt-level regressions alongside overall averages. These 16 prompts include
previously monitored examples and cover eight repeated motion templates; this remains
a screening evaluation, not a claim of broad generalization or full-length video quality.

## Validation

Run `python -m pytest tests/video -q`, focused Ruff checks, and shell syntax checks
before submission. Tests cover adapter extraction, update mismatch, paired inputs,
adapter restoration, 64-video/32-pair exports, and preservation of four-condition behavior.
