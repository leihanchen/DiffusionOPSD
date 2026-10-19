# Evaluate base, step 7, and step 50 on 64 prompts

Generate 192 videos (64 held-out prompts, seed=prompt index, base and policy_default adapters at updates 7 and 50). Reuse inference investigation runner. Native UniPC, flow shift 5, 1280x704, 121 frames, 50 steps, guidance 5, 24 FPS, empty negative prompt. Validate recorded checkpoint updates and hashes, complete adapter replacement, and identical inputs across the three conditions.

Add opt-in native checkpoint comparisons, step50-step7 deltas alongside both base comparisons, table-focused W&B logging, prompt strata 1–2/3–16/17–64, and verification of exported videos. Keep legacy modes compatible. Save every completed sample immediately and preserve partial outputs on failure. Export blinded review files.

Slurm: Killarney aip-wangcs/gpubase_h100_b3, 1 node/task, 2 H100 GPUs, 4 CPUs, 128 GiB RAM, 24 hours, no successors. GPU 0 policy/text/VAE, GPU 1 scorers. Expected duration 10–14 hours from previous ~181 sec/video. Offline W&B to cvis_tmu/diffusionopsd; later verify all history, tables and 192 uploaded videos. Name contains job ID, purpose, cluster.

Validation: focused regression tests, full video suite, lint and shell syntax, local checkpoint/model/scorer preflight, scheduler dry run. Acceptance: 192 finite/readable 121-frame 24-FPS videos, 64 complete triplets and 192 comparison rows, completed manifest and verified online W&B. Preserve training configuration, other work, and untracked SIF; no commit or push requested.
