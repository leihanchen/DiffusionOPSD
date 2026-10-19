# Wan2.2 base-model inference ablation

Run from the repository root:

```bash
mkdir -p logs
sbatch scripts/ablate_wan_inference.slurm
```

This standalone experiment uses the base Wan2.2-TI2V-5B checkpoint and native
UniPC. It does not load a trained adapter or change the training configuration.
The first nonempty prompt from `data/video_motion/test.txt` is generated with
seeds 0 and 1000 under each condition:

| Condition | Width × height | Steps |
| --- | --- | --- |
| control | 1280 × 704 | 50 |
| steps30 | 1280 × 704 | 30 |
| resolution480 | 832 × 480 | 50 |

All conditions use 121 frames, guidance 5, an empty negative prompt, BF16 model
and VAE, and 8 fps playback. Settings live in
`config/wan22_inference_ablation.json`; validation fixes this experiment's scope.
Control and steps30 receive identical initial noise for each seed. Resolution480
uses the same seed with a different noise shape, so it is not an identical-noise
comparison. A fresh scheduler is constructed for every generation.

The Slurm allocation is two A100 GPUs, eight CPUs, 128 GiB RAM, and two hours.
Policy, text encoder, and VAE use GPU 0; geometry and VideoReward scorers use GPU 1.

## Outputs and W&B

`logs/wan22-ablation-JOBID/` contains six MP4s, first/middle/last PNGs, initial and
final latents, per-sample scheduler traces, CSV/JSONL scores, paired differences,
and a manifest recording configuration, source revision, and model metadata.
Failed runs preserve completed samples and write `failure.json`.

W&B is explicitly offline. All six videos are recorded under
`ablation/videos/{control,steps30,resolution480}`, with condition, seed,
resolution, and steps in their captions. Each condition has two history entries.
Metrics include geometry components, motion, VideoReward VQ/MQ/TA/Overall,
elapsed time, and peak CUDA memory. W&B summaries include condition means and
variant-minus-control differences. After completion, sync the run with:

```bash
wandb sync --entity cvis_tmu logs/wan22-ablation-JOBID/wandb/offline-run-*
```

One prompt and two seeds provide a diagnostic comparison, not a general quality
estimate. Inspect videos alongside scores; resolution changes pixel-based
motion and geometry scales. GPU inference is validated by the submitted job;
CPU tests exercise experiment orchestration and real offline W&B media logging
with stub models.
