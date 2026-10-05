# Wan2.2 OPSD full training pass

Submit once from the repository root:

```bash
mkdir -p logs
sbatch scripts/train_opsd_video_wan_epoch.slurm
```

The separate preset `config/wan22_ti2v_epoch.py` trains on each of the 200
`data/video_motion/train.txt` prompts once, in a saved random order. Four
workers each process one prompt and four clips per optimizer update: 50 updates
and 800 training rollouts. The Wan2.2 grid is 1280×704, 17 frames, custom Euler
30 steps, guidance 5. The per-clip training microbatch remains one.

The job requests four Narval nodes, each with three A100 GPUs, 12 CPU cores,
and 448 GiB host RAM in the 24-hour `gpubase_bygpu_b3` partition. On each
worker, GPU 0 holds the LoRA policy, GPU 1 holds the scorers, and GPU 2 holds
the VAE. Workers aggregate rewards before advantage normalization and aggregate
gradients before each synchronized optimizer step. Only worker 0 evaluates and
logs to offline W&B. Held-out evaluation runs at updates 0, 25, and 50.

The initial job ID defines `logs/wan22-epoch-JOBID/`. Each completed update
atomically replaces `training_state.pt`, which contains the optimizer, both
adapters, prompt order/position, RNG states, tracker, calibration, and evaluation
baseline. This is the resume checkpoint; the inference-ready model is written
only at update 50 as `policy_50.pt`. A completed job with work remaining submits
one `afterok` successor with the same run ID and logs it in `job_chain.txt`.
Failed jobs do not advance the chain. Each segment writes `job_SEGMENT_ID.json`
and an offline W&B run in the shared run directory. W&B runs share a group name
based on the initial job ID and use absolute optimizer-update steps.

The first production job records per-worker GPU allocation peaks, process RSS,
and update phase times. It runs on the four-node allocation directly; no
three-GPU comparison is performed. The 720p resource fit and wall time remain
unknown until this job executes. A hard time limit or failure preserves the last
completed-update checkpoint. When investigating a failed job, check Slurm
stdout/stderr and `training_state.pt` before resubmitting with
`WAN_EPOCH_RUN_ID=INITIAL_JOB_ID`.

The container `containers/diffusionopsd.sif` is a local artifact and is not
committed. The Slurm job uses the repository's `src/` overlay and offline model
cache. To sync a completed segment manually, use the offline run directory
under `logs/wan22-epoch-JOBID/wandb/`.
