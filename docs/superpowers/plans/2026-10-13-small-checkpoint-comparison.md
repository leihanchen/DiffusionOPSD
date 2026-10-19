# Small checkpoint comparison

Compare base, saved step 5 and saved step 7 on the first two held-out prompts, seeds 0 and 1: six videos, four base comparisons. Reuse Euler 30 steps, 1280x704, 17 frames, guidance 5, 8 FPS. Extend existing inference runner with a mutually exclusive checkpoint manifest; retain legacy modes. Validate complete adapter replacement and checkpoint update numbers. Save finite sample records and videos immediately, log offline W&B, sync and verify after completion.

Allocation: Killarney aip-wangcs/gpubase_h100_b2, one H100 >=80GB, four CPUs, 64 GiB RAM, 30 minutes. Sequential inference, all modules on cuda:0. No automatic resubmission. Preserve unrelated changes; no commit requested.

Tasks: checkpoint-list loading and generalized comparisons; entrypoint and Slurm configuration; focused and full video tests; submission and outcome monitoring.
