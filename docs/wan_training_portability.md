# Moving the Wan training directory

Copy the complete repository directory, including `containers/diffusionopsd.sif`, `hf/`, `video_reward_ckpts/`, prompt files, and the `logs/` directories containing checkpoints and calibration references. Git does not include the model/cache/container artifacts.

From the copied directory, use the same relative submission arguments. For example, this validates the one-node resume request without submitting:

```bash
python scripts/submit_opsd_video_wan_scaled.py \
  --nodes 1 --gpus-per-node 4 --single-node-smoke --max-updates 1 \
  --mem 640G --time 04:00:00 \
  --resume-from logs/wan22-two-gpu-probe-4962523/training_state_step4.pt \
  --allow-topology-change --expected-resume-update 4 --eval-freq 2 \
  --run-id relocated-probe --dry-run
```

Remove `--dry-run` to submit. No repository-location edits are needed. The scaled entry point still uses its submission wrapper to supply its allocation and training options; the standalone epoch/debug entry points retain their existing submission commands.

## Paths that follow the copy

- The launcher determines the repository root from its own file location. Paths in saved launch options and exported training settings are relative to that root. It also works when invoked from a different working directory.
- Successor submissions load configuration and checkpoints from the copied directory. Legacy saved options containing an absolute `config/` path are resolved to the copied configuration when present.
- Checkpoints save repository-relative calibration references and output-directory metadata. When loading existing checkpoints with absolute `logs/.../calibration/...` references, the loader prefers the copied files. Missing relative references fail explicitly.
- Legacy checkpoint configuration hashes remain enforced. If the source directory moved, the loader reconstructs the original run directory from its recorded calibration paths, then verifies the complete original hash. Other training settings cannot be changed through this compatibility path.
- The offline preflight check converts VideoReward's Qwen loader path to `video_reward_ckpts/Qwen2-VL-2B-Instruct`. This local, atomic normalization requires no downloads and also handles a copied older cache.

Absolute paths are still resolved at runtime for container mounts and model loading. These are derived from the current location. An explicitly supplied artifact outside the repository remains external and must remain accessible. Historical submission logs may retain the original location; they do not determine new submissions.

The folder copy does not change Slurm accounts, GPU types, memory requirements, the cluster module environment, or already-submitted jobs. The current deployment requests Narval A100 resources. Another cluster may require allocation/environment changes even though its local artifact paths move correctly.

## Validation

CPU tests cover a real directory rename between submission and successor launch, legacy hash reconstruction with incompatible settings still rejected, relocated calibration references, and idempotent model-config normalization. The original step-4 checkpoint was also loaded on CPU: its legacy hash verified using a simulated relocated source directory, and both calibration references round-tripped through repository-relative storage. The full video suite passed (144 tests). The existing local offline artifact tree passed preflight after normalization.
