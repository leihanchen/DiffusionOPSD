# Narval isolated-worker NCCL check

The communication-only allocation **5286646** requested one node, two A100 GPUs, four CPU cores, 128 GiB host RAM, and 15 minutes. It completed in 4 minutes 51 seconds. Tests used two processes, each with one visible GPU; their GPU UUIDs differed and local CUDA operations passed.

| Environment change | Result |
| --- | --- |
| Defaults | First metadata gather failed in `ncclP2pImportShareableBuffer`, P2P/CUMEM, CUDA 101: invalid device ordinal |
| `NCCL_CUMEM_ENABLE=0` alone | Legacy P2P/IPC import failed: invalid argument |
| `NCCL_P2P_DISABLE=1` alone | Shared-memory cuMem import failed in `ncclShmImportShareableBuffer`, CUDA 101 |
| `NCCL_P2P_DISABLE=1`, `NCCL_CUMEM_HOST_ENABLE=0` | Metadata gather, tensor reductions, backward pass and synchronized toy optimizer update passed on both ranks |

The failure is isolated to CUDA memory-handle imports between workers in the current Slurm/container stack. This test does not determine which driver, container, or device-isolation component causes the imports to fail. It does establish a working transport workaround: disable GPU P2P and use POSIX host shared memory. `NCCL_CUMEM_ENABLE=0` is unnecessary for this workaround.

Observed environment: A100 40 GiB, driver 580.178.04, PyTorch 2.14.0+cu130, CUDA runtime 13.0, NCCL 2.30.7. Logs reported shared-memory transport for the passing test. The passing step's Slurm MaxRSS was approximately 1.49 GiB per task; 16 GiB per node provides ample headroom for a communication-only check. Host RAM for real Wan training must be sized separately.

The scalable training launcher now supplies the two verified environment defaults explicitly to Apptainer. Explicit environment overrides remain available for later investigation. Training settings and the existing three-GPU entry points are unchanged.

## Exact two-GPU worker check

Submit from the repository root, with `logs/` present:

```bash
mkdir -p logs
sbatch scripts/probe_wan_nccl.slurm
```

This standalone diagnostic requests one node, four A100 GPUs, two workers, two CPU cores per worker, 16 GiB per node, and ten minutes. It uses Slurm's task GPU isolation and the training container. Both local GPUs are exercised before NCCL initialization. The test validates four distinct GPU UUIDs, object gathering, tensor reductions, and gradients normalized by the retained sample count. The gradient checks include balanced retention and a worker with no retained samples. Both ranks must obtain the serial-reference gradient and identical optimizer results.

Results are written to `logs/nccl-probe-JOBID/rank_0.json`, `rank_1.json`, and `summary.json`; Slurm stdout and stderr use `logs/wan-nccl-probe-JOBID.out` and `.err`. A passed communication probe does not verify model loading, OPSD target construction, training memory consumption, or a Wan optimizer update.

Original experiment evidence is retained locally under `logs/nccl-diagnostic-5217175/`. The full training retry 5280212 was held during diagnosis and requires a separate decision before release.
