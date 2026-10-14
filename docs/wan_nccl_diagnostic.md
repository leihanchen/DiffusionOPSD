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

## Killarney: route-based socket selection (job 6160389)

Job 6160389 failed during process-group initialization, before model loading.
NCCL selected `ib0` and peers could not reach rank 0 at `10.0.1.172:33939`
(`No route to host`). Node hostname `kn172` resolves to `10.1.1.172`.
The container also lacked usable `libibverbs`, causing fallback to socket transport.
The logs establish a failed socket path, but do not establish whether routing,
firewall policy, or fabric availability caused the failure.

Scalable workers now select the IPv4 interface whose address matches the kernel's
route source for `MASTER_ADDR`, before initializing NCCL. This runs inside each
worker, so it does not depend on a cluster name, an interface named `eth0`, or the
login node's network layout. An explicit `NCCL_SOCKET_IFNAME` takes precedence.
Missing, ambiguous, or loopback route mappings fail with an actionable error.
The `nccl_network` trace records the selected interface and addresses.

This is Linux/IPv4 discovery. It does not prove peer reachability, install RDMA
libraries, select an RDMA HCA, or guarantee optimal network performance. IPv6 or
special multi-rail deployments should provide an explicit interface override.
No `NCCL_IB_DISABLE` workaround is applied.

Verification: focused route/override/failure tests, the full video suite, and a real
login-node route check passed. Job 6166490 completed successfully on two H100
nodes, one worker with two GPUs per node: metadata gathering, tensor reductions,
and balanced/zero-retention optimizer updates passed. Job 6166321 (the superseded
hardcoded eth0 test) was cancelled.

The full eight-worker, four-node training recovery job 6166613 subsequently
completed updates 6 and 7 successfully, with consistent adapter states and a
saved checkpoint. This verifies the tested communication and short training
workload; full-epoch stability remains unverified.
