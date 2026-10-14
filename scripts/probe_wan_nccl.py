#!/usr/bin/env python3
"""Check isolated Wan worker GPU placement and NCCL without loading models."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import traceback

import torch
import torch.distributed as dist

from diffusionopsd.video.nccl_network import configure_socket_interface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-nodes", type=int, default=1)
    parser.add_argument("--gpus-per-worker", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"rank": rank, "status": "FAILED"}
    try:
        result["network"] = configure_socket_interface(os.environ["MASTER_ADDR"])
        print(json.dumps({"event": "network", **result["network"]}), flush=True)
        count = torch.cuda.device_count()
        if world != 2 or count != args.gpus_per_worker:
            raise ValueError(f"Expected two workers with {args.gpus_per_worker} GPUs each; got {world=}, {count=}")
        torch.cuda.set_device(0)
        placement = {
            "rank": rank,
            "local_rank": int(os.environ["SLURM_LOCALID"]),
            "host": socket.gethostname(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_count": count,
            "gpu_uuids": [str(torch.cuda.get_device_properties(d).uuid) for d in range(count)],
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "nccl_cumem_host_enable": os.environ.get("NCCL_CUMEM_HOST_ENABLE"),
        }
        result["placement"] = placement
        print(json.dumps({"event": "placement", **placement}), flush=True)
        for device in range(count):
            value = torch.full((8,), float(rank + 1), device=f"cuda:{device}")
            if not torch.all(value == rank + 1).item():
                raise RuntimeError(f"Local GPU {device} failed")
            torch.cuda.synchronize(device)
        dist.init_process_group(
            "nccl", rank=rank, world_size=world, timeout=timedelta(seconds=60), device_id=torch.device("cuda:0")
        )
        placements = [None] * world
        dist.all_gather_object(placements, placement)
        uuids = [uuid for item in placements for uuid in item["gpu_uuids"]]
        if len({p["host"] for p in placements}) != args.expected_nodes or len(set(uuids)) != world * count:
            raise RuntimeError(f"Workers must use distinct GPUs on the expected number of nodes: {placements}")
        for size in (8, 1_000_000):
            value = torch.full((size,), float(rank + 1), device="cuda:0")
            dist.all_reduce(value)
            if not torch.all(value == 3).item():
                raise RuntimeError("Tensor reduction disagrees with serial sum")
        # Both ranks retain a sample, then only rank 0 retains two samples.
        cases = []
        for retained in ((1, 1), (2, 0)):
            parameter = torch.nn.Parameter(torch.zeros(1, device="cuda:0"))
            optimizer = torch.optim.SGD([parameter], lr=0.1)
            targets = [rank + 1] if retained == (1, 1) else ([1, 2] if rank == 0 else [])
            parameter.grad = torch.zeros_like(parameter)
            for target in targets:
                ((parameter - target) ** 2).sum().backward()
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(sum(retained))
            if not torch.allclose(parameter.grad, torch.tensor([-3.0], device="cuda:0")):
                raise RuntimeError("Retained-count-normalized gradient disagrees with serial reference")
            optimizer.step()
            weights = [torch.zeros_like(parameter) for _ in range(world)]
            dist.all_gather(weights, parameter.detach())
            if not all(torch.allclose(weight, torch.tensor([0.3], device="cuda:0")) for weight in weights):
                raise RuntimeError("Optimizer states disagree across workers")
            cases.append({"retained": retained, "gradient": parameter.grad.item(), "parameter": parameter.item()})
        dist.barrier(device_ids=[0])
        dist.destroy_process_group()
        result.update(status="PASSED", placements=placements, gradient_cases=cases)
        print(json.dumps({"event": "probe_passed", **result}), flush=True)
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        (args.output_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
