"""Minimal NCCL smoke test launched via torch.distributed.run.

Each rank:
  1) init_process_group("nccl")
  2) allocate one tensor on its own GPU
  3) all_reduce it
  4) print result
  5) destroy_process_group

If this dies with "unhandled cuda error", the issue is in NCCL bring-up
(P2P / fabricmanager / topology), not in train.py / model / batch_size.

Launched by scripts/diagnose_nccl.sh; not for direct use.
"""

import os
import socket
import sys
import time

import torch
import torch.distributed as dist


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    host = socket.gethostname()

    torch.cuda.set_device(local_rank)
    print(
        f"[rank {rank}/{world} host={host} local_rank={local_rank}] "
        f"cuda={torch.cuda.is_available()} dev={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(local_rank)}",
        flush=True,
    )

    t0 = time.perf_counter()
    print(f"[rank {rank}] init_process_group('nccl') ...", flush=True)
    dist.init_process_group(backend="nccl")
    t1 = time.perf_counter()
    print(f"[rank {rank}] init OK in {t1 - t0:.2f}s", flush=True)

    tensor = torch.full((4,), float(rank + 1), device=f"cuda:{local_rank}")
    print(f"[rank {rank}] before all_reduce: {tensor.tolist()}", flush=True)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    print(f"[rank {rank}] after  all_reduce: {tensor.tolist()}  (expected={sum(range(1, world + 1))})", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
