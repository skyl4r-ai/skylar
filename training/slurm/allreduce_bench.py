# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
All-reduce bandwidth between the GPUs of a job: the number that decides multi-node efficiency.

    torchrun --nnodes N --nproc-per-node 4 ... training/slurm/allreduce_bench.py --out results/allreduce_N.json

Same NCCL and the same tensors as training (no MPI build, unlike nccl-tests). For each size it prints the
algorithm bandwidth and the bus bandwidth in the nccl-tests convention (busbw = algbw * 2(n-1)/n, which
is comparable across GPU counts), and the time of one gradient all-reduce of the 990M model (2.05 GB in
bf16, 4.1 GB in fp32). Run it once with NCCL_DEBUG=INFO: between nodes the log has to say NET/IB
(InfiniBand). NET/Socket means TCP, and on 2x4 A100 that halved the training throughput (30/09/2026).
"""
import argparse
import json
import os
import socket
import time

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes_mb", default="16,64,256,1024,2048,4096")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--grad_params", type=float, default=1.027e9, help="model size for the gradient line")
    ap.add_argument("--out", default=None, help="JSON with every measurement (rank 0)")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    esize = torch.finfo(dtype).bits // 8
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    n_nodes = len(set(hosts))

    rows = []
    if rank == 0:
        print(f"all-reduce {a.dtype}: {world} GPUs on {n_nodes} nodes ({', '.join(sorted(set(hosts)))})")
        print(f"{'size MB':>9} {'time ms':>9} {'algbw GB/s':>11} {'busbw GB/s':>11}")
    for mb in [float(x) for x in a.sizes_mb.split(",")]:
        n = int(mb * 2**20) // esize
        x = torch.ones(n, dtype=dtype, device="cuda")
        for _ in range(a.warmup):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        t = time.perf_counter()
        for _ in range(a.iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / a.iters
        t_max = torch.tensor([dt], device="cuda")
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX)     # the slowest rank is the real time
        dt = float(t_max)
        algbw = n * esize / dt / 1e9
        busbw = algbw * 2 * (world - 1) / world if world > 1 else 0.0
        rows.append({"size_mb": mb, "time_ms": dt * 1e3, "algbw_gbs": algbw, "busbw_gbs": busbw})
        if rank == 0:
            print(f"{mb:>9.0f} {dt * 1e3:>9.2f} {algbw:>11.2f} {busbw:>11.2f}")
        del x
        torch.cuda.empty_cache()

    if rank == 0:
        big = rows[-1]
        grad_gb = a.grad_params * esize / 1e9
        t_grad = grad_gb / big["algbw_gbs"] if big["algbw_gbs"] > 0 else None
        print(f"gradient all-reduce of {a.grad_params / 1e9:.2f}B params in {a.dtype} ({grad_gb:.2f} GB): "
              f"~{t_grad * 1e3:.0f} ms at the {big['size_mb']:.0f} MB bandwidth" if t_grad else "")
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w") as f:
                json.dump({"world": world, "nodes": n_nodes, "hosts": sorted(set(hosts)), "dtype": a.dtype,
                           "rows": rows, "grad_allreduce_ms": t_grad * 1e3 if t_grad else None,
                           "nccl_version": ".".join(map(str, torch.cuda.nccl.version())),
                           "t": time.time()}, f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
