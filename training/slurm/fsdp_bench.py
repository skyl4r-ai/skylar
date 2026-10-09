# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Memory and throughput of the 4B / 8B Skylar 2 with sharded data parallelism (FSDP2), the measurement a
multi-node GPU-hour budget is built on. Not a trainer: random tokens, a few steps, numbers out.

    torchrun --nnodes N --nproc-per-node 4 ... training/slurm/fsdp_bench.py --preset 4b --seq_len 8192 \\
        [--hsdp] [--grad_ckpt] [--muon gather] [--compile] --out results/fsdp_4b_N.json

- FSDP2 (`fully_shard`) on every transformer block and on the whole model, bf16 parameters in compute and
  fp32 gradient reduction (MixedPrecisionPolicy). `--hsdp`: shard inside the node, replicate across nodes
  (2-D mesh): the all-gathers stay on NVLink and only the gradient reduction crosses InfiniBand.
- The model is built on CPU with its real initialisation (KDA's A_log/dt_bias, GatedNorm...), identical on
  every rank (same seed); fully_shard moves each rank's shard to its GPU.
- `--muon gather`: Muon on sharded matrices as in Moonlight (arXiv 2502.16982, distributed Muon): momentum on
  the local shard, the full matrix gathered for Newton-Schulz, the local rows of the update applied. The rest
  of the parameters on AdamW. `--muon none`: AdamW everywhere (the baseline of the optimizer cost).
Prints and writes: step time, tokens/s per GPU, peak memory per GPU, and the share of the step spent in the
optimizer.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor

from models.config import get_config
from models.decoder import Skylar2ForCausalLM
from models.layers.attn_res import DepthState
from training.optim import default_decay, is_muon_matrix

V2 = dict(kda_ratio="3:1", attn_res=True, attn_res_mode="block", attn_res_block_size=8, gated_norm=16,
          attn_out_gate="perhead", hidden_act="situ_glu", dropout=0.0)


def newton_schulz(g, steps=5, eps=1e-7):
    """Muon's quintic Newton-Schulz iteration (the coefficients of torch.optim.Muon / Keller Jordan)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return x.T if transposed else x


class CheckpointedBlock(torch.nn.Module):
    """The model's AttnRes-safe activation checkpoint (models/decoder.py), moved INSIDE the FSDP unit.

    With FSDP2 the checkpoint must not wrap the sharded block: its pre-forward hook (the all-gather of the
    parameters) would then sit inside the checkpointed region, and in the backward the recompute finds the
    parameters already gathered, records one op less and the saved tensors shift by one (CheckpointError on
    2 A100, 09/10/2026; invisible on 1 GPU, where the all-gather is a no-op). Here FSDP wraps this module, and
    this module checkpoints the block, on a snapshot of the depth state as the model does."""

    def __init__(self, block, mode, block_size):
        super().__init__()
        self.block, self.mode, self.block_size = block, mode, block_size

    def forward(self, x, kv_cache=None, block_mask=None, attention_mask=None, use_cache=False, cu_seqlens=None,
                depth=None):
        def _run(inp, _f=depth.freeze() if depth is not None else None):
            st = None if _f is None else DepthState.thaw(_f, self.mode, self.block_size)
            out, cache = self.block(inp, kv_cache=None, block_mask=block_mask, attention_mask=attention_mask,
                                    use_cache=False, cu_seqlens=cu_seqlens, depth=st)
            return (out, cache) if st is None else (out, cache, *st.new_tensors(len(_f[0])))
        res = torch.utils.checkpoint.checkpoint(_run, x, use_reentrant=False)
        if depth is not None:
            depth.adopt(list(res[2:]), pushes=2)
        return res[0], res[1]


class GatherMuon:
    """Muon for FSDP2-sharded matrices (Moonlight 2502.16982): Nesterov momentum on the local shard, all-gather
    of the momentum-corrected gradient, Newton-Schulz on the full matrix, local rows of the update applied,
    update RMS matched to AdamW (0.2 * sqrt(max(m, n)), as torch.optim.Muon's match_rms_adamw)."""

    def __init__(self, params, lr, weight_decay, momentum=0.95, ns_steps=5):
        self.params = list(params)
        self.lr, self.wd, self.mom, self.ns = lr, weight_decay, momentum, ns_steps
        self.buf = {}

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            b = self.buf.get(p)
            if b is None:
                b = self.buf[p] = torch.zeros_like(g)
            b.mul_(self.mom).add_(g)
            u = g.add(b, alpha=self.mom)                       # Nesterov
            full = u.full_tensor() if isinstance(u, DTensor) else u
            o = newton_schulz(full, self.ns)
            o = o * (0.2 * math.sqrt(max(full.shape)))
            if isinstance(p, DTensor):
                o = distribute_tensor(o.to(p.dtype), p.device_mesh, p.placements, src_data_rank=None)
            p.mul_(1 - self.lr * self.wd)
            p.add_(o, alpha=-self.lr)

    def zero_grad(self, set_to_none=True):
        pass                                                  # the grads are freed with the model's zero_grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="4b")
    ap.add_argument("--n_layers", type=int, default=None, help="override (a quick test on a small GPU)")
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--vocab", type=int, default=64000)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--hsdp", action="store_true", help="shard inside the node, replicate across nodes")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile every block")
    ap.add_argument("--muon", default="gather", choices=["gather", "none"])
    ap.add_argument("--reshard_after_forward", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="nccl", help="gloo only to test several ranks on one GPU")
    a = ap.parse_args()

    dist.init_process_group(a.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    gpn = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    nodes = max(1, world // gpn)
    if a.hsdp and nodes > 1:
        mesh = init_device_mesh("cuda", (nodes, gpn), mesh_dim_names=("replicate", "shard"))
    else:
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("shard",))

    torch.manual_seed(0)
    over = dict(V2)
    if a.n_layers:
        over["n_layers"] = a.n_layers
    cfg = get_config(a.preset, vocab_size=a.vocab, **over)
    for attr in ("max_seq_len", "max_position_embeddings"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, max(a.seq_len, getattr(cfg, attr, 0) or 0))
    t_build = time.time()
    model = Skylar2ForCausalLM(cfg)                             # on CPU, real init, same on every rank
    n_params = sum(p.numel() for p in model.parameters())
    # Order inside each FSDP unit (the torchtitan order): the AttnRes-safe checkpoint around the block, then
    # torch.compile around that, then fully_shard outside (see CheckpointedBlock). Compiling the block INSIDE the
    # checkpoint fails on 2+ ranks: the recompute meets freshly gathered parameters, recompiles another graph and
    # the saved tensors no longer match (CheckpointError, 2 A100, 09/10/2026; eager is fine).
    # The model's own gradient_checkpointing stays off.
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for i, blk in enumerate(model.blocks):
        if a.grad_ckpt:
            blk = CheckpointedBlock(blk, getattr(cfg, "attn_res_mode", "block"), getattr(cfg, "attn_res_block_size", 8))
        if a.compile:
            blk = torch.compile(blk, dynamic=False)
        model.blocks[i] = blk
        fully_shard(blk, mesh=mesh, mp_policy=mp, reshard_after_forward=bool(a.reshard_after_forward))
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=bool(a.reshard_after_forward))
    model.cuda()
    t_build = time.time() - t_build

    muon_p, decay_p, plain_p = [], [], []
    for n, p in model.named_parameters():
        if a.muon == "gather" and is_muon_matrix(n, p):
            muon_p.append(p)
        elif default_decay(n, p):
            decay_p.append(p)
        else:
            plain_p.append(p)
    adamw = torch.optim.AdamW([{"params": decay_p, "weight_decay": 0.1}, {"params": plain_p, "weight_decay": 0.0}],
                              lr=1e-4, betas=(0.9, 0.95))
    muon = GatherMuon(muon_p, lr=1e-4, weight_decay=0.1) if muon_p else None

    g = torch.Generator().manual_seed(1 + rank)
    x = torch.randint(3, a.vocab, (a.batch_size, a.seq_len), generator=g)
    for c in torch.randint(1, a.seq_len, (4,), generator=g).tolist():
        x[:, c] = 1
    y = torch.roll(x, -1, 1)
    doc = (x == 1).cumsum(1).to(torch.int32)
    x, y, doc = x.cuda(), y.cuda(), doc.cuda()

    torch.cuda.reset_peak_memory_stats()
    times, opt_times = [], []
    for i in range(a.warmup + a.steps):
        torch.cuda.synchronize(); t0 = time.time()
        loss = model(input_ids=x, labels=y, document_ids=doc, loss_only=True)["loss"]
        loss.backward()
        torch.cuda.synchronize(); t1 = time.time()
        adamw.step()
        if muon is not None:
            muon.step()
        adamw.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t2 = time.time()
        if i >= a.warmup:
            times.append(t2 - t0); opt_times.append(t2 - t1)
        if rank == 0:
            print(f"  step {i}: {t2 - t0:.2f}s (optimizer {t2 - t1:.2f}s) loss {loss.item():.3f}", flush=True)
    times.sort(); opt_times.sort()
    step_s = times[len(times) // 2]
    peak = torch.cuda.max_memory_allocated() / 2**30
    peak_t = torch.tensor([peak], device="cuda")
    dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
    tok_gpu = a.batch_size * a.seq_len / step_s
    res = {"preset": a.preset, "n_layers": cfg.n_layers, "params_b": n_params / 1e9, "world": world, "nodes": nodes,
           "mesh": "hsdp" if (a.hsdp and nodes > 1) else "fsdp", "seq_len": a.seq_len, "batch_size": a.batch_size,
           "grad_ckpt": a.grad_ckpt, "compile": a.compile, "muon": a.muon, "step_s": step_s,
           "optimizer_s": opt_times[len(opt_times) // 2], "tok_s_per_gpu": tok_gpu, "tok_s_total": tok_gpu * world,
           "peak_gib_max": float(peak_t), "gpu": torch.cuda.get_device_name(), "build_s": t_build}
    if rank == 0:
        print(json.dumps(res, indent=1))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
