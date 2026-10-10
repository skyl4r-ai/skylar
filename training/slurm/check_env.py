# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Is this machine able to train Skylar? The first thing to run on a new cluster or pod, on one GPU node:

    python training/slurm/check_env.py [--json out.json]

Checks, each PASS/FAIL with the reason:
  1. versions: torch, CUDA runtime, driver, triton, fla, accelerate, transformers (against constraints.txt);
  2. the GPUs: how many, model, memory, compute capability;
  3. the driver can run this torch build: a CUDA op (the driver older than the wheel's CUDA is the n.1 risk);
  4. Triton compiles and runs a kernel on the GPU (it needs ptxas and a C compiler for its launcher);
  5. flash-linear-attention: a KDA chunk forward + backward on the GPU (the kernels Skylar 2 trains with);
  6. NCCL is available to torch.distributed;
  7. RDMA for more than one node: InfiniBand/RoCE devices and libibverbs. Without the library NCCL falls back to TCP
     sockets without an error (on 30/09/2026 a 2-node cluster ran at 50% for this). Required with --multinode,
     reported otherwise.
Then the architecture gates: python eval/bin.gate_arch_v2.py (18 on a GPU).
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append({"check": name, "ok": True, "detail": detail})
        print(f"  [PASS] {name:<28} {detail}")
    except Exception as e:
        results.append({"check": name, "ok": False, "detail": f"{type(e).__name__}: {e}"})
        print(f"  [FAIL] {name:<28} {type(e).__name__}: {str(e)[:300]}")


def pinned():
    pins = {}
    f = ROOT / "constraints.txt"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.split("#")[0].strip()
            if "==" in line:
                k, v = line.split("==", 1)
                pins[k.strip().lower()] = v.strip()
    return pins


def versions():
    import importlib.metadata as md
    pins, out, bad = pinned(), [], []
    for pkg in ["torch", "triton", "flash-linear-attention", "fla-core", "accelerate", "transformers", "tokenizers"]:
        try:
            v = md.version(pkg)
        except md.PackageNotFoundError:
            v = None
        want = pins.get(pkg)
        if v is None or (want and not v.startswith(want)):
            bad.append(f"{pkg} {v} (want {want})")
        out.append(f"{pkg}={v}")
    if bad:
        raise RuntimeError("; ".join(bad))
    return " ".join(out)


def gpus():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    n = torch.cuda.device_count()
    p = torch.cuda.get_device_properties(0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.split("\n")[0].strip()
    return (f"{n} x {p.name}, {p.total_memory / 2**30:.0f} GiB, sm_{p.major}{p.minor}; driver {smi}; "
            f"torch built for CUDA {torch.version.cuda}")


def cuda_op():
    import torch
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    b = (a @ a).float().sum().item()
    return f"bf16 matmul ok ({b:.1f})"


def triton_kernel():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(o_ptr + i, tl.load(x_ptr + i, mask=m) + tl.load(y_ptr + i, mask=m), mask=m)

    x = torch.randn(10000, device="cuda")
    y = torch.randn(10000, device="cuda")
    o = torch.empty_like(x)
    add[(triton.cdiv(10000, 1024),)](x, y, o, 10000, BLOCK=1024)
    torch.cuda.synchronize()
    err = (o - (x + y)).abs().max().item()
    if err > 1e-6:
        raise RuntimeError(f"wrong result, max err {err}")
    ptxas = os.environ.get("TRITON_PTXAS_PATH") or "bundled"
    return f"triton {triton.__version__} kernel ok (ptxas {ptxas}, cc {shutil.which('gcc') or shutil.which('cc')})"


def kda_kernel():
    """The real Skylar 2 path: a small hybrid model (KDA 3:1 + AttnRes + GatedNorm) with document masking,
    forward + backward on the GPU through the fla kernels, the calls the training makes."""
    import torch
    sys.path.insert(0, str(ROOT))
    from models.config import get_config
    from models.decoder import Skylar2ForCausalLM
    cfg = get_config("test", vocab_size=1024, kda_ratio="3:1", attn_res=True, attn_res_mode="block",
                     attn_res_block_size=2, gated_norm=16, attn_out_gate="perhead", hidden_act="situ_glu")
    m = Skylar2ForCausalLM(cfg).cuda()
    x = torch.randint(3, 1024, (2, 512), device="cuda")
    x[:, 200] = 1
    doc = (x == 1).cumsum(1).to(torch.int32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(input_ids=x, labels=torch.roll(x, -1, 1), document_ids=doc, loss_only=True)["loss"]
    loss.backward()
    torch.cuda.synchronize()
    bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    if not torch.isfinite(loss) or bad:
        raise RuntimeError(f"non-finite loss or gradients ({bad[:3]})")
    n_kda = sum(1 for mod in m.modules() if type(mod).__name__ == "SkylarKDA")
    if n_kda == 0:
        raise RuntimeError("the model has no SkylarKDA layer: the kernels were not exercised")
    return f"hybrid fwd+bwd ok, loss {loss.item():.3f}, {n_kda} KDA modules"


def nccl():
    import torch.distributed as dist
    if not dist.is_nccl_available():
        raise RuntimeError("torch.distributed has no NCCL")
    import torch
    return f"NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}"


def rdma():
    import ctypes
    sysfs = Path("/sys/class/infiniband")
    devs = sorted(p.name for p in sysfs.iterdir()) if sysfs.is_dir() else []
    try:
        ctypes.CDLL("libibverbs.so.1")
        lib = True
    except OSError:
        lib = False
    if devs and lib:
        return f"{len(devs)} RDMA device(s) ({', '.join(devs[:6])}), libibverbs found: NCCL can use NET/IB"
    if devs:
        raise RuntimeError(f"RDMA devices {', '.join(devs[:6])} but no libibverbs.so.1: NCCL will use TCP. "
                           f"Install rdma-core / libibverbs1 + ibverbs-providers (or load the cluster's module)")
    raise RuntimeError("no RDMA device in /sys/class/infiniband: between nodes NCCL will use TCP sockets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--multinode", action="store_true", help="RDMA is required (a run on more than one node)")
    a = ap.parse_args()
    print(f"host {platform.node()}, python {sys.version.split()[0]}, {platform.platform()}")
    for name, fn in [("versions", versions), ("gpus", gpus), ("cuda op (driver vs wheel)", cuda_op),
                     ("triton kernel", triton_kernel), ("fla KDA kernel", kda_kernel), ("nccl", nccl)]:
        check(name, fn)
    if a.multinode:
        check("rdma (multi-node)", rdma)
    else:
        try:
            print(f"  [INFO] {'rdma (multi-node)':<28} {rdma()}")
        except Exception as e:
            print(f"  [INFO] {'rdma (multi-node)':<28} {e} (not needed on one node)")
    ok = all(r["ok"] for r in results)
    print("ENV OK" if ok else "ENV NOT OK: fix the FAIL lines first")
    if a.json:
        Path(a.json).write_text(json.dumps({"host": platform.node(), "ok": ok, "checks": results}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
