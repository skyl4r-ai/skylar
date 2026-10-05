"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Architecture ablation on a proxy model — the protocol of docs/PAPER_V2.md §5.

Trains one proxy per (variant, seed) with `training/bin.pretrain.py`, then evaluates every run on the
SAME fixed validation windows, so that differences between variants can be paired by seed. Runs that
already have a result.json are skipped: the ablation resumes where it stopped.

    python eval/bin.arch_ablation.py --data <tokenized_dir> --out runs/ablation \\
        --variants noar,block,block_gn --seeds 1234,777,2024 --lr 1.2e-3
    python eval/bin.arch_ablation.py --out runs/ablation --report

Variants (all on the Skylar 2 hybrid: KDA 3:1, per-head output gate, SiTU-GLU, document masking):
    noar       no AttnRes
    window     AttnRes over the 13 most recent sub-layer outputs (evaluated, not adopted)
    block      block AttnRes, blocks of 8 sub-layers
    block_gn   block AttnRes + gated normalisation (rank 16)
    muon:<v>   variant <v> trained with Muon instead of AdamW
Default proxy: 36 layers, width 512, 4/2 heads of 128, FFN 1408 (the depth and layout of the 990M).
Optional: --bpb_set <dir of .txt slices> adds bits-per-byte (eval/bin.bits_per_byte.py).
"""
import argparse
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

VARIANTS = {
    "noar": [],
    "window": ["--attn_res", "--attn_res_mode", "window", "--attn_res_block", "6"],
    "block": ["--attn_res", "--attn_res_mode", "block", "--attn_res_block_size", "8"],
    "block_gn": ["--attn_res", "--attn_res_mode", "block", "--attn_res_block_size", "8",
                 "--gated_norm", "16"],
}


def variant_flags(name):
    if name.startswith("muon:"):
        return VARIANTS[name[5:]] + ["--optimizer", "muon"]
    return VARIANTS[name]


def train(args, variant, seed, out):
    steps = int(args.tokens // (args.seq_len * args.batch_size * args.grad_accum))
    cmd = [sys.executable, str(REPO / "training/bin.pretrain.py"),
           "--preset", "medium", "--d_model", str(args.d_model), "--n_layers", str(args.n_layers),
           "--n_heads", str(args.n_heads), "--n_kv_heads", str(args.n_kv_heads), "--d_ff", str(args.d_ff),
           "--data", args.data, "--tokenizer", args.tokenizer or os.path.join(args.data, "tokenizer.json"),
           "--seq_len", str(args.seq_len), "--batch_size", str(args.batch_size),
           "--grad_accum", str(args.grad_accum), "--steps", str(steps), "--warmup", str(max(1, steps // 10)),
           "--lr", str(args.lr), "--lr_schedule", "wsd", "--lr_decay_ratio", "0.2", "--min_lr_ratio", "0.1",
           "--kda_ratio", "3:1", "--attn_out_gate", "perhead", "--hidden_act", args.hidden_act,
           "--doc_masking", "--dropout", "0", "--compile", "--grad_ckpt", "--eval_every", "150",
           "--log_every", "1", "--no_checkpoints", "--seed", str(seed), "--out", str(out),
           "--sampler", "random",          # the sampler of the report's runs (random windows)
           *variant_flags(variant)]
    env = {**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    with open(out / "train.log", "w") as log:
        return subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def evaluate(args, ckpt):
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO))
    from data.memmap_dataset import MemmapTokenDataset
    from models.decoder import Skylar2ForCausalLM
    from tokenizers import Tokenizer

    model = Skylar2ForCausalLM.from_pretrained(str(ckpt)).cuda().eval()
    tok = Tokenizer.from_file(args.tokenizer or os.path.join(args.data, "tokenizer.json"))
    bos = tok.token_to_id("<bos>")
    ds = MemmapTokenDataset(args.data, seq_len=args.seq_len, seed=0)
    tot = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(args.val_batches):            # the SAME windows for every run
            x, y, doc = ds.get_batch("val", 4, "cuda", rng=np.random.default_rng(10_000 + i),
                                     return_doc_ids=True, bos_id=bos)
            tot += model(input_ids=x, labels=y, document_ids=doc)["loss"].item()
    res = {"val_ce_fixed": tot / args.val_batches, "params": sum(p.numel() for p in model.parameters())}
    if args.bpb_set:
        spec = importlib.util.spec_from_file_location("bpb", REPO / "eval/bin.bits_per_byte.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        bpb = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for path in sorted(Path(args.bpb_set).glob("*.txt")):
                r = mod.bits_per_byte(model, tok, path.read_text(), "cuda", seq_len=args.seq_len)
                if r:
                    bpb[path.stem] = r["bpb"]
        res["bpb"] = bpb
        res["bpb_mean"] = sum(bpb.values()) / len(bpb) if bpb else None
    return res


def report(out_dir):
    rows = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("*/result.json"))]
    by = {}
    for r in rows:
        if "val_ce_fixed" in r:
            by.setdefault(r["variant"], {})[r["seed"]] = r
    print("\n| variant | n | val. CE (mean ± sd) | tokens/s |\n|---|---:|---:|---:|")
    for v, d in by.items():
        ce = [x["val_ce_fixed"] for x in d.values()]
        sd = statistics.stdev(ce) if len(ce) > 1 else float("nan")
        print(f"| {v} | {len(ce)} | {statistics.mean(ce):.4f} ± {sd:.4f} | "
              f"{statistics.mean(x.get('tok_s', 0) for x in d.values()):,.0f} |")
    names = list(by)
    print()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(by[a]) & set(by[b]))
            if len(common) < 2:
                continue
            diff = [by[a][s]["val_ce_fixed"] - by[b][s]["val_ce_fixed"] for s in common]
            mu, sd = statistics.mean(diff), statistics.stdev(diff)
            t = mu / (sd / math.sqrt(len(diff))) if sd > 0 else float("inf")
            print(f"{a} - {b}: {mu:+.4f} (paired t = {t:.2f}, {len(diff) - 1} dof, "
                  f"{sum(x < 0 for x in diff)}/{len(diff)} seeds favour {a})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="tokenized dir: tokenizer.json, pretokenized_meta.json (with val_shards), shards/")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="noar,block,block_gn")
    ap.add_argument("--seeds", default="1234,777,2024")
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--tokens", type=float, default=30e6)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=36)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_kv_heads", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=1408)
    ap.add_argument("--hidden_act", default="situ_glu", choices=["swiglu", "situ_glu"])
    ap.add_argument("--val_batches", type=int, default=128, help="x4 windows of seq_len")
    ap.add_argument("--bpb_set", default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(args.out)
        return
    if not args.data:
        ap.error("--data is required unless --report")
    for seed in (int(s) for s in args.seeds.split(",")):
        for variant in args.variants.split(","):
            out = Path(args.out) / f"{variant.replace(':', '_')}_s{seed}"
            if (out / "result.json").exists():
                continue
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True)
            rc = train(args, variant, seed, out)
            res = {"variant": variant, "seed": seed, "lr": args.lr, "rc": rc}
            metrics = [json.loads(l) for l in open(out / "metrics.jsonl")] if (out / "metrics.jsonl").exists() else []
            if metrics:
                res["tok_s"] = round(metrics[-1]["tok_s"])
            if rc == 0 and (out / "final").exists():
                res.update(evaluate(args, out / "final"))
                shutil.rmtree(out / "final", ignore_errors=True)
            (out / "result.json").write_text(json.dumps(res, indent=1))
            print(f"{variant} seed {seed}: {res.get('val_ce_fixed')}", flush=True)
    report(args.out)


if __name__ == "__main__":
    main()
