"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Grows a trained checkpoint wider (beta): the 990M becomes a ~3.9B that computes the same function, and the pretraining
goes on from it with `training/bin.pretrain.py --init_from <out>`. How and why: models/growth.py.

    python training/bin.upscale.py --model <out>/last --out <grown> [--factor 2] [--noise_snr_db 10] [--seed 0]

Grow from a checkpoint taken at constant learning rate (`last` or a WSM snapshot of the run), not from the merged or
annealed model: restarting from a decayed checkpoint costs loss (arXiv 2403.08763), and growth fails late in a cosine
schedule but holds in the stable phase of WSD (arXiv 2511.04981).

Before saving, the grown model is checked against the source on the same batch (two documents, so the KDA state and
the short conv reset at the boundary): if the logits differ by more than rounding, nothing is written. The result goes
in <out>/growth.json.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.decoder import Skylar2ForCausalLM               # noqa: E402
from models.growth import grow_width, compare_logits       # noqa: E402

TOL = {"cpu": 1e-5, "cuda": 5e-2}     # max|dlogit| / max|logit|: fp32 rounding on the CPU; on the GPU the KDA kernels


def main():
    ap = argparse.ArgumentParser(description="Grow a Skylar checkpoint wider (beta)")
    ap.add_argument("--model", required=True, help="the source checkpoint (a save_pretrained directory)")
    ap.add_argument("--out", required=True, help="where to write the grown model")
    ap.add_argument("--factor", type=int, default=2, help="width multiplier (2: the most reliable, arXiv 2605.13405)")
    ap.add_argument("--noise_snr_db", type=float, default=10.0,
                    help="symmetry-breaking noise, signal to noise in dB (HyperCloning tested 10); 0 = none: the "
                         "copies then stay identical through training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check_tokens", type=int, default=256, help="length of the check batch (0 = no check)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="where to run the check: cpu = fp32, exact to rounding; cuda = the fla kernels")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"--out {out} is not empty")
    t0 = time.time()
    src = Skylar2ForCausalLM.from_pretrained(args.model).eval()
    n_src = sum(p.numel() for p in src.parameters())
    print(f"[upscale] source {args.model}: {n_src / 1e9:.3f}B params, d_model {src.config.d_model}, "
          f"heads {src.config.n_heads}/{src.config.n_kv_heads}, d_ff {src.config.d_ff}", flush=True)
    big = grow_width(src, args.factor, noise_snr_db=args.noise_snr_db or None, seed=args.seed).eval()
    n_big = sum(p.numel() for p in big.parameters())
    c = big.config
    print(f"[upscale] grown {args.factor}x: {n_big / 1e9:.3f}B params, d_model {c.d_model}, heads "
          f"{c.n_heads}/{c.n_kv_heads}, KDA heads {c.kda_heads}, d_ff {c.d_ff} ({time.time() - t0:.0f}s)", flush=True)

    check = None
    if args.check_tokens:
        T, dev = args.check_tokens, torch.device(args.device)
        g = torch.Generator().manual_seed(args.seed)
        ids = torch.randint(16, src.config.vocab_size, (1, T), generator=g)
        doc = torch.zeros(1, T, dtype=torch.long)
        doc[:, T // 2:] = 1
        dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
        a, b = src.to(dev, torch.float32), big.to(dev, torch.float32)
        with torch.autocast("cuda", dtype=dtype, enabled=args.device == "cuda"):
            check = compare_logits(a, b, ids.to(dev), document_ids=doc.to(dev))
        check["device"] = args.device
        ok = check["max_rel"] < TOL[args.device]
        print(f"[upscale] check on {T} tokens ({args.device}): max|dlogit|/max|logit| {check['max_rel']:.1e}, "
              f"loss {check['loss_source']:.6f} -> {check['loss_grown']:.6f}: {'OK' if ok else 'FAILED'}", flush=True)
        if not ok:
            sys.exit("[upscale] the grown model does not compute the source's function: nothing written")
        big = big.cpu()

    out.mkdir(parents=True, exist_ok=True)
    big.save_pretrained(out)          # fp32, like the checkpoints the trainer writes
    tok = Path(args.model) / "tokenizer.json"
    if tok.exists():
        shutil.copy2(tok, out / "tokenizer.json")
    (out / "growth.json").write_text(json.dumps({
        "source": str(Path(args.model).resolve()), "factor": args.factor, "noise_snr_db": args.noise_snr_db,
        "seed": args.seed, "params_source": n_src, "params_grown": n_big, "check": check,
        "date": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print(f"[upscale] written {out} ({time.time() - t0:.0f}s). Next: training/bin.pretrain.py --init_from {out} "
          f"(same data; a learning rate no higher than a run from scratch of this size, with a warmup: on our proxy the "
          f"grown model did best at a lower one)", flush=True)


if __name__ == "__main__":
    main()
