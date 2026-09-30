"""
WSM checkpoint merge — the "decay" of a constant-LR run, done post-hoc (paper 2507.17634).

Instead of decaying the LR online, the trainer keeps it CONSTANT (--lr_schedule constant) and dumps
weights-only snapshots during the burst (--wsm_every_tok -> <out>/wsm/snap_*). This script AVERAGES
the last N of those snapshots. By Theorem 3.1 of the paper a uniform average of consecutive checkpoints
is algebraically a LINEAR LR-decay over that window's gradients — so the merge IS the cooldown, and we
get to pick the window POST-HOC (10B/20B/30B) and select on COBOLEval, not on loss.

  # merge the last ~20B-token window, uniform mean (the default / near-optimal):
  python training/bin.merge_wsm.py --snapshots_dir checkpoints/run4b/wsm --window_tok 20e9 \
         --method mean --out checkpoints/run4b/merged_mean_20B

  # explicit list, inverse-sqrt weighting (a post-hoc variant to try against mean on COBOLEval):
  python training/bin.merge_wsm.py --ckpts a,b,c,d --method 1sqrt --out /tmp/merged

  # verify it reloads + runs a forward after saving:
  python training/bin.merge_wsm.py --snapshots_dir .../wsm --last_n 10 --out .../merged --verify

Rules baked in (WSD-WSM schedule, paper 2507.17634):
- Default method = MEAN (uniform). Near-optimal, robust; 1-sqrt is a slight-edge variant to try post-hoc.
- NEVER EMA / convex weighting (all weight on the last ckpt) — it emulates an EXPONENTIAL decay, the
  paper's WORST merge: it crushes the older checkpoints that carry the general-code signal.
- WSM degrades with too few checkpoints: >=8-10 in the window, 2 collapses. We WARN below 8, never silently.
- Non-float tensors (rare persistent int buffers) are NOT averaged — kept from the last ckpt (identical across).
"""
import argparse, glob, json, math, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from models.config import Skylar2Config
from models.decoder import Skylar2ForCausalLM

ROOT = Path(__file__).resolve().parent.parent


# ── load a HF save_pretrained() state_dict, handling single-file AND sharded, safetensors AND .bin ──
def load_state_dict_any(d):
    d = Path(d)
    idx_st = d / "model.safetensors.index.json"
    one_st = d / "model.safetensors"
    idx_bin = d / "pytorch_model.bin.index.json"
    one_bin = d / "pytorch_model.bin"
    if idx_st.exists():
        from safetensors.torch import load_file
        wm = json.loads(idx_st.read_text())["weight_map"]
        sd = {}
        for shard in sorted(set(wm.values())):
            sd.update(load_file(str(d / shard)))
        return sd
    if one_st.exists():
        from safetensors.torch import load_file
        return load_file(str(one_st))
    if idx_bin.exists():
        wm = json.loads(idx_bin.read_text())["weight_map"]
        sd = {}
        for shard in sorted(set(wm.values())):
            sd.update(torch.load(d / shard, map_location="cpu", weights_only=True))
        return sd
    if one_bin.exists():
        return torch.load(one_bin, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no model weights (safetensors/bin) in {d}")


def snapshot_meta(d):
    """(step, tokens) from wsm_snapshot.json if present, else (None, None)."""
    f = Path(d) / "wsm_snapshot.json"
    if f.exists():
        m = json.loads(f.read_text())
        return m.get("step"), m.get("tokens")
    return None, None


def collect_snapshots(snapshots_dir):
    """All subdirs that look like a saved model, ordered oldest->newest by step (fallback: name)."""
    dirs = [Path(p).parent for p in glob.glob(str(Path(snapshots_dir) / "*" / "config.json"))]
    dirs = [d for d in dirs if not d.name.endswith(".tmp")]
    def key(d):
        step, tok = snapshot_meta(d)
        return (0, step) if step is not None else (1, d.name)
    return sorted(dirs, key=key)


def select_window(dirs, window_tok=None, last_n=None):
    """Pick the tail of the (ordered) snapshot list by token-window or count. dirs is oldest->newest."""
    if window_tok:
        toks = [snapshot_meta(d)[1] for d in dirs]
        if any(t is None for t in toks):
            raise SystemExit("--window_tok needs wsm_snapshot.json (tokens) in every snapshot; "
                             "use --last_n instead for snapshots without it.")
        last = toks[-1]
        return [d for d, t in zip(dirs, toks) if t > last - window_tok]
    if last_n:
        return dirs[-last_n:]
    return dirs


def merge_weights(n, method):
    """Return the n merge coefficients (oldest->newest), summing to 1."""
    if method == "mean":
        return [1.0 / n] * n                       # uniform = LINEAR decay (Thm 3.1), near-optimal
    if method == "1sqrt":
        # inverse-sqrt weighting: more weight on recent ckpts, sub-linearly (between uniform and EMA).
        # Heuristic stand-in for the paper's 1-sqrt merge — SELECT the final model on COBOLEval, not here.
        raw = [math.sqrt(k + 1) for k in range(n)]  # k=0 oldest
        s = sum(raw)
        return [r / s for r in raw]
    raise SystemExit(f"unknown --method {method} (mean|1sqrt; EMA is forbidden by design)")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--snapshots_dir", help="dir of WSM snapshots (<out>/wsm)")
    src.add_argument("--ckpts", help="explicit comma-separated ckpt dirs, oldest->newest")
    ap.add_argument("--window_tok", type=float, default=None, help="keep tail within this token window")
    ap.add_argument("--last_n", type=int, default=None, help="keep the last N snapshots")
    ap.add_argument("--method", default="mean", choices=["mean", "1sqrt"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json to save (else copied from a ckpt)")
    ap.add_argument("--verify", action="store_true", help="reload the merged model + run a forward")
    args = ap.parse_args()

    if args.ckpts:
        dirs = [Path(x) for x in args.ckpts.split(",") if x.strip()]
    else:
        dirs = collect_snapshots(args.snapshots_dir)
    dirs = select_window(dirs, args.window_tok, args.last_n)
    n = len(dirs)
    if n < 2:
        raise SystemExit(f"need >=2 checkpoints to merge, got {n}")
    if n < 8:
        print(f"[warn] merging only {n} ckpts — WSM degrades below ~8-10 (2 collapses). Widen the window.",
              flush=True)

    w = merge_weights(n, args.method)
    toks = [snapshot_meta(d)[1] for d in dirs]
    span = (f"{(toks[0] or 0)/1e9:.1f}..{(toks[-1] or 0)/1e9:.1f}B" if all(t is not None for t in toks) else "n/a")
    print(f"merging {n} ckpts | method={args.method} | token-span={span}", flush=True)
    for d, wi in zip(dirs, w):
        print(f"  w={wi:.4f}  {d.name}", flush=True)

    # incremental fp32 accumulation: peak RAM ~= one fp32 state_dict + one loaded ckpt (not n of them)
    acc, out_dtype = None, {}
    for wi, d in zip(w, dirs):
        sd = load_state_dict_any(d)
        if acc is None:
            acc = {}
            for k, v in sd.items():
                out_dtype[k] = v.dtype
                acc[k] = v.float() * wi if v.is_floating_point() else v.clone()
        else:
            for k, v in sd.items():
                if k not in acc:
                    raise SystemExit(f"key mismatch across ckpts: {k} missing in first ckpt")
                if v.is_floating_point():
                    acc[k].add_(v.float() * wi)
                else:
                    acc[k] = v          # non-float buffer: keep last (identical across ckpts)
        del sd
    merged = {k: (acc[k].to(out_dtype[k]) if acc[k].is_floating_point() else acc[k]) for k in acc}

    cfg = Skylar2Config.from_pretrained(dirs[0])
    model = Skylar2ForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(merged, strict=False)
    if missing or unexpected:
        # tied lm_head (weight-tied to token_emb for <8b) legitimately isn't in the state_dict — tolerate it,
        # but surface anything else so a real key mismatch can't pass silently.
        benign = all("lm_head" in m for m in missing) and not unexpected
        print(f"[load] missing={list(missing)} unexpected={list(unexpected)}"
              f"{'  (tied lm_head — ok)' if benign else '  [!] unexpected mismatch'}", flush=True)
        if not benign:
            raise SystemExit("state_dict mismatch beyond tied lm_head — refusing to save a wrong merge")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    # tokenizer: explicit path wins, else copy from the first snapshot that has one
    tok_src = args.tokenizer
    if not tok_src:
        for d in dirs:
            if (d / "tokenizer.json").exists():
                tok_src = str(d / "tokenizer.json")
                break
    if tok_src and Path(tok_src).exists():
        import shutil
        shutil.copyfile(tok_src, out / "tokenizer.json")
    (out / "merge_manifest.json").write_text(json.dumps({
        "method": args.method, "n_ckpts": n, "weights": w,
        "sources": [d.name for d in dirs], "token_span": span,
    }, indent=2))
    print(f"saved merged model -> {out}", flush=True)

    if args.verify:
        del model, acc, merged
        m2 = Skylar2ForCausalLM.from_pretrained(str(out))
        # I layer KDA (v2) girano solo su CUDA: kernel Triton, nessun fallback CPU.
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m2.to(dev).eval()
        with torch.no_grad():
            x = torch.randint(0, cfg.vocab_size, (1, 16), device=dev)
            out_dict = m2(input_ids=x)
            logits = out_dict["logits"] if isinstance(out_dict, dict) else out_dict.logits
        assert logits.shape == (1, 16, cfg.vocab_size), logits.shape
        assert torch.isfinite(logits).all(), "non-finite logits in merged model"
        print(f"[verify] reload OK, forward OK, logits {tuple(logits.shape)} finite", flush=True)


if __name__ == "__main__":
    main()
