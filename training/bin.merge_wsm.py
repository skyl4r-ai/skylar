# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
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

  # explicit list, the 1-sqrt shape (a post-hoc variant to try against mean on COBOLEval):
  python training/bin.merge_wsm.py --ckpts a,b,c,d --method 1sqrt --out /tmp/merged

  # verify it reloads + runs a forward after saving:
  python training/bin.merge_wsm.py --snapshots_dir .../wsm --last_n 10 --out .../merged --verify

Rules baked in (WSD-WSM schedule, paper 2507.17634):
- Default method = MEAN (uniform). Near-optimal, robust; 1-sqrt is a slight-edge variant to try post-hoc
  (WSM Table 3: 64.06 against 63.95 for the mean).
- Every source is checked for non-finite values before it is averaged, and its sha256 goes in the manifest.
- NEVER EMA / convex weighting (all weight on the last ckpt) — it emulates an EXPONENTIAL decay, the
  paper's WORST merge: it crushes the older checkpoints that carry the general-code signal.
- WSM degrades with too few checkpoints: >=8-10 in the window, 2 collapses. We WARN below 8, never silently.
- Non-float tensors (rare persistent int buffers) are NOT averaged: they must be identical across the sources.
- A snapshot marked `terminal` (the last weights of a job that stopped inside a window) is merged only when it is the
  newest: followed by newer snapshots it is a job boundary of the chain and is left out (an explicit --ckpts list is
  taken as given).
- Snapshots are ordered by the TOKENS they were saved at, never by step (a resume with another global batch renumbers
  the steps), and every source must have the same tensors (names, shapes, dtypes), the same config.json and the same
  tokenizer. --start_tok/--end_tok pin the window to a phase explicitly (e.g. the burst: --start_tok 180e9).
"""
import argparse, glob, hashlib, json, math, os, sys
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
    """All subdirs that look like a saved model, ordered oldest->newest by the tokens they were saved at. Until 10/2026
    they were ordered by step: after a resume with a doubled batch a snapshot at 198B tokens (step ~47k) sorted before
    one at 180B (step ~86k), and the window was cut from the wrong end. Without tokens everywhere: by name, with a
    warning (--window_tok and --start_tok/--end_tok then refuse)."""
    dirs = [Path(p).parent for p in glob.glob(str(Path(snapshots_dir) / "*" / "config.json"))]
    dirs = [d for d in dirs if not d.name.endswith(".tmp")]
    toks = [snapshot_meta(d)[1] for d in dirs]
    if dirs and all(t is not None for t in toks):
        return [d for _, d in sorted(zip(toks, dirs))]
    print(f"[warn] {sum(t is None for t in toks)} snapshots without wsm_snapshot.json: ordered by name", flush=True)
    return sorted(dirs, key=lambda d: d.name)


def is_terminal(d):
    """True for the last weights of a job that stopped inside a window (wsm_snapshot.json says "terminal")."""
    try:
        return bool(json.loads((Path(d) / "wsm_snapshot.json").read_text()).get("terminal"))
    except (OSError, ValueError, AttributeError):
        return False


def drop_job_boundaries(dirs):
    """A terminal snapshot (bin.pretrain.py, terminal_snapshot) is the end of the run only when nothing newer follows
    it. One with newer snapshots after it is where a job of the chain stopped (its deadline, a SIGTERM) and the next
    went on: it is left out, so the merge stays on the window's marks. Since 15/10/2026 every planned stop inside a
    window writes one, so that a run the GPU-hour budget ends between two marks keeps its newest weights."""
    keep = [d for i, d in enumerate(dirs) if not (is_terminal(d) and i < len(dirs) - 1)]
    for d in dirs:
        if d not in keep:
            print(f"[merge] {d.name}: the last weights of a job that the chain continued, left out", flush=True)
    return keep


def select_window(dirs, window_tok=None, last_n=None, start_tok=None, end_tok=None):
    """The snapshots of the window, oldest->newest: within [start_tok, end_tok] (a phase, explicit), then the last
    window_tok tokens of what remains (counted from the newest snapshot's tokens) or the last N."""
    if window_tok or start_tok is not None or end_tok is not None:
        toks = [snapshot_meta(d)[1] for d in dirs]
        if any(t is None for t in toks):
            raise SystemExit("--window_tok/--start_tok/--end_tok need wsm_snapshot.json (tokens) in every snapshot; "
                             "use --last_n instead for snapshots without it.")
        keep = [(t, d) for t, d in zip(toks, dirs)
                if (start_tok is None or t >= start_tok) and (end_tok is None or t <= end_tok)]
        if window_tok and keep:
            newest = max(t for t, _ in keep)
            keep = [(t, d) for t, d in keep if t > newest - window_tok]
        dirs = [d for _, d in keep]
    if last_n:
        return dirs[-last_n:]
    return dirs


def check_order(dirs):
    """Tokens strictly increasing along the list (also an explicit --ckpts list): two sources at the same token count
    are the same point twice (or two branches), one older listed after a newer one inverts the 1-sqrt weights."""
    toks = [snapshot_meta(d)[1] for d in dirs]
    if all(t is not None for t in toks):
        for (a, ta), (b, tb) in zip(zip(dirs, toks), zip(dirs[1:], toks[1:])):
            if tb <= ta:
                raise SystemExit(f"{b.name} ({tb:,} tokens) comes after {a.name} ({ta:,} tokens): the sources must be "
                                 f"in increasing token order, each once")
    return toks


def _sha256_file(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def same_model_and_tokenizer(dirs):
    """Every source has the config.json of the first (except the bookkeeping keys) and the same tokenizer.json:
    averaging two architectures or two vocabularies gives a model that loads and means nothing. Returns the hashes."""
    def cfg(d):
        c = json.loads((d / "config.json").read_text())
        for k in ("_name_or_path", "transformers_version", "torch_dtype", "dtype"):
            c.pop(k, None)
        return c
    ref = cfg(dirs[0])
    for d in dirs[1:]:
        c = cfg(d)
        if c != ref:
            diff = sorted(k for k in set(c) | set(ref) if c.get(k) != ref.get(k))
            raise SystemExit(f"{d.name}: config.json differs from {dirs[0].name} on {diff}")
    toks = {d.name: _sha256_file(d / "tokenizer.json") for d in dirs if (d / "tokenizer.json").exists()}
    if len(set(toks.values())) > 1:
        raise SystemExit("the sources carry different tokenizers: " + ", ".join(f"{n} {h[:12]}" for n, h in toks.items()))
    return (hashlib.sha256(json.dumps(ref, sort_keys=True).encode()).hexdigest(),
            next(iter(toks.values()), None))


def merge_weights(n, method):
    """Return the n merge coefficients (oldest->newest), summing to 1."""
    if method == "mean":
        return [1.0 / n] * n                       # uniform = LINEAR decay (Thm 3.1), near-optimal
    if method == "1sqrt":
        # The merge equivalent of the 1-sqrt cooldown (Hägele et al. 2405.18392; WSM 2507.17634 Fig. 2d): the gradient
        # segment i (1 = oldest) weighs sum_{j>=i} c_j, which must follow 1 - sqrt((i-1)/n), so
        # c_j = sqrt(j/n) - sqrt((j-1)/n): the most weight on the OLDEST snapshot, falling towards the newest
        # (n = 10: 0.316 ... 0.051). Until 13/10/2026 this used weights growing with recency, the opposite shape.
        return [math.sqrt(j / n) - math.sqrt((j - 1) / n) for j in range(1, n + 1)]
    raise SystemExit(f"unknown --method {method} (mean|1sqrt; EMA is forbidden by design)")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--snapshots_dir", help="dir of WSM snapshots (<out>/wsm)")
    src.add_argument("--ckpts", help="explicit comma-separated ckpt dirs, oldest->newest")
    ap.add_argument("--window_tok", type=float, default=None, help="keep tail within this token window")
    ap.add_argument("--last_n", type=int, default=None, help="keep the last N snapshots")
    ap.add_argument("--start_tok", type=float, default=None,
                    help="only snapshots saved at >= this many tokens (pins the phase, e.g. the burst start)")
    ap.add_argument("--end_tok", type=float, default=None, help="only snapshots saved at <= this many tokens")
    ap.add_argument("--method", default="mean", choices=["mean", "1sqrt"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json to save (else copied from a ckpt)")
    ap.add_argument("--verify", action="store_true", help="reload the merged model + run a forward")
    args = ap.parse_args()

    if args.ckpts:
        dirs = [Path(x) for x in args.ckpts.split(",") if x.strip()]
    else:
        dirs = drop_job_boundaries(collect_snapshots(args.snapshots_dir))
    dirs = select_window(dirs, args.window_tok, args.last_n, args.start_tok, args.end_tok)
    n = len(dirs)
    if n < 2:
        raise SystemExit(f"need >=2 checkpoints to merge, got {n}")
    if n < 8:
        print(f"[warn] merging only {n} ckpts — WSM degrades below ~8-10 (2 collapses). Widen the window.",
              flush=True)

    toks = check_order(dirs)
    cfg_hash, tok_hash = same_model_and_tokenizer(dirs)
    w = merge_weights(n, args.method)
    span = (f"{(toks[0] or 0)/1e9:.1f}..{(toks[-1] or 0)/1e9:.1f}B" if all(t is not None for t in toks) else "n/a")
    print(f"merging {n} ckpts | method={args.method} | token-span={span}", flush=True)
    for d, wi in zip(dirs, w):
        print(f"  w={wi:.4f}  {d.name}", flush=True)

    # incremental fp32 accumulation: peak RAM ~= one fp32 state_dict + one loaded ckpt (not n of them)
    acc, out_dtype, digests = None, {}, {}
    for wi, d in zip(w, dirs):
        sd = load_state_dict_any(d)
        bad = [k for k, v in sd.items() if v.is_floating_point() and not torch.isfinite(v).all()]
        if bad:
            raise SystemExit(f"{d.name}: non-finite values in {len(bad)} tensors (first: {bad[0]}): leave this "
                             f"snapshot out of the window (--ckpts) and look at the run around it")
        h = hashlib.sha256()
        for f in sorted(d.glob("*.safetensors")) or sorted(d.glob("*.bin")):
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
        digests[d.name] = h.hexdigest()
        if acc is None:
            acc, shapes = {}, {}
            for k, v in sd.items():
                out_dtype[k], shapes[k] = v.dtype, tuple(v.shape)
                acc[k] = v.float() * wi if v.is_floating_point() else v.clone()
        else:
            # The same tensors in every source: a key missing from a later snapshot used to keep a partial sum,
            # under-weighted, with no error (only extra keys were refused).
            if set(sd) != set(acc):
                raise SystemExit(f"{d.name}: tensors differ from {dirs[0].name}: missing {sorted(set(acc) - set(sd))[:5]}, "
                                 f"extra {sorted(set(sd) - set(acc))[:5]}")
            for k, v in sd.items():
                if tuple(v.shape) != shapes[k] or v.dtype != out_dtype[k]:
                    raise SystemExit(f"{d.name}: {k} is {tuple(v.shape)} {v.dtype}, {shapes[k]} {out_dtype[k]} in "
                                     f"{dirs[0].name}")
                if v.is_floating_point():
                    acc[k].add_(v.float() * wi)
                elif not torch.equal(v, acc[k]):
                    raise SystemExit(f"{d.name}: the non-float buffer {k} differs from {dirs[0].name}")
        del sd
    merged = {k: (acc[k].to(out_dtype[k]) if acc[k].is_floating_point() else acc[k]) for k in acc}
    bad = [k for k, v in merged.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        raise SystemExit(f"the merge has non-finite values in {len(bad)} tensors (first: {bad[0]}): not saved")

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
        "sources": [d.name for d in dirs], "tokens": toks, "token_span": span, "sha256": digests,
        "config_sha256": cfg_hash, "tokenizer_sha256": tok_hash,
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
