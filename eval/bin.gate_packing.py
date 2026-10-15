"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Gates of best-fit packing (data/packing.py, data/bin.pack_index.py, `--pack` in training/bin.pretrain.py).

On a synthetic corpus written here (documents from 2 to 6x the window, so some are split), in seconds:
  P1  index: two builds give the same bytes; every document that fits a window is in exactly one window, whole;
      a longer one becomes chunks of seq_len + 1 tokens overlapping by one, so each of its next-token pairs is a
      target exactly once (keep) or only its first chunk is kept (drop); no window holds more than seq_len + 1 tokens.
  P2  loader: over a full pass of packed windows, tokens, targets (-100 across a document, a chunk edge and the
      padding) and per-position document ids are those of the index; the same samples on 1 and 3 ranks; a mixture
      reads packed windows too; without document ids, with another seq_len, other tails or a stale index it refuses.
      A folder whose meta says "validation": "none" (the COBOL burst) trains on every shard and adds no validation
      window to a mixture; without the marker its last shard would leave training to become validation.
  P3  without packing the windows are the consecutive slices of every shard (the behaviour before packing).
  P4  (CUDA) on the launch architecture (KDA 3:1, block AttnRes, gated norm, output gate, SiTU), in float32: the
      hidden states and the loss of packed windows equal those of their documents run one by one, so no document
      sees another; without the boundaries they differ by orders of magnitude (the control).

    python eval/bin.gate_packing.py

Exit code 0 = every gate passed, 1 = at least one failed.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from data.memmap_dataset import MemmapTokenDataset, MixtureTokenDataset  # noqa: E402
from data.packing import index_dir, segments  # noqa: E402

PASS, FAIL, TODO = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m", "\033[33mTODO\033[0m"
results = []
T = 64
VOCAB = 500


def report(gate, ok, detail, todo=False):
    results.append(None if todo else ok)
    print(f"  [{TODO if todo else (PASS if ok else FAIL)}] {gate:<34} {detail}")


def write_corpus(d, n_shards=3, seed=0):
    """Shards of <bos> body <eos> documents (bos 1, eos 2, pad 0), lengths mostly short and some up to 6 windows."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(d, "shards"), exist_ok=True)
    recs = []
    for i in range(n_shards):
        docs = []
        for _ in range(300):
            L = int(rng.integers(2, 40)) if rng.random() < 0.85 else int(rng.integers(40, 6 * T))
            docs.append(np.concatenate([[1], rng.integers(11, VOCAB, L - 2), [2]]).astype("<u2"))
        arr = np.concatenate(docs)
        fn = f"shard_{i:06d}.bin"
        arr.tofile(os.path.join(d, "shards", fn))
        recs.append({"index": i, "filename": fn, "num_tokens": int(len(arr))})
    meta = {"dtype": "uint16", "vocab_size": VOCAB, "total_tokens": sum(r["num_tokens"] for r in recs),
            "shards": recs, "val_shards": [n_shards - 1], "special_ids": {"<pad>": 0, "<bos>": 1, "<eos>": 2}}
    with open(os.path.join(d, "pretokenized_meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


def build(d, tails):
    r = subprocess.run([sys.executable, os.path.join(REPO, "data", "bin.pack_index.py"), "--data", d, "--seq_len",
                        str(T), "--tails", tails, "--workers", "2", "--check", "3"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout[-500:] + r.stderr[-1500:])
    return json.load(open(os.path.join(index_dir(d, T, tails), "index.json"))), r.stdout


def gate_p1_index(d):
    for tails in ("keep", "drop"):
        try:
            info, _ = build(d, tails)
            sha = info["index_sha256"]
            info2, out = build(d, tails)
            same = info2["index_sha256"] == sha
            # pairs of every document covered once (keep) / first chunk only (drop), on shard 0
            tok = np.fromfile(os.path.join(d, "shards", "shard_000000.bin"), dtype="<u2")
            s, L = segments(tok, 1, T, tails)
            bos = np.flatnonzero(tok == 1)
            ends = np.append(bos[1:], len(tok))
            pairs = np.zeros(len(tok), dtype=np.int64)        # pairs[i]: times the pair (i, i+1) is a target
            for a, n in zip(s.tolist(), L.tolist()):
                pairs[a:a + n - 1] += 1
            inside = np.ones(len(tok), dtype=bool)
            inside[ends - 1] = False                           # (eos, next bos) is not a pair of a document
            if tails == "keep":
                ok_pairs = (pairs[inside] == 1).all() and (pairs[~inside] == 0).all()
            else:
                first = np.zeros(len(tok), dtype=bool)
                for b, e in zip(bos.tolist(), ends.tolist()):
                    first[b:min(e, b + T + 1) - 1] = True
                ok_pairs = (pairs[first] == 1).all() and (pairs[~first] == 0).all()
            report(f"P1 index ({tails})", same and ok_pairs and "check OK" in out,
                   f"{info['windows']} windows, {info['docs_split']} documents split, padding "
                   f"{info['padding_share']:.2%} · deterministic {same} · every pair once {ok_pairs} · check "
                   f"{'OK' if 'check OK' in out else 'FAILED'}")
        except Exception as e:
            report(f"P1 index ({tails})", False, f"{type(e).__name__}: {str(e)[:200]}")


def gate_p2_loader(d):
    import torch
    for tails in ("keep", "drop"):
        try:
            ds = MemmapTokenDataset(d, seq_len=T, seed=3, pack=tails)
            idx, bad, seen = ds.pack, 0, []
            for split in ("train", "val"):
                for k in range(ds.n_windows(split)):
                    x, y, doc = ds.batch_of(split, [k], seed=9, return_doc_ids=True)
                    x, y, doc = x[0].numpy(), y[0].numpy(), doc[0].numpy()
                    shard, w = ds._locate(split, k, 9)
                    st, ln = idx.window(shard.pos, w)
                    pos = 0
                    for j, (a, n) in enumerate(zip(st.tolist(), ln.tolist())):
                        full = np.asarray(shard[a:a + n], dtype=np.int64)
                        m = min(T, pos + n)
                        bad += not np.array_equal(x[pos:m], full[:m - pos])
                        bad += not (doc[pos:m] == j).all()
                        mt = min(T, pos + n - 1)
                        bad += not np.array_equal(y[pos:mt], full[1:mt - pos + 1])
                        if pos + n - 1 < T:
                            bad += y[pos + n - 1] != -100
                        seen.append((shard.pos, a, n))
                        pos += n
                    bad += not ((y[pos:] == -100).all() and (doc[pos:] == len(st)).all())
            all_pieces = set()
            for p in range(idx.info["shards"]):
                a, b = int(idx.bin_ptr[idx.shard_ptr[p]]), int(idx.bin_ptr[idx.shard_ptr[p + 1]])
                all_pieces |= {(p, int(s), int(n)) for s, n in zip(idx.seg_start[a:b], idx.seg_len[a:b])}
            once = len(seen) == len(set(seen)) == len(all_pieces) and set(seen) == all_pieces
            # same samples whatever the number of ranks
            rows = {}
            for world in (1, 3):
                got = []
                for bi in range(6 if world == 3 else 18):
                    for r in range(world):
                        got.append((bi * world + r, ds.get_batch_at("train", 2, bi, rank=r, world=world, seed=5,
                                                                     return_doc_ids=True)[0]))
                rows[world] = torch.cat([t for _, t in sorted(got, key=lambda g: g[0])])
            ranks = torch.equal(rows[1], rows[3])
            report(f"P2 loader ({tails})", bad == 0 and once and ranks,
                   f"full pass of {len(seen)} pieces: mismatches {bad}, each piece once {once} · 1 vs 3 ranks "
                   f"identical {ranks}")
        except Exception as e:
            report(f"P2 loader ({tails})", False, f"{type(e).__name__}: {str(e)[:200]}")
    # mixture and refusals
    try:
        d2 = d.rstrip("/") + "_second"
        if not os.path.exists(d2):
            os.symlink(d, d2)
        mx = MixtureTokenDataset([(d, 0.6), (d2, 0.4)], seq_len=T, seed=1, pack="keep")
        x, y, doc = mx.get_batch_at("train", 4, 2, seed=1, return_doc_ids=True)
        mix_ok = x.shape == (4, T) and bool((doc[:, 1:] >= doc[:, :-1]).all())
        refusals = []
        ds = MemmapTokenDataset(d, seq_len=T, seed=3, pack="keep")
        for name, fn in (("no doc ids", lambda: ds.get_batch_at("train", 1, 0)),
                         ("other seq_len", lambda: MemmapTokenDataset(d, seq_len=T * 2, pack="keep")),
                         ("other tails", lambda: __import__("data.packing", fromlist=["PackIndex"]).PackIndex(
                             index_dir(d, T, "keep"), d, T, "drop"))):
            try:
                fn()
                refusals.append(f"{name}: ACCEPTED")
            except (ValueError, FileNotFoundError):
                refusals.append(f"{name}: refused")
        meta_p = os.path.join(d, "pretokenized_meta.json")
        keep_meta = open(meta_p).read()
        try:
            with open(meta_p, "a") as f:
                f.write(" ")                                   # another meta: the index is stale
            MemmapTokenDataset(d, seq_len=T, pack="keep")
            refusals.append("stale index: ACCEPTED")
        except ValueError:
            refusals.append("stale index: refused")
        finally:
            open(meta_p, "w").write(keep_meta)
        ok = mix_ok and all(r.endswith("refused") for r in refusals)
        report("P2 mixture and refusals", ok, f"mixture batch {tuple(x.shape)} · " + ", ".join(refusals))
    except Exception as e:
        report("P2 mixture and refusals", False, f"{type(e).__name__}: {str(e)[:200]}")


def gate_p2_no_validation(d):
    try:
        b = d.rstrip("/") + "_burst"
        write_corpus(b, n_shards=4, seed=7)
        meta_p = os.path.join(b, "pretokenized_meta.json")
        meta = json.load(open(meta_p))
        meta["val_shards"] = []
        json.dump(meta, open(meta_p, "w"))
        plain = MemmapTokenDataset(b, seq_len=T)                       # no marker: the last-shard heuristic
        heuristic = len(plain.splits["train"]) == 3 and len(plain.splits["val"]) == 1
        meta["validation"] = "none"
        json.dump(meta, open(meta_p, "w"))
        build(b, "keep")
        nv = MemmapTokenDataset(b, seq_len=T, pack="keep")
        alone = len(nv.splits["train"]) == 4 and nv.splits["val"] == [] and nv.n_windows("val") == 0
        base = MemmapTokenDataset(d, seq_len=T, seed=3, pack="keep")
        mx = MixtureTokenDataset([(d, 0.9), (b, 0.1)], seq_len=T, seed=1, pack="keep")
        same_val = [p.n_windows("val") for p in mx.parts] == [base.n_windows("val"), 0] \
            and mx.n_windows("val") == base.n_windows("val")
        x, y, doc = mx.get_batch_at("train", 100, 0, seed=1, return_doc_ids=True)
        from_b = sum(mx.locate(k)[0] == 1 for k in range(100))
        report("P2 folder without validation", heuristic and alone and same_val and from_b == 10 and x.shape[0] == 100,
               f"without the marker 1 of 4 shards leaves training {heuristic} · with \"validation\": \"none\" 4 shards "
               f"train, no validation window {alone} · mixture validation = the other folder's {same_val} · "
               f"{from_b}/100 samples from it")
    except Exception as e:
        report("P2 folder without validation", False, f"{type(e).__name__}: {str(e)[:200]}")


def gate_p3_unpacked(d):
    try:
        ds = MemmapTokenDataset(d, seq_len=T, seed=3)
        shards = ds.splits["train"]
        W = [(len(s) - 1) // T for s in shards]
        bad = 0
        for k in range(sum(W)):
            shard, w = ds._locate("train", k, 4)
            bad += not np.array_equal(ds.window_at("train", k, 4), np.asarray(shard[w * T:w * T + T + 1]))
        x, y, doc = ds.batch_of("train", [0, 1], 4, return_doc_ids=True, bos_id=1)
        bos_doc = bool(torch_equal((x == 1).cumsum(1), doc)) and bool((y[:, :-1] == x[:, 1:]).all())
        report("P3 unpacked = consecutive slices", bad == 0 and ds.n_windows("train") == sum(W) and bos_doc,
               f"{sum(W)} windows, mismatches {bad} · doc ids from <bos> {bos_doc}")
    except Exception as e:
        report("P3 unpacked = consecutive slices", False, f"{type(e).__name__}: {str(e)[:200]}")


def torch_equal(a, b):
    import torch
    return torch.equal(a.to(torch.int64), b.to(torch.int64))


def gate_p4_isolation(d):
    """Every piece of packed windows, run alone, gives the same hidden states and the same loss as inside the window.
    The control runs the same windows without the boundaries: the pieces after the first must then differ by orders
    of magnitude more (on a model just initialised the LOSS barely depends on the context, 1e-5 without boundaries:
    the hidden states are the measure; the loss is reported too)."""
    import torch as t
    if not t.cuda.is_available():
        report("P4 packed = documents alone", False, "the KDA kernels need CUDA: not runnable here", todo=True)
        return
    try:
        from models.config import get_config
        from models.decoder import Skylar2ForCausalLM
        cfg = get_config("test", vocab_size=VOCAB, kda_ratio="3:1", attn_res=True, attn_res_mode="block",
                         attn_res_block_size=2, gated_norm=16, attn_out_gate=True, hidden_act="situ_glu")
        cfg.max_seq_len, cfg.dropout = 4 * T, 0.0
        t.manual_seed(0)
        m = Skylar2ForCausalLM(cfg).cuda().float().eval()
        worst_h, worst_loss, leak, n_docs = 0.0, 0.0, float("inf"), 0
        for tails in ("keep", "drop"):
            ds = MemmapTokenDataset(d, seq_len=T, seed=3, pack=tails)
            x, y, doc = ds.get_batch_at("train", 8, 1, seed=2, return_doc_ids=True, device="cuda")
            with t.no_grad():
                packed = m(input_ids=x, labels=y, document_ids=doc, return_hidden=True)
                blind = m(input_ids=x, return_hidden=True)["hidden"]
                tot, cnt = 0.0, 0
                for b in range(x.shape[0]):
                    for j in doc[b].unique().tolist():
                        sel = (doc[b] == j).nonzero().flatten()
                        yy = y[b, sel]
                        n = int((yy != -100).sum())
                        if n == 0:
                            continue                               # the padding
                        alone = m(input_ids=x[b, sel][None], labels=yy[None], return_hidden=True)
                        h = alone["hidden"][0]
                        scale = h.abs().max()
                        worst_h = max(worst_h, ((packed["hidden"][b, sel] - h).abs().max() / scale).item())
                        if int(sel[0]) > 0:                        # a piece after another one in its window
                            leak = min(leak, ((blind[b, sel] - h).abs().max() / scale).item())
                        tot, cnt, n_docs = tot + alone["loss"].item() * n, cnt + n, n_docs + 1
            worst_loss = max(worst_loss, abs(packed["loss"].item() - tot / cnt) / (tot / cnt))
        report("P4 packed = documents alone", worst_h < 5e-3 and worst_loss < 1e-5 and leak > 0.1,
               f"launch architecture in float32, {n_docs} pieces: hidden states {worst_h:.1e} (threshold 5e-3), loss "
               f"{worst_loss:.1e} (threshold 1e-5) · without the boundaries {leak:.1e} (must be > 0.1)")
    except Exception as e:
        report("P4 packed = documents alone", False, f"{type(e).__name__}: {str(e)[:200]}")


def main():
    print(f"\nPacking gates — synthetic corpus, windows of {T} tokens\n")
    tmp = tempfile.mkdtemp(prefix="gate_packing_")
    try:
        d = os.path.join(tmp, "corpus")
        write_corpus(d)
        gate_p1_index(d)
        gate_p2_loader(d)
        gate_p2_no_validation(d)
        gate_p3_unpacked(d)
        gate_p4_isolation(d)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = sum(r is True for r in results)
    print(f"\n  {ok}/{sum(r is not None for r in results)} gates passed, "
          f"{sum(r is None for r in results)} not runnable here\n")
    sys.exit(0 if all(r is not False for r in results) else 1)


if __name__ == "__main__":
    main()
