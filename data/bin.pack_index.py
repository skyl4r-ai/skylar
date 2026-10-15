#!/usr/bin/env python3
# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Build the best-fit packing index of a tokenized folder (data/packing.py, arXiv 2404.10830), once, on CPU.

    python data/bin.pack_index.py --data <folder> --seq_len 8192 --tails keep [--workers 32]

Reads every shard once (on v6, 424 GB), writes `<folder>/pack/T<seq_len>_<tails>/` (about 10 bytes per document, ~1.5
GB on v6) and prints what the windows hold: documents read whole, documents split, padding, and the share of input
positions that do not see their document's start. The trainer reads it with `--pack keep|drop` and refuses an index
built for another seq_len, other tails or other shards. `--check N` rebuilds N shards at random and compares them
with the index (and that every document that fits a window is in exactly one window, whole).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.packing import PACK_VERSION, TAILS, PackIndex, index_dir, meta_sha, pack_shard, segments  # noqa: E402

_DTYPE = {"uint16": "<u2", "uint32": "<u4"}


def _shard_job(a):
    path, dtype, bos_id, seq_len, tails = a
    tok = np.fromfile(path, dtype=dtype)
    s, L, counts = pack_shard(tok, bos_id, seq_len, tails)
    # statistics of this shard
    starts = np.flatnonzero(tok == bos_id)
    n_docs = int(starts.size)
    lens = np.diff(np.append(starts, len(tok)))
    C = seq_len + 1
    long_ = lens > C
    tail_tok = int((lens[long_] - C).sum())           # positions of the chunks after the first, roughly
    return s, L, counts, {"tokens": int(len(tok)), "docs": n_docs, "docs_split": int(long_.sum()),
                          "tail_tokens": tail_tok, "seg_tokens": int(L.sum(dtype=np.int64))}


def build(data_dir, seq_len, tails, workers):
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "pretokenized_meta.json").read_text())
    dtype = _DTYPE[meta["dtype"]]
    ids = meta.get("special_ids") or {}
    bos_id, pad_id = int(ids.get("<bos>", 1)), int(ids.get("<pad>", 0))
    paths = [str(data_dir / "shards" / r["filename"]) for r in meta["shards"]]
    out = index_dir(data_dir, seq_len, tails)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    t0 = time.time()
    starts, lens, counts, stats = [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (s, L, c, st) in enumerate(ex.map(_shard_job, [(p, dtype, bos_id, seq_len, tails) for p in paths],
                                                  chunksize=4)):
            starts.append(s), lens.append(L), counts.append(c), stats.append(st)
            if (i + 1) % 500 == 0 or i + 1 == len(paths):
                print(f"  {i + 1:,}/{len(paths):,} shards, {time.time() - t0:.0f}s", flush=True)
    seg_start, seg_len = np.concatenate(starts), np.concatenate(lens)
    bin_counts = np.concatenate(counts)
    bin_ptr = np.zeros(len(bin_counts) + 1, dtype=np.int64)
    np.cumsum(bin_counts, out=bin_ptr[1:])
    shard_ptr = np.zeros(len(paths) + 1, dtype=np.int64)
    np.cumsum([len(c) for c in counts], out=shard_ptr[1:])
    for name, arr in (("seg_start", seg_start), ("seg_len", seg_len), ("bin_ptr", bin_ptr), ("shard_ptr", shard_ptr)):
        np.save(tmp / f"{name}.npy", arr)
    tot = {k: sum(s[k] for s in stats) for k in stats[0]}
    n_bins = int(len(bin_counts))
    positions = n_bins * (seq_len + 1)
    info = {"version": PACK_VERSION, "seq_len": seq_len, "tails": tails, "bos_id": bos_id, "pad_id": pad_id,
            "meta_sha256": meta_sha(data_dir), "shards": len(paths), "windows": n_bins, "segments": int(len(seg_len)),
            "tokens": tot["tokens"], "docs": tot["docs"], "docs_split": tot["docs_split"],
            "docs_split_share": tot["docs_split"] / max(1, tot["docs"]),
            "tokens_in_windows": tot["seg_tokens"],
            "padding_share": 1 - tot["seg_tokens"] / positions,
            "without_doc_start_share": (tot["tail_tokens"] / max(1, tot["tokens"])) if tails == "keep" else 0.0,
            "dropped_tokens_share": (tot["tail_tokens"] / max(1, tot["tokens"])) if tails == "drop" else 0.0,
            "consecutive_windows": int(sum((s["tokens"] - 1) // seq_len for s in stats)),
            "build_seconds": round(time.time() - t0, 1)}
    h = hashlib.sha256()
    for name in ("seg_start", "seg_len", "bin_ptr", "shard_ptr"):
        h.update((tmp / f"{name}.npy").read_bytes())
    info["index_sha256"] = h.hexdigest()
    (tmp / "index.json").write_text(json.dumps(info, indent=1))
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    return out, info


def check(data_dir, seq_len, tails, n, seed=0):
    """Rebuild n shards at random and compare; every document that fits is in one window, whole, exactly once."""
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "pretokenized_meta.json").read_text())
    dtype = _DTYPE[meta["dtype"]]
    idx = PackIndex(index_dir(data_dir, seq_len, tails), data_dir, seq_len, tails)
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(len(meta["shards"]), size=min(n, len(meta["shards"])), replace=False).tolist())
    C = seq_len + 1
    for i in picks:
        tok = np.fromfile(data_dir / "shards" / meta["shards"][i]["filename"], dtype=dtype)
        s, L, counts = pack_shard(tok, idx.bos_id, seq_len, tails)
        lo, hi = int(idx.shard_ptr[i]), int(idx.shard_ptr[i + 1])
        a, b = int(idx.bin_ptr[lo]), int(idx.bin_ptr[hi])
        assert np.array_equal(np.diff(idx.bin_ptr[lo:hi + 1]), counts), f"shard {i}: window sizes differ"
        assert np.array_equal(idx.seg_start[a:b], s) and np.array_equal(idx.seg_len[a:b], L), f"shard {i}: segments"
        per_bin = np.diff(idx.bin_ptr[lo:hi + 1])
        sums = np.add.reduceat(np.asarray(idx.seg_len[a:b], dtype=np.int64), np.cumsum(per_bin) - per_bin)
        assert (sums <= C).all(), f"shard {i}: a window over {C} tokens"
        ds, dl = segments(tok, idx.bos_id, seq_len, tails)
        fits = {(int(x), int(y)) for x, y in zip(ds, dl)}
        got = list(zip(np.asarray(idx.seg_start[a:b]).tolist(), np.asarray(idx.seg_len[a:b]).tolist()))
        assert len(got) == len(set(got)) == len(fits) and set(got) == fits, f"shard {i}: a piece missing or twice"
        bos = np.flatnonzero(tok == idx.bos_id)
        ends = np.append(bos[1:], len(tok))
        short = {(int(x), int(e - x)) for x, e in zip(bos, ends) if e - x <= C}
        assert short <= set(got), f"shard {i}: a document that fits is not whole in a window"
        print(f"  shard {i}: {hi - lo:,} windows, {b - a:,} pieces, identical; every document of <= {C} tokens whole",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="tokenized folder (pretokenized_meta.json + shards/)")
    ap.add_argument("--seq_len", type=int, required=True, help="the trainer's --seq_len")
    ap.add_argument("--tails", choices=TAILS, default="keep",
                    help="chunks after the first of a document longer than a window: train on them (keep) or not")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--check", type=int, default=0, help="after the build, rebuild N shards at random and compare")
    ap.add_argument("--check_only", action="store_true", help="only the check of an existing index")
    a = ap.parse_args()
    if not a.check_only:
        out, info = build(a.data, a.seq_len, a.tails, a.workers)
        print(f"index -> {out}")
        print(f"  {info['windows']:,} windows of {a.seq_len + 1:,} tokens ({info['consecutive_windows']:,} consecutive "
              f"ones), padding {info['padding_share']:.3%}")
        print(f"  {info['docs']:,} documents, {info['docs_split_share']:.2%} longer than a window (split); input "
              f"positions without their document's start {info['without_doc_start_share']:.2%}, tokens left out "
              f"{info['dropped_tokens_share']:.2%} ({info['build_seconds']:.0f}s)")
    if a.check or a.check_only:
        check(a.data, a.seq_len, a.tails, a.check or 3)
        print("check OK")


if __name__ == "__main__":
    main()
