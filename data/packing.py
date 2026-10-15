# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Best-fit packing of a tokenized corpus into training windows (Ding et al., "Fewer Truncations Improve Language
Modeling", arXiv 2404.10830).

Cutting the token stream into consecutive windows of seq_len tokens (the default of `MemmapTokenDataset`) cuts
documents: on the v6 corpus at 8,192 tokens, 13.1% of the documents are cut and 32.9% of the input positions do not
see the start of their document (measured on 40 random shards). The model then predicts code whose
definitions are in another window; the paper measures it on HumanEval as names used without a definition, 5.10% of
the samples with consecutive windows against 2.41% packed (Table 9), at the same perplexity (Table 10).

Here every document is first split only if it is longer than a window, into chunks of seq_len + 1 tokens that overlap
by one (every next-token pair of the document is a target exactly once); then the documents and chunks are packed
into windows of seq_len + 1 tokens by best-fit decreasing, shard by shard, never split again. A document that fits a
window is always read whole. The rest of a window is padding (0.05% of the positions on v6 at 8,192).

`tails`: what to do with the chunks after the first one of a document longer than a window. "keep" trains on them as
the paper does (they do not see their document's start); "drop" leaves them out, so no input position ever misses
its document's start, at the cost of those tokens (15.3% of v6 at 8,192, 2.6% at 32,768, 0.67% at 65,536).

The index of one folder lives in `<folder>/pack/T<seq_len>_<tails>/` (`bin.pack_index.py` builds it, once, on CPU):
  seg_start.npy  uint32  offset of every segment (document or chunk) inside its shard
  seg_len.npy    uint32  its length in tokens
  bin_ptr.npy    int64   segments of window b are seg_*[bin_ptr[b]:bin_ptr[b+1]]
  shard_ptr.npy  int64   windows of shard i (meta order) are bin_ptr indices shard_ptr[i]:shard_ptr[i+1]
  index.json             seq_len, tails, bos/pad ids, sha256 of pretokenized_meta.json, counts
It is a pure function of the shards, seq_len and tails: two builds give the same bytes.
"""
import bisect
import hashlib
import json
from pathlib import Path

import numpy as np

PACK_VERSION = 1
TAILS = ("keep", "drop")


def index_dir(data_dir, seq_len, tails):
    """Where the index of folder `data_dir` for windows of `seq_len` tokens lives."""
    return Path(data_dir) / "pack" / f"T{int(seq_len)}_{tails}"


def meta_sha(data_dir):
    """sha256 of the folder's pretokenized_meta.json: an index built on other shards is refused."""
    return hashlib.sha256((Path(data_dir) / "pretokenized_meta.json").read_bytes()).hexdigest()


def segments(tokens, bos_id, seq_len, tails="keep"):
    """(starts, lengths) of the documents and chunks of one shard, in shard order. A document starts at every <bos>
    (and at offset 0); one longer than seq_len + 1 tokens becomes chunks of seq_len + 1 tokens starting every seq_len
    tokens, the last one shorter; with tails="drop" only its first chunk is kept. A piece of one token has no target
    and is left out."""
    assert tails in TAILS, tails
    n = len(tokens)
    C, T = seq_len + 1, seq_len
    doc_s = np.flatnonzero(np.asarray(tokens) == bos_id).astype(np.int64)
    if doc_s.size == 0 or doc_s[0] != 0:
        doc_s = np.concatenate([[0], doc_s])
    doc_e = np.append(doc_s[1:], n)
    L = doc_e - doc_s
    # chunks per document: 1 if it fits, else ceil((L - 1) / T) (the last chunk ends at the document's end)
    k = np.where(L <= C, 1, -(-(L - 1) // T))
    if tails == "drop":
        k = np.ones_like(k)
    idx = np.repeat(np.arange(len(L)), k)
    first = np.repeat(np.cumsum(k) - k, k)
    j = np.arange(len(idx)) - first                       # chunk number inside its document
    starts = doc_s[idx] + j * T
    lens = np.minimum(C, doc_e[idx] - starts)
    keep = lens >= 2
    return starts[keep], lens[keep]


def best_fit(lens, capacity):
    """Best-fit decreasing (2404.10830 §3): longest first, each into the open window with the least room that still
    fits it, else a new window. Returns the window of every item and the number of windows. Deterministic: ties
    in length keep the input order, ties in room go to the lowest window."""
    order = np.argsort(-np.asarray(lens), kind="stable")
    room = []            # sorted (free, window) of the open windows
    where = np.empty(len(lens), dtype=np.int64)
    n_bins = 0
    for i in order.tolist():
        L = int(lens[i])
        p = bisect.bisect_left(room, (L, -1))
        if p == len(room):
            b = n_bins
            n_bins += 1
            free = capacity - L
        else:
            free, b = room.pop(p)
            free -= L
        where[i] = b
        if free > 0:
            bisect.insort(room, (free, b))
    return where, n_bins


def pack_shard(tokens, bos_id, seq_len, tails="keep"):
    """The packed windows of one shard: (seg_start, seg_len, bin_counts), segments grouped by window and in shard order
    inside a window, bin_counts = segments per window."""
    s, L = segments(tokens, bos_id, seq_len, tails)
    where, n_bins = best_fit(L, seq_len + 1)
    order = np.lexsort((s, where))                        # by window, then by position in the shard
    counts = np.bincount(where, minlength=n_bins)
    return s[order].astype(np.uint32), L[order].astype(np.uint32), counts.astype(np.int64)


class PackIndex:
    """A built index, memory-mapped. `window(shard_pos, w)` gives the segments of window w of the shard at position
    shard_pos of the meta's shard list."""

    def __init__(self, path, data_dir=None, seq_len=None, tails=None):
        self.path = Path(path)
        f = self.path / "index.json"
        if not f.is_file():
            raise FileNotFoundError(
                f"no packing index in {self.path}: build it once with\n"
                f"  python data/bin.pack_index.py --data {data_dir or '<folder>'} --seq_len {seq_len or '<T>'} "
                f"--tails {tails or 'keep'}")
        self.info = json.loads(f.read_text())
        if self.info.get("version") != PACK_VERSION:
            raise ValueError(f"{self.path}: index version {self.info.get('version')}, this code reads {PACK_VERSION}: "
                             f"build it again")
        if seq_len is not None and self.info["seq_len"] != seq_len:
            raise ValueError(f"{self.path}: built for seq_len {self.info['seq_len']}, the run uses {seq_len}")
        if tails is not None and self.info["tails"] != tails:
            raise ValueError(f"{self.path}: built with tails={self.info['tails']}, the run asks {tails}")
        if data_dir is not None and self.info["meta_sha256"] != meta_sha(data_dir):
            raise ValueError(f"{self.path}: built on another pretokenized_meta.json than {data_dir}'s: build it again")
        self.seg_start = np.load(self.path / "seg_start.npy", mmap_mode="r")
        self.seg_len = np.load(self.path / "seg_len.npy", mmap_mode="r")
        self.bin_ptr = np.load(self.path / "bin_ptr.npy", mmap_mode="r")
        self.shard_ptr = np.load(self.path / "shard_ptr.npy")
        self.seq_len, self.tails = self.info["seq_len"], self.info["tails"]
        self.bos_id, self.pad_id = self.info["bos_id"], self.info["pad_id"]

    def n_bins(self, shard_pos):
        return int(self.shard_ptr[shard_pos + 1] - self.shard_ptr[shard_pos])

    def window(self, shard_pos, w):
        b = int(self.shard_ptr[shard_pos]) + int(w)
        lo, hi = int(self.bin_ptr[b]), int(self.bin_ptr[b + 1])
        return np.asarray(self.seg_start[lo:hi], dtype=np.int64), np.asarray(self.seg_len[lo:hi], dtype=np.int64)


def assemble(shard, starts, lens, seq_len, pad_id):
    """Tokens (seq_len + 1), per-position segment ids and per-target validity of one packed window. Segment j holds
    positions of its j-th piece; the padding after them is one more segment. Target t (token t + 1 predicted at t)
    is valid only inside one segment: no target crosses a document, a chunk edge or the padding."""
    C = seq_len + 1
    tok = np.full(C, pad_id, dtype=np.int64)
    seg = np.empty(C, dtype=np.int32)
    pos = 0
    for j, (s, L) in enumerate(zip(starts.tolist(), lens.tolist())):
        tok[pos:pos + L] = shard[s:s + L]
        seg[pos:pos + L] = j
        pos += L
    seg[pos:] = len(starts)
    valid = (seg[1:] == seg[:-1]) & (seg[:-1] < len(starts))     # the padding predicts nothing
    return tok, seg, valid
