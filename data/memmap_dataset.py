# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Reusable memmap token dataset + background prefetcher — the streaming data layer.

Memmaps sharded little-endian token files instead of loading the whole corpus into one
RAM tensor: O(1) RAM regardless of corpus size, next-token windows sampled either as a
shuffled pass WITHOUT replacement (`get_batch_at`, the pretraining default) or at random
(`get_batch`), train/val split, long-context ready. This is the piece the framework was missing — the
`bin.*` trainers used to define their dataset inline; now pretrain/eval share this module.

Shard format (written by the tokenizer step): little-endian **uint16** (`<u2`) or
**uint32** (`<u4`), auto-detected from `pretokenized_meta.json["dtype"]`. Documents are
packed as `<bos> … <eos> …` back to back. Random windows cross document boundaries on
purpose (the model learns the `<eos><bos>` reset) — packed-LM standard. For attention that
must NOT cross documents, pass `bos_id=` to `get_batch(..., return_doc_ids=True)` and feed
the returned per-position document ids to a FlexAttention block mask.

Usage:
    from data.memmap_dataset import MemmapTokenDataset, Prefetcher
    ds = MemmapTokenDataset("data/tokenized", seq_len=2048)
    x, y = ds.get_batch("train", batch_size=16, device="cuda")          # (B,T) int64, random
    x, y = ds.get_batch_at("train", 16, batch_index=0, device="cuda")   # shuffled pass, no repeats
    pf = Prefetcher(ds, "train", 16, "cuda", seed=1234, indexed=True)    # overlap load+compute
    x, y = pf.next()

Several folders as one corpus, with weights chosen at launch (`MixtureTokenDataset`, same interface;
`bin.pretrain.py --data_mix "a=0.7,b=0.3"`):
    mix = MixtureTokenDataset([("data/code", 0.7), ("data/text", 0.3)], seq_len=2048)
    x, y = mix.get_batch_at("train", 16, batch_index=0, device="cuda")
"""
import hashlib
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

# meta["dtype"] string -> little-endian numpy dtype
_DTYPE_MAP = {"uint16": "<u2", "uint32": "<u4"}

# How many shards may stay mapped at once. Each live np.memmap costs one file descriptor for
# its whole lifetime (CPython dups the fd inside mmap), so mapping every shard up front dies
# with OSError 24 once a corpus grows past `ulimit -n` — the 4B mix has 31,905 shards against
# a default limit of 1024. Re-opening on demand costs microseconds.
_OPEN_SHARDS = 256

_MASK64 = (1 << 64) - 1


def _mix64(x):
    """splitmix64 finaliser: a cheap, well-mixed 64-bit hash."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _permute(i, n, key, rounds=4):
    """Bijection of [0, n): a balanced Feistel network on the smallest even power of two >= n,
    walking the cycle back into range. Deterministic in (key, i) and O(1) in memory, so the order of
    the ~60M windows of a 500B-token corpus is never materialised."""
    bits = max(2, (n - 1).bit_length())
    bits += bits & 1
    half = bits // 2
    low = (1 << half) - 1
    while True:
        left, right = i >> half, i & low
        for r in range(rounds):
            left, right = right, left ^ (_mix64(key ^ (r << 56) ^ right) & low)
        i = (left << half) | right
        if i < n:
            return i


class _ShardCache:
    """LRU of open memmaps, shared by every shard of one dataset. Thread-safe: the Prefetcher
    reads from a background thread while the main thread may also pull a batch."""

    def __init__(self, capacity=_OPEN_SHARDS):
        self.capacity = capacity
        self._open = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path, dtype):
        with self._lock:
            mm = self._open.pop(path, None)
            if mm is None:
                mm = np.memmap(path, dtype=dtype, mode="r")
                while len(self._open) >= self.capacity:
                    self._open.popitem(last=False)      # drop the least recently used
            self._open[path] = mm
            return mm


class _LazyShard:
    """One shard behaving like the array it maps: `len()` and slicing, opened on first touch.

    The length comes from the meta (`num_tokens`) or from the file size, so building the
    dataset touches no file at all — the 4B corpus goes from ~125s of constructor to instant.
    """

    __slots__ = ("path", "dtype", "_len", "_cache")

    def __init__(self, path, dtype, n_tokens, cache):
        self.path, self.dtype, self._len, self._cache = str(path), dtype, int(n_tokens), cache

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        return self._cache.get(self.path, self.dtype)[item]


class MemmapTokenDataset:
    def __init__(self, data_dir, seq_len=2048, val_frac=0.01, seed=1234):
        self.dir = Path(data_dir)
        self.seq_len = seq_len
        meta = json.loads((self.dir / "pretokenized_meta.json").read_text())
        dt = meta["dtype"]
        assert dt in _DTYPE_MAP, f"expected dtype uint16/uint32, got {dt!r}"
        self.np_dtype = _DTYPE_MAP[dt]
        self.vocab_size = meta["vocab_size"]
        self.meta = meta

        # Shards are described, not opened: `num_tokens` comes from the meta (falling back to
        # the file size), so a 31,905-shard corpus costs zero file descriptors here.
        self._cache = _ShardCache()
        itemsize = np.dtype(self.np_dtype).itemsize
        shards, idxs = [], []
        for rec in meta["shards"]:
            p = self.dir / "shards" / rec["filename"]
            n = rec.get("num_tokens")
            if n is None:
                n = os.path.getsize(p) // itemsize
            if n > seq_len + 1:
                shards.append(_LazyShard(p, self.np_dtype, n, self._cache))
                idxs.append(rec["index"])
        if not shards:
            raise RuntimeError(f"no shard longer than seq_len+1={seq_len + 1} in {data_dir}")

        # Prefer an EXPLICIT per-source val set written by the mixer (meta['val_shards']);
        # those shards are a representative mix + identity-disjoint from train. Else fall
        # back to the last-val_frac-of-shards heuristic.
        val_set = set(meta.get("val_shards") or [])
        if val_set:
            train = [s for s, i in zip(shards, idxs) if i not in val_set]
            val = [s for s, i in zip(shards, idxs) if i in val_set]
            self.splits = {"train": train or shards, "val": val or shards[-1:]}
        else:
            n_val = max(1, int(round(len(shards) * val_frac))) if len(shards) > 1 and val_frac > 0 else 0
            self.splits = {"train": shards[: len(shards) - n_val] or shards,
                           "val": shards[len(shards) - n_val:] if n_val else shards[-1:]}
        # length-weighted shard sampling probabilities per split
        self._w = {}
        for k, sl in self.splits.items():
            lens = np.array([len(s) for s in sl], dtype=np.float64)
            self._w[k] = lens / lens.sum()
        self.rng = np.random.default_rng(seed)
        self.total_tokens = meta["total_tokens"]
        self._win_cum = {}

    def get_batch(self, split, batch_size, device=None, rng=None,
                  return_doc_ids=False, bos_id=None):
        """Random (B, seq_len) next-token windows. Returns int64 (x, y) — or (x, y, doc_ids)
        when return_doc_ids=True and bos_id is given (doc_ids[b,t] = #<bos> seen up to t,
        i.e. a per-position document index for document-masked attention)."""
        r = rng if rng is not None else self.rng   # prefetch thread passes its OWN rng (thread-safety)
        shards = self.splits[split]
        w = self._w[split]
        T = self.seq_len
        xb = np.empty((batch_size, T), dtype=np.int64)
        yb = np.empty((batch_size, T), dtype=np.int64)
        for i in range(batch_size):
            si = r.choice(len(shards), p=w)
            s = shards[si]
            off = int(r.integers(0, len(s) - T - 1))
            chunk = np.asarray(s[off: off + T + 1], dtype=np.int64)
            xb[i] = chunk[:-1]
            yb[i] = chunk[1:]
        return self._to_batch(xb, yb, device, return_doc_ids, bos_id)

    def _window_index(self, split):
        """Cumulative count of non-overlapping windows of seq_len+1 tokens per shard (the last token
        of a window is the first target of the next)."""
        cum = self._win_cum.get(split)
        if cum is None:
            T = self.seq_len
            cum = np.cumsum([(len(s) - 1) // T for s in self.splits[split]], dtype=np.int64)
            self._win_cum[split] = cum
        return cum

    def get_batch_at(self, split, batch_size, batch_index, device=None, rank=0, world=1, seed=0,
                     return_doc_ids=False, bos_id=None, offset=0):
        """Batch number `batch_index` of a shuffled pass WITHOUT replacement over the split's
        non-overlapping windows. Sample k = offset + (batch_index * world + rank) * batch_size + j is
        window perm(k mod W) of pass k // W, with a new permutation at every pass. Deterministic in
        (seed, k): a resumed run continues exactly where it stopped, and the ranks of a data-parallel
        run never read the same window. With random windows (`get_batch`) a run of 0.6 corpora sees
        only 45% of the corpus and repeats the rest; here it sees 60%, once.

        `offset` is the number of samples the whole run has already consumed. The window of sample k
        does not depend on world or batch_size, so a run resumed on a different number of GPUs passes
        offset = samples consumed and batch_index from 0: it reads the next samples, none twice."""
        T = self.seq_len
        xb = np.empty((batch_size, T), dtype=np.int64)
        yb = np.empty((batch_size, T), dtype=np.int64)
        base = offset + (batch_index * world + rank) * batch_size
        for j in range(batch_size):
            chunk = self.window_at(split, base + j, seed)
            xb[j] = chunk[:-1]
            yb[j] = chunk[1:]
        return self._to_batch(xb, yb, device, return_doc_ids, bos_id)

    def window_at(self, split, k, seed=0):
        """The seq_len+1 tokens of sample k of the shuffled passes (int64): window perm(k mod W) of
        pass k // W, with a new permutation at every pass."""
        cum = self._window_index(split)
        n = int(cum[-1])
        T = self.seq_len
        p = _permute(k % n, n, _mix64(_mix64(seed) ^ (k // n)))
        si = int(np.searchsorted(cum, p, side="right"))
        w = p - (int(cum[si - 1]) if si else 0)
        return np.asarray(self.splits[split][si][w * T: w * T + T + 1], dtype=np.int64)

    def _to_batch(self, xb, yb, device, return_doc_ids, bos_id):
        import torch
        x = torch.from_numpy(xb)
        y = torch.from_numpy(yb)
        doc = None
        if return_doc_ids and bos_id is not None:
            doc = (x == bos_id).cumsum(dim=1).to(torch.int32)   # per-position document index
        if device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if doc is not None:
                doc = doc.to(device, non_blocking=True)
        return (x, y, doc) if return_doc_ids else (x, y)

    def n_windows(self, split):
        """Number of non-overlapping windows in a split: one pass of `get_batch_at`."""
        return int(self._window_index(split)[-1])


def _name_key(name):
    """Stable 64-bit key of a folder name: a folder keeps its order when others are added or removed."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


class MixtureTokenDataset:
    """
    Several tokenized folders read as one corpus, with the weights chosen at launch instead of when the corpus
    is tokenized (DoReMi, arXiv 2305.10429, also sets them at training time). Changing the mix no longer means
    tokenizing again: the corpus grows by adding a folder.

    - `sources`: [(path, weight), ...]. Weights are shares of SAMPLES (windows of seq_len tokens), normalised.
    - Every folder keeps its own shuffled passes (`MemmapTokenDataset.window_at`, seeded by the folder name):
      a folder whose weight is above its share of tokens is repeated, with a new order at every pass; one
      below its share is read in part.
    - Which folder sample k comes from is a pattern of `block` samples, repeated. Every folder gets
      round(weight * block) of them (largest remainder), spread by a smooth weighted round-robin, so any
      stretch of the run holds each folder in its proportion. The pattern and the windows are pure functions
      of k: a run resumed on another number of GPUs continues exactly, as with one folder.
    - The weights can change at a resume (a COBOL burst, a new folder): `state` holds, per folder, the samples
      it had already given (`base`) and the sample `k0` where the current weights started. `restate()` builds
      it from the checkpoint's state, so in every folder no window is read twice and none is skipped.
      A folder that leaves the mix and comes back starts again from its first window.
    """

    def __init__(self, sources, seq_len=2048, seed=1234, block=10_000, state=None):
        assert sources, "no source"
        self.names = [Path(p).name for p, _ in sources]
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"two sources with the same folder name: {self.names}")
        w = np.array([float(x) for _, x in sources], dtype=np.float64)
        if (w <= 0).any():
            raise ValueError(f"weights must be > 0: {dict(zip(self.names, w))}")
        self.weights = w / w.sum()
        self.parts = [MemmapTokenDataset(p, seq_len=seq_len, seed=seed + i) for i, (p, _) in enumerate(sources)]
        p0 = self.parts[0]
        for name, p in zip(self.names, self.parts):
            if (p.vocab_size, p.np_dtype) != (p0.vocab_size, p0.np_dtype):
                raise ValueError(f"{name}: vocab {p.vocab_size} {p.np_dtype}, {self.names[0]}: "
                                 f"{p0.vocab_size} {p0.np_dtype}; all folders need the same tokenizer")
        self.seq_len, self.vocab_size, self.np_dtype = seq_len, p0.vocab_size, p0.np_dtype
        self.total_tokens = sum(p.total_tokens for p in self.parts)
        self.splits = {k: [s for p in self.parts for s in p.splits[k]] for k in ("train", "val")}
        self.rng = np.random.default_rng(seed)
        self.block = block
        self.quota, self._src, self._local, self._prefix = self._pattern(self.weights, block)
        self.state = state or {"k0": 0, "base": {n: 0 for n in self.names}, "block": block,
                               "quota": dict(zip(self.names, self.quota.tolist()))}

    @staticmethod
    def _pattern(weights, block):
        """Quotas per block (largest remainder) and, for every position of the block, its folder, its index
        among that folder's samples in the block, and the per-folder counts before it."""
        raw = weights * block
        q = np.floor(raw).astype(np.int64)
        for i in np.argsort(-(raw - q), kind="stable")[: block - int(q.sum())]:
            q[i] += 1
        if (q == 0).any():
            raise ValueError(f"a weight is below 1/{block}: raise the block or the weight")
        S = len(q)
        src = np.empty(block, dtype=np.int64)
        local = np.empty(block, dtype=np.int64)
        prefix = np.zeros((block + 1, S), dtype=np.int64)
        cur = np.zeros(S, dtype=np.int64)
        given = np.zeros(S, dtype=np.int64)
        for pos in range(block):                       # smooth weighted round-robin (nginx)
            cur += q
            s = int(np.argmax(cur))
            cur[s] -= block
            src[pos], local[pos] = s, given[s]
            given[s] += 1
            prefix[pos + 1] = given
        assert (given == q).all()
        return q, src, local, prefix

    def restate(self, old, samples_done):
        """The state for these weights from a checkpoint's `old` state, at `samples_done` samples since the
        data started: the same if nothing changed, otherwise every folder's base moves to what it has given."""
        if old is None:
            return self.state
        if old.get("block") == self.block and old.get("quota") == dict(zip(self.names, self.quota.tolist())):
            self.state = old
            return old
        names = list(old["quota"])
        q = np.array([old["quota"][n] for n in names], dtype=np.int64)
        _, _, _, prefix = self._pattern(q / q.sum(), old["block"])
        b, r = divmod(samples_done - old["k0"], old["block"])
        given = {n: old["base"][n] + b * int(q[i]) + int(prefix[r][i]) for i, n in enumerate(names)}
        self.state = {"k0": samples_done, "base": {n: given.get(n, 0) for n in self.names}, "block": self.block,
                      "quota": dict(zip(self.names, self.quota.tolist()))}
        return self.state

    def locate(self, k):
        """(folder index, sample index inside that folder) of global sample k."""
        b, r = divmod(k - self.state["k0"], self.block)
        assert b >= 0, f"sample {k} is before the start of these weights ({self.state['k0']})"
        s = int(self._src[r])
        return s, self.state["base"][self.names[s]] + b * int(self.quota[s]) + int(self._local[r])

    def given(self, k):
        """{folder: samples it has given} after the first k samples."""
        b, r = divmod(k - self.state["k0"], self.block)
        return {n: self.state["base"][n] + b * int(self.quota[i]) + int(self._prefix[r][i])
                for i, n in enumerate(self.names)}

    def passes(self, k, split="train"):
        """{folder: passes over its windows} after the first k samples: how much each folder is repeated."""
        return {n: g / self.parts[i].n_windows(split) for i, (n, g) in enumerate(self.given(k).items())}

    def get_batch_at(self, split, batch_size, batch_index, device=None, rank=0, world=1, seed=0,
                     return_doc_ids=False, bos_id=None, offset=0):
        """As `MemmapTokenDataset.get_batch_at`: sample k = offset + (batch_index * world + rank) *
        batch_size + j, taken from its folder's own shuffled pass."""
        T = self.seq_len
        xb = np.empty((batch_size, T), dtype=np.int64)
        yb = np.empty((batch_size, T), dtype=np.int64)
        base = offset + (batch_index * world + rank) * batch_size
        for j in range(batch_size):
            s, i = self.locate(base + j)
            chunk = self.parts[s].window_at(split, i, _mix64(seed ^ _name_key(self.names[s])))
            xb[j] = chunk[:-1]
            yb[j] = chunk[1:]
        return self.parts[0]._to_batch(xb, yb, device, return_doc_ids, bos_id)

    def get_batch(self, split, batch_size, device=None, rng=None, return_doc_ids=False, bos_id=None):
        """Random windows: the folder by weight, then a shard by length and an offset, as in
        `MemmapTokenDataset.get_batch`."""
        r = rng if rng is not None else self.rng
        T = self.seq_len
        xb = np.empty((batch_size, T), dtype=np.int64)
        yb = np.empty((batch_size, T), dtype=np.int64)
        for i in range(batch_size):
            p = self.parts[int(r.choice(len(self.parts), p=self.weights))]
            shards = p.splits[split]
            s = shards[r.choice(len(shards), p=p._w[split])]
            off = int(r.integers(0, len(s) - T - 1))
            chunk = np.asarray(s[off: off + T + 1], dtype=np.int64)
            xb[i] = chunk[:-1]
            yb[i] = chunk[1:]
        return self.parts[0]._to_batch(xb, yb, device, return_doc_ids, bos_id)

    def n_windows(self, split):
        return sum(p.n_windows(split) for p in self.parts)


class Prefetcher:
    """Background-thread prefetch: pulls the NEXT batch (memmap reads + H2D copy) while the GPU is
    busy on the current step, so the GPU never starves on the dataloader. ~16% throughput recovery
    on the memmap loader. Drop-in: pf.next() replaces ds.get_batch(split, bs, device).

    Extra get_batch kwargs (e.g. return_doc_ids=True, bos_id=...) pass through via **batch_kwargs.
    The thread keeps its OWN rng (independent of the main-thread val sampler) and that rng state is
    save/restorable (rng_state/set_rng_state) so a resumed run does not re-see the same windows.

    indexed=True serves `ds.get_batch_at` batches start, start+1, ... instead: the position is the
    number of micro-batches already trained (or `offset`, the samples already consumed by the whole
    run, with start=0), so a resume needs no saved sampler state."""
    def __init__(self, ds, split, batch_size, device, depth=4, seed=0, indexed=False, start=0,
                 rank=0, world=1, offset=0, **batch_kwargs):
        import threading, queue
        self.ds, self.split, self.bs, self.device = ds, split, batch_size, device
        self.batch_kwargs = batch_kwargs
        self.indexed, self._next, self.rank, self.world, self.seed = indexed, start, rank, world, seed
        self.offset = offset
        self._rng = np.random.default_rng(seed)    # independent rng → no race with main-thread val batches
        self._q = queue.Queue(maxsize=depth)
        self._stop = False
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        while not self._stop:
            try:
                if self.indexed:
                    batch = self.ds.get_batch_at(self.split, self.bs, self._next, self.device,
                                                 rank=self.rank, world=self.world, seed=self.seed,
                                                 offset=self.offset, **self.batch_kwargs)
                    self._next += 1
                else:
                    batch = self.ds.get_batch(self.split, self.bs, self.device,
                                              rng=self._rng, **self.batch_kwargs)
                self._q.put(batch)
            except Exception as e:                 # surface loader errors on the main thread
                self._q.put(e); return

    def next(self):
        item = self._q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def rng_state(self):
        return self._rng.bit_generator.state

    def set_rng_state(self, state):
        self._rng.bit_generator.state = state

    def close(self):
        self._stop = True
        try:                                   # drain so a blocked put() unblocks and the worker exits
            while True:
                self._q.get_nowait()
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "data/tokenized"
    ds = MemmapTokenDataset(d, seq_len=512)
    print(f"dtype={ds.np_dtype} vocab={ds.vocab_size} total_tokens={ds.total_tokens:,} "
          f"train_shards={len(ds.splits['train'])} val_shards={len(ds.splits['val'])} "
          f"train_windows~{ds.n_windows('train'):,}")
    try:
        import torch  # noqa
        x, y = ds.get_batch("train", 4)
        assert (y[:, :-1] == x[:, 1:]).all(), "labels must be inputs shifted by 1"
        print(f"batch x={tuple(x.shape)} y={tuple(y.shape)} dtype={x.dtype} max_id={int(x.max())}")
        x2, y2, doc = ds.get_batch("train", 2, return_doc_ids=True, bos_id=1)
        print(f"doc_ids shape={tuple(doc.shape)} dtype={doc.dtype} (monotonic per row: "
              f"{bool((doc[:, 1:] >= doc[:, :-1]).all())})")
        print("shift-by-1 + doc_ids checks: OK")
    except ImportError:
        print("(torch not importable here — batch test skipped)")
