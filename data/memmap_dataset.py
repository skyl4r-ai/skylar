"""
Reusable memmap token dataset + background prefetcher — the streaming data layer.

Memmaps sharded little-endian token files instead of loading the whole corpus into one
RAM tensor: O(1) RAM regardless of corpus size, random-window next-token sampling,
train/val split, long-context ready. This is the piece the framework was missing — the
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
    x, y = ds.get_batch("train", batch_size=16, device="cuda")          # (B,T) int64
    pf = Prefetcher(ds, "train", 16, "cuda", seed=1234)                  # overlap load+compute
    x, y = pf.next()
"""
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

    def get_batch(self, split, batch_size, device=None, rng=None,
                  return_doc_ids=False, bos_id=None):
        """Random (B, seq_len) next-token windows. Returns int64 (x, y) — or (x, y, doc_ids)
        when return_doc_ids=True and bos_id is given (doc_ids[b,t] = #<bos> seen up to t,
        i.e. a per-position document index for document-masked attention)."""
        import torch
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
        """Approx number of non-overlapping seq_len windows in a split."""
        return sum(len(s) // self.seq_len for s in self.splits[split])


class Prefetcher:
    """Background-thread prefetch: pulls the NEXT batch (memmap reads + H2D copy) while the GPU is
    busy on the current step, so the GPU never starves on the dataloader. ~16% throughput recovery
    on the memmap loader. Drop-in: pf.next() replaces ds.get_batch(split, bs, device).

    Extra get_batch kwargs (e.g. return_doc_ids=True, bos_id=...) pass through via **batch_kwargs.
    The thread keeps its OWN rng (independent of the main-thread val sampler) and that rng state is
    save/restorable (rng_state/set_rng_state) so a resumed run does not re-see the same windows."""
    def __init__(self, ds, split, batch_size, device, depth=4, seed=0, **batch_kwargs):
        import threading, queue
        self.ds, self.split, self.bs, self.device = ds, split, batch_size, device
        self.batch_kwargs = batch_kwargs
        self._rng = np.random.default_rng(seed)    # independent rng → no race with main-thread val batches
        self._q = queue.Queue(maxsize=depth)
        self._stop = False
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        while not self._stop:
            try:
                self._q.put(self.ds.get_batch(self.split, self.bs, self.device,
                                              rng=self._rng, **self.batch_kwargs))
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
