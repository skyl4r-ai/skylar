"""
Reusable tokenizer library — train a (code-aware) ByteLevel BPE and tokenize a corpus
into little-endian uint16/uint32 memmap shards. This is the importable, code-model-ready
primitive the framework was missing: the `bin.tokenizer.py` entry-point is the IT-legal
recipe (uint32, LiteToken pruning, S3); THIS module is the small generic core shared by
projects (and destined for the pip package so a dev can build a model from scratch).

Code-aware touches (StarCoder/CodeLlama-style), toggleable:
  - digits split (individual_digits) so numeric literals / PIC 9(7) don't waste vocab
    and the model learns arithmetic digit-by-digit;
  - vocab that fits uint16 -> half the RAM/disk of uint32, which is what the streaming
    memmap dataloader wants. dtype is selectable (uint16|uint32).

The shards written here are read back by `data.memmap_dataset` (dtype auto-detected from
pretokenized_meta.json). Special tokens stay ChatML/SFT-compatible.

  from data.tokenizer import SPECIAL_TOKENS, train, tokenize_to_shards, quality
  tok = train(48128, docs, Path("tokenizer.json"))            # code-aware, uint16-ready
  records, total, n = tokenize_to_shards(tok, docs, out_dir)  # <bos>+ids+<eos> per doc
"""
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>",
    "<|im_start|>", "<|im_end|>",
    "<think>", "</think>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
]

# meta["dtype"] string -> little-endian numpy dtype (mirrors data.memmap_dataset)
_DTYPE = {"uint16": "<u2", "uint32": "<u4"}


def train(vocab_size, docs, save_path, *, digits_split=True, max_token_length=64,
          min_frequency=2, add_bos_eos_processor=True, show_progress=False):
    """Train a ByteLevel BPE on `docs`. digits_split=True prepends a Digits pre-tokenizer
    (code-aware); ByteLevel preserves whitespace/indent exactly. Saves to save_path and
    returns the Tokenizer. Deterministic given the same corpus + params."""
    tok = Tokenizer(models.BPE())
    pre = [pre_tokenizers.Digits(individual_digits=True)] if digits_split else []
    pre.append(pre_tokenizers.ByteLevel(add_prefix_space=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence(pre)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=max_token_length,
        show_progress=show_progress,
    )
    tok.train_from_iterator(docs, trainer=trainer)
    tok.decoder = decoders.ByteLevel()
    if add_bos_eos_processor:
        bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
        tok.post_processor = processors.TemplateProcessing(
            single="<bos>:0 $A:0 <eos>:0",
            special_tokens=[("<bos>", bos), ("<eos>", eos)],
        )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(save_path))
    return tok


def verify_roundtrip(tok, docs_sample, *, limit=256):
    """Hard lossless check: decode(encode(x)) == x byte-for-byte over up to `limit` docs.
    Returns {n, ok, fail, rate, mismatches:[(idx, first_diff_char)]}. This is the check the
    inline build_mix/build_tokenizer round-trip only *printed* (and only on the first ~30
    chars) — here it's exact and reusable, so a caller can assert on it."""
    n = ok = 0
    mism = []
    for i, d in enumerate(docs_sample):
        if i >= limit:
            break
        n += 1
        # skip_special_tokens=False: gli special-token (ChatML/think/tool) sono CONTENUTO
        # del documento nel pretrain -> devono sopravvivere al round-trip, non sparire.
        rt = tok.decode(tok.encode(d, add_special_tokens=False).ids, skip_special_tokens=False)
        if rt == d:
            ok += 1
        elif len(mism) < 5:
            j = next((k for k in range(min(len(rt), len(d))) if rt[k] != d[k]), min(len(rt), len(d)))
            mism.append((i, j))
    return {"n": n, "ok": ok, "fail": n - ok, "rate": round(ok / max(n, 1), 4), "mismatches": mism}


def tokenize_to_shards(tok, docs, out_dir, shard_tokens=8_000_000, start_idx=0,
                       enc_batch=4000, *, dtype="uint16", verify_sample=0):
    """Write <bos>+ids+<eos> per doc into little-endian shards of the chosen dtype
    (uint16|uint32), indices from start_idx. Uses tok.encode_batch (Rust, multi-threaded)
    — ~Ncores faster than per-doc encode, which matters at the ~8B-token mix scale.
    Returns (records, total_tokens, n_docs). NOTE: sets tok.post_processor = None (bos/eos
    added manually here) — retrain/reload the tokenizer if you need the processor after.

    Correctness guards (default off to keep the generic path byte-identical):
      - dtype uint16 requires vocab <= 65536, else ids would be silently truncated -> raise;
      - verify_sample>0 runs verify_roundtrip on the first N docs and raises if not lossless."""
    if dtype not in _DTYPE:
        raise ValueError(f"dtype must be one of {list(_DTYPE)}, got {dtype!r}")
    if dtype == "uint16" and tok.get_vocab_size() > 65536:
        raise ValueError(f"vocab {tok.get_vocab_size()} > 65536 does not fit uint16 "
                         f"(ids would be truncated) — use dtype='uint32'")
    np_dtype = _DTYPE[dtype]
    out_dir = Path(out_dir)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    tok.post_processor = None  # we add bos/eos manually
    if verify_sample > 0:  # peek the first N docs, assert lossless, then chain them back
        import itertools
        head = list(itertools.islice(docs, verify_sample))
        vr = verify_roundtrip(tok, head, limit=verify_sample)
        if vr["fail"]:
            raise ValueError(f"round-trip NOT lossless: {vr['fail']}/{vr['n']} docs "
                             f"(rate {vr['rate']}), first mismatches {vr['mismatches']}")
        docs = itertools.chain(head, docs)
    records, buf, total, idx, n_docs = [], [], 0, start_idx, 0

    def flush():
        nonlocal buf, idx
        if not buf:
            return
        arr = np.asarray(buf, dtype=np_dtype)
        fn = f"shard_{idx:06d}.bin"
        (shards_dir / fn).write_bytes(arr.tobytes())
        records.append({"index": idx, "filename": fn, "num_tokens": len(buf)})
        idx += 1; buf = []

    docbuf = []

    def encode_flush():
        nonlocal total, n_docs
        if not docbuf:
            return
        for e in tok.encode_batch(docbuf, add_special_tokens=False):
            buf.append(bos); buf.extend(e.ids); buf.append(eos)
            total += len(e.ids) + 2; n_docs += 1
        docbuf.clear()
        if len(buf) >= shard_tokens:
            flush()

    for d in docs:
        docbuf.append(d)
        if len(docbuf) >= enc_batch:
            encode_flush()
    encode_flush()
    flush()
    return records, total, n_docs


def quality(tok, docs_sample):
    """bytes/token + vocab utilization over a sample of docs."""
    nb = nt = 0
    seen = set()
    for d in docs_sample:
        nb += len(d.encode("utf-8"))
        ids = tok.encode(d, add_special_tokens=False).ids
        nt += len(ids); seen.update(ids)
    return {"bytes_per_token": round(nb / max(nt, 1), 2),
            "vocab_utilization": round(len(seen) / tok.get_vocab_size(), 4),
            "unique_seen": len(seen)}
