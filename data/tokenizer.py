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
import errno
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


class OutOfSpace(RuntimeError):
    """Il filesystem di destinazione ha rifiutato una scrittura (ENOSPC/EDQUOT).

    Non e' un errore di programmazione ma un ARRESTO ORDINATO, e sostituisce la morte a meta'
    shard: chi la cattura trova su disco un checkpoint valido e riparte da li'. Su volumi di
    rete con quota (MooseFS) e' l'unico modo affidabile di accorgersene — `df` mostra il
    filesystem condiviso sottostante (petabyte liberi) e non la quota del proprio volume."""

# malloc_trim(0) — restituisce al SO la memoria liberata dall'allocatore. Serve a tenere la RSS
# piatta durante tokenizzazioni lunghe multi-thread (vedi tokenize_to_shards.flush). Assente
# fuori da glibc (musl/macOS): in quel caso resta None e non si fa nulla.
try:
    import ctypes as _ct
    _malloc_trim = _ct.CDLL("libc.so.6").malloc_trim
    _malloc_trim.argtypes = [_ct.c_size_t]
except Exception:                                    # pragma: no cover
    _malloc_trim = None

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
                       enc_batch=4000, *, dtype="uint16", verify_sample=0,
                       enc_chars=16_000_000, max_doc_chars=4_000_000,
                       skip_docs=0, checkpoint_every=0, on_checkpoint=None):
    """Write <bos>+ids+<eos> per doc into little-endian shards of the chosen dtype
    (uint16|uint32), indices from start_idx. Uses tok.encode_batch (Rust, multi-threaded)
    — ~Ncores faster than per-doc encode, which matters at the ~8B-token mix scale.
    Returns (records, total_tokens, n_docs). NOTE: sets tok.post_processor = None (bos/eos
    added manually here) — retrain/reload the tokenizer if you need the processor after.

    Correctness guards (default off to keep the generic path byte-identical):
      - dtype uint16 requires vocab <= 65536, else ids would be silently truncated -> raise;
      - verify_sample>0 runs verify_roundtrip on the first N docs and raises if not lossless.

    Ripresa dopo un'interruzione (`skip_docs` + `checkpoint_every`/`on_checkpoint`):
      una tokenizzazione da centinaia di miliardi di token dura giorni, e senza stato QUALSIASI
      inciampo (quota, kill, riavvio della macchina) butta via tutto il lavoro. `on_checkpoint`
      viene chiamata ogni `checkpoint_every` shard con (records, total_tokens, n_docs_letti);
      chi la riceve salva quel numero e al rilancio lo ripassa come `skip_docs`.
      Il conto torna solo se `docs` e' DETERMINISTICO (stessa sorgente, stesso seed): la ripresa
      riesegue la stessa sequenza e scarta cio' che e' gia' su disco, quindi zero duplicati e
      zero buchi. Lo scarto costa la rilettura, non la tokenizzazione (la parte cara).
      Il checkpoint e' preso solo in un istante SINCRONIZZATO — batch di encoding svuotato e
      shard chiuso — perche' e' l'unico in cui "documenti letti" e "documenti su disco"
      coincidono; altrove il numero salvato sarebbe in anticipo e la ripresa perderebbe dati."""
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
    docs = iter(docs)
    for _ in range(skip_docs):          # ripresa: scarta cio' che e' gia' su disco
        if next(docs, None) is None:
            break
    if verify_sample > 0:  # peek the first N docs, assert lossless, then chain them back
        import itertools
        head = list(itertools.islice(docs, verify_sample))
        vr = verify_roundtrip(tok, head, limit=verify_sample)
        if vr["fail"]:
            raise ValueError(f"round-trip NOT lossless: {vr['fail']}/{vr['n']} docs "
                             f"(rate {vr['rate']}), first mismatches {vr['mismatches']}")
        docs = itertools.chain(head, docs)
    records, buf, total, idx, n_docs, n_split = [], [], 0, start_idx, 0, 0
    n_in = skip_docs                    # documenti LETTI dal generatore (l'ancora della ripresa)

    def flush():
        nonlocal buf, idx
        if not buf:
            return
        arr = np.asarray(buf, dtype=np_dtype)
        fn = f"shard_{idx:06d}.bin"
        try:
            (shards_dir / fn).write_bytes(arr.tobytes())
        except OSError as e:
            # Uno shard scritto a meta' e' peggio di uno assente: ha una taglia plausibile e
            # verrebbe letto come dati buoni. Si toglie di mezzo prima di propagare l'errore.
            (shards_dir / fn).unlink(missing_ok=True)
            if e.errno in (errno.ENOSPC, errno.EDQUOT):
                raise OutOfSpace(f"spazio esaurito scrivendo {fn} "
                                 f"({len(buf):,} token, shard #{idx})") from e
            raise
        records.append({"index": idx, "filename": fn, "num_tokens": len(buf)})
        idx += 1; buf = []
        # RSS: encode_batch alloca da molte arene glibc (una per thread) e la memoria liberata
        # NON torna al SO da sola -> su run lunghi la RSS cresce monotona fino all'OOM (misurato:
        # 64GB dopo 4h43 su un mix da 300B token). malloc_trim la restituisce a ogni shard.
        # Preferito a MALLOC_ARENA_MAX, che limita le arene ma serializza i thread (-90% throughput).
        if _malloc_trim is not None:
            _malloc_trim(0)

    docbuf, docchars = [], 0

    def encode_flush():
        nonlocal total, n_docs, docchars
        if not docbuf:
            return
        for e in tok.encode_batch(docbuf, add_special_tokens=False):
            buf.append(bos); buf.extend(e.ids); buf.append(eos)
            total += len(e.ids) + 2; n_docs += 1
            # Lo scarico va DENTRO il ciclo. Con il controllo dopo il ciclo, `buf` accumulava
            # l'INTERO batch prima di scrivere: un batch di documenti lunghi lo faceva esplodere
            # (una lista Python costa ~36 byte per token, non 2). Misurato su un mix da 300B:
            # RSS piatta a 5 GB per 5 ore, poi 5 -> 57 GB in 4 minuti e OOM.
            if len(buf) >= shard_tokens:
                flush()
        docbuf.clear()
        docchars = 0

    def pieces(d):
        """Spezza i documenti mostruosi. I tetti sul LOTTO non bastano: un documento entra
        sempre INTERO (`buf.extend(e.ids)` non si interrompe a meta'), quindi uno solo da 1 GB
        vale ~250M token in una lista Python (~9 GB) piu' l'oggetto Encoding lato Rust — decine
        di GB allocati in pochi minuti, senza altro I/O. E' stata la causa di quattro OOM
        consecutivi, sempre allo stesso punto del mix; negli shard scritti si vedono documenti
        da 3,9M token e il colpevole (non scritto) era piu' grande.
        Tagliare non altera l'addestramento: il contesto del modello e' 8192 token, quindi un
        documento da un milione di token e' gia' spezzato in centinaia di finestre a valle."""
        nonlocal n_split
        if len(d) <= max_doc_chars:
            yield d
            return
        n_split += 1
        for i in range(0, len(d), max_doc_chars):
            yield d[i:i + max_doc_chars]

    def sync_checkpoint():
        """Porta lettura e scrittura allo stesso punto — svuota il batch di encoding e chiude
        lo shard corrente — e solo allora annuncia lo stato. Fuori da questo istante `n_in`
        conterebbe anche documenti letti ma ancora in volo, e riprendere da li' li perderebbe."""
        encode_flush()
        flush()
        if on_checkpoint is not None:
            on_checkpoint(records, total, n_in)

    next_ckpt = checkpoint_every
    for d0 in docs:
        # Il controllo sta PRIMA dell'incremento: qui `n_in` conta i documenti gia' processati
        # e d0 non lo e' ancora. Incrementando prima, il checkpoint annuncerebbe un documento
        # che non e' su disco e la ripresa lo salterebbe.
        if checkpoint_every and (idx - start_idx) >= next_ckpt:
            sync_checkpoint()
            next_ckpt = (idx - start_idx) + checkpoint_every
        n_in += 1
        for d in pieces(d0):
            docbuf.append(d)
            docchars += len(d)
            # Il batch si chiude a `enc_batch` documenti OPPURE a `enc_chars` caratteri, quale dei
            # due arriva prima. Contare solo i documenti lega il consumo di memoria alla loro taglia
            # (che varia di ordini di grandezza tra un file di codice e una conversazione): il tetto
            # in caratteri rende il picco costante e indipendente dai dati.
            if len(docbuf) >= enc_batch or docchars >= enc_chars:
                encode_flush()
    encode_flush()
    flush()
    if on_checkpoint is not None:       # stato finale: l'ultima parola spetta comunque al meta
        on_checkpoint(records, total, n_in)
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
