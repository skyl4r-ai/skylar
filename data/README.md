# `data/` · from raw documents to token shards

```mermaid
flowchart LR
    raw["raw documents<br/>PDF · HTML · JSON · TXT"] --> build["corpus builder<br/>pretrain-pipeline/"]
    build --> txt["clean corpus<br/>&lt;bos&gt;doc&lt;eos&gt;"]
    txt --> tok["BPE tokenizer<br/>tokenizer.py"]
    tok --> shards["uint16 / uint32 shards<br/>+ pretokenized_meta.json"]
    shards --> loader["memmap loader<br/>memmap_dataset.py"]
    loader --> trainer["training/bin.pretrain.py"]
```

## Tokenizer

Two entry points share the same special tokens:

- [`tokenizer.py`](tokenizer.py) is the library: a code-aware ByteLevel BPE, with digits split one by one
  and uint16 shards whenever the vocabulary fits, which halves memory and disk.
- [`bin.tokenizer.py`](bin.tokenizer.py) is the command line: uint32 shards and optional S3 upload.

```python
from pathlib import Path
from data.tokenizer import train, tokenize_to_shards

tok = train(48128, docs, Path("tokenizer.json"))
records, total, n = tokenize_to_shards(tok, docs, out_dir)   # <bos> + ids + <eos> per document
```

```bash
python data/bin.tokenizer.py --data corpus/ --output .datasets/tokenized --vocab_size 40960
```

**Special tokens:** `<pad>` `<bos>` `<eos>` `<|im_start|>` `<|im_end|>` `<think>` `</think>`
`<tool_call>` `</tool_call>` `<tool_response>` `</tool_response>`.

## Shard format

A tokenized directory holds `tokenizer.json`, `pretokenized_meta.json` (with the `dtype`) and
`shards/shard_000000.bin …`: little-endian uint16 or uint32, documents packed back to back as
`<bos> … <eos>`. `bin.tokenizer.py` also writes a SHA-256 checksum per shard.

[`memmap_dataset.py`](memmap_dataset.py) reads them without loading the corpus into memory:

```python
from data.memmap_dataset import MemmapTokenDataset, Prefetcher
ds = MemmapTokenDataset("data/tokenized", seq_len=2048)
x, y = ds.get_batch_at("train", 16, batch_index=0, device="cuda")   # shuffled pass, no repeats
pf = Prefetcher(ds, "train", 16, "cuda", seed=1234, indexed=True)   # loads while the GPU computes
```

Windows cross document boundaries on purpose. When attention, or a recurrent state, must not cross them,
the loader returns per-position document ids for the mask.

### Packed windows

Consecutive windows cut documents. [`packing.py`](packing.py) packs whole documents into the windows instead, by
best-fit decreasing (Ding et al., [arXiv 2404.10830](https://arxiv.org/abs/2404.10830)). A document that fits a
window is always read whole; a longer one is split into chunks of a window, and the rest of a window is padding.

```bash
python data/bin.pack_index.py --data <folder> --seq_len 8192 --tails keep --check 20   # once per folder, CPU
```

The index (`<folder>/pack/T<seq_len>_<tails>/`, about 10 bytes per document) is a pure function of the shards. The
loader reads it with `MemmapTokenDataset(..., pack="keep")`, and the trainer with `--pack keep`. `--tails drop` leaves
out the chunks after the first one of a document longer than a window, so that no position misses its document's
start.

The windows give per-position document ids from the index, in which every chunk is a document of its own. A target
never crosses a document, a chunk edge or the padding: the trainer sees it as label -100. Packing needs document
masking, so without it the loader refuses; an index built for another `seq_len`, other tails or other shards is
refused too. `eval/bin.gate_packing.py` checks all of it on a synthetic corpus.

### Several folders

`MixtureTokenDataset` reads several tokenized directories as one corpus, with weights chosen when training starts
(`bin.pretrain.py --data_mix "code=0.7,text=0.2,math=0.1"`). A corpus grows by adding a directory, and the mix
changes without tokenizing again. Every directory keeps its own shuffled passes: a weight above its share of tokens
repeats it, with a new order at every pass. The weights can also change when a run resumes, and every directory
continues where it stopped. All directories must carry the same `tokenizer.json`.

Validation comes from each directory's `val_shards` (listed in `pretokenized_meta.json`); without them the loader
takes the last 1% of the shards, at least one. A directory whose meta says `"validation": "none"` trains on every
shard and gives no validation windows.

## Corpus builder

[`pretrain-pipeline/`](pretrain-pipeline/) turns raw documents into a clean, deduplicated, shuffled corpus.
It was built for Italian legal and institutional text: EUR-Lex, Gazzetta Ufficiale, Banca d'Italia.

```bash
python data/pretrain-pipeline/bin.pretrain_builder.py /path/to/raw_documents -o .datasets/pretokenized --max-gb 5
```

```mermaid
flowchart TB
    raw["raw files"] --> ext["1 · extraction<br/>Docling on GPU · trafilatura · JSON"]
    ext --> norm["2 · normalisation<br/>ftfy · NFKC"]
    norm --> col["3 · two-column PDF repair"]
    col --> prose["4 · prose filter<br/>semantic chunks · optional GPU perplexity"]
    prose --> pii["5 · PII redaction<br/>Presidio NER, Italian"]
    pii --> q["6 · quality · spam · exact and near dedup"]
    q --> out["7 · global shuffle · write &lt;bos&gt;doc&lt;eos&gt;"]
    raw -. "curated ChatML .txt:<br/>no filters" .-> out
```

| step | technology | purpose |
|:--|:--|:--|
| PDF extraction | Docling (GPU), pymupdf fallback | layout-aware reading order |
| PII redaction | Microsoft Presidio + spaCy | names → `<PERSONA>`, and `<EMAIL>`, `<TELEFONO>`, `<CF>`, `<IBAN>`, `<CC>` |
| perplexity | `facebook/xglm-564M`, optional (`--no-gpu-perplexity` to skip) | drops incoherent or repetitive text |
| near-dedup | MinHash LSH at 0.80 (datasketch) | removes fuzzy duplicates |

- **Curated ChatML `.txt` files bypass every filter**, so the model sees `<|im_start|>` / `<|im_end|>`
  from the first steps.
- **Consequence:** these files are neither redacted nor deduplicated. Only put text there that is
  already clean.

## Other tools

| file | what it does |
|:--|:--|
| [`bin.sft_data_to_jsonl.py`](bin.sft_data_to_jsonl.py), [`bin.sft_data_shaffle.py`](bin.sft_data_shaffle.py) | build and shuffle SFT JSONL |
| [`ted-processor/`](ted-processor/) | downloads and parses TED Europa tenders, 1993 to 2026, into an Italian corpus |
| [`bin.shard_dgt_data.py`](bin.shard_dgt_data.py) | splits one very large legal text file into shards |
