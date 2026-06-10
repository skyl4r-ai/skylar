# Skylar — Project Structure

> ⚠️ **Nota:** questo documento è in parte ASPIRAZIONALE e non riflette esattamente l'albero su disco
> (descrive es. `cli/`, `configs/`, `data/ingestion/`, `model.py`, `train.py` che potrebbero non esistere).
> Gli entry-point reali sono i file `bin.*.py` (es. `training/bin.pretrain.py`, `inference/bin.chat.py`) e i
> moduli importabili in `models/`. Fai riferimento al file listing reale del repo, non solo a questo doc.

```
skylar/
│
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── CLAUDE.md
├── README.md
│
│
│   ┌─────────────────────────────────────────────┐
│   │              CODICE (committato)             │
│   └─────────────────────────────────────────────┘
│
├── models/                            # ── ARCHITETTURE ──
│   ├── __init__.py
│   ├── config.py                      # tutte le config + preset (test, small, medium, ...)
│   ├── decoder.py                     # GPT decoder-only (NanoTransformer)
│   ├── embedder.py                    # futuro: embedding model
│   ├── heads.py                       # LM head, classificazione, pooling
│   └── layers/
│       ├── __init__.py
│       ├── attention.py               # GQA, MHA, Flash Attention
│       ├── ffn.py                     # SwiGLU, MLP
│       ├── norm.py                    # RMSNorm, LayerNorm
│       ├── rope.py                    # RoPE positional encoding
│       └── kv_cache.py               # KV-Cache per inference
│
│
├── data/                              # ── PIPELINE DATI ──
│   ├── __init__.py
│   │
│   ├── ingestion/                     # da dove arrivano i dati
│   │   ├── __init__.py
│   │   ├── extractors.py             # PDF, HTML, JSON, TXT, XML
│   │   ├── hf_loader.py              # HuggingFace streaming (mc4_it, etc.)
│   │   └── mixer.py                  # merge + peso sorgenti multiple
│   │
│   ├── cleaning/                      # come si puliscono
│   │   ├── __init__.py
│   │   ├── normalizer.py             # TextNormalizer (ftfy, NFKC, whitespace)
│   │   ├── pdf_cleaner.py            # header/footer, paragraph merge
│   │   ├── column_merge.py           # EUR-Lex two-column corruption
│   │   ├── prose_filter.py           # chunk scoring, perplexity, word-salad
│   │   ├── repetition.py             # n-gram repetition scorer
│   │   └── pipeline.py               # PII, quality, spam, dedup composable
│   │
│   ├── tokenizer.py                   # train/load BPE tokenizer
│   └── shuffle.py                     # byte-offset global shuffle
│
│
├── training/                          # ── LOOP DI TRAINING ──
│   ├── __init__.py
│   ├── pretrain.py                    # pre-training loop (next-token)
│   ├── sft.py                         # supervised fine-tuning (ChatML)
│   ├── distill.py                     # knowledge distillation (teacher→student)
│   ├── contrastive.py                 # contrastive training (embedding model)
│   ├── dpo.py                         # futuro: Direct Preference Optimization
│   ├── scheduler.py                   # LR scheduler, warmup, cosine decay
│   ├── optimizer.py                   # AdamW config, weight decay groups
│   └── utils.py                       # grad clipping, checkpointing, EMA
│
│
├── eval/                              # ── VALUTAZIONE ──
│   ├── __init__.py
│   ├── perplexity.py                  # perplexity su held-out set
│   ├── generation.py                  # genera + confronta (BLEU, ROUGE, qualitativo)
│   ├── benchmarks.py                  # ITA bench: MMLU-it, HellaSwag-it, ARC-it
│   ├── embedding.py                   # futuro: MTEB, STS, retrieval eval
│   ├── chat_eval.py                   # futuro: MT-Bench style, judge LLM
│   └── report.py                      # genera report markdown/HTML dei risultati
│
│
├── inference/                         # ── GENERAZIONE E SERVING ──
│   ├── __init__.py
│   ├── generate.py                    # sampling, beam search, top-k/p, temperature
│   ├── chat.py                        # ChatML multi-turn, system prompt
│   ├── embed.py                       # futuro: batch encode testi → vettori
│   └── server.py                      # futuro: FastAPI/vLLM serving
│
│
├── cli/                               # ── ENTRY POINT (sottili) ──
│   ├── __init__.py
│   │
│   │   # Data
│   ├── build_corpus.py                # file locali → corpus pulito
│   ├── build_pretrain.py              # HuggingFace → corpus pulito
│   ├── build_mix.py                   # mixer multi-sorgente con pesi
│   ├── train_tokenizer.py             # allena BPE tokenizer
│   │
│   │   # Training
│   ├── train_decoder.py               # pre-training decoder
│   ├── train_sft.py                   # fine-tuning SFT
│   ├── train_distill.py               # distillation teacher→student
│   ├── train_embedder.py              # training embedding model
│   │
│   │   # Eval
│   ├── run_eval.py                    # tutti i benchmark in un comando
│   ├── run_perplexity.py              # perplexity su file specifico
│   │
│   │   # Inference
│   ├── chat.py                        # chat interattiva terminale
│   ├── generate.py                    # genera testo da prompt
│   └── serve.py                       # lancia server API
│
│
├── utils/                             # ── UTILITÀ CONDIVISE ──
│   ├── __init__.py
│   ├── logging.py                     # structlog / rich logging setup
│   ├── io.py                          # load/save checkpoint, safetensors
│   ├── distributed.py                 # multi-GPU, FSDP, DeepSpeed helpers
│   └── chat_format.py                 # ChatML template, loss mask
│
│
├── configs/                           # ── YAML OVERRIDE ──
│   ├── pretrain_small.yaml
│   ├── pretrain_medium.yaml
│   ├── sft_small.yaml
│   ├── distill_qwen_to_small.yaml
│   └── eval_ita.yaml
│
│
├── scripts/                           # ── ONE-OFF / AUTOMAZIONE ──
│   ├── download_eurlex.sh             # scarica PDF da EUR-Lex
│   ├── download_mc4.sh                # pre-download HuggingFace
│   ├── convert_hf.py                  # converti checkpoint → HuggingFace format
│   └── push_to_hub.py                 # upload su HuggingFace Hub
│
│
│   ┌─────────────────────────────────────────────┐
│   │              DATI (in .gitignore)            │
│   └─────────────────────────────────────────────┘
│
├── datasets/
│   ├── raw/                           # file originali non toccati
│   │   ├── eurlex/
│   │   ├── bankitalia/
│   │   ├── gazzetta/
│   │   └── web/
│   ├── processed/                     # output pipeline → .txt puliti
│   └── tokenized/                     # pretokenized.pt
│
├── checkpoints/                       # pesi modello
│   ├── decoder_small/
│   ├── decoder_medium/
│   ├── sft_small/
│   └── embedder_v1/
│
└── logs/                              # training logs, eval results
    ├── wandb/
    └── eval_reports/
```


## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "skylar"
version = "0.1.0"
requires-python = ">=3.10"
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}

[tool.setuptools.packages.find]
include = ["models*", "data*", "training*", "eval*", "inference*", "cli*", "utils*"]
```


## .gitignore

```gitignore
# Dati pesanti
datasets/
checkpoints/
logs/

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Venv
.venv/
venv/

# IDE
.vscode/
.idea/
.DS_Store

# Claude Code personale
CLAUDE.local.md

# Pesi modello
*.pt
*.pth
*.bin
*.safetensors

# Logs
*.log
wandb/
```


## Setup

```bash
cd skylar/
python -m venv .venv
source .venv/bin/activate
pip install -e .
```