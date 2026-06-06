# Skylar `medium_plus` (236M) — Validated Results

**One from-scratch 236M Italian base → generative + dense + sparse + classifier.**
Every number below is reproducible from this repo (scripts named per row). Models compared head-to-head
use the **same pool and the same metric code** (`eval/bench_retrieval.py`), so they are directly comparable
across size and tokenizer. Nothing here is cherry-picked: where Skylar loses, it says so.

- **Architecture:** Qwen3-style decoder (RMSNorm · RoPE · GQA · QK-Norm · SwiGLU · µP), HF-native.
- **Pretrain:** 1.12B tokens of Italian legal/normative text (EUR-Lex, Banca d'Italia, Gazzetta Ufficiale,
  normattiva, TED), 4 epochs, ~19h on one RTX 4090. Val loss **2.16**, health-check perplexity **15.4**.
- **License:** Apache-2.0. Fully local / offline. No data leaves the box.

---

## 1. Generative — public Italian benchmarks

Likelihood-based multiple-choice (the lm-eval method), `eval/bench_ita.py`. Headline = best of acc / acc_norm.

| Task | medium (100M) | **medium_plus (236M)** | random |
|:--|:--:|:--:|:--:|
| XCOPA-it (causal commonsense, 500) | 0.546 | **0.562** ⭐ | 0.50 |
| HellaSwag-it (9193) | 0.279 | 0.292 | 0.25 |
| Belebele-it (900) | 0.244 | 0.267 | 0.25 |

**Read:** real, if modest, signal on Italian causal commonsense (+6 pts over random). On knowledge-heavy
benchmarks it sits near chance — expected: the corpus is narrow legal/normative Italian, not Wikipedia.
This is a **domain model, not a general-knowledge LLM**.

## 2. Generative — chat / grounded behaviour (the intended role)

`eval/validate_grounded.py` (temp 0.3) and `eval/validate_chat.py --lang both` (temp 0.7) on the SFT model.

| Probe | Result |
|:--|:--|
| Answer-from-context | ✅ correct ("…fondata nel 1893, sede a Roma in via Nazionale") |
| Classify → one word | ✅ `credito` |
| Extract → JSON | ✅ `{"importo": "1.500 euro", "scadenza": "31 dicembre 2024"}` |
| Query generation (RAG) | ✅ 3 clean search queries |
| **Refuse when not in context** | ✅ "Non presente nel contesto" |
| Clean stop (`<\|im_end\|>`) | ✅ 6/6 grounded · 12/13 full battery |

- **Italian, open-domain facts:** fluent and grammatical, **but hallucinates** (e.g. wrong Constitution date).
  Not its job — parametric recall is not what a 236M model is for.
- **English:** non-functional (word-salad / switches back to Italian). Skylar is **Italian-only by design**.

**Verdict:** usable, non-garbage **for grounded Italian RAG** (answer/extract/classify/refuse from retrieved
context) — the exact production role. Not usable as an open-domain chatbot or in English.

## 3. Retrieval — vs off-the-shelf multilingual SOTA

`eval/bench_retrieval.py`. The Skylar embedder = the 236M base + a cheap contrastive fine-tune
(InfoNCE, in-batch negatives, same-context false-negative masking) on Italian QA. `bge-m3` and `e5-base`
are evaluated **zero-shot**. Single relevant doc per query.

### 3a. SQuAD-it test — open domain (7609 queries / 1988 contexts)

| Model | Params | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Skylar-embed — base (no retrieval tuning) | 236M | 0.285 | 0.540 | 0.659 | 0.397 | 0.459 |
| **Skylar-embed — IT-QA fine-tune** | **236M** | **0.550** | **0.807** | 0.884 | 0.660 | **0.714** |
| `intfloat/multilingual-e5-base` | 278M | 0.707 | 0.909 | 0.949 | 0.795 | 0.833 |
| `BAAI/bge-m3` | 568M | 0.703 | 0.903 | 0.940 | 0.789 | 0.826 |

### 3b. In-domain banking concepts — fair for all (150 held-out queries / 30 contexts)

| Model | Params | R@1 | nDCG@10 |
|:--|:--:|:--:|:--:|
| Skylar-embed (IT-QA fine-tune) | 236M | 0.927 | 0.967 |
| `intfloat/multilingual-e5-base` | 278M | 0.980 | 0.993 |
| `BAAI/bge-m3` | 568M | 0.987 | 0.995 |

**Read:** a from-scratch 236M Italian model, with a ~30-min contrastive fine-tune, becomes a **functional
retriever** — top-1 in 55% of cases, top-5 in 81% (open domain). It reaches **~78% of the R@1** and
**~86% of the nDCG** of `bge-m3` while being **2.4× smaller** and fully local. It does **not** beat the
multilingual SOTA on accuracy (`e5-base`, same size class, leads). The gap traces to the **narrow
1.12B-token pretrain**, not the contrastive recipe: more negatives (batch 128 → 256) gave **no** gain
(in-batch accuracy already saturated), so the bottleneck is the backbone's lexical coverage, not the head.

## 4. Sparse + classifier (in-domain validation)

Same base, `from_decoder()`, no re-pretrain. `training/bin.sparse.py`, `bin.classify.py`.

| Product | Metric |
|:--|:--|
| Sparse (SPLADE-style) | Recall@1 **1.000** · ~5 non-zeros/query · interpretable lexical weights |
| Classifier | 5-way banking intent accuracy **1.00** (held-out) |

> In-domain synthetic validation — a correctness/plumbing check that the discriminative heads train and
> serve from the shared backbone, not a public-benchmark claim.

---

## What we can declare

1. **A complete from-scratch Italian NLP stack from one 236M base** — generative + dense + sparse +
   classifier — Apache-2.0, fully local/offline.
2. **Usable for grounded Italian RAG**: 6/6 on answer/extract/classify/refuse-from-context, clean stopping.
3. **A functional Italian retriever** at **~85% of `bge-m3`'s nDCG while 2.4× smaller** (SQuAD-it test).
4. Honest limits: **not** a factual oracle, **not** an English model, **does not beat** multilingual SOTA on
   retrieval accuracy. Its edge is **size, locality, transparency, and stack completeness** — not raw scores.

**Publishing focus:** an open, transparent, fully-local **Italian legal/normative domain model + RAG stack**
for on-prem/sovereign deployments — not a general-purpose or SOTA-chasing release.
