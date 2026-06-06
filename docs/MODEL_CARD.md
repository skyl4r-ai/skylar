# Skylar — Model Family Card & Benchmarks

**From-scratch Italian LLM stack.** One 236M base (Qwen3-style decoder: RMSNorm · RoPE · GQA · QK-Norm ·
SwiGLU · µP, HF-native), pretrained on **1.12B tokens** of Italian legal/normative text (EUR-Lex, Banca
d'Italia, Gazzetta Ufficiale, normattiva, TED) — then turned into **generative + dense + sparse + classifier**
products via `from_decoder()`, **no re-pretrain**. Apache-2.0. Runs **fully local / offline**.

> **Read this first — honest scope.** Skylar is a **domain specialist for grounded Italian RAG**, *not* a
> general-purpose assistant. It is **Italian-only** (English is not fluent) and it is **undertrained by
> Chinchilla** (1.12B tokens for 236M params ≈ 5× below compute-optimal). It does **not** beat multilingual
> SOTA retrievers on accuracy. Where it loses, this card says so. Published claims must match this card.

---

## The family at a glance

| Model | Params | Type | Trained on | Headline result | HF-ready? |
|:--|:--:|:--|:--|:--|:--:|
| **Skylar-236M-Base** | 236M | decoder LM | 1.12B tok IT legal | val ppl **15.4** · XCOPA-it **0.562** | ✅ *domain/research* |
| **Skylar-236M-Chat** | 236M | SFT (ChatML) | grounded IT SFT | grounded tasks **6/6** · clean stop | ✅ *grounded RAG* |
| **Skylar-Embed-236M** | 236M | dense retriever | + IT-QA contrastive | SQuAD-it R@1 **0.55** · nDCG **0.71** | ✅ *functional* |
| **Skylar-Sparse-236M** | 236M | SPLADE sparse | + 30-concept synth | in-domain R@1 1.0 (synthetic) | ⚠️ *demo only* |
| **Skylar-Intent-236M** | 236M | classifier | + 328 synth rows | 5-way acc 1.0 (synthetic) | ⚠️ *example only* |
| Skylar-107M-Base/Chat | 107M | decoder LM/SFT | 1.12B tok IT legal | XCOPA-it 0.546 | ➖ *superseded* |

Baselines used for comparison (zero-shot, off-the-shelf): `BAAI/bge-m3` (568M), `intfloat/multilingual-e5-base` (278M).

---

## 1. Generative — public Italian benchmarks

Likelihood-based multiple-choice (lm-eval method), `eval/bench_ita.py`. Headline = best of acc / acc_norm.

| Task (n) | Skylar-107M | **Skylar-236M** | random |
|:--|:--:|:--:|:--:|
| XCOPA-it — causal commonsense (500) | 0.546 | **0.562** ⭐ | 0.50 |
| HellaSwag-it — completion (9193) | 0.279 | 0.292 | 0.25 |
| Belebele-it — reading comprehension (900) | 0.244 | 0.267 | 0.25 |

Real (if modest) signal on Italian causal commonsense; near-chance on knowledge-heavy tasks — expected for a
236M model trained on a narrow legal corpus. **A domain model, not a knowledge oracle.**

## 2. Generative — grounded / chat behaviour (the intended role)

`eval/validate_grounded.py` (temp 0.3) · `eval/validate_chat.py --lang both` (temp 0.7), on Skylar-236M-Chat.

| Probe | Result |
|:--|:--|
| Answer-from-context | ✅ correct |
| Classify → one word | ✅ `credito` |
| Extract → JSON | ✅ `{"importo": "1.500 euro", "scadenza": "31 dicembre 2024"}` |
| Query generation (RAG) | ✅ 3 clean search queries |
| Refuse when not in context | ✅ "Non presente nel contesto" |
| Clean stop (`<\|im_end\|>`) | ✅ 6/6 grounded · 12/13 full battery |
| Open-domain facts (IT) | ⚠️ fluent **but hallucinates** |
| English | ❌ non-functional (Italian-only) |

**Usable for grounded Italian RAG.** Not an open-domain chatbot, not an English model.

## 3. Retrieval — vs off-the-shelf multilingual SOTA

`eval/bench_retrieval.py` — identical pool & metric code for every model. Skylar-Embed = the 236M base + a
~30-min contrastive fine-tune (InfoNCE, in-batch negatives, same-context false-negative masking) on Italian
QA. `bge-m3` / `e5-base` are **zero-shot**. One relevant doc per query.

### SQuAD-it test — open domain (7609 queries / 1988 contexts)

| Model | Params | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Skylar-Embed — base (no retrieval tuning) | 236M | 0.285 | 0.540 | 0.659 | 0.397 | 0.459 |
| **Skylar-Embed — IT-QA fine-tune** | **236M** | **0.550** | **0.807** | 0.884 | 0.660 | **0.714** |
| `intfloat/multilingual-e5-base` | 278M | 0.707 | 0.909 | 0.949 | 0.795 | 0.833 |
| `BAAI/bge-m3` | 568M | 0.703 | 0.903 | 0.940 | 0.789 | 0.826 |

### In-domain banking concepts — fair for all (150 held-out queries / 30 contexts)

| Model | Params | R@1 | nDCG@10 |
|:--|:--:|:--:|:--:|
| Skylar-Embed (IT-QA fine-tune) | 236M | 0.927 | 0.967 |
| `intfloat/multilingual-e5-base` | 278M | 0.980 | 0.993 |
| `BAAI/bge-m3` | 568M | 0.987 | 0.995 |

**The from-scratch 236M retriever reaches ~78% of bge-m3's R@1 and ~86% of its nDCG, at 2.4× fewer params and
fully local.** It does **not** beat the multilingual SOTA on accuracy. The ceiling (~0.55 R@1) is the
backbone, not the recipe: doubling in-batch negatives (128→256) gave **no** gain (in-batch acc already ~0.99).

## 4. Sparse + classifier — in-domain validation only

Same base, `from_decoder()`. **These are plumbing/demos, not benchmarked artifacts** — trained on small
synthetic in-domain data, no public-benchmark or vs-SOTA comparison yet.

| Product | Metric (in-domain, synthetic) |
|:--|:--|
| Sparse (SPLADE-style) | Recall@1 1.000 · ~5 non-zeros/query · interpretable lexical weights |
| Classifier | 5-way banking intent accuracy 1.00 |

---

## Reproduce

```bash
# generative public benchmarks
python eval/bench_ita.py --model checkpoints/skylar-mp-base/final --tasks xcopa_it,hellaswag_it,belebele_it
# grounded + chat (IT/EN)
python eval/validate_grounded.py --model checkpoints_sft/skylar-mp-chat/best
python eval/validate_chat.py     --model checkpoints_sft/skylar-mp-chat/best --lang both
# retrieval vs bge-m3 / e5  (same pool, same metrics)
python eval/bench_retrieval.py --dataset squad_it \
    --models "skylar=checkpoints_embed/skylar-mp-embed-squad,e5=intfloat/multilingual-e5-base,bge-m3"
```

## Bottom line

A complete, transparent, **fully-local Italian legal/normative RAG stack from a single 236M base** —
generative + dense retrieval, both **honestly benchmarked**, plus experimental sparse/classifier heads.
Its edge is **size, locality, transparency and stack completeness**, not raw scores. Apache-2.0,
© A. Ivanovitch. Deep methodology in [`RESULTS.md`](RESULTS.md).
