# Post-training suite — Skylar (medium_plus 236M)

Infrastruttura costruita per portare il base `medium_plus` a prodotto. Tutto
**locale, deterministico, zero API**. Tre verticali pronti + due estensioni
architetturali pianificate (sparse / YaRN) da applicare a pretrain finito.

Convenzione: i `bin.<name>.py` sono entry-point (si lanciano, non si importano).
Sempre `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `.venv/bin/python`.

---

## 1. Generativo: SFT → preferenze (ORPO/SimPO)

**SFT** (già rodato su medium): `training/bin.sft.py`, mix `.datasets/sft/sft_mp_mix.jsonl`
(38535: chat×1 + grounded×2 + JSON-distill×1 + task×2; query_gen stutter già fixato).

**Preferenze reference-free** (NUOVO, `training/bin.preference.py`): ORPO (default) o SimPO.
Niente reference model congelato (≠ DPO) → metà memoria, ideale per modelli piccoli.
- Dati: `data/gen_preference_data.py` → `.datasets/pref/preference_it.jsonl` (2451 coppie).
  `chosen` = grounded/format-corretto, `rejected` = i fallimenti reali (allucinazione,
  JSON-dump al posto di 1 parola, niente rifiuto, ripetizione degenere).
- Maschera risposta = `utils.chatML.create_loss_mask` (identica a SFT → compone pulito).
```
.venv/bin/python training/bin.preference.py \
  --base_model checkpoints_sft/skylar-mp-chat/best \
  --data .datasets/pref/preference_it.jsonl --loss orpo --epochs 1 --lr 8e-6 --bf16 \
  --out_dir checkpoints_pref/skylar-mp-orpo
```
Diagnostica: `margin` (logp_chosen − logp_rejected, deve salire) e `pref_acc` (deve → 1).

## 2. Embeddings: encoder dal decoder (LLM2Vec/E5-Mistral)

`models/embedder.py SkylarEmbedder` (mean/cls/last pool, L2-norm); `from_decoder()` copia il tronco del decoder
pretrained. **Dal 10/10/2026** embedder, sparse e classificatore condividono `models/encoder_base.py`, cioè il tronco
stesso del decoder: bidirezionali su un modello denso (come il 236M, invariato bit per bit), causali con l'ultimo token
`<eos>` su Skylar 2 ibrido, i cui strati KDA leggono solo da sinistra a destra. Controlli: `eval/bin.gate_encoders.py`.
- **Trainer NUOVO** `training/bin.contrastive.py`: InfoNCE + in-batch negatives, temperatura 0.05.
- Dati: `data/gen_contrastive_data.py` → `.datasets/embed/contrastive_it.jsonl` (3600 coppie
  query↔passaggio su 30 concetti legali/bancari con definizioni reali).
```
.venv/bin/python training/bin.contrastive.py \
  --base_model checkpoints/skylar-mp-base/final \
  --data .datasets/embed/contrastive_it.jsonl --epochs 3 --bf16 --batch_size 128 \
  --out_dir checkpoints_embed/skylar-mp-embed
```
- CLI: `inference/bin.embed.py` (similarità query↔doc). Eval: `eval/eval_embeddings.py`
  (Recall@1/@5/MRR su query in frasi NON viste in train → misura generalizzazione).
- **Contrastive serve SOLO all'embedder** — NON al generativo né alla preferenza (segnali diversi).
- Miglioria futura (LLM2Vec): breve step MNTP (masked-next-token bidirezionale) prima del contrastive.

### 2b. Embeddings SPARSE (SPLADE / BGE-M3) — l'altra metà del hybrid retrieval
Stesso base, NESSUN re-pretrain: lo sparse proietta gli hidden state sul VOCABOLARIO (riusa la
tied token-emb = lm_head che il pretrain già allena) → `log(1+ReLU)` + max-pool → vettore sparso
sul vocab (per lo più zeri, peso solo sui termini rilevanti). Interpretabile (le dim = token).
- Modello NUOVO `models/sparse_encoder.py` `SkylarSparseEncoder` (+ `from_decoder`, `top_terms`, `flops_regularizer`).
- Trainer NUOVO `training/bin.sparse.py`: InfoNCE su DOT-product (non coseno) + FLOPS reg (rampa quadratica) → sparsità. Stesse coppie `contrastive_it.jsonl`. Diagnostica: `acc` + `L0` (nnz/query, deve crollare).
```
.venv/bin/python training/bin.sparse.py --base_model checkpoints/skylar-mp-base/final \
  --data .datasets/embed/contrastive_it.jsonl --epochs 3 --bf16 --batch_size 96 \
  --out_dir checkpoints_embed/skylar-mp-sparse
```
- **dense + sparse dallo STESSO peso 236M** = stack hybrid stile BGE-M3, nativo in Qdrant.
- Nota: lo smoke su modello test random non è sparso (L0 alto) — sparsità/semantica emergono col backbone pretrained + training pieno. Se early-instabile (CE esplode pre-sparsità), aggiungere un floor a `reg_scale` o scalare gli score.

## 3. Benchmark pubblici italiani (metrica pubblicabile)

`eval/bench_ita.py` (likelihood-based MC, acc + acc_norm, stile lm-eval, prepende `<bos>`):
- `xcopa_it` (causale, random 50%), `hellaswag_it` (random 25%), `belebele_it` (random 25%).
- Run a pretrain finito su `checkpoints/skylar-mp-base/final` + confronto vs medium 100M.
```
.venv/bin/python eval/bench_ita.py --model checkpoints/skylar-mp-base/final \
  --tasks xcopa_it,hellaswag_it,belebele_it
```

---

## Estensioni architetturali — DA FARE A PRETRAIN FINITO (toccano file live)

> Regola di sicurezza: `decoder.py / attention.py / rope.py / config.py / block.py` sono
> in RAM nel processo di pretrain; editarli ora rischia un resume rotto. Si applicano DOPO.

### A. Sparse variant (dense + sparse per coprire mercato) — Sliding Window Attention
Tecnica utile *oggi* (Mistral/Gemma2), non kernel di ricerca. L'infra FlexAttention c'è già
(`create_document_block_mask` con `mask_mod`). Implementazione:
1. `config.py`: nuovi campi `attention_type="dense"` (default, retro-compatibile) e
   `sliding_window=None`; nuovo preset `medium_plus_sparse` = medium_plus + `sliding_window=1024`.
2. `attention.py`: nel mask_mod aggiungere `windowed = (q_idx - kv_idx) < window`, combinato con
   causale (+ same_doc se packing). Path FlexAttention → block-sparse efficiente. Fallback SDPA:
   maschera causale + banda finestra. Default-off riproduce ESATTAMENTE il comportamento attuale.
3. **Il modello sparse va PRE-ALLENATO sparse** (non basta runtime): è una 2ª run pretrain con
   `--preset medium_plus_sparse`. NB: a seq 2048 lo sliding window rende poco (finestra ~1024);
   il valore vero è a contesto lungo (8K+). → ha senso SOLO se accoppiato a contesto esteso.
   Per "dense + sparse market split" a 2K il guadagno è marginale; valutare se vale la 2ª run.

### B. YaRN / contesto esteso — SOLO se serve long-context
`rope.py` ora è RoPE vanilla (cache lazy, no scaling). `rope_theta=1e6` già dà margine.
Per 8K-16K *reali* serve YaRN (NTK-by-parts) + breve adattamento long-context. Per la RAG
attuale (chunk < 512) **non serve oggi** → rimandato. Implementazione YaRN ≈ 30 righe in rope.py
(scaling di `inv_freq` per banda di frequenza + fattore di attenzione), attivabile da config.

---

## Ordine di esecuzione autonomo (a pretrain finito ~01:30)
1. Gate `eval/bin.eval_base_model.py` su mp-base/final.
2. **Bench ITA** su mp-base + confronto vs medium 100M (writeup).
3. **SFT** (`sft_mp_mix.jsonl`) → `validate_grounded` + `validate_chat` + test empirico IT/EN.
4. **ORPO** sul chat → ri-valida (margine/pref_acc + grounded/chat).
5. **Contrastive embedder** da mp-base/final → `eval_embeddings` (Recall@k).
6. (Opz.) Sparse + YaRN se decidiamo di coprire il long-context.
