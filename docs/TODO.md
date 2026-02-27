# Training Roadmap — TODO

---

## ✅ 1. µP (Maximal Update Parameterization) — COMPLETATO

Gli HP ottimali del modello piccolo (proxy) trasferiscono esattamente al modello grande. Non serve re-tunare lr, init, ecc. quando si scala da 107M → 1B → 4B.

- **Ref:** paper u-µP (ICLR 2025 Spotlight)
- **Chi lo usa:** Cerebras (BTLM-3B), EleutherAI, DeepSeek
- **Implementazione:** µP scaling rules in `config.py` e `model.py` (per-layer lr, init scaling, attention logit scaling)

---

## ✅ 2. WSD Learning Rate Schedule — COMPLETATO

Warmup → stabile ad alto lr → decay solo alla fine (ultimo 10-20%). Si può fermare in qualsiasi momento durante la fase stabile e fare il decay a qualsiasi punto.

- **Ref:** arxiv 2602.06797 (feb 2026), paper "Pre-training LLM without Learning Rate Decay Enhances SFT" (OpenReview 2025)
- **Chi lo usa:** DeepSeek-V3, MiniCPM, OLMo 2, ERNIE 4.5, GLM-4.5
- **Implementazione:** `--lr_schedule cosine|wsd` in `train.py`, funzione `get_lr()`

---

## ⬜ 3. Muon Optimizer — PRIORITÀ MEDIA

Ottimizzatore con ortogonalizzazione del gradiente via Newton-Schulz. Ogni update dei weight è una matrice ortonormale → step spectralmente controllati.

- **Ref:** "Muon is Scalable for LLM Training" (feb 2025)
- **Chi lo usa:** GLM-4.5
- **Impatto:** convergenza potenzialmente più veloce a parità di compute, ma meno maturo di AdamW
- **Azione:** backlog, da valutare come esperimento futuro

---

## ⬜ 4. Smooth-SwiGLU per FP8 — PRIORITÀ BASSA

SwiGLU può causare instabilità in FP8 training su run lunghi (>450B token) perché i pesi si allineano e producono outlier.

- **Ref:** paper FOG (oct 2025)
- **Impatto:** a 4B con training bf16 non è un problema. Rilevante solo se si passa a FP8 training.
- **Azione:** nessuna per ora

---

## ⬜ 5. RoPE + NoPE Hybrid (Sliding Window) — PRIORITÀ BASSA

Alternare layer con RoPE e layer senza (NoPE), con sliding window attention sui layer RoPE. Migliora il retrieval a context lunghi (128K+).

- **Ref:** "Rope to Nope and Back Again" (jan 2025)
- **Impatto:** con max context 32-65K, RoPE standard è sufficiente. Rilevante solo per 128K+.
- **Azione:** nessuna per ora

---

## ⬜ 6. Data Repetition > Data Scaling per SFT — RILEVANTE ORA

Pochi sample ripetuti per molte epoch battono tanti sample per 1 epoch. Con 200 sample ripetuti, OLMo3-7B supera il training su dataset completo con meno catastrophic forgetting.

- **Ref:** arxiv 2602.11149 (feb 2026)
- **Impatto:** con 2588 conversazioni SFT regtech, selezionare i migliori ~200-500 e trainare per più epoch potrebbe dare risultati migliori
- **Azione:** aggiungere parametro `--sft_epochs` in `train_sft.py`