# Training Cheat Sheet — Chinchilla & Hyperparameters

## 1. Quanti token servono? (Chinchilla)

```
token_ottimali = n_params × 20
```

| Modello | Params | Token ottimali |
|---------|--------|----------------|
| test    | 5M     | 100M           |
| small   | 30M    | 600M           |
| medium  | 107M   | 2.14B          |
| large   | 358M   | 7.16B          |
| 1B      | 1B     | 20B            |

> **Post-Chinchilla (LLaMA-style):** puoi overtrainare fino a 100-200 tok/param
> se vuoi un modello più piccolo ma più smart a inference.
> Undertraining (< 20 tok/param) degrada sempre.

---

## 2. Quanti step servono?

⚠️ Nel nostro `train.py`, `step` = **micro-step** (ogni batch), NON optimizer update.

```
optimizer_updates = token_totali ÷ (effective_batch × seq_len)
max_steps         = optimizer_updates × grad_accum
```

### Esempio concreto (medium, 2.18B token):

```
effective_batch = batch_size × grad_accum = 16 × 4 = 64
token_per_optimizer_update = 64 × 16384 = 1,048,576 (~1M)
optimizer_updates = 2,180,000,000 ÷ 1,048,576 ≈ 2,080

max_steps = 2,080 × 4 (grad_accum) = 8,320 micro-step
```

### Formula rapida:

```
max_steps = token_totali ÷ (batch_size × seq_len)
          = 2,180,000,000 ÷ (16 × 16,384)
          = 8,316
```

### Verifica dal log:

```
packed_chunks ÷ batch_size = micro-step per epoca
126,512 ÷ 16 = 7,907 → arrotonda a ~8,000-8,400
```

---

## 3. Epoche

```
epoche = max_steps × batch_size × seq_len ÷ token_totali
```

- **Chinchilla puro:** 1 epoca (vedi ogni token 1 volta)
- **Overtrain (LLaMA-style):** 2-4 epoche ok, oltre 4 rischi overfitting
- **< 1 epoca:** stai undertrainando, sprechi compute

---

## 4. Learning Rate

| Parametro | Formula / Regola |
|-----------|-----------------|
| **max_lr** | 3e-4 per ≤300M params, 1.5e-4 per ~1B, 1e-4 per ≥3B |
| **min_lr** | max_lr × 0.1 (il 10%) |
| **warmup** | 1-5% degli optimizer updates |
| **schedule** | cosine (standard) o WSD (DeepSeek-style) |

### Warmup nel nostro script (micro-step!):

```
warmup_optimizer_updates = optimizer_updates × 0.05   (5%)
warmup_steps = warmup_optimizer_updates × grad_accum
```

Esempio: 2,080 updates × 0.05 = 104 → × 4 = **416 micro-step**
(arrotonda a 400-800)

### LR scaling con batch size:

Se raddoppi effective_batch, puoi scalare LR di √2:
```
lr_scaled = lr_base × √(new_batch ÷ base_batch)
```

---

## 5. Batch Size

| VRAM GPU | batch_size consigliato (seq_len=16K) |
|----------|--------------------------------------|
| 24 GB    | 1-2                                  |
| 40 GB    | 4-8                                  |
| 80 GB    | 8-16                                 |
| 192 GB   | 16-32                                |

> L'attention scala O(seq²), quindi con seq_len=16K il batch
> è il bottleneck, non i parametri del modello.

### Effective batch ottimale (regola empirica):

```
effective_batch_tokens ≈ 0.5M - 2M token
effective_batch = target_tokens ÷ seq_len

Esempio: 1M ÷ 16,384 = 61 → arrotonda a 64
→ batch_size=16, grad_accum=4
```

---

## 6. Altri Hyperparameters

| Param | Valore | Note |
|-------|--------|------|
| weight_decay | 0.1 | Standard per LLM |
| grad_clip | 1.0 | Standard |
| beta1 | 0.9 | AdamW default |
| beta2 | 0.95 | LLM standard (non 0.999!) |
| dropout | 0.0 | Se Chinchilla-optimal (dati sufficienti) |
| dropout | 0.1 | Se overtrain o dataset piccolo |

---

## 7. Quick Reference — Comando completo

```bash
# Variabili da calcolare
TOKENS=2180000000
BATCH=16
SEQ=16384
GRAD_ACCUM=4

# max_steps = token ÷ (batch × seq)
MAX_STEPS=$((TOKENS / (BATCH * SEQ)))  # = 8316

# warmup = 5% di max_steps
WARMUP=$((MAX_STEPS / 20))  # = 415

bash train.runpod.sh \
  --preset medium \
  --bf16 --compile \
  --batch_size $BATCH \
  --grad_accum $GRAD_ACCUM \
  --lr 2e-4 \
  --lr_schedule cosine \
  --warmup_steps $WARMUP \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  --max_steps $MAX_STEPS \
  --eval_every $((MAX_STEPS / 8)) \
  --save_every $((MAX_STEPS / 4)) \
  --sample_every $((MAX_STEPS / 8)) \
  --dropout 0.0
```