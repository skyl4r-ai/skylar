# esport key

export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=

# Train new tokenizer + upload to S3

python pre_tokenize.py \
--data data/pretrain \
--output data/tokenized_corpus \
--vocab_size 40960 \
--s3_bucket sophia-ai-dset \
--s3_prefix skylar/pretrain_v1 \
--s3_region eu-south-1

# Train model from s3

python train.py \
--s3_bucket sophia-ai-dset \
--s3_prefix skylar/pretrain_v1 \
--s3_region eu-south-1 \
--preset medium \
--bf16 \
--max_steps 50000

# Scarica il best

python download_model.py \
--checkpoint best \
--output ./my_model \
--s3_bucket sophia-ai-dset \
--s3_prefix skylar/pretrain_v1 \
--s3_region eu-south-1

### beta 1 => effettive pre-training | 4090

bash train.runpod.sh --host root@38.80.152.146 --port 30664 \
--preset medium \
--bf16 \
--compile \
--batch_size 4 \
--grad_accum 8 \
--lr 3e-4 \
--lr_schedule cosine \
--warmup_steps 200 \
--weight_decay 0.1 \
--grad_clip 1.0 \
--max_steps 4200 \
--eval_every 200 \
--save_every 500 \
--sample_every 200 \
--dropout 0.0

### beta 1 => effettive pre-training | b200

bash train.runpod.sh --host root@38.80.152.146 --port 30664 \
--preset medium \
--bf16 \
--compile \
--batch_size 16 \
--grad_accum 4 \
--lr 2e-4 \
--lr_schedule cosine \
--warmup_steps 400 \
--weight_decay 0.1 \
--grad_clip 1.0 \
--max_steps 8400 \
--eval_every 1000 \
--save_every 2000 \
--sample_every 1000 \
--dropout 0.0 \
--out_dir /workspace/skylar-100M-Base

## training best 4.7k conversations

python train_sft.py \
  --data ./data/distilled_sft_dataset.jsonl \
  --base_model ./checkpoints/skylar-100M-Base \
  --seq_len 1024 \
  --batch_size 4 \
  --grad_accum 4 \
  --epochs 5 \
  --lr 8e-6 \
  --warmup_steps 60 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  --lr_schedule cosine \
  --eval_every 50 \
  --save_every 100 \
  --sample_every 50 \
  --bf16 \
  --out_dir ./checkpoints/skylar-100M-Chat-v3

##==================================
## 1. La curva della loss — il segnale principale

# Media mobile scende → tutto ok, oscillazioni normali attese
     ↓↗↓↓↗↓↓↗↓↓

# Media mobile piatta + oscillazioni enormi → lr troppo alto
     ─↗↘↗↘↗↘─

# Media mobile scende poi risale → overfitting → smetti
     ↓↓↓↓__↗↗↗

# Media mobile non scende mai → lr troppo basso o dati rotti
     ─────────────

## 2. Train loss vs Val loss — il segnale di overfitting
train_loss = 0.8,  val_loss = 0.9  → ok, differenza piccola
train_loss = 0.3,  val_loss = 1.8  → overfitting grave → modello inutile
train_loss = 1.8,  val_loss = 1.9  → underfitting → allena di più

# 3. I sample generati — il segnale più onesto
ogni 50 step guarda cosa genera il modello
se genera spazzatura dopo 500 step → qualcosa è rotto
se genera testo sensato → sta imparando

# Epoche
- non pensare alle epoche → pensa ai token visti
- token visti = batch_size × grad_accum × seq_len × steps

regola empirica:
dataset piccolo (<100M token) → 3-5 epoche
dataset medio (1B token)      → 1-2 epoche
dataset grande (10B+ token)   → meno di 1 epoca

Se vedi val_loss che risale → smetti subito, non finire le epoche.

**Learning rate:**
```
non indovinarlo → calcolalo

lr ottimale ≈ 0.003 / sqrt(dimensione modello)

modello 50M  → lr ≈ 4e-4  (training da zero)
modello 50M  → lr ≈ 1e-5  (fine-tuning)
modello 500M → lr ≈ 1e-4  (training da zero)
```

**Warmup:**
```
warmup_steps = 1% degli step totali

se hai 1000 step totali → warmup = 10
se hai 10000 step totali → warmup = 100
```

---

## Il segnale di "pesi spappolati" che cerchi

Questo si chiama **catastrophic forgetting** o **training instability**. I segnali sono:
```
1. loss scende a zero troppo veloce → lr troppo alto
2. il modello genera sempre la stessa frase → mode collapse
3. il modello genera caratteri a caso → pesi esplosi
4. val_loss molto peggio di train_loss → overfitting grave
5. grad_norm sempre > 1.0 → gradienti instabili → abbassa lr

grad_norm < 0.5   → training stabile
grad_norm 0.5-1.0 → normale
grad_norm > 1.0   → viene clippato, attenzione
grad_norm > 10.0  → qualcosa è rotto → abbassa lr subito

######## dovremmo aggiungere questo

# fuori dal loop — inizializza
loss_history = []

# dentro il logging
loss_history.append(raw_loss)
if len(loss_history) > 20:
    loss_history.pop(0)
moving_avg = sum(loss_history) / len(loss_history)

print(f"  step {step:6d}/{args.max_steps} | "
      f"loss {raw_loss:.4f} | avg {moving_avg:.4f} | "
      f"lr {display_lr:.2e} | grad_norm {grad_norm:.3f} | "
      f"{tok_per_sec:.0f} tok/s")