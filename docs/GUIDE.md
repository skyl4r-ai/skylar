# NanoTransformer — Pipeline Guide

## Step 0: Tokenizer + Pre-tokenization (una volta sola)

Sì, **si fa una volta sola**. Il tokenizer e il `pretokenized.pt` valgono per tutti i preset.

```bash
python pre_tokenize.py \
  --data data/pretrain/ \
  --vocab_size 40960 \
  --output data/
```

Output: `data/tokenizer.json` + `data/pretokenized.pt`

> ⚠️ Se cambi il corpus o il vocab_size, devi rifare questo step.
> Ma cambiare preset (small→4b) **non** richiede ri-tokenizzazione.

---

## Step 1: Pre-training

### Small (~40M) — RTX 4090, ~5 ore

```bash
python train.py \
  --tokenized_data data/pretokenized.pt \
  --tokenizer data/ \
  --preset small \
  --bf16 --compile \
  --batch_size 0 --grad_accum 8 \
  --lr 6e-4 --warmup_steps 500 \
  --max_steps 50000 \
  --lr_schedule wsd --lr_decay_ratio 0.1 \
  --wandb --wandb_run "small-v1"
```

| Parametro | Valore | Note |
|---|---|---|
| LR | 6e-4 | Modelli piccoli tollerano LR alta |
| Batch effettivo | auto × 8 | ~256-512 token/batch equiv |
| Schedule | WSD 10% | Stabile, può stoppare/riprendere |
| Steps | 50K | ~1.8B token (2× Chinchilla) |

### Medium (~107M) — RTX 4090, ~10 ore

```bash
python train.py \
  --tokenized_data data/pretokenized.pt \
  --tokenizer data/ \
  --preset medium \
  --bf16 --compile \
  --batch_size 0 --grad_accum 8 \
  --lr 4e-4 --warmup_steps 750 \
  --max_steps 100000 \
  --lr_schedule wsd --lr_decay_ratio 0.1 \
  --wandb --wandb_run "medium-v1"
```

| Parametro | Valore | Note |
|---|---|---|
| LR | 4e-4 | Tra small (6e-4) e large (3e-4) |
| Steps | 100K | ~3.6B token (quasi 2× Chinchilla) |
| Primo modello SFT-ready | ✅ | Chat strutturate funzionano |

### Large (~358M) — RTX 4090/5090, ~4 giorni

```bash
python train.py \
  --tokenized_data data/pretokenized.pt \
  --tokenizer data/ \
  --preset large \
  --bf16 --compile --gradient_checkpointing \
  --batch_size 0 --grad_accum 16 \
  --lr 3e-4 --warmup_steps 1000 \
  --max_steps 200000 \
  --lr_schedule wsd --lr_decay_ratio 0.1 \
  --wandb --wandb_run "large-v1"
```

| Parametro | Valore | Note |
|---|---|---|
| LR | 3e-4 | Standard per ~350M |
| Grad checkpoint | ✅ | Necessario su 24GB |
| Steps | 200K | ~7B token |

### XL (~1B) — RTX PRO 6000 o DGX H100

```bash
python train.py \
  --tokenized_data data/pretokenized.pt \
  --tokenizer data/ \
  --preset xl \
  --bf16 --compile --gradient_checkpointing \
  --batch_size 0 --grad_accum 32 \
  --lr 2e-4 --warmup_steps 2000 \
  --max_steps 500000 \
  --lr_schedule wsd --lr_decay_ratio 0.1 \
  --wandb --wandb_run "xl-v1"
```

| Parametro | Valore | Note |
|---|---|---|
| LR | 2e-4 | Scala con la radice dei params |
| Grad accum | 32 | Batch effettivo grande per stabilità |
| Steps | 500K | ~20B token |
| ❌ RTX 4090 | OOM | Serve ≥48GB VRAM |

### 4B (Qwen3-4B) — DGX H100

```bash
python train.py \
  --tokenized_data data/pretokenized.pt \
  --tokenizer data/ \
  --preset 4b \
  --bf16 --compile --gradient_checkpointing \
  --batch_size 0 --grad_accum 64 \
  --lr 1.5e-4 --warmup_steps 4000 \
  --max_steps 2000000 \
  --lr_schedule wsd --lr_decay_ratio 0.1 \
  --wandb --wandb_run "4b-v1"
```

| Parametro | Valore | Note |
|---|---|---|
| LR | 1.5e-4 | Qwen3 usa ~1.5e-4 per 4B |
| Grad accum | 64 | Batch effettivo ~4M token |
| Steps | 2M | ~80B token (Chinchilla) |
| Multi-GPU | `--multi_gpu` con accelerate | Necessario per tempi ragionevoli |

---

## Step 2: SFT (Chat Fine-tuning)

### Small / Medium — test rapido

```bash
python train_sft.py \
  --data data/sft_train.jsonl \
  --base_model checkpoints/final \
  --bf16 --compile \
  --batch_size 4 --grad_accum 4 \
  --lr 2e-5 --warmup_steps 50 \
  --epochs 5 \
  --lr_schedule cosine \
  --wandb --wandb_run "sft-small"
```

### Large / XL / 4B — SFT reale

```bash
python train_sft.py \
  --data data/sft_train.jsonl \
  --base_model checkpoints/final \
  --bf16 --compile --gradient_checkpointing \
  --batch_size 0 --grad_accum 8 \
  --lr 2e-5 --warmup_steps 100 \
  --epochs 5 \
  --lr_schedule cosine \
  --wandb --wandb_run "sft-large"
```

| Parametro | Tutti i preset | Note |
|---|---|---|
| LR | **2e-5** | 10-15× più bassa del pre-training |
| Epochs | **5** | Multi-epoch > dataset grande × 1 epoch |
| Schedule | **cosine** | Per SFT corto, cosine va bene |
| Batch eff. | auto × 8 = ~32 | Batch grande + LR bassa = stabile |

> Con 2588 esempi: 5 epoch ≈ 800 step (batch=4, accum=4).
> Guarda la `gnorm` nei primi 100 step: se scende → buon segno.

---

## Regole d'oro

| Scala | LR pre-train | LR SFT | Warmup | Grad accum |
|---|---|---|---|---|
| Small (40M) | 6e-4 | 2e-5 | 500 | 8 |
| Medium (107M) | 4e-4 | 2e-5 | 750 | 8 |
| Large (358M) | 3e-4 | 2e-5 | 1000 | 16 |
| XL (1B) | 2e-4 | 1e-5 | 2000 | 32 |
| 4B | 1.5e-4 | 1e-5 | 4000 | 64 |

**LR scaling**: ÷2 ogni ~10× parametri.

**WSD vs Cosine**: WSD per pre-training (può stoppare/riprendere nella fase stabile). Cosine per SFT (breve, converge).

**µP** (quando scali): tuna HP su small con `--mup_base_d_model 512`, poi usa gli stessi HP su xl/4b con lo stesso flag.

---

## Cheat Sheet

```bash
# ── PIPELINE COMPLETO ──

# 1. Tokenizer (una volta)
python pre_tokenize.py --data data/pretrain/ --vocab_size 40960 --output data/

# 2. Pre-train
python train.py --tokenized_data data/pretokenized.pt --tokenizer data/ \
  --preset small --bf16 --compile --lr_schedule wsd --max_steps 50000

# 3. SFT
python train_sft.py --data data/sft_train.jsonl \
  --base_model checkpoints/final --bf16 --compile --epochs 5

# 4. Chat
python chat.py --model checkpoints_sft/best
```