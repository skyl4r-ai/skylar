# Guida Parametri Training — NanoTransformer

---

## Come calcolare `max_steps` dalle epoche

```python
steps_per_epoch = n_token / (seq_len * batch_size * grad_accum)
max_steps       = steps_per_epoch * epoche_desiderate
warmup_steps    = int(max_steps * 0.02)  # 2% degli step totali

# esempio:
# 10M token, seq_len=1024, batch=4, grad_accum=4, 3 epoche
steps_per_epoch = 10_000_000 / (1024 * 4 * 4) = 610
max_steps       = 610 * 3 = 1830
warmup_steps    = int(1830 * 0.02) = 36
```

---

## PRE-TRAINING

### Quando usarlo
Dataset grande (>10M token), stai allenando da zero, vuoi che il modello impari la lingua.

### Parametri per dimensione modello

| Parametro | Modello ~10M | Modello ~50M | Modello ~200M |
|-----------|-------------|-------------|--------------|
| `--lr` | `3e-4` | `2e-4` | `1e-4` |
| `--seq_len` | `512` | `1024` | `2048` |
| `--batch_size` | `8` | `4` | `2` |
| `--grad_accum` | `4` | `8` | `16` |
| `--weight_decay` | `0.1` | `0.1` | `0.1` |
| `--grad_clip` | `1.0` | `1.0` | `1.0` |
| `--dropout` | `0.0` | `0.0` | `0.0` |
| `--warmup_steps` | `2% di max_steps` | `2% di max_steps` | `2% di max_steps` |
| `--lr_schedule` | `cosine` o `wsd` | `cosine` o `wsd` | `cosine` o `wsd` |

### Parametri per dimensione dataset

| Dataset | Epoche consigliate | Note |
|---------|-------------------|------|
| <10M token | 5-10 | Piccolo, ripassa spesso |
| 10M–100M token | 3-5 | Standard |
| 100M–1B token | 1-3 | Già tanto |
| >1B token | 1 o meno | Non finisci l'epoca |

### Effective batch size consigliato
```
pre-training → effective batch = batch_size × grad_accum = 32–256
```

### Segnali di run sana
```
grad_norm stabile sotto 1.0
train loss scende nella media mobile
val loss segue train loss con gap < 15%
i sample generati ogni 500 step migliorano
```

---

## SFT (Supervised Fine-Tuning)

### Quando usarlo
Hai un modello pre-trainato, vuoi insegnargli un comportamento specifico (chat, istruzioni, dominio).

### Parametri per dimensione dataset

| Parametro | Dataset piccolo (<5k esempi) | Dataset medio (5k–50k) | Dataset grande (>50k) |
|-----------|----------------------------|------------------------|----------------------|
| `--lr` | `5e-6` | `1e-5` | `2e-5` |
| `--epochs` / `max_steps` | 1-2 epoche | 2-3 epoche | 1-2 epoche |
| `--batch_size` | `2–4` | `4–8` | `8–16` |
| `--grad_accum` | `8–16` | `4–8` | `2–4` |
| `--weight_decay` | `0.01` | `0.05` | `0.1` |
| `--grad_clip` | `1.0` | `1.0` | `1.0` |
| `--dropout` | `0.05` | `0.1` | `0.1` |
| `--warmup_steps` | `5% di max_steps` | `3% di max_steps` | `2% di max_steps` |
| `--lr_schedule` | `cosine` | `cosine` | `cosine` o `wsd` |

### Differenze chiave rispetto al pre-training

| | Pre-training | SFT |
|--|-------------|-----|
| LR | più alto (`1e-4`–`3e-4`) | più basso (`5e-6`–`2e-5`) |
| Epoche | 1-3 su dataset grande | 1-3 su dataset piccolo |
| Dropout | `0.0` (dati abbondanti) | `0.05`–`0.1` (pochi dati) |
| Overfitting | raro | rischio alto |
| Effective batch | 32–256 | 16–64 |

---

## Scegliere tra Cosine e WSD

| Situazione | Schedule consigliato |
|-----------|---------------------|
| Sai esattamente quanti step farai | `cosine` |
| Vuoi poter estendere il training | `wsd` |
| Fine-tuning veloce | `cosine` |
| Pre-training lungo e iterativo | `wsd` |
| Non sai quanti step servono | `wsd` |

```bash
# WSD — parametro aggiuntivo
--lr_schedule wsd --lr_decay_ratio 0.1
# ultimo 10% degli step = decay finale
# puoi estendere la fase stable senza ripartire
```

---

## Segnali di allarme durante il training

| Segnale | Causa probabile | Soluzione |
|---------|----------------|-----------|
| `grad_norm > 2.0` costante | lr troppo alto | dimezza lr |
| loss oscilla senza scendere | lr troppo alto | dimezza lr |
| loss non scende mai | lr troppo basso o dati rotti | raddoppia lr o controlla dati |
| `val_loss >> train_loss` | overfitting | aumenta weight_decay o riduci epoche |
| loss va a `nan` | lr troppo alto o dati corrotti | dimezza lr, controlla dati |
| sample generano sempre la stessa frase | mode collapse | abbassa lr, controlla dati |

---

## Checklist prima di ogni run costosa

```
□ run di debug 100 step — loss scende?
□ sample ogni 10 step — testo sensato dopo 50 step?
□ grad_norm — sotto 1.0?
□ calcola max_steps dalle epoche che vuoi
□ warmup = 2-5% di max_steps
□ effective batch = batch_size × grad_accum nel range giusto
□ hai guardato manualmente 50-100 esempi del dataset?
   (90% dei problemi vengono dai dati)
```

```bash
# template run di debug
python train.py \
  --data tuoi_dati \
  --max_steps 100 \
  --eval_every 10 \
  --sample_every 10 \
  --log_every 1 \
  --bf16
```

---

## Valori fissi — non toccarli quasi mai

```bash
--grad_clip 1.0       # rete di sicurezza standard
--weight_decay 0.1    # pre-training / 0.01-0.05 SFT con pochi dati
--bf16                # sempre se la GPU lo supporta
```

---

## Formula rapida per stimare il costo

```python
token_per_step  = seq_len * batch_size * grad_accum
token_totali    = token_per_step * max_steps
token_nel_dataset = n_token * epoche

# se token_totali >> token_nel_dataset → stai overfittando
# se token_totali << token_nel_dataset → stai undertrainando
```