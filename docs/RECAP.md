# NanoTransformer — Recap Progetto (22 Feb 2026)

## Cosa è
Un transformer from scratch, architettura identica a Qwen3-4B-Instruct: RMSNorm, RoPE, SwiGLU, GQA, QK-Norm, KV-Cache, Flash Attention, ChatML. Pipeline completa: pre-training → SFT → chat.

## Struttura progetto
```
~/htdocs/sophia-core-server/model/skylar/
├── config.py          # NanoTransformerConfig + 11 preset (test→96b)
├── model.py           # NanoTransformer (architettura Qwen3-identica)
├── train.py           # Pre-training (text → base model)
├── train_sft.py       # SFT (base model → chat model)
├── chat_format.py     # ChatML format + loss mask
├── generate.py        # Generazione testo base model
├── chat.py            # Chat interattiva post-SFT
├── prepare_data.py    # Pre-tokenizzazione dataset grandi
├── CLAUDE.md          # Istruzioni per Claude Code
├── TEST.md            # 13 test automatici
├── README.md          # Documentazione completa
├── checkpoints/       # Modello pre-trained small (40M)
│   ├── best/          # val_loss=3.02 (USARE QUESTO come base)
│   └── final/         # train_loss=2.64
├── checkpoints_sft/   # SFT attempts (IGNORARE — small troppo piccolo per SFT)
└── data/
    ├── pretrain/       # 2 file .txt, 7.1 GB totale
    ├── pretokenized.pt # ~1.8 miliardi token pre-tokenizzati (pronto all'uso)
    ├── sft_train.jsonl # 2588 esempi SFT (JSONL ChatML format)
    └── tokenizer.json  # Copia del tokenizer (in data/)
```

## Hardware
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- RAM: 64GB
- OS: Linux (Ubuntu)
- Python 3.12, PyTorch 2.8.0+cu128, venv in `.venv`

## Preset disponibili (config.py)
| Preset | Params | Context | GPU necessaria | Tempo stimato |
|--------|--------|---------|----------------|---------------|
| test | 5M | 4K | Qualsiasi | Minuti |
| small | 40M | 8K | 1x 4090 | ~5 ore |
| medium | 125M | 16K | 1x 4090 | 2-3 giorni |
| large | 350M | 32K | 1x A100 80GB | 1-2 settimane |
| xl | 1B | 32K | 1x H100 | 2-4 settimane |
| 4b | 4B | 64K | 2-4x H100 | 1-3 mesi |
| 7b | 7B | 64K | 4-8x H100 | 2-4 mesi |
| 14b-96b | 14B-96B | 64-128K | Cluster | 3-9 mesi |

## Cosa è stato fatto

### 1. Pre-training small (40M) — ✅ COMPLETATO
- Dataset: 7.1 GB testo (italiano + inglese), 1.8 miliardi di token
- Tokenizer: BPE 40960 vocab, allenato su sample 500MB
- Training: 100K step, batch 2, grad_accum 16, bf16, ~4.5 ore su 4090
- Risultato: val_loss=3.02, genera italiano grammaticale ma ripetitivo
- Checkpoint: `checkpoints/best`

### 2. Pre-tokenizzazione — ✅ COMPLETATA
- `data/pretokenized.pt` — 1.8 miliardi token pronti
- Riusabile per qualsiasi preset (medium, large, ecc.)
- Non serve ri-tokenizzare: `--tokenized_data data/pretokenized.pt`

### 3. SFT small (40M) — ❌ FALLITO (modello troppo piccolo)
- 2588 esempi JSONL → loss scende ma genera spazzatura
- 40M parametri non hanno capacità per apprendere formato ChatML + risposte
- Conclusione: SFT funziona solo da medium (125M) in su

## Bug risolti (importanti per Claude Code)

### 1. OOM tokenizer su file grandi
**Problema**: BPE tokenizer (Rust) crashava su file >1GB. Processo terminato con SIGKILL.
**Fix**: `train.py` e `prepare_data.py` ora:
- Tokenizer allenato su sample 500MB (`_sample_iterator`)
- File tokenizzati in chunk da 50MB (`_tokenize_source`)
- `load_text_data()` restituisce Path objects, non legge file interi

### 2. OOM RAM durante tokenizzazione in train.py
**Problema**: lista Python di 1.4B token = 39GB RAM → crash.
**Fix**: usare `prepare_data.py` separatamente. Salva su disco file per file con numpy, concatena alla fine. Max 10-15GB RAM.

### 3. OOM VRAM durante training
**Problema**: batch_size=8 + seq_len=8192 + vocab=40960 → logits 5.4GB → OOM.
**Fix**: batch_size=2, grad_accum=16 (effective batch 32 invariato), --gradient_checkpointing.

### 4. ByteLevel decoder mancante nel tokenizer
**Problema**: `tokenizer.decode()` produceva Ġ, Ã¹, Ċ invece di spazi e accenti.
**Fix**: aggiunto `tokenizer.decoder = decoders.ByteLevel()` in tutti i file:
- `train.py` — in `build_tokenizer()` (root cause)
- `prepare_data.py` — in `build_tokenizer()`
- `train_sft.py` — in `ensure_tokenizer_ready()`
- `generate.py` — dopo load
- `chat.py` — dopo load
**Nota**: il tokenizer salvato in `checkpoints/` NON ha il decoder nel JSON. I file lo aggiungono al volo al load. Se si rifa il training con i file aggiornati, il decoder sarà nel JSON dalla nascita.

### 5. Resize embeddings distruggeva pesi pre-trained
**Problema**: aggiunta token ChatML (40960→40962) ricreava embedding da zero.
**Fix**: in `train_sft.py`, copia vecchi pesi prima del resize:
```python
old_emb = model.token_emb.weight.data.clone()
model.token_emb = torch.nn.Embedding(vocab_size, config.d_model)
model.token_emb.weight.data[:old_emb.shape[0]] = old_emb
```

### 6. Loss = 0.000 durante SFT (pre-fix resize)
**Causa**: embeddings tutti random → modello non produce nulla di sensato → labels tutti ignorati.
**Fix**: stesso del punto 5.

## Speed optimizations implementate
Tutti opt-in con flag:
1. `--compile` — torch.compile(), ~20-30% più veloce (solo CUDA)
2. `--gradient_checkpointing` — riduce VRAM attivazioni
3. `--packing` (default True) — elimina padding, 100% token utili
4. `--tokenized_data` — skip tokenizzazione, loading istantaneo
5. `--multi_gpu` — HuggingFace Accelerate per multi-GPU
6. `--batch_size 0` — auto-stima da VRAM disponibile

## Prossimo passo: Pre-training MEDIUM (125M)

Il tokenizer e i dati pre-tokenizzati sono già pronti. Comando:

```bash
cd ~/htdocs/sophia-core-server/model/skylar
source .venv/bin/activate

python train.py \
    --tokenized_data data/pretokenized.pt \
    --tokenizer checkpoints/ \
    --preset medium \
    --vocab_size 40960 \
    --max_steps 100000 \
    --batch_size 2 \
    --grad_accum 16 \
    --lr 3e-4 \
    --bf16 \
    --gradient_checkpointing \
    --compile
```

Stima: 2-3 giorni su RTX 4090. Dopo completamento:

```bash
python train_sft.py \
    --data data/sft_train.jsonl \
    --base_model checkpoints/best \
    --max_steps 500 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 1e-5 \
    --bf16 \
    --gradient_checkpointing \
    --eval_every 50 \
    --sample_every 50
```

**ATTENZIONE SFT**: con 2588 esempi e modello piccolo, l'overfitting è rapido. Monitorare val_loss — fermare se risale. 500 step con lr 1e-5 dovrebbe bastare.

## Dataset SFT
- `data/sft_train.jsonl` — 2588 esempi formato ChatML
- Formato: `{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- I dati sono dell'utente, non demo

## Istruzioni critiche per Claude Code
1. **NON leggere MAI i file in `data/pretrain/`** — sono 4GB ciascuno, crashano Claude Code
2. **NON leggere `data/pretokenized.pt`** — è ~14GB
3. Leggere solo file `.py` del progetto
4. Dopo ogni modifica, verificare syntax: `python -c "import ast; ast.parse(open('file.py').read())"`
5. Per test completi: "Esegui TEST.md"

## Note tecniche
- Architettura 4b è IDENTICA a Qwen3-4B-Instruct (d_model=2560, n_heads=32, n_kv_heads=8, n_layers=36, d_ff=6912)
- Differenza da Qwen3: solo dati di training e compute
- Tutti i componenti: RMSNorm, RoPE con theta scaling, SwiGLU (2/3 ratio), GQA, QK-Norm, KV-Cache, Flash Attention, weight tying, ChatML
- Il tokenizer ha vocabolario bilingue italiano/inglese (dataset è misto)
- Context window calibrato per dimensione modello (8K small, 16K medium, 32K large, 64K 4b+)