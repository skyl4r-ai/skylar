# TEST.md — Validazione completa del progetto

Esegui TUTTI i test in ordine. Se qualcosa fallisce, fermati, riporta l'errore esatto con traceback, e proponi il fix.

Alla fine riporta una tabella riassuntiva con ✅ o ❌ per ogni test.

---

## 0. Syntax check

Verifica che tutti i file Python parsino senza errori di sintassi:

```bash
python -c "
import ast, sys
files = ['config.py', 'model.py', 'train.py', 'train_sft.py', 'chat_format.py', 'chat.py', 'generate.py', 'build_dataset.py', 'push_to_hub.py']
ok = True
for f in files:
    try:
        with open(f) as fh:
            ast.parse(fh.read())
        print(f'  ✅ {f}')
    except SyntaxError as e:
        print(f'  ❌ {f} — {e}')
        ok = False
if not ok:
    sys.exit(1)
print('\n  Tutti i file parsano correttamente.')
"
```

---

## 1. Config & Presets

Verifica che tutti i preset si istanzino e che i vincoli (d_model % n_heads, n_heads % n_kv_heads) siano rispettati:

```bash
python -c "
from config import get_config, PRESETS

print('  Testing all presets...')
for name in PRESETS:
    try:
        cfg = get_config(name)
        assert cfg.d_model % cfg.n_heads == 0, f'd_model % n_heads != 0'
        assert cfg.n_heads % cfg.n_kv_heads == 0, f'n_heads % n_kv_heads != 0'
        assert cfg.rope_theta > 0, f'rope_theta <= 0'
        assert cfg.max_seq_len > 0, f'max_seq_len <= 0'
        assert hasattr(cfg, 'qk_norm'), 'missing qk_norm'
        assert hasattr(cfg, 'rope_theta'), 'missing rope_theta'
        print(f'  ✅ {name}: d={cfg.d_model}, heads={cfg.n_heads}, kv={cfg.n_kv_heads}, layers={cfg.n_layers}, ctx={cfg.max_seq_len}, rope_theta={cfg.rope_theta}')
    except Exception as e:
        print(f'  ❌ {name}: {e}')
print()
"
```

---

## 2. Model — istanziazione, forward, generate

Verifica con il preset test che il modello si crea, fa forward, genera, e conta i parametri:

```bash
python -c "
import torch
from config import get_config
 NanoTransformer

device = 'cpu'
cfg = get_config('test', vocab_size=1000)
model = NanoTransformer(cfg).to(device)
print(f'  ✅ Modello creato: {model.count_params():,} params')

# Forward senza labels
x = torch.randint(0, 1000, (2, 64))
out = model(x)
assert 'logits' in out, 'missing logits'
assert 'loss' in out, 'missing loss key'
assert 'kv_cache' in out, 'missing kv_cache'
assert out['logits'].shape == (2, 64, 1000), f'logits shape wrong: {out[\"logits\"].shape}'
assert out['loss'] is None, 'loss should be None without labels'
print(f'  ✅ Forward (no labels): logits {out[\"logits\"].shape}')

# Forward con labels
y = torch.randint(0, 1000, (2, 64))
out = model(x, labels=y)
assert out['loss'] is not None, 'loss should not be None with labels'
assert out['loss'].dim() == 0, 'loss should be scalar'
print(f'  ✅ Forward (with labels): loss={out[\"loss\"].item():.4f}')

# Backward
out['loss'].backward()
grad_count = sum(1 for p in model.parameters() if p.grad is not None)
total_params = sum(1 for p in model.parameters())
print(f'  ✅ Backward: {grad_count}/{total_params} params have gradients')

# Generate
model.eval()
prompt = torch.randint(0, 1000, (1, 10))
gen = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
assert gen.shape[0] == 1, 'batch should be 1'
assert gen.shape[1] == 30, f'expected 30 tokens, got {gen.shape[1]}'
print(f'  ✅ Generate: input 10 tokens → output {gen.shape[1]} tokens')

# Generate con KV-cache disabilitato
gen2 = model.generate(prompt, max_new_tokens=20, use_cache=False)
assert gen2.shape[1] == 30
print(f'  ✅ Generate (no cache): {gen2.shape[1]} tokens')

print()
"
```

---

## 3. QK-Norm

Verifica che QK-Norm sia presente e funzionante, e che sia disattivabile:

```bash
python -c "
import torch
from config import get_config
 NanoTransformer

# Con QK-Norm (default)
cfg = get_config('test', vocab_size=1000, qk_norm=True)
model = NanoTransformer(cfg)
has_qk = hasattr(model.blocks[0].attn, 'q_norm')
assert has_qk, 'q_norm missing with qk_norm=True'
print(f'  ✅ QK-Norm attivo: q_norm e k_norm presenti')

# Senza QK-Norm
cfg2 = get_config('test', vocab_size=1000, qk_norm=False)
model2 = NanoTransformer(cfg2)
has_qk2 = hasattr(model2.blocks[0].attn, 'q_norm') and model2.blocks[0].attn.qk_norm
assert not has_qk2, 'q_norm should not be active with qk_norm=False'
print(f'  ✅ QK-Norm disattivabile')

# Entrambi producono output valido
x = torch.randint(0, 1000, (1, 32))
out1 = model(x)
out2 = model2(x)
assert out1['logits'].shape == out2['logits'].shape
print(f'  ✅ Entrambi producono output corretto')
print()
"
```

---

## 4. GQA

Verifica che GQA funzioni con diversi rapporti:

```bash
python -c "
import torch
from config import NanoTransformerConfig
 NanoTransformer

configs = [
    ('MHA (4:4)', dict(d_model=64, n_heads=4, n_kv_heads=4, n_layers=2, d_ff=128, max_seq_len=64)),
    ('GQA (4:2)', dict(d_model=64, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=128, max_seq_len=64)),
    ('GQA (4:1)', dict(d_model=64, n_heads=4, n_kv_heads=1, n_layers=2, d_ff=128, max_seq_len=64)),
]

x = torch.randint(0, 100, (1, 32))
for name, params in configs:
    cfg = NanoTransformerConfig(vocab_size=100, **params)
    model = NanoTransformer(cfg)
    out = model(x)
    gen = model.generate(torch.randint(0, 100, (1, 5)), max_new_tokens=10)
    print(f'  ✅ {name}: forward ok, generate ok ({model.count_params():,} params)')

print()
"
```

---

## 5. Save / Load (HuggingFace compatibility)

Verifica save_pretrained e from_pretrained:

```bash
python -c "
import torch, os, shutil
from config import get_config
 NanoTransformer

path = '/tmp/nano_test_save'
if os.path.exists(path):
    shutil.rmtree(path)

# Salva
cfg = get_config('test', vocab_size=1000)
model = NanoTransformer(cfg)
model.save_pretrained(path)
assert os.path.exists(os.path.join(path, 'config.json')), 'config.json missing'
print(f'  ✅ save_pretrained: {os.listdir(path)}')

# Carica
model2 = NanoTransformer.from_pretrained(path)
print(f'  ✅ from_pretrained: {model2.count_params():,} params')

# Verifica che i pesi siano identici
x = torch.randint(0, 1000, (1, 32))
with torch.no_grad():
    out1 = model(x)
    out2 = model2(x)
diff = (out1['logits'] - out2['logits']).abs().max().item()
assert diff < 1e-5, f'outputs differ by {diff}'
print(f'  ✅ Output identici dopo save/load (diff={diff:.2e})')

# Weight tying check
if cfg.tie_weights:
    same = model2.token_emb.weight.data_ptr() == model2.lm_head.weight.data_ptr()
    print(f'  ✅ Weight tying dopo load: {\"attivo\" if same else \"NON ATTIVO — ERRORE\"}')
    assert same, 'Weight tying broken after load!'

# Cleanup
shutil.rmtree(path)
print()
"
```

---

## 6. Chat format & Loss mask

Verifica il formato ChatML e la loss mask token-based:

```bash
python -c "
from chat_format import format_chatml, create_loss_mask, make_conversation, make_json_example, SPECIAL_TOKENS

# Test format
msgs = [
    {'role': 'system', 'content': 'You are helpful.'},
    {'role': 'user', 'content': 'Hi!'},
    {'role': 'assistant', 'content': 'Hello!'},
]
text = format_chatml(msgs)
assert '<|im_start|>system' in text
assert '<|im_start|>user' in text
assert '<|im_start|>assistant' in text
assert '<|im_end|>' in text
print(f'  ✅ format_chatml: corretto')

# Test generation prompt
text2 = format_chatml(msgs, add_generation_prompt=True)
assert text2.endswith('<|im_start|>assistant\n') or text2.rstrip().endswith('assistant')
print(f'  ✅ add_generation_prompt: corretto')

# Test helpers
conv = make_conversation(system='Test', turns=[('Q', 'A')])
assert len(conv['messages']) == 3
print(f'  ✅ make_conversation: {len(conv[\"messages\"])} messages')

json_ex = make_json_example('Parse this', {'key': 'value'})
assert len(json_ex['messages']) == 3
print(f'  ✅ make_json_example: ok')

# Test loss mask (se accetta messages)
import inspect
sig = inspect.signature(create_loss_mask)
params = list(sig.parameters.keys())
print(f'  ℹ️  create_loss_mask signature: {params}')

print()
"
```

---

## 7. Tokenizer

Verifica che il tokenizer BPE si alleni e funzioni:

```bash
python -c "
import os, shutil
from train import build_tokenizer

path = '/tmp/nano_test_tok'
if os.path.exists(path):
    shutil.rmtree(path)

texts = [
    'The quick brown fox jumps over the lazy dog.',
    'Il gatto mangia il pesce nel giardino.',
    'Machine learning is transforming the world.',
    'La tecnologia moderna cambia tutto.',
] * 100  # Ripeti per avere abbastanza dati

tokenizer = build_tokenizer(texts, vocab_size=500, save_path=path)
assert tokenizer.get_vocab_size() <= 500
print(f'  ✅ Tokenizer trainato: {tokenizer.get_vocab_size()} tokens')

# Encode/decode
encoded = tokenizer.encode('Hello world')
decoded = tokenizer.decode(encoded.ids)
print(f'  ✅ Encode: \"Hello world\" → {encoded.ids[:10]}...')
print(f'  ✅ Decode: {encoded.ids[:10]}... → \"{decoded}\"')

# Special tokens
assert tokenizer.token_to_id('<pad>') is not None
assert tokenizer.token_to_id('<bos>') is not None
assert tokenizer.token_to_id('<eos>') is not None
print(f'  ✅ Special tokens presenti')

# Save/load
from train import load_tokenizer
tok2 = load_tokenizer(path)
assert tok2.get_vocab_size() == tokenizer.get_vocab_size()
print(f'  ✅ Save/load tokenizer ok')

shutil.rmtree(path)
print()
"
```

---

## 8. Training end-to-end (pre-training)

Lancia un mini training completo e verifica che la loss scenda:

```bash
python train.py \
    --preset test \
    --vocab_size 1000 \
    --max_steps 100 \
    --batch_size 4 \
    --grad_accum 1 \
    --lr 1e-3 \
    --warmup_steps 10 \
    --log_every 10 \
    --eval_every 50 \
    --save_every 50 \
    --sample_every 50 \
    --out_dir /tmp/nano_test_pretrain \
    --num_workers 0 \
    --seed 42 2>&1 | tee /tmp/nano_pretrain_log.txt

# Verifica che la loss sia scesa
python -c "
import re
with open('/tmp/nano_pretrain_log.txt') as f:
    text = f.read()
losses = re.findall(r'loss\s+([\d.]+)', text)
if len(losses) >= 2:
    first = float(losses[0])
    last = float(losses[-1])
    print(f'  Loss: {first:.4f} → {last:.4f}')
    if last < first:
        print(f'  ✅ Loss scesa di {first - last:.4f}')
    else:
        print(f'  ❌ Loss NON scesa!')
else:
    print(f'  ❌ Non riesco a leggere le loss dal log')
"
```

---

## 9. Checkpoint resume

Verifica che il resume funzioni (modello + optimizer state):

```bash
python -c "
import os
path = '/tmp/nano_test_pretrain'

# Verifica che i checkpoint esistano
checkpoints = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
print(f'  Checkpoint trovati: {checkpoints}')

for cp in checkpoints:
    cp_path = os.path.join(path, cp)
    files = os.listdir(cp_path)
    has_config = 'config.json' in files
    has_model = any('safetensors' in f or 'bin' in f for f in files)
    has_tokenizer = 'tokenizer.json' in files
    has_training_state = 'training_state.pt' in files

    status = '✅' if (has_config and has_model) else '❌'
    extras = []
    if has_tokenizer: extras.append('tokenizer')
    if has_training_state: extras.append('optimizer_state')
    print(f'  {status} {cp}: config={has_config}, model={has_model}, extras={extras}')

print()
"

# Testa resume effettivo
python train.py \
    --preset test \
    --vocab_size 1000 \
    --max_steps 150 \
    --batch_size 4 \
    --grad_accum 1 \
    --log_every 10 \
    --out_dir /tmp/nano_test_pretrain \
    --resume /tmp/nano_test_pretrain/step_50 \
    --num_workers 0 2>&1 | head -20
```

---

## 10. SFT end-to-end

Testa il pipeline SFT completo:

```bash
# Genera dataset demo
python chat_format.py 2>&1 | tail -5

# Lancia SFT
python train_sft.py \
    --preset test \
    --vocab_size 1000 \
    --max_steps 100 \
    --batch_size 4 \
    --grad_accum 1 \
    --lr 1e-3 \
    --warmup_steps 10 \
    --log_every 10 \
    --eval_every 50 \
    --save_every 50 \
    --sample_every 50 \
    --out_dir /tmp/nano_test_sft \
    --num_workers 0 2>&1 | tee /tmp/nano_sft_log.txt

python -c "
import re
with open('/tmp/nano_sft_log.txt') as f:
    text = f.read()
losses = re.findall(r'loss\s+([\d.]+)', text)
if len(losses) >= 2:
    first = float(losses[0])
    last = float(losses[-1])
    print(f'  SFT Loss: {first:.4f} → {last:.4f}')
    if last < first:
        print(f'  ✅ SFT loss scesa')
    else:
        print(f'  ❌ SFT loss NON scesa!')
"
```

---

## 11. Generation & Chat (se SFT completato)

```bash
# Test generate.py su modello pre-trained
python generate.py \
    --model /tmp/nano_test_pretrain/final \
    --prompt "The" \
    --max_tokens 50 \
    --temperature 0.8 2>&1

echo "---"

# Test chat.py su modello SFT (non interattivo, una domanda)
echo "What is AI?" | timeout 10 python chat.py \
    --model /tmp/nano_test_sft/final \
    --max_tokens 50 2>&1 || echo "  ℹ️  Chat test completato (o timeout)"
```

---

## 12. Speed features (se implementate)

Testa solo le feature che sono state implementate. Skippa quelle che non ci sono.

```bash
python -c "
import subprocess, sys

tests = {
    'torch.compile': ['python', 'train.py', '--preset', 'test', '--vocab_size', '1000', '--max_steps', '20', '--num_workers', '0', '--compile'],
    'gradient_checkpointing': ['python', 'train.py', '--preset', 'test', '--vocab_size', '1000', '--max_steps', '20', '--num_workers', '0', '--gradient_checkpointing'],
    'packing': ['python', 'train.py', '--preset', 'test', '--vocab_size', '1000', '--max_steps', '20', '--num_workers', '0', '--packing'],
}

for name, cmd in tests.items():
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f'  ✅ {name}: funziona')
        elif 'unrecognized arguments' in result.stderr:
            print(f'  ⏭️  {name}: non ancora implementato (skip)')
        else:
            print(f'  ❌ {name}: errore (exit code {result.returncode})')
            # Stampa le ultime 3 righe di stderr
            lines = result.stderr.strip().split('\n')
            for l in lines[-3:]:
                print(f'      {l}')
    except subprocess.TimeoutExpired:
        print(f'  ⚠️  {name}: timeout (120s)')
    except Exception as e:
        print(f'  ❌ {name}: {e}')

print()
"
```

---

## 13. Cleanup

```bash
rm -rf /tmp/nano_test_pretrain /tmp/nano_test_sft /tmp/nano_test_save /tmp/nano_test_tok /tmp/nano_pretrain_log.txt /tmp/nano_sft_log.txt
echo "  ✅ Cleanup completato"
```

---

## Tabella finale

Dopo tutti i test, stampa questa tabella compilata con i risultati:

```
  ┌─────┬──────────────────────────────┬────────┐
  │  #  │ Test                         │ Stato  │
  ├─────┼──────────────────────────────┼────────┤
  │  0  │ Syntax check                 │ ✅/❌  │
  │  1  │ Config & Presets             │ ✅/❌  │
  │  2  │ Model forward & generate     │ ✅/❌  │
  │  3  │ QK-Norm                      │ ✅/❌  │
  │  4  │ GQA                          │ ✅/❌  │
  │  5  │ Save / Load HF               │ ✅/❌  │
  │  6  │ Chat format & loss mask      │ ✅/❌  │
  │  7  │ Tokenizer                    │ ✅/❌  │
  │  8  │ Pre-training end-to-end      │ ✅/❌  │
  │  9  │ Checkpoint resume            │ ✅/❌  │
  │ 10  │ SFT end-to-end              │ ✅/❌  │
  │ 11  │ Generation & Chat            │ ✅/❌  │
  │ 12  │ Speed features               │ ✅/⏭️   │
  └─────┴──────────────────────────────┴────────┘
```

Se qualcosa è ❌, proponi il fix ma NON applicarlo senza conferma.