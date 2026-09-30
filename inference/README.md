# `inference/` · chat, generate, embed

For a trained model on your own machine. To serve a published model behind an OpenAI-compatible API,
use the [`skylar`](https://pypi.org/project/skylar/) pip package: `pip install skylar`, then
`skylar chat` or `skylar serve`.

## Chat

A streaming terminal chat built with `rich`, with ChatML turns and the last three exchanges kept as
history.

```bash
python inference/bin.chat.py --model checkpoints_sft/best
python inference/bin.chat.py --model checkpoints_sft/best --system "Sei un esperto programmatore COBOL."
```

| command | effect |
|:--|:--|
| `/system <text>` | set the system prompt |
| `/temp <0.0-2.0>` | temperature |
| `/topk <n>` · `/topp <0.0-1.0>` | top-k and nucleus sampling |
| `/rep <1.0-2.0>` | repetition penalty |
| `/max <n>` | maximum tokens per answer |
| `/clear` · `/reset` | clear the conversation |
| `/config` | show the current settings |
| `/help` · `/quit` | help, exit |

## Generate

```bash
python inference/bin.generate.py --model checkpoints/final --prompt "IDENTIFICATION DIVISION." --n 3
```

Sampling (temperature, top-k, top-p, repetition penalty) and the cache live in the model itself,
`Skylar2ForCausalLM.generate()` and `generate_streaming()`, so every script samples the same way.

## Embed

```bash
python inference/bin.embed.py --model <embedder_dir> \
    --query "Cos'è il TAEG?" \
    --docs "Il TAEG è il costo totale annuo di un finanziamento." "L'IBAN identifica un conto corrente."
```

Encodes texts into L2-normalised vectors with `SkylarEmbedder` and ranks the documents by cosine
similarity. `--text "…" --show_vector` prints the raw vector.
