# skylar

A tiny **runtime + CLI for the Skylar model family** — local, sovereign, from-scratch LLMs you
load, run, and serve with one `pip install`. It covers **generative chat**, **embeddings /
retrieval**, and a **COBOL code specialist** — 236M–390M class, runnable on a single GPU or CPU,
no data leaving your machine.

Models live under [`Sophia-AI` on HuggingFace](https://huggingface.co/Sophia-AI):
`Skylar-236M-Base` · `Skylar-236M-Chat` · `Skylar-236M-Embed` · `Skylar-390M-Cobol`.

## Install

```bash
pip install skylar
# optional HTTP server:
pip install "skylar[serve]"
```

## Use it — CLI

```bash
# chat with any Skylar generative model (no forced persona — steer it with --system)
skylar chat --model Sophia-AI/Skylar-236M-Chat --system "Sei un assistente che risponde dal contesto."

# embeddings / retrieval (any SkylarEmbedder model)
skylar embed --model Sophia-AI/Skylar-236M-Embed --query "prestito casa" --docs "mutuo" "meteo"

# one-shot generation (HF repo id or a local checkpoint dir)
skylar generate --model Sophia-AI/Skylar-236M-Chat --prompt "..."

# the COBOL specialist — completes a COBOL stub into a full, compilable program
#   (auto-downloads Skylar-390M-Cobol; it's a stub completer, not a chatbot)
skylar cobol --example
skylar cobol --stub-file my_task.cbl --compile        # your own stub + GnuCOBOL check

# OpenAI-compatible server (needs the [serve] extra) — auto-detects generative vs embedding
skylar serve --model <any-skylar-model> --port 8000
#   generative model -> POST /generate            POST /v1/chat/completions
#   embedder model   -> POST /v1/embeddings
```

Decoding is greedy by default (`--temperature 0.0`); there is **no forced system prompt** — pass
`--system "..."` to steer a chat model. (The `skylar cobol` subcommand handles the COBOL prompt
format for you.)

## Use it — Python

```python
import skylar

# generative chat — pass your own system prompt (no forced persona)
m = skylar.load("Sophia-AI/Skylar-236M-Chat")          # HF repo id or a local dir
print(m.generate("Domanda: dove ha sede la Banca d'Italia?",
                 system="Rispondi solo dal contesto fornito."))
for delta in m.stream("..."):                          # streaming
    print(delta, end="", flush=True)

# embeddings / retrieval
e = skylar.load_embedder("Sophia-AI/Skylar-236M-Embed")
ranked = e.rank("costo del denaro", ["la BCE alza i tassi", "ricetta pizza"])

# the COBOL specialist — a stub completer (not a chatbot)
c = skylar.load("Sophia-AI/Skylar-390M-Cobol")
print(c.complete_cobol(my_stub))                       # -> full, compilable COBOL program
```

`skylar` also registers the architecture with 🤗 Transformers, so this works too:

```python
import skylar  # registers nano-transformer
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Sophia-AI/Skylar-236M-Chat")
```

## What's inside

The Skylar models use a custom decoder (`NanoTransformer`, Qwen3-style: RMSNorm + RoPE + GQA +
QK-Norm + SwiGLU), trained 100% from scratch (no third-party pretrained weights). This package
vendors the architecture so the published weights load anywhere — no private framework needed.

## License

Apache-2.0. Models & code IP: A. Ivanovitch (Sophia AI).
