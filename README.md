<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brand/skylar-logo-dark-2048.png">
  <img alt="SKYL4R" src="docs/brand/skylar-logo-light-2048.png" width="420">
</picture>

### A from-scratch language-model framework, and the small local models it trains for legacy code.

[![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗_models-Skyl4r--Ai-FFD21E?style=flat-square)](https://huggingface.co/Skyl4r-Ai)
[![PyPI](https://img.shields.io/pypi/v/skylar?style=flat-square&label=pip%20install%20skylar&color=4F46E5)](https://pypi.org/project/skylar/)
[![License](https://img.shields.io/badge/code-Apache_2.0-blue?style=flat-square)](LICENSE)

**by [A. Ivanovitch](https://www.linkedin.com/in/aleksandr-ivanovitch-brunelli), CEO [SKYL4R](https://skyl4r.ai)**

</div>

<br>

Every layer, every rotation, every gradient is written by hand in plain PyTorch, with no black boxes and
no third-party weights: from a 9M test model to 128B-parameter presets, from raw documents to a chat
model, in one codebase that stays readable.

<div align="center">

### ⌨️ Skylar writes COBOL. It compiles, runs, and returns the right answer. 100% local, 100% from scratch.

<img src="docs/skylar-cobol-demo.gif" alt="A from-scratch Skylar model generates COBOL; GnuCOBOL compiles and runs it" width="760"/>

<img src="docs/coboleval_benchmark.png" alt="COBOLEval: the from-scratch Skylar COBOL models (390M and 980M) beat the 7B code models on the same harness" width="860"/>

</div>

On **COBOLEval** (146 problems, compiled and executed by GnuCOBOL, same harness for every model),
**Skylar-980M-Cobol beats Qwen2.5-Coder-7B, CodeLlama-7B and StarCoder2-7B** on both compile rate and
pass@1, with about 7× fewer parameters, fully local and trained from scratch. It is a research preview
and a COBOL specialist, not a general chatbot: read the
[model card](https://huggingface.co/Skyl4r-Ai/Skylar-980M-Cobol). The scorer is public, and the result can be
reproduced with one command: [`eval/`](eval/README.md).

```bash
pip install skylar

# the COBOL specialist: writes, explains and modifies COBOL, 100% local
skylar chat --model Skyl4r-Ai/Skylar-980M-Cobol --system "Sei un esperto programmatore COBOL."

# Italian chat and semantic retrieval (236M)
skylar chat  --model Skyl4r-Ai/Skylar-236M-Chat
skylar embed --model Skyl4r-Ai/Skylar-236M-Embed --query "prestito casa" --docs "mutuo" "meteo"

# System One (beta): what does this COBOL program print? Calibrated probabilities, no text generated
#   state.json = {"cobol": "<program>", "stdin": "<input>"}, questions.json = {"output": {"type": "choice", ...}}
skylar decide --model Skyl4r-Ai/Skylar-980M-Cobol-Decider --state-file state.json --questions questions.json
```

## 🌱 Models that grow (beta)

A trained Skylar model becomes one **twice as wide** (d_model, attention and recurrent heads, FFN) that computes
**exactly the same function** at step zero, then keeps learning: HyperCloning (arXiv 2409.12903) carried to every
piece of Skylar 2. Every weight still comes from a model trained from random init: from scratch, in two stages.

```bash
python training/bin.upscale.py --model <run>/last --out <grown>      # checks the logits before writing
python training/bin.pretrain.py --init_from <grown> --data <tokenized_dir> ... --resume auto
```

## ⚖️ System One: typed, calibrated decisions (beta)

A model that does not generate text: give it a state and typed questions (`choice` among options, `score` on a
rubric, `noul` true/false) and it returns **calibrated probabilities**, in the schema of Jev's API. It is a new head
on the same base model, trained in about an hour on one GPU; one temperature fitted on held-out cases calibrates it.

On the published [Skylar-980M-Cobol-Base](https://huggingface.co/Skyl4r-Ai/Skylar-980M-Cobol-Base):

| test | Skylar 980M decider | Qwen2.5-Coder-1.5B-Instruct, zero-shot |
|:--|--:|--:|
| "What does this COBOL program print?" (1,157 questions, labels from real execution) | **0.836** | 0.359 |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), the open dataset behind Laya (2,000 test questions) | **0.750** | 0.416 |

After calibration the expected calibration error is 0.019 and 0.024: when it says 80%, it is right about 80% of the
time. Laya (ModernBERT, 421M) scores 0.767 on typed-decisions. The baseline is zero-shot and the decider is trained:
the comparison says a general model of that size cannot make these decisions untrained, not that our trunk is
better. Deciding whether two programs print the same thing, which takes running them, is still at chance.

The COBOL decider is public: [Skylar-980M-Cobol-Decider](https://huggingface.co/Skyl4r-Ai/Skylar-980M-Cobol-Decider).

```python
import skylar                                   # pip install "skylar>=0.6.0"

d = skylar.load_decider("Skyl4r-Ai/Skylar-980M-Cobol-Decider")
d.decide({"cobol": open("program.cbl").read(), "stdin": "12\n30\n"}, {
    "output": {"type": "choice", "instructions": "What does the COBOL program in `cobol` print on the input in `stdin`?",
               "criteria": {"A": "0042", "B": "0041", "C": "0084"}}})
# {"output": {"type": "choice", "probabilities": {"A": ..., "B": ..., "C": ...}, "choice": ..., "confidence": ...}}
```

```bash
skylar serve --model Skyl4r-Ai/Skylar-980M-Cobol-Decider --port 8000  # POST /v1/decide {"state", "questions"}
python training/bin.decide.py --base_model <decoder> --train train.parquet --test test.parquet --bf16 --out <decider>
```

## 🧬 Skylar 2

The next architecture, built for long programs read whole on local hardware: three recurrent layers for
every attention layer, attention over depth instead of a plain residual sum, gated normalisation and the
Muon optimiser. With the options off it is the previous decoder, bit for bit.

```mermaid
flowchart TB
    emb(["tokens → embedding"]) --> grp
    subgraph grp ["36 layers = 9 × (3 KDA + 1 attention), read across depth by block Attention Residuals"]
        direction LR
        k1["KDA"] --> k2["KDA"] --> k3["KDA"] --> att["attention"]
    end
    grp --> head(["GatedNorm → LM head → next token"])
```

| component | what it does |
|:--|:--|
| **KDA hybrid, 3:1** | 27 of 36 layers keep a fixed-size recurrent state instead of a cache that grows with the context; every fourth layer keeps full attention for exact recall |
| **Block Attention Residuals** | each sub-layer reads a learned mixture of earlier depths, on a fused kernel that makes the step 7% faster |
| **GatedNorm** | a low-rank gate after each normalisation, which rescales without the activation outliers behind loss spikes |
| **SiTU-GLU** | a bounded SwiGLU, for low-precision inference without retraining |
| **Muon** | the optimiser for the hidden matrices; flat over a 4× range of learning rates |

**No Skylar 2 model has been trained at scale yet.** Architecture, flags and sizes: [`models/`](models/README.md).

## ⚡ Quick start

```bash
git clone https://github.com/skyl4r-ai/skylar.git && cd skylar
python -m venv .venv && source .venv/bin/activate
pip install -e .

# a model from scratch, on the GPU you have
python training/bin.pretrain.py --preset small \
    --data <tokenized_dir> --tokenizer <tokenizer.json> --out checkpoints/small

# talk to it after fine-tuning
python inference/bin.chat.py --model checkpoints_sft/best
```

## 🔄 One base, many products

```mermaid
flowchart TB
    pre["📚 pre-training"] --> sft["💬 SFT"]
    pre --> emb["🔎 dense embedder"]
    pre --> sp["🧮 sparse retriever"]
    pre --> cls["🏷️ classifier"]
    pre --> dec["⚖️ System One decider"]
    pre --> grow["🌱 a 2× wider base"]
    sft --> pref["🎯 ORPO · SimPO"]
    sft --> rl["🧪 GRPO · RFT"]
```

The chat model, the retrieval models, the classifier and the decider all start from the same pre-trained weights,
and a trained base can grow into a bigger one that starts the cycle again.

| | what is inside |
|:--|:--|
| [**`models/`**](models/README.md) | Skylar 2, the base decoder (RMSNorm, RoPE, GQA, QK-Norm, SwiGLU, FlexAttention, µP), 17 presets, the encoders, the decider, width growth |
| [**`training/`**](training/README.md) | pre-training from one GPU to several nodes, checkpoint merging, growth, SFT, preference, RL, distillation, retrieval heads, typed decisions |
| [**`data/`**](data/README.md) | corpus builder with PII redaction and dedup, code-aware BPE, memory-mapped shards |
| [**`inference/`**](inference/README.md) | streaming chat, generation, embeddings |
| [**`eval/`**](eval/README.md) | the COBOLEval scorer (compile, run, compare), architecture gates, bits per byte, cache parity, Italian benchmarks |

Everything is Hugging Face-native: `save_pretrained`, `from_pretrained`, `AutoModelForCausalLM` and
`push_to_hub` work as for any model.

## 📊 From 9M to 128B

| | presets and parameters |
|:--|:--|
| **small and research** | `test` 9M · `small` 52M · `small_plus` 88M · `medium` 125M · `medium_plus` 268M · `large` 381M · `gold` 426M |
| **the code line** | `1B` 1.04B · `1B_D` 1.0B · `2b` 2.1B |
| **Qwen3 shapes** | `4b` 3.8B · `8b` 7.2B · `14b` 13.5B · `32b` 31.5B |
| **frontier scale** | `64b` 62B · `96b` 100B · `128b` 128B |

Parameters with a 64,000-token vocabulary. The `4b` to `32b` presets have the shapes of the official
Qwen3 models, and every preset from `1B_D` up can be built as a Skylar 2 hybrid.
[All presets →](models/README.md#all-presets)

## 🤗 Published models

| model | size | what it is | result |
|:--|--:|:--|:--|
| [**Skylar-980M-Cobol**](https://huggingface.co/Skyl4r-Ai/Skylar-980M-Cobol) | 980M | COBOL specialist, instruction-tuned | COBOLEval: **80.1%** compile, **7.5%** pass@1 |
| [Skylar-980M-Cobol-Base](https://huggingface.co/Skyl4r-Ai/Skylar-980M-Cobol-Base) | 980M | its pre-trained base | |
| [Skylar-980M-Cobol-Decider](https://huggingface.co/Skyl4r-Ai/Skylar-980M-Cobol-Decider) | 980M | System One on COBOL (beta) | "what does it print?": **0.836** |
| [Skylar-236M-Chat](https://huggingface.co/Skyl4r-Ai/Skylar-236M-Chat) | 236M | grounded Italian chat | 6/6 grounded tasks |
| [Skylar-236M-Embed](https://huggingface.co/Skyl4r-Ai/Skylar-236M-Embed) | 236M | Italian embeddings | SQuAD-it R@1 0.55 · nDCG@10 0.71 |
| [Skylar-236M-Base](https://huggingface.co/Skyl4r-Ai/Skylar-236M-Base) | 236M | Italian legal base | XCOPA-it 0.562 |

The 236M models were trained on 1.12B tokens of Italian legal text; what they do well, and where they fall short,
is on each model card.

## 📚 Documentation

Every folder has its own README: [`models/`](models/README.md), [`training/`](training/README.md),
[`data/`](data/README.md), [`inference/`](inference/README.md), [`eval/`](eval/README.md). Every script prints its
options with `--help`.

## 📜 License

The framework's source code is licensed under the **[Apache License 2.0](LICENSE)**. Copyright © 2026
**Aleksandr Ivanovitch**, who holds all intellectual property rights in this software; see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

**Model weights are licensed separately**, on each model card. Skylar 2 models will be released under the
**PolyForm Noncommercial License 1.0.0**.

<br>

<div align="center">

*"The best way to understand a Transformer is to build one."*

**Built with 🔥 by [Aleksandr Ivanovitch](https://www.linkedin.com/in/aleksandr-ivanovitch-brunelli)**

<sub>Copyright © 2026 Aleksandr Ivanovitch · Apache License 2.0</sub>

</div>
