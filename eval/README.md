# `eval/` · how we know a model works

Every stage has its own test, from a change to the architecture to the released weights. Most of them are
pass or fail, and none of them trusts the training loss alone.

```mermaid
flowchart TB
    code["code change"] --> gates["18 architecture gates"]
    gates --> pre["pre-training"]
    pre --> val["validation loss · bits per byte"]
    pre --> ckpt["cache parity · base health check"]
    ckpt --> sft["fine-tuning"]
    sft --> cobol["COBOLEval<br/>compile · run · compare"]
    sft --> chat["chat and grounded tasks"]
```

## COBOLEval: does the COBOL compile, run and return the right answer?

[COBOLEval](https://github.com/BloopAI/COBOLEval) (BloopAI, MIT) turns the 146 HumanEval problems into
COBOL subprograms. [`bin.coboleval.py`](bin.coboleval.py) asks the model for each program, links it with the
calling program of every test, compiles both with GnuCOBOL, runs the binary and compares the value it writes
with the expected one. It reports two numbers:

- **CSR** (compile success rate): the share of problems whose program compiles;
- **pass@1:** the share whose program compiles and passes every test.

**To reproduce the published result of Skylar-980M-Cobol:**

```bash
git clone https://github.com/BloopAI/COBOLEval eval/COBOLEval             # the problems, not vendored here
micromamba create -y -p tools/gnucobol-env -c conda-forge gnucobol=3.2   # GnuCOBOL 3.2, no root needed

COBC=tools/gnucobol-env/bin/cobc python eval/bin.coboleval.py \
    --model Skyl4r-Ai/Skylar-980M-Cobol --greedy --out samples.jsonl
```

| model | CSR | pass@1 | decoding |
|:--|--:|--:|:--|
| Skylar-980M-Cobol | **80.1%** | **7.5%** | greedy, seed 0, 146 problems |

**Checked:**
- On the samples of the published run, this scorer gives the published numbers, with the same outcome on
  every one of the 146 problems.
- Greedy generation from the checkpoint on the Hub reproduces those samples byte for byte, on the problems
  we compared.

**Other models** are scored by the same code: `--samples file.jsonl` takes one `{"task_id", "completion"}`
per line, generated anywhere.

GnuCOBOL 3.2 is required. The 3.1 package of Ubuntu does not accept the source format COBOLEval compiles
with, and every program would look like a compile failure, so the scorer stops if it finds 3.1.

## Before training: is the code right?

[`bin.gate_arch_v2.py`](bin.gate_arch_v2.py) runs 18 checks in seconds, and exits with 1 if any fails.

| gate | question |
|:--|:--|
| v1 parity | with every Skylar 2 option off, are the logits bit-identical to the frozen code the published models were trained with? |
| parameter count | does every component cost exactly the parameters the formula says? |
| initialisation | did every matrix receive its initialisation, including the new ones? |
| hybrid wiring | is every layer the type the config asks for? |
| document isolation | does the recurrent state stop at document boundaries in packed sequences? |
| cache | does generation with the cache match a full recomputation? |
| Attention Residuals | does the fused kernel match a float64 reference, forward and backward? |
| checkpointing | are gradients the same with and without activation checkpointing? |
| KDA without CUDA | does the PyTorch recurrence match the Triton kernel? |

```bash
python eval/bin.gate_arch_v2.py                  # preset test, seconds
python eval/bin.gate_arch_v2.py --preset 1B_D    # the real size; does not fit in 24 GB (stops at the
                                                 # AttnRes gate, which keeps a float64 reference)
python eval/bin.gate_encoders.py                 # the dense, sparse and classifier encoders, dense and hybrid
python eval/bin.gate_growth.py                   # a model grown 2x wider computes the same function (beta)
python eval/bin.gate_decider.py                  # typed decisions: input, padding, learning, temperature (beta)
python eval/bin.gate_packing.py                  # packed windows (--pack): index, loader, no document sees another
```

[`bin.arch_ablation.py`](bin.arch_ablation.py) is an ablation protocol. It trains one proxy
per variant and seed, and evaluates all of them on the same fixed windows, so differences are paired by
seed. It resumes where it stopped.

## During pre-training: is it learning?

- **Validation cross-entropy** on fixed windows, from the trainer, at every evaluation.
- **Bits per byte** on a folder of frozen texts (`bin.pretrain.py --bpb_set <dir>`, measured by [`bpb.py`](bpb.py)
  with a sliding window). Perplexity cannot compare models with different vocabularies, because a bigger token
  carries more bytes; bits per byte divides by the bytes of the text itself, so every model is measured in the same
  unit.

## After pre-training

| tool | question | pass |
|:--|:--|:--|
| [`bin.cache_parity.py`](bin.cache_parity.py) | on trained weights, does decoding with the cache give the same tokens as recomputing everything? | fp32 argmax identical at every step |
| [`bin.eval_base_model.py`](bin.eval_base_model.py) | is the base ready for fine-tuning? perplexity, coherence, staying on topic, loops, leaked training artefacts | read the report |

## After fine-tuning

| tool | question |
|:--|:--|
| [`bin.coboleval.py`](bin.coboleval.py) | does the COBOL compile, run and return the right answer? (above) |
| [`validate_chat.py`](validate_chat.py) | does the chat model answer sensibly, and does it stop, emitting `<\|im_end\|>`? |
| [`validate_grounded.py`](validate_grounded.py) | the tasks a small model is for: answer from context, classify, extract JSON, refuse when the answer is absent |
| [`bin.diagnose_sft.py`](bin.diagnose_sft.py) | eight checks on the "never stops" failure: special tokens, loss mask, the probability of `<\|im_end\|>` |
| [`bin.diagnose_sft_drift.py`](bin.diagnose_sft_drift.py) | why a chat model talks nonsense: tokenisation, mask, drift from the base, embedding health |
| [`bin.eval_sft_model_token_stop.py`](bin.eval_sft_model_token_stop.py) | a quick probe: the raw tokens a chat model emits after a greeting |

## Public benchmarks

**Italian generation** ([`bench_ita.py`](bench_ita.py)): multiple choice scored by likelihood, the method of
lm-eval-harness.

| | Skylar-236M-Base | random |
|:--|:-:|:-:|
| XCOPA-it | **0.562** | 0.50 |
| HellaSwag-it | 0.292 | 0.25 |
| Belebele-it | 0.267 | 0.25 |

**Italian retrieval** ([`bench_retrieval.py`](bench_retrieval.py)): SQuAD-it test, 7,609 questions over
1,988 contexts, the same search and metrics for every model.

| model | parameters | R@1 | nDCG@10 |
|:--|--:|--:|--:|
| Skylar-236M-Embed | 236M | 0.55 | 0.71 |
| multilingual-e5-base | 278M | 0.71 | 0.83 |
| bge-m3 | 568M | 0.70 | 0.83 |

The 236M embedder does not beat these models on accuracy: it reaches 86% of the nDCG of bge-m3 at 2.4×
fewer parameters, from the same base as the chat model. [`eval_embeddings.py`](eval_embeddings.py) is the in-domain
check, with held-out phrasings.

## Data

[`bin.dataset_diagnostic.py`](bin.dataset_diagnostic.py) reports documents and tokens per source and flags
an unbalanced corpus. [`bin.dataset_diagnostic_sft.py`](bin.dataset_diagnostic_sft.py) does the same for a
conversation dataset.
