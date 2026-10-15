# `training/` · from pre-training to a usable model

One pre-trained decoder feeds every later stage. The chat model, the preference-tuned model and the
retrieval models all start from the same weights, with no second pre-training.

```mermaid
flowchart LR
    pre["Pre-training<br/>bin.pretrain.py"] --> merge["WSM merge<br/>bin.merge_wsm.py"]
    merge --> sft["SFT<br/>bin.sft.py"]
    sft --> pref["Preference<br/>ORPO · SimPO"]
    sft --> rl["RL with verifiable rewards<br/>grpo.py · rft.py"]
    merge --> emb["Dense embedder<br/>InfoNCE"]
    merge --> sparse["Sparse retriever<br/>SPLADE"]
    merge --> cls["Classifier"]
```

| stage | script | output |
|:--|:--|:--|
| pre-training | [`bin.pretrain.py`](bin.pretrain.py) | causal language model |
| checkpoint merge | [`bin.merge_wsm.py`](bin.merge_wsm.py) | the cooldown of a constant-LR run, done after training |
| growth (beta) | [`bin.upscale.py`](bin.upscale.py) | a model 2× wider computing the same function, trained on with `--init_from` |
| supervised fine-tuning | [`bin.sft.py`](bin.sft.py) | ChatML chat model, loss on assistant tokens only |
| preference | [`bin.preference.py`](bin.preference.py) | ORPO or SimPO, reference-free |
| reinforcement learning | [`grpo.py`](grpo.py), [`rft.py`](rft.py) | GRPO and rejection-sampling fine-tuning with a pluggable verifiable reward |
| distillation | [`bin.distill.py`](bin.distill.py) | online logit distillation, full or top-k |
| dense embedder | [`bin.contrastive.py`](bin.contrastive.py) | `SkylarEmbedder` (InfoNCE, in-batch negatives) |
| sparse retriever | [`bin.sparse.py`](bin.sparse.py) | `SkylarSparseEncoder` (SPLADE-style) |
| classifier | [`bin.classify.py`](bin.classify.py) | `SkylarClassifier` |
| typed decisions (beta) | [`bin.decide.py`](bin.decide.py) | `SkylarDecider`: choice / score / noul with calibrated probabilities |
| optimiser | [`optim.py`](optim.py) | Muon on the hidden matrices, AdamW on the rest |

Every script prints its options with `--help`.

## Pre-training

The trainer streams memory-mapped token shards (uint16 or uint32, see [`data/`](../data/README.md)),
runs on one GPU or many through Accelerate, and resumes after a crash with model, optimiser, RNG and
sampler state.

```bash
# a small model on one GPU
python training/bin.pretrain.py --preset medium --data <tokenized_dir> --tokenizer <tokenizer.json> \
    --seq_len 2048 --batch_size 8 --grad_accum 16 --lr <lr> --warmup <steps> --out checkpoints/medium

# resume where it stopped
python training/bin.pretrain.py ... --resume auto          # = <out>/last, or the newest usable last.prevN
```

`--resume auto` starts from random weights only at a run's first launch: a run that has trained and has no usable
checkpoint stops with an error, and the checkpoint must match the run (model config, tokenizer, `--seed`, the
`--data_mix` folders).

**The Skylar 2 architecture** is switched on by flags (explained in [`models/`](../models/README.md)); learning rate,
batch and schedule are yours to choose:

```bash
python training/bin.pretrain.py --preset 1B_D --data <tokenized_dir> --tokenizer <tokenizer.json> --seq_len 8192 \
    --kda_ratio 3:1 --attn_res --attn_res_mode block --attn_res_block_size 8 --gated_norm 16 \
    --attn_out_gate perhead --hidden_act situ_glu --optimizer muon --doc_masking --compile ...
```

`--doc_masking` is required with recurrent layers: without document boundaries the recurrent state would
flow from one packed document into the next, so the trainer refuses to start.

| feature | flags |
|:--|:--|
| data | memory-mapped shards with SHA-256 checksums; sampler without repeats (`--sampler permutation`); several tokenized folders with weights (`--data_mix`); best-fit packed windows (`--pack keep|drop`, [`data/`](../data/README.md)) |
| schedules | `--lr_schedule cosine`, `wsd` (warmup-stable-decay), `constant` |
| checkpoint merge | with `--lr_schedule constant`, weight-only snapshots every `--wsm_every_tok` tokens or inside `--wsm_window START:END:EVERY`, then `bin.merge_wsm.py` |
| memory | loss computed in chunks, so the full logits never exist at once; activation checkpointing (`--grad_ckpt`) |
| checkpoints | Hugging Face format, `last` plus milestones (`--milestones`), `--keep_last N`, optional async upload to S3 |
| telemetry | `metrics.jsonl`, `train_steps.jsonl`, `status.json`, `heartbeat.json`; GPU power and energy from NVML (`--telemetry_s`); bits per byte on a set of frozen texts (`--bpb_set`); the model's continuation of fixed prompts (`--samples_prompts`); optional W&B (`--wandb`) |
| long runs | `--deadline` and SIGTERM/SIGUSR1 save `last` and exit cleanly; `--ckpt_every_min`; a non-finite gradient skips the update and `--max_nonfinite` stops the run without overwriting `last`; `--skip_tokens_at` resumes from an earlier checkpoint and steps over the data that followed it |
| optimiser | `--wd` on the matrices; `--wd_embed` for the token embedding |
| validation | a fixed set per data folder (`--val_windows`), the same for any number of GPUs |

### More than one GPU

```bash
# one node
accelerate launch --num_processes 4 --mixed_precision no training/bin.pretrain.py ...

# several nodes, run on each node
torchrun --nnodes $NUM_NODES --node_rank $NODE_RANK --nproc_per_node 4 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    training/bin.pretrain.py ... --resume auto
```

- **`--mixed_precision no` is deliberate.** bf16 is applied by the trainer around the forward pass;
  Accelerate's bf16 path would cast the full logits to fp32.
- **`--ckpt_per_node`** makes every node write its own `<out>/last`, for clusters without a shared disk.
- **Resume on a different number of GPUs:** the sampler position is counted in samples, so a run can continue
  on more or fewer GPUs without repeating or skipping data.
- **`--ddp_comm bf16`** halves the gradient traffic between nodes; `--dist_timeout_min` turns a hung
  collective into an error.

## Checkpoint merge

```bash
python training/bin.merge_wsm.py --snapshots_dir <out>/wsm --window_tok <tokens> --method mean --out <out>/merged
```

A uniform average of consecutive checkpoints of a constant-LR run is equivalent to a linear decay over that
window (WSM, arXiv 2507.17634), so the cooldown is chosen after training.

## Growing a model (beta)

```bash
python training/bin.upscale.py --model <out>/last --out <grown>
python training/bin.pretrain.py --init_from <grown> --data <tokenized_dir> ... --resume auto
```

`bin.upscale.py` makes the model twice as wide (d_model, heads, KDA heads, d_ff; same depth and vocabulary) with the
same function and checks it on a batch before writing; `--init_from` starts a run from it. How it works:
[`models/growth.py`](../models/growth.py); gates: [`eval/bin.gate_growth.py`](../eval/bin.gate_growth.py).

## Supervised fine-tuning

```bash
python training/bin.sft.py --base_model <pretrained_dir> --data sft_train.jsonl --epochs <n> --lr <lr> --bf16
```

| feature | detail |
|:--|:--|
| format | ChatML (`<\|im_start\|>role … <\|im_end\|>`) |
| loss mask | only assistant tokens and the closing `<\|im_end\|>` carry gradient |
| stopping | extra weight on `<\|im_end\|>`, so the model learns to end its turn |
| optimiser | AdamW or Muon (`--optimizer`) |
