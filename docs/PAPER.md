# Skylar: A Modern Decoder-Only NanoTransformer with FlexAttention Masking, Grouped Query Attention and Rotary Position Embeddings

**Abstract** — We present Skylar, a decoder-only Transformer language model built from first principles in PyTorch. Skylar implements the same architectural blueprint adopted by leading open-weight models (LLaMA 2/3, Mistral, Qwen3) while remaining fully transparent: every component — from normalization to positional encoding to the gating mechanism — is written explicitly without relying on high-level library abstractions. The architecture features RMSNorm pre-normalization, Rotary Position Embeddings (RoPE), SwiGLU-activated feed-forward networks, Grouped Query Attention (GQA) with QK-Norm, explicit per-head dimensionality (Qwen3-style rectangular projections), Flash Attention via PyTorch's scaled dot-product kernel, FlexAttention block-sparse document masking for zero-overhead packed sequence training, a KV-cache for efficient autoregressive generation, and µP (Maximal Update Parameterization) for hyperparameter transfer from small proxy models to arbitrary target widths. All components are integrated within the HuggingFace `PreTrainedModel` interface, enabling native serialization, loading, and hub distribution. We describe the architecture, its design rationale, the parameter-count scaling family from 6M to 128B parameters — with the 4B through 32B presets verified identical to Qwen3's published configurations — and the three-stage training pipeline (pre-training, supervised fine-tuning, inference). Skylar demonstrates that a production-grade LLM architecture can be reproduced transparently and scaled across five orders of magnitude with a single, unified codebase.

---

## 1. Introduction

Recent advances in large language models have converged on a remarkably consistent architecture: a decoder-only Transformer with pre-normalization, rotary positional encodings, gated feed-forward networks, and grouped attention heads (Touvron et al., 2023; Jiang et al., 2023; Yang et al., 2025). Despite this convergence, most production models are distributed as opaque checkpoints, and the few open-source implementations wrap significant complexity inside library-level abstractions that obscure the underlying mechanics.

Skylar addresses this gap. It is a GPT-style decoder-only Transformer in which every subcomponent — the normalization layers, the rotary embedding computation, the grouped attention mechanism, the gated activation function, the document masking logic, the µP scaling rules — is implemented as an explicit, self-contained PyTorch module. The goal is threefold: (1) to provide a faithful, auditable reproduction of the modern LLM architecture, (2) to serve as a practical, scalable foundation for training language models from scratch, and (3) to enable efficient hyperparameter search via µP transfer from small proxy models to large targets.

The model is fully compatible with HuggingFace Transformers, inheriting `PreTrainedModel` and `PretrainedConfig` so that standard workflows — `save_pretrained`, `from_pretrained`, `push_to_hub` — work natively. A family of twelve preset configurations spans from a 6M-parameter debug model to a 128B-parameter frontier-scale model, all sharing the same code path. The 4B through 32B presets reproduce the exact architectures of Qwen3-4B, Qwen3-8B, Qwen3-14B, and Qwen3-32B as published on HuggingFace, differing only in vocabulary size.

## 2. Architecture

Skylar follows the decoder-only Transformer paradigm (Radford et al., 2018; 2019). An input sequence of token indices is mapped to a continuous representation via a learned embedding table, processed through a stack of identical Transformer blocks, and projected back to vocabulary logits by a linear head. The architecture diverges from the original GPT formulation in eight key ways, each motivated by empirical findings from the 2023–2025 generation of open models: (1) RMSNorm pre-normalization, (2) Rotary Position Embeddings, (3) SwiGLU feed-forward networks, (4) Grouped Query Attention with explicit head dimensionality, (5) QK-Norm, (6) FlexAttention document masking for packed training, (7) µP for hyperparameter transfer, and (8) three-path attention dispatch.

### 2.1. Model Overview

The forward pass proceeds as follows:

1. **Token Embedding.** Input token IDs $x \in \mathbb{Z}^{B \times T}$ are mapped to $\mathbf{E} \in \mathbb{R}^{B \times T \times d}$ by a learned embedding matrix $W_e \in \mathbb{R}^{V \times d}$, where $V$ is the vocabulary size and $d$ is the model dimension. A dropout layer is applied after embedding.

2. **Transformer Blocks.** The embedded representation passes through $L$ identical decoder blocks (Sections 2.2–2.6). When gradient checkpointing is enabled and the model is in training mode (with no KV-cache), activations are recomputed during the backward pass to reduce VRAM usage.

3. **Final Normalization.** The output of the last block is normalized by a final RMSNorm layer.

4. **Language Model Head.** A linear projection $W_{lm} \in \mathbb{R}^{d \times V}$ maps the normalized hidden states to vocabulary logits. Optionally, $W_{lm}$ shares weights with $W_e$ (weight tying). When µP is active, the logits are scaled by $1 / m_w$ where $m_w = d / d_{\text{base}}$ is the width multiplier, ensuring that logit magnitudes remain stable as model width grows (Section 3).

5. **Loss.** When target labels are provided, the cross-entropy loss is computed with `ignore_index=-100`, allowing selective masking of non-target tokens (critical for SFT loss masking).

### 2.2. RMSNorm (Pre-Normalization)

Each Transformer block applies normalization *before* the attention and feed-forward sublayers (pre-norm residual connection), following the convention established by GPT-3 and adopted universally by LLaMA, Mistral, and Qwen.

Skylar uses Root Mean Square Layer Normalization (Zhang & Sennrich, 2019) instead of the standard LayerNorm. Given an input vector $\mathbf{x} \in \mathbb{R}^d$:

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

where $\boldsymbol{\gamma} \in \mathbb{R}^d$ is a learnable scale parameter and $\epsilon = 10^{-6}$. Unlike LayerNorm, RMSNorm does not compute or subtract the mean, reducing computation and improving numerical stability at scale.

The implementation casts the input to `float32` for the normalization computation and casts back to the input dtype afterwards, ensuring stability under mixed-precision training.

### 2.3. Rotary Position Embeddings (RoPE)

Skylar uses Rotary Position Embeddings (Su et al., 2024) to encode positional information, replacing both sinusoidal and learned positional embeddings.

For a query or key vector $\mathbf{q} \in \mathbb{R}^{d_h}$ at position $m$, RoPE applies a rotation in the complex plane:

$$\text{RoPE}(\mathbf{q}, m) = \mathbf{q} \odot \cos(m\boldsymbol{\theta}) + \text{rotate\_half}(\mathbf{q}) \odot \sin(m\boldsymbol{\theta})$$

where $\boldsymbol{\theta}_i = \text{base}^{-2i/d_h}$ for $i = 0, 1, \ldots, d_h/2 - 1$, and `rotate_half` splits the vector into two halves and negates the first before concatenation.

The base frequency $\theta_{\text{base}}$ is configurable per preset (parameter `rope_theta`), following the observation that higher base values improve extrapolation to longer sequences. Skylar uses two distinct strategies:

| Preset Group | Context | `rope_theta` | Rationale |
|:--|:-:|:-:|:--|
| `test` | 4K | 100,000 | Minimal context for testing |
| `small` | 8K | 500,000 | Short context, moderate extrapolation |
| `medium` | 16K | 1,000,000 | Standard Qwen3 base frequency |
| `large`, `xl` | 32K | 5,000,000 | Custom presets, extended base for long context |
| `4b`–`128b` | 32K–128K | 1,000,000 | Matches Qwen3 official value for all model sizes |

The Qwen3-aligned presets (`4b` through `128b`) uniformly use $\theta_{\text{base}} = 1{,}000{,}000$, which is the value used by all Qwen3 models regardless of context length. The custom presets (`large`, `xl`) use a higher base to explore extended context capabilities at intermediate model sizes.

The `RotaryEmbedding` module pre-computes and caches the cosine and sine tables up to `max_seq_len`. If a longer sequence is encountered at runtime, the cache is transparently rebuilt. RoPE requires that $d_h$ is even; the implementation enforces this with an assertion.

**Advantages over alternatives.** RoPE encodes relative position through the geometry of the rotation rather than through additive bias. This enables natural extrapolation beyond the training context length (especially with NTK-aware scaling), requires no additional parameters, and integrates cleanly with the KV-cache during autoregressive generation.

### 2.4. Causal Self-Attention with Grouped Query Attention

The attention module is the architectural core of Skylar and incorporates five interrelated mechanisms: Grouped Query Attention, explicit head dimensionality, QK-Norm, KV-caching, and three-path attention dispatch with FlexAttention support.

#### 2.4.1. Grouped Query Attention (GQA)

In standard Multi-Head Attention (MHA), query, key, and value projections all produce $h$ heads. Grouped Query Attention (Ainslie et al., 2023) reduces the key and value projections to $h_{kv} < h$ heads, where each KV head is shared across $r = h / h_{kv}$ query heads. This yields three regimes:

| Configuration | Condition | Memory Savings |
|:-:|:-:|:-:|
| MHA (standard) | $h_{kv} = h$ | 1$\times$ (baseline) |
| GQA | $1 < h_{kv} < h$ | $r\times$ reduction in KV-cache |
| MQA | $h_{kv} = 1$ | $h\times$ reduction in KV-cache |

For the SDPA attention paths, Skylar implements GQA via the `repeat_kv` function, which expands the $h_{kv}$ key/value heads to $h$ heads by repeating each KV head $r$ times before the dot-product computation:

```
k_expanded = repeat_kv(k, n_rep)   # (B, h_kv, T, d_h) → (B, h, T, d_h)
v_expanded = repeat_kv(v, n_rep)
```

The expansion is zero-copy when $r = 1$ (MHA fallback). For the FlexAttention path, GQA is handled natively via the `enable_gqa=True` flag, avoiding K/V expansion entirely and saving memory (Section 2.4.6).

#### 2.4.2. Explicit Head Dimension (Qwen3-Style)

In standard implementations, the per-head dimension is implicitly derived as $d_h = d / h$, requiring $d$ to be divisible by $h$. Skylar supports an explicit `d_head` parameter, decoupling the per-head dimension from the model width. When `d_head` is set and $h \cdot d_h \neq d$, the query and output projections become rectangular:

- $W_q \in \mathbb{R}^{d \times (h \cdot d_h)}$ — query projection (potentially rectangular)
- $W_k \in \mathbb{R}^{d \times (h_{kv} \cdot d_h)}$ — reduced key projection
- $W_v \in \mathbb{R}^{d \times (h_{kv} \cdot d_h)}$ — reduced value projection
- $W_o \in \mathbb{R}^{(h \cdot d_h) \times d}$ — output projection (potentially rectangular)

This design is adopted directly from Qwen3 (Yang et al., 2025), which uses $d_h = 128$ fixed across all model sizes. For example, Qwen3-4B has $d = 2560$ and $h = 32$, so $h \cdot d_h = 4096 > d$: the Q projection maps from 2560 to 4096 dimensions, providing greater attention capacity without increasing the model's hidden dimension. Similarly, Qwen3-32B has $d = 5120$ and $h = 64$, so $h \cdot d_h = 8192 > d$.

When `d_head` is not set (default), the standard relationship $d_h = d / h$ is used and projections are square.

#### 2.4.3. QK-Norm

Following Qwen3 (Yang et al., 2025), Skylar applies RMSNorm independently to the query and key tensors *after* linear projection and reshaping into per-head form, and *before* RoPE application:

$$\mathbf{q}' = \text{RMSNorm}(\mathbf{q}), \quad \mathbf{k}' = \text{RMSNorm}(\mathbf{k})$$

The normalization operates on the per-head dimension $d_h$. This prevents the magnitude of the dot product $\mathbf{q}'^T \mathbf{k}'$ from growing unboundedly as training progresses, which would cause the softmax to saturate and gradients to vanish. QK-Norm is controlled by the `qk_norm` configuration flag (default: `True`).

#### 2.4.4. KV-Cache

During autoregressive generation, Skylar maintains a per-layer cache of past key and value tensors. At each generation step, only the new token's key and value are computed and concatenated with the cached history:

```
k = torch.cat([k_prev, k_new], dim=2)
v = torch.cat([v_prev, v_new], dim=2)
```

RoPE positions are offset by the cache length to ensure correct positional encoding. The KV-cache reduces the per-token generation cost from $O(T^2)$ to $O(T)$ in the attention computation, where $T$ is the total sequence length.

The incremental decoding path enforces $T = 1$ when a KV-cache is present, preventing future-token leakage that would occur with chunked decoding under a non-causal mask. A comprehensive validation routine (`_validate_kv_cache`) checks tensor shapes, device placement, and dimensional consistency before each forward pass, converting silent shape mismatches into actionable error messages.

#### 2.4.5. µP Attention Scaling

Under standard parameterization (SP), the attention logits are scaled by $1 / \sqrt{d_h}$, which is the PyTorch default for `scaled_dot_product_attention`. Under µP (Section 3), the scale becomes $1 / d_h$ — a sharper normalization that keeps the attention entropy distribution stable as model width grows. This prevents the attention pattern from becoming either too uniform or too peaked at larger widths, which is critical for hyperparameter transferability. When µP is disabled (`mup_base_d_model=None`), the standard $1 / \sqrt{d_h}$ scaling is used.

#### 2.4.6. Three Attention Paths

The attention module dispatches to one of three computation paths, selected automatically based on the available inputs:

**Path 1: FlexAttention with block-sparse document mask.** When `document_ids` are provided and FlexAttention is available (PyTorch $\geq$ 2.5), a block-sparse mask is constructed via `create_block_mask`. The mask function enforces two constraints per token pair: (1) causal ordering ($q_{\text{idx}} \geq kv_{\text{idx}}$), and (2) same-document membership ($\text{doc}[q] = \text{doc}[kv]$). The resulting `BlockMask` uses $O(T)$ memory — independent of sequence length — compared to $O(T^2)$ for a dense mask. GQA is handled natively via FlexAttention's `enable_gqa=True` flag, avoiding the need to expand K/V tensors and saving memory proportional to the GQA ratio. This is the same technique used by LLaMA, Qwen, Mistral, and OLMo for packed sequence training at scale.

**Path 2: SDPA with dense attention mask (fallback).** When `document_ids` are provided but FlexAttention is unavailable, a dense block-diagonal mask is constructed as an additive attention bias: $0.0$ for allowed positions, $-\infty$ for blocked positions. This requires $O(T^2)$ memory and becomes impractical for sequence lengths beyond ~8K. K/V heads are expanded via `repeat_kv` for GQA.

**Path 3: Standard causal SDPA.** The default path for non-packed training and autoregressive generation. PyTorch's `F.scaled_dot_product_attention` with `is_causal=True` (training/prefill) or `is_causal=False` (cached generation) dispatches to FlashAttention-2 (Dao, 2023) or memory-efficient attention kernels when available. K/V heads are expanded via `repeat_kv` for GQA.

**Note on attention dropout.** The FlexAttention path (Path 1) does not currently apply attention-probability dropout. When using FlexAttention with `dropout > 0`, the model raises a `NotImplementedError` to prevent silent behavior differences between training paths.

### 2.5. SwiGLU Feed-Forward Network

The feed-forward sublayer uses the SwiGLU activation (Shazeer, 2020), which combines a gated linear unit with the SiLU (Swish) activation:

$$\text{FFN}(\mathbf{x}) = W_2 \big(\text{SiLU}(W_1 \mathbf{x}) \odot W_3 \mathbf{x}\big)$$

where $W_1, W_3 \in \mathbb{R}^{d \times d_{ff}}$ and $W_2 \in \mathbb{R}^{d_{ff} \times d}$. The gating operation ($W_3$) adds a third weight matrix compared to standard FFNs but consistently improves quality per FLOP (Shazeer, 2020; Touvron et al., 2023). A dropout layer is applied after $W_2$.

### 2.6. Transformer Block

Each of the $L$ Transformer blocks follows the pre-norm residual pattern:

$$\mathbf{h} = \mathbf{x} + \text{Attention}(\text{RMSNorm}(\mathbf{x}))$$
$$\mathbf{y} = \mathbf{h} + \text{FFN}(\text{RMSNorm}(\mathbf{h}))$$

This ordering (normalize-then-sublayer) is more stable than post-norm during training at scale and is standard across LLaMA, Mistral, Qwen, and Gemma.

### 2.7. Weight Initialization

Weights are initialized from $\mathcal{N}(0, \sigma)$ with initialization scaling rules that depend on whether µP is active.

**Standard Parameterization (SP).** When `mup_base_d_model` is `None`:

- All linear weights: $\sigma = 0.02$
- Embedding weights: $\sigma = 0.02$
- Residual output projections ($W_o$, $W_2$): $\sigma_{\text{residual}} = \frac{0.02}{\sqrt{2L}}$

The residual scaling follows GPT-2 and accounts for the accumulation of residual contributions across $L$ layers, preventing the variance of activations from growing with depth.

**µP (Maximal Update Parameterization).** When `mup_base_d_model` is set, initialization scales with the width multiplier $m_w = d / d_{\text{base}}$ (Section 3):

- Hidden linear weights: $\sigma = \frac{0.02}{\sqrt{m_w}}$ — scales down as width grows
- Embedding weights: $\sigma = 0.02$ — unchanged (input parameter in µP theory)
- Residual output projections: $\sigma_{\text{residual}} = \frac{0.02}{\sqrt{2L} \cdot \sqrt{m_w}}$

When weight tying is active, the LM head (which shares the embedding weight tensor) is not re-initialized, preserving the embedding initialization.

### 2.8. Weight Tying

When `tie_weights=True` (default for $\leq$ 4B presets), the token embedding matrix $W_e$ and the language model head $W_{lm}$ share the same parameter tensor. This reduces total parameter count — significantly so for smaller models where the embedding constitutes a large fraction of parameters — and provides an implicit regularization that encourages the embedding and output spaces to remain aligned.

Following the Qwen3 convention, weight tying is disabled (`tie_weights=False`) for models at the 8B scale and above. At these scales, the embedding represents a small fraction of total parameters, and separate embedding and LM head weights provide additional model capacity with negligible relative parameter overhead. Skylar's `8b`, `14b`, `32b`, `64b`, `96b`, and `128b` presets all use `tie_weights=False`, matching the corresponding Qwen3 configurations.

## 3. µP: Maximal Update Parameterization

### 3.1. Motivation

Hyperparameter tuning is one of the most expensive aspects of training large language models. The learning rate, weight decay, batch size, and other hyperparameters that work well for a small model often fail catastrophically at larger scales — a problem known as hyperparameter non-transferability under standard parameterization (SP).

Maximal Update Parameterization (µP) (Yang et al., 2022) provides a principled solution: by scaling initialization, learning rates, and attention logits according to the model width, optimal hyperparameters found on a small *proxy* model transfer directly to a larger *target* model. This technique has been adopted by Cerebras, EleutherAI, and DeepSeek for production-scale training.

### 3.2. Scaling Rules

µP is activated by setting `mup_base_d_model` in the configuration to the hidden dimension of the proxy model. The width multiplier is $m_w = d_{\text{target}} / d_{\text{base}}$. Skylar implements four scaling rules:

| Component | SP (Standard) | µP |
|:--|:-:|:-:|
| Hidden weight init | $\mathcal{N}(0, 0.02)$ | $\mathcal{N}(0, 0.02 / \sqrt{m_w})$ |
| Embedding init | $\mathcal{N}(0, 0.02)$ | $\mathcal{N}(0, 0.02)$ (unchanged) |
| Attention logit scale | $1 / \sqrt{d_h}$ | $1 / d_h$ |
| Output logit scale | $1$ | $1 / m_w$ |
| Hidden weight LR | `base_lr` | `base_lr` / $m_w$ |
| Embedding LR | `base_lr` | `base_lr` (unchanged) |

The rationale for each rule derives from the theory of tensor programs (Yang et al., 2022):

- **Init scaling** ($1/\sqrt{m_w}$): keeps the initial forward-pass signal magnitude constant across widths. Without this, wider models would have larger initial activations.
- **Attention scaling** ($1/d_h$ instead of $1/\sqrt{d_h}$): ensures the entropy of the attention distribution remains stable as width grows. Under SP, wider models tend to produce sharper attention patterns initially, biasing early training dynamics.
- **Output scaling** ($1/m_w$): prevents the logit magnitude from growing with width, which would distort the loss landscape.
- **LR scaling** ($\text{lr}/m_w$ for hidden weights): ensures that the magnitude of weight updates relative to weight magnitudes remains constant across widths.

### 3.3. Optimizer Parameter Groups

The `mup_param_groups(base_lr, weight_decay)` method constructs parameter groups with per-parameter learning rate scaling:

| Group | Parameters | Learning Rate | Weight Decay |
|:--|:--|:-:|:-:|
| `embedding` | Token embedding (and LM head if tied) | `base_lr` | 0.0 |
| `hidden` | All 2D weight matrices | `base_lr / m_w` | `weight_decay` |
| `nodecay` | 1D parameters (norms, biases) | `base_lr` | 0.0 |

When weight tying is active, `lm_head.weight` is the same tensor as `token_emb.weight` and naturally receives the embedding learning rate — which is the correct µP treatment for tied output parameters. When µP is disabled ($m_w = 1$), all groups receive `base_lr` and the method reduces to standard AdamW groups.

## 4. Configuration and Scaling

Skylar defines a family of twelve preset configurations through a unified `NanoTransformerConfig` class (extending HuggingFace `PretrainedConfig`). All presets share the same code path; only the hyperparameters change.

### 4.1. Configuration Parameters

| Parameter | Description | Default |
|:--|:--|:-:|
| `vocab_size` | Vocabulary size | 40,960 |
| `d_model` | Hidden dimension | 256 |
| `n_heads` | Number of query attention heads | 8 |
| `n_kv_heads` | Number of key/value heads (GQA) | `n_heads` |
| `d_head` | Per-head dimension (explicit Qwen3-style) | `None` (= `d_model / n_heads`) |
| `n_layers` | Number of Transformer blocks | 6 |
| `d_ff` | Feed-forward intermediate dimension | 512 |
| `max_seq_len` | Maximum sequence length | 1,024 |
| `dropout` | Dropout probability | 0.1 |
| `bias` | Use bias in linear layers | `False` |
| `tie_weights` | Tie embedding and LM head weights | `True` |
| `qk_norm` | Apply RMSNorm to Q and K | `True` |
| `rope_theta` | RoPE base frequency | 10,000.0 |
| `mup_base_d_model` | Proxy model width for µP (None = disabled) | `None` |

When `d_head` is `None`, the per-head dimension is derived as `d_model // n_heads`, requiring divisibility. When set explicitly, Q and O projections may be rectangular (Section 2.4.2).

### 4.2. Preset Family

The table below summarizes the twelve presets. Preset names for 4B and above follow the Qwen3 naming convention (parameter counts computed with Qwen3's vocabulary of 151,936 tokens); the "Params" column shows approximate counts with Skylar's vocabulary of 40,960 tokens.

| Preset | $d$ | $h$ | $h_{kv}$ | $d_h$ | $L$ | $d_{ff}$ | Context | ~Params | Tie | Comparable To |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:--|
| `test` | 128 | 4 | 4 | 32 | 4 | 256 | 4K | ~6M | Yes | Debug / CI |
| `small` | 512 | 8 | 4 | 64 | 8 | 1,024 | 8K | ~40M | Yes | DistilGPT-2 |
| `medium` | 768 | 12 | 4 | 64 | 12 | 2,048 | 16K | ~107M | Yes | GPT-2 Small |
| `large` | 1,024 | 16 | 4 | 64 | 28 | 2,816 | 32K | ~358M | Yes | GPT-2 Medium |
| `xl` | 1,536 | 16 | 4 | 96 | 32 | 5,120 | 32K | ~1.0B | Yes | TinyLLaMA / Qwen3-0.6B |
| `4b` | 2,560 | 32 | 8 | **128** | 36 | 9,728 | 32K | ~3.6B | Yes | **Qwen3-4B** (identical) |
| `8b` | 4,096 | 32 | 8 | **128** | 36 | 12,288 | 32K | ~7.3B | No | **Qwen3-8B** (identical) |
| `14b` | 5,120 | 40 | 8 | **128** | 40 | 17,408 | 32K | ~13.2B | No | **Qwen3-14B** (identical) |
| `32b` | 5,120 | 64 | 8 | **128** | 64 | 25,600 | 128K | ~29.6B | No | **Qwen3-32B** (identical) |
| `64b` | 8,192 | 64 | 8 | 128 | 72 | 28,672 | 128K | ~61B | No | LLaMA-2-70B scale |
| `96b` | 10,240 | 80 | 8 | 128 | 80 | 32,768 | 128K | ~99B | No | — |
| `128b` | 11,264 | 88 | 8 | 128 | 92 | 32,768 | 128K | ~128B | No | GPT-4 class |

**Bold $d_h$ values** denote explicit `d_head=128` (Qwen3-style). For the `4b` and `32b` presets, $h \cdot d_h \neq d$, producing rectangular Q/O projections (Section 2.4.2):

- `4b`: $d = 2560$, $h \cdot d_h = 32 \times 128 = 4096$ → Q: $\mathbb{R}^{2560 \times 4096}$
- `32b`: $d = 5120$, $h \cdot d_h = 64 \times 128 = 8192$ → Q: $\mathbb{R}^{5120 \times 8192}$

### 4.3. Scaling Observations

Several patterns emerge across the preset family:

- **GQA ratio increases with model size**: $r = 1$ (MHA) for test, $r = 2$ for small, $r = 3$ for medium, $r = 4$ for large/xl/4b/8b, $r = 5$ for 14b, $r = 8$ for 32b/64b, $r = 10$ for 96b, $r = 11$ for 128b. This reflects the increasing relative cost of the KV-cache at scale, where more aggressive KV sharing yields greater memory and throughput benefits.
- **Context length scales with capacity**: 4K for test, 8K for small, 16K for medium, 32K for large through 14b, and 128K for 32b+.
- **Weight tying follows Qwen3**: enabled for $\leq$ 4B, disabled for $\geq$ 8B. At small scales, the shared embedding constitutes a significant fraction of parameters; at large scales, separate weights provide more capacity with minimal relative overhead.
- **All presets use `bias=False` and `qk_norm=True`**, consistent with modern best practices.
- **Fixed head dimension for Qwen3-aligned presets**: all 4b–128b presets use $d_h = 128$, allowing direct architectural comparison with Qwen3 published models.

### 4.4. Architectural Parity with Qwen3

The `4b`, `8b`, `14b`, and `32b` presets reproduce the exact architectures of their corresponding Qwen3 models, as verified against the `config.json` files published on HuggingFace. Only the vocabulary size differs (40,960 vs. 151,936).

| Component | Skylar (`4b`–`32b`) | Qwen3 (4B–32B) |
|:--|:-:|:-:|
| Type | Decoder-only | Decoder-only |
| Normalization | RMSNorm (Pre-Norm) | RMSNorm (Pre-Norm) |
| Positional encoding | RoPE ($\theta = 10^6$) | RoPE ($\theta = 10^6$) |
| FFN activation | SwiGLU | SwiGLU |
| Attention | GQA (8 KV heads) | GQA (8 KV heads) |
| Per-head dim | 128 (explicit) | 128 (explicit) |
| QK-Norm | RMSNorm on Q, K | RMSNorm on Q, K |
| KV-Cache | Yes | Yes |
| Flash Attention | PyTorch SDPA | Flash Attention 2 |
| Weight tying | Yes (4B) / No (8B+) | Yes (4B) / No (8B+) |
| Bias in Linear | No | No |

The architectures are functionally identical. Differences are limited to vocabulary size, training data, and compute scale. Additional Skylar features not present in the Qwen3 reference implementation — FlexAttention document masking and µP scaling — are orthogonal to the architecture and can be selectively enabled without altering the core model structure.

### 4.5. Training Compute Estimates

The following estimates are based on measured throughput (marked ✅) or projected throughput for representative hardware configurations. All times include typical failure-recovery overhead ("failure tax") for multi-node training.

| Preset | Chinchilla-Optimal Tokens | RTX 4090 (24GB) | RTX 5090 / PRO 6000 | 5× DGX H100 (40 GPU) |
|:--|:-:|:--|:--|:--|
| `small` (~40M) | ~800M | ~5h ✅ (100K tok/s) | — | — |
| `medium` (~107M) | ~2.1B | ~10h ✅ (51K tok/s) | ~13h (75K tok/s) | — |
| `large` (~358M) | ~7B | ~4 days (19K tok/s) | ~3 days (28K tok/s) | ~4h (500K tok/s) |
| `xl` (~1.0B) | ~20B | OOM | ~19 days (12K tok/s) | ~16h (700K tok/s) |
| `4b` (~3.6B) | ~80B | OOM | ~9 months (3.5K tok/s) | ~8 days (240K tok/s) |

Chinchilla-optimal token counts follow the scaling law $D \approx 20 \times N$ (Hoffmann et al., 2022), where $N$ is the parameter count. Larger presets (8B+) require multi-node clusters and are documented in the configuration source.

## 5. Training Pipeline

Skylar employs a three-stage training pipeline that mirrors the methodology used by production LLMs.

### 5.1. Stage 1: Pre-Training

The base model is trained on raw text corpora using a standard causal language modeling objective. The loss is the cross-entropy between the model's next-token predictions and the ground truth, computed across all non-padding positions.

**Optimization.** AdamW optimizer with cosine learning rate decay and linear warmup. Mixed-precision training (bf16/fp16) is used by default. Gradient accumulation and gradient clipping (max norm 1.0) are supported. When µP is active, `mup_param_groups()` provides per-parameter learning rate scaling (Section 3.3).

**Data.** Input can be raw text files, directories of text files, or HuggingFace datasets. A BPE tokenizer is trained on the corpus using the HuggingFace `tokenizers` library (Rust-backed for performance). Sequence packing eliminates padding waste by concatenating tokenized documents into fixed-length chunks. When packing is enabled and `document_ids` are provided, the model automatically applies document masking (Section 2.4.6) to prevent cross-document attention leakage.

**Efficiency.** Optional `torch.compile()` for kernel fusion; gradient checkpointing to trade compute for memory; pre-tokenized data loading from `.pt` files for I/O-bound scenarios; multi-GPU support via HuggingFace Accelerate (DDP/FSDP).

### 5.2. Stage 2: Supervised Fine-Tuning (SFT)

The pre-trained base model is fine-tuned on structured conversations in ChatML format:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

**Loss masking.** A critical design choice: the loss is computed *only* on assistant response tokens. All other tokens (system prompts, user messages, special tokens) are masked with `ignore_index=-100`. This focuses the model's learning capacity entirely on generating appropriate responses, rather than wasting capacity on reproducing prompts. The mask is generated by `create_loss_mask()` which parses the ChatML structure and identifies assistant spans.

### 5.3. Stage 3: Inference

Autoregressive generation supports temperature scaling, top-$k$ filtering, top-$p$ (nucleus) sampling, repetition penalty, greedy decoding, and early stopping on EOS tokens. The KV-cache is used by default for efficient token-by-token generation.

**Greedy decoding.** When temperature $\leq 0$, the model selects the token with the highest logit (argmax), producing fully deterministic output.

**Temperature scaling.** For temperature $> 0$, logits are divided by the temperature parameter before sampling. Lower temperatures produce more deterministic output; higher temperatures increase diversity.

**Top-$k$ filtering.** Only the $k$ highest-probability tokens are kept; all others are set to $-\infty$.

**Top-$p$ (nucleus) filtering.** Tokens are sorted by probability in descending order. The cumulative probability is computed, and all tokens beyond the $p$ threshold are masked.

**Repetition penalty.** A penalty factor is applied to tokens that have already appeared in the generated sequence (Keskar et al., 2019). For each previously-seen token: if its logit is positive, it is divided by the penalty; if negative, it is multiplied. A penalty of 1.0 has no effect; values in the range 1.1–1.3 are recommended for reducing repetitive output.

**Context management.** If the prompt exceeds `max_seq_len`, it is truncated to the last `max_seq_len` tokens with a warning. The maximum number of generated tokens is clamped to avoid exceeding the context window.

**Sampling.** After filtering, a softmax is applied and the next token is sampled from the resulting distribution via multinomial sampling. Generation stops when an EOS token is produced or the maximum token count is reached. The EOS token ID can be a single integer or a list of integers for models with multiple stop tokens.

## 6. Implementation Details

### 6.1. HuggingFace Integration

`NanoTransformer` extends `PreTrainedModel` and declares:

- `config_class = NanoTransformerConfig` — links the model to its configuration
- `supports_gradient_checkpointing = True` — enables the HuggingFace checkpointing API
- `_tied_weights_keys = ["lm_head.weight"]` — declares which parameters are tied for correct serialization

Standard HuggingFace methods (`get_input_embeddings`, `get_output_embeddings`, `set_output_embeddings`) are implemented to support the full ecosystem (tokenizer integration, generation utilities, model parallelism).

### 6.2. Gradient Checkpointing

When enabled, activations are not stored during the forward pass through each Transformer block. Instead, they are recomputed during the backward pass. This reduces peak VRAM usage from $O(L)$ to approximately $O(\sqrt{L})$ at the cost of ~33% additional compute. The implementation uses PyTorch's `torch.utils.checkpoint.checkpoint` with `use_reentrant=False` for compatibility with modern autograd.

Gradient checkpointing is automatically disabled during generation (when `kv_cache is not None`) since no backward pass is needed.

### 6.3. KV-Cache Validation

The `_validate_kv_cache` method performs comprehensive runtime checks on the KV-cache structure before each forward pass:

- **Type**: verifies the cache is a list/tuple of length $L$ (one entry per layer)
- **Per-layer structure**: each entry is a (key, value) tuple of tensors
- **Device**: all cache tensors reside on the same device as the input
- **Shape**: each tensor has shape $(B, h_{kv}, T_{\text{cache}}, d_h)$ with correct batch size, number of KV heads, and head dimension
- **Consistency**: key and value sequence lengths match within each layer

These checks convert silent shape mismatches — which would otherwise produce incorrect attention outputs or cryptic CUDA errors — into descriptive `ValueError` messages with the expected and actual dimensions.

### 6.4. Numerical Stability

Several design choices prioritize numerical stability:

- RMSNorm casts inputs to `float32` before computing the norm, then casts back
- QK-Norm prevents attention score explosion at large scale
- Scaled residual initialization ($\sigma \propto 1/\sqrt{2L}$, further scaled by $1/\sqrt{m_w}$ under µP) controls variance growth across depth and width
- RoPE frequencies are pre-computed in `float32`
- µP attention scaling ($1/d_h$) provides tighter logit control than standard $1/\sqrt{d_h}$
- Autocast-aware dtype detection for dense packing masks under mixed-precision training

### 6.5. Parameter Counting

The `count_params()` method provides exact parameter counts with an option to exclude the embedding table. This is useful because weight tying means the embedding parameters are shared — reporting "non-embedding" parameters gives a clearer picture of the model's unique learned capacity.

## 7. Design Rationale

### 7.1. Why RMSNorm over LayerNorm?

RMSNorm eliminates the mean computation and centering step of LayerNorm. Empirically, the mean subtraction provides negligible benefit for Transformer language models while adding compute and potential numerical issues. RMSNorm has become the de facto standard following its adoption in LLaMA (Touvron et al., 2023).

### 7.2. Why RoPE over Learned Positional Embeddings?

Learned positional embeddings are fixed at training time — the model cannot generalize to sequence lengths beyond what it was trained on. RoPE encodes position through rotation in the complex plane, which naturally captures relative position. With techniques like NTK-aware scaling, RoPE allows context extension post-training without retraining.

### 7.3. Why SwiGLU over GELU?

SwiGLU (Shazeer, 2020) empirically outperforms GELU and ReLU activations at equal FLOP budgets. The gating mechanism ($W_3$) introduces a multiplicative interaction that allows the network to learn more expressive feature combinations. The additional parameter matrix ($W_3$) increases the FFN parameter count by ~50%, but the quality improvement per FLOP justifies the cost.

### 7.4. Why GQA over Standard MHA?

At inference time, the KV-cache grows linearly with sequence length and number of KV heads. For a 32B model with 128K context, standard MHA would require hundreds of gigabytes of KV-cache alone. GQA reduces this by a factor of $r = h/h_{kv}$ (ranging from 4$\times$ to 11$\times$ across Skylar presets) with negligible quality degradation (Ainslie et al., 2023). This makes long-context generation feasible on practical hardware.

### 7.5. Why QK-Norm?

Without normalization, the dot product $\mathbf{q}^T \mathbf{k}$ can grow unboundedly during training, causing the softmax to concentrate on a single position (attention collapse). QK-Norm constrains the norms of $\mathbf{q}$ and $\mathbf{k}$, bounding the attention logits and maintaining gradient flow. This is the key architectural difference between Qwen2 and Qwen3 (Yang et al., 2025).

### 7.6. Why Explicit Head Dimension?

Decoupling $d_h$ from $d / h$ allows models to maintain a fixed attention subspace dimensionality (128) across all scales, regardless of the model width and head count. This has two advantages: (1) it provides greater attention capacity when $h \cdot d_h > d$ (as in the 4b and 32b presets), and (2) it simplifies cross-scale comparison and enables consistent RoPE frequency allocation across model sizes. Qwen3 adopted this design for all model sizes from 0.6B to 235B.

### 7.7. Why FlexAttention for Document Masking?

Packed sequence training — concatenating multiple documents into a single training sequence to eliminate padding waste — requires attention masking to prevent cross-document information leakage. The naive approach (dense $T \times T$ mask) consumes $O(T^2)$ memory, which is prohibitive for long sequences. FlexAttention constructs a block-sparse mask representation using a simple Python function, achieving $O(T)$ memory with zero runtime overhead compared to standard causal attention. This is the production technique used by all major open-weight model trainers.

### 7.8. Why µP?

Training a single run of a 4B+ parameter model costs thousands of GPU-hours. If the hyperparameters are suboptimal, that compute is wasted. µP enables a principled workflow: tune all hyperparameters on a small proxy model (e.g., 50M parameters, trainable in minutes on a single GPU), then transfer them directly to the target scale. The theory guarantees that the optimal learning rate, weight decay, and other HPs are preserved across widths, eliminating the need for expensive grid searches at scale.

### 7.9. Why Weight Tying?

The embedding layer maps tokens to semantic vectors; the LM head maps semantic vectors back to token probabilities. Both operate on the boundary between discrete token space and continuous representation space. Sharing their weights enforces consistency between these two mappings and reduces parameter count. The benefit is most pronounced for small models; for large models ($\geq$ 8B), the embedding fraction is small enough that untying provides net capacity gains.

### 7.10. Why ChatML?

ChatML is the most widely adopted chat template format, used by Qwen, Mistral, and OpenAI models. Its structured delimiters (`<|im_start|>`, `<|im_end|>`) enable reliable parsing for loss masking and are compatible with the broadest range of tools and inference frameworks.

## 8. Comparison with Related Work

| Feature | GPT-2 | LLaMA 2 | Mistral 7B | Qwen3 | **Skylar** |
|:--|:-:|:-:|:-:|:-:|:-:|
| Normalization | LayerNorm | RMSNorm | RMSNorm | RMSNorm | **RMSNorm** |
| Position encoding | Learned | RoPE | RoPE | RoPE | **RoPE** |
| FFN activation | GELU | SwiGLU | SwiGLU | SwiGLU | **SwiGLU** |
| Attention | MHA | GQA | GQA | GQA | **GQA** |
| QK-Norm | No | No | No | Yes | **Yes** |
| Explicit $d_h$ | No | No | No | Yes (128) | **Yes (128)** |
| Residual order | Post-norm | Pre-norm | Pre-norm | Pre-norm | **Pre-norm** |
| KV-Cache | No | Yes | Yes | Yes | **Yes** |
| FlexAttention | No | No | No | No† | **Yes** |
| µP | No | No | No | No | **Yes** |
| Bias | Yes | No | No | No | **No** |
| Weight tying | Yes | No | No | Configurable | **Configurable** |

† Qwen3's published implementation does not include FlexAttention, though the underlying training infrastructure may use equivalent techniques.

Skylar adopts the full modern stack (RMSNorm + RoPE + SwiGLU + GQA + QK-Norm + explicit $d_h$) and extends it with FlexAttention document masking and µP hyperparameter transfer, placing it architecturally at parity with Qwen3 while providing additional training-efficiency features.

## 9. Conclusion

Skylar is a transparent, production-grade implementation of the modern decoder-only Transformer architecture. By writing every component explicitly — from the RMSNorm computation to the GQA head expansion to the RoPE frequency tables to the FlexAttention mask function to the µP scaling rules — the codebase serves both as a practical training framework and as educational reference material.

The architecture achieves exact structural parity with Qwen3 (4B through 32B), demonstrating that the core design of modern LLMs is fully reproducible from first principles. The addition of FlexAttention document masking provides zero-overhead packed sequence training at any sequence length, and µP enables efficient hyperparameter discovery by transferring optimal settings from small proxy models to target widths.

The unified codebase scales from 6M to 128B parameters without code changes. The HuggingFace integration enables seamless deployment within the established ecosystem. The three-path attention dispatch adapts automatically to the available hardware and training configuration, requiring no manual intervention from the user.

Skylar represents both a starting point and a methodology: the architecture is complete, the scaling pathway is defined, and the µP framework reduces the cost of finding optimal training hyperparameters by orders of magnitude. The path to a competitive model requires data, compute, and iteration — all of which are facilitated by the transparency and modularity of the implementation.

---

## References

- Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. *arXiv:2305.13245*.

- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *arXiv:2307.08691*.

- He, H. & Guessous, D. (2024). FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention. *PyTorch Blog*, August 2024.

- Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training Compute-Optimal Large Language Models. *arXiv:2203.15556*.

- Jiang, A. Q., Sablayrolles, A., Mensch, A., et al. (2023). Mistral 7B. *arXiv:2310.06825*.

- Keskar, N. S., McCann, B., Varshney, L. R., Xiong, C., & Socher, R. (2019). CTRL: A Conditional Transformer Language Model for Controllable Generation. *arXiv:1909.05858*.

- Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving Language Understanding by Generative Pre-Training. *OpenAI Technical Report*.

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Technical Report*.

- Shazeer, N. (2020). GLU Variants Improve Transformer. *arXiv:2002.05202*.

- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., & Liu, Y. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. *Neurocomputing*, 568, 127063.

- Touvron, H., Lavril, T., Izacard, G., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv:2302.13971*.

- Yang, A., Yang, B., Zhang, B., et al. (2025). Qwen3 Technical Report. *arXiv:2505.09388*.

- Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., & Gao, J. (2022). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. *arXiv:2203.03466*.

- Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer Normalization. *NeurIPS 2019*.

---

*Skylar v3.0 — February 2026*
