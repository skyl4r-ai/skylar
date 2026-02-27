"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | CTO of Sophia AI | 2026
=================================================================

GPT-style decoder-only Transformer — PyTorch + HuggingFace.

Every component is written explicitly (no nn.TransformerDecoder),
so you can see, modify, and learn from every line.

Supports:
  - FlexAttention (PyTorch 2.5+) for zero-overhead document masking
    during packed sequence training — same technique used by LLaMA,
    Qwen, Mistral, and OLMo at scale.
  - µP (Maximal Update Parameterization) for HP transfer from small
    proxy models to large targets — used by Cerebras, EleutherAI,
    DeepSeek. Tune HPs on a 50M proxy, transfer to 4B.
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, __version__ as _tf_version

from models.config import NanoTransformerConfig
from models.layers.block import TransformerBlock
from models.layers.norm import RMSNorm
from models.layers.attention import (
    HAS_FLEX_ATTENTION,
    create_document_block_mask,
    make_packing_mask,
)
from models.layers.kv_cache import validate_kv_cache

logger = logging.getLogger(__name__)


class NanoTransformer(PreTrainedModel):
    """
    GPT-style Decoder-Only Transformer.

    Modern architecture choices:
      - RMSNorm (instead of LayerNorm)
      - RoPE (instead of sinusoidal/learned positional embeddings)
      - SwiGLU FFN (instead of GELU)
      - GQA — Grouped Query Attention (fewer KV heads → less memory)
      - KV-Cache for fast generation
      - Flash Attention (via PyTorch scaled_dot_product_attention)
      - FlexAttention document masking for packed training (zero overhead)
      - Optional weight tying
      - µP for HP transfer across model widths

    Document masking (packed training):
      Pass document_ids to forward() and the model automatically:
      - Uses FlexAttention block_mask if available (O(T) memory, zero overhead)
      - Falls back to dense attention_mask if not (O(T²) memory)
      - Does nothing during generation (kv_cache mode)

    µP (Maximal Update Parameterization):
      Set config.mup_base_d_model to the proxy model width. Then:
      - Hidden weight init scales as 0.02 / √(width_mult)
      - Attention logits scale as 1/d_head (not 1/√d_head)
      - LM head output scales by 1/width_mult
      - Use mup_param_groups() for per-parameter LR scaling in optimizer
      Net effect: HPs tuned on proxy transfer to any target width.

    HuggingFace compatible:
      - model.save_pretrained("path")
      - NanoTransformer.from_pretrained("path")
    """

    config_class = NanoTransformerConfig
    supports_gradient_checkpointing = True
    _tied_weights_keys = (
        {"lm_head.weight": "token_emb.weight"}
        if int(_tf_version.split(".")[0]) >= 5
        else ["lm_head.weight"]
    )

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        # µP: output logit scaling
        # Divides lm_head output by width_mult so that logit magnitude
        # stays constant as model width grows. When µP is off, this is 1.0.
        self._mup_output_alpha = 1.0 / config.mup_width_mult

        # ── Weight initialization ── v4|v5
        self.post_init()

        # Residual projection scaled by depth (GPT-2 style)
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        if config.mup_base_d_model is not None:
            residual_std /= math.sqrt(config.mup_width_mult)
        for pn, p in self.named_parameters():
            if pn.endswith("W_o.weight") or pn.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        """
        Weight initialization with µP scaling.

        SP (standard): all weights init to N(0, 0.02)
        µP:
          - Embedding:       N(0, 0.02)  — unchanged
          - Hidden weights:  N(0, 0.02 / √width_mult)  — scales down with width
          - 1D params:       unchanged (norms, biases)
        """
        std = 0.02

        if isinstance(module, nn.Linear):

            if self.config.tie_weights and module is self.lm_head:
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                return

            if self.config.mup_base_d_model is not None:
                std /= math.sqrt(self.config.mup_width_mult)
            nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Embedding init unchanged in µP — it's an "input" parameter
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.token_emb

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, **kwargs):
        """Ensure lm_head shares weights with token_emb after loading."""
        super().tie_weights(**kwargs)
        if self.config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def mup_param_groups(self, base_lr, weight_decay):
        """
        Build optimizer parameter groups with µP per-parameter LR scaling.

        Returns standard groups when µP is disabled.

        µP rules:
          - Embedding params:       lr = base_lr, wd = 0
          - 1D params (norms):      lr = base_lr, wd = 0
          - Hidden 2D weights:      lr = base_lr / width_mult, wd = weight_decay

        With tied weights, lm_head.weight IS token_emb.weight, so it
        naturally gets the embedding LR — correct for µP.
        """
        wm = self.config.mup_width_mult

        embedding_params = []
        decay_params = []
        nodecay_params = []

        embedding_ids = {id(self.token_emb.weight)}
        if self.config.tie_weights:
            embedding_ids.add(id(self.lm_head.weight))

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in embedding_ids:
                embedding_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        groups = [
            {"params": embedding_params, "lr": base_lr, "weight_decay": 0.0,
             "name": "embedding"},
            {"params": decay_params, "lr": base_lr / wm, "weight_decay": weight_decay,
             "name": "hidden"},
            {"params": nodecay_params, "lr": base_lr, "weight_decay": 0.0,
             "name": "nodecay"},
        ]

        return [g for g in groups if g["params"]]

    def _validate_kv_cache(self, kv_cache, batch_size, x_device):
        validate_kv_cache(kv_cache, self.blocks, batch_size, x_device)

    def forward(self, input_ids, labels=None, kv_cache=None, document_ids=None,
                attention_mask=None, use_cache=False):
        """
        Args:
            input_ids:      (B, T) token indices
            labels:         (B, T) target token indices (optional, for training)
            kv_cache:       list of (k, v) tuples per layer (for generation)
            document_ids:   (B, T) document IDs for packed training (optional).
                            Enables automatic document masking via FlexAttention.
            attention_mask: (B, 1, T, T) dense additive mask (backward compat).
            use_cache: Bool if use cache.

        Returns:
            dict with 'logits', 'loss' (if labels), 'kv_cache'
        """
        B, T = input_ids.shape
        x = self.drop(self.token_emb(input_ids))

        if kv_cache is not None:
            self._validate_kv_cache(kv_cache, batch_size=B, x_device=x.device)

        # ── Build attention mask from document_ids ──
        block_mask = None
        if document_ids is not None and kv_cache is None:
            if HAS_FLEX_ATTENTION:
                block_mask = create_document_block_mask(
                    document_ids, self.config.n_heads, device=input_ids.device,
                )
            else:
                if input_ids.is_cuda and torch.is_autocast_enabled():
                    # PyTorch 2.x compat
                    if hasattr(torch, "get_autocast_dtype"):
                        mask_dtype = torch.get_autocast_dtype("cuda")
                    else:
                        mask_dtype = torch.get_autocast_gpu_dtype()
                else:
                    mask_dtype = x.dtype

                attention_mask = make_packing_mask(document_ids, dtype=mask_dtype)

        collect_cache = bool(use_cache or kv_cache is not None)
        new_cache = [] if collect_cache else None

        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            if self.gradient_checkpointing and self.training and kv_cache is None:
                x, cache_i = torch.utils.checkpoint.checkpoint(
                    block, x, layer_cache, block_mask, attention_mask, False,
                    use_reentrant=False,
                )
            else:
                x, cache_i = block(
                    x,
                    kv_cache=layer_cache,
                    block_mask=block_mask,
                    attention_mask=attention_mask,
                    use_cache=collect_cache,
                )

            if collect_cache:
                new_cache.append(cache_i)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # µP: scale output logits to keep magnitude stable across widths
        if self._mup_output_alpha != 1.0:
            logits = logits * self._mup_output_alpha

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss, "kv_cache": new_cache}

    def count_params(self, non_embedding=False):
        """Count parameters (optionally excluding embedding)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_emb.weight.numel()
        return n

    # ─────────────────────────────────────────────────────────────
    # Static generation
    # ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=200, temperature=0.8, top_k=50,
                 top_p=0.9, repetition_penalty=1.0, eos_token_id=None, use_cache=True):
        """
        Autoregressive generation with KV-cache, top-k, top-p, temperature,
        and repetition penalty.

        Args:
            input_ids:          (1, T) starting tokens
            max_new_tokens:     how many tokens to generate
            temperature:        sampling temperature (lower = more deterministic)
            top_k:              keep only top-k logits
            top_p:              nucleus sampling threshold
            repetition_penalty: penalize already-generated tokens (1.0 = no penalty,
                                >1.0 = discourage repeats, 1.1-1.3 recommended)
            eos_token_id:       stop generation at this token (int or list of ints)
            use_cache:          use KV-cache for faster generation

        Returns:
            (1, T + generated) generated token ids
        """
        assert input_ids.shape[0] == 1, f"generate() supports batch_size=1, got {input_ids.shape[0]}"

        if eos_token_id is None:
            eos_set = set()
        elif isinstance(eos_token_id, int):
            eos_set = {eos_token_id}
        else:
            eos_set = set(eos_token_id)

        max_seq_len = self.config.max_seq_len
        prompt_len = input_ids.shape[1]

        if prompt_len > max_seq_len:
            logger.warning(
                f"Prompt length ({prompt_len}) exceeds max_seq_len ({max_seq_len}). "
                f"Truncating to last {max_seq_len} tokens."
            )
            input_ids = input_ids[:, -max_seq_len:]
            prompt_len = max_seq_len

        max_gen = min(max_new_tokens, max_seq_len - prompt_len)
        if max_gen < max_new_tokens:
            logger.warning(
                f"Clamping max_new_tokens from {max_new_tokens} to {max_gen} "
                f"(max_seq_len={max_seq_len}, prompt={prompt_len})"
            )
        if max_gen <= 0:
            logger.warning("No room to generate — prompt fills entire context window.")
            return input_ids

        was_training = self.training
        self.eval()
        kv_cache = None

        for _ in range(max_gen):
            if use_cache and kv_cache is not None:
                idx_input = input_ids[:, -1:]
            else:
                idx_input = input_ids[:, -max_seq_len:]
                kv_cache = None

            out = self.forward(idx_input, kv_cache=kv_cache, use_cache=use_cache)
            logits = out["logits"][:, -1, :]

            if use_cache:
                kv_cache = out["kv_cache"]

            if repetition_penalty != 1.0:
                seen_tokens = input_ids[0].unique()
                score = logits[0, seen_tokens]
                logits[0, seen_tokens] = torch.where(
                    score > 0,
                    score / repetition_penalty,
                    score * repetition_penalty,
                )

            # DOPO (fix)
            if temperature <= 0:
                # Greedy: prendi il token col logit più alto, zero casualità
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = -float("inf")

                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[mask] = -float("inf")
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() in eos_set:
                break

        if was_training:
            self.train()
        return input_ids

    # ─────────────────────────────────────────────────────────────
    # Streaming generation
    # ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate_streaming(self, input_ids, max_new_tokens=200, temperature=0.8, top_k=50,
                           top_p=0.9, repetition_penalty=1.0, eos_token_id=None, use_cache=True):
        """
        Autoregressive generation that yields one token ID at a time.

        Same logic as generate() — KV-cache, top-k, top-p, temperature,
        repetition penalty — but yields instead of accumulating.

        Args:
            Same as generate().

        Yields:
            int: token ID at each step. Stops after eos or max_new_tokens.
        """
        assert input_ids.shape[0] == 1, f"generate_streaming() supports batch_size=1, got {input_ids.shape[0]}"

        if eos_token_id is None:
            eos_set = set()
        elif isinstance(eos_token_id, int):
            eos_set = {eos_token_id}
        else:
            eos_set = set(eos_token_id)

        max_seq_len = self.config.max_seq_len
        prompt_len = input_ids.shape[1]

        if prompt_len > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]
            prompt_len = max_seq_len

        max_gen = min(max_new_tokens, max_seq_len - prompt_len)
        if max_gen <= 0:
            return

        was_training = self.training
        self.eval()
        kv_cache = None

        for _ in range(max_gen):
            if use_cache and kv_cache is not None:
                idx_input = input_ids[:, -1:]
            else:
                idx_input = input_ids[:, -max_seq_len:]
                kv_cache = None

            out = self.forward(idx_input, kv_cache=kv_cache, use_cache=use_cache)
            logits = out["logits"][:, -1, :]

            if use_cache:
                kv_cache = out["kv_cache"]

            if repetition_penalty != 1.0:
                seen_tokens = input_ids[0].unique()
                score = logits[0, seen_tokens]
                logits[0, seen_tokens] = torch.where(
                    score > 0,
                    score / repetition_penalty,
                    score * repetition_penalty,
                )

            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = -float("inf")

                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[mask] = -float("inf")
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            token_id = next_token.item()

            yield token_id

            if token_id in eos_set:
                break

        if was_training:
            self.train()
