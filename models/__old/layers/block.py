"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | 2026
=================================================================

Transformer block — shared building block for decoder and embedder.

The block itself is mask-agnostic: the caller decides causal vs
bidirectional by passing (or not passing) block_mask / attention_mask.

  Decoder  → passes causal mask → autoregressive
  Embedder → passes nothing (or padding mask) → bidirectional
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .norm import RMSNorm
from .attention import CausalSelfAttention
from .ffn import FeedForward


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer block.

    Architecture (identical to Qwen3, LLaMA 3, Mistral):
        x → RMSNorm → Attention → residual
          → RMSNorm → SwiGLU FFN → residual

    Args:
        config: model config with d_model, n_heads, n_kv_heads, etc.
    """

    def __init__(self, config: object) -> None:
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        block_mask: object | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            x:              (B, T, D) hidden states.
            kv_cache:       (K, V) from previous step — generation only.
            block_mask:     FlexAttention BlockMask — packed causal training.
            attention_mask: (B, 1, T, T) dense additive mask — fallback.
            use_cache:      return updated KV cache.

        Returns:
            (hidden_states, kv_cache | None)
        """
        attn_out, new_cache = self.attn(
            self.ln1(x),
            kv_cache=kv_cache,
            block_mask=block_mask,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache
