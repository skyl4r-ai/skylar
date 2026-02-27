"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | CTO of Sophia AI | 2026
=================================================================

Rotary Positional Embedding (RoPE) — used in LLaMA, Mistral, etc.
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) — used in LLaMA, Mistral, etc."""

    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    """Apply rotary embeddings to Q and K."""
    cos = cos.to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
