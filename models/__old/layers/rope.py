"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | 2026
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
        self.max_seq_len = max_seq_len
        # cos/sin are kept as PLAIN attributes (not registered buffers) and rebuilt
        # lazily from inv_freq. Registering them as non-persistent buffers is unsafe:
        # transformers v5 from_pretrained() does not restore non-persistent buffers,
        # leaving them uninitialized (NaN) — every forward of a LOADED checkpoint
        # (SFT, chat, generate, resume) would then produce NaN. inv_freq IS persistent
        # and reloads correctly, so recomputing from it is robust to that.
        self._cos = None
        self._sin = None
        self._cached_len = 0

    def _build_cache(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos()
        self._sin = emb.sin()
        self._cached_len = seq_len

    def _cache_valid(self, seq_len):
        return (
            self._cos is not None
            and seq_len <= self._cached_len
            and not self._cos.is_meta
            and self._cos.device == self.inv_freq.device
        )

    def forward(self, seq_len):
        # Lazily (re)build on first use, on growth, or after a device move / load.
        if not self._cache_valid(seq_len):
            self._build_cache(max(seq_len, self.max_seq_len), self.inv_freq.device)
        return self._cos[:seq_len], self._sin[:seq_len]


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
