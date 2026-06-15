"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | CTO of Sophia AI | 2026
=================================================================

Normalization layers.

RMSNorm — Root Mean Square Layer Normalization (used in LLaMA, Gemma, etc.).
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA, Gemma, etc.)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight.type_as(x)
