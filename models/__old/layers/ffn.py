"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | 2026
=================================================================

Feed-Forward Networks.

SwiGLU FFN — used in LLaMA, Mistral, etc.
"""

import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network (used in LLaMA, Mistral, etc.)."""

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
