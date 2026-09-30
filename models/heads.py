"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | 2026
=================================================================

Output heads for Skylar2ForCausalLM.

The default LM head (nn.Linear + µP output scaling + weight tying)
lives directly in decoder.py to preserve HuggingFace checkpoint
compatibility (weight path: "lm_head.weight" tied with "token_emb.weight").

This module provides additional heads for future use:
  - ClassificationHead — sequence classification
  - RewardHead — scalar reward for RLHF
"""

import torch
import torch.nn as nn

from models.layers.norm import RMSNorm


class ClassificationHead(nn.Module):
    """
    Classification head on top of the last hidden state.

    Applies pooling → norm → dropout → linear projection.
    """

    def __init__(self, d_model, num_classes, dropout=0.1, pool_mode="last"):
        super().__init__()
        self.pool_mode = pool_mode
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (B, T, D) last hidden states from the decoder
            attention_mask: (B, T) binary mask (1 = real token, 0 = pad)

        Returns:
            (B, num_classes) classification logits
        """
        if self.pool_mode == "last":
            if attention_mask is not None:
                seq_lengths = (attention_mask.sum(dim=1).long() - 1).clamp(min=0)   # F7
                pooled = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), seq_lengths]  # F2: device=
            else:
                pooled = hidden_states[:, -1, :]
        elif self.pool_mode == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden_states.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

        return self.proj(self.dropout(self.norm(pooled)))


class RewardHead(nn.Module):
    """
    Scalar reward head for RLHF reward modeling.

    Projects last hidden state to a single scalar value.
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, 1, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (B, T, D) last hidden states from the decoder
            attention_mask: (B, T) binary mask

        Returns:
            (B,) scalar rewards
        """
        if attention_mask is not None:
            seq_lengths = (attention_mask.sum(dim=1).long() - 1).clamp(min=0)   # F7
            pooled = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), seq_lengths]  # F2: device=
        else:
            pooled = hidden_states[:, -1, :]

        return self.proj(self.dropout(self.norm(pooled))).squeeze(-1)
