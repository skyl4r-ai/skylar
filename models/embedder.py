"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | CTO of Sophia AI | 2026
=================================================================

Embedding model — future expansion.

Currently the token embedding (nn.Embedding) lives directly in decoder.py.
This module is a placeholder for a dedicated embedding model (e.g. for
retrieval, semantic search, sentence embeddings) that can share the
decoder's backbone.

Bidirectional Transformer embedding model.

Same architecture as NanoTransformer decoder (same blocks, same layers),
but:
  - No causal mask → attention sees the full sequence (bidirectional)
  - No KV cache → not autoregressive
  - No LM head → pooling layer outputs a single vector per sequence
  - L2 normalization → unit vectors for cosine similarity

Identical to how E5, GTE, BGE, Nomic-Embed work internally.

Training:
  Contrastive loss (InfoNCE) in training/contrastive.py — not here.
  This module only defines the architecture.

Usage:
    from models.embedder import SkylarEmbedder
    from models.config import NanoTransformerConfig

    config = NanoTransformerConfig.from_preset("small")
    model = SkylarEmbedder(config)

    # (B, T) → (B, D) normalized embeddings
    out = model(input_ids, attention_mask=padding_mask)
    embeddings = out["embeddings"]  # unit vectors, ready for cosine sim
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from models.config import NanoTransformerConfig
from models.layers.block import TransformerBlock
from models.layers.norm import RMSNorm

logger = logging.getLogger(__name__)


class SkylarEmbedder(PreTrainedModel):
    """Bidirectional Transformer for dense text embeddings.

    Architecture:
        token_emb → dropout → N × TransformerBlock → RMSNorm → pool → normalize

    Pooling strategies:
        - "mean":  mean of non-padded token hidden states (default, best overall)
        - "cls":   first token hidden state (needs [CLS] token in data)
        - "last":  last non-padded token hidden state (works with decoder init)

    HuggingFace compatible:
        model.save_pretrained("path")
        SkylarEmbedder.from_pretrained("path")
    """

    config_class = NanoTransformerConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: NanoTransformerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        # ── Same layers as decoder ────────────────────────────
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = RMSNorm(config.d_model)

        # ── Pooling config ────────────────────────────────────
        self.pool_strategy = getattr(config, "pool_strategy", "mean")

        # ── Init ──────────────────────────────────────────────
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _pool(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Reduce (B, T, D) → (B, D) using the configured strategy.

        Args:
            hidden: (B, T, D) final hidden states after ln_f.
            mask:   (B, T) with 1 for real tokens, 0 for padding.
                    If None, all tokens are real.
        """
        if mask is None:
            mask = torch.ones(
                hidden.shape[0], hidden.shape[1],
                dtype=torch.bool, device=hidden.device,
            )
        else:
            mask = mask.bool()

        if self.pool_strategy == "cls":
            return hidden[:, 0]

        if self.pool_strategy == "last":
            # Index of last non-padded token per sequence
            seq_lengths = mask.sum(dim=1) - 1                    # (B,)
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[batch_idx, seq_lengths]

        # Default: mean pooling (masked)
        mask_expanded = mask.unsqueeze(-1).float()               # (B, T, 1)
        summed = (hidden * mask_expanded).sum(dim=1)             # (B, D)
        counts = mask_expanded.sum(dim=1).clamp(min=1)           # (B, 1)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids:      (B, T) token indices.
            attention_mask: (B, T) padding mask — 1 for real, 0 for pad.
            labels:         unused, accepted for HuggingFace Trainer compat.

        Returns:
            dict with:
              "embeddings": (B, D) L2-normalized embeddings
              "last_hidden": (B, T, D) full hidden states (for probing/debug)
        """
        x = self.drop(self.token_emb(input_ids))

        # ── Build padding mask for attention (not causal!) ────
        # (B, T) → (B, 1, 1, T) additive mask: 0 for real, -inf for pad
        attn_mask = None
        if attention_mask is not None:
            attn_mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e9

        # ── Transformer blocks (bidirectional) ────────────────
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(
                    block, x, None, None, attn_mask, False,
                    use_reentrant=False,
                )
            else:
                x, _ = block(x, attention_mask=attn_mask, use_cache=False)

        x = self.ln_f(x)

        # ── Pool + normalize ──────────────────────────────────
        pooled = self._pool(x, attention_mask)
        embeddings = F.normalize(pooled, p=2, dim=-1)

        return {
            "embeddings": embeddings,
            "last_hidden": x,
        }

    def count_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_emb.weight.numel()
        return n

    @classmethod
    def from_decoder(
        cls,
        decoder_path: str,
        pool_strategy: str = "mean",
    ) -> "SkylarEmbedder":
        """Initialize embedder from a pre-trained decoder checkpoint.

        Loads all shared weights (embedding, blocks, ln_f) from the
        decoder. Skips lm_head since the embedder doesn't have one.

        This is the standard approach:
          1. Pre-train decoder on next-token prediction (cheap, lots of data)
          2. Init embedder from decoder weights
          3. Fine-tune with contrastive loss (expensive, less data)

        Same technique used by E5-Mistral, GTE-Qwen, NV-Embed.

        Args:
            decoder_path: path to decoder checkpoint dir
            pool_strategy: "mean", "cls", or "last"
        """
        from models.decoder import NanoTransformer

        logger.info("Loading decoder from %s", decoder_path)
        decoder = NanoTransformer.from_pretrained(decoder_path)
        config = decoder.config
        config.pool_strategy = pool_strategy

        embedder = cls(config)

        # Copy shared weights
        embedder.token_emb.load_state_dict(decoder.token_emb.state_dict())
        embedder.blocks.load_state_dict(decoder.blocks.state_dict())
        embedder.ln_f.load_state_dict(decoder.ln_f.state_dict())

        n_params = embedder.count_params() / 1e6
        logger.info(
            "Initialized embedder (%.1fM params) from decoder, pool=%s",
            n_params, pool_strategy,
        )
        return embedder

    # ─────────────────────────────────────────────────────────────
    # Static generation
    # ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Batch encode for inference — handles large inputs.

        Args:
            input_ids:      (N, T) all sequences to encode.
            attention_mask: (N, T) padding mask.
            batch_size:     process this many at a time.

        Returns:
            (N, D) normalized embeddings.
        """
        was_training = self.training
        self.eval()

        all_embs = []
        N = input_ids.shape[0]

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_ids = input_ids[start:end]
            batch_mask = attention_mask[start:end] if attention_mask is not None else None

            out = self.forward(batch_ids, attention_mask=batch_mask)
            all_embs.append(out["embeddings"])

        if was_training:
            self.train()

        return torch.cat(all_embs, dim=0)