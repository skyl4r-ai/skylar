"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | CTO of Sophia AI | 2026
=================================================================

SkylarClassifier — BERT-style discriminative (non-generative) head on the
bidirectional backbone. For tasks that EXTRACT / CLASSIFY rather than generate:
intent / topic / sentiment classification, routing, filtering.

Same recipe as the embedder: from_decoder() reuses the pretrained backbone, runs
it bidirectionally (full attention, no causal mask), pools, and a small head maps
to class logits. Trained with cross-entropy.

Token-level extraction (NER / span QA) would use a per-token head on the same
backbone (not yet implemented) — same idea, per-token logits instead of pooled.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from models.config import NanoTransformerConfig
from models.layers.block import TransformerBlock
from models.layers.norm import RMSNorm

logger = logging.getLogger(__name__)


class SkylarClassifier(PreTrainedModel):
    """Bidirectional sequence classifier (BERT-style)."""

    config_class = NanoTransformerConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: NanoTransformerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.num_labels = int(getattr(config, "num_labels", 2))
        self.pool_strategy = getattr(config, "pool_strategy", "mean")
        self.gradient_checkpointing = False

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, self.num_labels, bias=True)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _pool(self, hidden, mask):
        if mask is None:
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        mask = mask.bool()
        if self.pool_strategy == "cls":
            return hidden[:, 0]
        if self.pool_strategy == "last":
            idx = (mask.sum(1) - 1).clamp(min=0)             # F7: all-pad -> 0, not -1 wrap
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]
        me = mask.unsqueeze(-1).float()
        return (hidden * me).sum(1) / me.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.drop(self.token_emb(input_ids))
        # F1: avoid the causal fallback path (attention.py PATH 3) when no mask is given.
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attn_mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e9
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(
                    block, x, None, None, attn_mask, False, use_reentrant=False)
            else:
                x, _ = block(x, attention_mask=attn_mask, use_cache=False)
        x = self.ln_f(x)
        pooled = self._pool(x, attention_mask)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"logits": logits, "loss": loss}

    @classmethod
    def from_decoder(cls, decoder_path: str, num_labels: int, pool_strategy: str = "mean"):
        from models.decoder import NanoTransformer
        decoder = NanoTransformer.from_pretrained(decoder_path)
        config = decoder.config
        config.num_labels = num_labels
        config.pool_strategy = pool_strategy
        model = cls(config)
        model.token_emb.load_state_dict(decoder.token_emb.state_dict())
        model.blocks.load_state_dict(decoder.blocks.state_dict())
        model.ln_f.load_state_dict(decoder.ln_f.state_dict())
        logger.info("Initialized SkylarClassifier (%d labels) from decoder", num_labels)
        return model
