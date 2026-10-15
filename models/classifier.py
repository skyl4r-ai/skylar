"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

SkylarClassifier — BERT-style discriminative (non-generative) head on the
bidirectional backbone. For tasks that EXTRACT / CLASSIFY rather than generate:
intent / topic / sentiment classification, routing, filtering.

Same recipe as the embedder: from_decoder() reuses the pretrained backbone (the
decoder's own trunk, models/encoder_base.py), pools, and a small head maps to
class logits. Bidirectional on a dense model; causal on Skylar 2 (KDA hybrid),
where it reads the last token, as GPT-style classifiers do. Trained with
cross-entropy.

Token-level extraction (NER / span QA) would use a per-token head on the same
backbone (not yet implemented) — same idea, per-token logits instead of pooled.
"""
import logging

import torch.nn as nn
import torch.nn.functional as F

from models.encoder_base import SkylarEncoderBase

logger = logging.getLogger(__name__)


class SkylarClassifier(SkylarEncoderBase):
    """Sequence classifier: (B, T) → (B, num_labels) logits."""

    _head_prefixes = ("classifier.",)

    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_labels = int(getattr(config, "num_labels", 2))
        self.classifier = nn.Linear(config.d_model, self.num_labels, bias=True)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        x, mask = self.hidden_states(input_ids, attention_mask)
        logits = self.classifier(self.pool(x, mask))
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}

    @classmethod
    def from_decoder(cls, decoder_path: str, num_labels: int, pool_strategy: str | None = None,
                     encoder_attention: str = "auto"):
        """A pretrained decoder's trunk + a fresh linear head over num_labels classes."""
        return cls._from_decoder(decoder_path, num_labels=num_labels, pool_strategy=pool_strategy,
                                 encoder_attention=encoder_attention)
