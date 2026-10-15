"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

SkylarEmbedder: dense text embeddings for retrieval and semantic search, on the decoder's backbone.

    token_emb → the decoder's trunk (models/encoder_base.py) → pool → L2 normalize

On a dense model (Skylar 1, e.g. Skylar-236M-Embed) the trunk reads bidirectionally and pools the mean, as E5, GTE and
BGE do. On Skylar 2 (KDA hybrid) it reads causally and pools the state of an appended <eos>, as E5-Mistral (2401.00368)
and jina-code-embeddings (2508.21290) do: the recurrent layers read left to right only.

Training: contrastive (InfoNCE) in training/bin.contrastive.py. This module only defines the model.

    from models.embedder import SkylarEmbedder
    model = SkylarEmbedder.from_decoder("checkpoints/skylar2-990m")        # the decoder's weights, no re-pretrain
    ids, mask = model.tokenize(tokenizer, ["testo uno", "testo due"], max_len=512)
    emb = model(ids, attention_mask=mask)["embeddings"]                     # (B, D) unit vectors
"""

import logging

import torch
import torch.nn.functional as F

from models.encoder_base import SkylarEncoderBase

logger = logging.getLogger(__name__)


class SkylarEmbedder(SkylarEncoderBase):
    """Dense embeddings: (B, T) → (B, D) L2-normalized vectors. HuggingFace compatible (save/from_pretrained)."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.post_init()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids:      (B, T) token ids, padded on the right.
            attention_mask: (B, T) 1 for real tokens, 0 for padding.
            labels:         unused, accepted for HuggingFace Trainer compatibility.

        Returns:
            "embeddings": (B, D) L2-normalized; "last_hidden": (B, T, D) the states after the final norm.
        """
        x, mask = self.hidden_states(input_ids, attention_mask)
        pooled = self.pool(x, mask)
        return {"embeddings": F.normalize(pooled, p=2, dim=-1), "last_hidden": x}

    @classmethod
    def from_decoder(cls, decoder_path: str, pool_strategy: str | None = None,
                     encoder_attention: str = "auto") -> "SkylarEmbedder":
        """A pretrained decoder's weights in an embedder, then fine-tuned contrastively (the E5-Mistral, GTE-Qwen,
        NV-Embed recipe). pool_strategy None = the default for the reading mode ("last" causal, "mean" bidirectional)."""
        return cls._from_decoder(decoder_path, pool_strategy=pool_strategy, encoder_attention=encoder_attention)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
               batch_size: int = 32) -> torch.Tensor:
        """(N, T) → (N, D) normalized embeddings, a batch at a time."""
        was_training = self.training
        self.eval()
        out = []
        for s in range(0, input_ids.shape[0], batch_size):
            e = min(s + batch_size, input_ids.shape[0])
            m = attention_mask[s:e] if attention_mask is not None else None
            out.append(self.forward(input_ids[s:e], attention_mask=m)["embeddings"])
        if was_training:
            self.train()
        return torch.cat(out, dim=0)
