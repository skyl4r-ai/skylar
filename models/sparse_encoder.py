"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

SkylarSparseEncoder — learned SPARSE lexical representations (SPLADE-style)
for hybrid retrieval (the sparse half of a dense+sparse BGE-M3-like stack).

Idea (SPLADE / BGE-M3 sparse): instead of a dense vector, produce a sparse
vector over the VOCABULARY — most entries zero, weight only on the few terms
that matter. Computed by projecting each token's hidden state onto the vocab
(reusing the tied token-embedding matrix — exactly the lm_head we already
pretrain), passing through log(1 + ReLU(·)) to get non-negative term weights,
then max-pooling over the sequence:

    h = backbone(tokens)                          # (B, T, D)
    logits = h @ token_emb.weight.T               # (B, T, V)   tied projection
    weights = log(1 + relu(logits))               # (B, T, V)   non-negative
    rep_j = max over tokens i of weights[:, i, j] # (B, V)      sparse vector

Retrieval score = dot(query_rep, doc_rep). Interpretable: the non-zero dims
ARE vocabulary tokens. Trained contrastively + FLOPS sparsity regularization.

NO re-pretrain: from_decoder() copies the decoder backbone; the vocab head is
the (tied) token embedding the decoder already learned. The backbone is the
decoder's own trunk (models/encoder_base.py): bidirectional on a dense model,
causal on Skylar 2 (KDA hybrid), where position i weighs the vocabulary from
the text up to i and the max over positions sees the whole text.
"""
import logging

import torch
import torch.nn.functional as F

from models.encoder_base import SkylarEncoderBase

logger = logging.getLogger(__name__)


class SkylarSparseEncoder(SkylarEncoderBase):
    """SPLADE-style sparse lexical encoder: (B, T) → (B, V) non-negative term weights."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Returns {"sparse": (B, V)}: max over the real tokens of log(1 + relu(h @ token_emb.T))."""
        x, mask = self.hidden_states(input_ids, attention_mask)
        logits = F.linear(x, self.token_emb.weight)               # (B, T, V) tied vocab projection
        weights = torch.log1p(F.relu(logits)) * mask[:, :, None].to(logits.dtype)
        return {"sparse": weights.max(dim=1).values}

    @classmethod
    def from_decoder(cls, decoder_path: str, encoder_attention: str = "auto") -> "SkylarSparseEncoder":
        """Init from a pretrained decoder (its trunk; the vocab projection is the tied token embedding)."""
        model = cls._from_decoder(decoder_path, encoder_attention=encoder_attention)
        if not getattr(model.config, "tie_weights", True):
            logger.warning("from_decoder: decoder is NOT weight-tied — the SPLADE vocab projection "
                           "reuses token_emb, not the trained lm_head; copy lm_head for a faithful init.")
        return model

    @torch.no_grad()
    def encode(self, input_ids, attention_mask=None, batch_size: int = 32):
        was_training = self.training
        self.eval()
        outs = []
        for s in range(0, input_ids.shape[0], batch_size):
            e = min(s + batch_size, input_ids.shape[0])
            m = attention_mask[s:e] if attention_mask is not None else None
            outs.append(self.forward(input_ids[s:e], attention_mask=m)["sparse"])
        if was_training:
            self.train()
        return torch.cat(outs, 0)

    @staticmethod
    def top_terms(sparse_vec, tokenizer, k: int = 12):
        """Human-readable top weighted vocab terms for one (V,) sparse vector."""
        vals, idx = sparse_vec.topk(min(k, sparse_vec.shape[-1]))
        out = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            if v <= 0:
                break
            tok = tokenizer.id_to_token(i)
            out.append((tok, round(v, 3)))
        return out



def flops_regularizer(sparse_batch):
    """SPLADE FLOPS loss: pushes whole-batch sparsity. sparse_batch (B, V) >= 0."""
    return (sparse_batch.mean(dim=0) ** 2).sum()
