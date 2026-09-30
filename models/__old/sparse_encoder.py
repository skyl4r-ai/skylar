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

    h = bidirectional_backbone(tokens)            # (B, T, D)
    logits = h @ token_emb.weight.T               # (B, T, V)   tied projection
    weights = log(1 + relu(logits))               # (B, T, V)   non-negative
    rep_j = max over tokens i of weights[:, i, j] # (B, V)      sparse vector

Retrieval score = dot(query_rep, doc_rep). Interpretable: the non-zero dims
ARE vocabulary tokens. Trained contrastively + FLOPS sparsity regularization.

NO re-pretrain: from_decoder() copies the decoder backbone; the vocab head is
the (tied) token embedding the decoder already learned.
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


class SkylarSparseEncoder(PreTrainedModel):
    """Bidirectional SPLADE-style sparse lexical encoder."""

    config_class = NanoTransformerConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: NanoTransformerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Returns dict:
          "sparse":  (B, V) non-negative sparse term-weight vector (max-pooled)
        """
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
        x = self.ln_f(x)                                          # (B, T, D)

        # Tied vocab projection (reuses pretrained token embedding = lm_head).
        logits = F.linear(x, self.token_emb.weight)              # (B, T, V)
        weights = torch.log1p(F.relu(logits))                    # (B, T, V) >= 0

        if attention_mask is not None:
            weights = weights * attention_mask[:, :, None].to(weights.dtype)
        sparse = weights.max(dim=1).values                       # (B, V) SPLADE-max
        return {"sparse": sparse}

    def count_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_emb.weight.numel()
        return n

    @classmethod
    def from_decoder(cls, decoder_path: str) -> "SkylarSparseEncoder":
        """Init from a pretrained decoder — copies token_emb, blocks, ln_f."""
        from models.decoder import NanoTransformer
        decoder = NanoTransformer.from_pretrained(decoder_path)
        model = cls(decoder.config)
        model.token_emb.load_state_dict(decoder.token_emb.state_dict())
        model.blocks.load_state_dict(decoder.blocks.state_dict())
        model.ln_f.load_state_dict(decoder.ln_f.state_dict())
        if not getattr(decoder.config, "tie_weights", True):   # F5
            logger.warning("from_decoder: decoder is NOT weight-tied — the SPLADE vocab projection "
                           "reuses token_emb, not the trained lm_head; copy lm_head for a faithful init.")
        logger.info("Initialized SkylarSparseEncoder (%.1fM) from decoder",
                    model.count_params() / 1e6)
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
