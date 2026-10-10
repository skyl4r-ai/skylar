"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================
The trunk shared by the encoders (dense embedder, sparse encoder, classifier).

The modules are built as in the decoder (same names, so a decoder checkpoint loads into them key for key) and the
forward pass is the decoder's own `_trunk`: every architecture flag the decoder has (KDA hybrid, AttnRes, GatedNorm,
output gate, SiTU) works in the encoders too, with no second implementation to keep in step.

How the encoder reads the sequence (`config.encoder_attention`):
  - "bidirectional": every token sees every other one (padding masked). Possible only on a model without recurrent
    layers: it is how the dense Skylar 1 models (e.g. Skylar-236M-Embed) were trained.
  - "causal": every token sees the ones before it, as in pretraining. Required on a model with KDA layers, which are
    recurrent and read left to right. The sequence is padded on the right, so the padding never reaches a real token.
  - "auto" (default): causal if the model has recurrent layers, bidirectional otherwise.

Pooling (`config.pool_strategy`): "mean", "last" or "cls"; by default "last" when causal (the state of the last token,
an appended <eos>, has read the whole text: E5-Mistral 2401.00368, jina-code-embeddings 2508.21290) and "mean" when
bidirectional.
"""

import logging

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from models.config import NanoTransformerConfig, Skylar2Config
from models.decoder import Skylar2ForCausalLM
from models.layers.attn_res import AttnResMixer
from models.layers.block import TransformerBlock
from models.layers.norm import make_norm

logger = logging.getLogger(__name__)

ENCODER_ATTENTION = ("auto", "bidirectional", "causal")
POOL_STRATEGIES = ("mean", "last", "cls")


class SkylarEncoderBase(PreTrainedModel):
    """The decoder's trunk without the language-model head, read causally or bidirectionally."""

    config_class = NanoTransformerConfig
    supports_gradient_checkpointing = True
    _head_prefixes = ()          # parameters of the subclass's own head: not in a decoder checkpoint

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)])
        self.attn_res = bool(getattr(config, "attn_res", False))
        self.attn_res_mode = getattr(config, "attn_res_mode", "block")
        self.attn_res_block_size = getattr(config, "attn_res_block_size", 8)
        if self.attn_res:
            window = None
            if self.attn_res_mode == "window":
                w = getattr(config, "attn_res_block", None)
                window = None if not w else 2 * w + 1
            self.res_final = AttnResMixer(config.d_model, window=window, rms_plus_eps=window is not None)
        self.ln_f = make_norm(config)
        self.mtp = None          # the decoder's trunk reads it: no MTP heads in an encoder
        mode = getattr(config, "encoder_attention", "auto") or "auto"
        if mode not in ENCODER_ATTENTION:
            raise ValueError(f"encoder_attention must be one of {ENCODER_ATTENTION}, got {mode!r}")
        if mode == "bidirectional" and self._has_recurrent_layers:
            raise ValueError("encoder_attention='bidirectional' on a model with KDA layers: they are recurrent and "
                             "read left to right. Use 'causal' (or 'auto').")
        pool = getattr(config, "pool_strategy", None)
        if pool is not None and pool not in POOL_STRATEGIES:
            raise ValueError(f"pool_strategy must be one of {POOL_STRATEGIES}, got {pool!r}")

    _trunk = Skylar2ForCausalLM._trunk     # the decoder's forward pass, shared (models/decoder.py)

    @property
    def _has_recurrent_layers(self):
        return any(getattr(b, "layer_type", "attention") == "kda" for b in self.blocks)

    @property
    def causal(self):
        mode = getattr(self.config, "encoder_attention", "auto") or "auto"
        return mode == "causal" or (mode == "auto" and self._has_recurrent_layers)

    @property
    def pool_strategy(self):
        return getattr(self.config, "pool_strategy", None) or ("last" if self.causal else "mean")

    @property
    def append_eos(self):
        """Causal with last-token pooling: the text ends with <eos>, whose state has read all of it."""
        return self.causal and self.pool_strategy == "last"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_emb.weight.numel()
        return n

    # ── the forward pieces ──────────────────────────────────────────────────────────────────────
    def hidden_states(self, input_ids, attention_mask=None):
        """(B, T) ids and (B, T) padding mask (1 = token, 0 = right padding) → (B, T, D) states after the final norm,
        and the mask (all ones when none was given)."""
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        if self.causal:
            x, _, _ = self._trunk(input_ids)
        else:
            # never None here: without a mask the attention would take its causal path
            additive = (1.0 - attention_mask[:, None, None, :].float()) * -1e9
            x, _, _ = self._trunk(input_ids, attention_mask=additive)
        return x, attention_mask

    def pool(self, hidden, mask):
        """(B, T, D) → (B, D) with the configured strategy, over the real tokens only."""
        mask = mask.bool()
        strategy = self.pool_strategy
        if strategy == "cls":
            return hidden[:, 0]
        if strategy == "last":
            idx = (mask.sum(dim=1) - 1).clamp(min=0)              # an all-padding row reads position 0
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def tokenize(self, tokenizer, texts, max_len, pad_id=None):
        """Texts → right-padded (input_ids, attention_mask) the way this encoder reads them: with <eos> appended when
        it pools the last token. `tokenizer` is a `tokenizers.Tokenizer`."""
        eos = tokenizer.token_to_id("<eos>") if self.append_eos else None
        if pad_id is None:
            pad_id = tokenizer.token_to_id("<pad>") or 0
        room = max_len - (1 if eos is not None else 0)
        ids = [tokenizer.encode(t, add_special_tokens=False).ids[:room] + ([eos] if eos is not None else [])
               for t in texts]
        width = max(1, max(len(x) for x in ids))
        input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(ids), width), dtype=torch.long)
        for i, x in enumerate(ids):
            if x:
                input_ids[i, :len(x)] = torch.tensor(x)
                attention_mask[i, :len(x)] = 1
        return input_ids, attention_mask

    # ── loading ─────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Opens the checkpoints of both config names (Skylar 2 'skylar2', Skylar 1 'nano-transformer')."""
        if "config" not in kwargs:
            try:
                saved, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path)
            except Exception:
                saved = {}
            if saved.get("model_type") == Skylar2Config.model_type:
                kwargs["config"] = Skylar2Config.from_pretrained(pretrained_model_name_or_path)
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    @classmethod
    def _from_decoder(cls, decoder_path, **config_updates):
        """The decoder's trunk (embedding, blocks, AttnRes read, final norm) copied into a new encoder; the
        subclass's head starts fresh. Any other difference between the two is an error, not a silent skip."""
        decoder = Skylar2ForCausalLM.from_pretrained(decoder_path)
        config = decoder.config
        for k, v in config_updates.items():
            setattr(config, k, v)
        model = cls(config)
        missing, unexpected = model.load_state_dict(decoder.state_dict(), strict=False)
        missing = [k for k in missing if not k.startswith(cls._head_prefixes)] if cls._head_prefixes else missing
        unexpected = [k for k in unexpected if not k.startswith(("lm_head.", "mtp."))]
        if missing or unexpected:
            raise RuntimeError(f"{cls.__name__}.from_decoder({decoder_path}): weights that do not match the decoder: "
                               f"missing {missing[:6]}, unexpected {unexpected[:6]}")
        logger.info("%s (%.1fM params) from %s: %s, pool=%s", cls.__name__, model.count_params() / 1e6, decoder_path,
                    "causal" if model.causal else "bidirectional", model.pool_strategy)
        return model
