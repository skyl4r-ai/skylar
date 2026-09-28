"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | 2026
=================================================================

Transformer block — shared building block for decoder and embedder.

The block itself is mask-agnostic: the caller decides causal vs
bidirectional by passing (or not passing) block_mask / attention_mask.

  Decoder  → passes causal mask → autoregressive
  Embedder → passes nothing (or padding mask) → bidirectional
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .norm import RMSNorm
from .attention import CausalSelfAttention
from .ffn import FeedForward
from .attn_res import AttnResMixer


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer block.

    Architecture (identical to Qwen3, LLaMA 3, Mistral):
        x → RMSNorm → Attention → residual
          → RMSNorm → SwiGLU FFN → residual

    v2 (docs/ARCH_V2.md): il layer di mixing temporale può essere **attention** o
    **KDA ricorrente**, scelto da `config.layer_types[layer_idx]`; e il residual
    stream può essere sostituito da **AttnRes**. Entrambi sono opt-in — con
    `layer_idx=None` (il default) il blocco è esattamente quello v1, che è ciò che
    tiene in piedi `embedder.py`, `sparse_encoder.py` e `classifier.py`, che lo
    costruiscono senza sapere niente di tutto questo.

    Args:
        config:    model config with d_model, n_heads, n_kv_heads, etc.
        layer_idx: posizione in profondità. Serve a KDA per la cache e per sapere
                   se questo layer è ricorrente o full-attention.
    """

    def __init__(self, config: object, layer_idx: int | None = None) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = RMSNorm(config.d_model)

        types = getattr(config, "layer_types", None)
        self.layer_type = "attention" if (layer_idx is None or not types) else types[layer_idx]
        if self.layer_type == "kda":
            from .kda import SkylarKDA          # import locale: fla serve solo se si usa KDA
            self.attn = SkylarKDA(config, layer_idx=layer_idx)
        else:
            self.attn = CausalSelfAttention(config)

        self.ln2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

        # AttnRes: due punti di applicazione per blocco (prima dell'attention e
        # prima della FFN). Il terzo, alla fine di tutto, sta nel decoder.
        self.attn_res = bool(getattr(config, "attn_res", False))
        if self.attn_res:
            w = getattr(config, "attn_res_block", None)
            window = None if not w else 2 * w + 1
            # Il primo punto in assoluto vede UNA sola sorgente (gli embedding): la
            # softmax su un elemento vale 1.0 qualunque sia la query, quindi quei 2·d
            # parametri non possono ricevere gradiente. Crearli comunque significa
            # parametri morti — e in DDP un parametro senza gradiente e' un errore,
            # non un dettaglio estetico.
            self.res_attn = None if layer_idx == 0 else AttnResMixer(config.d_model, window=window)
            self.res_ffn = AttnResMixer(config.d_model, window=window)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        block_mask: object | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        residuals: list | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            x:              (B, T, D) hidden states.
            kv_cache:       (K, V) per l'attention, o (stato, conv) per KDA.
            block_mask:     FlexAttention BlockMask — packed causal training.
            attention_mask: (B, 1, T, T) dense additive mask — fallback.
            use_cache:      return updated cache.
            cu_seqlens:     confini dei documenti impacchettati. Per l'attention è
                            ridondante (c'è la block_mask); per KDA è l'UNICO modo
                            di impedire allo stato di attraversare i documenti.
            residuals:      lista delle sorgenti AttnRes. Se None → residual stream v1.

        Returns:
            (hidden_states, cache | None)
        """
        kw = dict(kv_cache=kv_cache, block_mask=block_mask,
                  attention_mask=attention_mask, use_cache=use_cache)
        if self.layer_type == "kda":
            # Solo KDA sa cosa farsene: passarlo all'attention cambierebbe la sua
            # firma e romperebbe i tre modelli discriminativi che la condividono.
            kw["cu_seqlens"] = cu_seqlens

        if residuals is None:
            attn_out, new_cache = self.attn(self.ln1(x), **kw)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x, new_cache

        # AttnRes: il residual stream non è più una somma. Ogni punto sceglie da
        # quale profondità leggere, e l'uscita di ogni sotto-layer diventa una
        # sorgente nuova invece di essere sommata a un accumulatore.
        h = residuals[-1] if self.res_attn is None else self.res_attn(residuals)
        attn_out, new_cache = self.attn(self.ln1(h), **kw)
        residuals.append(attn_out)
        ffn_out = self.ffn(self.ln2(self.res_ffn(residuals)))
        residuals.append(ffn_out)
        return ffn_out, new_cache
