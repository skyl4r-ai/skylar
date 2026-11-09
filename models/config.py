"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Model configuration — HuggingFace compatible: `Skylar2Config`, model_type "skylar2".

`NanoTransformerConfig` (model_type "nano-transformer") is the name of the older checkpoints
(Skylar-236M, Skylar-980M-Cobol): same fields, same code, kept so they load unchanged.

Supports:
  - Explicit d_head (Qwen3-style) for non-square attention projections
  - µP (Maximal Update Parameterization) for hyperparameter transfer

When d_head is set explicitly and d_head != d_model // n_heads,
Q/O projections become rectangular (d_model ↔ n_heads × d_head).
This is exactly how Qwen3-4B and Qwen3-32B work.

When mup_base_d_model is set, init/lr/attention are scaled so that
optimal HPs found on a small proxy transfer to any width.
"""

from transformers import PretrainedConfig


class Skylar2Config(PretrainedConfig):
    model_type = "skylar2"

    def __init__(
            self,
            vocab_size=40960,
            d_model=256,
            n_heads=8,
            n_kv_heads=None,
            d_head=None,
            n_layers=6,
            d_ff=512,
            max_seq_len=1024,
            dropout=0.1,
            bias=False,
            tie_weights=True,
            qk_norm=True,
            rope_theta=10000.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            # µP: set to proxy model width to enable HP transfer
            mup_base_d_model=None,
            # ── Skylar 2 — ogni default riproduce il comportamento v1 ──
            hidden_act="swiglu",
            situ_beta1=4.0,
            situ_beta2=25.0,
            attn_out_gate=False,   # False | "perhead" (consigliato) | "fullrank"
            attn_res=False,
            attn_res_mode="block",   # "block" (Kimi, Skylar 2) | "full" | "window" (valutata e scartata)
            attn_res_block_size=8,   # block: sotto-layer per blocco (8 → 9 blocchi sul 990M)
            attn_res_block=6,        # solo window: S → finestra di 2S+1 sorgenti
            gated_norm=0,            # rango del gate di GatedNorm sui pre-norm (0 = spento, 16 consigliato)
            layer_types=None,
            kda_ratio=None,
            kda_heads=None,
            kda_gate="lowrank",
            kda_conv_size=4,
            nope_on_attention=False,
            mtp_layers=0,
            mtp_loss_weight=0.3,
            head_fp32=False,
            rope_doc_positions=False,
            rope_fp32=False,
            **kwargs,
    ):
        # config.json stores the explicit head size as "_d_head" (the attribute behind the d_head property): read it
        # back as d_head, or a model with d_model not divisible by n_heads fails the check below at load time.
        if d_head is None and kwargs.get("_d_head") is not None:
            d_head = kwargs.pop("_d_head")
        kwargs.pop("_d_head", None)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        # GQA: n_kv_heads < n_heads → Grouped Query Attention
        #       n_kv_heads == n_heads → standard MHA (default)
        #       n_kv_heads == 1 → Multi-Query Attention (MQA)
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )

        # Explicit head dimension (Qwen3-style).
        # When None (default): d_head = d_model // n_heads (standard).
        # When set explicitly: Q/O projections can be rectangular.
        #   e.g. Qwen3-4B: d=2560, H=32, d_head=128 → Q: 2560→4096
        self._d_head = d_head
        if d_head is None:
            assert d_model % n_heads == 0, (
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}) "
                f"when d_head is not set explicitly"
            )

        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.bias = bias
        self.tie_weights = tie_weights
        self.qk_norm = qk_norm
        self.rope_theta = rope_theta
        self.mup_base_d_model = mup_base_d_model

        # ── v2 ─────────────────────────────────────────────────────────────
        # Invariante: con questi default il modello è BIT-IDENTICO alla v1,
        # così i checkpoint pubblicati si caricano senza toccare
        # il loro config.json — che non contiene nessuno di questi campi.
        self.hidden_act = hidden_act              # "swiglu" (v1) | "situ_glu"
        self.situ_beta1 = situ_beta1              # K3 usa 4.0/25.0, tarati sul LORO
        self.situ_beta2 = situ_beta2              # regime (hidden 7168) → da ri-tarare
        # Gate d'uscita prima di W_o. Due forme, 127x di differenza in parametri:
        #   "perhead"  Linear(d, H)          → +18.560 sul 990M
        #   "fullrank" Linear(d, H·d_head)   → +2.359.424
        # Misurate su 5 seed: INDISTINGUIBILI (t=0.61, 4 gdl, il segno cambia fra
        # seed). A parita' si prende quella che costa 127x meno.
        self.attn_out_gate = attn_out_gate
        self.attn_res = attn_res                  # attenzione softmax sulla profondità
        # Quali sorgenti vede ogni punto (models/layers/attn_res.py). "window" è stata
        # valutata e scartata: Kimi la misura quasi inutile (Tab. 4 di 2603.15031), a 36
        # layer non si addestra e costa molto più tempo per step.
        self.attn_res_mode = attn_res_mode
        self.attn_res_block_size = attn_res_block_size
        self.attn_res_block = attn_res_block
        # GatedNorm (arXiv 2601.22966): gate sigmoid low-rank dopo ln1/ln2/ln_f.
        self.gated_norm = gated_norm
        self.kda_gate = kda_gate                  # "lowrank" (tiene le teste) | "fullrank"
        self.kda_conv_size = kda_conv_size
        self.nope_on_attention = nope_on_attention
        self.mtp_layers = mtp_layers
        self.mtp_loss_weight = mtp_loss_weight
        # Testa d'uscita in float32 fuori dall'autocast: in bf16 i logit vengono arrotondati prima della cross-entropy
        # in float32. Sta nel config perché training ed eval facciano lo stesso conto.
        self.head_fp32 = head_fp32
        # RoPE. rope_doc_positions: nel training impacchettato ogni documento parte dalla
        # posizione 0 invece che dalla sua posizione nella finestra (AnchorAttention, arXiv 2411.13476 §3.2, Tab. 5:
        # «+ Reset»). rope_fp32: angoli in float64 e rotazione di q e k in float32, cast al dtype di q dopo. Spenti,
        # il modello è quello di prima.
        self.rope_doc_positions = rope_doc_positions
        self.rope_fp32 = rope_fp32

        # KDA: nessun GQA, quindi q/k/v/o sono tutte d × (H_kda·d_head). Copiare il
        # conteggio di teste di Kimi gonfia il modello; il default lo DIMENSIONA a
        # parità con l'attention che sostituisce → H_kda = (n_heads + n_kv_heads)/2.
        # Sui preset 1B_D, 2b e 4b esce intero (8, 10, 20).
        self.kda_ratio = kda_ratio
        if kda_heads is not None:
            self.kda_heads = kda_heads
        else:
            h = (n_heads + self.n_kv_heads) / 2
            self.kda_heads = int(h) if h == int(h) else None

        # `layer_types` è la fonte di verità di quale layer è cosa. Se è None ma
        # kda_ratio è dato, si deriva; se entrambi sono None il modello è tutto
        # attention, cioè la v1.
        self.layer_types = layer_types or self._resolve_layer_types(kda_ratio, n_layers)

        # transformers ties lm_head to the embedding whenever `tie_word_embeddings` is True, and that is its
        # default: without this line a preset with tie_weights=False (8b and up) was tied all the same. An older
        # saved config carries tie_word_embeddings=True and keeps loading as it was saved.
        kwargs.setdefault("tie_word_embeddings", tie_weights)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @staticmethod
    def _resolve_layer_types(kda_ratio, n_layers):
        """
        Espande "3:1" nella lista per-layer, es. per 36 layer:
            [kda, kda, kda, attention] × 9  →  27 ricorrenti, 9 full-attention.

        L'ultimo layer del gruppo è quello full-attention (come Kimi Linear): la
        sequenza parte ricorrente e "richiama" con l'attenzione esatta a chiudere
        ogni blocco. Con `kda_ratio=None` torna tutto attention, cioè la v1.
        """
        if not kda_ratio:
            return ["attention"] * n_layers
        try:
            n_kda, n_attn = (int(x) for x in str(kda_ratio).split(":"))
        except ValueError:
            raise ValueError(f"kda_ratio va scritto come 'N:M' (es. '3:1'), non {kda_ratio!r}")
        if n_kda < 0 or n_attn < 1:
            raise ValueError(f"kda_ratio {kda_ratio!r}: serve almeno un layer full-attention "
                             f"per gruppo, o il modello non ha richiamo esatto")
        period = n_kda + n_attn
        types = ["kda" if (i % period) < n_kda else "attention" for i in range(n_layers)]
        if types[-1] != "attention":
            # Un modello che chiude su un layer ricorrente perde il richiamo esatto
            # proprio dove serve (l'ultimo hidden state va al lm_head).
            types[-1] = "attention"
        return types

    @property
    def n_kda_layers(self):
        return sum(1 for t in self.layer_types if t == "kda")

    @property
    def d_head(self):
        """Per-head dimension. Explicit or inferred from d_model // n_heads."""
        if self._d_head is not None:
            return self._d_head
        return self.d_model // self.n_heads

    @property
    def mup_width_mult(self):
        """Width multiplier for µP scaling. Returns 1.0 when µP is disabled."""
        if self.mup_base_d_model is None:
            return 1.0
        return self.d_model / self.mup_base_d_model


class NanoTransformerConfig(Skylar2Config):
    """Name and model_type of the older checkpoints: their config.json says
    "nano-transformer". Same fields and code as Skylar2Config."""
    model_type = "nano-transformer"


# ── Preset configs ──────────────────────────────────────────
#
# Architettura: Qwen3-style (RMSNorm Pre-Norm, RoPE, SwiGLU, GQA, QK-Norm)
#
# Modelli da 4B in su: architettura IDENTICA ai Qwen3 ufficiali
# (verificata su config.json HuggingFace). Solo vocab_size diverso.
#
# Qwen3 usa d_head=128 fisso per tutti i modelli — quando d_model/n_heads ≠ 128,
# le projection Q/O diventano rettangolari (più capacità attentiva).
#
# rope_theta=1_000_000 per tutti i modelli Qwen3 ufficiali.
#
# Calcolo parametri: con vocab=40960 vs vocab=151936 (Qwen3).
# I nomi dei preset (4b, 8b, 14b, 32b) si riferiscono alla scala Qwen3
# (calcolata con il loro vocab), non al nostro conteggio parametri esatto.
#

PRESETS = {
    # test: ~6M params | ctx 4K: smoke tests, CPU ok.
    "test": dict(
        d_model=128, n_heads=4, n_kv_heads=4, n_layers=4, d_ff=256,
        max_seq_len=4096, rope_theta=100000.0,
    ),

    # small: ~40M params | ctx 8K.
    "small": dict(
        d_model=512, n_heads=8, n_kv_heads=4, n_layers=8, d_ff=1024,
        max_seq_len=8192, rope_theta=500000.0,
    ),

    # small_plus: ~70M params | ctx 16K.
    "small_plus": dict(
        d_model=640, n_heads=10, n_kv_heads=5, n_layers=10, d_ff=1792,
        max_seq_len=16384, rope_theta=1000000.0,
    ),

    # medium: ~107M params | ctx 16K.
    "medium": dict(
        d_model=768, n_heads=12, n_kv_heads=4, n_layers=12, d_ff=2048,
        max_seq_len=16384, rope_theta=1000000.0,
    ),

    # medium_plus: ~236M params | ctx 16K: the width of `large` with 18 layers (d_model/L 57), ff/dm 2.75, GQA 4:1, d_head 64.
    "medium_plus": dict(
        d_model=1024, n_heads=16, n_kv_heads=4, n_layers=18, d_ff=2816,
        max_seq_len=16384, rope_theta=1000000.0,
    ),

    # large: ~358M params | ctx 16K: deep and thin (28 layers at width 1024).
    "large": dict(
        d_model=1024, n_heads=16, n_kv_heads=4, n_layers=28, d_ff=2816,
        max_seq_len=16384, rope_theta=5000000.0,
    ),

    # gold: ~393M params | ctx 16K: d_head 128, ff/dm 2.70, d_model/L 64, GQA 5:1, tied weights.
    "gold": dict(
        d_model=1280, n_heads=10, n_kv_heads=2, d_head=128, n_layers=20, d_ff=3456,
        max_seq_len=16384, rope_theta=1000000.0,
    ),

    # 1B: ~1.0B params | ctx 32K.
    "1B": dict(
        d_model=1536, n_heads=16, n_kv_heads=4, n_layers=32, d_ff=5120,
        max_seq_len=32768, rope_theta=5000000.0,
    ),

    # 1B_D: ~980M params (vocab 48000) | ctx 16K: d_head 128, 12 heads, GQA 3:1, d_ff 4096, 36 layers, tied weights.
    "1B_D": dict(
        d_model=1536, n_heads=12, n_kv_heads=4, d_head=128, n_layers=36, d_ff=4096,
        max_seq_len=16384, rope_theta=1000000.0,
    ),

    # 2b: ~2.1B params (vocab 64000) | ctx 32K: between 1B_D and 4b; d_head 128, GQA 4:1, d_ff/d_model 3.5.
    "2b": dict(
        d_model=2048, n_heads=16, n_kv_heads=4, d_head=128, n_layers=36, d_ff=7168,
        max_seq_len=32768, rope_theta=1000000.0,
    ),

    # 4b: the architecture of Qwen3-4B (its config.json on HuggingFace), another vocab; d_head 128, tied weights.
    "4b": dict(
        d_model=2560, n_heads=32, n_kv_heads=8, d_head=128, n_layers=36, d_ff=9728,
        max_seq_len=32768, rope_theta=1000000.0,
    ),

    # 8b: the architecture of Qwen3-8B, another vocab; untied weights (as Qwen3 from 8B up).
    "8b": dict(
        d_model=4096, n_heads=32, n_kv_heads=8, d_head=128, n_layers=36, d_ff=12288,
        max_seq_len=32768, rope_theta=1000000.0, tie_weights=False,
    ),

    # 14b: the architecture of Qwen3-14B, another vocab; untied weights.
    "14b": dict(
        d_model=5120, n_heads=40, n_kv_heads=8, d_head=128, n_layers=40, d_ff=17408,
        max_seq_len=32768, rope_theta=1000000.0, tie_weights=False,
    ),

    # 32b: the architecture of Qwen3-32B, another vocab; d_head 128, untied weights.
    "32b": dict(
        d_model=5120, n_heads=64, n_kv_heads=8, d_head=128, n_layers=64, d_ff=25600,
        max_seq_len=131072, rope_theta=1000000.0, tie_weights=False,
    ),

    # 64b: ~61B params | ctx 128K, d_head 128 (LLaMA-2-70B scale).
    "64b": dict(
        d_model=8192, n_heads=64, n_kv_heads=8, d_head=128, n_layers=72, d_ff=28672,
        max_seq_len=131072, rope_theta=1000000.0, tie_weights=False,
    ),

    # 96b: ~99B params | ctx 128K, d_head 128.
    "96b": dict(
        d_model=10240, n_heads=80, n_kv_heads=8, d_head=128, n_layers=80, d_ff=32768,
        max_seq_len=131072, rope_theta=1000000.0, tie_weights=False,
    ),

    # 128b: ~128B params | ctx 128K, d_head 128.
    "128b": dict(
        d_model=11264, n_heads=88, n_kv_heads=8, d_head=128, n_layers=92, d_ff=32768,
        max_seq_len=131072, rope_theta=1000000.0, tie_weights=False,
    ),
}


def get_config(preset="small", vocab_size=40960, **overrides):
    """Get a preset config with optional overrides."""
    params = {**PRESETS[preset], "vocab_size": vocab_size, **overrides}
    return Skylar2Config(**params)
