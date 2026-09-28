"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | 2026
=================================================================

Model configuration — HuggingFace compatible.

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


class NanoTransformerConfig(PretrainedConfig):
    model_type = "nano-transformer"

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
            **kwargs,
    ):
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
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

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
# Calcolo parametri: con vocab=40960 (nostro) vs vocab=151936 (Qwen3).
# I nomi dei preset (4b, 8b, 14b, 32b) si riferiscono alla scala Qwen3
# (calcolata con il loro vocab), non al nostro conteggio parametri esatto.
#

PRESETS = {
    # ─── TEST ───────────────────────────────────────────────
    # ~6M params | ctx 4K
    # GPU: qualsiasi, CPU ok | Tempo: minuti
    # Chinchilla: ~120M token | Dati .bin uint32: ~0.48 GB | Raw stimato: ~0.36–0.72 GB
    "test": dict(
        d_model=128, n_heads=4, n_kv_heads=4, n_layers=4, d_ff=256,
        max_seq_len=4096, rope_theta=100000.0,
    ),

    # ─── SMALL ──────────────────────────────────────────────
    # ~40M params | ctx 8K
    # RTX 4090: ~100K tok/s → 1.8B token in ~5 ore ✅ MISURATO
    # Genera italiano grammaticale. Troppo piccolo per SFT chat.
    # Chinchilla: ~800M token | Noi: 1.8B (~2.25× optimal) ✅
    # Dati .bin uint32: ~3.2 GB (target) | ~7.2 GB (1.8B)
    # Raw stimato: ~2.4–4.8 GB (target) | ~5.4–10.8 GB (1.8B)
    "small": dict(
        d_model=512, n_heads=8, n_kv_heads=4, n_layers=8, d_ff=1024,
        max_seq_len=8192, rope_theta=500000.0,
    ),
    "small_plus": dict(
        d_model=640, n_heads=10, n_kv_heads=5, n_layers=10, d_ff=1792,
        max_seq_len=16384, rope_theta=1000000.0,
    ),
    # ~70M params (con vocab 40960, tie_weights=True)
    # Chinchilla: ~1.4B token ✅

    # ─── MEDIUM ─────────────────────────────────────────────
    # ~107M params | ctx 16K
    # RTX 4090: ~51K tok/s → 1.8B token in ~10 ore ✅ MISURATO
    # RTX 5090: ~75K tok/s → 3.6B token in ~13 ore
    # RTX PRO 6000: ~80K tok/s → 3.6B token in ~12 ore
    # Primo modello con SFT chat funzionante. Risposte strutturate.
    # Chinchilla: 2.1B token | 6.3–12.6 GB raw
    "medium": dict(
        d_model=768, n_heads=12, n_kv_heads=4, n_layers=12, d_ff=2048,
        max_seq_len=16384, rope_theta=1000000.0,
    ),

    # ─── MEDIUM_PLUS — spot tra medium e large, prod-usable su RTX 4090 ───
    # ~236M params (vocab 32768, tie_weights=True) | ctx 16K | Nostro preset.
    # Stessa width di `large` (1024) ma 18 layer (non 28): aspect dm/L=57 (large
    # è deep-thin 37, sub-ottimale), ff/dm=2.75, GQA 4:1, head quadrate hd=64.
    # Regime data-constrained (Muennighoff 2023): sui nostri 1.12B token unici,
    # 4 epoche (4.48B, ~19:1 ≈ Chinchilla-matched) penalità trascurabile; 5 epoche
    # (5.6B, ~24:1) leggero over-train per spingere la qualità (inference-optimal).
    # RTX 4090: throughput da misurare (smoke), atteso ~70-90K tok/s con
    # expandable_segments + bf16. Pensato anche come encoder per embeddings.
    "medium_plus": dict(
        d_model=1024, n_heads=16, n_kv_heads=4, n_layers=18, d_ff=2816,
        max_seq_len=16384, rope_theta=1000000.0,
    ),

    # ─── LARGE ──────────────────────────────────────────────
    # ~358M params | ctx 32K | Nostro preset, non Qwen3.
    # RTX 4090: ~19K tok/s → 7B token in ~4 giorni
    # RTX 5090: ~28K tok/s → 7B token in ~3 giorni
    # RTX PRO 6000: ~30K tok/s → 7B token in ~3 giorni
    # 5× DGX H100: ~500K tok/s → 7B token in ~4 ore
    # Chat solida, JSON output affidabile, istruzioni complesse.
    # Chinchilla: ~7.0B token (349M × 20) | a 1.8B token = solo ~26% (under-train, NON optimal)
    # Nota: large è "deep-thin" (28 layer @ 1024 width, aspect dm/L=37). Per ~350M un
    #       1280×18 (aspect ~71) allenerebbe più veloce. Vedi preset `gold` qui sotto.
    "large": dict(
        d_model=1024, n_heads=16, n_kv_heads=4, n_layers=28, d_ff=2816,
        max_seq_len=16384, rope_theta=5000000.0,
    ),

    # ─── GOLD — best prod-usable per RTX 4090 ───────────────
    # ~393M params (vocab 32768) | ctx 16K | Nostro preset, ben proporzionato.
    # d_head=128 (Qwen-style), ff/dm=2.70, aspect dm/L=64, GQA 5:1, tie_weights=True.
    # RTX 4090: floor 6.3G, +grad-checkpointing ~10.3G → entra comodo (no OOM).
    # Velocità stimata ~13K tok/s su 4090.
    # Chinchilla: ~7.9B token (20×) | specialista @10×: ~4B token (~2.7 giorni).
    # SFT chat funzionante (>125M). Inference ~0.8 GB bf16 → deployable ovunque.
    "gold": dict(
        d_model=1280, n_heads=10, n_kv_heads=2, d_head=128, n_layers=20, d_ff=3456,
        max_seq_len=16384, rope_theta=1000000.0,
    ),

    # ─── 1B ───────────────────────────────────────────
    # ~1.0B params | ctx 32K | Nostro preset, non Qwen3.
    # RTX 4090: ❌ OOM (24GB non basta per training)
    # RTX 5090: ~12K tok/s → 20B token in ~19 giorni
    # RTX PRO 6000: ~12K tok/s → 20B token in ~19 giorni (comodo, 96GB)
    # 5× DGX H100: ~700K tok/s → 20B token in ~16 ore (con failure tax)
    # Primo modello con chat di qualità vera. Code gen, ragionamento.
    # Chinchilla: ~20B token | pretokenized .bin uint32 ≈ ~80 GB ⚠️
    "1B": dict(
        d_model=1536, n_heads=16, n_kv_heads=4, n_layers=32, d_ff=5120,
        max_seq_len=32768, rope_theta=5000000.0,
    ),

    # ─── 1B_D — preset "D", la code-SLM COBOL seria (from-scratch) ───
    # ~980M params (vocab 48000) / ~956M (vocab 32768) | ctx 16K.
    # Scelte (vs il "1B" sopra): d_head=128 ESPLICITO (Qwen3-style, come gold/4b/8b),
    # n_heads=12 (1536/128), GQA 3:1 (n_kv=4), d_ff=4096 (SwiGLU ff/dm=2.67 come gold),
    # 36 layer → aspect 42.7 = DEEP, headroom di reasoning (il gap che il gold non chiuse).
    # tie_weights=True (preset <8b). Chinchilla 20×: ~19.13B token | shard uint16 ≈ ~38 GB.
    # RTX 4090: ❌ OOM in training → RunPod (5090 / PRO 6000 / H100 ~$420/~7g).
    "1B_D": dict(
        d_model=1536, n_heads=12, n_kv_heads=4, d_head=128, n_layers=36, d_ff=4096,
        max_seq_len=16384, rope_theta=1000000.0,
    ),

    # ─── 4B — Qwen3-4B ─────────────────────────────────────
    # Architettura IDENTICA a Qwen3-4B (config.json verificato su HuggingFace).
    # 4.0B params con vocab Qwen3 (151936) | ~3.6B con nostro vocab (40960)
    # d_head=128 esplicito: Q/O projection rettangolari (2560↔4096).
    # tie_weights=True come Qwen3.
    # RTX PRO 6000: ~3.5K tok/s → 80B token in ~9 mesi
    # 5× DGX H100: ~240K tok/s → 80B token in ~8 giorni (con failure tax)
    # Chinchilla: ~72B token (su ~3.6B params) | pretokenized .bin uint32 ≈ ~288 GB 🔴
    "4b": dict(
        d_model=2560, n_heads=32, n_kv_heads=8, d_head=128, n_layers=36, d_ff=9728,
        max_seq_len=32768, rope_theta=1000000.0,
    ),

    # ─── 8B — Qwen3-8B ─────────────────────────────────────
    # Architettura IDENTICA a Qwen3-8B (config.json verificato su HuggingFace).
    # 8.2B params con vocab Qwen3 | ~7.3B con nostro vocab
    # d_head=128 = d_model/n_heads → projection quadrate.
    # tie_weights=False come Qwen3 (≥8B non lega i pesi).
    # 5× DGX H100 (40 GPU): ~170K tok/s → 160B in ~20 giorni (con failure tax)
    # 10× DGX H100 (80 GPU): ~300K tok/s → 160B in ~10 giorni
    # Chinchilla: ~146B token (su ~7.3B params) | pretokenized .bin uint32 ≈ ~584 GB 🔴
    "8b": dict(
        d_model=4096, n_heads=32, n_kv_heads=8, d_head=128, n_layers=36, d_ff=12288,
        max_seq_len=32768, rope_theta=1000000.0, tie_weights=False,
    ),

    # ─── 14B — Qwen3-14B ───────────────────────────────────
    # Architettura IDENTICA a Qwen3-14B (config.json verificato su HuggingFace).
    # 14.8B params con vocab Qwen3 | ~13.2B con nostro vocab
    # tie_weights=False come Qwen3.
    # 5× DGX H100: ~85K tok/s → 300B in ~2.5 mesi (con failure tax)
    # 10× DGX H100: ~150K tok/s → 300B in ~44 giorni
    # Chinchilla: ~264B token (su ~13.2B params) | pretokenized .bin uint32 ≈ ~1.056 TB 🔴
    # Costo cloud stimato: ~$150K
    "14b": dict(
        d_model=5120, n_heads=40, n_kv_heads=8, d_head=128, n_layers=40, d_ff=17408,
        max_seq_len=32768, rope_theta=1000000.0, tie_weights=False,
    ),

    # ─── 32B — Qwen3-32B ───────────────────────────────────
    # Architettura IDENTICA a Qwen3-32B (config.json verificato su HuggingFace).
    # 32.8B params con vocab Qwen3 | ~29.6B con nostro vocab
    # d_head=128 esplicito: Q/O projection rettangolari (5120↔8192).
    # tie_weights=False come Qwen3.
    # 10× DGX H100 (80 GPU): ~55K tok/s → 640B in ~9 mesi (con failure tax ×3)
    # 32× H100 cloud: ~28K tok/s → 640B in ~9 mesi
    # Chinchilla: ~592B token (su ~29.6B params) | pretokenized .bin uint32 ≈ ~2.368 TB 🔴
    # Costo cloud stimato: ~$1.7M
    "32b": dict(
        d_model=5120, n_heads=64, n_kv_heads=8, d_head=128, n_layers=64, d_ff=25600,
        max_seq_len=131072, rope_theta=1000000.0, tie_weights=False,
    ),

    # ─── 64B ────────────────────────────────────────────────
    # ~61B params (nostro vocab) | ctx 128K
    # Nessun Qwen3 corrispondente. Scala LLaMA-2-70B.
    # d_head=128 (d_model/n_heads = 128, projection quadrate).
    # 128× H100 cloud: ~32K tok/s → 1.3T in ~15 mesi (con failure tax ×3)
    # Chinchilla: ~1.22T token (su ~61B params) | pretokenized .bin uint32 ≈ ~4.88 TB 🔴
    # Costo cloud stimato: ~$9M
    "64b": dict(
        d_model=8192, n_heads=64, n_kv_heads=8, d_head=128, n_layers=72, d_ff=28672,
        max_seq_len=131072, rope_theta=1000000.0, tie_weights=False,
    ),

    # ─── 96B ────────────────────────────────────────────────
    # ~99B params (nostro vocab) | ctx 128K
    # Nessun Qwen3 corrispondente. Oltre LLaMA-2-70B.
    # d_head=128 (d_model/n_heads = 128, projection quadrate).
    # 256× H100 cloud: ~32K tok/s → 1.9T in ~20 mesi (con failure tax ×3)
    # Chinchilla: ~1.98T token (su ~99B params) | pretokenized .bin uint32 ≈ ~7.92 TB 🔴
    # Costo cloud stimato: ~$30M
    "96b": dict(
        d_model=10240, n_heads=80, n_kv_heads=8, d_head=128, n_layers=80, d_ff=32768,
        max_seq_len=131072, rope_theta=1000000.0, tie_weights=False,
    ),

    # ─── 128B ───────────────────────────────────────────────
    # ~128B params (nostro vocab) | ctx 128K
    # Nessun Qwen3 corrispondente. Scala GPT-4 class.
    # d_head=128 (d_model/n_heads = 128, projection quadrate).
    # 512× H100 cloud: ~40K tok/s → 2.5T in ~24 mesi (con failure tax ×3)
    # Chinchilla: ~2.56T token (su ~128B params) | pretokenized .bin uint32 ≈ ~10.24 TB 🔴
    # Costo cloud stimato: ~$55M
    "128b": dict(
        d_model=11264, n_heads=88, n_kv_heads=8, d_head=128, n_layers=92, d_ff=32768,
        max_seq_len=131072, rope_theta=1000000.0, tie_weights=False,
    ),
}


def get_config(preset="small", vocab_size=40960, **overrides):
    """Get a preset config with optional overrides."""
    params = {**PRESETS[preset], "vocab_size": vocab_size, **overrides}
    return NanoTransformerConfig(**params)
