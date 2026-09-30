"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Attention layers.

Multi-Head Causal Self-Attention with:
  - GQA (Grouped Query Attention)
  - RoPE (via layers.rope)
  - KV-Cache
  - FlexAttention document masking (PyTorch 2.5+)
  - µP attention scaling
  - Flash Attention (via PyTorch SDPA)
"""

import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary_emb
from .kv_cache import repeat_kv

logger = logging.getLogger(__name__)

# FlexAttention: zero-overhead document masking (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False
    logger.info("FlexAttention not available (requires PyTorch >= 2.5). "
                "Falling back to dense mask for packing.")


# flex_attention must be compiled ON ITS OWN to get the fused kernel. Compiling the whole model is not
# enough: KDA's Triton kernels and gradient checkpointing break the graph, and outside a compiled region
# flex_attention silently falls back to a reference implementation that materialises the full T×T score
# matrix (at seq 8192 on the 990M: OOM on 24 GB, and far slower everywhere). Compiled lazily, CUDA only.
# Inside a compiled model the call must stay outside the model's graph as well: traced from the model
# (no activation checkpointing, `--compile`) it ran the reference implementation, found by the first
# B200 smoke on 30/09/2026 (bs 4 at seq 8192: 171 GiB and 13k tok/s). `recursive=False` keeps dynamo
# off this frame only, so the separately compiled kernel below still compiles.
_flex_compiled = None


@torch.compiler.disable(recursive=False)
def _flex(q, k, v, **kw):
    global _flex_compiled
    if not q.is_cuda:
        return flex_attention(q, k, v, **kw)
    if _flex_compiled is None:
        _flex_compiled = torch.compile(flex_attention, dynamic=False)
    return _flex_compiled(q, k, v, **kw)


# ─────────────────────────────────────────────────────────────
# DOCUMENT MASKING (for packed sequence training)
# ─────────────────────────────────────────────────────────────

def create_document_block_mask(document_ids, n_heads, device=None):
    """
    Create a FlexAttention block mask for packed document training.

    This is the same technique used by LLaMA, Qwen, Mistral, and OLMo:
    tokens can only attend to tokens within the same document, preventing
    cross-document attention leakage in packed sequences.

    Uses FlexAttention's block-sparse representation: O(T) memory instead
    of O(T²) for a dense mask. Zero VRAM overhead at any sequence length.

    Args:
        document_ids: (B, T) integer tensor — each value identifies which
                      document the token belongs to.
        n_heads:      number of Q attention heads (for mask dimensions)
        device:       target device

    Returns:
        BlockMask object for use with flex_attention()
    """
    B, T = document_ids.shape
    if device is not None and document_ids.device != device:
        document_ids = document_ids.to(device)

    # Mask function: attend if (1) causal and (2) same document
    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_doc = document_ids[b, q_idx] == document_ids[b, kv_idx]
        return causal & same_doc

    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=device)


@torch.compiler.disable
def build_cu_seqlens(document_ids):
    """
    Confini dei documenti in forma cumulativa, per i layer ricorrenti.

    Un layer di attention riceve una *maschera*: gli si dice quali coppie di token
    non devono vedersi. Un layer ricorrente non ha coppie — ha uno stato che scorre.
    L'unico modo di separare due documenti è dirgli DOVE azzerarlo, e questo è il
    formato che i kernel di `fla` (e di flash-attn) si aspettano: gli offset di
    inizio di ogni segmento nella sequenza appiattita.

        document_ids  [[0,0,0,1,1], [0,0,1,1,1]]   (B=2, T=5)
        →             [0, 3, 5, 7, 10]

    Senza questo, in un batch impacchettato lo stato ricorrente porta il contesto di
    un programma COBOL dentro il successivo. Non è un crash: è un modello peggiore.

    Args:
        document_ids: (B, T) interi — a quale documento appartiene ogni token
    Returns:
        (n_segmenti + 1,) int32 su device, monotona crescente, che parte da 0
    """
    B, T = document_ids.shape
    bounds = [0]
    offset = 0
    # I confini si calcolano sugli indici, non sui valori: due documenti diversi
    # possono avere lo stesso id in righe diverse del batch, e ogni riga è comunque
    # una sequenza a sé (lo stato non attraversa le righe).
    changes = (document_ids[:, 1:] != document_ids[:, :-1])
    for b in range(B):
        idx = torch.nonzero(changes[b], as_tuple=False).flatten() + 1
        for i in idx.tolist():
            bounds.append(offset + i)
        offset += T
        bounds.append(offset)
    cu = torch.tensor(bounds, dtype=torch.int32, device=document_ids.device)
    # Il numero di documenti cambia a ogni batch. Sotto torch.compile ogni lunghezza nuova ricompilava
    # i layer che ricevono `cu_seqlens` (30/09/2026: una ricompilazione a ogni batch, e in un run lungo
    # si passa il limite e i layer tornano non compilati): la dimensione e' dichiarata variabile.
    torch._dynamo.maybe_mark_dynamic(cu, 0)
    return cu


def make_packing_mask(document_ids, dtype=torch.bfloat16):
    """
    Build a dense block-diagonal attention mask for packed sequences.

    Fallback for environments without FlexAttention (PyTorch < 2.5).
    WARNING: O(T²) memory — impractical for seq_len > ~8K.

    Args:
        document_ids: (B, T) integer tensor — document membership per token
        dtype:        mask dtype (must match model dtype)

    Returns:
        (B, 1, T, T) additive attention mask (0.0 = attend, -inf = block)
    """
    same_doc = document_ids.unsqueeze(-1) == document_ids.unsqueeze(-2)
    T = document_ids.shape[1]
    causal = torch.tril(torch.ones(T, T, device=document_ids.device, dtype=torch.bool))
    allowed = same_doc & causal
    mask = torch.where(
        allowed,
        torch.tensor(0.0, dtype=dtype, device=document_ids.device),
        torch.tensor(float("-inf"), dtype=dtype, device=document_ids.device),
    )
    return mask.unsqueeze(1)


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with GQA, RoPE, KV-Cache,
    FlexAttention document masking, and µP attention scaling.

    Three attention paths (selected automatically):
      1. FlexAttention + block_mask  → packed training, zero overhead
      2. SDPA + dense attention_mask → fallback for packing without FlexAttention
      3. SDPA + is_causal=True       → generation, non-packed training

    FlexAttention handles GQA natively (enable_gqa=True) — no repeat_kv needed,
    saving memory by not expanding K/V tensors.

    µP attention scaling:
      SP (standard):  scale = 1/√d_head  (PyTorch default)
      µP:             scale = 1/d_head   (ensures stable attention entropy at any width)
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_heads % config.n_kv_heads == 0, (
            f"n_heads ({config.n_heads}) must be divisible by n_kv_heads ({config.n_kv_heads})"
        )

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.d_head = config.d_head  # explicit or d_model // n_heads
        assert self.d_head % 2 == 0, (
            f"d_head ({self.d_head}) must be even for RoPE rotate_half()"
        )

        self.use_gqa = self.n_rep > 1

        # Q/O projections can be rectangular when d_head != d_model/n_heads
        # (e.g. Qwen3-4B: d=2560, H=32, d_head=128 → Q: 2560→4096)
        self.d_attn = config.n_heads * self.d_head
        self.W_q = nn.Linear(config.d_model, self.d_attn, bias=config.bias)
        self.W_k = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=config.bias)
        self.W_v = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=config.bias)
        self.W_o = nn.Linear(self.d_attn, config.d_model, bias=config.bias)

        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # ── v2: NoPE (docs/PAPER_V2.md §3.7) ──
        # In un ibrido con KDA la posizione arriva dalla ricorrenza dei layer
        # ricorrenti, quindi i layer full-attention possono farne a meno: è ciò che
        # dà a K3 l'extrapolazione oltre la finestra addestrata senza YaRN.
        # ⚠️ Se KDA non fornisse abbastanza segnale posizionale, il danno NON si vede
        # dalla loss — solo dal gate G8 (retrieval a 8k/16k/32k), a run finito.
        # Con NoPE la cache RoPE non viene nemmeno allocata (33 MB/layer a 32K ctx).
        self.use_rope = not getattr(config, "nope_on_attention", False)
        self.rope = (RotaryEmbedding(self.d_head, config.max_seq_len, base=config.rope_theta)
                     if self.use_rope else None)

        # ── v2: output gate (docs/PAPER_V2.md §3.5) ──
        #   y = W_o[ σ(W_gate·x) ⊙ RMSNorm(attn_out) ]
        # Il gate si calcola dall'INPUT del layer, non dall'uscita dell'attention:
        # così la sua decisione non dipende da quanto l'attention ha già prodotto, ed
        # è il meccanismo che rimuove l'attention sink (le teste che scaricano
        # probabilità sul primo token per non attendere a nulla).
        # NOME: deliberatamente NON `*.W_o.weight` né `*.w2.weight`, i due pattern su
        # cui decoder.py fa l'init depth-scaled — un gate scalato in profondità non
        # avrebbe senso.
        # Due forme, e la differenza di costo e' 128x:
        #   "fullrank" → un valore per CANALE d'uscita: Linear(d, H·d_head)
        #   "perhead"  → un valore per TESTA:          Linear(d, H)
        # arXiv 2505.06708 valida il "head-specific sigmoid gate" ma l'abstract non
        # fissa quale delle due (testano 30 varianti). K3 usa la piena. Misuriamo.
        g = getattr(config, "attn_out_gate", False)
        self.gate_mode = ("fullrank" if g is True else (g or None)) if g else None
        self.out_gate = self.gate_mode is not None
        if self.out_gate:
            gate_dim = self.d_attn if self.gate_mode == "fullrank" else self.n_heads
            self.W_gate = nn.Linear(config.d_model, gate_dim, bias=config.bias)
            self.o_norm = RMSNorm(self.d_head)

        # µP: scale attention logits by 1/d_head instead of 1/√d_head
        # This keeps attention entropy stable as width grows.
        # When µP is off (mup_base_d_model=None), attn_scale=None → PyTorch default 1/√d_head
        if config.mup_base_d_model is not None:
            self.attn_scale = 1.0 / self.d_head
        else:
            self.attn_scale = None

    def forward(self, x, kv_cache=None, block_mask=None, attention_mask=None, use_cache=False):
        B, T, D = x.shape

        if kv_cache is not None and T > 1:
            raise ValueError(
                f"kv_cache incremental path expects T=1, got T={T}. "
                "Current implementation sets is_causal=False when kv_cache is present, "
                "so chunked decoding with T>1 would leak future tokens inside the chunk. "
                "Use T=1 incremental decode or implement a chunk causal mask."
            )

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # KV-Cache for fast autoregressive generation
        cache_active = bool(use_cache or kv_cache is not None)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            if self.use_rope:
                offset = k_prev.shape[2]
                cos, sin = self.rope(offset + T)
                cos, sin = cos[offset:offset + T], sin[offset:offset + T]
                q, k = apply_rotary_emb(q, k, cos, sin)
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        elif self.use_rope:
            cos, sin = self.rope(T)
            q, k = apply_rotary_emb(q, k, cos, sin)

        new_cache = (k, v) if cache_active else None

        # ── Attention: three paths ──

        if block_mask is not None and kv_cache is None:

            if self.training and self.attn_dropout.p > 0:
                # FlexAttention has no dropout_p argument; attention-probability
                # dropout is simply not applied on the packed path. Modern LLM
                # pretraining uses attn dropout = 0, so warn once (the warnings
                # module dedupes) and proceed instead of hard-crashing the run.
                # resid_dropout still applies after W_o.
                warnings.warn(
                    "FlexAttention packed path does not apply attention dropout "
                    f"(dropout={self.attn_dropout.p}); proceeding without it. "
                    "Set dropout=0 to silence, or --no-packing for SDPA dropout.",
                    RuntimeWarning, stacklevel=2,
                )

            # PATH 1: FlexAttention with document masking
            # GQA handled natively — no need to expand K/V
            attn_out = _flex(
                q, k, v,
                block_mask=block_mask,
                enable_gqa=self.use_gqa,
                scale=self.attn_scale,
            )

        elif attention_mask is not None:
            # PATH 2: SDPA with dense attention mask (fallback)
            k_expanded = repeat_kv(k, self.n_rep)
            v_expanded = repeat_kv(v, self.n_rep)
            attn_out = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
                scale=self.attn_scale,
            )

        else:
            # PATH 3: Standard causal SDPA (generation, non-packed training)
            k_expanded = repeat_kv(k, self.n_rep)
            v_expanded = repeat_kv(v, self.n_rep)
            attn_out = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True if kv_cache is None else False,
                scale=self.attn_scale,
            )

        if self.out_gate:
            # RMSNorm per-testa sull'uscita (attn_out è ancora (B, H, T, d_head)),
            # poi il gate calcolato dall'input. Ordine come K3: normalizza, poi filtra.
            attn_out = self.o_norm(attn_out)
            gate = torch.sigmoid(self.W_gate(x))
            if self.gate_mode == "perhead":
                # (B, T, H) → un fattore per testa, ripetuto sui suoi d_head canali
                gate = gate.unsqueeze(-1).expand(B, T, self.n_heads, self.d_head).reshape(B, T, self.d_attn)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_attn)
            attn_out = attn_out * gate
        else:
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_attn)
        return self.resid_dropout(self.W_o(attn_out)), new_cache
