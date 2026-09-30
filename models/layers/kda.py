"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

KDA — Kimi Delta Attention: il layer ricorrente dell'ibrido (docs/PAPER_V2.md §3.2, §4.4).

Al posto di confrontare ogni token con tutti i precedenti (costo che cresce col
quadrato del contesto e con una KV-cache che cresce lineare), KDA porta avanti uno
**stato a matrice** di dimensione fissa, aggiornato token per token con la delta rule:

    S_t = (I − β_t k_t k_tᵀ) · Diag(α_t) · S_{t−1} + β_t k_t v_tᵀ

`Diag(α_t)` è un gate di dimenticanza **per canale** — ogni dimensione dello stato
decide da sola quanto ricordare — e il termine `(I − β k kᵀ)` cancella dallo stato
ciò che la chiave corrente sta per sovrascrivere. Il risultato per noi: sul 990M ibrido
una sequenza tiene 18 KiB per token invece di 72, più 14 MiB fissi (3,9× meno a 32k
token), e con più prompt lunghi la generazione fa 1,9× i token al secondo
(docs/PAPER_V2.md §6.9). Su COBOL, dove un programma più i suoi copybook sono lunghi,
è la differenza fra contesto lungo nominale e contesto lungo usabile.

## Cosa è nostro e cosa no

Nostre: le proiezioni, i nomi, il dimensionamento, la cache, l'init. Del kernel
`flash-linear-attention` (MIT) è solo la **ricorrenza chunkwise in Triton** — lo
stesso rapporto che abbiamo con flash-attn o con cuBLAS. Su CPU e MPS, dove Triton non
gira, la stessa ricorrenza è fatta token per token in PyTorch (`kda_torch`), senza fla. Non ci sono pesi di altri:
la regola "100% from-scratch" resta intatta.

## Il dimensionamento — perché H_kda non è il numero di teste di Kimi

KDA non ha GQA: `q`, `k`, `v`, `o` sono tutte `d × key_dim`. Copiare le teste di
Kimi (`H_kda = d/128`) gonfierebbe il modello del 10-16%. Imponendo invece la parità
con l'attention che il layer sostituisce esce `H_kda = (n_heads + n_kv_heads)/2`, e
l'ibrido costa **+1-2%**. Il conto sta in `utils/bin.arch_budget.py`.

## Due cose da sapere prima di usarlo

1. **`cu_seqlens` non è un extra.** Non esiste una maschera di attention da mascherare:
   esiste uno *stato*. Senza i confini dei documenti lo stato scorre da un programma
   COBOL al successivo. Non crasha — impara peggio, e la loss non lo mostra.
2. **La cache non è una KV-cache.** È `(stato_ricorrente, stati_della_conv)`, di forma
   completamente diversa dalla tupla `(k, v)` rank-4 dei layer full-attention. Un
   modello ibrido ha quindi una cache **eterogenea**: vedi `models/layers/kv_cache.py`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm

_FLA_HINT = (
    "KDA richiede flash-linear-attention (MIT).\n"
    "    pip install flash-linear-attention einops\n"
    "Serve solo per i layer ricorrenti: senza, il modello gira normalmente in "
    "configurazione tutta-attention (kda_ratio=None)."
)


def _load_kernels():
    try:
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
        return chunk_kda, fused_recurrent_kda
    except ImportError as e:
        raise ImportError(f"{_FLA_HINT}\n(causa: {e})") from e


def kda_torch(q, k, v, g, beta, A_log, dt_bias, lower_bound, initial_state=None,
              output_final_state=False, cu_seqlens=None):
    """KDA in plain PyTorch, token by token: the path for CPU and MPS, where the Triton kernels of
    flash-linear-attention do not run. Same computation as `fla.ops.kda` with the flags SkylarKDA uses
    (use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, safe_gate with lower_bound,
    state_v_first), checked against the kernels by gate G12 (relative error ~1e-3, the kernels' own
    internal precision):
        q, k  <- q / ||q||, k / ||k||   (eps 1e-6), q scaled by K^-1/2
        a     =  lower_bound * sigmoid(exp(A_log) * (g + dt_bias))     per-channel log-decay in (lower_bound, 0)
        S     <- S * exp(a)                                             decay along K; S is [V, K] ("v first")
        S     <- S + sigmoid(beta) * (v - S k) k^T                      delta rule
        o     =  S q
    q, k, g: (B, T, H, K); v: (B, T, H, V); beta: (B, T, H); initial_state: (B, H, V, K) or None;
    cu_seqlens (varlen, B = 1): the state restarts at every segment. Returns (o in v's dtype, final
    state float32 — one per segment with cu_seqlens — or None)."""
    if cu_seqlens is not None:
        outs, states = [], []
        bounds = cu_seqlens.tolist()
        for s, e in zip(bounds[:-1], bounds[1:]):
            o, st = kda_torch(q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e], A_log, dt_bias,
                              lower_bound, None, output_final_state)
            outs.append(o)
            states.append(st)
        return torch.cat(outs, 1), (torch.cat(states, 0) if output_final_state else None)
    B, T, H, K = q.shape
    dtype = v.dtype
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * K ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    a = lower_bound * torch.sigmoid(A_log.float().exp().view(1, 1, H, 1)
                                    * (g + dt_bias.float().view(1, 1, H, K)))
    beta = torch.sigmoid(beta)
    S = (initial_state.float().clone() if initial_state is not None
         else q.new_zeros(B, H, v.shape[-1], K))
    o = torch.empty(B, T, H, v.shape[-1], dtype=torch.float32, device=q.device)
    for t in range(T):
        S = S * a[:, t].exp().unsqueeze(-2)
        kt = k[:, t]
        u = (v[:, t] - torch.einsum("bhvk,bhk->bhv", S, kt)) * beta[:, t].unsqueeze(-1)
        S = S + u.unsqueeze(-1) * kt.unsqueeze(-2)
        o[:, t] = torch.einsum("bhvk,bhk->bhv", S, q[:, t])
    return o.to(dtype), (S if output_final_state else None)


class ShortConv(nn.Module):
    """
    Convoluzione causale depthwise (kernel 4) su q, k, v prima della ricorrenza.

    Dà al layer una finestra locale esatta che lo stato ricorrente, essendo una
    compressione, non può garantire. In decode mantiene uno stato di `kernel-1`
    token — piccolo, ma va salvato nella cache o la generazione diverge dal training.
    """

    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.dim, self.kernel_size = dim, kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)

    def forward(self, x, state=None, pos_in_seg=None):
        """
        Args:
            x:          (B, T, D)
            state:      (B, D, K-1) coda del passo precedente, in decode
            pos_in_seg: (T,) distanza di ogni token dall'inizio del SUO documento.
                        Se assente, la conv tratta la riga come un unico documento.

        La conv va isolata ai confini esattamente come lo stato ricorrente. Il punto
        non è tanto che tre token vedano il documento precedente: è che quella
        contaminazione entra nella ricorrenza e da lì si propaga a TUTTO il documento
        successivo. Verificato: senza questa maschera l'isolamento fallisce su tutte
        le posizioni, non solo sulle prime tre.
        """
        xt = x.transpose(1, 2)                                  # (B, D, T)
        K = self.kernel_size
        if state is None:
            xt = F.pad(xt, (K - 1, 0))
        else:
            xt = torch.cat([state, xt], dim=-1)
        new_state = xt[..., -(K - 1):] if K > 1 else None

        if pos_in_seg is None:
            out = self.conv(xt)
        else:
            # Convoluzione causale scritta come K somme sfalsate, così ogni tap si
            # può spegnere quando pescherebbe da PRIMA dell'inizio del documento.
            # Costa quanto la conv (K multiply-add su un tensore (B,D,T)) e vale solo
            # nel percorso varlen.
            T = x.shape[1]
            w = self.conv.weight                                 # (D, 1, K)
            out = None
            for j in range(K):
                back = K - 1 - j                                 # quanti token indietro
                tap = xt[..., j:j + T] * (pos_in_seg >= back)
                term = w[:, 0, j].unsqueeze(0).unsqueeze(-1) * tap
                out = term if out is None else out + term
        return F.silu(out.transpose(1, 2)), new_state


def positions_in_segment(cu_seqlens, total):
    """
    Distanza di ogni token dall'inizio del proprio documento.

    `cu_seqlens` dice DOVE cominciano i documenti; questo dice, per ogni token,
    quanto è lontano dall'inizio del suo. È ciò che serve alla short conv per sapere
    quali tap può usare, e si calcola una volta sola per forward invece che per layer.
    """
    idx = torch.arange(total, device=cu_seqlens.device)
    starts = cu_seqlens[:-1]
    # per ogni token, l'ultimo inizio di documento che lo precede
    seg = torch.searchsorted(starts, idx, right=True) - 1
    return idx - starts[seg]


class SkylarKDA(nn.Module):
    """
    Un layer ricorrente KDA, sostituto drop-in di `CausalSelfAttention` nel blocco.

    Espone la stessa firma di forward — `(x, kv_cache, block_mask, attention_mask,
    use_cache)` più `cu_seqlens` — e restituisce `(output, new_cache)`, così il
    blocco non deve sapere quale dei due layer sta ospitando.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        d = config.d_model
        self.d_model = d
        self.head_dim = config.d_head
        self.n_heads = config.kda_heads
        if self.n_heads is None:
            raise ValueError(
                f"kda_heads non derivabile: (n_heads + n_kv_heads)/2 = "
                f"({config.n_heads} + {config.n_kv_heads})/2 non è intero. "
                f"Passa kda_heads esplicitamente."
            )
        self.key_dim = self.n_heads * self.head_dim

        # Proiezioni. NOTA SUI NOMI: la proiezione d'uscita si chiama `W_o` come
        # nell'attention, e non `o_proj`. Non è estetica — `decoder.py:110-112`
        # applica l'init depth-scaled 0.02/√(2L) cercando i nomi `W_o.weight` e
        # `w2.weight`. Un modulo chiamato `o_proj` salterebbe l'init IN SILENZIO,
        # senza errori, dando solo un modello peggiore.
        self.W_q = nn.Linear(d, self.key_dim, bias=False)
        self.W_k = nn.Linear(d, self.key_dim, bias=False)
        self.W_v = nn.Linear(d, self.key_dim, bias=False)
        self.W_o = nn.Linear(self.key_dim, d, bias=False)

        self.conv_size = getattr(config, "kda_conv_size", 4)
        self.use_conv = self.conv_size > 1
        if self.use_conv:
            self.q_conv = ShortConv(self.key_dim, self.conv_size)
            self.k_conv = ShortConv(self.key_dim, self.conv_size)
            self.v_conv = ShortConv(self.key_dim, self.conv_size)

        # Gate di dimenticanza per canale, low-rank (d → head_dim → key_dim).
        # È la variante Kimi Linear: in full-rank costerebbe 5× e obbligherebbe a
        # tagliare le teste per restare a parità di parametri, dimezzando lo stato.
        self.f_a = nn.Linear(d, self.head_dim, bias=False)
        self.f_b = nn.Linear(self.head_dim, self.key_dim, bias=False)

        # Gate d'uscita, anch'esso low-rank. È il motivo per cui l'output gate
        # esplicito serve solo ai 9 layer full-attention: qui c'è già.
        gate_full = getattr(config, "kda_gate", "lowrank") == "fullrank"
        self.gate_fullrank = gate_full
        if gate_full:
            self.g_full = nn.Linear(d, self.key_dim, bias=True)
        else:
            self.g_a = nn.Linear(d, self.head_dim, bias=False)
            self.g_b = nn.Linear(self.head_dim, self.key_dim, bias=True)

        self.b_proj = nn.Linear(d, self.n_heads, bias=False)

        # Decadimento log-spazio con lower bound: g = lower_bound · σ(...) mantiene
        # il log-decay in un intervallo che i tile bf16 delle TensorCore reggono —
        # è ciò che rende il kernel interamente TensorCore invece che misto.
        self.A_log = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        dt = torch.exp(torch.rand(self.key_dim, dtype=torch.float32)
                       * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(0.001)))
                       + torch.log(torch.tensor(0.001))).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        # Marcati come in `fla`: vanno nel gruppo SENZA weight decay, insieme a norm,
        # bias ed embedding. Sono parametri di dinamica, non di capacità: decaderli
        # spinge il gate verso un decadimento fisso.
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True

        self.o_norm = RMSNorm(self.head_dim)
        self.lower_bound = -5.0
        self.dropout = nn.Dropout(config.dropout)

    # ── stato ricorrente: quanto costa davvero ──
    def state_bytes(self, batch_size=1):
        """Byte dello stato ricorrente per sequenza — il numero da confrontare con la KV-cache."""
        return batch_size * self.n_heads * self.head_dim * self.head_dim * 4

    # Triton kernels on CUDA; `kda_torch` elsewhere, or here when switched off (the parity gate G12).
    use_kernels = True

    def forward(self, x, kv_cache=None, block_mask=None, attention_mask=None,
                use_cache=False, cu_seqlens=None):
        B, T, _ = x.shape

        rec_state, conv_states = (kv_cache if kv_cache is not None else (None, None))
        cq, ck, cv = conv_states if conv_states is not None else (None, None, None)

        # In modalità varlen si lavora sulla sequenza APPIATTITA fin da subito: la
        # conv e la ricorrenza devono vedere gli stessi confini, e `cu_seqlens` è
        # espressa in quello spazio.
        xk = x.reshape(1, B * T, -1) if cu_seqlens is not None else x
        pos = positions_in_segment(cu_seqlens, B * T) if cu_seqlens is not None else None

        q, k, v = self.W_q(xk), self.W_k(xk), self.W_v(xk)
        if self.use_conv:
            q, cq = self.q_conv(q, cq, pos)
            k, ck = self.k_conv(k, ck, pos)
            v, cv = self.v_conv(v, cv, pos)
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)

        g = self.f_b(self.f_a(xk))
        beta = self.b_proj(xk)

        H, Dh = self.n_heads, self.head_dim

        # Formato varlen: i kernel di `fla` vogliono il batch APPIATTITO in una sola
        # sequenza — (1, B·T, ...) — con `cu_seqlens` che indicizza dentro di essa.
        # È lo stesso formato di flash-attn. `build_cu_seqlens` produce già offset
        # nello spazio appiattito (avanza di T per ogni riga del batch), quindi qui
        # basta la reshape: i confini fra righe sono già confini di segmento.
        if cu_seqlens is not None:
            shape = (1, B * T, H, Dh)
            beta = beta.reshape(1, B * T, H)
        else:
            shape = (B, T, H, Dh)
        q = q.reshape(*shape)
        k = k.reshape(*shape)
        v = v.reshape(*shape)
        g = g.reshape(*shape)

        # `g` esce da f_proj come attivazione GREZZA: è il kernel a trasformarla in
        # decadimento log-spazio, con `use_gate_in_kernel=True`, calcolando
        # `-exp(A_log) · softplus(g + dt_bias)`, il clamp a `[lower_bound, 0)` e
        # **la somma cumulativa dentro il chunk**, tutto fuso in un passaggio.
        #
        # ⚠️ Non provare a precalcolarlo qui per "ridurre la dipendenza". Con
        # `use_gate_in_kernel=False` il kernel non si aspetta il log-decay per
        # posizione ma il suo **cumsum di chunk già scalato per 1/ln2**
        # (`kda_gate_chunk_cumsum(..., scale=RCP_LN2)`): passargli il valore
        # per-posizione compila, gira, e dà un risultato sbagliato del 35% —
        # misurato. È una convenzione interna al kernel, non un'interfaccia.
        #
        # `A_log` e `dt_bias` viaggiano per `**kwargs`: non compaiono fra i parametri
        # nominati di `chunk_kda`, ma entrambe le versioni di `fla` li leggono da lì
        # (`chunk.py`: `assert "A_log" in kwargs`). Un controllo di compatibilità
        # basato su `inspect.signature` darebbe quindi un falso allarme — va provato
        # chiamando, ed è ciò che fa il gate.
        #
        # Il kernel chunkwise è per il training e il prefill; in decode (T=1) la
        # forma ricorrente fusa è più veloce, perché non c'è nessun chunk da riempire.
        if x.is_cuda and self.use_kernels:
            chunk_kda, fused_recurrent_kda = _load_kernels()
            kernel = fused_recurrent_kda if T == 1 else chunk_kda
            o, rec_state = kernel(
                q=q, k=k, v=v, g=g, beta=beta,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=rec_state,
                output_final_state=bool(use_cache or kv_cache is not None),
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=self.lower_bound,
                state_v_first=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, rec_state = kda_torch(q, k, v, g, beta, self.A_log, self.dt_bias, self.lower_bound,
                                     initial_state=rec_state,
                                     output_final_state=bool(use_cache or kv_cache is not None),
                                     cu_seqlens=cu_seqlens)

        gate = self.g_full(x) if self.gate_fullrank else self.g_b(self.g_a(x))  # gate su x (B,T,D)
        o = self.o_norm(o.reshape(B, T, H, Dh)) * torch.sigmoid(gate.view(B, T, H, Dh))
        o = o.reshape(B, T, self.key_dim)

        new_cache = None
        if use_cache or kv_cache is not None:
            new_cache = (rec_state, (cq, ck, cv) if self.use_conv else None)
        return self.dropout(self.W_o(o)), new_cache

    def extra_repr(self):
        return (f"d_model={self.d_model}, n_heads={self.n_heads}, head_dim={self.head_dim}, "
                f"stato={self.state_bytes()/1024:.0f} KB/seq")
