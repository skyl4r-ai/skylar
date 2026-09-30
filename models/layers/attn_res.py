"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | 2026
=================================================================

AttnRes — attenzione softmax sulla PROFONDITÀ (Kimi, arXiv 2603.15031).

In un transformer normale ogni sotto-layer legge un solo tensore, il residual
stream, che è la somma di tutto quello che è venuto prima: `x = x + attn(x)`. Le
sorgenti sono sommate con peso 1, sempre, per ogni token.

AttnRes sostituisce quella somma con una scelta: le uscite precedenti restano
disponibili come sorgenti separate, e ogni punto di lettura le pesa con una softmax
contro una pseudo-query imparata, decidendo **per token** da quale profondità leggere.

    k_l = RMSNorm(v_l)
    p   = softmax_l( k_l · q )         q = pseudo-query del punto, imparata (init 0)
    o   = Σ_l p_l · v_l

Costa 2·d parametri per punto (la query e il peso della norm): sul 990M sono 72 punti
e 221.184 parametri, lo 0.02% del modello.

## Quali sorgenti: tre forme

  block   l'embedding + la SOMMA di ogni blocco chiuso + la somma parziale del
          blocco aperto (Kimi §3.2; `fla/models/lightnet`). È quella di Skylar 2:
          blocchi da 8 sotto-layer, al massimo 11 sorgenti su 36 layer.
  full    tutte le uscite dei sotto-layer, embedding compreso (fino a 2L+1).
  window  solo le ultime W uscite (W = 2S+1): l'embedding e i primi layer escono
          dalla vista dopo pochi layer.

`window` è stata valutata e scartata. Kimi la misura quasi inutile (Tab. 4, 16 layer:
baseline 1.766 · window 1.764 · block 1.746 · full 1.737), perché il valore sta
nell'accesso ai layer LONTANI, non nell'averne molti vicini. Da noi a 12 layer si
addestra, a 36 non impara (norma del gradiente all'init 418 contro 46) e sul 990M
costa il 62% in più per step. `block` sul kernel fuso, con la pre-norm piegata dentro,
rende lo step del 990M più veloce che senza AttnRes. Misure: docs/PAPER_V2.md.

## Il percorso in torch (CPU e riferimento dei test)

Il riferimento di fla impila le sorgenti: `torch.stack` di L tensori [B·T, D], 13.7 GiB
di attivazioni sul 990M a B·T=65536. Non serve: poiché
RMSNorm(v)·(w⊙q) = (v·(w⊙q))·(mean(v²)+ε)^(-1/2), il punteggio di ogni sorgente si
calcola con **due riduzioni su D che producono scalari**, e nessun tensore [B·T, D]
viene materializzato. Su GPU `block` e `full` passano invece dal kernel Triton fuso
di fla (forward + backward), confrontato in fp64 in `eval/bin.gate_arch_v2.py`.
"""

import torch
import torch.nn as nn

try:                                   # kernel Triton fuso (forward + backward) di fla
    from fla.ops.attnres import fused_attnres as _fused_attnres
except Exception:                      # CPU, o fla non installata: resta la forma in torch
    _fused_attnres = None


def _use_fused(sources):
    """Il kernel fla vuole CUDA e sorgenti della stessa forma; il resto va in torch."""
    return (_fused_attnres is not None and AttnResMixer.use_fused
            and sources[0].is_cuda and all(s.shape == sources[0].shape for s in sources))


class AttnResMixer(nn.Module):
    """
    Un punto di applicazione di AttnRes.

    Args:
        d_model: larghezza del modello
        eps:     epsilon della RMSNorm sulle chiavi
        window:  quante sorgenti recenti considerare (None = tutte, forma piena)
    """

    # Interruttore globale del kernel fuso: i gate lo spengono per confrontare
    # il kernel con la forma in torch sugli stessi pesi.
    use_fused = True

    def __init__(self, d_model, eps=1e-6, window=None, rms_plus_eps=False):
        super().__init__()
        # La RMSNorm delle chiavi è quella della definizione (Kimi, fla):
        # v·(mean(v²)+ε)^(-1/2). La finestra divide invece per (rms+ε), la forma con cui
        # è stata misurata: le due coincidono con sorgenti grandi e, con le pseudo-query
        # a zero, danno comunque pesi uniformi all'init.
        self.rms_plus_eps = rms_plus_eps
        # `query` è la pseudo-query del punto; `key_weight` è la scala della RMSNorm
        # applicata alle sorgenti prima del prodotto scalare. Insieme: 2·d parametri.
        self.query = nn.Parameter(torch.zeros(d_model))
        self.key_weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.window = window
        self.d_model = d_model

    def reset_parameters(self):
        # query a zero ⇒ logit tutti nulli ⇒ softmax uniforme: all'inizio AttnRes
        # è la MEDIA delle sorgenti. È un punto di partenza neutro e stabile, e il
        # modello impara a sbilanciarla. Con init casuale partirebbe da un routing
        # arbitrario, che è rumore che si porta dietro per tutto il training.
        nn.init.zeros_(self.query)
        nn.init.ones_(self.key_weight)

    def forward(self, residuals, return_weights=False):
        """
        Args:
            residuals: lista di tensori (B, T, D) — le sorgenti, in ordine di profondità
        Returns:
            (B, T, D) mix pesato, più opzionalmente i pesi softmax (L, B, T)
        """
        srcs = residuals if self.window is None else residuals[-self.window:]
        if len(srcs) == 1:
            return (srcs[0], None) if return_weights else srcs[0]

        dt, shape = srcs[0].dtype, srcs[0].shape
        wq = (self.key_weight * self.query).to(dt)             # (D,)
        n = self.d_model

        # Entrambe le riduzioni sono MATVEC, non prodotti elementwise seguiti da
        # somma: `v @ wq` e `‖v‖` non materializzano mai un [B·T, D] intermedio, e
        # sulle tensor core l'accumulo è comunque in fp32. È qui che sta il
        # risparmio di memoria — un `(v * wq).sum(-1)` scritto in modo ingenuo
        # alloca un temporaneo grande quanto la sorgente, per ogni sorgente.
        logits = []
        for v in srcs:
            flat = v.reshape(-1, n)
            dot = flat @ wq                                              # (B·T,)
            if self.rms_plus_eps:
                # Forma della finestra. Anche l'ordine delle operazioni conta: Inductor
                # fonde diversamente, e sulla 4090 la variante riordinata sfora la
                # memoria condivisa.
                rms = torch.linalg.vector_norm(flat, dim=-1) * (n ** -0.5)   # (B·T,)
                logits.append(dot.float() / (rms.float() + self.eps))
            else:
                ms = torch.linalg.vector_norm(flat, dim=-1).float().square() / n
                logits.append(dot.float() * torch.rsqrt(ms + self.eps))
        p = torch.stack(logits, 0).softmax(0)                            # (L, B·T)

        # Accumulo nel dtype delle sorgenti: l'upcast a fp32 qui raddoppierebbe la
        # memoria senza guadagno numerico (la softmax è già stata fatta in fp32).
        out = srcs[0].reshape(-1, n) * p[0].unsqueeze(-1).to(dt)
        for i in range(1, len(srcs)):
            out = out + srcs[i].reshape(-1, n) * p[i].unsqueeze(-1).to(dt)
        out = out.view(shape)
        return (out, p.view(len(srcs), *shape[:-1])) if return_weights else out

    def mix(self, sources, norm=None):
        """
        Mix delle sorgenti seguito dalla norm del punto di lettura (ln1/ln2/ln_f).

        Con il kernel fuso la RMSNorm viene piegata dentro il kernel (una lettura
        delle sorgenti in meno); se la norm è una GatedRMSNorm resta da applicare
        solo il suo gate. La finestra passa sempre dalla forma in torch.
        """
        if len(sources) == 1:
            return norm(sources[0]) if norm is not None else sources[0]
        if self.window is None and _use_fused(sources):
            # Tutte nello stesso dtype: l'embedding arriva in fp32, le uscite dei
            # sotto-layer in bf16 sotto autocast.
            dt = sources[-1].dtype
            srcs = [s if s.dtype == dt else s.to(dt) for s in sources]
            y = _fused_attnres(self.query, srcs, self.key_weight,
                               output_rms_weight=None if norm is None else norm.weight,
                               rms_eps=self.eps)
            gate = getattr(norm, "gate", None)
            return gate(y) if gate is not None else y
        # La finestra gira fuori da torch.compile: dentro lo stato di profondità
        # Inductor fonde le 13 sorgenti in una riduzione persistente che sulla 4090
        # sfora la memoria condivisa (139 KB su 101). Stessa matematica, solo eager.
        h = _eager_forward(self, sources) if self.window is not None else self.forward(sources)
        return norm(h) if norm is not None else h

    def extra_repr(self):
        return f"d_model={self.d_model}, window={self.window or 'full'}"


_eager_forward = torch.compiler.disable(AttnResMixer.forward)


class DepthState:
    """
    Le sorgenti di AttnRes durante UN forward, per le tre forme.

    Il decoder ne crea uno per forward con l'embedding; ogni sotto-layer ci
    spinge la sua uscita con `push()`, e ogni punto di lettura chiede `sources()`.
    In `block` le uscite si sommano nel blocco aperto e, ogni `block_size`
    sotto-layer, la somma diventa una sorgente chiusa; in `full`/`window` ogni
    uscita è una sorgente (la finestra la applica il mixer).
    """

    def __init__(self, x, mode="block", block_size=8):
        if mode not in ("block", "full", "window"):
            raise ValueError(f"attn_res_mode {mode!r}: usa 'block', 'full' o 'window'")
        if mode == "block" and (not block_size or block_size < 1):
            raise ValueError(f"attn_res_block_size deve essere >= 1, non {block_size!r}")
        self.mode, self.block_size = mode, block_size
        self.done = [x]          # embedding, poi blocchi chiusi (block) o ogni uscita
        self.partial = None      # block: somma delle uscite del blocco aperto
        self.count = 0           # uscite di sotto-layer viste finora

    def sources(self):
        return self.done if self.partial is None else self.done + [self.partial]

    def push(self, y):
        self.count += 1
        if self.mode != "block":
            self.done.append(y)
            return
        self.partial = y if self.partial is None else self.partial + y
        if self.count % self.block_size == 0:
            self.done.append(self.partial)
            self.partial = None

    # ── gradient checkpointing ──
    # Il checkpoint riesegue il blocco nel backward: deve ripartire dallo stato di
    # PRIMA, non da quello già avanzato dai layer successivi. `freeze` fotografa lo
    # stato, `thaw` lo ricostruisce dentro il checkpoint, `adopt` applica fuori i
    # tensori nuovi che il checkpoint restituisce.

    def freeze(self):
        return tuple(self.done), self.partial, self.count

    @classmethod
    def thaw(cls, frozen, mode, block_size):
        done, partial, count = frozen
        st = cls.__new__(cls)
        st.mode, st.block_size = mode, block_size
        st.done, st.partial, st.count = list(done), partial, count
        return st

    def new_tensors(self, n_done_before):
        """I tensori da restituire dal checkpoint: sorgenti chiuse nuove + parziale."""
        return self.done[n_done_before:] + ([] if self.partial is None else [self.partial])

    def adopt(self, tensors, pushes):
        """Applica le uscite di un checkpoint: quante chiudono un blocco si ricalcola dal conteggio."""
        count, open_, closed = self.count, self.partial is not None, 0
        for _ in range(pushes):
            count += 1
            if self.mode != "block":
                closed += 1
                continue
            open_ = True
            if count % self.block_size == 0:
                closed, open_ = closed + 1, False
        if len(tensors) != closed + int(open_):
            raise RuntimeError(f"checkpoint AttnRes: attesi {closed + int(open_)} tensori, "
                               f"arrivati {len(tensors)}")
        self.done.extend(tensors[:closed])
        self.partial = tensors[closed] if open_ else None
        self.count = count


class ResidualStack:
    """
    Le sorgenti vive durante un forward, con la finestra applicata.

    Non è un nn.Module: è solo il contenitore che il decoder passa da un blocco al
    successivo. Serve perché in AttnRes il residual stream non esiste più come
    tensore unico — esiste una LISTA, e chi la tiene deve anche buttare via ciò che
    è fuori finestra, o la memoria cresce con la profondità.
    """

    def __init__(self, window=None):
        self.window = window
        self.sources = []

    def append(self, x):
        self.sources.append(x)
        if self.window is not None and len(self.sources) > self.window:
            # Fuori finestra: si lascia andare il riferimento. I tensori restano nel
            # grafo di autograd finché servono al backward, ma non li teniamo noi.
            self.sources = self.sources[-self.window:]
        return self

    def __len__(self):
        return len(self.sources)
