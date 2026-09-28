"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | 2026
=================================================================

AttnRes — attenzione softmax sulla PROFONDITÀ (arXiv 2603.15031).

In un transformer normale ogni blocco legge un solo tensore, il residual stream,
che è la somma di tutto quello che è venuto prima: `x = x + attn(x)`. Le sorgenti
sono sommate con peso 1, sempre, per ogni token.

AttnRes toglie quella somma e la sostituisce con una scelta: il residual stream
viene AZZERATO a ogni confine di blocco, le uscite dei blocchi precedenti restano
disponibili come sorgenti separate, e ogni punto di applicazione le pesa con una
softmax — decidendo **per token** da quale profondità leggere.

    k_l = RMSNorm(v_l)
    p   = softmax_l( k_l · q )         q = pseudo-query del punto, imparata
    o   = Σ_l p_l · v_l

Costa 2·d parametri per punto di applicazione (la query e il peso della norm):
sul 990M sono 224k parametri in tutto, lo 0.02% del modello. È il modo più
economico che conosciamo di aggiungere espressività senza aggiungere size — che è
esattamente il problema del 980M (capacity-bound: il pass@1 non si è mai mosso).

## La riformulazione che elimina il costo di memoria

L'implementazione di riferimento (`fla/ops/attnres/naive.py`) impila le sorgenti:
`torch.stack` di L tensori [B·T, D]. Sul 990M a B·T=65536 sono **13.7 GiB** di
attivazioni vive, ed è la ragione per cui si finisce a usare finestre corte.

Non serve. Poiché RMSNorm(v)·(w⊙q) = (v·(w⊙q)) / rms(v), il punteggio di ogni
sorgente si calcola con **due riduzioni su D che producono scalari**:

    score_l = (v_l · wq) / rms(v_l)          wq = w ⊙ q, precalcolato una volta

Nessun tensore [B·T, D] viene mai materializzato: il costo in memoria scende da
W·B·T·D a W·B·T, cioè **1/D**, ed è la stessa matematica bit per bit (verificato
contro il riferimento fla in `eval/bin.gate_arch_v2.py`). Con questa forma la
scelta fra finestra corta e forma piena torna a essere una scelta di espressività
invece che una resa alla HBM.
"""

import torch
import torch.nn as nn


class AttnResMixer(nn.Module):
    """
    Un punto di applicazione di AttnRes.

    Args:
        d_model: larghezza del modello
        eps:     epsilon della RMSNorm sulle chiavi
        window:  quante sorgenti recenti considerare (None = tutte, forma piena)
    """

    def __init__(self, d_model, eps=1e-6, window=None):
        super().__init__()
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
            rms = torch.linalg.vector_norm(flat, dim=-1) * (n ** -0.5)   # (B·T,)
            logits.append(dot.float() / (rms.float() + self.eps))
        p = torch.stack(logits, 0).softmax(0)                            # (L, B·T)

        # Accumulo nel dtype delle sorgenti: l'upcast a fp32 qui raddoppierebbe la
        # memoria senza guadagno numerico (la softmax è già stata fatta in fp32).
        out = srcs[0].reshape(-1, n) * p[0].unsqueeze(-1).to(dt)
        for i in range(1, len(srcs)):
            out = out + srcs[i].reshape(-1, n) * p[i].unsqueeze(-1).to(dt)
        out = out.view(shape)
        return (out, p.view(len(srcs), *shape[:-1])) if return_weights else out

    def extra_repr(self):
        return f"d_model={self.d_model}, window={self.window or 'full'}"


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
