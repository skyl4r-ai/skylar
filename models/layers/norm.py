"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Normalization layers.

RMSNorm — Root Mean Square Layer Normalization (used in LLaMA, Gemma, etc.).
GatedRMSNorm — RMSNorm followed by a low-rank sigmoid self-gate (GatedNorm).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA, Gemma, etc.)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight.type_as(x)


class GatedRMSNorm(RMSNorm):
    """
    GatedNorm (Qiu et al., arXiv 2601.22966): RMSNorm seguita da un gate sigmoid
    elementwise, calcolato dall'uscita stessa con un collo di bottiglia low-rank:

        y  = RMSNorm(x)
        y' = y ⊙ σ(W_up · SiLU(W_down · y))        W_down: d→r, W_up: r→d

    Perché: senza un riscalamento esplicito la rete lo ottiene facendo crescere
    outlier di attivazione, ed è da lì che nascono i picchi di loss a learning
    rate alto. Il gate fornisce il riscalamento direttamente. Numeri delle fonti:
    −0.006 di loss su un 2B denso (2601.22966); −0.004/−0.005 sopra AttnRes e, a
    3× l'LR ottimale tenuto costante, picchi da 32.0 a 3.2 ogni 10k step
    (Qwen, arXiv 2608.30320, Tab. 6 e Fig. 12). Per noi conta il secondo numero:
    il pretrain WSD-WSM tiene l'LR costante per quasi tutto il run.

    Costo: 2·d·r parametri per norm (r=16 → 49k sul 990M, per norm).
    `gate()` è separato da `forward()` perché con Block AttnRes la RMSNorm viene
    piegata dentro il kernel fuso: lì serve solo il gate, applicato dopo.
    """

    def __init__(self, dim, rank=16, eps=1e-6):
        super().__init__(dim, eps)
        self.gate_down = nn.Linear(dim, rank, bias=False)
        self.gate_up = nn.Linear(rank, dim, bias=False)

    def gate(self, y):
        return y * torch.sigmoid(self.gate_up(F.silu(self.gate_down(y))))

    def forward(self, x):
        return self.gate(super().forward(x))


def make_norm(config, dim=None):
    """La norm dei punti di lettura del residuo: gated se `config.gated_norm` (il rango) > 0."""
    d = dim or config.d_model
    rank = getattr(config, "gated_norm", 0) or 0
    return GatedRMSNorm(d, rank=rank) if rank else RMSNorm(d)
