"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Feed-Forward Networks.

Due varianti, scelte da `config.hidden_act`:
  - "swiglu"   (default, v1) — SwiGLU: silu(w1·x) ⊙ w3·x
  - "situ_glu" (v2)          — SwiGLU con entrambi i fattori illimitati
                               soft-clippati, così l'attivazione è LIMITATA.

Perché SiTU-GLU (docs/PAPER_V2.md §3.6): in SwiGLU sia `w1·x` che `w3·x`
sono illimitati, quindi il prodotto può esplodere e la scala delle attivazioni
non è nota a priori. SiTU li passa in un tanh scalato:

    situ(x) = β₁·tanh(w1·x / β₁) ⊙ σ(w1·x) ⊙ β₂·tanh(w3·x / β₂)

Per attivazioni piccole tanh(z) ≈ z e la funzione **coincide con SwiGLU**; per
attivazioni grandi satura, e il modulo dell'uscita è limitato da β₁·β₂ (= 100 coi
default). Un range noto è la precondizione per quantizzare a FP8/FP4 senza
ri-addestrare — motivo per cui entra ora e non dopo.

⚠️ β₁=4 / β₂=25 sono i valori di Kimi K3, tarati sul LORO regime (hidden 7168,
ramo routed a 4 matmul). Per noi sono un punto di partenza **da ri-tarare**, non
un numero fondato: vedi la regola delle fonti in CLAUDE.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Il core di SiTU è tutto elementwise: senza fusione costa ~+24% sul forward del
# blocco (misurato sulla 4090). `torch.compile` lo fonde in un kernel unico e il
# costo torna in linea con SwiGLU. Si compila alla prima chiamata, non all'import.
_situ_compiled = None


def _situ_core(g, u, beta1: float, beta2: float):
    return (beta1 * torch.tanh(g / beta1)) * torch.sigmoid(g) * (beta2 * torch.tanh(u / beta2))


def _get_situ_kernel():
    global _situ_compiled
    if _situ_compiled is None:
        try:
            # `dynamic=True` NON è un dettaglio. Con `dynamic=False` inductor
            # specializza il kernel sulla lunghezza di sequenza, e in generazione la
            # lunghezza cresce di uno a ogni token: dopo 8 ricompilazioni torch hit
            # il `recompile_limit` e ricade in eager — silenziosamente, con solo un
            # warning. La fusione che rende SiTU gratuito (+0.6% invece di +24%)
            # sparirebbe proprio in inferenza. Il core è tutto elementwise, quindi le
            # forme dinamiche non costano nulla.
            _situ_compiled = torch.compile(_situ_core, dynamic=True)
        except Exception:
            _situ_compiled = _situ_core       # torch senza inductor: funziona, più lento
    return _situ_compiled


class FeedForward(nn.Module):
    """SwiGLU (default) o SiTU-GLU, a parità di parametri."""

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.hidden_act = getattr(config, "hidden_act", "swiglu")
        if self.hidden_act not in ("swiglu", "situ_glu"):
            raise ValueError(f"hidden_act {self.hidden_act!r} sconosciuto "
                             f"(attesi 'swiglu' o 'situ_glu')")
        self.beta1 = float(getattr(config, "situ_beta1", 4.0))
        self.beta2 = float(getattr(config, "situ_beta2", 25.0))

    def forward(self, x):
        if self.hidden_act == "swiglu":
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        h = _get_situ_kernel()(self.w1(x), self.w3(x), self.beta1, self.beta2)
        return self.dropout(self.w2(h))
