"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

MTP — Multi-Token Prediction (stile DeepSeek-V3, arXiv 2412.19437).

Il modello normale impara a predire il token successivo. MTP gli chiede, in più,
di predire anche quello **dopo ancora**, con una testa dedicata che riceve sia lo
stato nascosto del trunk sia l'embedding del token che sta per arrivare:

    h'_i = M · [ RMSNorm(h_i) ; RMSNorm(emb(x_{i+1})) ]
    logits_i = lm_head( blocco( h'_i ) )        → target: x_{i+2}

Due motivi per averlo, molto diversi fra loro:

1. **Segnale di training più denso.** Predire due passi avanti costringe la
   rappresentazione a pianificare invece di limitarsi a completare. Su codice, dove
   una riga vincola le successive (una `PIC` in WORKING-STORAGE decide come si
   scriverà la MOVE trenta righe dopo), è il tipo di struttura che vogliamo.
2. **Speculative decoding gratis.** La testa MTP propone il token t+2, il trunk lo
   verifica in un colpo solo: 2-3× di throughput in inferenza. ⚠️ Oggi
   `skylar serve` NON lo implementa — quindi questo secondo motivo è potenziale,
   non incassato. Vale la pena saperlo prima di pagare il 3% di parametri.

## Onestà sulle fonti (regola CLAUDE.md)

L'evidenza pubblica è **DeepSeek-V3, un MoE da 671B**. Non esiste nessuna ablation
pubblicata a ~1-4B denso, e MTP non compare nel report di Kimi K3. È quindi la
componente v2 con la giustificazione più debole: sta dietro flag, spenta di
default, e si decide con una misura nostra (docs/PAPER_V2.md §3.7).

## Il vincolo di memoria, che è reale

Una seconda testa su vocab 64000 raddoppia i logits: a batch 4×8192 sono 4.19 GB
in bf16, 8.4 GB se la cross-entropy fa l'upcast a fp32. Per questo la loss qui è
**calcolata a blocchi** (`chunk_size`): i logits di ogni blocco vivono, producono
il loro contributo e vengono liberati, invece di materializzare l'intero tensore.
Senza, per farlo stare si dovrebbe tagliare il batch — e cambiare i token/step
cambia l'LR ottimale, cioè si introdurrebbe una variabile in più proprio nel run
che deve dimostrare le altre.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.block import TransformerBlock
from models.layers.norm import RMSNorm


def chunked_ce_loss(hidden, lm_head, labels, chunk_size=1024, ignore_index=-100):
    """
    Cross-entropy senza mai materializzare tutti i logits insieme.

    `logits = lm_head(hidden)` su (B, T, 64000) è il singolo tensore più grosso del
    forward — più grosso del modello. Qui si processa un blocco di posizioni alla
    volta: ogni blocco produce i suoi logits, il suo contributo alla loss, e muore.
    Il risultato è identico alla versione intera (media pesata sui token validi),
    la memoria di picco scende di ~T/chunk_size.
    """
    flat_h = hidden.reshape(-1, hidden.shape[-1])
    flat_y = labels.reshape(-1)
    total = flat_h.shape[0]

    loss_sum = flat_h.new_zeros((), dtype=torch.float32)
    n_valid = flat_h.new_zeros((), dtype=torch.float32)
    for i in range(0, total, chunk_size):
        h, y = flat_h[i:i + chunk_size], flat_y[i:i + chunk_size]
        logits = lm_head(h).float()
        valid = (y != ignore_index).sum()
        if valid == 0:
            continue
        loss_sum = loss_sum + F.cross_entropy(
            logits, y, ignore_index=ignore_index, reduction="sum")
        n_valid = n_valid + valid
    return loss_sum / n_valid.clamp(min=1)


class MTPHead(nn.Module):
    """
    Una testa di predizione a distanza k (k=1 → predice due token avanti).

    Condivide `token_emb`, `ln_f` e `lm_head` con il trunk: la testa aggiunge solo
    la proiezione di fusione e un blocco transformer. Alla fine del pretrain si
    **scarta** — il modello deployato non la porta, quindi il costo in inferenza è
    zero anche se in training è +3%.
    """

    def __init__(self, config, depth=1):
        super().__init__()
        self.depth = depth
        d = config.d_model
        self.norm_h = RMSNorm(d)
        self.norm_e = RMSNorm(d)
        # Fusione: [stato del trunk ; embedding del token che arriva] → d.
        # NOME: non finisce in `W_o.weight` né `w2.weight`, quindi NON riceve l'init
        # depth-scaled — corretto, non è una proiezione residuale (trappola #1).
        self.fuse = nn.Linear(2 * d, d, bias=False)
        # `layer_idx=None` → blocco full-attention anche in un modello ibrido. La
        # testa MTP vede una finestra corta e non deve portare stato ricorrente
        # attraverso i passi: un layer ricorrente qui complicherebbe la cache senza
        # dare niente.
        #
        # E AttnRes va DISATTIVATO in questo blocco. AttnRes sceglie fra le sorgenti
        # accumulate lungo la profondità; qui di sorgente ce n'è una sola, quella
        # appena fusa. Ereditando il flag dal config si costruirebbero due mixer che
        # nessuno chiama: parametri senza gradiente, cioè un errore in DDP a run
        # avviato. Trovato dal test, non dalla lettura.
        block_cfg = copy.copy(config)
        block_cfg.attn_res = False
        self.block = TransformerBlock(block_cfg, layer_idx=None)

    def forward(self, hidden, token_emb_fn, input_ids):
        """
        Args:
            hidden:       (B, T, D) uscita del trunk, PRIMA di ln_f
            token_emb_fn: la embedding condivisa del modello
            input_ids:    (B, T) i token in ingresso
        Returns:
            (B, T', D) stati nascosti della testa, e le posizioni valide T' = T - depth
        """
        k = self.depth
        h = hidden[:, :-k]                       # posizione i
        nxt = token_emb_fn(input_ids[:, k:])     # embedding del token i+k
        fused = self.fuse(torch.cat([self.norm_h(h), self.norm_e(nxt)], dim=-1))
        out, _ = self.block(fused)
        return out


class MTPModule(nn.Module):
    """Le teste MTP di un modello (oggi sempre una: `config.mtp_layers`)."""

    def __init__(self, config):
        super().__init__()
        n = int(getattr(config, "mtp_layers", 0) or 0)
        self.heads = nn.ModuleList([MTPHead(config, depth=i + 1) for i in range(n)])
        self.loss_weight = float(getattr(config, "mtp_loss_weight", 0.3))

    def __len__(self):
        return len(self.heads)

    def loss(self, hidden, token_emb_fn, input_ids, labels, ln_f, lm_head, chunk_size=1024):
        """
        Loss ausiliaria media sulle teste, già pesata da `mtp_loss_weight`.

        Restituisce anche il dettaglio per testa: serve perché la loss loggata
        cambia di significato appena MTP è acceso. Se non si separa la CE della
        testa principale, la curva smette di essere confrontabile col run
        precedente — e il confronto è il deliverable.
        """
        if not self.heads:
            return None, {}
        losses, detail = [], {}
        for i, head in enumerate(self.heads):
            k = head.depth
            # La testa predice il token i+1+k a partire dallo stato i
            y = labels[:, k:]
            h = head(hidden, token_emb_fn, input_ids)
            l = chunked_ce_loss(ln_f(h), lm_head, y, chunk_size=chunk_size)
            losses.append(l)
            detail[f"mtp{k}"] = l.detach()
        return self.loss_weight * torch.stack(losses).mean(), detail
