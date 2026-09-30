"""
=================================================================
@copyright: A. Ivanovitch | skyl4r.ai | 2026
=================================================================

Optimizer construction shared by the trainers: AdamW, or Muon on the hidden linear maps with AdamW on
every other parameter (docs/PAPER_V2.md §6.6).

A model pretrained with Muon should be post-trained with Muon: Moonlight (arXiv 2502.16982, Table 6)
finds Muon in pretraining + Muon in SFT the best pairing, and no significant advantage of Muon over
AdamW in SFT when the pretraining optimizer was the other one.

    from training.optim import build_optimizer
    opt, info = build_optimizer(model, lr=2e-5, weight_decay=0.1, optimizer="muon")
    print(info)                        # counts of decayed / non-decayed / Muon tensors
"""

import torch

# The projections that are true linear maps (attention, KDA, feed-forward): the only parameters on which
# Muon is meaningful. Embedding and head, norms, biases, AttnRes pseudo-queries and the low-rank gates
# (KDA, gated normalisation, output gate) stay on AdamW, as in 2608.30320 §3.1.
MUON_MATRICES = {"W_q", "W_k", "W_v", "W_o", "w1", "w2", "w3"}


class OptimizerSet:
    """Several optimizers driven as one, so the training loop does not change."""

    def __init__(self, opts):
        self.opts = list(opts)

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self):
        for o in self.opts:
            o.step()

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizer_set": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, state):
        for o, st in zip(self.opts, state["optimizer_set"]):
            o.load_state_dict(st)


def is_muon_matrix(name, p):
    parts = name.split(".")
    return p.dim() == 2 and len(parts) >= 2 and parts[-2] in MUON_MATRICES


def default_decay(name, p):
    """Weight decay on matrices only. Norm weights and biases are excluded (decay pulls a norm towards
    switching its layer off), and so are the parameters that `fla` marks `_no_weight_decay` (A_log and
    dt_bias of KDA: they set the dynamics of the forget gate, not capacity)."""
    return p.dim() >= 2 and not getattr(p, "_no_weight_decay", False)


def build_optimizer(model, lr, weight_decay=0.0, optimizer="adamw", betas=(0.9, 0.95),
                    decay=default_decay, **adamw_kwargs):
    """AdamW over all trainable parameters, with weight decay where `decay(name, param)` is True;
    or, with optimizer="muon", Muon (Nesterov 0.95, 5 Newton-Schulz steps, update scaled to the RMS of
    AdamW so that `lr` and `weight_decay` keep AdamW's scale) over the hidden linear maps and AdamW over
    the rest. Returns (optimizer, info string). The learning rate of every group is set through
    `optimizer.param_groups`, whichever the type."""
    if optimizer not in ("adamw", "muon"):
        raise ValueError(f"optimizer must be 'adamw' or 'muon', got {optimizer!r}")
    decayed, plain, muon = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if optimizer == "muon" and is_muon_matrix(n, p):
            muon.append(p)
        elif decay(n, p):
            decayed.append(p)
        else:
            plain.append(p)
    opt = torch.optim.AdamW([{"params": decayed, "weight_decay": weight_decay},
                             {"params": plain, "weight_decay": 0.0}],
                            lr=lr, betas=betas, **adamw_kwargs)
    if muon:
        opt = OptimizerSet([opt, torch.optim.Muon(
            muon, lr=lr, weight_decay=weight_decay, momentum=0.95, nesterov=True, ns_steps=5,
            adjust_lr_fn="match_rms_adamw")])
    info = (f"weight decay {weight_decay} on {len(decayed)} tensors; {len(plain)} without"
            + (f"; Muon on {len(muon)} matrices" if muon else ""))
    return opt, info
