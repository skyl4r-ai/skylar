"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Optimizer construction shared by the trainers: AdamW, or Muon on the hidden linear maps with AdamW on
every other parameter (the Skylar 2 report §6.6).

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
        return {"optimizer_set": [o.state_dict() for o in self.opts], "types": [_kind(o) for o in self.opts]}

    def load_state_dict(self, state):
        """All or nothing: a saved set with fewer optimizers (zip would leave the last one fresh, its momentum lost
        without a word) or of other types is refused before anything is loaded."""
        saved = state.get("optimizer_set") if isinstance(state, dict) else None
        if saved is None:
            raise ValueError("this checkpoint holds one optimizer, the run builds several (--optimizer changed?)")
        kinds = [_kind(o) for o in self.opts]
        if len(saved) != len(self.opts) or state.get("types", kinds) != kinds:
            raise ValueError(f"the checkpoint holds {len(saved)} optimizers {state.get('types', '')}, the run builds "
                             f"{len(self.opts)} {kinds}")
        for o, st in zip(self.opts, saved):
            have = [len(g["params"]) for g in st["param_groups"]]
            want = [len(g["params"]) for g in o.param_groups]
            if have != want:
                raise ValueError(f"{_kind(o)}: parameter groups of {have} tensors in the checkpoint, {want} in the run")
        for o, st in zip(self.opts, saved):
            o.load_state_dict(st)


def _kind(o):
    """The optimizer's class name, through accelerate's wrapper."""
    return type(getattr(o, "optimizer", o)).__name__


def is_muon_matrix(name, p):
    parts = name.split(".")
    return p.dim() == 2 and len(parts) >= 2 and parts[-2] in MUON_MATRICES


def default_decay(name, p):
    """Weight decay on matrices only. Norm weights and biases are excluded (decay pulls a norm towards
    switching its layer off), and so are the parameters that `fla` marks `_no_weight_decay` (A_log and
    dt_bias of KDA: they set the dynamics of the forget gate, not capacity)."""
    return p.dim() >= 2 and not getattr(p, "_no_weight_decay", False)


def build_optimizer(model, lr, weight_decay=0.0, optimizer="adamw", betas=(0.9, 0.95),
                    decay=default_decay, embed_weight_decay=None, **adamw_kwargs):
    """AdamW over all trainable parameters, with weight decay where `decay(name, param)` is True;
    or, with optimizer="muon", Muon (Nesterov 0.95, 5 Newton-Schulz steps, update scaled to the RMS of
    AdamW so that `lr` and `weight_decay` keep AdamW's scale) over the hidden linear maps and AdamW over
    the rest. Returns (optimizer, info string). The learning rate of every group is set through
    `optimizer.param_groups`, whichever the type.
    embed_weight_decay: None = the token embedding (with tied weights, also the output head) decays like every
    matrix; a number = its own AdamW group with that decay (0 in OLMo 2, arXiv 2501.00656 §3.4.2, and in SmolLM3).
    It changes the groups of the optimizer: decided at launch, a resume across the change is refused."""
    if optimizer not in ("adamw", "muon"):
        raise ValueError(f"optimizer must be 'adamw' or 'muon', got {optimizer!r}")
    decayed, plain, muon, embed = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if optimizer == "muon" and is_muon_matrix(n, p):
            muon.append(p)
        elif embed_weight_decay is not None and n.split(".")[-2:] == ["token_emb", "weight"]:
            embed.append(p)
        elif decay(n, p):
            decayed.append(p)
        else:
            plain.append(p)
    groups = [{"params": decayed, "weight_decay": weight_decay}, {"params": plain, "weight_decay": 0.0}]
    if embed:
        groups.append({"params": embed, "weight_decay": embed_weight_decay})
    opt = torch.optim.AdamW(groups, lr=lr, betas=betas, **adamw_kwargs)
    if muon:
        opt = OptimizerSet([opt, torch.optim.Muon(
            muon, lr=lr, weight_decay=weight_decay, momentum=0.95, nesterov=True, ns_steps=5,
            adjust_lr_fn="match_rms_adamw")])
    info = (f"weight decay {weight_decay} on {len(decayed)} tensors; {len(plain)} without"
            + (f"; token embedding {embed_weight_decay}" if embed else "")
            + (f"; Muon on {len(muon)} matrices" if muon else ""))
    return opt, info
