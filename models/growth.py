"""
=================================================================
@copyright: A. Ivanovitch | CEO SKYL4R | 2026
=================================================================

Growing a trained Skylar model wider (beta, 10/10/2026).

`grow_width(model, factor=2)` returns a model `factor` times wider (d_model, attention heads q and kv, KDA heads,
d_ff) with the same depth, d_head, vocabulary and layer types, that computes THE SAME FUNCTION as the source before
any training. Training then goes on from there: a 4B grown from a trained 990M instead of a 4B from random weights.
It is HyperCloning (Apple, arXiv 2409.12903), carried to every piece of Skylar 2: KDA, AttnRes, GatedNorm, the
per-head output gate, SiTU-GLU.

How the function is kept (k = factor):
  - every hidden vector of the grown model is the source's one repeated k times: x' = [x, x];
  - a weight dimension that produces a hidden vector is repeated ([W; W], the bias too); a dimension that reads one
    is split in k blocks W/k + eta_j with sum_j eta_j = 0, exact because the k copies it reads are equal;
  - heads: the copies go after the originals, so q head i still reads kv head i // n_rep, the copy of its own;
  - RMSNorm: the weight is repeated, the mean of squares of [x, x] is that of x; the per-head norms (QK-Norm, o_norm,
    KDA's o_norm) do not change, d_head does not;
  - KDA: state, short conv, gates and decay are per head or per channel, so duplicated heads give duplicated outputs;
  - AttnRes: key_weight is repeated and the pseudo-query is split like a reading dimension: the logits do not change;
  - tied lm_head: E' = [E, E] against y' = [y, y] would give k times the logits, so ln_f.weight is divided by k
    (LEMON, arXiv 2310.07999, App. C.2); a gated ln_f then reads y/k, and its gate_down is split without the 1/k.

The noise eta breaks the symmetry. With dropout 0, as in our pretraining, a clone without noise gets the same gradient
on every copy forever and stays the small model with more parameters (HyperCloning sees the symmetry break during
training "potentially due to random operations such as dropout"). The noise changes the weights, not the function.
Default 10 dB of signal to noise, the value HyperCloning tested (§3). It has to be random: two copies scaled by
different constants (LEMON) get proportional gradients, and Adam and Muon, which are scale invariant, would still
give them the same update.

Not supported: µP (its logit scale depends on the width) and MTP heads: both raise.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.attn_res import AttnResMixer


def grown_config(config, factor=2):
    """The config of the model `factor` times wider: same depth, d_head, vocabulary and layer types."""
    if int(factor) != factor or factor < 2:
        raise ValueError(f"factor must be an integer >= 2, got {factor!r}")
    k = int(factor)
    if getattr(config, "mup_base_d_model", None) is not None:
        raise ValueError("grow_width does not support µP: its output scale depends on the width")
    if getattr(config, "mtp_layers", 0):
        raise ValueError("grow_width does not support MTP heads: drop them first (they are discarded after "
                         "pretraining anyway)")
    if any(t == "kda" for t in config.layer_types) and config.kda_heads is None:
        raise ValueError("a KDA model without kda_heads")
    new = copy.deepcopy(config)
    new._d_head = config.d_head            # explicit: d_model // n_heads would give the same, say it anyway
    new.d_model = config.d_model * k
    new.n_heads = config.n_heads * k
    new.n_kv_heads = config.n_kv_heads * k
    new.d_ff = config.d_ff * k
    if config.kda_heads is not None:
        new.kda_heads = config.kda_heads * k
    return new


def _widen(w, k, out_dim=None, in_dim=None, in_scale=1.0, gen=None, snr_db=None):
    """One parameter, widened k times. `out_dim`: the dimension that produces a hidden vector, repeated.
    `in_dim`: the dimension that reads one, split in k blocks W / (k·in_scale) + eta_j with sum_j eta_j = 0
    (`in_scale` = the scale of each copy it reads: 1, or 1/k after ln_f). Each output copy gets its own noise."""
    w = w.detach().float().cpu()

    def split():
        if in_dim is None:
            return w
        base = w / (k * in_scale)
        blocks = [base.clone() for _ in range(k)]
        if snr_db:
            std = float(base.pow(2).mean().sqrt()) * 10 ** (-snr_db / 20)
            if std > 0:
                noise = [torch.randn(base.shape, generator=gen) * std for _ in range(k - 1)]
                noise.append(-torch.stack(noise).sum(0))
                blocks = [b + n for b, n in zip(blocks, noise)]
        return torch.cat(blocks, dim=in_dim)

    if out_dim is None:
        return split()
    return torch.cat([split() for _ in range(k)], dim=out_dim)


def _roles(module, pname, ndim):
    """For each dimension of a parameter: 'in' if it reads a hidden vector, else 'out'."""
    if isinstance(module, nn.Linear) and pname == "weight":
        return ("out", "in")
    if isinstance(module, AttnResMixer) and pname == "query":
        return ("in",)
    return ("out",) * ndim


@torch.no_grad()
def grow_width(model, factor=2, noise_snr_db=10.0, seed=0):
    """A model `factor` times wider computing the same function as `model` (see the module docstring).
    `noise_snr_db` None or 0 = no noise (an exact clone that stays symmetric: for tests). Returns the new model, on
    the CPU, each parameter in the source's dtype."""
    k = int(factor)
    cfg = grown_config(model.config, k)
    gen = torch.Generator().manual_seed(seed)
    new = type(model)(cfg)
    tied = bool(cfg.tie_weights)

    done = set()
    for name, src in model.named_modules():
        dst = new.get_submodule(name) if name else new
        for pname, p in src.named_parameters(recurse=False):
            full = f"{name}.{pname}" if name else pname
            if tied and full == "lm_head.weight":
                continue                                   # the same tensor as token_emb.weight
            q = getattr(dst, pname)
            out_dim = in_dim = None
            for d, role in enumerate(_roles(src, pname, p.dim())):
                if q.shape[d] == p.shape[d]:
                    continue
                if q.shape[d] != k * p.shape[d]:
                    raise RuntimeError(f"{full}: {tuple(p.shape)} -> {tuple(q.shape)} is not a {k}x widening")
                if role == "in":
                    in_dim = d
                else:
                    out_dim = d
            in_scale = 1.0
            if tied and full == "ln_f.gate_down.weight":
                in_scale = 1.0 / k                          # it reads y/k: see the docstring
            w = _widen(p, k, out_dim=out_dim, in_dim=in_dim, in_scale=in_scale, gen=gen, snr_db=noise_snr_db)
            if tied and full == "ln_f.weight":
                w = w / k
            if w.shape != q.shape:
                raise RuntimeError(f"{full}: built {tuple(w.shape)}, the grown model wants {tuple(q.shape)}")
            q.data = w.to(p.dtype)
            done.add(id(q))
    missing = [n for n, q in new.named_parameters() if id(q) not in done]
    if missing:
        raise RuntimeError(f"parameters of the grown model with no source: {missing[:6]}")
    for name, b in model.named_buffers():
        nb = new.get_buffer(name)
        if nb.shape == b.shape:
            nb.copy_(b)
    if tied:
        new.tie_weights()
    new.train(model.training)
    return new


@torch.no_grad()
def compare_logits(source, grown, input_ids, document_ids=None):
    """The same batch through both models: the largest logit difference (absolute and relative to the largest
    logit) and both losses. The function is kept if the differences are rounding."""
    a = source(input_ids, labels=input_ids, document_ids=document_ids)
    b = grown(input_ids, labels=input_ids, document_ids=document_ids)
    la, lb = a["logits"].float(), b["logits"].float()
    diff = (la - lb).abs().max().item()
    return {"max_abs": diff, "max_rel": diff / max(la.abs().max().item(), 1e-12),
            "loss_source": a["loss"].item(), "loss_grown": b["loss"].item()}


@torch.no_grad()
def copy_similarity(model, factor=2):
    """How far the copies have moved apart: the cosine between the first and the second copy of the token
    embedding (1.0 = still identical, the symmetry is not broken)."""
    k = int(factor)
    e = model.token_emb.weight.float()
    d = e.shape[1] // k
    return F.cosine_similarity(e[:, :d].flatten(), e[:, d:2 * d].flatten(), dim=0).item()
