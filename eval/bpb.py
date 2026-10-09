# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Bits per byte with a sliding window, the measure of the stopping rule (docs/PAPER_V2.md §7): every token is
predicted with at least `seq_len - stride` tokens of context (except in the first window), so the number does
not depend on where the cuts fall. Comparable across tokenisers and vocabularies, unlike perplexity.

One definition, used by `eval/bin.bits_per_byte.py` (a checkpoint, one process) and by the pretraining loop
(`--bpb_set`, the windows split across the ranks, the sums all-reduced).
"""
import math

import torch


@torch.no_grad()
def bpb_sums(model, ids, device, seq_len=2048, stride=1024, rank=0, world=1):
    """(nats, predicted tokens) over the windows of `ids` that belong to this rank (window w goes to rank
    w % world). Summed over the ranks they give exactly the single-process result."""
    total_nats, n_pred = 0.0, 0
    for w, start in enumerate(range(0, len(ids) - 1, stride)):
        chunk = ids[start:start + seq_len + 1]
        if len(chunk) < 2:
            break
        if w % world != rank:
            continue
        x = torch.tensor([chunk[:-1]], device=device)
        y = torch.tensor([chunk[1:]], device=device)
        skip = 0 if start == 0 else (seq_len - stride)
        logits = model(x)["logits"].float()
        total_nats += torch.nn.functional.cross_entropy(logits[0, skip:], y[0, skip:], reduction="sum").item()
        n_pred += y.shape[1] - skip
    return total_nats, n_pred


def bits_per_byte(model, tok, text, device, seq_len=2048, stride=1024):
    """bpb of `text` with the sliding window (single process)."""
    n_bytes = len(text.encode("utf-8"))
    ids = tok.encode(text).ids
    if len(ids) < 2:
        return None
    total_nats, n_pred = bpb_sums(model, ids, device, seq_len, stride)
    return {
        "bpb": total_nats / math.log(2) / n_bytes,
        "bytes": n_bytes,
        "tokens": n_pred,
        "bytes_per_token": n_bytes / max(n_pred, 1),
    }
