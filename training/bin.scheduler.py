# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
"""
Reserved module — the LR schedule currently lives INLINE.

Warmup + cosine / WSD (warmup-stable-decay) annealing are implemented inline in the
entry-point trainers (see the `lr_at()` / schedule logic in `bin.pretrain.py` and
`bin.sft.py`). This file is a placeholder for a future extraction of a shared scheduler;
nothing imports it today. It is not a missing feature.
"""
