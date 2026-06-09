"""
Reserved module — optimizer construction currently lives INLINE.

AdamW setup and the µP per-layer parameter groups (`model.mup_param_groups(...)`) are built
directly inside the entry-point trainers `bin.pretrain.py` and `bin.sft.py`, not in a shared
module. This file is a placeholder for a future extraction of a common optimizer factory;
nothing imports it today. It is not a missing feature.
"""
