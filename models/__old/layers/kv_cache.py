"""
=================================================================
@copyright: A. Ivanovitch | CEO MwSpace | 2026
=================================================================

KV-Cache utilities for inference.

- repeat_kv: expand KV heads for GQA
- validate_kv_cache: runtime validation of cache shapes
"""

import torch


def repeat_kv(x, n_rep):
    """Repeat KV heads to match the number of Q heads (for GQA)."""
    if n_rep == 1:
        return x
    B, n_kv_heads, T, d_head = x.shape
    return (
        x[:, :, None, :, :]  # (B, n_kv, 1, T, d)
        .expand(B, n_kv_heads, n_rep, T, d_head)
        .reshape(B, n_kv_heads * n_rep, T, d_head)
    )


def validate_kv_cache(kv_cache, blocks, batch_size, x_device):
    """
    Validate KV-cache structure and shapes against model blocks.

    Args:
        kv_cache:    list of (k, v) tuples per layer
        blocks:      nn.ModuleList of TransformerBlock
        batch_size:  expected batch size
        x_device:    expected device
    """
    if not isinstance(kv_cache, (list, tuple)):
        raise TypeError(f"kv_cache must be list/tuple, got {type(kv_cache).__name__}")

    if len(kv_cache) != len(blocks):
        raise ValueError(
            f"kv_cache length ({len(kv_cache)}) != n_layers ({len(blocks)})"
        )

    for i, layer_kv in enumerate(kv_cache):
        if not isinstance(layer_kv, (list, tuple)) or len(layer_kv) != 2:
            raise TypeError(
                f"kv_cache[{i}] must be (k, v) tuple, got {type(layer_kv).__name__}"
            )

        k, v = layer_kv
        if not torch.is_tensor(k) or not torch.is_tensor(v):
            raise TypeError(f"kv_cache[{i}] entries must be tensors")

        if k.device != x_device or v.device != x_device:
            raise ValueError(
                f"kv_cache[{i}] device mismatch: "
                f"k={k.device}, v={v.device}, expected={x_device}"
            )

        attn = blocks[i].attn
        expected_heads = attn.n_kv_heads
        expected_d_head = attn.d_head

        # Expected shape: (B, n_kv_heads, T_cache, d_head)
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                f"kv_cache[{i}] tensors must be rank-4, got k.ndim={k.ndim}, v.ndim={v.ndim}"
            )

        if k.shape[0] != batch_size or v.shape[0] != batch_size:
            raise ValueError(
                f"kv_cache[{i}] batch mismatch: "
                f"k.shape[0]={k.shape[0]}, v.shape[0]={v.shape[0]}, expected={batch_size}"
            )

        if k.shape[1] != expected_heads or v.shape[1] != expected_heads:
            raise ValueError(
                f"kv_cache[{i}] n_kv_heads mismatch: "
                f"k.shape[1]={k.shape[1]}, v.shape[1]={v.shape[1]}, expected={expected_heads}"
            )

        if k.shape[3] != expected_d_head or v.shape[3] != expected_d_head:
            raise ValueError(
                f"kv_cache[{i}] d_head mismatch: "
                f"k.shape[3]={k.shape[3]}, v.shape[3]={v.shape[3]}, expected={expected_d_head}"
            )

        if k.shape[2] != v.shape[2]:
            raise ValueError(
                f"kv_cache[{i}] seq_len mismatch between k and v: "
                f"{k.shape[2]} vs {v.shape[2]}"
            )
