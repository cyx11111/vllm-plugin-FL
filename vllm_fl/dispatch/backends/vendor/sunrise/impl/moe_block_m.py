# Copyright (c) 2026 BAAI. All rights reserved.

"""``BLOCK_SIZE_M`` for the fused-MoE expert GEMMs on Sunrise.

FlagGems pins 32 for every shape. ``BLOCK_SIZE_M`` is the one tile axis the
two expert GEMMs must share, because ``moe_align_block_size`` pads to it.

INT8 uses rows per expert: a short block at small batch, a taller one once
the block fills. High-E, wide-expert layers add a taller prefill-only band
(see ``int8/moe_tiles.py``). BF16 uses the same axis with different bands:
routing spreads a batch over experts, so token count is not the real batch.

The two dtypes still disagree and are decided separately. Nothing here is
keyed on N: only INT8 gets per-stage tiles, and those no longer come through
this function.
"""

from __future__ import annotations

# Rows-per-expert bounds and the BLOCK_SIZE_M each band takes. Ordered; the first
# bound that fits wins.
_BANDS = ((16, 16), (32, 32))
_WIDE_BATCH = 64

# Rows per expert at which the INT8 kernel switches from a 16-row block to a
# 32-row one.
_INT8_TALL_MIN_ROWS = 8
_INT8_BLOCK_M = 16
_INT8_TALL_BLOCK_M = 32
_INT8_WIDE_BLOCK_M = 64
_INT8_WIDE_TALL_MIN_ROWS = 32


def block_size_m_int8(
    rows_per_expert: float, use_wide_prefill_block: bool = False
) -> int:
    """``BLOCK_SIZE_M`` for the INT8 W8A8 expert GEMMs of one layer."""
    if use_wide_prefill_block and rows_per_expert >= _INT8_WIDE_TALL_MIN_ROWS:
        return _INT8_WIDE_BLOCK_M
    if rows_per_expert >= _INT8_TALL_MIN_ROWS:
        return _INT8_TALL_BLOCK_M
    return _INT8_BLOCK_M


def block_size_m(rows_per_expert: float) -> int:
    """``BLOCK_SIZE_M`` for the BF16 expert GEMMs of one layer."""
    for max_rows, block_m in _BANDS:
        if rows_per_expert < max_rows:
            return block_m
    return _WIDE_BATCH


def rows_per_expert(m: int, top_k: int, num_experts: int) -> float:
    """Routed rows one expert sees, the axis ``block_size_m`` is tuned on."""
    if num_experts <= 0:
        return float(m) * top_k
    return float(m) * top_k / num_experts


__all__ = ["block_size_m", "block_size_m_int8", "rows_per_expert"]
