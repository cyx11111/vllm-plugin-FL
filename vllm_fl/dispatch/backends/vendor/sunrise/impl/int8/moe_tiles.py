# Copyright (c) 2026 BAAI. All rights reserved.

"""Tile shapes for the INT8 W8A8 routed-MoE grouped GEMMs on Sunrise.

FlagGems pins 32x64x64 / warps=8 for every shape and dtype. That tile is
close to optimal for BF16 experts, not for INT8.

This module owns per-stage N and K. ``BLOCK_SIZE_M`` is shared (align pads
to it; both GEMMs index that padding) and lives in ``impl/moe_block_m.py``.

A stage's tile depends on **K depth** and **rows per expert**
(``M * topk / E``). Deep K (gemm1, K is hidden) takes a narrow deep step at
small batch and a wide shallow one when the block is tall. Shallow K
(gemm2, K is per-rank expert width) is epilogue-heavy: few K steps per
tile, so the choice is mostly N-width vs K-depth, plus a wider
``GROUP_SIZE_M`` swizzle to keep the expert weight hot. Shapes outside
these rules get ``None`` and keep the pinned config.

High-E layers whose per-rank width is itself past the generic "deep"
threshold would otherwise label gemm2 as gemm1; that override is keyed on
tensor geometry. ``BLOCK_SIZE_K`` must divide K (unmasked K loop); see
``_fit_block_k``.
"""

from __future__ import annotations

from typing import Optional

# (BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages)
_DEEP_K_SMALL_BATCH = (64, 256, 8, 3)
_DEEP_K_LARGE_BATCH = (128, 128, 8, 3)
_SHALLOW_K_SMALL_BATCH_DEEP = (128, 128, 8, 3)
_SHALLOW_K_SMALL_BATCH = (256, 64, 8, 3)
_SHALLOW_K_TALL_SHORT_K = (256, 64, 16, 3)
_SHALLOW_K_TALL_LONG_K = (128, 64, 8, 3)
_WIDE_EXPERT_STAGE = (64, 128, 8, 3)

# Lowest K at which a stage counts as deep, i.e. is the hidden dimension rather
# than a per-rank expert width.
_DEEP_K_MIN_K = 1536
# Rows per expert at which a 32-row block starts winning; both stages change
# shape here, because the taller block is what the wider tiles need.
_TALL_MIN_ROWS = 8
# Above this K the small-batch shallow stage wants a 128-deep step.
_SHALLOW_K_DEEP_MIN_K = 512
# From this K up, a tall-block shallow stage takes the 128-wide tile. Below it
# there are too few K steps to amortize anything narrower than 256.
_SHALLOW_K_NARROW_TILE_MIN_K = 128
# Coverage bound: the narrowest operand the shallow rule covers.
_SHALLOW_K_MIN_N = 256

# Block swizzle widths. Deep K is compute-bound and keeps vLLM's default.
_DEEP_K_GROUP = 8
_SHALLOW_K_NARROW_GROUP = 64
_SHALLOW_K_WIDE_GROUP = 16
# K at which the shallow stage stops wanting the widest swizzle.
_SHALLOW_K_WIDE_GROUP_MIN_K = 256
# At the K where the tile narrows, a batch this tall wants the group to narrow
# with it; shorter batches keep the widest group.
_SHALLOW_K_WIDE_GROUP_MIN_ROWS = 32

# Shallowest step the INT8 MMA still runs at: the sunrise backend declares
# min_dot_size (8, 8, 16) for INT8 operands.
_MIN_BLOCK_K = 16

# A hole in the generic K-based stage classifier. Keep this bounded:
# wider, non-128-steppable geometry was slower.
_WIDE_EXPERT_MIN_EXPERTS = 256
_WIDE_EXPERT_MIN_WIDTH = 1536
_WIDE_EXPERT_MAX_WIDTH = 2048

# (BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages, GROUP_SIZE_M)
StageTile = tuple[int, int, int, int, int]


def _fit_block_k(k: int, block_k: int) -> Optional[int]:
    """Deepest step at or below ``block_k`` that divides ``k``.

    The kernel's K loop is unmasked, so a step that leaves a tail is not
    expressible and the launch would fall back to vLLM's kernel. Most MoE
    geometries have a power-of-two K and keep the intended step; the ones
    that do not take a shallower one rather than losing the kernel.
    """
    while block_k >= _MIN_BLOCK_K:
        if k % block_k == 0:
            return block_k
        block_k //= 2
    return None


def _fitted(tile: tuple[int, int, int, int], k: int, group: int) -> Optional[StageTile]:
    block_n, block_k, num_warps, num_stages = tile
    fitted = _fit_block_k(k, min(block_k, k))
    if fitted is None:
        return None
    return (block_n, fitted, num_warps, num_stages, group)


def _shallow_k_group(k: int, rows_per_expert: float) -> int:
    if k >= _SHALLOW_K_WIDE_GROUP_MIN_K:
        return _SHALLOW_K_WIDE_GROUP
    if (
        k >= _SHALLOW_K_NARROW_TILE_MIN_K
        and rows_per_expert >= _SHALLOW_K_WIDE_GROUP_MIN_ROWS
    ):
        return _SHALLOW_K_WIDE_GROUP
    return _SHALLOW_K_NARROW_GROUP


def stage_tile(n: int, k: int, rows_per_expert: float) -> Optional[StageTile]:
    """Tile for one expert GEMM, or ``None`` for a shape this table does not cover."""
    if k >= _DEEP_K_MIN_K:
        if rows_per_expert >= _TALL_MIN_ROWS:
            return _fitted(_DEEP_K_LARGE_BATCH, k, _DEEP_K_GROUP)
        return _fitted(_DEEP_K_SMALL_BATCH, k, _DEEP_K_GROUP)
    if n < _SHALLOW_K_MIN_N:
        return None
    group = _shallow_k_group(k, rows_per_expert)
    if rows_per_expert < _TALL_MIN_ROWS:
        if k >= _SHALLOW_K_DEEP_MIN_K:
            return _fitted(_SHALLOW_K_SMALL_BATCH_DEEP, k, group)
        return _fitted(_SHALLOW_K_SMALL_BATCH, k, group)
    if k >= _SHALLOW_K_NARROW_TILE_MIN_K:
        return _fitted(_SHALLOW_K_TALL_LONG_K, k, group)
    return _fitted(_SHALLOW_K_TALL_SHORT_K, k, group)


def is_wide_expert(
    w1_shape: tuple[int, ...], w2_shape: tuple[int, ...]
) -> bool:
    """Whether this layer is in the high-E, wide-expert region."""
    if len(w1_shape) != 3 or len(w2_shape) != 3:
        return False
    experts, doubled_width, hidden = (int(v) for v in w1_shape)
    experts2, hidden2, width = (int(v) for v in w2_shape)
    return (
        experts == experts2
        and experts >= _WIDE_EXPERT_MIN_EXPERTS
        and hidden == hidden2
        and doubled_width == 2 * width
        and _WIDE_EXPERT_MIN_WIDTH <= width <= _WIDE_EXPERT_MAX_WIDTH
        and hidden % 128 == 0
        and width % 128 == 0
    )


def plan(
    w1_shape, w2_shape, rows_per_expert: float
) -> Optional[tuple[StageTile, StageTile]]:
    """``(gemm1_tile, gemm2_tile)`` when both stages are covered, else None.

    All or nothing on purpose: ``BLOCK_SIZE_M`` is pinned for the layer once a
    plan exists, so it must not be pinned for one stage while the other keeps a
    tile that was never matched against it.
    """
    if len(w1_shape) != 3 or len(w2_shape) != 3:
        return None
    if is_wide_expert(tuple(w1_shape), tuple(w2_shape)):
        tile1 = _fitted(_WIDE_EXPERT_STAGE, int(w1_shape[2]), _DEEP_K_GROUP)
        tile2 = _fitted(
            _WIDE_EXPERT_STAGE, int(w2_shape[2]), _SHALLOW_K_WIDE_GROUP
        )
        if tile1 is None or tile2 is None:
            return None
        return tile1, tile2
    tile1 = stage_tile(int(w1_shape[1]), int(w1_shape[2]), rows_per_expert)
    tile2 = stage_tile(int(w2_shape[1]), int(w2_shape[2]), rows_per_expert)
    if tile1 is None or tile2 is None:
        return None
    return tile1, tile2


def with_tile(config: dict, tile: StageTile) -> dict:
    """Return ``config`` with one stage's N/K tile and launch params."""
    block_n, block_k, num_warps, num_stages, group_m = tile
    return {
        **config,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


__all__ = ["is_wide_expert", "plan", "stage_tile", "with_tile"]
