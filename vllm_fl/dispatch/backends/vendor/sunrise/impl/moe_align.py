# Copyright (c) 2026 BAAI. All rights reserved.

"""``moe_align_block_size`` for Sunrise/PTPU.

FlagGems' TLE path is CUDA-only. Its fallback counts into a
``(num_experts + 1, num_experts)`` histogram and prefix-sums it column-wise
-- O(E^2) and independent of token count.

The per-chunk histogram only exists to give each token a deterministic rank
within its expert. Rank only decides which ``BLOCK_SIZE_M`` tile a token
lands in, and any permutation within an expert is bit-identical, so a flat
``[num_experts]`` counter is enough (O(numel + E)).

That trades quadratic work for atomic contention: every token increments
one of ``num_experts`` addresses. Privatizing counters across groups
unclogs that; ``_group_count`` scales rows with both batch and expert
count, with a scratch-size cap so few-expert models get more rows rather
than overflowing the memset. ``tl.atomic_add`` already returns the old
value, so one contended pass is enough and scatter is arithmetic.

The three phases mirror FlagGems (fill+count, prefix, scatter+label) in
three launches instead of four.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from vllm.utils.math_utils import round_up

# One tile width for every phase. The atomic pass wants many small programs,
# not few large ones.
BLOCK = 256

# Widest private-counter scratch worth zeroing per call, as groups * num_experts.
# Contention is numel/(num_experts*groups), so fewer experts need more rows.
# Cap the product so the memset stays bounded while the row count follows
# the geometry.
_MAX_SCRATCH_COUNTERS = 2048
# Floor, so an expert count wider than the budget still gets some rows.
_MIN_GROUP_CAP = 8
# Tokens per counter address below which privatizing cannot pay for itself.
_TOKENS_PER_ADDRESS = 4


def _group_count(numel: int, num_experts: int) -> int:
    """Private counter rows: enough to unclog the atomics, and no more."""
    cap = max(_MIN_GROUP_CAP, _MAX_SCRATCH_COUNTERS // max(1, num_experts))
    target = numel // (_TOKENS_PER_ADDRESS * num_experts)
    groups = 1
    while groups < cap and groups * 2 <= target:
        groups *= 2
    return groups


@triton.jit(do_not_specialize=["numel"])
def _fill_and_rank(
    topk_ids_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    counts_ptr,
    ranks_ptr,
    numel,
    max_padded,
    max_blocks,
    GROUPS: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Prime both outputs with their skip values, histogram into a private row,
    and keep the rank the counting atomic already returned."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # ``numel`` is one past the last valid token index, so the expert GEMM masks
    # off every slot the scatter phase does not overwrite.
    tl.store(sorted_ids_ptr + offs, numel, mask=offs < max_padded)
    # -1 marks a block no expert owns; the expert GEMM checks for it and skips.
    tl.store(expert_ids_ptr + offs, -1, mask=offs < max_blocks)

    mask = offs < numel
    expert = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    row = (pid % GROUPS) * num_experts
    rank = tl.atomic_add(counts_ptr + row + expert, 1, mask=mask)
    tl.store(ranks_ptr + offs, rank, mask=mask)


@triton.jit
def _starts_and_group_offsets(
    counts_ptr,
    starts_ptr,
    goff_ptr,
    num_tokens_post_pad_ptr,
    GROUPS: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    block_size: tl.constexpr,
):
    """Where each expert's padded run begins, and where each group sits inside it."""
    group = tl.arange(0, GROUPS)[:, None]
    expert = tl.arange(0, BLOCK_EXPERT)[None, :]
    counts = tl.load(
        counts_ptr + group * num_experts + expert,
        mask=expert < num_experts,
        other=0,
    )

    # Exclusive prefix over the groups: a group's tokens follow every earlier
    # group's within the same expert.
    tl.store(
        goff_ptr + group * num_experts + expert,
        tl.cumsum(counts, axis=0) - counts,
        mask=expert < num_experts,
    )

    offs = tl.arange(0, BLOCK_EXPERT)
    mask = offs < num_experts
    total = tl.sum(counts, axis=0)
    aligned = tl.cdiv(total, block_size) * block_size
    # Padding lanes must not contribute to the running sum.
    aligned = tl.where(mask, aligned, 0)

    tl.store(starts_ptr + offs, tl.cumsum(aligned, axis=0) - aligned, mask=mask)
    # The label phase needs the unaligned totals to know how many blocks it owns.
    tl.store(starts_ptr + num_experts + offs, total, mask=mask)
    tl.store(num_tokens_post_pad_ptr, tl.sum(aligned, axis=0))


@triton.jit(do_not_specialize=["numel"])
def _scatter_and_label(
    topk_ids_ptr,
    ranks_ptr,
    starts_ptr,
    goff_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    numel,
    GROUPS: tl.constexpr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Place each token at a position arithmetic already fixed, and stamp the
    block owners."""
    pid = tl.program_id(0)

    # Experts own disjoint, block-aligned regions, so no two programs here write
    # the same expert_ids slot.
    if pid < num_experts:
        start = tl.load(starts_ptr + pid)
        total = tl.load(starts_ptr + num_experts + pid)
        base = start // block_size
        for i in range(tl.cdiv(total, block_size)):
            tl.store(expert_ids_ptr + base + i, pid)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    expert = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    rank = tl.load(ranks_ptr + offs, mask=mask, other=0)
    row = (pid % GROUPS) * num_experts
    start = tl.load(starts_ptr + expert, mask=mask, other=0)
    goff = tl.load(goff_ptr + row + expert, mask=mask, other=0)
    tl.store(sorted_ids_ptr + start + goff + rank, offs, mask=mask)


def moe_align_block_size_sunrise(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group ``topk_ids`` into block-aligned, per-expert runs.

    Returns ``(sorted_ids, expert_ids, num_tokens_post_pad)`` with the same
    contract as the FlagGems and CUDA implementations:

    * ``sorted_ids`` ``[max_num_tokens_padded]`` — flat token indices grouped by
      expert, each expert's run padded up to a multiple of ``block_size``. Unused
      slots hold ``topk_ids.numel()``, which the expert GEMM treats as invalid.
    * ``expert_ids`` ``[max_num_m_blocks]`` — expert owning each block, ``-1``
      for blocks nobody owns.
    * ``num_tokens_post_pad`` ``[1]`` — total padded length.

    Tokens are ordered arbitrarily within an expert (see the module docstring);
    the grouping and the padded layout are what the GEMM requires.
    """
    del ignore_invalid_experts  # FlagGems ignores it too; kept for signature parity

    if topk_ids.numel() == 0:
        empty = torch.empty((0,), dtype=torch.int32, device=topk_ids.device)
        return (
            empty,
            empty.clone(),
            torch.zeros((1,), dtype=torch.int32, device=topk_ids.device),
        )

    topk_ids = topk_ids.contiguous()
    numel = topk_ids.numel()
    device = topk_ids.device
    groups = _group_count(numel, num_experts)

    # Buffer sizing must match the FlagGems wrapper exactly: the expert GEMM
    # derives its M grid from sorted_ids.size(0), not from num_tokens_post_pad.
    max_padded = numel + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_padded = round_up(max_padded, block_size)
    if numel < num_experts:
        # At most one block per token can ever be occupied.
        max_padded = min(numel * block_size, max_padded)
    max_blocks = triton.cdiv(max_padded, block_size)

    sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    ranks = torch.empty((numel,), dtype=torch.int32, device=device)
    # counts feeds atomics and must start at zero; the group offsets, the starts
    # and the per-expert totals ride along in the same allocation so the whole
    # scratch costs one memset instead of four.
    scratch = torch.zeros(
        (2 * groups + 2, num_experts), dtype=torch.int32, device=device
    )
    counts = scratch[:groups]
    goff = scratch[groups : 2 * groups]
    starts = scratch[2 * groups :]

    _fill_and_rank[(triton.cdiv(max_padded, BLOCK),)](
        topk_ids,
        sorted_ids,
        expert_ids,
        counts,
        ranks,
        numel,
        max_padded,
        max_blocks,
        GROUPS=groups,
        num_experts=num_experts,
        BLOCK=BLOCK,
    )
    _starts_and_group_offsets[(1,)](
        counts,
        starts,
        goff,
        num_tokens_post_pad,
        GROUPS=groups,
        num_experts=num_experts,
        BLOCK_EXPERT=triton.next_power_of_2(num_experts),
        block_size=block_size,
    )
    _scatter_and_label[(max(triton.cdiv(numel, BLOCK), num_experts),)](
        topk_ids,
        ranks,
        starts,
        goff,
        sorted_ids,
        expert_ids,
        numel,
        GROUPS=groups,
        num_experts=num_experts,
        block_size=block_size,
        BLOCK=BLOCK,
    )

    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


__all__ = ["moe_align_block_size_sunrise"]
