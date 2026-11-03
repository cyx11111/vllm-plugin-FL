# Copyright (c) 2026 BAAI. All rights reserved.

"""Prefill attention on the vendor's paged kernel instead of FlagGems' varlen.

``flash_attn_varlen_func`` from ``flaggems_vllm`` serves every attention
regime from one generic Triton kernel. Prefill instead uses
``llm_kernel_ptpu.paged_prefill_attention_fwd`` on the same paged cache.

The ragged kernel is faster but needs contiguous per-request K/V, so it
cannot serve chunked prefill after the first chunk or a prefix-cache hit.
The paged kernel takes separate ``qo_ind`` / ``kv_ind`` cumsums against the
real cache, covering all prefill. Decode is declined: the sibling decode
kernel in the same wheel is not bit-reproducible past a work-unit budget.

The vendor wants a flat ``[slot, kv_head, head_dim]`` buffer plus a
per-token index list. Converting vLLM's block table is one small kernel per
step (shared by all layers). Slot stride is read from
``key_cache.stride(0)`` so hybrid interleaved K/V slabs stay correct.

Set ``VLLM_FL_SUNRISE_BENCH_BASELINE_PREFILL_ATTN=1`` to keep FlagGems.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BASELINE_ENV = "VLLM_FL_SUNRISE_BENCH_BASELINE_PREFILL_ATTN"

_UNAVAILABLE: Optional[str] = None
_DECISION_LOGGED = False

# Rows per program when expanding the block table. The kernel is bandwidth
# trivial either way; this only bounds the grid.
_EXPAND_TILE = 256


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _disabled() -> bool:
    return _env_on(_BASELINE_ENV)


def available() -> bool:
    """Whether this path could run at all, without importing the wheel."""
    if _disabled():
        return False
    return importlib.util.find_spec("llm_kernel_ptpu") is not None


def disable(reason: str) -> None:
    """Stop offering this path for the rest of the process."""
    global _UNAVAILABLE
    if _UNAVAILABLE is None:
        _UNAVAILABLE = reason
        logger.warning("prefill-attn: disabled (%s)", reason)


def _lib():
    """The vendor wheel, or None with the reason recorded once."""
    global _UNAVAILABLE
    if _UNAVAILABLE is not None:
        return None
    try:
        import llm_kernel_ptpu

        return llm_kernel_ptpu
    except Exception as exc:  # noqa: BLE001
        _UNAVAILABLE = str(exc)
        logger.info(
            "prefill-attn: llm_kernel_ptpu unavailable (%s); prefill keeps the "
            "FlagGems varlen kernel.",
            exc,
        )
        return None


@triton.jit
def _expand_block_table_kernel(
    table_ptr,
    cu_kv_ptr,
    seq_lens_ptr,
    out_ptr,
    table_stride,
    slots_per_block,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    """Flatten a block table into the per-token slot list the vendor reads.

    ``out[cu_kv[b] + j] = table[b, j // BLOCK_SIZE] * slots_per_block
                          + j % BLOCK_SIZE``

    for every request ``b`` and every one of its ``seq_lens[b]`` cached tokens.
    ``slots_per_block`` is a runtime value because it is derived from the
    cache's own stride and differs between the dense and hybrid layouts.
    """
    req = tl.program_id(0)
    length = tl.load(seq_lens_ptr + req)
    offs = tl.program_id(1) * TILE + tl.arange(0, TILE)
    mask = offs < length
    block = tl.load(
        table_ptr + req * table_stride + offs // BLOCK_SIZE,
        mask=mask,
        other=0,
    )
    slot = block * slots_per_block + offs % BLOCK_SIZE
    tl.store(out_ptr + tl.load(cu_kv_ptr + req) + offs, slot, mask=mask)


def flat_cache_view(cache: torch.Tensor) -> Optional[torch.Tensor]:
    """``[block, block_size, kv_head, dim]`` seen as ``[slot, kv_head, dim]``.

    A plain ``reshape`` is only a view for the dense layout. For the hybrid
    interleaved slab the blocks are not adjacent, so the flat view is built from
    the strides instead, keeping this tensor's own storage offset -- which is
    what distinguishes the K half from the V half of the same slab.

    Returns None if the trailing dimensions are not packed, in which case there
    is no such view and the caller should decline.
    """
    if cache.dim() != 4:
        return None
    num_blocks, block_size, heads, dim = cache.shape
    if cache.stride(3) != 1 or cache.stride(2) != dim:
        return None
    slot_stride = cache.stride(1)
    if slot_stride != heads * dim:
        return None
    block_stride = cache.stride(0)
    if block_stride % slot_stride:
        return None
    slots_per_block = block_stride // slot_stride
    if slots_per_block < block_size:
        return None
    # Size to the last addressable slot rather than to
    # num_blocks * slots_per_block, which for the value half of an interleaved
    # slab would run past the end of the storage.
    slots = (num_blocks - 1) * slots_per_block + block_size
    return cache.as_strided((slots, heads, dim), (slot_stride, dim, 1))


def slots_per_block(cache: torch.Tensor) -> int:
    """How far apart consecutive blocks are, in slots."""
    return cache.stride(0) // cache.stride(1)


def build_kv_indices(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    total_kv: int,
    max_seq_len: int,
    block_size: int,
    cache_slots_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-token slot list and its per-request cumsum.

    ``total_kv`` comes from the host, so nothing here has to read a device
    tensor to size an allocation. Both outputs are int32, which is what the
    vendor kernel takes.
    """
    device = block_table.device
    lens = seq_lens[:num_reqs]

    cu_kv = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    torch.cumsum(lens, 0, out=cu_kv[1:])

    kv_indices = torch.empty(total_kv, dtype=torch.int32, device=device)
    grid = (num_reqs, triton.cdiv(max_seq_len, _EXPAND_TILE))
    _expand_block_table_kernel[grid](
        block_table,
        cu_kv,
        lens,
        kv_indices,
        block_table.stride(0),
        cache_slots_per_block,
        BLOCK_SIZE=block_size,
        TILE=_EXPAND_TILE,
    )
    return kv_indices, cu_kv


def forward(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    qo_ind: torch.Tensor,
    kv_ind: torch.Tensor,
    kv_indices: torch.Tensor,
    max_query_len: int,
    scale: float,
) -> bool:
    """Run the vendor prefill kernel, or return False having done nothing."""
    global _DECISION_LOGGED
    lib = _lib()
    if lib is None:
        return False

    k_flat = flat_cache_view(key_cache)
    v_flat = flat_cache_view(value_cache)
    if k_flat is None or v_flat is None:
        disable("kv cache has no flat slot view")
        return False

    try:
        lib.paged_prefill_attention_fwd(
            query,
            k_flat,
            v_flat,
            output,
            qo_ind,
            kv_ind,
            kv_indices,
            None,
            None,
            max_query_len,
            scale,
            0.0,
            True,
        )
    except Exception:  # noqa: BLE001
        # The FlagGems kernel can always serve the same call, and the output
        # buffer has not been consumed yet, so falling back is safe -- but a
        # kernel that failed once is not worth re-entering every layer.
        logger.warning(
            "prefill-attn: vendor kernel raised; prefill returns to the "
            "FlagGems varlen kernel for the rest of the run",
            exc_info=True,
        )
        disable("raised during forward")
        return False

    if not _DECISION_LOGGED:
        _DECISION_LOGGED = True
        logger.info(
            "prefill-attn: prefill now runs llm_kernel_ptpu's paged kernel; "
            "decode and cascade keep FlagGems. Disable with %s=1.",
            _BASELINE_ENV,
        )
    return True
