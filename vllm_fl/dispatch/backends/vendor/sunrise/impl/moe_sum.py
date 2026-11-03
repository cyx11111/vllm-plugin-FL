# Copyright (c) 2026 BAAI. All rights reserved.

"""``moe_sum`` for Sunrise/PTPU: FlagGems' kernel with a chosen tile.

The body is FlagGems' -- one program per (token, hidden block), accumulating
the top-k axis in fp32. Unrolling ``topk`` or tiling several tokens per
program does not help; the tile does.

FlagGems autotunes ``BLOCK_SIZE`` on ``["hidden_size", "topk"]`` and omits
token count, so whichever batch first hits a (hidden, topk) pins the config
for the rest of the process. This path picks a static tile instead so
arrival order does not decide decode vs prefill cost. The default covers
most shapes; covering a hidden row in one unmasked block only pays off
once the row is wide enough and the batch is deep enough.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The flat default: winner or close to it nearly everywhere.
DEFAULT_BLOCK = 1024

# Covering the hidden row in one unmasked block, which only pays off once the row
# is at least four default blocks wide and the batch is deep enough to matter.
WHOLE_ROW_BLOCK = 4096
WHOLE_ROW_MIN_HIDDEN = 4096
WHOLE_ROW_MIN_TOKENS = 1024

NUM_WARPS = 8

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def _moe_sum_kernel(
    inp_ptr,
    out_ptr,
    hidden,
    topk,
    stride_inp_token,
    stride_inp_topk,
    stride_out_token,
    BLOCK: tl.constexpr,
):
    """Sum one token's top-k rows into one output row, fp32 accumulation."""
    token = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < hidden

    base = inp_ptr + token * stride_inp_token + offs
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(topk):
        acc += tl.load(base + k * stride_inp_topk, mask=mask, other=0.0).to(tl.float32)

    tl.store(
        out_ptr + token * stride_out_token + offs,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def block_size(num_tokens: int, hidden: int) -> int:
    """``BLOCK_SIZE`` for a reduction of ``num_tokens`` rows of ``hidden``."""
    if (
        num_tokens >= WHOLE_ROW_MIN_TOKENS
        and hidden >= WHOLE_ROW_MIN_HIDDEN
        and hidden & (hidden - 1) == 0
    ):
        return min(hidden, WHOLE_ROW_BLOCK)
    return DEFAULT_BLOCK


def is_supported(inp: torch.Tensor, out: torch.Tensor) -> bool:
    """Whether the kernel can take this pair, as opposed to the torch fallback.

    vLLM only ever passes the 3-D/2-D case with a contiguous hidden axis; the
    wider ``[tokens, topk, ...]`` contract is left to the fallback rather than
    given an untested path here.
    """
    return (
        inp.dim() == 3
        and out.dim() == 2
        and inp.stride(2) == 1
        and out.stride(1) == 1
        and inp.dtype == out.dtype
        and inp.dtype in _SUPPORTED_DTYPES
        and inp.shape[0] > 0
        and inp.shape[1] > 0
        and inp.shape[2] > 0
    )


def moe_sum_triton(inp: torch.Tensor, out: torch.Tensor) -> None:
    """Reduce ``inp`` ``[tokens, topk, hidden]`` into ``out`` ``[tokens, hidden]``."""
    num_tokens, topk, hidden = inp.shape
    block = block_size(num_tokens, hidden)
    _moe_sum_kernel[(num_tokens, triton.cdiv(hidden, block))](
        inp,
        out,
        hidden,
        topk,
        inp.stride(0),
        inp.stride(1),
        out.stride(0),
        BLOCK=block,
        num_warps=NUM_WARPS,
    )


__all__ = [
    "DEFAULT_BLOCK",
    "WHOLE_ROW_BLOCK",
    "WHOLE_ROW_MIN_HIDDEN",
    "WHOLE_ROW_MIN_TOKENS",
    "block_size",
    "is_supported",
    "moe_sum_triton",
]
