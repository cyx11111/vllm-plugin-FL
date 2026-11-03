# Copyright (c) 2026 BAAI. All rights reserved.

"""
Sunrise (PTPU) activation operator implementations.

Routes ``silu_and_mul`` to ``torch_ptpu.sgl_kernel.silu_and_mul``.

The op is memory-bound. FlagGems sits far below achievable bandwidth at
every size; the vendor kernel is bit-identical where it accepts the input.

The vendor kernel is narrower: 2-byte floats, contiguous input, half width
a multiple of four. Unsupported inputs are handled here rather than left
to raise, because ``OpManager.call`` records a failure against the
*operator* -- one unsupported call would drop this backend for every later
``silu_and_mul``. Shipped shapes already satisfy the multiple-of-four
constraint, so the fallback is a guard rail.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch_ptpu.sgl_kernel import silu_and_mul as _ptpu_silu_and_mul

# What the vendor kernel accepts. fp32 fails to dispatch; a non-contiguous
# input and a half width outside this multiple both raise. Rank and row
# count are unconstrained.
_VENDOR_DTYPES = (torch.bfloat16, torch.float16)
_VENDOR_HALF_WIDTH_MULTIPLE = 4


def _vendor_supports(x: torch.Tensor) -> bool:
    """Whether ``_ptpu_silu_and_mul`` can take ``x`` without raising."""
    if x.dtype not in _VENDOR_DTYPES:
        return False
    if not x.is_contiguous():
        return False
    return (x.shape[-1] // 2) % _VENDOR_HALF_WIDTH_MULTIPLE == 0


def silu_and_mul_sunrise(obj, x: torch.Tensor) -> torch.Tensor:
    """SiLU activation followed by element-wise multiplication on PTPU.

    Args:
        obj: The calling object (unused; kept for dispatch ABI parity with the
            FlagGems / reference implementations).
        x: Input tensor of shape ``[..., 2 * d]``. ``[..., :d]`` is the
            "gate" half and ``[..., d:]`` is the "up" half (gate-up
            projection convention).

    Returns:
        Output tensor of shape ``[..., d]``.
    """
    if not _vendor_supports(x):
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]
    return _ptpu_silu_and_mul(x)
