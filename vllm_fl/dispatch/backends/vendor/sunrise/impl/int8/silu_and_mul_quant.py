# Copyright (c) 2026 BAAI. All rights reserved.

"""SiLU-and-mul fused with per-token INT8 activation quantization.

Routed experts quantize gemm1's activated output before gemm2; nothing reads
the BF16 in between. Fusing both in one kernel removes that round trip and
one launch. The activation stays in fp32 into the quantizer, so this is
slightly more accurate than the unfused pair.

``is_profitable`` is the shape boundary. Rows per program matter more than
the fusion itself: a program per row at MoE widths is bandwidth-starved, so
``_silu_and_mul_quant_rows_kernel`` tiles several rows. The tiled kernel
only takes widths a power-of-two tile covers exactly; see
``rows_per_program`` for the store-aliasing that decides it.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_INT8_MIN = -128
_INT8_MAX = 127
# Symmetric int8 uses 127 as the scale denominator so +/-amax maps to +/-127.
_SCALE_DENOM = 127.0
# Widest row a program handles in one shot before falling back to two passes.
_MAX_BLOCK = 8192
# Rows this narrow win at every batch size.
_ALWAYS_PROFITABLE_N = 512
# Wider rows only win while the output stays under this; past it the fused
# kernel is HBM-bound and one program per row streams worse than the
# vendor's silu, which the extra quant launch no longer pays for.
_MAX_PROFITABLE_ELEMS = 16 << 20

# Rows one program takes, by output width: (widest N, rows). The narrow
# widths are the MoE ones -- N is the per-rank expert width -- and they
# are where a program per row leaves most of the bandwidth unused.
_ROWS_PER_PROGRAM = ((64, 16), (128, 8), (512, 4))
# Tiles a batch must be worth before tiling beats a program per row.
_MIN_TILED_ROWS = 4


@triton.jit
def _silu_and_mul_quant_single_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    wscale_ptr,
    expert_ptr,
    N,
    x_row_stride,
    q_row_stride,
    eps,
    scale_denom,
    int8_min,
    int8_max,
    BLOCK: tl.constexpr,
    HAS_WSCALE: tl.constexpr,
    SFK: tl.constexpr,
    BLOCK_SFK: tl.constexpr,
):
    """Single-load path for N <= BLOCK: gate/up, amax and quantize in registers."""
    row = tl.program_id(0)
    x_row = x_ptr + row * x_row_stride

    cols = tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row + N + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_WSCALE:
        base = wscale_ptr + tl.load(expert_ptr + row) * 2 * N
        gate = gate * tl.load(base + cols, mask=mask, other=0.0)
        up = up * tl.load(base + N + cols, mask=mask, other=0.0)
    y = (gate * tl.sigmoid(gate)) * up

    scale = tl.maximum(tl.max(tl.abs(y)), eps) / scale_denom
    if SFK == 1:
        tl.store(s_ptr + row, scale)
    else:
        soffs = tl.arange(0, BLOCK_SFK)
        tl.store(s_ptr + row * SFK + soffs, scale, mask=soffs < SFK)

    inv_scale = 1.0 / scale
    x = y * inv_scale
    r = tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))
    r = tl.minimum(tl.maximum(r, int8_min), int8_max)
    tl.store(
        q_ptr + row * q_row_stride + cols,
        r.to(q_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _silu_and_mul_quant_rows_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    wscale_ptr,
    expert_ptr,
    N,
    rows,
    x_row_stride,
    q_row_stride,
    eps,
    scale_denom,
    int8_min,
    int8_max,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_WSCALE: tl.constexpr,
    SFK: tl.constexpr,
    BLOCK_SFK: tl.constexpr,
):
    """Several rows per program: the per-token amax is a reduction along axis 1.

    One row per program leaves the narrow MoE widths moving a few hundred bytes
    per program, which is why the single-row kernels above run at a few percent
    of peak there. Nothing about the operation needs a program per row.
    """
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    row_mask = r < rows
    mask = row_mask[:, None] & (cols[None, :] < N)

    base = x_ptr + r[:, None] * x_row_stride + cols[None, :]
    gate = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(base + N, mask=mask, other=0.0).to(tl.float32)
    if HAS_WSCALE:
        wbase = (
            wscale_ptr
            + tl.load(expert_ptr + r, mask=row_mask, other=0)[:, None] * 2 * N
            + cols[None, :]
        )
        gate = gate * tl.load(wbase, mask=mask, other=0.0)
        up = up * tl.load(wbase + N, mask=mask, other=0.0)
    y = (gate * tl.sigmoid(gate)) * up

    scale = tl.maximum(tl.max(tl.abs(y), axis=1), eps) / scale_denom
    if SFK == 1:
        tl.store(s_ptr + r, scale, mask=row_mask)
    else:
        soffs = tl.arange(0, BLOCK_SFK)
        tl.store(
            s_ptr + r[:, None] * SFK + soffs[None, :],
            scale[:, None],
            mask=row_mask[:, None] & (soffs[None, :] < SFK),
        )

    x = y * (1.0 / scale)[:, None]
    q = tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))
    q = tl.minimum(tl.maximum(q, int8_min), int8_max)
    tl.store(
        q_ptr + r[:, None] * q_row_stride + cols[None, :],
        q.to(q_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _silu_and_mul_quant_twopass_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    N,
    x_row_stride,
    q_row_stride,
    eps,
    scale_denom,
    int8_min,
    int8_max,
    BLOCK: tl.constexpr,
):
    """Two-pass path for N > BLOCK; the activation is recomputed, not re-read."""
    row = tl.program_id(0)
    x_row = x_ptr + row * x_row_stride

    amax = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        gate = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_row + N + cols, mask=mask, other=0.0).to(tl.float32)
        y = (gate * tl.sigmoid(gate)) * up
        amax = tl.maximum(amax, tl.max(tl.abs(y)))

    scale = tl.maximum(amax, eps) / scale_denom
    tl.store(s_ptr + row, scale)

    inv_scale = 1.0 / scale
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        gate = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_row + N + cols, mask=mask, other=0.0).to(tl.float32)
        y = (gate * tl.sigmoid(gate)) * up
        x = y * inv_scale
        r = tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))
        r = tl.minimum(tl.maximum(r, int8_min), int8_max)
        tl.store(
            q_ptr + row * q_row_stride + cols,
            r.to(q_ptr.dtype.element_ty),
            mask=mask,
        )


def is_fusable(x: torch.Tensor) -> bool:
    """Whether ``x`` is a gated activation input this kernel can consume.

    Rows must be contiguous because the kernel reaches the ``up`` half at a
    fixed ``+N`` element offset from the ``gate`` half.
    """
    return (
        x.dim() == 2
        and x.shape[-1] % 2 == 0
        and x.shape[-1] > 0
        and x.stride(-1) == 1
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    )


def is_profitable(x: torch.Tensor) -> bool:
    """Whether fusing beats the vendor silu plus a separate quant launch.

    Rows wider than ``_MAX_BLOCK`` need the two-pass kernel, which reads the
    gate and up halves twice; the unfused pair reads the narrow bf16 buffer
    for its second pass instead and wins those shapes. Everything else is a
    win except very large outputs at moderate widths.
    """
    rows, two_n = x.shape
    n = two_n // 2
    if n > _MAX_BLOCK:
        return False
    return n <= _ALWAYS_PROFITABLE_N or rows * n <= _MAX_PROFITABLE_ELEMS


def rows_per_program(n: int) -> int:
    """How many rows one program takes; 1 means keep the single-row kernels.

    The single-row kernels are not a special case of the tiled one -- a
    (1, BLOCK) tile compiles worse than a flat one -- so a width whose
    best count is 1 stays on them rather than going through the tiled
    kernel with ROWS=1.

    Widths that a power-of-two tile does not cover exactly are refused.
    The tiled store addresses ``row * N + col`` for col up to the tile
    width, so when the tile overhangs the row those lanes address the
    *next* row, which another program owns. They are masked off, but the
    mask is not always honoured. Covering the row exactly removes the
    column mask, and with it the aliasing.
    """
    if n != triton.next_power_of_2(n):
        return 1
    for width, count in _ROWS_PER_PROGRAM:
        if n <= width:
            return count
    return 1


def _triton(x, eps, wscale=None, expert_of_row=None, sfk=1, out=None, scale_out=None):
    rows, two_n = x.shape
    n = two_n // 2
    q = torch.empty((rows, n), device=x.device, dtype=torch.int8) if out is None else out
    s = (
        torch.empty((rows, sfk), device=x.device, dtype=torch.float32)
        if scale_out is None
        else scale_out
    )
    block = min(triton.next_power_of_2(n), _MAX_BLOCK)
    has_ws = wscale is not None
    block_sfk = triton.next_power_of_2(sfk)

    per_program = rows_per_program(n) if n <= block else 1
    # Too few rows to fill a few tiles; the tiled kernel only starts
    # paying once the batch is large enough.
    if rows < _MIN_TILED_ROWS * per_program:
        per_program = 1
    if per_program > 1:
        _silu_and_mul_quant_rows_kernel[(triton.cdiv(rows, per_program),)](
            x,
            q,
            s,
            wscale,
            expert_of_row,
            n,
            rows,
            x.stride(0),
            q.stride(0),
            eps,
            _SCALE_DENOM,
            _INT8_MIN,
            _INT8_MAX,
            ROWS=per_program,
            BLOCK=block,
            HAS_WSCALE=has_ws,
            SFK=sfk,
            BLOCK_SFK=block_sfk,
            num_warps=min(max(per_program * block // 256, 1), 8),
        )
        return q, s

    if n > block:
        # The two-pass kernel carries neither extra, and cannot be reached from
        # the routed experts: it wants N > 8192 and the per-rank expert width is
        # 512 at TP=1. Refusing beats silently dropping the weight scale.
        if has_ws or sfk != 1:
            raise ValueError(
                f"fused weight scale is not implemented for N={n} > {block}"
            )
        _silu_and_mul_quant_twopass_kernel[(rows,)](
            x, q, s, n, x.stride(0), q.stride(0), eps, _SCALE_DENOM,
            _INT8_MIN, _INT8_MAX, BLOCK=block,
            num_warps=min(max(block // 256, 1), 8),
        )
        return q, s

    _silu_and_mul_quant_single_kernel[(rows,)](
        x,
        q,
        s,
        wscale,
        expert_of_row,
        n,
        x.stride(0),
        q.stride(0),
        eps,
        _SCALE_DENOM,
        _INT8_MIN,
        _INT8_MAX,
        BLOCK=block,
        HAS_WSCALE=has_ws,
        SFK=sfk,
        BLOCK_SFK=block_sfk,
        num_warps=min(max(block // 256, 1), 8),
    )
    return q, s


def _torch(x, eps, wscale=None, expert_of_row=None, sfk=1, out=None, scale_out=None):
    n = x.shape[-1] // 2
    gate = x[:, :n].to(torch.float32)
    up = x[:, n:].to(torch.float32)
    if wscale is not None:
        # [rows, 2N] gathered by each row's expert, same as the kernel does.
        ws = wscale.to(torch.float32)[expert_of_row.long()]
        gate = gate * ws[:, :n]
        up = up * ws[:, n:]
    y = torch.nn.functional.silu(gate) * up
    scale = y.abs().amax(dim=-1, keepdim=True).clamp(min=eps) / _SCALE_DENOM
    q = (y / scale).round().clamp(_INT8_MIN, _INT8_MAX).to(torch.int8)
    if sfk != 1:
        scale = scale.expand(-1, sfk).contiguous()
    if out is not None:
        out.copy_(q)
        q = out
    if scale_out is not None:
        scale_out.copy_(scale)
        scale = scale_out
    return q, scale


def silu_and_mul_quant_int8(
    x: torch.Tensor,
    eps: float = 1e-10,
    wscale: torch.Tensor | None = None,
    expert_of_row: torch.Tensor | None = None,
    sfk: int = 1,
    out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``per_token_quant_int8(silu(x[:, :N]) * x[:, N:])`` in one pass.

    Returns ``(int8 [rows, N], fp32 scale [rows, sfk])``, matching what
    ``moe_kernel_quantize_input`` hands the second expert GEMM.

    Two extras exist for the routed experts, both defaulting off so the
    plain call is unchanged. Each removes a launch:

    ``wscale`` / ``expert_of_row``
        The first expert GEMM's per-channel weight scale, ``[E, 2N]``,
        applied to ``gate`` and ``up`` as they are loaded. It used to be a
        pass of its own that read and rewrote the whole bf16 input.
        Folding it in also drops a rounding: the scaled value no longer
        round-trips through bfloat16.

    ``sfk``
        Writes the per-token scale replicated across that many k-blocks,
        which is the layout the vendor blockwise GEMM wants, instead of
        leaving a second kernel to spread ``[rows, 1]`` into
        ``[rows, sfk]``.

    ``out`` / ``scale_out``
        Write into buffers the caller owns instead of allocating. The routed
        experts pass slices of a persistent pool so that nothing is allocated
        inside a captured decode graph; see ``impl/int8/moe_scratch.py``.
    """
    assert is_fusable(x), f"not a fusable gated activation input: {x.shape}"
    try:
        q, s = _triton(x, eps, wscale, expert_of_row, sfk, out, scale_out)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "silu_and_mul_quant_int8: Triton path failed (%s); "
            "falling back to torch.",
            e,
        )
        q, s = _torch(x, eps, wscale, expert_of_row, sfk, out, scale_out)
    return q, s
