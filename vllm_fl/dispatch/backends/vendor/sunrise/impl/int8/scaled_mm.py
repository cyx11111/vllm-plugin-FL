# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU INT8 scaled GEMM for the vLLM-native W8A8 linear path.

Drop-in replacement for ``flag_gems.ops.scaled_mm.scaled_mm`` on sunrise.

FlagGems' generic kernel cannot stay on the PTPU INT8 MMA path: it uses
int64 offsets and masks both operand loads inside the K loop, so its
autotuner stays on a shallow ``BLOCK_K``. This kernel uses 32-bit offsets,
clamped (not masked) K-loop loads, an ``EVEN_K`` specialisation, and an
accumulator threaded through ``tl.dot``.

Tile selection is static (see ``_select_config``) because these linears run
inside captured decode graphs. It is keyed on M, N and K: TP moves gate_up's
N and down's K in opposite directions, so a tile for one TP does not
transfer to another. Three regimes: decode (one M-tile, N sets program
count), a middle band that cuts N finer and spends the budget on deeper
``BLOCK_K``, and large batches where wide tiles win on their own.

``sd_kernel.int8_scaled_mm`` is complementary: decode and narrow shapes stay
here; only the consistently profitable large-prefill region is sent to SD.
``VLLM_FL_SUNRISE_BENCH_BASELINE_SD_INT8_SCALED_MM=1`` keeps this kernel.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# torch_ptpu exposes a tablas-backed per-channel W8A8 GEMM through
# ``sd_kernel.int8_scaled_mm``. It has the same quantization contract as
# this kernel, but not the same performance curve: it loses on decode and
# narrow shared-expert GEMMs, then wins once a large prefill supplies
# enough rows. Keep the dispatch inside that profitable region rather
# than replacing Triton wholesale.
_SD_MIN_M = 256
_SD_MIN_N = 4096
_SD_MIN_K = 2048
_SD_BASELINE_ENV = "VLLM_FL_SUNRISE_BENCH_BASELINE_SD_INT8_SCALED_MM"
_SD_MM_FN = None
_SD_UNAVAILABLE = False
_SD_DECISION_LOGGED = False

# tl.constexpr so the kernel can read them as globals; host code uses ``.value``.
SCALAR_SCALE = tl.constexpr(0)
VECTOR_SCALE = tl.constexpr(1)


@triton.jit
def _scaled_mm_kernel(
    A,
    B,
    ScaleA,
    ScaleB,
    Bias,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    SCALE_A_MODE: tl.constexpr,
    SCALE_B_MODE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Clamp instead of mask: every load stays in bounds and the K loop needs no
    # predication. Rows/columns past the edge accumulate junk that the store
    # mask below drops.
    offs_am = tl.minimum(offs_m, M - 1)
    offs_bn = tl.minimum(offs_n, N - 1)

    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out = acc.to(tl.float32)

    if SCALE_A_MODE == SCALAR_SCALE:
        scale_a = tl.full((BLOCK_M,), tl.load(ScaleA), dtype=tl.float32)
    else:
        scale_a = tl.load(ScaleA + offs_am)
    if SCALE_B_MODE == SCALAR_SCALE:
        scale_b = tl.full((BLOCK_N,), tl.load(ScaleB), dtype=tl.float32)
    else:
        scale_b = tl.load(ScaleB + offs_bn)

    out = out * scale_a[:, None] * scale_b[None, :]

    if HAS_BIAS:
        bias = tl.load(Bias + offs_bn).to(tl.float32)
        out += bias[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out.to(C.dtype.element_ty), mask=c_mask)


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for the batched regime,
# used once the launch has enough programs to fill the device. BLOCK_K=128
# is the smallest tile the PTPU INT8 MMA runs at full rate; a 256-deep tile
# busts the resource budget at these widths.
_CONFIG_MID_M = (64, 64, 128, 8, 2)
_CONFIG_LARGE_M = (64, 128, 128, 8, 2)

# The 128-wide tile only pays off when the GEMM is large on every axis.
# On skinny shapes it halves the program count without enough K work to
# hide the wider epilogue.
_LARGE_M = 512
_LARGE_N = 5120
_LARGE_K = 512

# Between decode and a device-filling batch the tiles above launch too few
# programs. Cut N finer, spend the freed budget on a deeper BLOCK_K, and
# let BLOCK_M grow only as far as the program count allows. Past this N
# there are already enough 64-wide N-tiles; cutting further wastes MMA width.
_BATCH_BLOCK_K = 256
_BATCH_NARROW_N = 5120
_BATCH_BLOCK_N_NARROW = 32
_BATCH_BLOCK_N_WIDE = 64

# Known bad band, currently masked rather than fixed: at N=5120 with a
# shallow K the narrow rule picks a tile that collapses around a typical
# graph-capture M. Deep K does not show it, and ``dense_gemm`` now takes
# the shapes that would reach the notch. Remember before widening
# ``dense_gemm``'s refusals.

# A short K wants few programs, each reading a wide slice of B. One or two
# K steps amortise nothing, so a tile that is narrow in N and deep in K
# pays twice. Below this K the rule goes wide and shallow instead.
_BATCH_DEEP_K_MIN_K = 1024

# Cutting N finer re-reads A once per N-tile. Affordable while M is small;
# ruinous once it is not. Hand over to the buckets above _BATCH_MAX_M
# except where they would be starved outright.
_BATCH_MAX_M = 128
_BATCH_STARVED = 32

# Grow BLOCK_M while the launch still clears this many programs. Below it
# the device runs dry.
_BATCH_MIN_PROGRAMS = 64
_BATCH_BLOCK_MS = (64, 32, 16)

# Past this the tile spills to local memory and the launch collapses.
# Caps BLOCK_K so the wide-N end of the rule falls back to _CONFIG_MID_M.
# _CONFIG_LARGE_M is twice as big and is only used at an M large enough
# to amortise the spill.
_BATCH_MAX_TILE = 1 << 19

# Programs _CONFIG_MID_M would launch. At or above this it fills the
# device and wins on its own; below, it is the narrow rule's band.
_BATCH_ABUNDANT = 128

# At decode M is below any useful BLOCK_M, so N alone decides program
# count. Cutting N at 64 starves small shapes; cut finer and spend the
# budget on a deeper BLOCK_K, but only while the program count stays
# inside a few hundred -- past that the narrow tile wastes MMA width.
# num_warps=4 is only safe at BLOCK_N=16 (a 256-deep tile otherwise
# spills to local memory).
_DECODE_M = 16

# A short K is the one case where cutting N finer backfires: there is no
# K loop to amortise the extra programs over, so the launch wants few and
# wide.
_DECODE_SHORT_K = 128
_DECODE_TILE_SHORT_K = (16, 64, 128, 8, 2)

# Otherwise cut N as fine as the program count tolerates.
_DECODE_TILES = (
    # (inclusive N bound, tile)
    (1024, (16, 16, 256, 4, 2)),
    (5120, (16, 32, 256, 8, 2)),
)
_DECODE_TILE_WIDE_N = (16, 64, 256, 8, 2)


# The 16-wide tile keeps winning above the decode bucket only while the
# launch stays inside a program budget: BLOCK_M=16 multiplies the program
# count as M grows, and past that the narrow tile is pure overhead. Only
# the narrowest bucket gets this treatment.
_DECODE_MAX_PROGRAMS = 128
_EXTEND_MAX_BLOCK_N = 16


def _decode_config(N: int, K: int):
    """Tile for a single-M-tile launch, keyed on the axes that supply them."""
    if K <= _DECODE_SHORT_K:
        return _DECODE_TILE_SHORT_K
    for bound, cfg in _DECODE_TILES:
        if N <= bound:
            return cfg
    return _DECODE_TILE_WIDE_N


def _decode_tile_extends(cfg, M: int, N: int) -> bool:
    """Is the decode tile still the right call above the decode bucket?"""
    block_m, block_n = cfg[0], cfg[1]
    if block_n > _EXTEND_MAX_BLOCK_N:
        return False
    programs = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    return programs <= _DECODE_MAX_PROGRAMS


def _mid_m_programs(M: int, N: int) -> int:
    """Programs ``_CONFIG_MID_M`` would launch on this shape."""
    block_m, block_n = _CONFIG_MID_M[0], _CONFIG_MID_M[1]
    return triton.cdiv(M, block_m) * triton.cdiv(N, block_n)


def _narrow_config(M: int, N: int, K: int):
    """Tile for the band between decode and a launch that fills the device."""
    if K <= _DECODE_SHORT_K:
        # Short K wants the same few-and-wide launch it wants at decode, and for
        # the same reason: there is no K loop to amortise more programs over.
        return _DECODE_TILE_SHORT_K

    narrow_n = N <= _BATCH_NARROW_N and K >= _BATCH_DEEP_K_MIN_K
    block_n = _BATCH_BLOCK_N_NARROW if narrow_n else _BATCH_BLOCK_N_WIDE
    n_tiles = triton.cdiv(N, block_n)

    # Largest BLOCK_M that still clears the program floor; the list is descending
    # so the first hit is the largest, and the last entry is the fallback for a
    # shape too small to clear it at any height.
    block_m = _BATCH_BLOCK_MS[-1]
    for candidate in _BATCH_BLOCK_MS:
        if candidate > max(triton.next_power_of_2(M), 16):
            continue
        if triton.cdiv(M, candidate) * n_tiles >= _BATCH_MIN_PROGRAMS:
            block_m = candidate
            break

    block_k = _BATCH_BLOCK_K if K >= _BATCH_DEEP_K_MIN_K else 128
    while block_k > 64 and block_m * block_n * block_k > _BATCH_MAX_TILE:
        block_k //= 2
    return block_m, block_n, block_k, 8, 2


def _select_config(M: int, N: int, K: int):
    """Static tile choice; no runtime benchmarking (decode runs under graphs)."""
    decode = _decode_config(N, K)
    mid_programs = _mid_m_programs(M, N)
    if M <= _DECODE_M or _decode_tile_extends(decode, M, N):
        cfg = decode
    elif mid_programs < _BATCH_ABUNDANT and (
        M <= _BATCH_MAX_M or mid_programs < _BATCH_STARVED
    ):
        cfg = _narrow_config(M, N, K)
    elif M > _LARGE_M and N >= _LARGE_N and K >= _LARGE_K:
        cfg = _CONFIG_LARGE_M
    else:
        cfg = _CONFIG_MID_M

    block_m, block_n, block_k, num_warps, num_stages = cfg
    # A tile larger than the operand only wastes MMA slots. BLOCK_M matters as
    # much as the other two: _CONFIG_MID_M is 64 rows tall, and on a 32-row batch
    # half of every program is padding.
    block_m = min(block_m, max(triton.next_power_of_2(M), 16))
    block_n = min(block_n, max(triton.next_power_of_2(N), 16))
    block_k = min(block_k, max(triton.next_power_of_2(K), 16))
    return block_m, block_n, block_k, num_warps, num_stages


def _normalize_scale(scale: torch.Tensor, expected: int, *, is_left: bool):
    """Mirror ``flag_gems.ops.scaled_mm._normalize_scale``."""
    if scale.numel() == 1:
        return scale.reshape(1).contiguous(), SCALAR_SCALE.value

    ok = scale.ndim == 1 and scale.shape[0] == expected
    if is_left:
        ok = ok or (scale.ndim == 2 and tuple(scale.shape) == (expected, 1))
    else:
        ok = ok or (scale.ndim == 2 and tuple(scale.shape) == (1, expected))
    if ok:
        return scale.reshape(expected).contiguous(), VECTOR_SCALE.value

    name = "scale_a" if is_left else "scale_b"
    want = f"({expected}, 1)" if is_left else f"(1, {expected})"
    raise RuntimeError(f"{name} must be a scalar tensor or have shape {want}")


def _use_sd_scaled_mm(M: int, N: int, K: int) -> bool:
    """Whether the tablas-backed GEMM is profitable in this region."""
    baseline = os.environ.get(_SD_BASELINE_ENV, "").strip().lower()
    if baseline in {"1", "true", "yes", "on"}:
        return False
    return M >= _SD_MIN_M and N >= _SD_MIN_N and K >= _SD_MIN_K


def _resolve_sd_scaled_mm():
    """Resolve the optional torch_ptpu entry point once."""
    global _SD_MM_FN, _SD_UNAVAILABLE
    if _SD_MM_FN is not None:
        return _SD_MM_FN
    if _SD_UNAVAILABLE:
        return None
    try:
        from torch_ptpu.sd_kernel import int8_scaled_mm as sd_scaled_mm
    except (ImportError, AttributeError) as exc:
        _SD_UNAVAILABLE = True
        logger.info(
            "Sunrise SD INT8 scaled_mm is unavailable (%s); keeping Triton.",
            exc,
        )
        return None
    _SD_MM_FN = sd_scaled_mm
    return _SD_MM_FN


def _try_sd_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Run the native per-channel GEMM when no layout conversion is needed."""
    global _SD_DECISION_LOGGED
    M, K = a.shape
    N = b.shape[1]
    if not _use_sd_scaled_mm(M, N, K):
        return None
    if output_dtype not in (torch.bfloat16, torch.float16):
        return None
    # This kernel reaches tangMemGetInfo, which throws while a stream is being
    # captured, and the raise leaves the capture unusable rather than just
    # failing the call. Decode graphs are captured at batch sizes that land in
    # this region, so being recorded is not optional: skip it and let a
    # capturable kernel be the one in the graph.
    from ..capture import is_capturing

    if is_capturing():
        return None

    # vLLM hands us weight.t(): [K,N] with stride (1,K), so transposing it back
    # is normally a free contiguous [N,K] view. Do not turn an unusual layout
    # into a per-call weight copy.
    weight = b.t()
    if not a.is_contiguous() or not weight.is_contiguous():
        return None

    sd_scaled_mm = _resolve_sd_scaled_mm()
    if sd_scaled_mm is None:
        return None
    if not _SD_DECISION_LOGGED:
        _SD_DECISION_LOGGED = True
        logger.info(
            "Sunrise INT8 scaled_mm: using torch_ptpu.sd_kernel for the "
            "large-prefill region M>=%d N>=%d K>=%d; smaller shapes keep "
            "the Triton kernel. Disable with %s=1.",
            _SD_MIN_M,
            _SD_MIN_N,
            _SD_MIN_K,
            _SD_BASELINE_ENV,
        )
    return sd_scaled_mm(
        a, weight, scale_a, scale_b, dtype=output_dtype, bias=bias
    )


def scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    scale_result: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``(a @ b) * scale_a * scale_b + bias`` for INT8 operands.

    Signature matches ``flag_gems.ops.scaled_mm.scaled_mm`` so the two are
    interchangeable behind ``patch_int8_native._resolve_fg_scaled_mm``.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise RuntimeError("scaled_mm expects 2D operands")
    if a.shape[1] != b.shape[0]:
        raise RuntimeError(
            f"shapes cannot be multiplied ({a.shape[0]}x{a.shape[1]} and "
            f"{b.shape[0]}x{b.shape[1]})"
        )
    if a.dtype != b.dtype:
        raise RuntimeError(f"operand dtype mismatch: {a.dtype} vs {b.dtype}")
    if scale_result is not None:
        raise NotImplementedError("scaled_mm: scale_result is not supported")

    M, K = a.shape
    N = b.shape[1]

    output_dtype = out_dtype or (out.dtype if out is not None else a.dtype)
    caller_supplied_out = out is not None
    if out is not None and tuple(out.shape) != (M, N):
        raise RuntimeError("Incompatible output shape")

    if M == 0 or N == 0:
        if out is None:
            out = torch.empty((M, N), dtype=output_dtype, device=a.device)
        return out

    scale_a, scale_a_mode = _normalize_scale(scale_a, M, is_left=True)
    scale_b, scale_b_mode = _normalize_scale(scale_b, N, is_left=False)
    if bias is not None:
        if bias.numel() != N:
            raise RuntimeError(f"Bias must be size {N} but got {bias.numel()}")
        bias = bias.reshape(N).contiguous()

    # The native API allocates its result. An explicit ``out`` would
    # therefore add a full output copy. Dynamic W8A8 linears use the
    # return-value form; unusual callers keep the in-place Triton contract.
    # A single-row activation scale normalizes to SCALAR_SCALE because one
    # per-token scale and one per-tensor scale are the same number, but the
    # native paths still want it (M=1 is single-request decode).
    per_token = scale_a_mode == VECTOR_SCALE.value or M == 1
    if (
        not caller_supplied_out
        and per_token
        and scale_b_mode == VECTOR_SCALE.value
    ):
        # deepgemm first: unlike SD it can be graph-captured.
        from . import dense_gemm

        dense_out = dense_gemm.try_scaled_mm(
            a, b, scale_a, scale_b, bias, output_dtype
        )
        if dense_out is not None:
            return dense_out

        sd_out = _try_sd_scaled_mm(
            a, b, scale_a, scale_b, bias, output_dtype
        )
        if sd_out is not None:
            return sd_out

    if out is None:
        out = torch.empty((M, N), dtype=output_dtype, device=a.device)

    # One of the two axes has to be unit-stride for the operand loads to
    # vectorize; a fully strided operand is materialized instead.
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    block_m, block_n, block_k, num_warps, num_stages = _select_config(M, N, K)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)

    _scaled_mm_kernel[grid](
        a,
        b,
        scale_a,
        scale_b,
        bias,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        SCALE_A_MODE=scale_a_mode,
        SCALE_B_MODE=scale_b_mode,
        HAS_BIAS=bias is not None,
        EVEN_K=(K % block_k == 0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


__all__ = ["scaled_mm"]
