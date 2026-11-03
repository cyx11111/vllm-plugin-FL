# Copyright (c) 2026 BAAI. All rights reserved.

"""Dense INT8 projections on ``deepgemm_ptpu``'s non-grouped GEMM.

Shipped W8A8 checkpoints quantize MLP / expert linears, so those INT8 GEMMs
land here. Attention stays BF16.

The vendor kernel folds per-token activation and per-channel weight scales as
``sum_kb (sum_{k in kb} A*B) * AS[m, kb] * BS[nb, kb]``. The per-token scale
does not vary along K and can be applied in the epilogue instead of being
replicated across k-blocks (an aten expand is host-heavy with FlagGems). The
per-channel weight scale is finer than any ``n_block_size``, so ``BS`` is all
ones and that scale also rides in the epilogue.

``n_block_size`` is 128: 64 looks attractive on some shapes but the vendor
config table is not reliable for it. ``torch_ptpu.sd_kernel.int8_scaled_mm``
calls ``tangMemGetInfo`` and throws during graph capture, so it cannot serve
a captured decode step.

Set ``VLLM_FL_SUNRISE_BENCH_BASELINE_DENSE_INT8_GEMM=1`` to keep the Triton
kernel for an A/B baseline.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 128

# With n_block_size=128 a narrower N is a single n-block, leaving the kernel
# almost nothing to parallelize over.
_MIN_N = 256

# The vendor kernel has a large fixed cost; below this K there is not enough
# work to amortize it.
_MIN_K = 1024

# The autotune table sometimes answers a mid-size batch with a decode-shaped
# M tile and then collapses. Rejecting a small tile that has to cover many
# rows keeps that region on Triton, in the vendor's own terms rather than as
# a shape blacklist.
_SMALL_M_TILE = 16
_M_BLOCK_SLACK = 4

_M_PER_BLOCK = re.compile(r"m_per_block=(\d+)")

_BASELINE_ENV = "VLLM_FL_SUNRISE_BENCH_BASELINE_DENSE_INT8_GEMM"

_SUPPORT: dict[tuple[int, int, int], bool] = {}
_ONES: dict[tuple[int, int, torch.device], torch.Tensor] = {}

# One flat all-ones buffer per device, sliced to whatever [M, k_blocks] the
# call needs. Both of the GEMM's scale operands are constant now that the
# epilogue applies the real scales, so there is nothing to key a cache on
# beyond size, and nothing to refill per call.
_ONES_A: dict[torch.device, torch.Tensor] = {}
_UNAVAILABLE: Optional[str] = None
_DECISION_LOGGED = False


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _disabled() -> bool:
    return _env_on(_BASELINE_ENV)


def available() -> bool:
    """Whether this path could run at all, without importing the wheel.

    For the startup log line, so it answers before device init and cannot be
    what triggers it. Whether a given call is taken is decided by ``can_run``.
    """
    if _disabled():
        return False
    return importlib.util.find_spec("deepgemm_ptpu") is not None


def _lib():
    """The vendor wheel, or None with the reason recorded once."""
    global _UNAVAILABLE
    if _UNAVAILABLE is not None:
        return None
    try:
        import deepgemm_ptpu

        return deepgemm_ptpu
    except Exception as exc:  # noqa: BLE001
        _UNAVAILABLE = str(exc)
        logger.info(
            "dense-int8: vendor GEMM unavailable (%s); INT8 linears keep the "
            "Triton kernel.",
            exc,
        )
        return None


@triton.jit
def _epilogue_kernel(
    c_ptr,
    a_scale_ptr,
    scale_ptr,
    bias_ptr,
    n,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``out[m, :] = out[m, :] * scale_a[m] * weight_scale + bias`` in fp32.

    The weight scale is per output channel, finer than any ``n_block_size`` the
    GEMM accepts, so it cannot ride in ``BS``. Reading the [N] scale directly
    rather than a materialised [M, N] broadcast keeps this a single pass.

    The per-token scale *could* ride in ``BS``, but only after a kernel wrote
    it into an [M, k_blocks] buffer every call. Taking it here instead costs
    one scalar load per row and lets that buffer be a constant.
    """
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n
    value = tl.load(c_ptr + row * n + offs, mask=mask, other=0.0).to(tl.float32)
    value *= tl.load(a_scale_ptr + row).to(tl.float32)
    value *= tl.load(scale_ptr + offs, mask=mask, other=0.0)
    if HAS_BIAS:
        value += tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(c_ptr + row * n + offs, value.to(tl.bfloat16), mask=mask)


def _m_per_block(dg, m: int, n: int, k: int) -> Optional[int]:
    """The smallest M tile the vendor would use, across best and fallback.

    Both are consulted because the kernel's own logs say it does not always take
    the best config, so a sound best does not excuse a degenerate fallback.
    """
    widths = []
    for fn in (dg.get_best_config_log, dg.get_fallback_config_log):
        try:
            found = _M_PER_BLOCK.search(fn(m, n, k, _BLOCK, _BLOCK, 1))
        except Exception:  # noqa: BLE001
            return None
        if found is None:
            return None
        widths.append(int(found.group(1)))
    return min(widths) if widths else None


def _profitable(dg, m: int, n: int, k: int) -> bool:
    key = (m, n, k)
    cached = _SUPPORT.get(key)
    if cached is not None:
        return cached

    # The config queries below are host-side vendor calls of unknown appetite
    # for allocation, and allocating is what a capture forbids. A shape first
    # seen while recording is declined rather than probed; warmup runs the same
    # shapes eagerly beforehand, so in practice the answer is already cached.
    from ..capture import is_capturing

    if is_capturing():
        return False

    ok = False
    try:
        if dg.is_support_int8_blockwise_gemm(m, n, k, _BLOCK, _BLOCK, 1):
            tile = _m_per_block(dg, m, n, k)
            ok = tile is not None and not (
                tile <= _SMALL_M_TILE and m > _M_BLOCK_SLACK * tile
            )
    except Exception as exc:  # noqa: BLE001
        # is_support has been seen to raise "no valid fallback config found,
        # M=1" for arguments it accepted moments earlier. Declining is right and
        # it must not escape into a forward pass.
        logger.debug("dense-int8: declining M=%d N=%d K=%d (%s)", m, n, k, exc)
        ok = False
    _SUPPORT[key] = ok
    return ok


def _ones_a_for(m: int, kb: int, device: torch.device) -> torch.Tensor:
    """The all-ones per-token ``BS``, as an [m, kb] view of one flat buffer.

    The GEMM wants this as a real K-major [M, k_blocks] buffer and rejects a
    stride-0 expanded view, but with the per-token scale moved into the epilogue
    nothing in it varies, so one buffer serves every shape and every call: a
    prefix of it, viewed, which is contiguous with the strides the GEMM wants.
    Sizes round up to a power of two so a widening prefill chunk reallocates a
    handful of times rather than on every new width.
    """
    need = m * kb
    buf = _ONES_A.get(device)
    if buf is not None and buf.numel() >= need:
        return buf[:need].view(m, kb)

    from ..capture import is_capturing

    if is_capturing():
        # torch.ones records a fill into the graph and takes its memory from
        # that graph's private pool, which later graphs reuse; caching it would
        # leave us reading another graph's bytes, and unlike a buffer rewritten
        # per call there would be nothing to correct it. Warmup runs these
        # shapes eagerly first, so in practice the buffer is already big enough.
        return torch.ones(m, kb, dtype=torch.float32, device=device)

    buf = torch.ones(
        1 << max(need - 1, 0).bit_length(), dtype=torch.float32, device=device
    )
    _ONES_A[device] = buf
    return buf[:need].view(m, kb)


def _ones_for(n: int, k: int, device: torch.device) -> torch.Tensor:
    """The all-ones ``BS``, cached per weight shape rather than per call."""
    key = (n, k, device)
    buf = _ONES.get(key)
    if buf is None:
        buf = torch.ones(
            n // _BLOCK, k // _BLOCK, dtype=torch.float32, device=device
        )
        _ONES[key] = buf
    return buf


def can_run(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    output_dtype: torch.dtype,
) -> bool:
    """Whether this module implements the GEMM it is being offered."""
    if _disabled():
        return False
    dg = _lib()
    if dg is None:
        return False

    # The GEMM emits bfloat16 and the epilogue writes the result in place.
    if output_dtype != torch.bfloat16:
        return False
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        return False
    if not a.is_contiguous():
        return False
    # vLLM hands over weight.t(): [K, N] strided (1, K), so transposing it back
    # is a free contiguous [N, K] view. Anything else would mean copying a
    # weight on every call.
    if b.stride(0) != 1 or not b.t().is_contiguous():
        return False
    # Per-token on the activation, per-channel on the weight. Scalars mean a
    # per-tensor scheme, which the folding below does not express.
    if scale_a.numel() != a.shape[0] or scale_b.numel() != b.shape[1]:
        return False
    if scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
        return False
    # The epilogue reads scale_a by row index and scale_b with unit stride, both
    # assuming unit stride. Callers come through _normalize_scale, which hands
    # over 1-D contiguous scales, so this only ever declines a caller that did
    # not.
    if not scale_a.is_contiguous() or not scale_b.is_contiguous():
        return False

    m, k = a.shape
    n = b.shape[1]
    # A partial k-block or n-block would carry a replicated scale over columns
    # that do not exist.
    if n % _BLOCK or k % _BLOCK:
        return False
    if n < _MIN_N or k < _MIN_K:
        return False
    return _profitable(dg, m, n, k)


def scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """``(a @ b) * scale_a * scale_b + bias`` through the vendor GEMM."""
    dg = _lib()
    m, k = a.shape
    n = b.shape[1]

    # Both of the GEMM's scale operands are all-ones constants; the real
    # scales are applied together in the epilogue. Handing the per-token
    # scale to the GEMM instead means writing an [M, k_blocks] copy of it
    # per call for arithmetic the epilogue was already set up to do.
    kb = k // _BLOCK
    out = dg.int8_blockwise_normal_gemm(
        a,
        b.t(),
        _ones_a_for(m, kb, a.device),
        _ones_for(n, k, a.device),
        _BLOCK,
        _BLOCK,
    )

    block_n = min(1024, triton.next_power_of_2(n))
    _epilogue_kernel[(m, triton.cdiv(n, block_n))](
        out,
        scale_a,
        scale_b,
        bias,
        n,
        HAS_BIAS=bias is not None,
        BLOCK_N=block_n,
    )
    return out


def try_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Run the vendor GEMM for this call, or decline and leave it to Triton."""
    global _DECISION_LOGGED
    if not can_run(a, b, scale_a, scale_b, output_dtype):
        return None
    if not _DECISION_LOGGED:
        _DECISION_LOGGED = True
        logger.info(
            "dense-int8: INT8 linears now run the deepgemm_ptpu GEMM where it "
            "is profitable; narrow-N, shallow-K and decode-tile shapes keep "
            "Triton. Disable with %s=1.",
            _BASELINE_ENV,
        )
    return scaled_mm(a, b, scale_a, scale_b, bias)
