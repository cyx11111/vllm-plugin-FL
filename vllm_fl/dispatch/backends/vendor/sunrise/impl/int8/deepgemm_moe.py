# Copyright (c) 2026 BAAI. All rights reserved.

"""INT8 W8A8 routed experts on the vendor grouped GEMM, with our quantization.

**It may not allocate inside a captured graph.** Doing so was wrong and silent:
with ``cudagraph_mode: FULL_DECODE_ONLY`` on Qwen3.6-35B-A3B-w8a8, GSM8K was
88.3% against 93.3% for the Triton dataflow it replaces, while this pipeline
with cudagraphs off was 93.3% all along. The cause was that ``fused_experts``
allocated about a dozen scratch tensors per layer per call -- the routing pair,
the permutation and its inverse, the sorted activation and its scale, both GEMM
outputs, the activation-quant pair, and the vendor's own workspace, which
``int8_blockwise_group_gemm`` allocates per call when handed no ``workspace=``.
At 40 layers that is some five hundred addresses baked into every decode graph,
where vLLM's own dataflow writes into the ``workspace13`` / ``workspace2``
buffers its caller allocated once and outside.

``impl/int8/moe_scratch.py`` hoists all of them and ``_prewarm`` settles every
host-side vendor query before the first capture; that restored both the
accuracy and the throughput, and is why this module is on the serving path.
``_scratch_is_safe`` keeps it that way by declining any capture the pool does
not cover, so the failure cannot come back by accident.

Two other explanations were tried first and neither was it, both measured at
88.3%: raising ``cudagraph_num_of_warmups`` so no shape is first seen inside a
capture, and refusing to decide or cache while ``is_capturing()``.

The vendor GEMM wants ``A`` physically grouped by expert with boundaries in
``cumsum``. vLLM's ``fused_experts_impl`` instead gathers through
``sorted_token_ids``, so this module implements the whole expert path rather
than swapping a single kernel. The caller falls back to vLLM's dataflow when
we refuse.

Adaptation notes:
* ``B`` is ``w.transpose(1, 2)`` of vLLM's contiguous ``[E, N, K]`` weight.
* ``cumsum`` accepts arbitrary group boundaries, so ``moe_align_block_size``
  padding is not needed.
* ``moe_sort`` writes an inclusive scan; handing the tail of that buffer
  converts it to the exclusive ``[E+1]`` boundaries the GEMM wants.
* ``n_block_size`` is only 64 or 128, so the per-channel weight scale cannot
  ride in ``BS``. ``BS`` is all ones and an epilogue applies the scale after
  the GEMM's BF16 store (the one extra rounding vs our Triton kernel).

Launch count matters on this device, so the inverse permutation is written by
the gather, ``BS`` is cached, and both weight scales fold into kernels that
already read those values. Quantization stays ours: the vendor's fused quant
truncates instead of rounding. Quantizing before routing also moves INT8
rather than BF16. The reduce is ours because vendor ``moe_reduce`` only
accepts ``top_k`` in {1, 2, 4, 8}.

Set ``VLLM_FL_SUNRISE_BENCH_BASELINE_DEEPGEMM_MOE=1`` to keep the Triton
grouped kernel and vLLM's dataflow.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from . import moe_scratch

logger = logging.getLogger(__name__)

# n_block_size and k_block_size handed to the vendor GEMM. Only 64 and 128
# are accepted for the N block, and which of the two is right is a property
# of the stage: the same shape can collapse on the wrong choice. Both are
# offered and the vendor is asked which it would rather have.
_NARROW_BLOCK = 64
_WIDE_BLOCK = 128


def _blocks() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The block sizes to offer the vendor, widest first.

    The two axes always get the same tuple: only equal ``(n_block, k_block)``
    is numerically sound (see ``_stage_blocks``). Widest first, because a
    tie on the vendor's own answer resolves to the wider block. Withholding
    the narrow one reproduces the original gate (only (128, 128) survives).
    """
    if _env_on("VLLM_FL_SUNRISE_BENCH_BASELINE_DEEPGEMM_MOE_WIDE_BLOCK"):
        both = (_WIDE_BLOCK,)
    else:
        both = (_WIDE_BLOCK, _NARROW_BLOCK)
    return both, both

# Per-shape record of the blocks to use, or None where the vendor GEMM should not
# be asked at all. Its own is_support_int8_blockwise_gemm() is not usable as a
# gate -- for identical arguments it has both returned True and raised "no valid
# fallback config found, M=1" -- so the answer is cached and a refusal is
# permanent for that shape.
_BLOCKS: dict[tuple[int, ...], Optional[tuple[int, int]]] = {}

# The vendor's autotune table has holes: where it has one it picks a
# degenerate N tile and collapses. What separates those cliffs from the
# wins is how far the tile sits below the block, not the tile's absolute
# width. Stating the rule as a ratio keeps it meaningful at both 64 and
# 128 blocks.
_MAX_SUBDIVISION = 2
_N_PER_BLOCK = re.compile(r"n_per_block=(\d+)")

# Shallowest K worth handing over. At K=64 one k-step leaves nothing but
# epilogue, and the config log reports a full-width tile so
# _MAX_SUBDIVISION cannot see it.
_MIN_GEMM_K = 128

_UNAVAILABLE: Optional[str] = None


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _disabled() -> bool:
    return _env_on("VLLM_FL_SUNRISE_BENCH_BASELINE_DEEPGEMM_MOE")


_WARNED_UNPOOLED = False


def _scratch_is_safe(
    hidden_states: torch.Tensor, topk_ids: torch.Tensor
) -> bool:
    """Refuse to record this pipeline into a graph it cannot be pooled for.

    Allocating inside a captured region is what made this module wrong: it cost
    5 points of GSM8K until ``moe_scratch`` hoisted the scratch out, and it was
    silent, so it does not get to happen again by accident. Pooling depends on
    the capture sizes being known, so if a call is being recorded and does not
    fit the pool, the Triton dataflow is recorded instead.

    Eager calls are unaffected. Nothing bakes their addresses, and eager is the
    configuration this pipeline was always correct in.
    """
    global _WARNED_UNPOOLED
    from ..capture import is_capturing

    if not is_capturing():
        return True

    tokens_max = moe_scratch.capture_max_tokens()
    rows = hidden_states.shape[0] * topk_ids.size(1)
    if tokens_max is not None and rows <= tokens_max * topk_ids.size(1):
        return True

    if not _WARNED_UNPOOLED:
        _WARNED_UNPOOLED = True
        logger.warning(
            "deepgemm-moe: declining a capture of %d rows that the scratch "
            "pool does not cover (capacity %s). The Triton dataflow is being "
            "recorded instead, which is correct but slower. If this is every "
            "decode size, the capture sizes did not reach moe_scratch.",
            rows,
            "unknown" if tokens_max is None else tokens_max * topk_ids.size(1),
        )
    return False


def _libs():
    """The two vendor wheels, or None with the reason recorded once."""
    global _UNAVAILABLE
    if _UNAVAILABLE is not None:
        return None
    try:
        import deepgemm_ptpu
        import llm_kernel_ptpu

        return deepgemm_ptpu, llm_kernel_ptpu
    except Exception as exc:  # noqa: BLE001
        _UNAVAILABLE = str(exc)
        logger.info(
            "deepgemm-moe: vendor grouped GEMM unavailable (%s); INT8 experts "
            "keep the Triton grouped kernel.",
            exc,
        )
        return None


def available() -> bool:
    """Whether this path could run at all, without importing the wheels.

    For the startup log line only, so it answers before device init and cannot
    be the thing that triggers it. Whether a given call is taken is ``can_run``.
    """
    if _disabled():
        return False
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("deepgemm_ptpu", "llm_kernel_ptpu")
    )


@triton.jit
def _gather_rows_kernel(
    src_ptr,
    src_scale_ptr,
    dst_ptr,
    dst_scale_ptr,
    row_ptr,
    inv_ptr,
    k,
    sfk,
    top_k,
    BLOCK_K: tl.constexpr,
    BLOCK_SFK: tl.constexpr,
):
    """Lay the quantized activation out in expert order for the vendor GEMM.

    ``row_ptr`` holds flat ``token * top_k + slot`` indices grouped by expert, so
    row ``i`` of the output is token ``row_ptr[i] // top_k``. The per-token scale
    is written out replicated across all ``sfk`` k-blocks at the same time: the
    GEMM multiplies each k-block's partial sum by its own scale, and a value that
    does not vary along K factors straight back out of that sum, which is what
    makes a per-token scale exact under a blockwise contract.

    The inverse permutation the reduce needs is written here too. It was
    its own kernel; this one already loads ``row_ptr[i]``, so the scatter
    is a store.
    """
    i = tl.program_id(0)
    tile = tl.program_id(1)
    flat = tl.load(row_ptr + i)
    token = flat // top_k

    offs = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < k
    tl.store(
        dst_ptr + i * k + offs,
        tl.load(src_ptr + token * k + offs, mask=mask, other=0),
        mask=mask,
    )

    if tile == 0:
        scale = tl.load(src_scale_ptr + token)
        soffs = tl.arange(0, BLOCK_SFK)
        tl.store(dst_scale_ptr + i * sfk + soffs, scale, mask=soffs < sfk)
        tl.store(inv_ptr + flat, i)


# The widest row this kernel will quantize in one pass, matching the cap in
# ``scaled_int8_quant``. Above it the row would need a second HBM read to find
# its amax, which is the round trip the fusion exists to remove, so a model that
# wide keeps the separate quantize and gather instead.
_QUANT_GATHER_MAX_K = 8192


@triton.jit
def _quant_gather_rows_kernel(
    src_ptr,
    dst_ptr,
    dst_scale_ptr,
    row_ptr,
    inv_ptr,
    k,
    sfk,
    top_k,
    src_row_stride,
    eps,
    scale_denom,
    int8_min,
    int8_max,
    BLOCK_K: tl.constexpr,
    BLOCK_SFK: tl.constexpr,
):
    """Quantize the activation straight into expert order, in one pass.

    This is ``_gather_rows_kernel`` with the per-token quantize folded in
    front of it. It used to be two launches: quantize the whole activation
    in token order, then permute the int8 copy into expert order. The int8
    buffer existed only to carry values between them.

    The arithmetic is ``_dynamic_scaled_int8_quant_single_kernel``'s, so
    the result is bit-identical to quantizing first. Each row recomputes
    the amax of its source token rather than reading a scale someone else
    computed. Every row of a token gets the same scale because it is the
    same reduction over the same values in the same order.
    """
    i = tl.program_id(0)
    flat = tl.load(row_ptr + i)
    token = flat // top_k

    cols = tl.arange(0, BLOCK_K)
    mask = cols < k
    y = tl.load(src_ptr + token * src_row_stride + cols, mask=mask, other=0.0)
    y = y.to(tl.float32)
    amax = tl.max(tl.abs(y))
    scale = tl.maximum(amax, eps) / scale_denom

    inv_scale = 1.0 / scale
    x = y * inv_scale
    r = tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))
    r = tl.minimum(tl.maximum(r, int8_min), int8_max)
    tl.store(dst_ptr + i * k + cols, r.to(dst_ptr.dtype.element_ty), mask=mask)

    soffs = tl.arange(0, BLOCK_SFK)
    tl.store(dst_scale_ptr + i * sfk + soffs, scale, mask=soffs < sfk)
    tl.store(inv_ptr + flat, i)


@triton.jit
def _reduce_kernel(
    src_ptr,
    weight_ptr,
    inv_ptr,
    out_ptr,
    wscale_ptr,
    expert_ptr,
    cols,
    TOP_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Weighted sum of a token's expert outputs, undoing the permutation.

    ``llm_kernel_ptpu.moe_reduce`` does this for ``top_k`` in {1, 2, 4, 8}
    only; other arities fail on the device. Doing it here drops that
    restriction.

    The second GEMM's per-channel weight scale is applied here rather than
    in a pass of its own, which also avoids a BF16 round-trip before the
    reduce widens back to fp32.
    """
    token = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs < cols
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for slot in tl.static_range(TOP_K):
        flat = token * TOP_K + slot
        row = tl.load(inv_ptr + flat)
        weight = tl.load(weight_ptr + flat)
        expert = tl.load(expert_ptr + row)
        scale = tl.load(wscale_ptr + expert * cols + offs, mask=mask, other=0.0)
        value = tl.load(src_ptr + row * cols + offs, mask=mask, other=0.0)
        acc += value.to(tl.float32) * scale * weight
    tl.store(out_ptr + token * cols + offs, acc.to(tl.bfloat16), mask=mask)


# The reduce's tile width. This kernel loads ``top_k`` rows of the [E, cols]
# weight scale per program; a tile that is too narrow makes that gather
# degrade, while a tile that is too wide leaves too few programs on the
# device. A narrower tile is used only while the program count stays small.
_REDUCE_TILE = 256
_REDUCE_NARROW_TILE = 128
_REDUCE_NARROW_MAX_PROGRAMS = 128


def _reduce_tile(tokens: int, cols: int) -> int:
    """The reduce's ``BLOCK_C``; see ``_REDUCE_TILE``."""
    wide = min(_REDUCE_TILE, triton.next_power_of_2(cols))
    narrow = min(_REDUCE_NARROW_TILE, triton.next_power_of_2(cols))
    if tokens * triton.cdiv(cols, narrow) <= _REDUCE_NARROW_MAX_PROGRAMS:
        return narrow
    return wide


_ONES: dict[tuple, torch.Tensor] = {}


_WS_BYTES: dict[tuple, int] = {}

# Per geometry: the byte size to give each stage's vendor workspace slot, taken
# over every capture size so the slot is never resized.
_WS_SLOT: dict[tuple, tuple[int, int]] = {}
_PREWARMED: set = set()


def _ws_bytes(dg, m: int, n: int, k: int, nb: int, kb: int, group: int) -> int:
    """The vendor's workspace size for one grouped GEMM, cached.

    ``int8_blockwise_group_gemm`` allocates this itself when handed no
    ``workspace=``, which inside a captured graph is one more baked address per
    stage per layer. Asking here is what lets the caller hand it a pooled
    buffer instead. It does not save the query -- the vendor makes its own
    either way -- it is only how ``_prewarm`` learns how large to make the
    slot. The vendor does not check the size it is given, so a slot cut for
    the widest capture size serves every narrower one.
    """
    key = (m, n, k, nb, kb, group)
    got = _WS_BYTES.get(key)
    if got is None:
        got = int(
            dg.get_int8_blockwise_group_gemm_workspace_bytes(
                m, n, k, nb, kb, group
            )
        )
        _WS_BYTES[key] = got
    return got


def _prewarm(
    dg,
    experts: int,
    n: int,
    k: int,
    width: int,
    top_k: int,
    device: torch.device,
) -> None:
    """Decide every capturable decode shape now, while nothing is recording.

    Three things in here decide on first touch and none may do it inside a
    capture: ``_stage_blocks`` and ``_ws_bytes`` ask the vendor host-side config
    queries, which allocate, and ``_ones_bs`` allocates a buffer it then caches
    -- from the recording graph's private memory pool, which later captures
    reuse. vLLM does not warm up before capturing (``cudagraph_num_of_warmups``
    defaults to 0), so nothing else would force these outside one. This runs
    from the first call, which is the eager profile run.
    """
    key = (experts, n, k, width, top_k, device)
    if key in _PREWARMED:
        return
    _PREWARMED.add(key)

    sizes = moe_scratch.capture_sizes()
    if not sizes:
        # Either nothing is captured, or the capture sizes could not be found
        # -- which silently turns the pool off and puts every allocation back
        # inside the captured region, the exact thing this is here to prevent.
        # It has happened once already, so it is said out loud.
        logger.warning(
            "deepgemm-moe: no cudagraph capture sizes, so the scratch pool is "
            "off and every expert call allocates. Harmless when serving "
            "eagerly; if cudagraphs are on, the hoisting is not in effect."
        )

    ws1 = ws2 = 0
    for tokens in sizes:
        rows = tokens * top_k
        stage1 = _stage_blocks(dg, rows, n, k, experts)
        stage2 = _stage_blocks(dg, rows, k, width, experts)
        if stage1 is None or stage2 is None:
            continue
        nb1, kb1 = stage1
        nb2, kb2 = stage2
        _ones_bs(experts, n // nb1, k // kb1, device)
        _ones_bs(experts, k // nb2, width // kb2, device)
        ws1 = max(ws1, _ws_bytes(dg, rows, n, k, nb1, kb1, experts))
        ws2 = max(ws2, _ws_bytes(dg, rows, k, width, nb2, kb2, experts))

    _WS_SLOT[key] = (ws1, ws2)
    if ws1:
        moe_scratch.slot("gemm1_workspace", (ws1,), torch.int8, device)
    if ws2:
        moe_scratch.slot("gemm2_workspace", (ws2,), torch.int8, device)

    if sizes:
        logger.info(
            "deepgemm-moe: scratch pooled for capture sizes up to %d (%d rows "
            "at top_k=%d); %d shapes decided eagerly, workspaces %d/%d bytes.",
            sizes[-1],
            sizes[-1] * top_k,
            top_k,
            len(sizes),
            ws1,
            ws2,
        )


def _ones_bs(experts: int, nb: int, sfk: int, device: torch.device):
    """The GEMM's ``BS`` argument, which is all ones and never changes.

    The weight scale is per channel and the GEMM's n-block cannot express
    it, so it is applied in an epilogue. Cache the ones buffer: building
    it with ``torch.ones`` every call is an allocation plus a fill kernel
    for a tensor that does not change.
    """
    key = (experts, nb, sfk, device)
    buf = _ONES.get(key)
    if buf is None:
        buf = torch.ones(
            experts, nb, sfk, dtype=torch.float32, device=device
        ).transpose(1, 2)
        _ONES[key] = buf
    return buf


def _n_per_block(
    dg, rows: int, n: int, k: int, n_block: int, k_block: int, experts: int
) -> Optional[int]:
    """The narrowest N tile the vendor would pick, across best and fallback.

    Both are consulted because the kernel logs say it does not always take the
    best one, and a block is only accepted if neither choice is degenerate.
    """
    widths = []
    for name in ("get_best_config_log", "get_fallback_config_log"):
        try:
            log = str(
                getattr(dg, name)(
                    rows, n, k, n_block, k_block, experts
                )
            )
        except Exception:  # noqa: BLE001
            continue
        found = _N_PER_BLOCK.search(log)
        if found:
            widths.append(int(found.group(1)))
    return min(widths) if widths else None


def _stage_blocks(
    dg, rows: int, n: int, k: int, experts: int
) -> Optional[tuple[int, int]]:
    """``(n_block, k_block)`` for one expert GEMM, or None to leave it alone.

    The two axes always get the same value: mixed pairs are accepted by
    ``is_support`` but compute the wrong thing (and can NaN the layer).
    Among equal pairs that fit, take the one the vendor reports the widest
    tile for; a large subdivision of the N block is a collapse. Ties go
    to the wider block.
    """
    blocks, _ = _blocks()
    key = (rows, n, k, experts, blocks)
    if key in _BLOCKS:
        return _BLOCKS[key]

    chosen: Optional[tuple[int, int]] = None
    if k >= _MIN_GEMM_K:
        best_tile = 0
        for block in blocks:
            if n % block or k % block:
                continue
            n_block = k_block = block
            try:
                if not dg.is_support_int8_blockwise_gemm(
                    rows, n, k, n_block, k_block, experts
                ):
                    continue
                tile = _n_per_block(
                    dg, rows, n, k, n_block, k_block, experts
                )
            except Exception as exc:  # noqa: BLE001
                # Its config lookup raises for shapes it has no entry for.
                # Treat that as a refusal of this block rather than letting
                # it escape mid-layer.
                logger.debug(
                    "deepgemm-moe: vendor GEMM refused M=%d N=%d K=%d "
                    "group=%d at block=%d (%s)",
                    rows,
                    n,
                    k,
                    experts,
                    block,
                    exc,
                )
                continue
            if tile is None or block > _MAX_SUBDIVISION * tile:
                logger.debug(
                    "deepgemm-moe: declining M=%d N=%d K=%d group=%d at "
                    "block=%d; the vendor would tile N by %s.",
                    rows,
                    n,
                    k,
                    experts,
                    block,
                    tile,
                )
                continue
            if tile > best_tile:
                best_tile, chosen = tile, (n_block, k_block)

    _BLOCKS[key] = chosen
    return chosen


def can_run(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    use_int8_w8a8: bool,
    per_channel_quant: bool,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    block_shape: Optional[list[int]],
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
) -> bool:
    """Whether this module implements the expert call it is being offered."""
    if _disabled():
        return False
    if not _scratch_is_safe(hidden_states, topk_ids):
        return False
    libs = _libs()
    if libs is None:
        return False
    dg, _ = libs

    if not use_int8_w8a8 or not per_channel_quant:
        return False
    if activation != "silu" or apply_router_weight_on_input:
        return False
    # Static activation scales, zero points, biases and blockwise weight scales
    # all mean a different dequant than the epilogue here performs.
    if any(
        t is not None
        for t in (w1_zp, w2_zp, a1_scale, a2_scale, w1_bias, w2_bias)
    ):
        return False
    if block_shape is not None:
        return False
    # Expert parallelism marks unrouted tokens with -1 and expects them to
    # contribute nothing; the grouped layout here has no slot for that yet.
    if expert_map is not None:
        return False

    # The vendor GEMM only emits bfloat16, and the reduce writes the layer output.
    if hidden_states.dtype != torch.bfloat16:
        return False
    if not hidden_states.is_contiguous():
        return False
    if w1.dtype != torch.int8 or w2.dtype != torch.int8:
        return False
    if w1.ndim != 3 or w2.ndim != 3:
        return False
    # w.transpose(1, 2) is handed to the GEMM as its K-major B, which is only a
    # view if the weight is contiguous.
    if not w1.is_contiguous() or not w2.is_contiguous():
        return False
    if w1_scale is None or w2_scale is None:
        return False
    if topk_weights.dtype != torch.float32:
        return False

    experts, n, k = w1.shape
    if global_num_experts not in (-1, experts):
        return False
    if w2.shape[0] != experts or w2.shape[1] != k or w2.shape[2] * 2 != n:
        return False
    # One scale per output channel, which is what the epilogue applies. A
    # per-tensor scale has the same rank and trailing extent, so the channel
    # count is what separates them.
    if w1_scale.shape != (experts, n, 1) or w2_scale.shape != (experts, k, 1):
        return False
    rows = topk_ids.numel()
    if rows == 0:
        return False

    # Both stages or neither: the routing, the permute and the two scale buffers
    # are shared, so there is no half of this to run. Each stage's blocks have to
    # divide its own N and K, and _stage_blocks returns None when none of them
    # do, which is what keeps every k-block full -- a partial block would carry a
    # replicated scale over columns that do not exist.
    #
    # ``silu_and_mul_quant_int8`` needs no check of its own here: it wants a 2-D
    # contiguous bf16/fp16/fp32 tensor of even width, and what it gets is a
    # freshly allocated contiguous bf16 [rows, 2 * width].
    if _stage_blocks(dg, rows, n, k, experts) is None:
        return False
    return _stage_blocks(dg, rows, k, w2.size(2), experts) is not None


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """The INT8 routed-expert path: our quantization, the vendor grouped GEMM."""
    dg, lk = _libs()

    from .scaled_int8_quant import (
        _INT8_MAX,
        _INT8_MIN,
        _QUANT_EPS,
        _SCALE_DENOM,
        scaled_int8_quant,
    )
    from .silu_and_mul_quant import silu_and_mul_quant_int8

    device = hidden_states.device
    tokens, k = hidden_states.shape
    experts, n, _ = w1.shape
    width = w2.size(2)
    top_k = topk_ids.size(1)
    rows = tokens * top_k

    geometry = (experts, n, k, width, top_k, device)
    _prewarm(dg, *geometry)

    # Same call ``can_run`` already made and cached, so this is a dict lookup.
    (nb1, kb1) = _stage_blocks(dg, rows, n, k, experts)
    (nb2, kb2) = _stage_blocks(dg, rows, k, width, experts)
    sfk1, sfk2 = k // kb1, width // kb2

    # Every buffer below is a prefix of a persistent pool slot when this call is
    # one a decode graph can capture, and a plain allocation otherwise. What
    # that avoids is a dozen allocations per layer per call landing inside the
    # captured region; see ``impl/int8/moe_scratch.py``.
    tokens_max = moe_scratch.capture_max_tokens()
    scratch = moe_scratch.Scratch(
        rows, None if tokens_max is None else tokens_max * top_k, device
    )

    # Routing: histogram, then the expert-major order and its group boundaries.
    # moe_exp_count accumulates rather than overwrites, so the histogram has to
    # start at zero -- a stale buffer yields out-of-range boundaries and the
    # GEMM faults on the device. The two live in one buffer because zeroing
    # twice is two launches of pure fixed cost. ``boundaries[0]`` is the leading
    # zero the GEMM wants and moe_sort writes the rest.
    routing = scratch.fixed(
        "routing", (2 * experts + 1,), torch.int32, zero=True
    )
    histogram = routing[:experts]
    boundaries = routing[experts:]
    permuted_rows = scratch.rows("permuted_rows", (), torch.int32)
    row_expert = scratch.rows("row_expert", (), torch.int32)
    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    lk.moe_exp_count(ids, histogram)
    lk.moe_sort(ids, histogram, boundaries[1:], permuted_rows, row_expert)

    # Quantize straight into expert order. The kernel writes the inverse
    # permutation the reduce needs on the side; both that and the quantize used
    # to be launches of their own.
    inverse = scratch.rows("inverse", (), torch.int32)
    a_sorted = scratch.rows("a_sorted", (k,), torch.int8)
    a_scale = scratch.rows("a_scale", (sfk1,), torch.float32)
    if k <= _QUANT_GATHER_MAX_K:
        block_k = triton.next_power_of_2(k)
        _quant_gather_rows_kernel[(rows,)](
            hidden_states,
            a_sorted,
            a_scale,
            permuted_rows,
            inverse,
            k,
            sfk1,
            top_k,
            hidden_states.stride(0),
            _QUANT_EPS,
            _SCALE_DENOM,
            _INT8_MIN,
            _INT8_MAX,
            BLOCK_K=block_k,
            BLOCK_SFK=triton.next_power_of_2(sfk1),
            num_warps=min(max(block_k // 256, 1), 8),
        )
    else:
        a1q, a1s, _ = scaled_int8_quant(hidden_states)
        a1s = a1s.reshape(tokens)
        block_k = min(1024, triton.next_power_of_2(k))
        _gather_rows_kernel[(rows, triton.cdiv(k, block_k))](
            a1q,
            a1s,
            a_sorted,
            a_scale,
            permuted_rows,
            inverse,
            k,
            sfk1,
            top_k,
            BLOCK_K=block_k,
            BLOCK_SFK=triton.next_power_of_2(sfk1),
        )

    ones1 = _ones_bs(experts, n // nb1, sfk1, device)
    ones2 = _ones_bs(experts, k // nb2, sfk2, device)

    # The vendor allocates its own workspace when given none. Hand it a pooled
    # one on the capturable path; the slot is sized for the widest capture size
    # and the kernel takes the prefix it needs, so passing all of it is safe.
    ws1 = ws2 = None
    slot_bytes = _WS_SLOT.get(geometry)
    if scratch.pooled and slot_bytes is not None:
        if slot_bytes[0]:
            ws1 = moe_scratch.slot(
                "gemm1_workspace", (slot_bytes[0],), torch.int8, device
            )
        if slot_bytes[1]:
            ws2 = moe_scratch.slot(
                "gemm2_workspace", (slot_bytes[1],), torch.int8, device
            )

    c1 = scratch.rows("c1", (n,), torch.bfloat16)
    dg.int8_blockwise_group_gemm(
        a_sorted,
        w1.transpose(1, 2),
        a_scale,
        ones1,
        nb1,
        kb1,
        boundaries,
        workspace=ws1,
        out=c1,
    )

    # gemm1's weight scale and the k-block replication of the activation
    # scale both ride inside this kernel; each was a launch of its own.
    a2q, a2_scale = silu_and_mul_quant_int8(
        c1,
        wscale=w1_scale.squeeze(-1),
        expert_of_row=row_expert,
        sfk=sfk2,
        out=scratch.rows("a2q", (width,), torch.int8),
        scale_out=scratch.rows("a2_scale", (sfk2,), torch.float32),
    )

    c2 = scratch.rows("c2", (k,), torch.bfloat16)
    dg.int8_blockwise_group_gemm(
        a2q,
        w2.transpose(1, 2),
        a2_scale,
        ones2,
        nb2,
        kb2,
        boundaries,
        workspace=ws2,
        out=c2,
    )

    # The second GEMM's weight scale rides into the reduce rather than taking a
    # pass of its own over C2.
    #
    # This one allocation is deliberately not pooled. It is the value this
    # function returns, so the caller decides how long it lives, and a pooled
    # buffer would be overwritten by the next layer's call. vLLM's MoE layers
    # ask for ``inplace``, in which case nothing is allocated here at all.
    out = hidden_states if inplace else torch.empty_like(hidden_states)
    block_c = _reduce_tile(tokens, k)
    _reduce_kernel[(tokens, triton.cdiv(k, block_c))](
        c2,
        topk_weights.reshape(-1),
        inverse,
        out,
        w2_scale.squeeze(-1),
        row_expert,
        k,
        TOP_K=top_k,
        BLOCK_C=block_c,
    )
    return out


def try_fused_experts(arguments: dict[str, Any]) -> Optional[torch.Tensor]:
    """Run the vendor path for ``fused_experts_impl``'s arguments, or decline."""
    if not can_run(
        arguments["hidden_states"],
        arguments["w1"],
        arguments["w2"],
        arguments["topk_weights"],
        arguments["topk_ids"],
        arguments.get("activation", "silu"),
        arguments.get("apply_router_weight_on_input", False),
        arguments.get("use_int8_w8a8", False),
        arguments.get("per_channel_quant", False),
        arguments.get("global_num_experts", -1),
        arguments.get("expert_map"),
        arguments.get("w1_scale"),
        arguments.get("w2_scale"),
        arguments.get("w1_zp"),
        arguments.get("w2_zp"),
        arguments.get("a1_scale"),
        arguments.get("a2_scale"),
        arguments.get("block_shape"),
        arguments.get("w1_bias"),
        arguments.get("w2_bias"),
    ):
        return None
    return fused_experts(
        arguments["hidden_states"],
        arguments["w1"],
        arguments["w2"],
        arguments["topk_weights"],
        arguments["topk_ids"],
        arguments["inplace"],
        arguments["w1_scale"],
        arguments["w2_scale"],
    )
