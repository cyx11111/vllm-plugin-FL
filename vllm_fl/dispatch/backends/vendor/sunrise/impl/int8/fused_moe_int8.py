# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU INT8 W8A8 grouped GEMM for the routed MoE experts.

Replaces ``vllm.model_executor.layers.fused_moe.fused_moe.fused_moe_kernel``
for per-channel weight scales, per-token activation scales, no bias, no
blockwise quantization.

The sunrise Triton backend's INT8 MMA only stays at rate when the whole K
loop stays on that path. vLLM's generic kernel cannot: it accumulates
``tl.dot`` into fp32 outside the MMA, masks both operand loads inside the
K loop, and uses int64 offsets. This kernel uses an int32 accumulator
threaded through ``tl.dot``, clamped (not masked) loads, and 32-bit
offsets -- the same fixes as ``impl/int8/scaled_mm``.

The remaining gap vs a dense INT8 GEMM is the down projection: K is the
per-rank expert width, so the stage is bandwidth-bound. ``num_stages``
beyond 2 and split-K are not useful here; the expert dimension already
supplies parallelism. Results are bit-identical to the kernel it
replaces (same integer dot product).

``can_run`` refuses anything this kernel does not implement:
* K must be divisible by ``BLOCK_SIZE_K`` (unmasked K loop).
* Offsets must fit in int32 (checked per launch).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import triton
import triton.language as tl

_INT32_MAX = (1 << 31) - 1


@triton.jit
def _fused_moe_int8_kernel(
    A,
    B,
    C,
    AScale,
    BScale,
    TopkWeights,
    SortedIds,
    ExpertIds,
    NumPostPad,
    N,
    K,
    EM,
    num_valid_tokens,
    num_rows,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_bse,
    stride_bsn,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_M >= tl.load(NumPostPad):
        return

    offs_token = tl.load(SortedIds + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    token_mask = offs_token < num_valid_tokens
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Padded rows carry a token id past the end. Clamping keeps the row address in
    # bounds; ``token_mask`` is what actually decides whether it is written.
    offs_cm = tl.minimum(offs_token, num_valid_tokens - 1)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    off_experts = tl.load(ExpertIds + pid_m)
    if off_experts == -1:
        # Expert belongs to another rank: this block of the output is zero.
        tl.store(c_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=C.dtype.element_ty),
                 mask=c_mask)
        return

    # Clamp rather than mask, so the K loop needs no predication and stays on the
    # INT8 MMA path. Padded rows and the N tail read a valid but meaningless
    # location; ``c_mask`` discards whatever they accumulate.
    a_row = tl.minimum(offs_token // top_k, num_rows - 1)
    offs_bn = tl.minimum(offs_cn, N - 1)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        B
        + off_experts * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    # int32 accumulation stays inside the MMA; K divides BLOCK_K by construction,
    # so there is no tail step to predicate.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K, BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out = acc.to(tl.float32)
    out *= tl.load(AScale + a_row * stride_asm)[:, None]
    out *= tl.load(BScale + off_experts * stride_bse + offs_bn * stride_bsn)[None, :]
    if MUL_ROUTED_WEIGHT:
        out *= tl.load(TopkWeights + offs_cm)[:, None]

    tl.store(c_ptrs, out.to(C.dtype.element_ty), mask=c_mask)


def _fits_int32(*offsets: int) -> bool:
    return all(o <= _INT32_MAX for o in offsets)


def can_run(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    sorted_token_ids: Optional[torch.Tensor],
    top_k: int,
    block_size_m: int,
    block_size_k: int,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[list[int]],
    B_bias: Optional[torch.Tensor],
) -> bool:
    """Whether this kernel implements the launch it is being offered."""
    if not use_int8_w8a8 or not per_channel_quant:
        return False
    if use_fp8_w8a8 or use_int8_w8a16 or use_int4_w4a16:
        return False
    if block_shape is not None or B_bias is not None:
        return False
    if sorted_token_ids is None:
        return False
    if A_scale is None or B_scale is None:
        return False
    if A.dtype != torch.int8 or B.dtype != torch.int8 or B.ndim != 3:
        return False
    # Per-token activation scales and per-channel weight scales, which is what the
    # epilogue assumes; anything coarser has a different dequant.
    if A_scale.numel() != A.size(0) or B_scale.ndim != 3 or B_scale.size(2) != 1:
        return False

    N, K = B.size(1), B.size(2)
    if K % block_size_k or block_size_k > K:
        return False
    # ``tl.arange`` wants a power of two, and the INT8 MMA's K granularity is 16.
    # The caller passes the same block the moe_align_block_size padding used, so
    # one expert per block follows from that rather than needing a check here.
    if block_size_m < 16 or block_size_m & (block_size_m - 1):
        return False
    if block_size_m > sorted_token_ids.size(0):
        return False

    num_valid_tokens = A.size(0) * top_k
    return _fits_int32(
        A.size(0) * A.stride(0) + K * A.stride(1),
        B.size(0) * B.stride(0),
        num_valid_tokens * C.stride(1) + N * C.stride(2),
    )


def invoke(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
) -> None:
    """Launch the grouped GEMM. Signature mirrors the caller it substitutes for."""
    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    block_k = config["BLOCK_SIZE_K"]
    N, K = B.size(1), B.size(2)

    EM = sorted_token_ids.size(0)
    if A.size(0) < block_m:
        # Same small-batch trim as the caller: with fewer tokens than a block,
        # each token's experts are distinct, so the later blocks are all padding.
        EM = min(EM, A.size(0) * top_k * block_m)
    grid = (triton.cdiv(EM, block_m) * triton.cdiv(N, block_n),)

    _fused_moe_int8_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        A.size(0) * top_k,
        A.size(0),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale.ndim == 2 else 1,
        B_scale.stride(0),
        B_scale.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=config.get("GROUP_SIZE_M", 8),
        num_warps=config.get("num_warps", 8),
        num_stages=config.get("num_stages", 2),
    )


__all__ = ["can_run", "invoke"]
