# Copyright (c) 2026 BAAI. All rights reserved.

"""Stable dtype adapters for the PTPU fused GDN decode kernel.

The native kernel selects one scalar template from ``a.dtype`` and applies it
to ``A_log`` and ``dt_bias`` too. Qwen stores those parameters as float32 while
decode activations are bfloat16. Passing them directly makes the kernel
reinterpret float32 storage as bfloat16, corrupting every recurrent update.
"""

from __future__ import annotations

import torch


_PARAM_CACHE: dict[
    tuple[int, int, torch.device, torch.dtype],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def cast_gdn_params_for_ptpu(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    activation_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stable parameter buffers matching the native activation dtype."""
    if A_log.dtype == activation_dtype and dt_bias.dtype == activation_dtype:
        return A_log, dt_bias
    if A_log.device != dt_bias.device:
        raise ValueError(
            f"GDN parameter device mismatch: {A_log.device} != {dt_bias.device}"
        )

    key = (
        A_log.data_ptr(),
        dt_bias.data_ptr(),
        A_log.device,
        activation_dtype,
    )
    cached = _PARAM_CACHE.get(key)
    if cached is not None and cached[0] is A_log and cached[1] is dt_bias:
        return cached[2], cached[3]

    cast_A_log = A_log.to(dtype=activation_dtype).contiguous()
    cast_dt_bias = dt_bias.to(dtype=activation_dtype).contiguous()
    _PARAM_CACHE[key] = (A_log, dt_bias, cast_A_log, cast_dt_bias)
    return cast_A_log, cast_dt_bias
