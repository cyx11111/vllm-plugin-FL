# Copyright (c) 2026 BAAI. All rights reserved.

"""Fuse the routed experts' a2 activation quant into silu_and_mul.

Between its two GEMMs, ``fused_experts_impl`` activates gemm1's output into
``intermediate_cache2`` and then immediately quantizes it per token; only
the int8 copy reaches gemm2, so the bf16 buffer exists purely to carry one
value from one kernel to the next. ``impl/int8/silu_and_mul_quant.py``
does both in one kernel.

The seam is awkward because the two halves live in different functions and
the one that could fuse them (the activation) runs before the one that
knows the quantization scheme. So the activation is *deferred* rather than
replaced: inside an experts call it records what it was asked to do and
returns, and the quantize that follows either consumes the record with the
fused kernel or, for any other scheme, runs the real activation first.
Scope exit settles a record nobody claimed.

``VLLM_FL_SUNRISE_BENCH_BASELINE_MOE_ACT_QUANT=1`` keeps the unfused pair.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import os
import sys
import threading

import torch

logger = logging.getLogger(__name__)

_PATCHED = False
# Set only while inside vLLM's ``fused_experts_impl``, where the activation and
# the quantize that consumes it are adjacent and nothing else reads the buffer
# in between. Deferring anywhere else would hand an unwritten tensor to whoever
# reads it next.
_STATE = threading.local()

_ORIG_ACTIVATION = None
_ORIG_QUANTIZE = None
_QUANTIZE_SIG = None


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _disabled() -> bool:
    return _env_on("VLLM_FL_SUNRISE_BENCH_BASELINE_MOE_ACT_QUANT")


def _flush() -> None:
    """Run the activation we deferred, for a consumer that cannot use the fusion."""
    deferred = getattr(_STATE, "deferred", None)
    if deferred is None:
        return
    _STATE.deferred = None
    activation, output, inp = deferred
    _ORIG_ACTIVATION(activation, output, inp)


@contextlib.contextmanager
def _fusion_scope():
    previous_active = getattr(_STATE, "active", False)
    previous_deferred = getattr(_STATE, "deferred", None)
    _STATE.active = True
    _STATE.deferred = None
    try:
        yield
    finally:
        _flush()
        _STATE.active = previous_active
        _STATE.deferred = previous_deferred


def _deferrable(activation, output, inp) -> bool:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from ..impl.int8.silu_and_mul_quant import is_fusable, is_profitable

    return (
        activation == MoEActivation.SILU
        and output.dim() == 2
        and is_fusable(inp)
        and output.size(-1) * 2 == inp.size(-1)
        and is_profitable(inp)
    )


def _apply_moe_activation(activation, output, inp):
    if getattr(_STATE, "active", False) and _deferrable(activation, output, inp):
        # Two deferrals cannot overlap: the second would strand the first.
        _flush()
        _STATE.deferred = (activation, output, inp)
        return output
    return _ORIG_ACTIVATION(activation, output, inp)


def _moe_kernel_quantize_input(*args, **kwargs):
    deferred = getattr(_STATE, "deferred", None)
    if deferred is None:
        return _ORIG_QUANTIZE(*args, **kwargs)

    bound = _QUANTIZE_SIG.bind(*args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments["A"]
    _, output, inp = deferred
    if (
        a is output
        and bound.arguments["quant_dtype"] is torch.int8
        and bound.arguments["per_act_token_quant"]
        and bound.arguments["block_shape"] is None
        and bound.arguments.get("ocp_mx_scheme") is None
        and not bound.arguments.get("quantization_emulation", False)
    ):
        from ..impl.int8.silu_and_mul_quant import silu_and_mul_quant_int8

        result = silu_and_mul_quant_int8(inp)
        # Only now: a throw above leaves the record for the scope to settle.
        _STATE.deferred = None
        return result

    _flush()
    return _ORIG_QUANTIZE(*args, **kwargs)


def _rebind(module, name, original, replacement) -> None:
    setattr(module, name, replacement)
    for other in list(sys.modules.values()):
        if other is None:
            continue
        if getattr(other, name, None) is original:
            setattr(other, name, replacement)


def apply_patch() -> bool:
    """Defer the MoE activation so the a2 quantize can absorb it."""
    global _PATCHED, _ORIG_ACTIVATION, _ORIG_QUANTIZE, _QUANTIZE_SIG
    if _PATCHED:
        return True
    if _disabled():
        logger.info(
            "moe-act-quant: fusion disabled by "
            "VLLM_FL_SUNRISE_BENCH_BASELINE_MOE_ACT_QUANT; the routed experts "
            "keep the separate silu_and_mul and per-token quant kernels."
        )
        return False

    try:
        import vllm.model_executor.layers.fused_moe.activation as vllm_act
        import vllm.model_executor.layers.fused_moe.fused_moe as vllm_moe
        import vllm.model_executor.layers.fused_moe.utils as vllm_utils
    except Exception as exc:  # noqa: BLE001
        logger.debug("moe-act-quant: vLLM fused_moe modules unavailable (%s)", exc)
        return False

    activation = getattr(vllm_act, "apply_moe_activation", None)
    quantize = getattr(vllm_utils, "moe_kernel_quantize_input", None)
    experts_impl = getattr(vllm_moe, "fused_experts_impl", None)
    if activation is None or quantize is None or experts_impl is None:
        logger.debug("moe-act-quant: hooks unavailable; skip.")
        return False
    if getattr(activation, "_fl_moe_act_quant", False):
        _PATCHED = True
        return True

    _ORIG_ACTIVATION = activation
    _ORIG_QUANTIZE = quantize
    _QUANTIZE_SIG = inspect.signature(quantize)

    @functools.wraps(experts_impl)
    def _fused_experts_impl(*args, **kwargs):
        with _fusion_scope():
            return experts_impl(*args, **kwargs)

    for wrapper, wrapped in (
        (_apply_moe_activation, activation),
        (_moe_kernel_quantize_input, quantize),
        (_fused_experts_impl, experts_impl),
    ):
        wrapper._fl_moe_act_quant = True  # type: ignore[attr-defined]
        wrapper._fl_original = wrapped  # type: ignore[attr-defined]

    _rebind(vllm_act, "apply_moe_activation", activation, _apply_moe_activation)
    _rebind(
        vllm_utils,
        "moe_kernel_quantize_input",
        quantize,
        _moe_kernel_quantize_input,
    )
    # Only the module that owns the adjacency, so an experts call reached any
    # other way keeps the stock activation.
    vllm_moe.fused_experts_impl = _fused_experts_impl

    _PATCHED = True
    logger.info(
        "moe-act-quant: routed experts now quantize a2 inside silu_and_mul "
        "(one kernel instead of two, no bf16 round trip)."
    )
    return True


__all__ = ["apply_patch"]
