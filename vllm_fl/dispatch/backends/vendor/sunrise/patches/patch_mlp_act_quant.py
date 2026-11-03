# Copyright (c) 2026 BAAI. All rights reserved.

"""Fuse a gated MLP's INT8 activation quant into silu_and_mul.

``patch_moe_act_quant`` did this for routed experts. The same pair still
runs unfused in every gated MLP: ``act_fn`` writes BF16, ``down_proj``
reads it back to quantize. ``impl/int8/silu_and_mul_quant.py`` does both
in one pass. The class is reused as the shared expert of MoE layers and
as the dense MLP, so one seam covers both.

The activation runs before anything knows the quantization scheme, so it
is deferred rather than replaced. Inside an MLP forward ``SiluAndMul``
records the request and returns the buffer unwritten; the following
quantize either consumes that record with the fused kernel or, for any
other scheme, runs the real activation first. Scope exit settles a
record nobody claimed. The buffer is the one the activation would have
allocated anyway.

Deferring is confined to that forward: adjacency is guaranteed only
there. On a BF16 model ``down_proj`` would read the unwritten buffer, so
each module probes: the first forward stays unfused and watches whether
quantize actually consumes the activation; only then does later work
defer. Unknown schemes fail safe.

``VLLM_FL_SUNRISE_BENCH_BASELINE_MLP_ACT_QUANT=1`` keeps the unfused pair.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import sys
import threading

import torch

logger = logging.getLogger(__name__)

_PATCHED = False
# Set only while inside a wrapped MLP forward. Deferring anywhere else would
# hand an unwritten tensor to whoever reads it next.
_STATE = threading.local()

_ORIG_SILU_FORWARD = None
_ORIG_QUANT = None

# The MLP classes whose forward has the gate_up / act / down shape above. Each
# is (module path, class name); a missing one is skipped rather than fatal, so
# this list can name classes a given vLLM does not ship.
_MLP_CLASSES = (
    ("vllm.model_executor.models.qwen2_moe", "Qwen2MoeMLP"),
)


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _disabled() -> bool:
    return _env_on("VLLM_FL_SUNRISE_BENCH_BASELINE_MLP_ACT_QUANT")


def _flush() -> None:
    """Run the activation we deferred, for a consumer that cannot use the fusion."""
    deferred = getattr(_STATE, "deferred", None)
    if deferred is None:
        return
    _STATE.deferred = None
    module, inp, output = deferred
    output.copy_(_ORIG_SILU_FORWARD(module, inp))


@contextlib.contextmanager
def _scope(mlp, *, probing: bool):
    previous = (
        getattr(_STATE, "active", False),
        getattr(_STATE, "deferred", None),
        getattr(_STATE, "probe", None),
    )
    _STATE.active = not probing
    _STATE.deferred = None
    # [module, activation output seen, consumed] while probing.
    _STATE.probe = [mlp, None, False] if probing else None
    try:
        yield
    finally:
        _flush()
        probe = _STATE.probe
        if probe is not None:
            mlp._fl_mlp_fuses = probe[2]
        _STATE.active, _STATE.deferred, _STATE.probe = previous


def _deferrable(inp: torch.Tensor) -> bool:
    from ..impl.int8.silu_and_mul_quant import is_fusable, is_profitable

    return is_fusable(inp) and is_profitable(inp)


def _silu_and_mul_forward(self, x, *args, **kwargs):
    plain = not args and not kwargs
    if plain and getattr(_STATE, "active", False) and _deferrable(x):
        # Two deferrals cannot overlap: the second would strand the first.
        _flush()
        out = torch.empty(
            (*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device
        )
        _STATE.deferred = (self, x, out)
        return out

    out = _ORIG_SILU_FORWARD(self, x, *args, **kwargs)
    probe = getattr(_STATE, "probe", None)
    if probe is not None and plain:
        # Watch the last activation of the forward; if the quantize takes it,
        # deferring it next time is safe.
        probe[1] = out
    return out


def _scaled_int8_quant(input, scale=None, azp=None, symmetric=True):
    # ``scale is None`` is dynamic per-token, which is what the fused kernel
    # computes; a static or asymmetric scheme is somebody else's arithmetic.
    consumable = scale is None and symmetric

    deferred = getattr(_STATE, "deferred", None)
    if deferred is not None:
        _, inp, output = deferred
        if input is output and consumable:
            from ..impl.int8.silu_and_mul_quant import silu_and_mul_quant_int8

            q, s = silu_and_mul_quant_int8(inp)
            # Only now: a throw above leaves the record for the scope to settle.
            _STATE.deferred = None
            return q, s, None
        _flush()

    probe = getattr(_STATE, "probe", None)
    if probe is not None and probe[1] is input and consumable:
        probe[2] = True
    return _ORIG_QUANT(input, scale, azp, symmetric)


def _rebind(module, name, original, replacement) -> None:
    setattr(module, name, replacement)
    for other in list(sys.modules.values()):
        if other is None:
            continue
        if getattr(other, name, None) is original:
            setattr(other, name, replacement)


def _wrap_mlp_forwards() -> int:
    """Put a fusion scope around every gated MLP forward we know the shape of."""
    import importlib

    wrapped = 0
    for module_path, class_name in _MLP_CLASSES:
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("mlp-act-quant: %s unavailable (%s)", module_path, exc)
            continue
        cls = getattr(module, class_name, None)
        if cls is None or getattr(cls.forward, "_fl_mlp_act_quant", False):
            continue
        original = cls.forward

        @functools.wraps(original)
        def _forward(self, *args, __original=original, **kwargs):
            fuses = getattr(self, "_fl_mlp_fuses", None)
            if fuses is False:
                return __original(self, *args, **kwargs)
            with _scope(self, probing=fuses is None):
                return __original(self, *args, **kwargs)

        _forward._fl_mlp_act_quant = True  # type: ignore[attr-defined]
        _forward._fl_original = original  # type: ignore[attr-defined]
        cls.forward = _forward
        wrapped += 1
    return wrapped


def apply_patch() -> bool:
    """Defer the gated MLP activation so its INT8 quantize can absorb it."""
    global _PATCHED, _ORIG_SILU_FORWARD, _ORIG_QUANT
    if _PATCHED:
        return True
    if _disabled():
        logger.info(
            "mlp-act-quant: fusion disabled by "
            "VLLM_FL_SUNRISE_BENCH_BASELINE_MLP_ACT_QUANT; gated MLPs keep the "
            "separate silu_and_mul and per-token quant kernels."
        )
        return False

    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.activation import SiluAndMul
    except Exception as exc:  # noqa: BLE001
        logger.debug("mlp-act-quant: vLLM modules unavailable (%s)", exc)
        return False

    quantize = getattr(ops, "scaled_int8_quant", None)
    if quantize is None:
        logger.debug("mlp-act-quant: scaled_int8_quant unavailable; skip.")
        return False
    if getattr(quantize, "_fl_mlp_act_quant", False):
        _PATCHED = True
        return True

    wrapped = _wrap_mlp_forwards()
    if not wrapped:
        logger.debug("mlp-act-quant: no known gated MLP class; skip.")
        return False

    # CustomOp resolves its forward at construction, but ``forward`` itself is
    # looked up on the class per call, so wrapping it catches every instance
    # whichever implementation was dispatched.
    _ORIG_SILU_FORWARD = SiluAndMul.forward
    _ORIG_QUANT = quantize
    for wrapper in (_silu_and_mul_forward, _scaled_int8_quant):
        wrapper._fl_mlp_act_quant = True  # type: ignore[attr-defined]
    _silu_and_mul_forward._fl_original = _ORIG_SILU_FORWARD  # type: ignore
    _scaled_int8_quant._fl_original = _ORIG_QUANT  # type: ignore

    SiluAndMul.forward = _silu_and_mul_forward
    _rebind(ops, "scaled_int8_quant", quantize, _scaled_int8_quant)

    _PATCHED = True
    logger.info(
        "mlp-act-quant: gated MLPs now quantize the down_proj activation "
        "inside silu_and_mul (one kernel instead of two, no bf16 round trip); "
        "%d MLP class(es) wrapped.",
        wrapped,
    )
    return True


__all__ = ["apply_patch"]
