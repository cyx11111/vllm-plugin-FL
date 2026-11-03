# Copyright (c) 2026 BAAI. All rights reserved.

"""Route BF16 dense linears to the vendor GEMM where it wins.

``torch.nn.functional.linear`` is the seam because that is where vLLM's
unquantized parallel linear layers actually land: ``UnquantizedLinearMethod``
calls it directly, and FlagGems cannot take it back because ``linear``, ``mm``
and ``addmm`` are all on ``VLLM_FL_FLAGOS_BLACKLIST`` in production. Patching
here rather than registering an ``aten::linear`` override keeps the decision in
Python and leaves every other matmul in the process untouched.

Why the work sits behind a custom op
------------------------------------
Not because production needs it. The serve entrypoint passes ``{"mode": 0,
"cudagraph_mode": "FULL_DECODE_ONLY"}``, and ``CompilationMode.NONE`` sets
``do_not_compile``, so the model runs eager and nothing traces it.

It is because the gate cannot survive being traced, and a traced config is one
flag away. Dynamo cannot enter ``deepgemm_ptpu``'s pybind entry points, so
asking the vendor whether it supports a shape aborts a fullgraph capture with
``Unsupported: Attempted to call function marked as skipped``; and the gate
wants ``M``, which is dynamic in a traced graph and so is not a number anything
could branch on there regardless.

So the wrapper decides only what is static -- dtype, device, rank, contiguity,
whether there is a bias -- all of which Dynamo evaluates and guards on happily.
Calls that clear those go to ``torch.ops.vllm.sunrise_bf16_linear``, which
Dynamo would record as an opaque call and which runs the real per-``M`` gate at
execution time, in eager Python, where the vendor can be asked things. Calls
that do not clear them are emitted as a plain ``aten::linear``.

The dispatch this adds is off the decode critical path either way: with
``FULL_DECODE_ONLY`` a decode step is replayed from a captured graph, where no
Python runs at all.

``patch_profile_decode`` also wraps ``F.linear``. The two compose in either
order -- each delegates to whatever it captured -- but this one is applied at
import time and the profiler installs after it, so the profiler's timings
include the routing decision, which is what you want when reading them.

``VLLM_FL_SUNRISE_BENCH_BASELINE_DENSE_BF16_GEMM=1`` makes the gate decline
everything, leaving ``aten::linear`` in place for an A/B baseline.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_PATCHED = False
_REGISTERED = False

# ``aten::linear`` as it was before this patch, called both when the static
# checks decline and when the runtime gate does.
_ORIG_LINEAR = None


def _impl(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The vendor GEMM if the runtime gate takes this shape, else torch."""
    from ..impl.bf16 import dense_gemm

    try:
        out = dense_gemm.try_linear(input, weight, None)
    except Exception:  # noqa: BLE001
        # A vendor GEMM that raises mid-forward must not take the model down;
        # the incumbent is always able to serve the same call.
        logger.warning(
            "dense-bf16: vendor GEMM raised, falling back to aten::linear "
            "for this shape and every later one",
            exc_info=True,
        )
        dense_gemm.disable("raised during forward")
        out = None
    if out is not None:
        return out
    return _ORIG_LINEAR(input, weight, None)


def _fake(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Output metadata for the trace, without running either kernel."""
    return torch.empty(
        (*input.shape[:-1], weight.shape[0]),
        dtype=input.dtype,
        device=input.device,
    )


def _register() -> bool:
    global _REGISTERED
    if _REGISTERED:
        return True
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op("sunrise_bf16_linear", _impl, fake_impl=_fake)
    _REGISTERED = True
    return True


def _static_ok(input: torch.Tensor, weight: torch.Tensor, bias) -> bool:
    """The part of the decision Dynamo can make while tracing.

    Everything here is a dtype, a rank, a stride or a static shape, so it is
    resolved at trace time and guarded. Nothing here touches the vendor
    library; ``M`` is deliberately not consulted, because it is dynamic.
    """
    # The vendor GEMM takes no bias; shipped projections are bias-free.
    if bias is not None:
        return False
    if input.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return False
    # The op is registered on the accelerator dispatch key only, so a host-side
    # linear -- and there are some, outside the model forward -- would fail in
    # the dispatcher, before ``_impl`` and its fallback can be reached.
    if input.is_cpu or input.device != weight.device:
        return False
    if weight.dim() != 2 or input.dim() < 2:
        return False
    # ``bf16_gemm_nt(A[M, K], B[N, K])`` is ``A @ B.t()``, which is what
    # ``F.linear`` computes with vLLM's [out_features, in_features] weight --
    # but only if both operands are K-major.
    if not weight.is_contiguous():
        return False
    if input.stride(-1) != 1:
        return False
    return input.shape[-1] == weight.shape[1]


def apply_patch() -> bool:
    """Wrap ``torch.nn.functional.linear`` with the vendor GEMM dispatch."""
    global _PATCHED, _ORIG_LINEAR
    if _PATCHED:
        return True

    from ..impl.bf16 import dense_gemm

    if not dense_gemm.available():
        logger.info(
            "dense-bf16: deepgemm_ptpu not installed; BF16 linears keep "
            "aten::linear."
        )
        return False

    import torch.nn.functional as F

    orig = F.linear
    if getattr(orig, "__sunrise_bf16_gemm__", False):
        _PATCHED = True
        return True

    if not _register():
        return False

    _ORIG_LINEAR = orig
    vendor = torch.ops.vllm.sunrise_bf16_linear

    def _linear(input, weight, bias=None):
        if _static_ok(input, weight, bias):
            return vendor(input, weight)
        return orig(input, weight, bias)

    _linear.__sunrise_bf16_gemm__ = True  # type: ignore[attr-defined]
    F.linear = _linear
    # The symbol is re-exported from torch.nn.functional and may already be
    # bound at a call site, so replace it in both places, as the profiler does.
    torch.nn.functional.linear = _linear

    _PATCHED = True
    logger.info(
        "dense-bf16: BF16 linears routed to deepgemm_ptpu where profitable."
    )
    return True
