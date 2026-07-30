# Copyright (c) 2026 BAAI. All rights reserved.

"""Enable vLLM native INT8 (compressed-tensors W8A8) on Sunrise/PTPU."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ENABLED = False
_OOT_KERNEL_REGISTERED = False
_QUANT_PATCHED = False
_MOE_ORACLE_PATCHED = False
_MOE_QUANT_CFG_PATCHED = False
_MOE_ACT_QUANT_PATCHED = False
_MM_PATCHED = False
_FG_MM_FN = None
_MM_SHAPE_LOGGED = False


def _register_oot_int8_kernel() -> bool:
    """Register a Triton INT8 ScaledMM kernel that is ``is_supported`` on OOT."""
    global _OOT_KERNEL_REGISTERED
    if _OOT_KERNEL_REGISTERED:
        return True
    try:
        from vllm.platforms import PlatformEnum
        from vllm.model_executor.kernels.linear import (
            _POSSIBLE_INT8_KERNELS,
            register_linear_kernel,
        )
        from vllm.model_executor.kernels.linear.scaled_mm.triton import (
            TritonInt8ScaledMMLinearKernel,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("native-int8: vLLM kernel registry unavailable (%s)", e)
        return False

    class OOTTritonInt8ScaledMMLinearKernel(TritonInt8ScaledMMLinearKernel):
        """``TritonInt8ScaledMMLinearKernel`` allowed on the OOT platform.

        Compute is unchanged (pure-Triton ``triton_scaled_mm`` + the patched
        activation quant); only platform gating is relaxed.
        """

        @classmethod
        def is_supported(cls, compute_capability=None):
            return True, None

        @classmethod
        def can_implement(cls, c):
            return True, None

    existing = _POSSIBLE_INT8_KERNELS.get(PlatformEnum.OOT, [])
    if any(k.__name__ == OOTTritonInt8ScaledMMLinearKernel.__name__ for k in existing):
        _OOT_KERNEL_REGISTERED = True
        return True

    register_linear_kernel(
        OOTTritonInt8ScaledMMLinearKernel, PlatformEnum.OOT, kernel_type="int8"
    )
    _OOT_KERNEL_REGISTERED = True
    logger.info(
        "native-int8: registered OOTTritonInt8ScaledMMLinearKernel for "
        "PlatformEnum.OOT."
    )
    return True


def _patch_scaled_int8_quant() -> bool:
    """Route ``vllm._custom_ops.scaled_int8_quant`` to sunrise Triton impl."""
    global _QUANT_PATCHED
    if _QUANT_PATCHED:
        return True
    try:
        import vllm._custom_ops as _vllm_ops
    except Exception as e:  # noqa: BLE001
        logger.debug("native-int8: vllm._custom_ops unavailable (%s)", e)
        return False

    from ..impl.int8.scaled_int8_quant import scaled_int8_quant as _sunrise_quant

    _orig = getattr(_vllm_ops, "scaled_int8_quant", None)

    def _fl_scaled_int8_quant(input, scale=None, azp=None, symmetric=True):
        return _sunrise_quant(input, scale=scale, azp=azp, symmetric=symmetric)

    _fl_scaled_int8_quant._fl_original = _orig  # type: ignore[attr-defined]
    _vllm_ops.scaled_int8_quant = _fl_scaled_int8_quant
    _QUANT_PATCHED = True
    logger.info(
        "native-int8: patched vllm._custom_ops.scaled_int8_quant -> "
        "sunrise impl/int8 scaled_int8_quant (Triton per-token)."
    )
    return True


def _resolve_fg_scaled_mm():
    """Return FlagGems' autotuned INT8 ``scaled_mm`` (cached)."""
    global _FG_MM_FN
    if _FG_MM_FN is not None:
        return _FG_MM_FN
    from flag_gems.ops.scaled_mm import scaled_mm as _fg
    _FG_MM_FN = _fg
    return _fg


def _patch_triton_scaled_mm() -> bool:
    """Route vLLM's ``triton_scaled_mm`` to FlagGems' autotuned ``scaled_mm``."""
    global _MM_PATCHED
    if _MM_PATCHED:
        return True
    try:
        import vllm.model_executor.kernels.linear.scaled_mm.triton as _mm_mod
    except Exception as e:  # noqa: BLE001
        logger.debug("native-int8: vLLM triton scaled_mm module unavailable (%s)", e)
        return False

    _orig = getattr(_mm_mod, "triton_scaled_mm", None)
    if _orig is None:
        logger.debug("native-int8: triton_scaled_mm symbol not found; skip.")
        return False
    if getattr(_orig, "_fl_native_int8_mm", False):
        _MM_PATCHED = True
        return True

    def _fl_triton_scaled_mm(
        input, weight, scale_a, scale_b, out_dtype, bias=None, *args, **kwargs
    ):
        global _MM_SHAPE_LOGGED
        fg = _resolve_fg_scaled_mm()

        # Vision encoder QKV (and any batched linear) may pass [..., K].
        # FlagGems / vLLM triton_scaled_mm only accept 2D [M, K]; CUDA
        # cutlass_scaled_mm flattens the same way before the GEMM.
        input_shape = input.shape
        if input.ndim != 2:
            input = input.reshape(-1, input_shape[-1])

        # FlagGems computes input[M,K] @ mat2[K,N]; orient weight to [K,N].
        mat2 = weight
        transposed = False
        if mat2.shape[0] != input.shape[1]:
            mat2 = mat2.t()
            transposed = True

        # FlagGems right-scale accepts scalar / [N] / [1,N] (not [N,1]).
        sb = scale_b
        if sb is not None and sb.ndim == 2 and sb.shape[-1] == 1 and sb.numel() > 1:
            sb = sb.reshape(-1)

        if not _MM_SHAPE_LOGGED:
            _MM_SHAPE_LOGGED = True
            logger.info(
                "native-int8: triton_scaled_mm->FlagGems scaled_mm active. "
                "input=%s (2d=%s) weight=%s (transposed=%s) scale_a=%s "
                "scale_b=%s out_dtype=%s bias=%s",
                tuple(input_shape), tuple(input.shape),
                tuple(weight.shape), transposed,
                tuple(scale_a.shape), tuple(scale_b.shape) if scale_b is not None
                else None, out_dtype, None if bias is None else tuple(bias.shape),
            )

        try:
            out = fg(input, mat2, scale_a, sb, bias=bias, out_dtype=out_dtype)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "native-int8: FlagGems scaled_mm failed (%s); falling back to "
                "vLLM triton_scaled_mm.", e,
            )
            out = _orig(input, weight, scale_a, scale_b, out_dtype, bias,
                        *args, **kwargs)

        if len(input_shape) != 2:
            out = out.view(*input_shape[:-1], out.shape[-1])
        return out

    _fl_triton_scaled_mm._fl_native_int8_mm = True  # type: ignore[attr-defined]
    _fl_triton_scaled_mm._fl_original = _orig  # type: ignore[attr-defined]

    # Patch the defining module, then rebind any module that already did
    # ``from ...scaled_mm.triton import triton_scaled_mm``.
    import sys

    _mm_mod.triton_scaled_mm = _fl_triton_scaled_mm
    for _mod in list(sys.modules.values()):
        if _mod is None:
            continue
        if getattr(_mod, "triton_scaled_mm", None) is _orig:
            _mod.triton_scaled_mm = _fl_triton_scaled_mm

    _MM_PATCHED = True
    logger.info(
        "native-int8: patched triton_scaled_mm -> FlagGems scaled_mm "
        "(autotuned INT8 GEMM) on sunrise."
    )
    return True


def _patch_int8_moe_oracle() -> bool:
    """Make the int8 MoE oracle return ``TritonExpertsFL`` on OOT."""
    global _MOE_ORACLE_PATCHED
    if _MOE_ORACLE_PATCHED:
        return True
    try:
        import sys

        from vllm.platforms import current_platform
        import vllm.model_executor.layers.fused_moe.oracle.int8 as _int8_oracle
        from vllm.model_executor.layers.fused_moe.oracle.int8 import Int8MoeBackend
        from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL
        from vllm_fl.utils import use_flaggems
    except Exception as e:  # noqa: BLE001
        logger.debug("native-int8: int8 MoE oracle unavailable (%s)", e)
        return False

    _orig = _int8_oracle.select_int8_moe_backend
    if getattr(_orig, "_fl_native_int8_moe", False):
        _MOE_ORACLE_PATCHED = True
        return True

    def _select_int8_moe_backend_oot(config, *args, **kwargs):
        # OOT + flaggems: force the FlagGems-routed experts class. This mirrors
        # ``select_unquantized_moe_backend_oot`` and bypasses the CUDA/ROCm-only
        # ``is_supported_config`` gate in the stock oracle.
        if current_platform.is_out_of_tree() and use_flaggems():
            return Int8MoeBackend.TRITON, TritonExpertsFL
        return _orig(config, *args, **kwargs)

    _select_int8_moe_backend_oot._fl_native_int8_moe = True  # type: ignore[attr-defined]
    _select_int8_moe_backend_oot._fl_original = _orig  # type: ignore[attr-defined]

    # Patch the oracle module attribute (covers modules that import the symbol
    # AFTER this runs). Then also rebind any module that already did a
    # ``from ...oracle.int8 import select_int8_moe_backend`` (e.g. the
    # compressed-tensors int8 MoE method), regardless of its module path.
    _int8_oracle.select_int8_moe_backend = _select_int8_moe_backend_oot
    for _mod in list(sys.modules.values()):
        if _mod is None:
            continue
        if getattr(_mod, "select_int8_moe_backend", None) is _orig:
            _mod.select_int8_moe_backend = _select_int8_moe_backend_oot

    _MOE_ORACLE_PATCHED = True
    logger.info(
        "native-int8: patched select_int8_moe_backend -> TritonExpertsFL on OOT "
        "(compressed-tensors INT8 W8A8 per-channel MoE routes through FlagGems)."
    )
    return True


def _patch_int8_moe_quant_config() -> bool:
    """Force dynamic per-token MoE configs onto W8A8 (not W8A16).

    ``CompressedTensorsW8A8Int8MoEMethod`` sets ``w13_input_scale`` /
    ``w2_input_scale`` to ``None`` for dynamic token quant, then calls
    ``make_int8_moe_quant_config(..., per_act_token_quant=True)``. Upstream
    interprets any ``None`` activation scale as W8A16 (``quant_dtype is None``),
    so ``use_int8_w8a8`` stays False and experts run int8×bf16. Re-route that
    case to ``int8_w8a8_moe_quant_config``.
    """
    global _MOE_QUANT_CFG_PATCHED
    if _MOE_QUANT_CFG_PATCHED:
        return True
    try:
        import sys

        import vllm.model_executor.layers.fused_moe.oracle.int8 as _oracle
        from vllm.model_executor.layers.fused_moe.config import (
            int8_w8a8_moe_quant_config,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("native-int8: MoE quant-config patch unavailable (%s)", e)
        return False

    _orig = _oracle.make_int8_moe_quant_config
    if getattr(_orig, "_fl_native_int8_moe_qcfg", False):
        _MOE_QUANT_CFG_PATCHED = True
        return True

    def _make_int8_moe_quant_config(
        w1_scale,
        w2_scale,
        a1_scale=None,
        a2_scale=None,
        per_act_token_quant: bool = False,
    ):
        if per_act_token_quant and (a1_scale is None or a2_scale is None):
            return int8_w8a8_moe_quant_config(
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                per_act_token_quant=True,
            )
        return _orig(
            w1_scale,
            w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            per_act_token_quant=per_act_token_quant,
        )

    _make_int8_moe_quant_config._fl_native_int8_moe_qcfg = True  # type: ignore[attr-defined]
    _make_int8_moe_quant_config._fl_original = _orig  # type: ignore[attr-defined]
    _oracle.make_int8_moe_quant_config = _make_int8_moe_quant_config
    for _mod in list(sys.modules.values()):
        if _mod is None:
            continue
        if getattr(_mod, "make_int8_moe_quant_config", None) is _orig:
            _mod.make_int8_moe_quant_config = _make_int8_moe_quant_config

    _MOE_QUANT_CFG_PATCHED = True
    logger.info(
        "native-int8: patched make_int8_moe_quant_config so dynamic "
        "per-token MoE uses W8A8 (not W8A16)."
    )
    return True


def _patch_moe_per_token_quant_int8() -> bool:
    """Replace vLLM's CUDA-only MoE ``per_token_quant_int8`` with sunrise impl.

    Stock kernel uses ``tl.extra.cuda.libdevice.round`` → TANG reports
    ``kernel function contain unknown call`` / ``TANG_ERROR_INVALID_IMAGE``.
    Required once MoE runs true W8A8 (see ``_patch_int8_moe_quant_config``).
    """
    global _MOE_ACT_QUANT_PATCHED
    if _MOE_ACT_QUANT_PATCHED:
        return True
    try:
        import sys

        import vllm.model_executor.layers.quantization.utils.int8_utils as _iu
    except Exception as e:  # noqa: BLE001
        logger.debug("native-int8: int8_utils unavailable (%s)", e)
        return False

    from ..impl.int8.scaled_int8_quant import scaled_int8_quant as _sunrise_quant

    _orig = getattr(_iu, "per_token_quant_int8", None)
    if _orig is None:
        return False
    if getattr(_orig, "_fl_native_int8_moe_act", False):
        _MOE_ACT_QUANT_PATCHED = True
        return True

    def _fl_per_token_quant_int8(x):
        q, s, _azp = _sunrise_quant(x, scale=None, azp=None, symmetric=True)
        return q, s

    _fl_per_token_quant_int8._fl_native_int8_moe_act = True  # type: ignore[attr-defined]
    _fl_per_token_quant_int8._fl_original = _orig  # type: ignore[attr-defined]
    _iu.per_token_quant_int8 = _fl_per_token_quant_int8

    # Rebind modules that already did ``from ...int8_utils import per_token_quant_int8``
    # (notably ``vllm.model_executor.layers.fused_moe.utils``).
    for _mod in list(sys.modules.values()):
        if _mod is None:
            continue
        if getattr(_mod, "per_token_quant_int8", None) is _orig:
            _mod.per_token_quant_int8 = _fl_per_token_quant_int8

    _MOE_ACT_QUANT_PATCHED = True
    logger.info(
        "native-int8: patched per_token_quant_int8 -> sunrise "
        "scaled_int8_quant (MoE W8A8 activation quant on sunrise)."
    )
    return True


def enable_native_int8() -> None:
    """Enable the vLLM-native compressed-tensors INT8 path on sunrise/ptpu.

    Idempotent; safe to call at import time even for non-INT8 models.
    """
    global _ENABLED
    if _ENABLED:
        return
    ok_kernel = _register_oot_int8_kernel()
    ok_quant = _patch_scaled_int8_quant()
    ok_mm = _patch_triton_scaled_mm()
    ok_moe_act = _patch_moe_per_token_quant_int8()
    ok_moe_qcfg = _patch_int8_moe_quant_config()
    ok_moe = _patch_int8_moe_oracle()
    _ENABLED = (
        ok_kernel
        and ok_quant
        and ok_mm
        and ok_moe_act
        and ok_moe_qcfg
        and ok_moe
    )


__all__ = ["enable_native_int8"]
