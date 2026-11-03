# Copyright (c) 2026 BAAI. All rights reserved.

"""Fused-MoE tile config for Sunrise.

Use FlagGems-vllm's embedded autotune table when the current device is in
it. PT200 has none; looking it up only spams a missing-config warning and
then uses an NVIDIA-ish default. In that case keep FlagGems'
``_sunrise_get_default_config`` (the PTPU-safe pin).

Two parts of that pin are replaced where the information exists:

* ``try_get_optimal_moe_config`` sets ``BLOCK_SIZE_M``, which the two
  expert GEMMs must share. The two kernels want opposite things, so INT8
  and BF16 take different bands; see ``impl/moe_block_m.py``. The seam
  cannot tell the dtypes apart, so it asks whether an INT8 tile plan is
  in scope for the layer.
* ``invoke_fused_moe_triton_kernel`` sets per-stage N/K. Only INT8 W8A8
  goes through this seam; it gets tiles from ``impl/int8/moe_tiles.py``
  and the launch from ``impl/int8/fused_moe_int8.py``. Anything that
  kernel does not implement is skipped per launch.

BF16 experts reach the GEMM through vllm_fl's own dispatch rather than
vLLM's ``fused_experts_impl``, so they get the ``BLOCK_SIZE_M`` rule but
not per-stage N/K.

``VLLM_FL_SUNRISE_BENCH_BASELINE_INT8_MOE_TILES=1`` restores the pinned
N/K (and vLLM's kernel, since the substitution rides on the tile plan);
``VLLM_FL_SUNRISE_BENCH_BASELINE_INT8_MOE_KERNEL=1`` keeps the tiles but
returns the launch to vLLM; ``VLLM_FL_SUNRISE_BENCH_BASELINE_MOE_BLOCK_M=1``
restores the pinned ``BLOCK_SIZE_M``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

_PATCHED = False
_HAS_FGV_TABLE: bool | None = None
_INT8_TILES_PATCHED = False
# Set only while inside vLLM's INT8 ``fused_experts_impl``; the config picker is
# called from there and cannot otherwise tell INT8 from BF16, because
# ``_get_config_dtype_str`` ignores ``use_int8_w8a8`` and returns None for both.
_INT8_SCOPE = threading.local()
# gemm1 (w13) and gemm2 (w2) deliberately share one config: BLOCK_SIZE_M also
# sizes the moe_align_block_size padding that both GEMMs index into, so they
# cannot disagree on it. Picking once matches what the callers already did.
_GEMM_STAGE = "gemm1"


def has_flaggems_vllm_embedded_moe_table() -> bool:
    """Whether FlagGems-vllm shipped a non-empty MoE autotune table for this GPU."""
    global _HAS_FGV_TABLE
    if _HAS_FGV_TABLE is not None:
        return _HAS_FGV_TABLE

    try:
        from flaggems_vllm.runtime.backend._sunrise.ops.fused_moe import (
            _get_device_name,
            get_embedded_moe_configs,
        )

        tables, fallback = get_embedded_moe_configs()
        name = _get_device_name()
        if name in tables and tables[name]:
            _HAS_FGV_TABLE = True
            return True
        mapped = fallback.get(name) if fallback else None
        _HAS_FGV_TABLE = bool(mapped and mapped in tables and tables[mapped])
    except Exception:  # noqa: BLE001
        _HAS_FGV_TABLE = False
    return _HAS_FGV_TABLE


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _int8_tiles_disabled() -> bool:
    return _env_on("VLLM_FL_SUNRISE_BENCH_BASELINE_INT8_MOE_TILES")


def _block_m_disabled() -> bool:
    return _env_on("VLLM_FL_SUNRISE_BENCH_BASELINE_MOE_BLOCK_M")


def _int8_kernel_disabled() -> bool:
    return _env_on("VLLM_FL_SUNRISE_BENCH_BASELINE_INT8_MOE_KERNEL")


@contextlib.contextmanager
def _int8_scope(tiles):
    """Publish the tile plan for the INT8 experts call we are inside of.

    ``tiles`` is ``(w1_shape, w2_shape, gemm1_tile, gemm2_tile)`` or ``None``.
    """
    previous = getattr(_INT8_SCOPE, "tiles", None)
    _INT8_SCOPE.tiles = tiles
    try:
        yield
    finally:
        _INT8_SCOPE.tiles = previous


def _int8_scope_tiles():
    return getattr(_INT8_SCOPE, "tiles", None)


def _forced_config():
    from vllm.model_executor.layers.fused_moe import get_config

    return get_config()


def _sunrise_moe_config_flaggems(w1_shape, w2_shape, top_k, dtype, M, block_shape=None):
    """FlagGems sunrise picker (PTPU-safe default when no autotune table)."""
    from flag_gems.fused import fused_moe as gems_moe
    from flag_gems.runtime.backend._sunrise.fused.fused_moe import (
        _sunrise_moe_config_patch,
    )

    override = _forced_config()
    if override:
        return override

    with _sunrise_moe_config_patch():
        return gems_moe.try_get_optimal_moe_config(
            w1_shape,
            w2_shape,
            top_k,
            dtype,
            M,
            w1_shape[0],
            block_shape=block_shape,
            gemm_stage=_GEMM_STAGE,
        )


def _sunrise_moe_config_flaggems_vllm(
    w1_shape, w2_shape, top_k, dtype, M, block_shape=None
):
    """FlagGems-vllm picker; only used when an embedded table exists."""
    from flaggems_vllm.runtime.backend._sunrise.ops.fused_moe import (
        try_get_optimal_moe_config as fgv_try,
    )

    override = _forced_config()
    if override:
        return override

    return fgv_try(
        w1_shape,
        w2_shape,
        top_k,
        dtype,
        M,
        w1_shape[0],
        block_shape=block_shape,
        gemm_stage=_GEMM_STAGE,
    )


def _plan_is_for(w1_shape, w2_shape) -> bool:
    """Whether the INT8 tile plan in scope belongs to the layer being configured."""
    tiles = _int8_scope_tiles()
    if tiles is None:
        return False
    return tuple(tiles[0]) == tuple(w1_shape) and tuple(tiles[1]) == tuple(w2_shape)


def _with_block_size_m(config, w1_shape, w2_shape, top_k, M):
    """Replace the pinned ``BLOCK_SIZE_M`` with the one for this layer."""
    if _block_m_disabled() or _forced_config():
        return config
    if len(w1_shape) != 3:
        return config

    from ..impl import moe_block_m

    rows = moe_block_m.rows_per_expert(M, top_k, int(w1_shape[0]))
    if _plan_is_for(w1_shape, w2_shape):
        # Same batch axis as BF16 but different bands: the two kernels
        # disagree about how tall a block they want. High-E, wide-expert
        # geometry adds one prefill-only band; narrower layers keep the
        # general two-band rule.
        from ..impl.int8 import moe_tiles

        use_wide_prefill_block = (
            moe_tiles.is_wide_expert(tuple(w1_shape), tuple(w2_shape))
            and int(w2_shape[2]) >= 2048
        )
        block_m = moe_block_m.block_size_m_int8(
            rows, use_wide_prefill_block=use_wide_prefill_block
        )
    else:
        block_m = moe_block_m.block_size_m(rows)
    if config.get("BLOCK_SIZE_M") == block_m:
        return config
    return {**config, "BLOCK_SIZE_M": block_m}


def _patch_int8_moe_tiles() -> bool:
    """Give the INT8 expert GEMMs their own tiles without touching BF16.

    ``fused_experts_impl`` is the only place that knows the run is INT8 *before*
    the config is picked, and ``invoke_fused_moe_triton_kernel`` is the only
    place that knows which of the two GEMMs is about to launch.
    """
    global _INT8_TILES_PATCHED
    if _INT8_TILES_PATCHED:
        return True
    if _int8_tiles_disabled():
        logger.info(
            "moe-config: INT8 MoE tiles disabled by "
            "VLLM_FL_SUNRISE_BENCH_BASELINE_INT8_MOE_TILES; keeping the "
            "FlagGems 32/64/64 pin for the W8A8 expert GEMMs."
        )
        return False

    import inspect

    import vllm.model_executor.layers.fused_moe.fused_moe as vllm_moe

    from ..impl.int8 import deepgemm_moe, fused_moe_int8, moe_tiles

    experts_impl = getattr(vllm_moe, "fused_experts_impl", None)
    invoke = getattr(vllm_moe, "invoke_fused_moe_triton_kernel", None)
    if experts_impl is None or invoke is None:
        logger.debug("moe-config: INT8 MoE tile hooks unavailable; skip.")
        return False
    if getattr(experts_impl, "_fl_int8_moe_tiles", False):
        _INT8_TILES_PATCHED = True
        return True

    experts_sig = inspect.signature(experts_impl)
    invoke_sig = inspect.signature(invoke)

    from ..impl import moe_block_m

    def _fused_experts_impl(*args, **kwargs):
        bound = experts_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments.get("use_int8_w8a8", False):
            # The vendor grouped GEMM needs the activation physically grouped by
            # expert, which vLLM's dataflow does not provide, so it replaces the
            # whole expert path rather than one kernel. It declines whatever it
            # does not implement -- including a capture its scratch pool cannot
            # cover -- and the Triton dataflow below takes over.
            out = deepgemm_moe.try_fused_experts(bound.arguments)
            if out is not None:
                return out

        tiles = None
        if bound.arguments.get("use_int8_w8a8", False):
            w1 = bound.arguments["w1"]
            w1_shape = tuple(w1.shape)
            w2_shape = tuple(bound.arguments["w2"].shape)
            # The deep-K stage's tile depends on the batch, and this is where the
            # batch is known: fused_experts_impl picks one config for the whole
            # call, so the routed rows per expert are fixed for both stages.
            topk_ids = bound.arguments["topk_ids"]
            num_experts = bound.arguments.get("global_num_experts", -1)
            if num_experts is None or num_experts <= 0:
                num_experts = w1_shape[0]
            rows = moe_block_m.rows_per_expert(
                int(topk_ids.shape[0]), int(topk_ids.shape[1]), int(num_experts)
            )
            stage_tiles = moe_tiles.plan(w1_shape, w2_shape, rows)
            if stage_tiles is not None:
                tiles = (w1_shape, w2_shape, *stage_tiles)
        with _int8_scope(tiles):
            return experts_impl(*args, **kwargs)

    def _invoke_fused_moe_triton_kernel(*args, **kwargs):
        tiles = _int8_scope_tiles()
        if tiles is None:
            return invoke(*args, **kwargs)

        bound = invoke_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        w1_shape, w2_shape, tile1, tile2 = tiles
        a = bound.arguments
        b_shape = tuple(a["B"].shape)
        if b_shape == w1_shape:
            tile = tile1
        elif b_shape == w2_shape:
            tile = tile2
        else:
            return invoke(*args, **kwargs)

        config = moe_tiles.with_tile(a["config"], tile)
        a["config"] = config

        if not _int8_kernel_disabled() and fused_moe_int8.can_run(
            a["A"],
            a["B"],
            a["C"],
            a["A_scale"],
            a["B_scale"],
            a["sorted_token_ids"],
            a["top_k"],
            config["BLOCK_SIZE_M"],
            config["BLOCK_SIZE_K"],
            a["use_fp8_w8a8"],
            a["use_int8_w8a8"],
            a["use_int8_w8a16"],
            a["use_int4_w4a16"],
            a["per_channel_quant"],
            a["block_shape"],
            a["B_bias"],
        ):
            return fused_moe_int8.invoke(
                a["A"],
                a["B"],
                a["C"],
                a["A_scale"],
                a["B_scale"],
                a["topk_weights"],
                a["sorted_token_ids"],
                a["expert_ids"],
                a["num_tokens_post_padded"],
                a["mul_routed_weight"],
                a["top_k"],
                config,
            )
        return invoke(*bound.args, **bound.kwargs)

    for wrapper, wrapped in (
        (_fused_experts_impl, experts_impl),
        (_invoke_fused_moe_triton_kernel, invoke),
    ):
        wrapper._fl_int8_moe_tiles = True  # type: ignore[attr-defined]
        wrapper._fl_original = wrapped  # type: ignore[attr-defined]

    vllm_moe.fused_experts_impl = _fused_experts_impl
    vllm_moe.invoke_fused_moe_triton_kernel = _invoke_fused_moe_triton_kernel
    _INT8_TILES_PATCHED = True
    logger.info(
        "moe-config: INT8 W8A8 expert GEMMs now run the sunrise grouped "
        "kernel on sunrise N/K tiles; BF16 keeps vLLM's kernel and its own "
        "rows-per-expert rule.",
    )
    if deepgemm_moe.available():
        logger.info(
            "moe-config: INT8 W8A8 expert calls are offered to the "
            "deepgemm_ptpu grouped GEMM first; set "
            "VLLM_FL_SUNRISE_BENCH_BASELINE_DEEPGEMM_MOE=1 for the Triton "
            "dataflow."
        )
    return True


def apply_patch() -> bool:
    """Rebind ``try_get_optimal_moe_config`` to FlagGems or FlagGems-vllm."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import vllm.model_executor.layers.fused_moe.fused_moe as vllm_moe
    except Exception as exc:  # noqa: BLE001
        logger.debug("moe-config: vLLM fused_moe module unavailable (%s)", exc)
        return False

    original = getattr(vllm_moe, "try_get_optimal_moe_config", None)
    if original is None:
        logger.debug("moe-config: try_get_optimal_moe_config not found; skip.")
        return False
    if getattr(original, "_fl_sunrise_moe_config", False):
        _PATCHED = True
        return True

    use_fgv = has_flaggems_vllm_embedded_moe_table()
    picker = (
        _sunrise_moe_config_flaggems_vllm if use_fgv else _sunrise_moe_config_flaggems
    )
    source = "FlagGems-vllm embedded table" if use_fgv else "FlagGems"

    def _try_get_optimal_moe_config(
        w1_shape, w2_shape, top_k, dtype, M, block_shape=None
    ):
        try:
            config = picker(
                w1_shape, w2_shape, top_k, dtype, M, block_shape=block_shape
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "moe-config: %s sunrise MoE config unavailable (%s); "
                "falling back to vLLM's NVIDIA-tuned heuristic, which can "
                "hang the w2 GEMM on PTPU.",
                source,
                exc,
            )
            return original(w1_shape, w2_shape, top_k, dtype, M, block_shape)

        return _with_block_size_m(config, w1_shape, w2_shape, top_k, M)

    _try_get_optimal_moe_config._fl_sunrise_moe_config = True  # type: ignore[attr-defined]
    _try_get_optimal_moe_config._fl_original = original  # type: ignore[attr-defined]

    vllm_moe.try_get_optimal_moe_config = _try_get_optimal_moe_config
    rebound = 0
    for module in list(sys.modules.values()):
        if module is None:
            continue
        if getattr(module, "try_get_optimal_moe_config", None) is original:
            module.try_get_optimal_moe_config = _try_get_optimal_moe_config
            rebound += 1

    _PATCHED = True
    if use_fgv:
        logger.info(
            "moe-config: fused-MoE tile config from FlagGems-vllm embedded "
            "table; rebound %d importer(s).",
            rebound,
        )
    else:
        logger.info(
            "moe-config: no FlagGems-vllm embedded MoE table for this device; "
            "using FlagGems _sunrise_get_default_config. rebound %d importer(s).",
            rebound,
        )

    try:
        _patch_int8_moe_tiles()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "moe-config: could not install the INT8 MoE tile hooks (%s); the "
            "W8A8 expert GEMMs keep the FlagGems 32/64/64 pin.",
            exc,
        )
    return True
