# Copyright (c) 2026 BAAI. All rights reserved.

"""Persistent scratch for the vendor grouped-GEMM expert path.

``deepgemm_moe.fused_experts`` allocates about a dozen tensors per layer per
call. Inside a captured decode graph each one is an address the graph bakes in,
and at 40 layers that is some five hundred allocations recorded per step. vLLM's
own expert dataflow does none of this: its caller hands it ``workspace13`` and
``workspace2``, allocated once and outside any graph.

That difference is the last untested explanation for why the vendor pipeline is
correct eagerly and wrong when captured (see the note on ``deepgemm_moe`` in the
README), so this module exists to remove it.

One rule makes it safe: a buffer a graph has captured must never be freed or
moved. So every slot here is allocated at the size the **largest cudagraph
capture size** needs, is never grown, and callers take a prefix view. A call
wider than that -- an eager prefill -- is not offered the pool at all and
allocates the way it always did.

Slots are shared across layers on purpose. Layers run one after another and
nothing here outlives the call that filled it, so one pool serves all of them;
keying per layer would multiply the footprint by the layer count for no reason.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_UNRESOLVED = object()

# This worker's cudagraph capture sizes. Empty once resolved means nothing is
# captured (eager, or no cudagraph support), in which case there is no baked
# address to protect and the pool stays out of the way.
_CAPTURE_SIZES: Any = _UNRESOLVED

# The config, recorded at init by the platform hook. See ``note_vllm_config``.
_VLLM_CONFIG: Any = None

_POOL: "dict[tuple, torch.Tensor]" = {}


def note_capture_sizes(sizes) -> None:
    """Record the sizes the model runner will actually capture.

    This is the dependable source and the one that decides: it runs in the
    worker process, after vLLM has resolved the config, before ``profile_run``
    reaches the experts, and it is the same list ``capture_model`` walks --
    already emptied if ``cudagraph_mode`` is NONE.
    """
    global _CAPTURE_SIZES
    _CAPTURE_SIZES = tuple(sorted(int(s) for s in sizes))


def note_vllm_config(vllm_config) -> None:
    """Record the config while someone still has it in hand.

    A second source, behind ``note_capture_sizes``, for anything that reaches
    the experts without going through the plugin's model runner.

    ``get_current_vllm_config()`` *raises* when no config context is active,
    and its own error message lists "model forward time" as a case where that
    happens -- which is exactly when this module is asked for the capture
    sizes, from inside ``fused_experts``. Reading it there silently yielded no
    sizes, which silently disabled the pool for a whole GSM8K run. The platform
    hook takes the config as an argument, so it can say so up front.

    The object is kept rather than the sizes because vLLM finalizes
    ``cudagraph_capture_sizes`` after that hook runs.
    """
    global _VLLM_CONFIG, _CAPTURE_SIZES
    _VLLM_CONFIG = vllm_config
    _CAPTURE_SIZES = _UNRESOLVED


def _config():
    if _VLLM_CONFIG is not None:
        return _VLLM_CONFIG
    try:
        from vllm.config import get_current_vllm_config_or_none

        return get_current_vllm_config_or_none()
    except Exception:  # noqa: BLE001 - no vllm, as in the unit tests
        return None


def capture_sizes() -> "tuple[int, ...]":
    """The decode sizes that will be captured, widest last. Empty if none."""
    global _CAPTURE_SIZES
    if _CAPTURE_SIZES is not _UNRESOLVED:
        return _CAPTURE_SIZES

    resolved: "tuple[int, ...]" = ()
    config = _config()
    if config is None:
        # Not an error on its own -- the unit tests run with no config at all --
        # but if this happens in a worker the pool is off and the caller is
        # allocating inside its capture again, so say so where it can be seen.
        logger.debug("moe-scratch: no vllm config; scratch pooling is off.")
    else:
        compilation = config.compilation_config
        sizes = getattr(compilation, "cudagraph_capture_sizes", None)
        mode = getattr(compilation, "cudagraph_mode", None)
        if sizes and (mode is None or str(mode).rsplit(".", 1)[-1] != "NONE"):
            resolved = tuple(sorted(int(s) for s in sizes))

    _CAPTURE_SIZES = resolved
    return resolved


def capture_max_tokens() -> Optional[int]:
    """The largest cudagraph capture size, or None if nothing is captured."""
    sizes = capture_sizes()
    return sizes[-1] if sizes else None


def slot(
    name: str,
    shape: "tuple[int, ...]",
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """A persistent tensor of exactly ``shape``, allocated at most once.

    ``shape`` has to be the largest any caller will ask for under this ``name``:
    the buffer is never reallocated, because a captured graph holds its address.
    Ask for a prefix view rather than a smaller slot.
    """
    key = (name, shape, dtype, device)
    buf = _POOL.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _POOL[key] = buf
    return buf


class Scratch:
    """Either prefix views of the pool, or fresh tensors, on one decision.

    ``rows`` is the call's row count and ``rows_max`` the pool's. When the call
    does not fit -- or there is no captured size to protect -- every request
    falls through to a plain allocation, which is what the pipeline did before
    this module existed.
    """

    __slots__ = ("_rows", "_rows_max", "_device", "pooled")

    def __init__(
        self, rows: int, rows_max: Optional[int], device: torch.device
    ) -> None:
        self._rows = rows
        self._rows_max = rows_max
        self._device = device
        self.pooled = rows_max is not None and rows <= rows_max

    def rows(self, name: str, tail: "tuple[int, ...]", dtype: torch.dtype):
        """A ``[rows, *tail]`` buffer, contiguous in the trailing dims."""
        if not self.pooled:
            return torch.empty(
                (self._rows, *tail), dtype=dtype, device=self._device
            )
        return slot(name, (self._rows_max, *tail), dtype, self._device)[
            : self._rows
        ]

    def fixed(
        self, name: str, shape: "tuple[int, ...]", dtype: torch.dtype, zero: bool
    ):
        """A buffer whose shape does not depend on the row count.

        ``zero`` is for the buffers a kernel accumulates into rather than
        overwrites; the pooled copy has to be cleared explicitly, where the
        fresh one came out of ``torch.zeros``.
        """
        if not self.pooled:
            factory = torch.zeros if zero else torch.empty
            return factory(shape, dtype=dtype, device=self._device)
        buf = slot(name, shape, dtype, self._device)
        if zero:
            buf.zero_()
        return buf


def reset_for_tests() -> None:
    """Drop the pool, the recorded config and the resolved sizes. Tests only."""
    global _CAPTURE_SIZES, _VLLM_CONFIG
    _POOL.clear()
    _CAPTURE_SIZES = _UNRESOLVED
    _VLLM_CONFIG = None


__all__ = [
    "Scratch",
    "capture_max_tokens",
    "capture_sizes",
    "note_capture_sizes",
    "note_vllm_config",
    "reset_for_tests",
    "slot",
]
