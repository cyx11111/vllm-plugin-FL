# Copyright (c) 2026 BAAI. All rights reserved.

"""Whether a PTPU graph capture is in progress right now.

Some vendor entry points cannot be recorded into a graph. The one that forced
this to exist is ``torch_ptpu.sd_kernel.int8_scaled_mm``, which reaches
``tangMemGetInfo`` and throws ``stream capture unsupported``; a raise at that
point does not merely lose the call, it leaves the capture broken, so
``capture_end`` then fails too and the process aborts. Catching the exception is
therefore not a fix -- the call has to not happen.

The constraint is narrower than it first appears. Replaying a captured graph
does not run any of this Python, so a kernel only has to be capturable to be
recorded, not to be used. Dispatch can consult this and pick a capturable kernel
while a graph is being recorded, then use the faster one when running eagerly.

torch_ptpu exposes no way to ask, so this counts the depth itself around
``PTPUGraph.capture_begin`` and ``capture_end``, which is what both
``torch.ptpu.graph`` and direct callers go through.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_depth = 0
_INSTALLED = False


def is_capturing() -> bool:
    """True while a graph capture is being recorded on this thread.

    Answers False when the tracker was never installed, which is the safe
    direction for callers that only use it to *avoid* a kernel: on a build with
    no graph support there is no capture to break.
    """
    return _depth > 0


def install() -> bool:
    """Wrap ``PTPUGraph``'s capture boundaries to track depth. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        import torch

        graph_cls = torch.ptpu.PTPUGraph
    except Exception as exc:  # noqa: BLE001
        logger.debug("capture tracker unavailable (%s)", exc)
        return False

    begin = getattr(graph_cls, "capture_begin", None)
    end = getattr(graph_cls, "capture_end", None)
    if begin is None or end is None:
        logger.debug("capture tracker: PTPUGraph has no capture boundaries")
        return False

    def capture_begin(self, *args, **kwargs):
        global _depth
        _depth += 1
        try:
            return begin(self, *args, **kwargs)
        except Exception:
            _depth -= 1
            raise

    def capture_end(self, *args, **kwargs):
        global _depth
        try:
            return end(self, *args, **kwargs)
        finally:
            _depth = max(0, _depth - 1)

    capture_begin._fl_original = begin  # type: ignore[attr-defined]
    capture_end._fl_original = end  # type: ignore[attr-defined]
    graph_cls.capture_begin = capture_begin
    graph_cls.capture_end = capture_end
    _INSTALLED = True
    return True
