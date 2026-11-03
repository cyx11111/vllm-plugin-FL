# Copyright (c) 2026 BAAI. All rights reserved.

"""Dense BF16 projections on ``deepgemm_ptpu``'s non-grouped GEMM.

FlagGems loses on large GEMMs, so ``linear`` / ``mm`` / ``addmm`` stay on
``VLLM_FL_FLAGOS_BLACKLIST`` and ``F.linear`` falls through to
``aten::linear``. This path takes the vendor GEMM whenever it reports a
config. The two agree numerically; this is not a speed-for-accuracy trade.

``_shallow_k_collapse`` refuses a bounded band of decode widths below
``K=256``, where the vendor GEMM collapses. The collapse is not visible in
the vendor config, so the rule is decided on shape before querying. The
band is refused whole rather than carved: an exception at one width would
not be monotonic in K. A single row and prefill widths stay on the vendor.

``is_support_bf16_gemm`` raises (slowly) for ``N=1``, the shared-expert
gate. Those weights are remembered so a new prefill width does not stall
the host again. Bias is declined: ``bf16_gemm_nt`` has no bias argument.

Set ``VLLM_FL_SUNRISE_BENCH_BASELINE_DENSE_BF16_GEMM=1`` to keep
``aten::linear`` for an A/B baseline.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Below this K the vendor GEMM collapses over a band of decode widths.
_MIN_K = 256

# The band ends here: 48 is the m_per_block the vendor moves to at that
# width. Above it, and on into prefill, shallow K is still taken.
_COLLAPSE_MAX_M = 48

# A single row is outside the collapse. Where the band starts depends on
# N, so the lower bound is the conservative one.
_COLLAPSE_MIN_M = 2

_BASELINE_ENV = "VLLM_FL_SUNRISE_BENCH_BASELINE_DENSE_BF16_GEMM"

# The vendor's narrowest n-tile is 8, so N=1 (shared-expert gate) raises
# "no valid fallback config found". The raise is caught, but the query is
# slow and the cache is keyed on M, so a new prefill width would pay it
# again. Decline N=1 without asking.
_MIN_N = 2

_FIELD = re.compile(r"(\w+)=(\d+)")

_DECIDED: dict[tuple[int, int, int], bool] = {}

# Weight shapes the vendor could not answer for, so a query that costs
# milliseconds is paid once per weight rather than once per batch width. Not
# folded into _DECIDED because that one is keyed on M.
_UNANSWERABLE: set[tuple[int, int]] = set()
_WORKSPACE_BYTES: dict[tuple[int, int, int], int] = {}
_WORKSPACE: dict[tuple[int, torch.device], torch.Tensor] = {}
_UNAVAILABLE: Optional[str] = None
_DECISION_LOGGED = False


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _disabled() -> bool:
    return _env_on(_BASELINE_ENV)


def available() -> bool:
    """Whether this path could run at all, without importing the wheel.

    For the startup log line, so it answers before device init and cannot be
    what triggers it. Whether a given call is taken is decided by ``can_run``.
    """
    if _disabled():
        return False
    return importlib.util.find_spec("deepgemm_ptpu") is not None


def disable(reason: str) -> None:
    """Stop offering this path for the rest of the process.

    For a caller that saw the vendor GEMM raise inside a forward pass: the
    incumbent can always serve the same call, but a kernel that failed once is
    not worth re-entering on every token.
    """
    global _UNAVAILABLE
    if _UNAVAILABLE is None:
        _UNAVAILABLE = reason


def _lib():
    """The vendor wheel, or None with the reason recorded once."""
    global _UNAVAILABLE
    if _UNAVAILABLE is not None:
        return None
    try:
        import deepgemm_ptpu

        return deepgemm_ptpu
    except Exception as exc:  # noqa: BLE001
        _UNAVAILABLE = str(exc)
        logger.info(
            "dense-bf16: vendor GEMM unavailable (%s); BF16 linears keep "
            "aten::linear.",
            exc,
        )
        return None


def _config(dg, m: int, n: int, k: int) -> Optional[dict[str, int]]:
    """The config the vendor reports as best, or None if it has none to report.

    Nothing reads the fields any more -- the gate asks only whether the vendor
    answered -- but they are parsed rather than discarded because answering with
    a config that has no ``m_per_block`` or ``n_per_warp`` in it is not an
    answer, and because a rejected shape is easier to explain from a log of what
    the vendor said it would do.

    Both queries are made and either one failing is a None. A shape the vendor
    cannot describe at all is one to leave alone.
    """
    best = None
    for fn in (dg.get_bf16_best_config_log, dg.get_bf16_fallback_config_log):
        try:
            text = fn(m, n, k, 1)
        except Exception:  # noqa: BLE001
            return None
        if not text:
            return None
        if best is None:
            best = {key: int(val) for key, val in _FIELD.findall(text)}
    if best is None or "m_per_block" not in best or "n_per_warp" not in best:
        return None
    return best


def _shallow_k_collapse(m: int, k: int) -> bool:
    """A shallow-K GEMM at a decode width, where the vendor falls over.

    Read off the shape rather than the config because the config does not
    distinguish the collapsed cells from their neighbours -- see the module
    docstring for the pair that shares a config and differs 2.3x. Being a shape
    test it settles the call without querying the vendor at all.
    """
    return k < _MIN_K and _COLLAPSE_MIN_M <= m < _COLLAPSE_MAX_M


def _profitable(dg, m: int, n: int, k: int) -> bool:
    key = (m, n, k)
    cached = _DECIDED.get(key)
    if cached is not None:
        return cached
    if (n, k) in _UNANSWERABLE:
        _DECIDED[key] = False
        return False

    # The support and config queries are host-side vendor calls of unknown
    # appetite for allocation, and allocating is what a capture forbids. A shape
    # first seen while recording is declined rather than probed; warmup runs the
    # same shapes eagerly beforehand, so in practice the answer is already
    # cached by the time a capture wants it.
    from ..capture import is_capturing

    if is_capturing():
        return False

    ok = False
    try:
        if dg.is_support_bf16_gemm(m, n, k, 1):
            ok = _config(dg, m, n, k) is not None
    except Exception as exc:  # noqa: BLE001
        # The INT8 sibling has seen is_support raise for arguments it accepted
        # moments earlier. Declining is right and it must not escape into a
        # forward pass.
        logger.debug("dense-bf16: declining M=%d N=%d K=%d (%s)", m, n, k, exc)
        ok = False
        # Asking again is slow, so remember the weight and not just this M
        # — but only once a second query agrees, because a spurious raise
        # would otherwise forfeit the shape for the whole run.
        try:
            dg.is_support_bf16_gemm(m, n, k, 1)
        except Exception:  # noqa: BLE001
            _UNANSWERABLE.add((n, k))
            logger.info(
                "dense-bf16: the vendor GEMM reports no config for N=%d K=%d; "
                "those linears keep aten::linear and are not asked about again.",
                n,
                k,
            )
    _DECIDED[key] = ok
    return ok


def _workspace_for(dg, m: int, n: int, k: int, device: torch.device):
    """The vendor scratch buffer, sized by the vendor and cached by size.

    Size is asked for rather than assumed. Cache the size so the vendor is
    not queried once per linear per token, and the buffer so nothing
    allocates inside a capture.
    """
    nbytes = _WORKSPACE_BYTES.get((m, n, k))
    if nbytes is None:
        try:
            nbytes = int(dg.get_bf16_gemm_nt_workspace_bytes(m, n, k))
        except Exception:  # noqa: BLE001
            nbytes = 0
        _WORKSPACE_BYTES[(m, n, k)] = nbytes
    if nbytes <= 0:
        return None
    key = (nbytes, device)
    buf = _WORKSPACE.get(key)
    if buf is None:
        buf = torch.empty(nbytes, dtype=torch.int8, device=device)
        _WORKSPACE[key] = buf
    return buf


def can_run(
    a: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> bool:
    """Whether this module implements the linear it is being offered."""
    if _disabled():
        return False
    # No bias argument on the vendor GEMM. Shipped projections are bias-free.
    if bias is not None:
        return False
    if a.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if weight.dim() != 2 or a.dim() < 2:
        return False
    # ``bf16_gemm_nt(A[M, K], B[N, K])`` is ``A @ B.t()``, which is exactly what
    # ``F.linear`` computes with vLLM's [out_features, in_features] weight, so
    # no transpose or copy is needed -- but only if both are K-major.
    if not weight.is_contiguous():
        return False
    if a.stride(-1) != 1:
        return False
    if a.shape[-1] != weight.shape[1]:
        return False

    dg = _lib()
    if dg is None:
        return False

    m = a.numel() // a.shape[-1]
    n, k = weight.shape
    if m <= 0:
        return False
    # Cheaper than letting the vendor tell us, and it tells us by raising.
    if n < _MIN_N:
        return False
    # Decided on the shape, so the vendor is not asked about a shape whose
    # answer would not be believed anyway.
    if _shallow_k_collapse(m, k):
        return False
    return _profitable(dg, m, n, k)


def linear(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``a @ weight.t()`` through the vendor GEMM."""
    dg = _lib()
    n, k = weight.shape
    flat = a.reshape(-1, k)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    m = flat.shape[0]

    out = dg.bf16_gemm_nt(
        flat,
        weight,
        _workspace_for(dg, m, n, k, flat.device),
        None,
        torch.bfloat16,
    )
    return out.reshape(*a.shape[:-1], n)


def try_linear(
    a: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Run the vendor GEMM for this call, or decline and leave it to torch."""
    global _DECISION_LOGGED
    if not can_run(a, weight, bias):
        return None
    if not _DECISION_LOGGED:
        _DECISION_LOGGED = True
        logger.info(
            "dense-bf16: BF16 linears now run the deepgemm_ptpu GEMM wherever "
            "the vendor reports a config for the shape; K<%d at decode widths "
            "keeps aten::linear. Disable with %s=1.",
            _MIN_K,
            _BASELINE_ENV,
        )
    return linear(a, weight)
