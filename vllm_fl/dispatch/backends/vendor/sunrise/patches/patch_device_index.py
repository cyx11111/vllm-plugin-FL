# Copyright (c) 2026 BAAI. All rights reserved.

"""Install ptpu-opmux's process-wide PTPU runtime compatibility fixes.

vLLM builds the model inside ``with torch.device("ptpu"):``, which has no
ordinal. FlagGems' sunrise ``empty`` then enters ``torch.ptpu.device(device)``.
``torch_ptpu.ptpu._get_device_index`` turns a ``torch.device`` into
``device.index`` and compares that with ``0``. An index-less device yields
``None``, so the comparison raises::

    TypeError: '>=' not supported between instances of 'NoneType' and 'int'

That aborts worker startup while constructing Qwen3 vision ``Conv3dLayer``.
Separately, ``torch.Event()`` starts with c10's ``device_index == -1`` sentinel.
torch_ptpu's DeviceGuard incorrectly compares that sentinel with the explicit
worker device when the event is first recorded, causing::

    RuntimeError: device index [-1] does not match the string device index [N]

The repairs live in ``ptpu_opmux.adapters.torch_ptpu`` and
``ptpu_opmux.adapters.ptpu_native``. Installing both here covers workers that
load the sunrise backend before ``ptpu_opmux.enable()``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def apply_patch() -> bool:
    """Install ptpu-opmux's Python and C++ runtime compatibility patches.

    Idempotent. Returns ``True`` when either patch is newly installed.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return False

    try:
        from ptpu_opmux.adapters.ptpu_native import install_native_runtime_compat
        from ptpu_opmux.adapters.torch_ptpu import install_device_index_patch
    except Exception as exc:  # pragma: no cover - non-PTPU env
        logger.debug(
            "Skipping PTPU runtime compatibility patches "
            "(ptpu_opmux unavailable): %s",
            exc,
        )
        return False

    try:
        index_patched = bool(install_device_index_patch())
        guard_patched = bool(install_native_runtime_compat())
    except Exception as exc:
        logger.warning("PTPU runtime compatibility patch failed: %s", exc)
        return False

    _PATCH_APPLIED = True
    if index_patched or guard_patched:
        logger.info(
            "Sunrise vendor: installed ptpu-opmux runtime compatibility "
            "patches (device-index=%s, DeviceGuard=%s)",
            index_patched,
            guard_patched,
        )
    else:
        logger.debug(
            "PTPU runtime compatibility patches already installed by ptpu-opmux"
        )
    return index_patched or guard_patched
