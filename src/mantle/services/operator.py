"""Platform operator resolution — sovereign-capable.

The platform operator (the root platform admin) can be established three ways,
tried in order so Mantle works both in the full platform AND standalone
(Origin-off):

1. Mantle's own ``platform.operator_id`` setting — set by Mantle-side bootstrap
   / provisioning, or synced from Origin previously.
2. ``config.OPERATOR_ID`` (env ``AGIENCE_OPERATOR_ID``) — a SOVEREIGN standalone
   Mantle names its operator here; no Origin required.
3. Origin's ``/internal/operator-id`` — the full-platform path, where the operator
   is bootstrapped on Origin. Non-fatal + skipped gracefully when Origin is absent.

Returns "" when the operator cannot be resolved (a fresh node before setup).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def resolve_operator_id(db: Any = None) -> str:
    from origin import config
    from mantle.services.platform_settings_service import settings as platform_settings

    op = platform_settings.get("platform.operator_id")
    if op:
        return str(op)

    cfg_op = getattr(config, "OPERATOR_ID", "") or ""
    if cfg_op:
        return str(cfg_op)

    try:
        from mantle.clients.origin_client import get_origin_client
        return str(get_origin_client().get_operator_id() or "")
    except Exception:
        logger.debug("operator_id: Origin fallback unavailable (non-fatal)", exc_info=True)
        return ""
