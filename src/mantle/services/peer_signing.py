"""Self-contained peer-auth signing for Mantle.

Mantle authenticates to peer services with a short-lived JWT signed by its own key
(``mantle.private.pem``). Kept local so the database depends on no shared trust
library — it reads its own key and signs with jose, nothing more.

Mantle is a verifier-first database, so this is a small surface: **one signature**.
The service JWT (``iss=sub=mantle``, ``principal_type=service``) is what
``clients/origin_client`` presents on its one outbound call to Origin. Mantle signs
nothing else — no user token, and no delegation.

Delegation is inbound-only here, and deliberately so. ``services/dependencies``
accepts ``principal_type == "delegation"`` on any route, requiring ``sub`` /
``act.sub`` / ``host_id``, so Mantle *verifies* RFC 8693
delegations; the peer that mints them is the authority issuer, not this module. A
signer for the direction Mantle never initiates would be an untested surface whose
only job is to drift out of agreement with the verifier that faces it.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from jose import jwt as _jwt

_NAME = "mantle"
_KID = "mantle-1"

#: Default lifetime of a signed service JWT. Short enough that a leaked token is a
#: window rather than a credential; long enough to survive one slow peer call.
_TTL = 300

_priv_pem: Optional[str] = None


def _keys_dir() -> Path:
    return Path(os.getenv("KEYS_DIR") or "/data/keys")


def init() -> None:
    """Load Mantle's private key at startup (fail-fast if absent)."""
    global _priv_pem
    _priv_pem = (_keys_dir() / f"{_NAME}.private.pem").read_text()


def _pem() -> str:
    if _priv_pem is None:
        init()
    return _priv_pem  # type: ignore[return-value]


# -- instance / host identity (read from the shared instance.uuid) -----------

def get_instance_namespace() -> Optional[uuid.UUID]:
    try:
        return uuid.UUID((_keys_dir() / "instance.uuid").read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def get_host_id() -> str:
    """The current-instance host artifact id, as `services/system_identity.py` resolves it."""
    ns = get_instance_namespace()
    return str(uuid.uuid5(ns, "agience/agience-host-current-instance")) if ns else ""


def get_system_principal_id() -> str:
    """The operator-rooted platform system principal id (platform automation acts AS it)."""
    ns = get_instance_namespace()
    return str(uuid.uuid5(ns, "platform/platform-system-principal")) if ns else ""


# -- signing -----------------------------------------------------------------

def sign_service_jwt(audience: str = "mantle", ttl_seconds: int = _TTL,
                     additional_claims: Optional[Dict[str, Any]] = None) -> str:
    """Service JWT (iss=sub=mantle, principal_type=service) for a peer service.

    The whole of Mantle's outbound signing. `sub` is Mantle itself, never a user:
    this token says "Mantle is calling", and it can say nothing else.
    """
    now = int(time.time())
    claims: Dict[str, Any] = {
        "iss": _NAME, "sub": _NAME, "aud": audience,
        "principal_type": "service", "iat": now, "exp": now + ttl_seconds,
    }
    if additional_claims:
        for k, v in additional_claims.items():
            claims.setdefault(k, v)
    return _jwt.encode(claims, _pem(), algorithm="RS256", headers={"kid": _KID})
