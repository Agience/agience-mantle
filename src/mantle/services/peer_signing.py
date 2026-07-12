"""Self-contained peer-auth signing for Mantle.

Mantle authenticates to peer services (Origin, Chorus, the embeddings host) with
short-lived JWTs signed by its OWN key (``mantle.private.pem``). Kept local so the
database depends on no shared trust library — it reads its own key and signs with
jose, nothing more. Tokens are claim-for-claim identical to what the old platform
signer produced, so peers verify them unchanged.

Mantle is a verifier-first database; this is the minimal signing it needs for its
outbound platform calls (api-key/grants lookups to Origin, the welcome-email
delegation to Iris via Chorus, embeddings requests to Prism). A purist
"verify-only Mantle" would relocate those calls out of the DB entirely — see
[[project_kit_trust_floor]].
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
_DELEGATION_TTL = 300

_priv_pem: Optional[str] = None


def _keys_dir() -> Path:
    return Path(os.getenv("KEYS_DIR") or "/data/keys")


def init() -> None:
    """Load Mantle's private key at startup (fail-fast if absent)."""
    global _priv_pem
    _priv_pem = (_keys_dir() / f"{_NAME}.private.pem").read_text()


def _pem() -> str:
    global _priv_pem
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
    """The current-instance host artifact id (the `host_id` claim on delegations)."""
    ns = get_instance_namespace()
    return str(uuid.uuid5(ns, "agience/agience-host-current-instance")) if ns else ""


def get_system_principal_id() -> str:
    """The operator-rooted platform system principal id (platform automation acts AS it)."""
    ns = get_instance_namespace()
    return str(uuid.uuid5(ns, "platform/platform-system-principal")) if ns else ""


# -- signing -----------------------------------------------------------------

def sign_service_jwt(audience: str = "mantle", ttl_seconds: int = _DELEGATION_TTL,
                     additional_claims: Optional[Dict[str, Any]] = None) -> str:
    """Service JWT (iss=sub=mantle, principal_type=service) for a peer service."""
    now = int(time.time())
    claims: Dict[str, Any] = {
        "iss": _NAME, "sub": _NAME, "aud": audience,
        "principal_type": "service", "iat": now, "exp": now + ttl_seconds,
    }
    if additional_claims:
        for k, v in additional_claims.items():
            claims.setdefault(k, v)
    return _jwt.encode(claims, _pem(), algorithm="RS256", headers={"kid": _KID})


def sign_delegation_jwt(*, audience: str, user_sub: str, host_id: Optional[str] = None,
                        ttl_seconds: int = _DELEGATION_TTL,
                        additional_claims: Optional[Dict[str, Any]] = None) -> str:
    """RFC 8693 delegation JWT — Mantle acting on behalf of a user (act.sub=mantle)."""
    now = int(time.time())
    claims: Dict[str, Any] = {
        "iss": _NAME, "sub": user_sub, "aud": audience,
        "act": {"sub": _NAME},
        "host_id": host_id if host_id is not None else get_host_id(),
        "principal_type": "delegation", "iat": now, "exp": now + ttl_seconds,
    }
    if additional_claims:
        for k, v in additional_claims.items():
            claims.setdefault(k, v)
    return _jwt.encode(claims, _pem(), algorithm="RS256", headers={"kid": _KID})
