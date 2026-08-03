"""Mantle auth service — verifier-only after 1.1e.

Origin owns all token issuance, password hashing, and OAuth flows. Mantle
retains:

- `verify_token` — JWT verification against Origin's public key (switches
  to JWKS-over-HTTP in 1.3).
- `verify_api_key` — raw `agc_xxx` Bearer verification via Origin's
  `/api-keys/verify` (HTTP).
- `verify_nonce` — stateless HMAC challenge-token validation.
- `hash_api_key` / `generate_api_key` — used by `services/grant_service.py`
  and the deprecated card-key rotation in `services/workspace_service.py`.
- `OAuth2Error` constants — used by gate/api-key error paths.
- `NONCE_TTL_SECONDS` / `JWT_ALGORITHM` / `ACCESS_TOKEN_EXPIRE_HOURS`.

All issuance helpers (`create_jwt_token`, `hash_password`, `is_person_allowed`,
etc.) moved to Origin alongside their last callers in 1.1e.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from mantle.db.store import Database
from jose import JWTError, jwt

from origin import config
from mantle.entities.api_key import APIKey as APIKeyEntity

logger = logging.getLogger(__name__)

JWT_ALGORITHM = "RS256"
ACCESS_TOKEN_EXPIRE_HOURS = 12
NONCE_TTL_SECONDS = 1800


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------
def verify_token(token: str, expected_audience: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Verify and decode an incoming JWT.

    Mantle verifies tokens with ONE generic issuer+JWKS verifier (`services.oidc`)
    — the database depends on no trust LIBRARY, only the manifest's inline JWKS that
    it reads itself. The verifier knows, uniformly:
    - the authority manifest's service anchors (platform-service JWTs:
      `iss ∈ {origin, mantle, chorus, crystal}`),
    - the Origin issuer (user tokens + delegations: `iss == AUTHORITY_ISSUER`),
    - and any configured external OIDC IdP (Entra/Auth0/Okta/...).

    The token's `iss` selects the JWKS; signature/iss/aud/exp are checked. Per-
    principal audience + claim-chain rules live in `services.dependencies`.
    """
    from mantle.services.oidc import get_oidc_verifier

    payload = get_oidc_verifier().verify(token, expected_audience=expected_audience)
    if payload is None:
        return None
    exp = payload.get("exp")
    if exp and datetime.now(timezone.utc).timestamp() > exp:
        return None
    return payload


def verify_api_key(db: Database, api_key: str) -> Optional[APIKeyEntity]:
    """Verify a raw `agc_xxx` token via the pluggable authz backend.

    `local` (default) verifies against Mantle's own api_keys in the lattice (`db`);
    `origin` delegates to Origin. Returns the APIKey entity (grants dropped here;
    callers that need grants use the backend's `verify_api_key` directly).
    """
    from mantle.services.grant_store import get_apikey_backend

    result = get_apikey_backend().verify_api_key(db, api_key)
    if result is None:
        return None
    api_key_entity, _grants = result
    return api_key_entity


def verify_nonce(
    token: str,
    key_id: str,
    artifact_id: str,
    secret: str,
    ttl_seconds: int = NONCE_TTL_SECONDS,
) -> bool:
    if not secret or not token:
        return False
    try:
        padding = "=" * (4 - len(token) % 4)
        decoded = base64.urlsafe_b64decode(token + padding).decode("utf-8")
        parts = decoded.split(":", 3)
        if len(parts) != 4:
            return False
        ts_str, nonce_artifact_id, nonce_key_id, sig = parts
        ts = int(ts_str)
    except Exception:
        return False
    if nonce_artifact_id != artifact_id or nonce_key_id != key_id:
        return False
    if int(time.time()) - ts > ttl_seconds:
        return False
    expected_payload = f"{ts_str}:{artifact_id}:{key_id}"
    expected_sig = hmac.new(
        secret.encode("utf-8"), expected_payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return secrets.compare_digest(sig, expected_sig)


# ---------------------------------------------------------------------------
# API key helpers — small, stateless, Mantle-owned
# ---------------------------------------------------------------------------
def generate_api_key() -> str:
    return f"agc_{secrets.token_bytes(16).hex()}"


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# OAuth2 error constants — used by error responses in gate / api-key paths.
# ---------------------------------------------------------------------------
class OAuth2Error:
    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED_CLIENT = "unauthorized_client"
    ACCESS_DENIED = "access_denied"
    UNSUPPORTED_RESPONSE_TYPE = "unsupported_response_type"
    INVALID_SCOPE = "invalid_scope"
    SERVER_ERROR = "server_error"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"
    INVALID_CLIENT = "invalid_client"
    INVALID_GRANT = "invalid_grant"
    UNSUPPORTED_GRANT_TYPE = "unsupported_grant_type"
