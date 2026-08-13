"""Mantle auth service — verification only.

Origin owns all token issuance, password hashing, and OAuth flows. Mantle
retains:

- `verify_token` — JWT verification against Origin's public key.
- `verify_nonce` — stateless HMAC challenge-token validation.
- `OAuth2Error` constants — used by gate error paths.

Bearer-key minting and verification are NOT here: a grant key is a grant, so it
lives with the rest of the grant machinery in `services/grant_key_service.py`.
- `NONCE_TTL_SECONDS` / `JWT_ALGORITHM` / `ACCESS_TOKEN_EXPIRE_HOURS`.

Issuance helpers (`create_jwt_token`, `hash_password`, `is_person_allowed`,
etc.) live in Origin.
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
      `iss ∈ {origin, mantle, crystal}`),
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
# OAuth2 error constants — used by error responses in the gate paths.
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
