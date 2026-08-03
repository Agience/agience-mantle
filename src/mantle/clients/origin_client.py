"""HTTP client for Mantle → Origin.

Sovereign model: Mantle owns authorization (grants + API keys) AND secret-artifact
material in its own store, so it no longer asks Origin for those. The one
remaining cross-service call is an OPTIONAL operator-id lookup — used only in the
full-platform deployment where the operator is bootstrapped on Origin; a
standalone Origin-off Mantle names its operator via `AGIENCE_OPERATOR_ID` instead
(see `services.operator.resolve_operator_id`). Mantle signs the call with its own
service identity (`mantle.private.pem`); Origin verifies via the inline JWKS in
the platform authority manifest.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from mantle.services import peer_signing

logger = logging.getLogger(__name__)


class OriginClient:
    """HTTP shim to Origin for cross-service identity lookups."""

    def __init__(self, base_uri: Optional[str] = None) -> None:
        from origin import config

        # Use the resolved config.ORIGIN_URI (defaults to http://localhost:8080,
        # overridden to http://origin:8080 in-cluster via the ORIGIN_URI env).
        # The previous hardcoded "http://origin:8080" fallback made the HOST mantle
        # (dev) unable to reach Origin → operator never resolved → provisioning
        # silently no-op'd.
        self._base = (base_uri or config.ORIGIN_URI or "http://localhost:8080").rstrip("/")
        self._client = httpx.Client(timeout=3.0)

    def _service_token(self) -> str:
        """Sign a short-lived service JWT addressed to Origin.

        Mantle's lifespan must have called `peer_signing.init()` at startup.
        Tests provide the same setup via the conftest fixture.
        """
        return peer_signing.sign_service_jwt(audience="origin")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._service_token()}",
            "Content-Type": "application/json",
        }

    def get_operator_id(self) -> Optional[str]:
        """Fetch the platform operator UUID from Origin.

        Used by Mantle when ``platform.operator_id`` is absent from its own
        platform_settings (e.g. after a factory reset that wiped Mantle's store
        but left Origin's SQLite intact).  Returns None when Origin
        is unreachable or setup has not yet completed.

        """
        try:
            resp = self._client.get(
                f"{self._base}/internal/operator-id",
                headers=self._headers(),
            )
        except httpx.HTTPError:
            logger.warning("Origin unreachable during get_operator_id", exc_info=True)
            return None
        if resp.status_code != 200:
            logger.warning("Origin /internal/operator-id returned %d", resp.status_code)
            return None
        try:
            return resp.json().get("operator_id") or None
        except ValueError:
            return None


# Module-level singleton
_origin_client: Optional[OriginClient] = None


def get_origin_client() -> OriginClient:
    global _origin_client
    if _origin_client is None:
        _origin_client = OriginClient()
    return _origin_client


def reset_origin_client() -> None:
    """Test hook."""
    global _origin_client
    _origin_client = None


# Keep `time` as an active import — referenced in inline cache logic added later
# without producing an unused-import warning today.
_ = time
