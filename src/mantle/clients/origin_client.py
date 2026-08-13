"""HTTP client for Mantle → Origin.

Sovereign model: Mantle owns authorization (grants, and the grant keys that ARE
grants) and secret-artifact material in its own store. The one cross-service call
this client makes is an optional operator-id lookup — used only in the full-platform
deployment where the operator is bootstrapped on Origin; a standalone Origin-off
Mantle names its operator via `AGIENCE_OPERATOR_ID` instead (see
`services.operator.resolve_operator_id`). Mantle signs the call with its own service
identity (`mantle.private.pem`); Origin verifies via the inline JWKS in the platform
authority manifest.

One method is the whole client. That is the measure of the coupling, so anything
added here is a new dependency on Origin and needs to be argued for as one.
"""

from __future__ import annotations

import logging
import socket
import time
from typing import Optional
from urllib.parse import urlsplit

import httpx

from mantle.services import peer_signing

logger = logging.getLogger(__name__)

#: How long to wait for the TCP connect itself — FOR THE WHOLE PROBE, not per address. A peer
#: that is not listening refuses or drops immediately on a LAN or loopback; the multi-second
#: default is a budget for a peer that IS there and slow, and spending it on a peer that is
#: absent is the entire cost of running standalone. Separate from the read budget below, which
#: a reachable Origin legitimately uses.
#:
#: ⚠ THIS IS THE BUDGET FOR THE PROBE, AND `_connect_timeout()` IS WHAT MAKES THAT TRUE.
#: httpcore hands the value to `socket.create_connection`, which applies it to EACH address
#: `getaddrinfo` returns and moves to the next one when it expires. `localhost` — the default
#: ORIGIN_URI's host — resolves to both ::1 and 127.0.0.1, so a name with two records cost
#: twice this, and a name with more cost more. MEASURED on a node with no Origin listening:
#: 1.00s of connect for a value that reads as 0.5, from the two loopback records alone; the
#: cost is per RECORD, so it is set by DNS rather than by anything stated here.
_CONNECT_TIMEOUT_SECONDS = 0.5

#: The rest of the request budget once a connection exists.
_REQUEST_TIMEOUT_SECONDS = 3.0

#: How long an "Origin is not answering" result stands before another attempt is made.
#: Origin is optional, so its absence is a steady state rather than an error to retry
#: through: one probe answers the question for the whole of boot, where three callers
#: resolve the operator in sequence. Bounded rather than permanent so a full-platform
#: node whose Origin starts after Mantle picks it up without a restart.
_UNREACHABLE_MEMO_SECONDS = 60.0


class OriginClient:
    """HTTP shim to Origin for cross-service identity lookups."""

    def __init__(self, base_uri: Optional[str] = None) -> None:
        from mantle import config

        # config.ORIGIN_URI carries its own default (http://localhost:8080), overridden
        # to http://origin:8080 in-cluster via the ORIGIN_URI env, so a host-side mantle
        # in dev reaches Origin at localhost rather than a container-only hostname. An
        # EMPTY value is not a missing value: it is a sovereign node declaring it has no
        # Origin, so no default is substituted for it and `enabled` goes false. Without
        # that, `ORIGIN_URI=""` — the obvious way to say "there is no Origin" — silently
        # became a localhost address that nothing answers.
        configured = base_uri if base_uri is not None else getattr(config, "ORIGIN_URI", "")
        self._base = (configured or "").strip().rstrip("/")
        self._client = httpx.Client(
            timeout=httpx.Timeout(_REQUEST_TIMEOUT_SECONDS, connect=_CONNECT_TIMEOUT_SECONDS)
        )
        #: monotonic deadline before which Origin is treated as absent without a call.
        self._unreachable_until = 0.0

    @property
    def enabled(self) -> bool:
        """False when no Origin URI is configured — a standalone node has nothing to call."""
        return bool(self._base)

    def _connect_timeout(self) -> float:
        """The connect budget for ONE address, so the probe as a whole costs the budget.

        `socket.create_connection` spends the timeout it is given on every address the host
        name resolves to, in turn. Dividing by that count is what turns a per-address value
        into a per-probe one — the alternative is a stated cost that silently multiplies by
        however many A/AAAA records a name happens to carry, which is not a number this
        process gets to choose.

        The common case is unchanged: a single-address host still gets the whole budget. Only
        a multi-record name — `localhost`, or an Origin behind round-robin DNS — divides, and
        it divides a budget that exists for a peer that is ABSENT. A peer that is present and
        merely slow is answered by `_REQUEST_TIMEOUT_SECONDS`, which this does not touch.

        Resolution failure yields one attempt's worth: a name that does not resolve has no
        addresses to spend the budget on, and httpx will fail on the same lookup regardless.
        """
        parts = urlsplit(self._base)
        host = parts.hostname
        if not host:
            return _CONNECT_TIMEOUT_SECONDS
        port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            count = len(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except OSError:
            return _CONNECT_TIMEOUT_SECONDS
        return _CONNECT_TIMEOUT_SECONDS / max(1, count)

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

        Used by Mantle when neither ``platform.operator_id`` in its own
        platform_settings nor ``AGIENCE_OPERATOR_ID`` names one — e.g. after a
        factory reset that wiped Mantle's store but left Origin's intact. Returns
        None when Origin is unconfigured, unreachable, answers non-200, or names no
        operator; the caller (``services.operator.resolve_operator_id``) treats them
        all the same and carries on.

        No Origin URI means no call at all, and a recent failure to reach one is
        remembered rather than re-attempted, so the supported standalone deployment
        pays nothing for a peer it does not have.
        """
        if not self.enabled:
            logger.debug("No ORIGIN_URI configured; resolving the operator locally")
            return None
        if time.monotonic() < self._unreachable_until:
            logger.debug("Origin recently unreachable; skipping get_operator_id")
            return None
        try:
            resp = self._client.get(
                f"{self._base}/internal/operator-id",
                headers=self._headers(),
                timeout=httpx.Timeout(
                    _REQUEST_TIMEOUT_SECONDS, connect=self._connect_timeout()
                ),
            )
        except httpx.HTTPError as exc:
            # An absent optional peer is an expected state, not an incident: the message
            # names it, and a stack trace of the socket layer adds nothing a reader of
            # this line needs. exc_info stays off.
            self._unreachable_until = time.monotonic() + _UNREACHABLE_MEMO_SECONDS
            logger.warning(
                "Origin at %s is unreachable (%s); resolving the operator locally. "
                "Suppressing further attempts for %ds.",
                self._base, type(exc).__name__, int(_UNREACHABLE_MEMO_SECONDS),
                exc_info=False,
            )
            return None
        self._unreachable_until = 0.0
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
