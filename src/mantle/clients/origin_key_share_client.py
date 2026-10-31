"""HTTP client for Origin's share of each master key (`/internal/key-shares`).

Under threshold custody (`MANTLE_MASTER_KEY_CUSTODY=threshold`) a principal's master key is split
2-of-3, and Origin holds share 2 (`search/mantle/threshold_custody.py`). This client is the whole of
Mantle's side of that: store a share, fetch a share. It is a second dependency on Origin, separate
from `origin_client.OriginClient` (the operator-id lookup) because the two differ in kind: that one
is optional and degrades to a local answer, and this one is required, so every failure here is
raised rather than read as an absent share. "Origin did not answer" and "Origin holds no share" are
different answers, and only the second may let a key be minted.

Each call carries Mantle's own service JWT (`peer_signing.sign_service_jwt(audience="origin")`),
which Origin verifies against the authority manifest's anchor for `mantle`, bound to that one call:
it lives `TOKEN_TTL` seconds, carries a fresh `jti` that Origin accepts once, and names the method
(`htm`), the route path (`htu`) and the principal (`principal_id`).

Only a 404 whose body says `"absent": true` is read as "no share". A 404 from anything else (a
proxy, an Origin without the route) is a failure like any other.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional
from urllib.parse import quote

import httpx

from mantle.services import peer_signing

logger = logging.getLogger(__name__)

#: Every call is bounded well inside the custody lock's 30 s (`threshold_custody.LOCK_TIMEOUT`),
#: since `put` calls Origin while holding it: a hung Origin refuses the put rather than holding it.
CALL_TIMEOUT = httpx.Timeout(8.0, connect=3.0)
_PATH = "/internal/key-shares/"

#: The lifetime of a custody token; Origin refuses one that lives longer than 60 seconds.
TOKEN_TTL = 30


class CustodianUnavailable(RuntimeError):
    """Origin could not answer for a share: unreachable, refused, or answered unexpectedly."""


class ShareConflict(CustodianUnavailable):
    """Origin holds a share of a different key for this principal and kept it."""


class OriginShareCustodian:
    """Store and fetch Origin's share of a principal's master key."""

    def __init__(self, base_uri: Optional[str] = None, client: Optional[httpx.Client] = None):
        if base_uri is None:
            from mantle import config
            base_uri = getattr(config, "ORIGIN_URI", "")
        self._base = (base_uri or "").strip().rstrip("/")
        if not self._base:
            raise ValueError("threshold custody needs ORIGIN_URI: Origin holds a share of each key")
        self._client = client or httpx.Client(timeout=CALL_TIMEOUT)

    def _url(self, principal_id: str) -> str:
        return f"{self._base}{_PATH}{quote(principal_id, safe='')}"

    def _headers(self, method: str, principal_id: str) -> dict:
        token = peer_signing.sign_service_jwt(
            audience="origin", ttl_seconds=TOKEN_TTL,
            additional_claims={"jti": uuid.uuid4().hex, "htm": method,
                               "htu": _PATH + principal_id, "principal_id": principal_id})
        return {"Authorization": f"Bearer {token}"}

    def get_share(self, principal_id: str) -> Optional[dict]:
        """`{"x", "y", "kcv"}`, or None when Origin answers that it holds none."""
        try:
            resp = self._client.get(self._url(principal_id),
                                    headers=self._headers("GET", principal_id))
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code == 404:
            try:
                absent = resp.json().get("absent") is True
            except (ValueError, AttributeError):
                absent = False
            if absent:
                return None
            raise CustodianUnavailable("Origin answered 404 without saying it holds no share")
        if resp.status_code != 200:
            raise CustodianUnavailable(f"Origin answered {resp.status_code} for a key share")
        body = resp.json()
        return {"x": body["x"], "y": body["y"], "kcv": body["kcv"]}

    def put_share(self, principal_id: str, share: dict, *,
                  supersedes: Optional[str] = None) -> str:
        """Store a share; returns Origin's outcome (`stored`, `unchanged`, `replaced`)."""
        body = dict(share)
        if supersedes:
            body["supersedes"] = supersedes
        try:
            resp = self._client.put(self._url(principal_id), json=body,
                                    headers=self._headers("PUT", principal_id))
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code == 409:
            raise ShareConflict(f"Origin holds a share of another key for {principal_id!r}")
        if resp.status_code != 200:
            raise CustodianUnavailable(f"Origin answered {resp.status_code} storing a key share")
        return resp.json().get("outcome", "stored")
