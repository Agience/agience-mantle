"""External OIDC IdP token verification — makes Mantle's auth IdP-agnostic.

Besides the platform's Origin-signed JWTs, Mantle can verify tokens issued
DIRECTLY by any configured trusted OIDC issuer (Microsoft Entra, Auth0, Okta, …).
Each issuer in ``config.TRUSTED_ISSUERS`` declares::

    {"issuer": "<iss>", "audience": "<client-id>",
     "jwks_uri": "<jwks url>"   # OR
     "jwks": {"keys": [...]} }  # inline (no fetch)

If only ``issuer`` is given, the JWKS URL is discovered from
``<issuer>/.well-known/openid-configuration``. Keys are cached and refreshed on a
kid miss (key rotation), at most once per issuer per ``_JWKS_MIN_REFETCH_S``. The
verifier checks signature + issuer + audience; the caller maps the token's stable
subject to a Mantle user (first-login provisioning handles new users — the same path as platform users).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from jose import jwt

from mantle import config

logger = logging.getLogger(__name__)

_JWKS_TTL_S = 3600.0
_FETCH_TIMEOUT_S = 5.0
#: Minimum spacing between JWKS fetches for one issuer, forced refetches on an unknown `kid`
#: included. See `OidcVerifier._keys`.
_JWKS_MIN_REFETCH_S = 30.0
_ALGS = ["RS256", "RS384", "RS512", "ES256"]


def _read_authority_manifest() -> Optional[dict]:
    """Read the on-disk authority manifest (KEYS_DIR/authority.manifest.json).

    Mantle reads it ITSELF — it is just a JSON file of per-issuer inline JWKS, so
    the database verifies tokens with no trust LIBRARY, only this generic seam.
    """
    keys_dir = os.getenv("KEYS_DIR") or "/data/keys"
    try:
        return json.loads((Path(keys_dir) / "authority.manifest.json").read_text())
    except (OSError, ValueError):
        return None

def _manifest_pinned_names() -> set:
    """The issuers the authority manifest pins: every anchor's name, and Origin's own issuer."""
    manifest = _read_authority_manifest() or {}
    names = set((manifest.get("trust_anchors") or {}).keys())
    auth_issuer = getattr(config, "AUTHORITY_ISSUER", None)
    if auth_issuer:
        names.add(auth_issuer)
    return names


def jwks_fetch_allowed(url: Optional[str], configured: Optional[list] = None) -> bool:
    """Is `url` an address Mantle may fetch a JWKS (or discovery document) from? Only an origin
    `MANTLE_JWKS_FETCH_ALLOWLIST` names (comma-separated `scheme://host[:port]`), over https,
    or one a configured issuer names (review 12, H1). Anything else, a link-local or internal
    address included, is never fetched."""
    from urllib.parse import urlsplit

    if not url:
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        return False
    origin = f"{parts.scheme}://{parts.netloc.lower()}"
    allowed = {o.strip().rstrip("/").lower()
               for o in (os.getenv("MANTLE_JWKS_FETCH_ALLOWLIST") or "").split(",") if o.strip()}
    for d in list(getattr(config, "TRUSTED_ISSUERS", []) or []) + list(configured or []):
        if not isinstance(d, dict):
            continue
        for key in ("issuer", "jwks_uri"):
            v = d.get(key)
            if isinstance(v, str) and v.startswith("https://"):
                p = urlsplit(v)
                allowed.add(f"{p.scheme}://{p.netloc.lower()}")
    return origin in allowed


# Fixed namespace for deriving a stable Agience user id from an external
# (issuer/tenant, subject) pair. Never change this — it would re-key every
# external user. See `OidcVerifier.external_user_id`.
_USER_NS = uuid.UUID("a9c1e0de-1d9f-4e7a-8b2c-9f0e1d2c3b4a")


class OidcVerifier:
    # Minimum spacing between refresh-on-miss reloads from the artifact store — an
    # unknown issuer may be a newly-added issuer artifact, but a flood of bad tokens
    # must not flood the DB. Startup + issuer.* events are the primary refresh; this
    # is the bounded real-time fallback.
    _MIN_REFRESH_INTERVAL_S = 10.0

    def __init__(self, trusted_issuers: Optional[list] = None) -> None:
        issuers = trusted_issuers if trusted_issuers is not None else getattr(config, "TRUSTED_ISSUERS", [])
        # Env-sourced external issuers (AGIENCE_TRUSTED_ISSUERS) — the bootstrap seed.
        self._env_issuers: list = [
            d for d in (issuers or []) if isinstance(d, dict) and d.get("issuer")
        ]
        self._jwks_cache: Dict[str, tuple] = {}   # issuer -> (keys, fetched_at)
        self._last_fetch: Dict[str, float] = {}   # issuer -> last JWKS fetch attempt
        self._fetch_locks: Dict[str, threading.Lock] = {}  # issuer -> held while fetching
        self._lock = threading.Lock()
        self._last_refresh = 0.0
        # Build the trust maps from the env seed + manifest anchors (no DB yet).
        self._rebuild([])

    def _rebuild(self, artifact_issuers: list, *, from_store: bool = False) -> None:
        """Rebuild the trust maps from configuration: the configured external issuers
        (`AGIENCE_TRUSTED_ISSUERS`) and the authority manifest's anchors (review 12, H1, H2).

        An issuer artifact never adds an issuer, a key or an audience: trust roots in
        configuration, never in who wrote a row (§51). Once the store has been read
        (`from_store`), a configured external issuer stays trusted only while a live
        artifact names it, so revoking the artifact narrows trust. The manifest always
        wins: its anchors, and Origin's own issuer, are taken from the manifest alone, and
        nothing configured or stored can name them."""
        pinned = _manifest_pinned_names()
        live = {d.get("issuer") for d in (artifact_issuers or [])
                if isinstance(d, dict) and d.get("issuer")}
        external: Dict[str, dict] = {}
        by_iss: Dict[str, dict] = {}
        for d in self._env_issuers:
            iss = d.get("issuer")
            if not iss or iss in pinned:
                continue
            platform = d.get("role", "external") == "platform"
            if from_store and not platform and iss not in live:
                continue
            by_iss[iss] = d
            if not platform:
                external[iss] = d
        self._external = external
        self._by_iss = by_iss
        # The platform's own services + Origin-signed user tokens are verified the
        # SAME generic way — load the manifest's service anchors as issuers too.
        self._load_manifest_anchors()

    def refresh_from_db(self, db: Any) -> None:
        """Reload trusted-issuer artifacts from the store and rebuild trust maps."""
        try:
            from mantle.services.issuers import load_issuer_configs
            arts = load_issuer_configs(db)
        except Exception:
            logger.debug("issuer refresh failed", exc_info=True)
            arts = []
        with self._lock:
            self._last_refresh = time.time()
            self._rebuild(arts, from_store=True)

    def refresh_if_unknown_iss(self, db: Any, token: str) -> bool:
        """If ``token``'s issuer is unknown, refresh from the store (throttled) and
        report whether a refresh ran (so the caller can retry verification). Lets a
        newly-added issuer artifact take effect without a restart."""
        try:
            iss = jwt.get_unverified_claims(token).get("iss", "")
        except Exception:
            return False
        if not iss or iss in self._by_iss:
            return False
        if (time.time() - self._last_refresh) < self._MIN_REFRESH_INTERVAL_S:
            return False
        self.refresh_from_db(db)
        return iss in self._by_iss

    def _load_manifest_anchors(self) -> None:
        """Register the authority manifest's service anchors as issuers, so this one
        verifier uniformly covers: platform-service JWTs (iss = origin/mantle/crystal),
        Origin-signed user tokens + delegations (iss = AUTHORITY_ISSUER), and external
        OIDC IdPs. Inline JWKS — no fetch. Explicitly-configured external issuers win
        over manifest anchors (setdefault)."""
        manifest = _read_authority_manifest()
        if not manifest:
            return
        anchors = manifest.get("trust_anchors", {})
        for name, anchor in anchors.items():
            jwks = anchor.get("jwks") if isinstance(anchor, dict) else None
            if isinstance(jwks, dict):
                # platform-service JWTs carry iss = the service NAME. The manifest wins.
                self._by_iss[name] = {"issuer": name, "jwks": jwks, "audience": None}
                self._external.pop(name, None)
        # Origin-signed user tokens + delegations carry iss = AUTHORITY_ISSUER (a URL),
        # signed by the origin key — verify against the origin anchor's JWKS, and nothing else.
        origin_jwks = (anchors.get("origin") or {}).get("jwks")
        auth_issuer = getattr(config, "AUTHORITY_ISSUER", None)
        if auth_issuer:
            self._external.pop(auth_issuer, None)
            if isinstance(origin_jwks, dict):
                self._by_iss[auth_issuer] = {"issuer": auth_issuer, "jwks": origin_jwks,
                                             "audience": None}
            else:
                self._by_iss.pop(auth_issuer, None)

    def is_trusted(self, iss: Optional[str]) -> bool:
        """True only for configured EXTERNAL OIDC IdPs — not the platform's own
        manifest anchors. Drives tenant namespacing + the dependencies aud-skip."""
        return bool(iss) and iss in self._external

    # -- multi-tenant identity ---------------------------------------------

    def tenant_for(self, iss: Optional[str]) -> Optional[str]:
        """Stable tenant key for an issuer. Each trusted issuer = one tenant.
        Operators may pin an explicit ``"namespace"`` (so the tenant survives an
        issuer-URL change / alias); otherwise the issuer URL IS the tenant key."""
        cfg = self._external.get(iss or "")
        if cfg is None:
            return None
        return cfg.get("namespace") or iss

    def external_user_id(self, claims: Dict[str, Any]) -> Optional[str]:
        """Derive a stable Agience user id from an external token's claims, or
        None if the token isn't from a trusted external issuer.

        The IdP ``sub`` is unique only WITHIN its issuer, so two different IdPs
        (or Entra tenants) can mint the same ``sub``. Namespacing the id by the
        issuer's tenant key keeps tenants isolated — the core multi-tenant
        guarantee. Deterministic: same (tenant, sub) → same user on every login.
        """
        iss = claims.get("iss", "")
        sub = claims.get("sub")
        if not sub or iss not in self._external:
            return None
        tenant = self.tenant_for(iss) or iss
        return str(uuid.uuid5(_USER_NS, f"{tenant}\n{sub}"))

    def verify(self, token: str, *, expected_audience: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Verify an external-IdP token. Returns claims, or None if not trusted /
        invalid. Raises nothing — callers treat None as 'reject'."""
        try:
            iss = jwt.get_unverified_claims(token).get("iss", "")
        except Exception:
            return None
        cfg = self._by_iss.get(iss)
        if cfg is None:
            return None
        try:
            kid = jwt.get_unverified_header(token).get("kid", "")
        except Exception:
            return None

        jwk = self._resolve_key(iss, cfg, kid)
        if jwk is None:
            logger.warning("oidc: no key for kid=%r at issuer=%s", kid, iss)
            return None

        aud = expected_audience or cfg.get("audience")
        if not aud and iss in self._external:
            # Confused-deputy guard (multi-tenant): a trusted EXTERNAL tenant IdP that
            # binds no audience would let a token it minted for a DIFFERENT relying
            # party be accepted here — a cross-tenant hole once several issuers are
            # trusted. Fail closed; register the issuer with an `audience` to trust it.
            # (Internal manifest anchors / the Origin issuer are NOT in `_external`;
            # their aud is validated downstream by `_validate_aud_for_principal`, so
            # they keep verifying without a per-issuer audience.)
            logger.warning(
                "oidc: rejecting token from external issuer=%s — no audience bound "
                "(register the issuer with an `audience`)", iss)
            return None
        options = {"verify_aud": bool(aud)}
        try:
            return jwt.decode(token, jwk, algorithms=_ALGS, issuer=iss, audience=aud, options=options)
        except Exception as exc:
            logger.warning("oidc: verify failed for issuer=%s: %r", iss, exc)
            return None

    # -- JWKS resolution ----------------------------------------------------

    def _resolve_key(self, iss: str, cfg: dict, kid: str) -> Optional[dict]:
        for key in self._keys(iss, cfg, force=False):
            if key.get("kid") == kid:
                return key
        # Miss → refresh once (handles key rotation).
        for key in self._keys(iss, cfg, force=True):
            if key.get("kid") == kid:
                return key
        return None

    def _keys(self, iss: str, cfg: dict, *, force: bool) -> List[dict]:
        """The issuer's keys: inline, cached, or fetched — at most one fetch per issuer per
        `_JWKS_MIN_REFETCH_S`.

        A `kid` is the token's own unverified header, so a caller can name a fresh unknown one on
        every request, and each miss asks for a forced refetch. The spacing is what stops that
        from becoming one outbound JWKS fetch per request: inside it, a forced or cold read
        answers from the cache (or with nothing), and a rotated key is picked up by the first
        miss after it. The spacing counts fetch attempts, failed ones included, so an issuer that
        is down is not hammered either.

        `_lock` guards only the cache and the attempt clock; the network I/O runs outside it, so
        a slow issuer holds up only callers waiting on that issuer's first keys, never another
        issuer's verification. A caller that arrives while the first fetch for a cold issuer is in
        flight waits for that fetch rather than starting a second one.
        """
        inline = cfg.get("jwks")
        if isinstance(inline, dict):
            return inline.get("keys", [])
        with self._lock:
            cached = self._jwks_cache.get(iss)
            now = time.time()
            if cached and not force and (now - cached[1]) < _JWKS_TTL_S:
                return cached[0]
            fetch_lock = self._fetch_locks.setdefault(iss, threading.Lock())
            throttled = (now - self._last_fetch.get(iss, float("-inf"))) < _JWKS_MIN_REFETCH_S
            if not throttled:
                self._last_fetch[iss] = now
                # Taken while `_lock` is held, so a caller that sees the attempt clock set also
                # finds the fetch lock held and waits on it below.
                fetch_lock.acquire()
        if throttled:
            if cached:
                return cached[0]
            with fetch_lock:
                pass
            with self._lock:
                cached = self._jwks_cache.get(iss)
            return cached[0] if cached else []
        try:
            jwks_uri = cfg.get("jwks_uri") or self._discover_jwks_uri(iss, self._env_issuers)
            if not jwks_uri or not jwks_fetch_allowed(jwks_uri, self._env_issuers):
                if jwks_uri:
                    logger.warning("oidc: JWKS for %s not fetched: %s is not an allowlisted "
                                   "address", iss, jwks_uri)
                return cached[0] if cached else []
            try:
                r = httpx.get(jwks_uri, timeout=_FETCH_TIMEOUT_S)
                r.raise_for_status()
                keys = r.json().get("keys", [])
            except Exception as exc:
                logger.warning("oidc: JWKS fetch failed for %s (%s): %r", iss, jwks_uri, exc)
                return cached[0] if cached else []
            with self._lock:
                self._jwks_cache[iss] = (keys, time.time())
            return keys
        finally:
            fetch_lock.release()

    @staticmethod
    def _discover_jwks_uri(iss: str, configured: Optional[list] = None) -> Optional[str]:
        url = iss.rstrip("/") + "/.well-known/openid-configuration"
        if not jwks_fetch_allowed(url, configured):
            return None
        try:
            r = httpx.get(url, timeout=_FETCH_TIMEOUT_S)
            r.raise_for_status()
            return r.json().get("jwks_uri")
        except Exception:
            return None


_verifier: Optional[OidcVerifier] = None
_verifier_lock = threading.Lock()


def get_oidc_verifier() -> OidcVerifier:
    global _verifier
    if _verifier is None:
        with _verifier_lock:
            if _verifier is None:
                _verifier = OidcVerifier()
    return _verifier


def reset_oidc_verifier() -> None:
    """Test hook / reload after config change."""
    global _verifier
    _verifier = None
