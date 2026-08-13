"""Trusted-issuer artifacts -> verifier trust config.

A trusted issuer (an external OIDC IdP, or the platform's own issuer) is the same
primitive the Authority is built from: an ``iss`` + a key set. Modeling it as a
``vnd.agience.issuer+json`` artifact makes adding / rotating trust a governable,
audited edit instead of env config. ``OidcVerifier`` resolves its trust set from
these artifacts; the authority manifest + ``AGIENCE_TRUSTED_ISSUERS`` env become a
bootstrap seed.

Trust boundary: ONLY issuer artifacts created by the platform system principal are
trusted. Mantle's create endpoint is label-blind, so the content-type alone must
never confer trust (that would let any user mint a trusted IdP). If the system
principal can't be determined, we trust NO issuer artifacts (fail closed).
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ISSUER_CONTENT_TYPE = "application/vnd.agience.issuer+json"

# Fields copied verbatim from the artifact context into the verifier config — the
# same shape as a ``config.TRUSTED_ISSUERS`` entry, so artifact-sourced issuers are
# verified identically to env-sourced ones.
_PASSTHROUGH = ("audience", "jwks", "jwks_uri", "namespace", "role")


def _system_principal_id() -> Optional[str]:
    try:
        from mantle.services.peer_signing import get_system_principal_id
        return get_system_principal_id()
    except Exception:
        return None


def load_issuer_configs(db: Any) -> List[Dict[str, Any]]:
    """Load trusted-issuer configs from committed, system-owned issuer artifacts.

    Returns dicts shaped like ``config.TRUSTED_ISSUERS`` entries:
        {"issuer", "audience"?, "jwks"|"jwks_uri", "namespace"?, "role"?}
    Empty list on any failure or if the system principal is unknown (fail closed).
    """
    if db is None:
        return []
    system_id = _system_principal_id()
    if not system_id:
        logger.warning(
            "issuer load: system principal unknown — trusting no issuer artifacts"
        )
        return []
    try:
        from mantle.db import backend as db_store
        arts = db_store.list_committed_artifacts_by_context_content_type(
            db, ISSUER_CONTENT_TYPE, created_by=system_id,
        )
    except Exception:
        logger.debug("issuer load: query failed", exc_info=True)
        return []

    out: List[Dict[str, Any]] = []
    for a in arts:
        ctx = getattr(a, "context", None)
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx) if ctx else {}
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(ctx, dict):
            continue
        iss = ctx.get("issuer")
        if not iss:
            continue
        cfg: Dict[str, Any] = {"issuer": iss}
        for k in _PASSTHROUGH:
            if ctx.get(k) is not None:
                cfg[k] = ctx[k]
        out.append(cfg)
    if out:
        logger.info("issuer load: %d trusted issuer artifact(s)", len(out))
    return out


# ---------------------------------------------------------------------------
# Admin create / list / revoke
#
# The privileged path that materializes a trusted issuer. The artifact is owned by
# the SYSTEM principal (so the loader trusts it) but records the authorizing admin
# in context (provenance roots to a person — never an unattributed service write).
# Creation goes through store.create_artifact so it fires the db-chokepoint event
# the watcher reacts to: an admin adds an IdP and it's live immediately.
# ---------------------------------------------------------------------------


def create_issuer_artifact(
    db: Any,
    *,
    issuer: str,
    authorized_by: str,
    jwks: Optional[dict] = None,
    jwks_uri: Optional[str] = None,
    audience: Optional[str] = None,
    namespace: Optional[str] = None,
    role: str = "external",
) -> Any:
    """Create a system-owned, committed trusted-issuer artifact. Raises ValueError
    on bad input, RuntimeError if the system principal is unavailable."""
    if not issuer:
        raise ValueError("issuer is required")
    if not jwks and not jwks_uri:
        raise ValueError("jwks or jwks_uri is required")
    if role not in ("external", "platform"):
        raise ValueError("role must be 'external' or 'platform'")
    if role == "external" and not audience:
        # Fail fast: an external tenant IdP MUST bind an audience, or Mantle's verifier
        # rejects its tokens (confused-deputy across tenants — see OidcVerifier.verify).
        raise ValueError("an external issuer must bind an 'audience'")
    system_id = _system_principal_id()
    if not system_id:
        raise RuntimeError("system principal unavailable — cannot own issuer artifact")

    ctx: Dict[str, Any] = {
        "content_type": ISSUER_CONTENT_TYPE, "issuer": issuer, "role": role,
        "authorized_by": authorized_by,
    }
    if audience is not None:
        ctx["audience"] = audience
    if jwks is not None:
        ctx["jwks"] = jwks
    if jwks_uri is not None:
        ctx["jwks_uri"] = jwks_uri
    if namespace is not None:
        ctx["namespace"] = namespace

    from mantle.db import backend as db_store
    from mantle.entities.artifact import Artifact
    art = Artifact(
        id=str(uuid.uuid4()), collection_id="", state="committed",
        created_by=system_id, modified_by=system_id, context=json.dumps(ctx),
        content_type=ISSUER_CONTENT_TYPE, name=issuer,
    )
    db_store.create_artifact(db, art)  # emits artifact.created -> watcher refreshes
    logger.info("issuer create: %s (authorized_by=%s)", issuer, authorized_by)
    return art


def list_issuer_artifacts(db: Any) -> List[Any]:
    """Return the committed, system-owned issuer artifacts (for the admin list)."""
    system_id = _system_principal_id()
    if db is None or not system_id:
        return []
    from mantle.db import backend as db_store
    return db_store.list_committed_artifacts_by_context_content_type(
        db, ISSUER_CONTENT_TYPE, created_by=system_id,
    )


def seed_platform_issuer_artifacts(db: Any) -> int:
    """Bootstrap-seed issuer artifacts from the authority manifest + env (#3 P1).

    Idempotent, create-if-missing: turns the platform's own trust (manifest service
    anchors + AUTHORITY_ISSUER, as `role=platform`) and `AGIENCE_TRUSTED_ISSUERS`
    (as `role=external`) into governable artifacts. The manifest/env are thereby
    demoted to a bootstrap SEED; the verifier reads trust from these artifacts and
    falls back to the manifest only for anchors not yet seeded (setdefault ordering
    in OidcVerifier._rebuild). Never overwrites — an admin's later edit/rotation
    survives a reboot. Returns the number created. Best-effort: a failure leaves the
    manifest fallback intact, so it is non-fatal.
    """
    if db is None:
        return 0
    system_id = _system_principal_id()
    if not system_id:
        logger.warning("issuer seed: system principal unknown; skipping (manifest fallback)")
        return 0

    # Already-present issuers (committed, system-owned) — skip these.
    existing: set = set()
    for a in list_issuer_artifacts(db):
        ctx = getattr(a, "context", None)
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx) if ctx else {}
            except (json.JSONDecodeError, TypeError):
                ctx = {}
        if isinstance(ctx, dict) and ctx.get("issuer"):
            existing.add(ctx["issuer"])

    from mantle import config
    from mantle.services.oidc import _read_authority_manifest

    desired: List[Dict[str, Any]] = []
    manifest = _read_authority_manifest()
    if manifest:
        anchors = manifest.get("trust_anchors", {}) or {}
        for name, anchor in anchors.items():
            jwks = anchor.get("jwks") if isinstance(anchor, dict) else None
            if isinstance(jwks, dict):
                desired.append({"issuer": name, "jwks": jwks, "role": "platform"})
        # Origin-signed USER tokens carry iss = AUTHORITY_ISSUER (a URL), verified
        # against origin's JWKS.
        origin_jwks = (anchors.get("origin") or {}).get("jwks")
        auth_issuer = getattr(config, "AUTHORITY_ISSUER", None)
        if isinstance(origin_jwks, dict) and auth_issuer:
            desired.append({"issuer": auth_issuer, "jwks": origin_jwks, "role": "platform"})

    for d in getattr(config, "TRUSTED_ISSUERS", []) or []:
        if isinstance(d, dict) and d.get("issuer"):
            desired.append({**d, "role": d.get("role", "external")})

    created = 0
    for d in desired:
        if d["issuer"] in existing:
            continue
        try:
            create_issuer_artifact(
                db, issuer=d["issuer"], authorized_by="bootstrap:manifest",
                jwks=d.get("jwks"), jwks_uri=d.get("jwks_uri"),
                audience=d.get("audience"), namespace=d.get("namespace"),
                role=d.get("role", "external"),
            )
            existing.add(d["issuer"])
            created += 1
        except Exception:
            logger.warning("issuer seed: failed for %s", d["issuer"], exc_info=True)
    if created:
        logger.info("issuer seed: created %d platform/env issuer artifact(s)", created)
    return created


def revoke_issuer_artifact(db: Any, artifact_id: str, *, by: str) -> bool:
    """Revoke trust in an issuer by archiving its artifact (the loader reads only
    committed). Returns False if not an issuer artifact. Emits artifact.updated ->
    the watcher drops it from the trust set."""
    from mantle.db import backend as db_store
    art = db_store.get_artifact(db, artifact_id)
    if art is None:
        return False
    try:
        ctx = json.loads(art.context) if isinstance(art.context, str) else (art.context or {})
    except (json.JSONDecodeError, TypeError):
        ctx = {}
    if ctx.get("content_type") != ISSUER_CONTENT_TYPE:
        return False
    art.state = "archived"
    art.modified_by = by
    db_store.update_artifact(db, art)  # emits artifact.updated -> watcher refreshes
    logger.info("issuer revoke: %s (by=%s)", artifact_id, by)
    return True


# ---------------------------------------------------------------------------
# Event-driven refresh
#
# The verifier's trust set is reloaded the instant an issuer artifact changes, so
# a governed issuer edit takes effect immediately — no restart, no waiting on the
# throttled refresh-on-miss (which remains the bounded fallback for events that
# can't reach this process, e.g. multi-replica where the in-process bus would be a
# Redis adapter).
# ---------------------------------------------------------------------------

# An event signals an issuer change when its name matches AND it carries our
# content_type (the db chokepoint stamps content_type on artifact.created/updated;
# the gateway emits issuer.* on typed ops). A full reload on any such event means a
# delete/rotate is reflected by re-reading the committed set.
_ISSUER_EVENT_NAMES = [
    "artifact.created", "artifact.updated", "artifact.deleted", "issuer.*",
]


def _handle_issuer_event(event: Any, get_db: Any = None) -> None:
    """Reload the verifier's trust set in response to an issuer-artifact event."""
    try:
        from mantle.services.oidc import get_oidc_verifier
        if get_db is None:
            from mantle.services.dependencies import get_store_db as get_db
        db = next(get_db())
        get_oidc_verifier().refresh_from_db(db)
        logger.info("issuer watch: trust refreshed on %s", getattr(event, "name", "?"))
    except Exception:
        logger.warning("issuer watch: refresh failed", exc_info=True)


async def watch_issuer_changes() -> None:
    """Subscribe to issuer-artifact change events and refresh the verifier on each.

    Runs for the app's lifetime (started in the lifespan). The throttled
    refresh-on-miss in ``resolve_auth`` is the fallback for missed events."""
    from mantle.events import event_bus
    flt = event_bus.EventFilter(
        content_type=ISSUER_CONTENT_TYPE, event_names=_ISSUER_EVENT_NAMES,
    )
    q = await event_bus.subscribe_filtered(flt)
    logger.info("issuer watch: subscribed to issuer-artifact change events")
    try:
        while True:
            event = await q.get()
            _handle_issuer_event(event)
    finally:
        await event_bus.unsubscribe_filtered(q)
