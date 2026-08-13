"""Grant keys and grant bundles — bearer credentials assembled out of grants.

A **grant key** is an ordinary :class:`~mantle.entities.grant.Grant` whose grantee is a
hashed bearer token rather than a user: ``grantee_type="grant_key"``,
``grantee_id=sha256(raw_token)``. Presenting the token in the Bearer slot makes the
holder act *as that grant* — its CRUDEASIO bits are the entire extent of what they may
do. There is no second permission vocabulary layered on top; the grant IS the scope.

A **grant bundle** is the same object with members. A member is a grant whose grantee is
the bundle itself (``grantee_type="grant"``, ``grantee_id=<bundle.id>``), which means
composition reuses :func:`get_active_grants_for_grantee` — the same lookup that expands
any other grantee — instead of introducing a second kind of edge with its own traversal,
its own expiry handling, and its own chance to disagree with the first.

Two consequences worth stating, because they are the point of the design:

* Members carry **independent** CRUDEASIO bits, so a single key can be read-only on one
  collection and read/write on another. Below each member, the existing per-edge
  ``propagate`` masks keep doing the fan-out (see :mod:`search.mantle.lightcone`).
* The bundle grant is a **ceiling**. Effective permission is
  ``member ∩ bundle`` (:meth:`Grant.masked_by`), so clearing one bit on the bundle
  narrows every member at once, and revoking the bundle revokes the whole set —
  without touching, or needing to find, any member.

Minting a key requires ``can_admin`` on the resource, and so does adding a member. The
bundle's own bits therefore never widen anything: every member was independently
authorized by someone who could already grant it.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from mantle.entities.grant import Grant as GrantEntity

logger = logging.getLogger(__name__)

#: Raw-token prefix. Deliberately distinct from the retired ``agc_`` API keys so a
#: stale key fails as "not a grant key" at parse time rather than as a hash miss —
#: the two are different diagnoses and only one of them means "your token is wrong".
KEY_PREFIX = "agk_"

#: Bundles may nest. The ceiling is an operational bound against a malformed store
#: (a cycle, or a chain built by a buggy writer), not a statement about how deep real
#: bundles go — exceeding it is a server fault, not a denial.
_MAX_BUNDLE_DEPTH = 16


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_token() -> str:
    """A fresh raw bearer token. Never stored — only its hash is."""
    return f"{KEY_PREFIX}{secrets.token_urlsafe(32)}"


def hash_token(raw_token: str) -> str:
    """The stored form of a bearer token."""
    return hashlib.sha256(raw_token.encode()).hexdigest()


def key_hint(raw_token: str) -> str:
    """Non-secret tail, so a human can tell two keys apart in a list."""
    return raw_token[-4:] if len(raw_token) >= 4 else ""


def _flags_from(
    flags: Optional[Dict[str, bool]] = None,
    role: Optional[str] = None,
) -> Dict[str, bool]:
    """Normalize a permission spec to a full CRUDEASIO map.

    Unspecified flags are False, not absent: a grant built from a partial map would
    inherit the entity's constructor defaults (``can_read=True``), which silently
    widens a caller who passed ``{"can_invoke": True}`` meaning only that.
    """
    if role:
        resolved = GrantEntity.permissions_for_role(role)
    else:
        resolved = dict(flags or {})
    return {flag: bool(resolved.get(flag, False)) for flag in GrantEntity.PERMISSION_FLAGS}


def _open_ceiling() -> Dict[str, bool]:
    """The bits a bundle root carries when the caller did not narrow it.

    All-True looks alarming and is not: a bundle root has no ``resource_id``, so it
    reaches nothing by itself, and every member it carries was separately authorized by
    someone holding ``can_admin`` on that member's resource. The ceiling exists to be
    *narrowed* later — starting it closed would mean a freshly-minted bundle silently
    granted nothing and every member had to be re-stated on the root.
    """
    return {flag: True for flag in GrantEntity.PERMISSION_FLAGS}


# ---------------------------------------------------------------------------
# Minting
# ---------------------------------------------------------------------------

def mint(
    db,
    *,
    user_id: str,
    name: str,
    resource_id: Optional[str] = None,
    flags: Optional[Dict[str, bool]] = None,
    role: Optional[str] = None,
    expires_at: Optional[str] = None,
    notes: Optional[str] = None,
) -> Tuple[GrantEntity, str]:
    """Create a grant key and return ``(grant, raw_token)``.

    The raw token is returned exactly once and is not recoverable afterwards — only
    its hash is stored, in ``grantee_id``.

    With *resource_id*, this is a single-resource key carrying *flags*. Without one,
    it is a bundle root: it reaches nothing on its own and acts purely as the ceiling
    over members added via :func:`add_member`.

    Authorization is the CALLER's job — mint only after checking ``can_admin`` on
    *resource_id*. This function is also reached by internal issuers (workspace card
    keys) that have already established authority by other means.
    """
    from mantle.db import backend as store

    raw_token = generate_token()
    now = _now_iso()

    if resource_id:
        bits = _flags_from(flags, role)
    else:
        # A bundle root with explicit flags is a caller deliberately pre-narrowing the
        # ceiling; without them it stays open. See `_open_ceiling`.
        bits = _flags_from(flags, role) if (flags or role) else _open_ceiling()

    grant = GrantEntity(
        id=str(uuid.uuid4()),
        resource_id=resource_id or "",
        grantee_type=GrantEntity.GRANTEE_GRANT_KEY,
        grantee_id=hash_token(raw_token),
        granted_by=user_id,
        key_hint=key_hint(raw_token),
        name=name,
        notes=notes,
        state=GrantEntity.STATE_ACTIVE,
        granted_at=now,
        expires_at=expires_at,
        created_time=now,
        modified_time=now,
        **bits,
    )
    store.create_grant(db, grant)
    invalidate_for(db, grant)
    return grant, raw_token


def add_member(
    db,
    *,
    bundle_id: str,
    resource_id: str,
    granted_by: str,
    flags: Optional[Dict[str, bool]] = None,
    role: Optional[str] = None,
    name: Optional[str] = None,
    expires_at: Optional[str] = None,
) -> GrantEntity:
    """Attach a resource to a bundle with its own CRUDEASIO bits.

    The member is a grant granted TO the bundle — that is the edge. Authorization
    (``can_admin`` on *resource_id*) is the caller's job.
    """
    from mantle.db import backend as store

    now = _now_iso()
    member = GrantEntity(
        id=str(uuid.uuid4()),
        resource_id=resource_id,
        grantee_type=GrantEntity.GRANTEE_GRANT,
        grantee_id=bundle_id,
        granted_by=granted_by,
        name=name,
        state=GrantEntity.STATE_ACTIVE,
        granted_at=now,
        expires_at=expires_at,
        created_time=now,
        modified_time=now,
        **_flags_from(flags, role),
    )
    store.create_grant(db, member)
    # NOT `_invalidate(bundle_id)`: a nested bundle names its immediate parent, while the
    # memo is keyed on the ROOT key. `invalidate_for` walks up to it.
    invalidate_for(db, member)
    return member


def list_members(db, bundle_id: str) -> List[GrantEntity]:
    """The bundle's direct members (one level, unmasked)."""
    from mantle.db import backend as store
    return store.get_active_grants_for_grantee(
        db, grantee_id=bundle_id, grantee_type=GrantEntity.GRANTEE_GRANT
    )


def list_keys_issued_by(db, user_id: str, include_revoked: bool = False) -> List[GrantEntity]:
    """Grant keys this user minted. Callers must redact the stored token hash."""
    from mantle.db.backend import query_documents, COLLECTION_GRANTS

    filters: dict = {
        "grantee_type": GrantEntity.GRANTEE_GRANT_KEY,
        "granted_by": user_id,
    }
    if not include_revoked:
        filters["state"] = GrantEntity.STATE_ACTIVE
    return query_documents(db, GrantEntity, COLLECTION_GRANTS, filters)


# ---------------------------------------------------------------------------
# Authentication + resolution
# ---------------------------------------------------------------------------

def authenticate(db, raw_token: str) -> Optional[GrantEntity]:
    """The active grant behind a raw bearer token, or None.

    Looks up by hash, so a wrong token is a miss rather than a comparison — there is
    no secret-dependent branch here to time.
    """
    if not raw_token or not raw_token.startswith(KEY_PREFIX):
        return None
    from mantle.db import backend as store

    matches = store.get_active_grants_for_grantee(
        db, grantee_id=hash_token(raw_token),
        grantee_type=GrantEntity.GRANTEE_GRANT_KEY,
    )
    return matches[0] if matches else None


def resolve(db, root: GrantEntity) -> List[GrantEntity]:
    """Every grant the bearer of *root* effectively holds.

    The root itself when it names a resource, plus each member of the bundle narrowed
    by the root's bits, recursively for nested bundles. Members are masked at every
    level, so a nested bundle can only ever narrow further.

    Returns detached copies; nothing here writes.
    """
    effective: List[GrantEntity] = []
    if root.resource_id:
        effective.append(root)

    seen = {root.id}
    # (bundle grant whose members to expand, ceiling those members are masked by)
    frontier: List[Tuple[GrantEntity, GrantEntity]] = [(root, root)]

    for _ in range(_MAX_BUNDLE_DEPTH):
        if not frontier:
            break
        nxt: List[Tuple[GrantEntity, GrantEntity]] = []
        for bundle, ceiling in frontier:
            for member in list_members(db, bundle.id):
                if member.id in seen:      # cycle guard: a malformed store must not hang auth
                    continue
                seen.add(member.id)
                masked = member.masked_by(ceiling)
                if masked.resource_id:
                    effective.append(masked)
                # A member may itself carry members (a nested bundle). It becomes the
                # next ceiling in its own MASKED form, so narrowing compounds down the
                # chain rather than resetting at each level.
                nxt.append((member, masked))
        frontier = nxt
    else:
        if frontier:
            logger.warning(
                "grant bundle exceeded depth %d; truncating (root=%s)",
                _MAX_BUNDLE_DEPTH, root.id,
            )

    return effective


def touch(db, root: GrantEntity) -> None:
    """Record use of a key. Best-effort — never fails a request."""
    try:
        from mantle.db import backend as store
        root.last_used_at = _now_iso()
        store.update_grant(db, root)
    except Exception:
        logger.debug("grant-key touch failed", exc_info=True)


def revoke(db, grant: GrantEntity, revoked_by: str) -> bool:
    """Soft-revoke a key or member. Members are left in place — the bundle they hang
    off is gone, so they resolve to nothing."""
    from mantle.db import backend as store

    now = _now_iso()
    grant.state = GrantEntity.STATE_REVOKED
    grant.revoked_by = revoked_by
    grant.revoked_at = now
    grant.modified_time = now
    updated = store.update_grant(db, grant) is not None
    invalidate_for(db, grant)
    return updated


# ---------------------------------------------------------------------------
# Cache invalidation — translating a grant into the principals it affects
# ---------------------------------------------------------------------------
#
# This is the ONE translator, and every mutation path goes through it, because the
# translation is not the identity function and getting it wrong is silent: the write
# lands, the invalidation call is made, and a memo entry nobody named keeps issuing
# keys against a grant the ledger has already withdrawn.


def principal_ids_for(db, grant: GrantEntity) -> Optional[set]:
    """The acting-principal ids whose memoized reachability *grant* changes.

    ``None`` means "cannot tell" and obliges the caller to clear the whole memo. That
    is the cheap side of the uncertainty: an over-wide clear costs one light-cone walk
    per affected principal, an under-wide one leaves a live key behind a dead grant.

    NOT ``grantee_id`` in the key cases, which is why this exists. The oracle memoizes
    by acting-principal id (``dependencies.resolve_auth`` → ``AuthContext.principal_id``
    → ``acting_principal``), and:

    * for a **grant key** that is the ROOT GRANT's id, while the stored ``grantee_id``
      is the token hash — a value the cache has never seen;
    * for a **bundle member** it is the root key at the top of the bundle chain, which
      ``grantee_id`` names only when the bundle is one level deep. A member of a nested
      bundle names the inner bundle, and the memo is keyed on the root.

    Everything else acts under its own id, which ``grantee_id`` does name.
    """
    from mantle.db import backend as store

    if grant.grantee_type == GrantEntity.GRANTEE_GRANT_KEY:
        return {grant.id} if grant.id else None

    if grant.grantee_type == GrantEntity.GRANTEE_GRANT:
        # Walk up the bundle chain to the root key. Every id on the way is included:
        # they are a handful of strings, an intermediate bundle can itself be a root
        # somebody holds directly, and clearing one extra entry costs one re-walk.
        ids: set = set()
        current = grant.grantee_id or ""
        for _ in range(_MAX_BUNDLE_DEPTH):
            if not current or current in ids:
                break                       # nowhere left to walk, or a cycle
            ids.add(current)
            parent = store.get_grant_by_id(db, current)
            if parent is None:
                break                       # the chain is broken; the root is unknown
            if parent.grantee_type != GrantEntity.GRANTEE_GRANT:
                return ids                  # `current` IS the root key — the principal
            current = parent.grantee_id or ""
        # Fell out of the walk without reaching a root: broken chain, cycle, or deeper
        # than the bound. Which principal is affected is now a guess, so clear all.
        return None

    if grant.grantee_type == GrantEntity.GRANTEE_GROUP:
        # A group is not a principal: nobody acts as one, so the memo is keyed on the
        # MEMBER ids — which this grant does not name and which no lookup here resolves.
        # Group grants reach nothing today (`lightcone.ledger_grantee_type` maps every
        # non-key principal to `user`), so this costs nothing now and stays correct on
        # the day expansion is added rather than becoming a silent miss then.
        return None

    if grant.grantee_type == GrantEntity.GRANTEE_INVITE:
        # `grantee_id` is a hashed claim token, so it names no principal and there is
        # nothing precise to clear. No memo entry can depend on an unclaimed invite —
        # no grantee lookup returns one — but invites are mutated rarely enough that
        # paying for a full clear is cheaper than being wrong about that.
        return None

    return {grant.grantee_id} if grant.grantee_id else None


def invalidate_for(db, grant: GrantEntity) -> None:
    """Drop every light-cone memo entry *grant*'s change can affect.

    Call after the write lands, not before: an entry dropped ahead of the commit can be
    refilled from the pre-change ledger by a concurrent request.
    """
    ids = principal_ids_for(db, grant)
    if ids is None:
        _invalidate(None)                   # None == clear everything (see `wiring`)
        return
    for principal_id in ids:
        _invalidate(principal_id)


def _invalidate(principal_id: Optional[str]) -> None:
    """Drop the light-cone memo for a principal after its reachability changed.

    ``None`` clears the whole memo. `LightConeGrantVerifier` memoizes
    (requester, type, action) for its TTL, so without this a key minted or revoked
    inside that window keeps answering with the old verdict — which for revocation is
    the security-relevant direction.

    Process-local. See `wiring.invalidate_grant_cache` for what that means under more
    than one worker, and what the wiring does about it.
    """
    try:
        from mantle.search.mantle.wiring import invalidate_grant_cache
        invalidate_grant_cache(principal_id or None)
    except Exception:      # a stale cache is a delay, not a failure
        logger.debug("grant-cache invalidation failed", exc_info=True)
