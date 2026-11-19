"""Bearer tokens an artifact issues (`services.on_create`, the ``token`` stamp).

A type whose field carries ``on_create: {"stamp": "token", "prefix": <prefix>, ...}`` makes each of
its artifacts a credential. The token is ``<prefix><root id>.<secret>``: it names the artifact that
holds its hash, so resolving it is one read of that lineage, with no index of hashes.

A token resolves while:

- the artifact's current version is live: archiving or deleting it ends the token;
- every collection it was created under is live: archiving the approval it sits in ends it too;
- its hash matches, compared in constant time;
- its ``ends`` field, when the rule names one, is in the future.

It authenticates as the person who created the artifact (the subject), acted by the agent its
``acts`` field names (the actor) or, when that is empty, as the person's own key, held to the ``reach`` ids and the ``actions`` the rule names (`AGENT_SCOPE`). Those
bounds are checked on every request by `check_access`, which also never goes past what the person
holds. A token that does not resolve is refused; it never falls through to another credential.
"""
from __future__ import annotations

import hmac
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

#: The shape of an artifact token: a type's prefix, the artifact's root id, a dot, the secret.
TOKEN_SHAPE = re.compile(r"^([a-z]{2,8}_)([A-Za-z0-9-]{1,128})\.([A-Za-z0-9_-]{20,})$")


def looks_like(token: str) -> bool:
    """Is ``token`` shaped like an artifact token? Grant keys carry no dot, and JWTs start
    ``ey`` with no underscore prefix."""
    return isinstance(token, str) and TOKEN_SHAPE.match(token) is not None


def _ctx(doc: Dict[str, Any]) -> Dict[str, Any]:
    ctx = doc.get("context")
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except ValueError:
            return {}
    return ctx if isinstance(ctx, dict) else {}


def _current(store_db: Any, root_id: str) -> Optional[Dict[str, Any]]:
    """The newest version of ``root_id`` when it is live; None when it is archived or gone."""
    from mantle.db.constants import STATE_LIVE, state_of
    versions = store_db.artifacts.versions_of(root_id) or []
    if not versions:
        return None
    doc = versions[-1]
    return doc if state_of(doc) == STATE_LIVE else None


def _chain_live(store_db: Any, root_id: str) -> bool:
    """Is every collection ``root_id`` was created under live?"""
    from mantle.db import lattice_api
    current, seen = root_id, {root_id}
    while True:
        parent = lattice_api.get_origin_parent(store_db, current)
        if not parent or not parent[0]:
            return True
        current = parent[0]
        if current in seen:
            return True
        seen.add(current)
        if _current(store_db, current) is None:
            return False


def resolve(store_db: Any, token: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], dict]]:
    """``(artifact, context, rule)`` for a live token, else None."""
    from mantle.services import on_create, type_offers
    m = TOKEN_SHAPE.match(token or "")
    if m is None:
        return None
    prefix, root_id = m.group(1), m.group(2)
    doc = _current(store_db, root_id)
    if doc is None:
        return None
    ctx = _ctx(doc)
    found = type_offers.token_rule(doc.get("content_type") or ctx.get("content_type") or "")
    if found is None:
        return None
    field, rule = found
    stored = ctx.get(field)
    if rule.get("prefix") != prefix or not isinstance(stored, str):
        return None
    if not hmac.compare_digest(stored, on_create.token_hash(token)):
        return None
    if rule.get("ends"):
        until = on_create.parse_time(ctx.get(rule["ends"]))
        if until is None or until <= datetime.now(timezone.utc):
            return None
    if not _chain_live(store_db, root_id):
        return None
    return doc, ctx, rule


def auth_of(store_db: Any, token: str):
    """The `AuthContext` a live token authenticates as, or None."""
    from mantle.services.dependencies import AGENT_SCOPE, JOB_REACH_ACTIONS, AuthContext
    found = resolve(store_db, token)
    if found is None:
        return None
    doc, ctx, rule = found
    person = str(doc.get("created_by") or "")
    if not person:
        return None
    reach = tuple(str(i) for i in (ctx.get(rule["reach"]) or [])) if rule.get("reach") else ()
    actions = (frozenset(ctx.get(rule["actions"]) or []) & JOB_REACH_ACTIONS
               if rule.get("actions") else frozenset())
    actor = str(ctx.get(rule["acts"]) or "") if rule.get("acts") else ""
    return AuthContext(
        principal_id=person, principal_type="user", user_id=person,
        actor=actor or None,
        scope=AGENT_SCOPE, job_reach=reach, reach_actions=actions,
        token_artifact=str(doc.get("root_id") or doc.get("id")),
        token_ends=ctx.get(rule["ends"]) if rule.get("ends") else None,
    )


__all__ = ["TOKEN_SHAPE", "auth_of", "looks_like", "resolve"]
