"""Role presets are Role artifacts in the platform's Roles collection.

A share or invite names a role; Mantle answers what it grants by reading the Role artifacts in
the collection now, so an operator edits a role, or adds one, by editing the collection. Each
Role's context is platform configuration kept in the open (`doc_boundary._readable`):
`{"role": <name>, "actions": [<CRUDEASIO action>, ...], "platform": true}`.

The platform seeds the collection and one Role per `Grant.ROLE_PRESETS` entry, each only when it
is missing: an operator's edit stands across restarts. Before the collection is seeded (a node
with no instance namespace yet), the presets are the seed itself.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from mantle.attenuation import ACTIONS
from mantle.db.store import Database
from mantle.entities.grant import Grant

logger = logging.getLogger(__name__)

ROLE_CT = "application/vnd.agience.role+json"
COLLECTION_CT = "application/vnd.agience.collection+json"
_NS = "platform"
_COLLECTION_SLUG = "roles"


def _role_slug(name: str) -> str:
    return f"role-{name}"


def roles_collection_id() -> str:
    """The Roles collection's id on this node, or empty before the instance namespace is known."""
    from mantle.services.seed_provisioning.loader import derive_uuid, get_instance_namespace
    ns = get_instance_namespace()
    return derive_uuid(ns, _NS, _COLLECTION_SLUG) if ns else ""


def role_id(name: str) -> str:
    """The seeded Role `name`'s id on this node, or empty before the instance namespace is known."""
    from mantle.services.seed_provisioning.loader import derive_uuid, get_instance_namespace
    ns = get_instance_namespace()
    return derive_uuid(ns, _NS, _role_slug(name)) if ns else ""


def _actions_of(bits: Dict[str, bool]) -> List[str]:
    return [a for a in ACTIONS if bits.get(f"can_{a}")]


def _context(name: str, actions: List[str]) -> dict:
    return {"content_type": ROLE_CT, "title": name.capitalize(), "role": name,
            "actions": actions, "platform": True}


def _seeded() -> Dict[str, Dict[str, bool]]:
    return {name: dict(bits) for name, bits in Grant.ROLE_PRESETS.items()}


def ensure_roles(db: Database, operator: str) -> bool:
    """Create the Roles collection and each seeded Role that is missing. An existing Role is left
    as it is. False when not yet possible."""
    from mantle.db import lattice_api as la
    from mantle.services import workspace_service
    from mantle.services.seed_provisioning.platform_email import _ensure_artifact
    from mantle.services.seed_provisioning.platform_operators import _exists

    coll = roles_collection_id()
    if not operator or not coll:
        return False
    if not _exists(db, coll):
        workspace_service.create_container(
            db, operator, content_type=COLLECTION_CT, name="Roles",
            description="The roles a share or invite offers; each Role lists the actions it grants.",
            context=json.dumps({"content_type": COLLECTION_CT, "title": "Roles"}),
            artifact_id=coll)
    for name, bits in Grant.ROLE_PRESETS.items():
        rid = role_id(name)
        if _exists(db, rid):
            continue
        _ensure_artifact(db, rid, content_type=ROLE_CT, name=name.capitalize(),
                         description=f"What the {name} role grants.",
                         context=_context(name, _actions_of(bits)), content="", owner=operator)
        if la._find_membership(db, coll, rid) is None:
            la.add_artifact_to_collection(db, coll, rid, origin=True)
    return True


def _ctx(doc: dict) -> dict:
    raw = doc.get("context")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def presets(db: Optional[Database]) -> Dict[str, Dict[str, bool]]:
    """{role name: CRUDEASIO bit map} from the Role artifacts in the Roles collection, fewest
    actions first; the seed when the collection holds none."""
    coll = roles_collection_id()
    if db is None or not coll:
        return _seeded()
    from mantle.db import lattice_api as la
    try:
        members = la.list_collection_artifacts(db, coll)
    except Exception:  # noqa: BLE001 — not seeded here yet
        return _seeded()
    found: Dict[str, List[str]] = {}
    for doc in members:
        ctx = _ctx(doc)
        if (ctx.get("content_type") or doc.get("content_type")) != ROLE_CT or ctx.get("platform") is not True:
            continue
        name = str(ctx.get("role") or "").strip().lower()
        actions = ctx.get("actions")
        if not name or not isinstance(actions, list):
            continue
        found[name] = [a for a in ACTIONS if a in {str(x) for x in actions}]
    if not found:
        return _seeded()
    ordered = sorted(found.items(), key=lambda kv: (len(kv[1]), kv[0]))
    return {name: {f"can_{a}": True for a in actions} for name, actions in ordered}


def permissions_for_role(db: Optional[Database], role: str) -> Dict[str, bool]:
    """The CRUDEASIO bit pattern the Role `role` grants. Raises `ValueError` for an unknown role."""
    known = presets(db)
    bits = known.get(role)
    if bits is None:
        raise ValueError(f"Unknown role {role!r}; valid roles: {sorted(known)}")
    return dict(bits)
