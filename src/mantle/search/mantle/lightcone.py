"""LightConeResolver — BFS over `collection_artifacts` origin edges.

Resolves the set of artifact IDs reachable from a principal's grants by
walking ``origin: true, relationship: null`` edges outbound through
the lattice, intersecting each edge's ``propagate`` mask with the requested
action.

CRUDEASIO lives in Mantle (the lattice grants collection). Grants are read
directly from `db_store.get_active_grants_for_grantee` — no Origin HTTP
calls. This keeps the resolver on the same data source as `check_access`.

After the lexical-backend retirement (Step 2.6.9), this is the only ACL path —
both MANTLE-SSE lexical and MANTLE vector search consume the resolver's
authorized artifact set. The legacy flat ACL filter that lived in the
the legacy lexical index query builder is gone.

See `.dev/features/mantle-mvp.md` § Layer 1.
"""

from __future__ import annotations

from typing import Optional, Set

from mantle.db import backend as db_store
from mantle.services.dependencies import _ACTION_FLAG_MAP


#: The ledger's ``grantee_type`` and the acting context's ``principal_type`` are
#: DIFFERENT AXES, and this is the one place that says so.
#:
#: ``grantee_type`` is a CREDENTIAL kind — a grant is held either by a principal id
#: (``"user"``) or by a hashed key (``"api_key"`` / ``"grant_key"``). That is the
#: whole vocabulary; see ``lattice_api.upsert_user_collection_grant``, which hard-codes
#: ``grantee_type="user"`` for every principal-held grant it mints.
#:
#: ``ActingPrincipal.principal_type`` is a wider ENTITY vocabulary —
#: ``user | api_key | server | mcp_client | grant_key | service | delegation``.
#:
#: ⛔ THIS MAPPING IS DERIVED FROM THE LEDGER, NOT CHOSEN. FORWARDING
#: ``principal_type`` VERBATIM IS A REGRESSION, NOT A FIX.
#: ``seed_provisioning/platform_email.py`` issues the platform SYSTEM principal
#: (``principal_type="service"``) its operator/authorizer/secret grants through
#: ``upsert_user_collection_grant``, which stores them as ledger ``"user"`` grants.
#: Passing ``"service"`` through as ``grantee_type`` matches nothing: every
#: system-principal grant in every existing store becomes invisible, and the
#: platform's own background work takes a SILENT FALSE DENY that returns the same
#: result as "no such grant". An entity kind with no credential of its own holds its
#: grants as a principal id, which in this ledger is spelled ``"user"``.
PRINCIPAL_GRANT_TYPE = "user"
_LEDGER_KEY_TYPES = frozenset({"api_key", "grant_key"})


def ledger_grantee_type(principal_type: Optional[str]) -> str:
    """Map an acting-context ``principal_type`` to the ledger's ``grantee_type``.

    Key-shaped principals keep their own kind (they hold grants as a hashed
    credential); every other entity kind acts as a principal id and therefore holds
    :data:`PRINCIPAL_GRANT_TYPE` grants. See the comment above for why this is not
    the identity function.
    """
    return principal_type if principal_type in _LEDGER_KEY_TYPES else PRINCIPAL_GRANT_TYPE


class LightConeResolver:
    """BFS over origin edges with `propagate` masks.

    ⛔ `max_depth=4` REMOVED 2026-07-30. The BFS terminates on its own `seen` set over a finite
    graph, so the cap only truncated. A grant more than 4 levels above an artifact produced a
    FALSE DENY that is indistinguishable from "no such artifact" — the authorization surface
    silently shrank with lattice depth."""

    def __init__(self, db) -> None:
        self._db = db

    def resolve(
        self,
        principal_id: str,
        action: str = "read",
        *,
        principal_type: str = "user",
    ) -> Set[str]:
        """Return artifact IDs the principal can reach for ``action``.

        Two-step traversal:

        1. Fetch the principal's grants from the lattice (grants collection). Filter
           to grants that allow ``action`` (CRUDEASIO flag check) and have a
           non-empty ``resource_id``.
        2. For each granted resource, BFS outbound through `origin: true,
           relationship: null`` edges in the lattice, pruning when an edge's
           ``propagate`` mask doesn't include the action. The traversal runs
           to exhaustion; `seen` is the termination guard.

        The returned set is the union of directly-granted IDs plus all
        descendants reachable through an unbroken chain of action-permitted
        origin edges.

        ``principal_type`` is the caller's ACTING-CONTEXT entity kind, not a ledger
        ``grantee_type``; :func:`ledger_grantee_type` maps between the two axes and
        its comment explains why they are not the same vocabulary.

        Returns an empty set when the principal has no relevant grants or
        the action name is unknown.
        """
        flag_attr: Optional[str] = _ACTION_FLAG_MAP.get(action)
        if flag_attr is None:
            return set()

        grants = db_store.get_active_grants_for_grantee(
            self._db,
            grantee_id=principal_id,
            grantee_type=ledger_grantee_type(principal_type),
        )
        granted_ids = [
            g.resource_id
            for g in grants
            if g.resource_id and getattr(g, flag_attr, False)
        ]
        if not granted_ids:
            return set()

        result: Set[str] = set(granted_ids)
        descendants = db_store.list_origin_descendants(self._db, granted_ids, action)
        result.update(descendants)
        return result
