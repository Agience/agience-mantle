"""Authorization record access for Mantle — SOVEREIGN.

Mantle is the authority for its own data plane: grants + API keys live in Mantle's
OWN STORE — the SQLite lattice, reached through ``db.backend`` — and Mantle enforces
authorization (the light-cone + grant checks in ``services.dependencies``) with NO call
to Origin. Origin is identity-only.

Historically this module was pluggable (`local` vs `origin` backend) so a
deployment could centralise authorization in Origin. That mode has been removed —
the platform is sovereign-only. The `get_*_backend()` accessors remain so existing
call sites don't churn; they always return the local (the lattice) backend.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from mantle.entities.api_key import APIKey as APIKeyEntity
from mantle.entities.grant import Grant as GrantEntity, grant_is_allow, grant_is_deny

logger = logging.getLogger(__name__)

# action → grant boolean flag (mirrors services.dependencies)
_ACTION_FLAGS = {
    "create": "can_create", "read": "can_read", "update": "can_update",
    "delete": "can_delete", "evict": "can_evict", "invoke": "can_invoke",
    "add": "can_add", "share": "can_share", "admin": "can_admin",
}


def _grant_dict(g: GrantEntity) -> dict:
    if hasattr(g, "to_dict"):
        try:
            return g.to_dict()
        except Exception:
            pass
    return {"id": getattr(g, "id", None), "effect": getattr(g, "effect", None)}


class LocalBackend:
    """Grants + API keys from Mantle's own lattice. No Origin dependency."""

    def verify_api_key(self, db, token) -> Optional[Tuple[APIKeyEntity, List[GrantEntity]]]:
        if db is None:
            return None
        from mantle.db import backend as dba
        from mantle.services.auth_service import hash_api_key
        key = dba.get_api_key_by_hash(db, hash_api_key(token))  # filters is_active + expiry
        if key is None:
            return None
        grants = dba.get_active_grants_for_grantee(db, grantee_id=str(key.id), grantee_type="api_key")
        return key, grants

    def check_grant(self, db, *, principal_id, resource_id, action) -> Optional[dict]:
        from mantle.db import backend as dba
        flag = _ACTION_FLAGS.get(action)
        grants = dba.get_active_grants_for_principal_resource(db, grantee_id=principal_id, resource_id=resource_id)
        allowed = False
        if flag:
            if any(getattr(g, flag, False) and grant_is_deny(g) for g in grants):
                allowed = False
            else:
                allowed = any(getattr(g, flag, False) and grant_is_allow(g) for g in grants)
        return {"allowed": bool(allowed), "grants": [_grant_dict(g) for g in grants]}

    def list_grants_by_principal_resource(self, db, *, grantee_id, resource_id) -> List[GrantEntity]:
        from mantle.db import backend as dba
        return dba.get_active_grants_for_principal_resource(db, grantee_id=grantee_id, resource_id=resource_id)

    def list_grants_by_grantee(self, db, grantee_id, grantee_type="user") -> List[GrantEntity]:
        from mantle.db import backend as dba
        return dba.get_active_grants_for_grantee(db, grantee_id=grantee_id, grantee_type=grantee_type)

    def lookup_grants_by_key(self, db, token) -> List[GrantEntity]:
        from mantle.db import backend as dba
        return dba.get_active_grants_by_key(db, token)

    def upsert_user_grant(self, db, *, user_id, resource_id, granted_by, flags, name=None) -> Tuple[Optional[GrantEntity], bool]:
        from mantle.db import backend as dba
        return dba.upsert_user_collection_grant(
            db, user_id=user_id, collection_id=resource_id, granted_by=granted_by,
            can_create=flags.get("can_create", False), can_read=flags.get("can_read", True),
            can_update=flags.get("can_update", False), can_delete=flags.get("can_delete", False),
            can_evict=flags.get("can_evict", False), can_invoke=flags.get("can_invoke", False),
            can_add=flags.get("can_add", False), can_share=flags.get("can_share", False),
            can_admin=flags.get("can_admin", False), name=name,
        )


_LOCAL = LocalBackend()


def get_apikey_backend() -> LocalBackend:
    """API-key verification backend (sovereign: always Mantle's the lattice)."""
    return _LOCAL


def get_grant_backend() -> LocalBackend:
    """Grant read/write backend (sovereign: always Mantle's the lattice)."""
    return _LOCAL
