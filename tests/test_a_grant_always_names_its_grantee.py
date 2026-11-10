"""Every grant names its grantee. 2026-11-09: seeding Origin's ledger found two grants with an
empty grantee, written by a direct grant whose request named none. They reach no one, and Origin
refuses them as malformed. The store refuses to write one, the route refuses the request, and
`mantle.system.drop_ungranteed_grants` drops the ones already stored."""
from __future__ import annotations

import pytest

from mantle.db import lattice_api as la
from mantle.entities.grant import Grant
from mantle.system import drop_ungranteed_grants as dug


def _grant(gid, grantee):
    return Grant(id=gid, resource_id="W", grantee_type="user", grantee_id=grantee,
                 granted_by="alice", can_read=True)


@pytest.fixture
def db(tmp_path):
    return la.LatticeDatabase(str(tmp_path / "g.db"), origin="g")


@pytest.mark.parametrize("grantee", ["", "   ", None])
def test_the_store_refuses_a_grant_with_no_grantee(db, grantee):
    with pytest.raises(la.GrantWithoutGrantee):
        la.create_grant(db, _grant("g-x", grantee))
    assert la.get_grant_by_id(db, "g-x") is None
    with pytest.raises(la.GrantWithoutGrantee):
        la.update_grant(db, _grant("g-x", grantee))


def test_the_stored_ones_are_dropped_and_the_rest_kept(db):
    la.create_grant(db, _grant("g-ok", "bob"))
    for gid in ("g-a", "g-b"):        # as an earlier writer stored them, past the check
        doc = la.to_lattice_doc(_grant(gid, ""))
        doc["content_type"] = la._GRANT_CT
        db.artifacts.put_artifact(doc)
    assert dug.ungranteed(db) == ["g-a", "g-b"]
    assert dug.drop_ungranteed_grants(db, dry_run=True) == {"found": 2, "dropped": 0}
    assert dug.drop_ungranteed_grants(db) == {"found": 2, "dropped": 2}
    assert dug.ungranteed(db) == [] and la.get_grant_by_id(db, "g-ok") is not None
    assert dug.drop_ungranteed_grants(db) == {"found": 0, "dropped": 0}       # idempotent


async def test_a_direct_grant_naming_no_grantee_is_refused_by_the_route(tmp_path, monkeypatch):
    """The path that wrote them: a direct grant whose request named no grantee."""
    from fastapi import HTTPException

    from mantle.routers import grants_router
    from mantle.services.acting_principal import acting_as
    from mantle.services.dependencies import AuthContext

    from .test_there_is_no_promotion_state import USER, real_node
    node = real_node(tmp_path, monkeypatch)
    auth = AuthContext(principal_id=USER, principal_type="user", user_id=USER)
    body = grants_router.CreateGrantRequest(resource_id="W", grantee_type="user", can_read=True)
    with acting_as(USER, principal_type="user"), pytest.raises(HTTPException) as e:
        await grants_router.create_grant_endpoint(body=body, auth=auth, store_db=node)
    assert e.value.status_code == 400 and "names its grantee" in e.value.detail


def test_every_grant_to_one_account_is_dropped_and_no_other(db):
    """`mantle.system.drop_grants_to`, for an account deactivated and retired (2026-11-10: the
    release smoke tests' throwaway accounts)."""
    from mantle.system import drop_grants_to as dgt
    for gid, who in (("g-1", "smoke"), ("g-2", "smoke"), ("g-3", "bob")):
        la.create_grant(db, _grant(gid, who))
    assert dgt.grants_to(db, "smoke") == ["g-1", "g-2"]
    assert dgt.drop_grants_to(db, "smoke", dry_run=True) == {"grantee": "smoke", "found": 2, "dropped": 0}
    assert dgt.drop_grants_to(db, "smoke") == {"grantee": "smoke", "found": 2, "dropped": 2}
    assert dgt.grants_to(db, "smoke") == [] and la.get_grant_by_id(db, "g-3") is not None
    with pytest.raises(ValueError):
        dgt.drop_grants_to(db, "  ")
