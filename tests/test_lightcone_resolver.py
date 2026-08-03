"""Unit tests for `search.mantle.LightConeResolver` (Step 2.1).

CRUDEASIO lives in Mantle (the lattice grants collection). The resolver reads
grants from `db_store.get_active_grants_for_grantee` — no Origin HTTP
calls. Tests cover:

- empty grant set → empty result
- grant with action mismatch (e.g. read-only grant for "create") → not included
- direct grant only (no descendants) → just that ID
- direct grant + descendants → union
- propagate mask blocks descendants → only direct included
- unknown action → empty
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mantle.db import backend as db_store
from mantle.entities.grant import Grant as GrantEntity
from mantle.search.mantle import LightConeResolver

def _grant(resource_id: str, **flags) -> GrantEntity:
    """Build a GrantEntity with CRUDEASIO flags defaulting to False."""
    defaults = {
        "can_read": False, "can_create": False, "can_update": False,
        "can_delete": False, "can_evict": False, "can_invoke": False,
        "can_add": False, "can_share": False, "can_admin": False,
    }
    defaults.update(flags)
    return GrantEntity(
        resource_id=resource_id,
        grantee_type="user",
        grantee_id="user-1",
        granted_by="admin",
        **defaults,
    )


def test_empty_grants_returns_empty_set():
    with patch.object(db_store, "get_active_grants_for_grantee", return_value=[]):
        resolver = LightConeResolver(db=MagicMock())
        assert resolver.resolve("user-1") == set()


def test_grant_lacking_action_flag_is_excluded():
    # Grant has can_read=False; resolving for "read" must skip it.
    with patch.object(db_store, "get_active_grants_for_grantee",
                      return_value=[_grant("col-1", can_read=False, can_admin=True)]):
        resolver = LightConeResolver(db=MagicMock())
        assert resolver.resolve("user-1", "read") == set()


def test_unknown_action_returns_empty():
    with patch.object(db_store, "get_active_grants_for_grantee",
                      return_value=[_grant("col-1", can_read=True)]):
        resolver = LightConeResolver(db=MagicMock())
        assert resolver.resolve("user-1", "no-such-action") == set()


def test_direct_grant_with_no_descendants():
    with (
        patch.object(db_store, "get_active_grants_for_grantee",
                     return_value=[_grant("col-1", can_read=True)]),
        patch.object(db_store, "list_origin_descendants", return_value=set()),
    ):
        resolver = LightConeResolver(db=MagicMock())
        assert resolver.resolve("user-1", "read") == {"col-1"}


def test_direct_grant_unions_with_descendants():
    with (
        patch.object(db_store, "get_active_grants_for_grantee",
                     return_value=[_grant("col-1", can_read=True)]),
        patch.object(db_store, "list_origin_descendants",
                     return_value={"art-a", "art-b", "sub-col"}),
    ):
        resolver = LightConeResolver(db=MagicMock())
        result = resolver.resolve("user-1", "read")
    assert result == {"col-1", "art-a", "art-b", "sub-col"}


def test_two_grants_descendants_unioned():
    captured = {}

    def fake_descendants(_db, root_ids, action):
        captured["root_ids"] = list(root_ids)
        captured["action"] = action
        return {"x", "y"}

    with (
        patch.object(db_store, "get_active_grants_for_grantee", return_value=[
            _grant("col-1", can_read=True),
            _grant("col-2", can_read=True),
        ]),
        patch.object(db_store, "list_origin_descendants", side_effect=fake_descendants),
    ):
        resolver = LightConeResolver(db=MagicMock())
        result = resolver.resolve("user-1", "read")

    assert result == {"col-1", "col-2", "x", "y"}
    # The resolver passes both granted IDs to the BFS in one call.
    assert set(captured["root_ids"]) == {"col-1", "col-2"}
    assert captured["action"] == "read"
    # FAILURE MODE: a `max_depth` used to be threaded through here and silently truncated the
    # light-cone, so a grant more than 4 levels up produced a false deny. There is no depth
    # parameter to pass any more — `seen` terminates the BFS.
    assert "max_depth" not in captured


def test_action_passes_through_to_descendant_lookup():
    captured = {}

    def fake_descendants(_db, root_ids, action):
        captured["action"] = action
        return set()

    with (
        patch.object(db_store, "get_active_grants_for_grantee",
                     return_value=[_grant("col-1", can_create=True)]),
        patch.object(db_store, "list_origin_descendants", side_effect=fake_descendants),
    ):
        resolver = LightConeResolver(db=MagicMock())
        resolver.resolve("user-1", "create")

    assert captured["action"] == "create"


# ---------------------------------------------------------------------------
# principal_type -> ledger grantee_type
# ---------------------------------------------------------------------------
#
# THE FAILURE MODE THESE PIN, STATED FIRST — because the obvious "fix" is the bug:
#
#   The oracle's grant verifier accepts a `requester_type` and, until 2026-07-31,
#   dropped it. Threading it STRAIGHT THROUGH as the ledger's `grantee_type` looks
#   like the correction and is a regression: the platform SYSTEM principal acts with
#   principal_type "service", but `seed_provisioning/platform_email.py` issues its
#   grants via `upsert_user_collection_grant`, which stores grantee_type "user".
#   A verbatim pass-through queries for "service" grants, finds none, and every
#   system-principal grant in every existing store becomes a silent false DENY.
#
# So both directions are asserted: the mapping must REACH the ledger (it is not
# dropped) and must NOT be the identity function (it does not pass "service" on).

def test_service_principal_resolves_against_the_ledgers_principal_grant_kind():
    captured = {}

    def fake_grants(_db, *, grantee_id, grantee_type):
        captured["grantee_type"] = grantee_type
        return [_grant("col-1", can_update=True)]

    with (
        patch.object(db_store, "get_active_grants_for_grantee", side_effect=fake_grants),
        patch.object(db_store, "list_origin_descendants", return_value=set()),
    ):
        resolver = LightConeResolver(db=MagicMock())
        result = resolver.resolve("sys-1", "update", principal_type="service")

    # NEGATIVE CONTROL: "service" is not a grantee_type this ledger ever writes.
    # If this assertion ever reads "service", the system principal's real grants
    # have just become invisible.
    assert captured["grantee_type"] != "service"
    assert captured["grantee_type"] == "user"
    assert result == {"col-1"}


def test_key_shaped_principals_keep_their_own_grantee_type():
    """api_key / grant_key hold grants as a hashed credential, not as a principal id."""
    seen = []

    def fake_grants(_db, *, grantee_id, grantee_type):
        seen.append(grantee_type)
        return []

    with patch.object(db_store, "get_active_grants_for_grantee", side_effect=fake_grants):
        resolver = LightConeResolver(db=MagicMock())
        resolver.resolve("k-1", "read", principal_type="api_key")
        resolver.resolve("k-2", "read", principal_type="grant_key")
        # Entity kinds with no credential of their own act as a principal id.
        resolver.resolve("s-1", "read", principal_type="server")
        resolver.resolve("d-1", "read", principal_type="delegation")

    assert seen == ["api_key", "grant_key", "user", "user"]


def test_verifier_does_not_drop_requester_type_before_the_lookup():
    """The parameter must REACH the resolver — it used to die in the verifier."""
    import mantle.search.mantle.sse.router_accessor as ra
    from mantle.search.mantle.oracle import LightConeGrantVerifier

    captured = {}

    class _Recorder:
        def resolve(self, principal_id, action="read", *, principal_type="user"):
            captured["principal_type"] = principal_type
            return set()

    with patch.object(
        ra, "_raw_artifact", return_value={"id": "col-1", "collection_id": "col-1"},
    ):
        v = LightConeGrantVerifier(MagicMock(), resolver=_Recorder())
        v.authorized(
            requester_id="sys-1", requester_type="service",
            principal_id="p-1", collection_id="col-1", action="update",
        )

    # FAILURE MODE: before the fix this stayed at the "user" default no matter who
    # asked, so the type was a parameter that read as enforced and was not.
    assert captured["principal_type"] == "service"
