"""Review 17, M2: Mantle's deny walk follows the whole creation chain whatever each edge conducts,
as Origin's `Reach.denies` does (review 16), so Mantle's own gate, the one a cached cell's
plaintext is served on, refuses what Origin refuses. Under threshold custody a cell served from
the cache also asks the store's verdict (Origin's, memoized 30 s)."""
# ruff: noqa: F811
from __future__ import annotations

import pytest

from mantle.attenuation import ACTIONS
from mantle.db import lattice_api as la
from mantle.entities.grant import Grant
from mantle.search.mantle import oracle as o
from mantle.services.acting_principal import acting_as

from .test_add_does_not_carry_read import OWNER, node  # noqa: F401

CAROL = "carol-1"


def _grant(gid, who, rid, actions, effect="allow"):
    return Grant(id=gid, resource_id=rid, grantee_type="user", grantee_id=who, granted_by=OWNER,
                 effect=effect, **{f"can_{a}": a in actions for a in ACTIONS})


def test_a_deny_above_a_narrowed_edge_holds_in_mantle_s_gate(node):
    from mantle.services import workspace_service as ws
    db, oracle = node
    with acting_as(OWNER, principal_type="user"):
        w = ws.create_container(db, OWNER, content_type="application/vnd.agience.collection+json",
                                name="W").id
        c = ws.create_container(db, OWNER, content_type="application/vnd.agience.collection+json",
                                name="C").id
        la.add_artifact_to_collection(db, w, c, origin=True, propagate=["create"])
        la.create_grant(db, _grant("g-cc", CAROL, c, ("read",)))
        la.create_grant(db, _grant("g-deny-cw", CAROL, w, ("read",), effect="deny"))
    assert oracle._verifier._denied(CAROL, "user", c, "read") is True
    with acting_as(CAROL, principal_type="user"):
        with pytest.raises(o.GrantDenied):
            oracle.authorize(c, c, o.KeyRequest(requester_id=CAROL, purpose=o.KeyPurpose.GRANT,
                                                action="read"))


def test_a_cached_cell_asks_the_store_s_verdict(monkeypatch):
    """`_load_cell` serves a cache hit only after the store's verdict, under threshold custody."""
    from mantle.search.mantle import engine

    asked = []

    class _Oracle:
        def authorize(self, principal_id, collection_id, request):
            asked.append("authorize")

        def verdict(self, principal_id, collection_id, request):
            asked.append("verdict")

    class _Cache:
        def get(self, *a):
            return [("chunk", 1)]

    loader = engine.MantleQueryEngine
    obj = loader.__new__(loader)
    obj._oracle, obj._cache = _Oracle(), _Cache()
    req = o.KeyRequest(requester_id="x", purpose=o.KeyPurpose.GRANT, action="read")
    out = obj._load_cell("p", "c", "k", req)
    assert asked == ["authorize", "verdict"] and out[0] == [("chunk", 1)]
