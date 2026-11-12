"""After a seed, every existing account gets the user grant set and the platform operators for each
pair that has never had a grant record; a grant once withheld (revoked) stays withheld (71/test,
2026-11-12: an account that signed in before the seed had no operators)."""
from __future__ import annotations

from mantle.services.seed_provisioning import loader as L
from mantle.services.seed_provisioning import platform_operators as P
from mantle.services.seed_provisioning import user_provisioning as U


def test_only_never_granted_pairs_are_written_and_a_revoked_grant_stays_withheld(monkeypatch):
    import mantle.db.backend as B
    import mantle.db.lattice_api as la
    import mantle.services.peer_signing as ps
    import mantle.services.seed_provisioning.platform_email as E

    public = [o["slug"] for o in P.OPERATORS if o.get("public")]
    ids = {P._COLLECTION_SLUG: "coll", **{s: f"id-{s}" for s in [o["slug"] for o in P.OPERATORS]}}
    first = ids[public[0]]
    docs = [  # alice: one operator revoked; bob: nothing on the operators; the system: excluded
        {"grantee_type": "user", "grantee_id": "alice", "resource_id": first, "state": "revoked"},
        {"grantee_type": "user", "grantee_id": "bob", "resource_id": "inbox-b", "state": "active"},
        {"grantee_type": "user", "grantee_id": "sys", "resource_id": "x", "state": "active"}]
    monkeypatch.setattr(la, "_grant_docs", lambda db, state=None: list(docs))
    monkeypatch.setattr(ps, "get_system_principal_id", lambda: "sys")
    monkeypatch.setattr(P, "ids", lambda: ids)
    monkeypatch.setattr(E, "_operator_id", lambda db: "op")
    written = []
    monkeypatch.setattr(B, "upsert_user_collection_grant",
                        lambda db, *, user_id, collection_id, **k: written.append((user_id, collection_id)))
    applied = []
    monkeypatch.setattr(U, "_apply_grant_set", lambda db, root, ctx, uid: applied.append(
        (uid, L._EXISTING_PAIRS.get())))

    out = U.provision_existing_accounts(object(), None)
    assert out["accounts"] == 2
    assert ("alice", first) not in written, "a revoked grant is never written again"
    assert {c for u, c in written if u == "alice"} == {ids[s] for s in public[1:]}
    assert {c for u, c in written if u == "bob"} == {ids[s] for s in public}
    assert not any(u == "sys" for u, _ in written)


def test_the_user_grant_set_skips_every_pair_with_a_record():
    class Report:
        grants_added = grants_skipped = 0
        errors: list = []
    calls = []
    import mantle.services.seed_provisioning.loader as mod
    real = mod.db_upsert_user_collection_grant
    mod.db_upsert_user_collection_grant = lambda db, **k: calls.append((k["user_id"], k["collection_id"])) or (None, True)
    try:
        acc = {("bob", "r-old"): {"name": "n", "flags": {}}, ("bob", "r-new"): {"name": "n", "flags": {}}}
        r = Report()
        with L.only_where_never_granted(frozenset({("bob", "r-old")})):
            L._apply_accumulated_grants(object(), acc, r)
    finally:
        mod.db_upsert_user_collection_grant = real
    assert calls == [("bob", "r-new")] and r.grants_skipped == 1


def test_a_sign_in_before_the_seed_grants_no_operator_that_does_not_exist_yet(monkeypatch):
    # The grant would name the operator before its creation, and its first key is minted only
    # while no grant names it: the seed could then never make it.
    import mantle.db.backend as B
    import mantle.db.lattice_api as la
    import mantle.services.seed_provisioning.platform_email as E

    public = [o["slug"] for o in P.OPERATORS if o.get("public")]
    ids = {P._COLLECTION_SLUG: "coll", **{o["slug"]: f"id-{o['slug']}" for o in P.OPERATORS}}
    made = {ids[public[0]]}
    monkeypatch.setattr(P, "ids", lambda: ids)
    monkeypatch.setattr(E, "_operator_id", lambda db: "op")
    monkeypatch.setattr(la, "get_artifact", lambda db, i, **k: object() if i in made else None)
    written = []
    monkeypatch.setattr(B, "upsert_user_collection_grant",
                        lambda db, *, user_id, collection_id, **k: written.append(collection_id))
    assert P.grant_to_account(object(), "alice") == 1
    assert written == [ids[public[0]]]


def test_each_provisioned_account_s_memoized_grants_are_dropped(monkeypatch):
    # An account that signed in before the seed has its grants memoized empty; its first read
    # after the seed must see the new ones (71/test, 2026-11-12: the first load waited 30 s).
    import mantle.db.lattice_api as la
    import mantle.services.peer_signing as ps
    from mantle.search.mantle import wiring

    docs = [{"grantee_type": "user", "grantee_id": "bob", "resource_id": "inbox-b", "state": "active"}]
    monkeypatch.setattr(la, "_grant_docs", lambda db, state=None: list(docs))
    monkeypatch.setattr(ps, "get_system_principal_id", lambda: "sys")
    monkeypatch.setattr(P, "grant_to_account", lambda db, a, skip=frozenset(): 0)
    dropped = []
    monkeypatch.setattr(wiring, "invalidate_grant_cache", lambda rid=None: dropped.append(rid))
    U.provision_existing_accounts(object(), None)
    assert dropped == ["bob"]
