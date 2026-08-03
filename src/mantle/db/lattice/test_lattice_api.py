"""Brick-1 tests for `db.lattice_api` — the function-compatible `db.lattice` surface over the lattice.

These prove the standalone-Mantle contract at its root: an `ArtifactEntity` round-trips through the
SAME function names/signatures the routers already call, against Mantle's OWN store (a temp SQLite
lattice) — no the lattice, no MinIO, no server, no network.
"""
import sqlite3

import pytest

try:
    from mantle.db import lattice_api as api
    from mantle.entities.artifact import Artifact
    from mantle.entities.grant import Grant
except ImportError:  # mantle dir itself on the path
    from mantle.db import lattice_api as api
    from mantle.entities.artifact import Artifact
    from mantle.entities.grant import Grant


@pytest.fixture(autouse=True)
def _test_master_key(monkeypatch):
    """A deterministic content master key — the key ORACLE isn't up in unit tests, but the
    envelope boundary itself must run for real (MEC1 at rest, plaintext on read)."""
    try:                                   # plain path first — the module doc_boundary imports
        from mantle.services import content_crypto
    except ImportError:
        from mantle.services import content_crypto
    monkeypatch.setattr(content_crypto, "_default_master_key",
                        lambda principal_id, collection_id=None: b"\x01" * 32)


@pytest.fixture()
def db(tmp_path):
    return api.open_database(str(tmp_path / "mantle-lattice.db"), origin="test-mantle")


# ── ordering keys (pure) ─────────────────────────────────────────────────────
def test_after_key_orders():
    a = api.after_key(None)
    assert api.after_key(a) > a
    assert api.after_key("Uz") > "Uz"


def test_mid_key_between():
    a, b = "U", api.after_key(api.after_key("U"))
    m = api.mid_key(a, b)
    assert a < m < b
    assert api.mid_key(None, None) == "U"
    assert api.mid_key("U", None) > "U"


# ── artifact CRUD round-trip against the standalone store ────────────────────
def _entity(aid="art-lattice-1", **kw):
    fields = dict(id=aid, collection_id="col-1", content="hello lattice",
                  content_type="text/markdown", state="committed",
                  name="brick one", created_by="test-mantle")
    fields.update(kw)
    return Artifact(**fields)


def test_create_and_get_round_trip(db):
    ent = _entity()
    api.create_artifact(db, ent)
    got = api.get_artifact(db, "art-lattice-1")
    assert got is not None
    assert got.id == "art-lattice-1"
    assert got.content == "hello lattice"
    assert got.collection_id == "col-1"
    assert got.state == "committed"
    # origin_root INHERITED from the collection (the key principal roots at the
    # collection's chain, never self — self-rooting gave every child its own key)
    assert got.origin_root == "col-1"


def test_update_reflects(db):
    api.create_artifact(db, _entity())
    api.update_artifact(db, _entity(content="hello again"))
    got = api.get_artifact(db, "art-lattice-1")
    assert got.content == "hello again"


def test_get_missing_is_none(db):
    assert api.get_artifact(db, "nope") is None


def test_delete_removes(db):
    api.create_artifact(db, _entity())
    assert api.delete_artifact(db, "art-lattice-1") is True
    assert api.get_artifact(db, "art-lattice-1") is None


def test_no_lattice_internals_leak(db):
    api.create_artifact(db, _entity())
    got = api.get_artifact(db, "art-lattice-1")
    d = got.to_dict()
    for k in ("_origin", "_seq", "_rev", "_fp", "_type"):
        assert k not in d


# ── brick 2: drafts / version history / children ─────────────────────────────
def _version(vid, root, state="committed", collection="col-1", content="v"):
    return _entity(aid=vid, root_id=root, state=state, collection_id=collection, content=content)


def test_draft_and_latest_committed(db):
    api.create_artifact(db, _version("v1", "root-A", content="first"))
    api.create_artifact(db, _version("v2", "root-A", content="second"))
    api.create_artifact(db, _version("v3-draft", "root-A", state="draft", content="wip"))

    draft = api.get_draft_artifact(db, "root-A", "col-1")
    assert draft is not None and draft.id == "v3-draft" and draft.state == "draft"
    assert api.get_draft_artifact(db, "root-A", "other-col") is None

    latest = api.get_latest_committed_artifact(db, "root-A")
    assert latest.id == "v2"                      # newest committed in proper time
    assert api.get_latest_committed_artifact(db, "root-A", "other-col") is None


def test_current_in_collection_prefers_draft(db):
    api.create_artifact(db, _version("c1", "root-B", content="committed"))
    assert api.get_current_in_collection(db, "col-1", "root-B").id == "c1"
    api.create_artifact(db, _version("c2-draft", "root-B", state="draft"))
    assert api.get_current_in_collection(db, "col-1", "root-B").id == "c2-draft"


def test_version_history_newest_first_committed_only(db):
    api.create_artifact(db, _version("h1", "root-C"))
    api.create_artifact(db, _version("h2", "root-C"))
    api.create_artifact(db, _version("h3-draft", "root-C", state="draft"))
    hist = api.list_version_history(db, "root-C")
    assert [a.id for a in hist] == ["h2", "h1"]   # newest first, drafts excluded


def test_list_draft_artifacts_by_collection(db):
    api.create_artifact(db, _version("d1", "root-D", state="draft", collection="col-X"))
    api.create_artifact(db, _version("d2", "root-E", state="draft", collection="col-X"))
    api.create_artifact(db, _version("d3", "root-F", state="draft", collection="col-Y"))
    drafts = api.list_draft_artifacts(db, "col-X")
    assert sorted(a.id for a in drafts) == ["d1", "d2"]


def test_children_containment_edges(db):
    api.create_artifact(db, _entity(aid="parent-1"))
    api.create_artifact(db, _entity(aid="kid-1"))
    api.create_artifact(db, _entity(aid="kid-2"))
    assert api.has_children(db, "parent-1") is False
    assert api.count_children(db, "parent-1") == 0
    db.graph.add_edge("parent-1", "kid-1", "contains")
    db.graph.add_edge("parent-1", "kid-2", "contains")
    db.graph.add_edge("parent-1", "kid-1", "references")   # typed relation ≠ containment
    assert api.has_children(db, "parent-1") is True
    assert api.count_children(db, "parent-1") == 2


# ── brick 3: collections + membership + order keys ───────────────────────────
def _mk_collection(db, cid="col-A"):
    api.create_collection(db, _entity(aid=cid, collection_id="",
                                      content_type="application/vnd.agience.collection+json",
                                      content="", name="a container"))
    return cid


def test_collection_crud_is_artifact_crud(db):
    _mk_collection(db, "col-A")
    got = api.get_collection_by_id(db, "col-A")
    assert got is not None and got.id == "col-A"
    got.name = "renamed"
    api.update_collection(db, got)
    assert api.get_collection_by_id(db, "col-A").name == "renamed"
    assert api.delete_collection(db, "col-A") is True
    assert api.get_collection_by_id(db, "col-A") is None


def test_membership_add_last_key_and_remove(db):
    _mk_collection(db, "col-A")
    api.create_artifact(db, _version("m1", "root-M1"))
    api.create_artifact(db, _version("m2", "root-M2"))
    assert api.get_last_order_key(db, "col-A") is None
    assert api.add_artifact_to_collection(db, "col-A", "root-M1") is True
    k1 = api.get_last_order_key(db, "col-A")
    assert k1 is not None
    assert api.add_artifact_to_collection(db, "col-A", "root-M2") is True
    k2 = api.get_last_order_key(db, "col-A")
    assert k2 > k1                                  # appended after the end
    assert api.remove_artifact_from_collection(db, "col-A", "root-M1") is True
    assert api.remove_artifact_from_collection(db, "col-A", "root-M1") is True   # idempotent
    assert len(api.list_collection_artifacts(db, "col-A")) == 1


def test_list_collection_artifacts_resolution_and_order(db):
    _mk_collection(db, "col-A")
    # root-X: committed in-collection, then a draft (draft must win)
    api.create_artifact(db, _version("x1", "root-X", collection="col-A", content="x committed"))
    api.create_artifact(db, _version("x2", "root-X", state="draft", collection="col-A", content="x draft"))
    # root-Y: committed only in ANOTHER collection (published link → committed fallback)
    api.create_artifact(db, _version("y1", "root-Y", collection="col-B", content="y committed"))
    # root-Z: archived in-collection (hidden unless include_archived)
    api.create_artifact(db, _version("z1", "root-Z", state="archived", collection="col-A"))
    api.add_artifact_to_collection(db, "col-A", "root-X")
    api.add_artifact_to_collection(db, "col-A", "root-Y", relationship="operator")
    api.add_artifact_to_collection(db, "col-A", "root-Z")
    rows = api.list_collection_artifacts(db, "col-A")
    by_root = {r["root_id"]: r for r in rows}
    assert set(by_root) == {"root-X", "root-Y"}                 # Z archived → hidden
    assert by_root["root-X"]["id"] == "x2"                      # draft preferred
    assert by_root["root-X"]["has_committed_version"] is True
    assert by_root["root-X"]["relationship"] is None            # containment
    assert by_root["root-Y"]["id"] == "y1"                      # committed-anywhere fallback
    assert by_root["root-Y"]["relationship"] == "operator"      # typed relation = the label
    assert [r["root_id"] for r in rows] == ["root-X", "root-Y"] # order_key order (insertion)
    rows_all = api.list_collection_artifacts(db, "col-A", include_archived=True)
    assert {r["root_id"] for r in rows_all} == {"root-X", "root-Y", "root-Z"}


def test_reorder_and_set_order_key(db):
    _mk_collection(db, "col-A")
    for i, root in enumerate(["root-R1", "root-R2", "root-R3"]):
        api.create_artifact(db, _version(f"r{i}", root))
        api.add_artifact_to_collection(db, "col-A", root)
    assert api.reorder_collection_artifacts(db, "col-A", ["root-R3", "root-R1", "root-R2"]) == 3
    rows = api.list_collection_artifacts(db, "col-A")
    assert [r["root_id"] for r in rows] == ["root-R3", "root-R1", "root-R2"]
    # a mid_key insert between the first two lands between them
    k1, k2 = rows[0]["order_key"], rows[1]["order_key"]
    assert api.set_edge_order_key(db, "col-A", "root-R2", api.mid_key(k1, k2)) is True
    rows = api.list_collection_artifacts(db, "col-A")
    assert [r["root_id"] for r in rows] == ["root-R3", "root-R2", "root-R1"]


def test_count_other_containers_for_root(db):
    _mk_collection(db, "col-A")
    _mk_collection(db, "col-B")
    api.create_artifact(db, _version("s1", "root-S"))
    api.add_artifact_to_collection(db, "col-A", "root-S")
    api.add_artifact_to_collection(db, "col-B", "root-S")
    assert api.count_other_containers_for_root(db, "root-S", "col-A") == 1
    assert api.count_other_containers_for_root(db, "root-S", "col-B") == 1


# ── brick 4: grants — the CRUDEASIO authorization plane ──────────────────────
def _grant(gid="g-1", **kw):
    fields = dict(id=gid, resource_id="col-A", grantee_type="user", grantee_id="user-1",
                  granted_by="admin-1", can_read=True)
    fields.update(kw)
    return Grant(**fields)


def test_grant_crud_round_trip(db):
    api.create_grant(db, _grant(can_update=True, name="editor grant"))
    g = api.get_grant_by_id(db, "g-1")
    assert g is not None and g.resource_id == "col-A" and g.grantee_id == "user-1"
    assert g.can_read is True and g.can_update is True and g.can_delete is False
    assert g.state == "active" and g.effect == "allow" and g.name == "editor grant"
    g.state = Grant.STATE_REVOKED
    api.update_grant(db, g)
    assert api.get_grant_by_id(db, "g-1").state == "revoked"


def test_grant_scoping_artifact_id_is_not_a_grant(db):
    api.create_artifact(db, _entity(aid="art-1"))
    assert api.get_grant_by_id(db, "art-1") is None      # collection scoping preserved


def test_active_grants_filters_state_and_expiry(db):
    api.create_grant(db, _grant("g-live"))
    api.create_grant(db, _grant("g-revoked", state="revoked"))
    api.create_grant(db, _grant("g-expired", expires_at="2001-01-01T00:00:00+00:00"))
    api.create_grant(db, _grant("g-future", expires_at="2999-01-01T00:00:00Z"))
    got = api.get_active_grants_for_principal_resource(db, "user-1", "col-A")
    assert sorted(g.id for g in got) == ["g-future", "g-live"]
    assert [g.id for g in api.get_active_grants_for_grantee(db, "user-1", "user")
            ] == sorted(g.id for g in got)


def test_light_cone_and_grants_for_collection(db):
    api.create_grant(db, _grant("g-a", resource_id="col-A"))
    api.create_grant(db, _grant("g-b", resource_id="col-B", can_read=False))   # unreadable
    api.create_grant(db, _grant("g-c", resource_id="col-C", grantee_id="user-2"))
    assert api.get_active_collection_ids_for_user(db, "user-1") == ["col-A"]
    api.create_grant(db, _grant("g-a2", resource_id="col-A", grantee_id="user-2", state="revoked"))
    assert sorted(g.id for g in api.get_grants_for_collection(db, "col-A")) == ["g-a", "g-a2"]


def test_upsert_user_collection_grant(db):
    g1, changed = api.upsert_user_collection_grant(
        db, user_id="user-9", collection_id="col-A", granted_by="admin-1")
    assert changed is True and g1.can_read is True and g1.can_update is False
    g2, changed = api.upsert_user_collection_grant(          # same flags → no-op
        db, user_id="user-9", collection_id="col-A", granted_by="admin-1")
    assert changed is False and g2.id == g1.id
    g3, changed = api.upsert_user_collection_grant(          # flag flip → in-place update
        db, user_id="user-9", collection_id="col-A", granted_by="admin-1", can_update=True)
    assert changed is True and g3.id == g1.id
    assert api.get_grant_by_id(db, g1.id).can_update is True


def test_grant_key_auth_by_token_hash(db):
    try:
        from mantle.services import auth_service
    except ImportError:
        from mantle.services import auth_service
    token = "sk-test-token-123"
    api.create_grant(db, _grant("g-key", grantee_type="grant_key",
                                grantee_id=auth_service.hash_api_key(token)))
    got = api.get_active_grants_by_key(db, token)
    assert [g.id for g in got] == ["g-key"]
    assert api.get_active_grants_by_key(db, "wrong-token") == []
    assert [g.id for g in api.get_active_grant_key_grants_for_collection(db, "col-A")] == ["g-key"]


def test_get_containers_for_user_via_light_cone(db):
    _mk_collection(db, "col-A")
    _mk_collection(db, "col-B")
    api.create_grant(db, _grant("g-a", resource_id="col-A"))
    api.create_grant(db, _grant("g-b", resource_id="col-B"))
    api.create_grant(db, _grant("g-x", resource_id="col-missing"))   # dangling grant → skipped
    got = api.get_containers_for_user(db, "user-1")
    assert sorted(c.id for c in got) == ["col-A", "col-B"]
    typed = api.get_containers_for_user(
        db, "user-1", content_type="application/vnd.agience.collection+json")
    assert sorted(c.id for c in typed) == ["col-A", "col-B"]
    assert api.get_containers_for_user(db, "user-1", content_type="text/markdown") == []


# ── brick 5: origin lineage & the propagation light-cone ─────────────────────
def test_origin_chain_and_root(db):
    # ws → col → art: two origin containment hops
    for aid in ("ws-1", "col-1x", "art-1x"):
        api.create_artifact(db, _entity(aid=aid))
    api.add_artifact_to_collection(db, "ws-1", "col-1x")
    api.add_artifact_to_collection(db, "col-1x", "art-1x", propagate=["read"])
    assert api.get_origin_parent(db, "art-1x") == ("col-1x", ["read"])
    assert api.get_origin_parent(db, "ws-1") is None
    assert api.get_origin_root(db, "art-1x") == "ws-1"
    assert api.get_origin_root(db, "ws-1") == "ws-1"
    # a non-origin LINK edge must not become a parent
    api.create_artifact(db, _entity(aid="other-col"))
    api.add_artifact_to_collection(db, "other-col", "art-1x", origin=False)
    assert api.get_origin_parent(db, "art-1x") == ("col-1x", ["read"])


def test_light_cone_propagation_mask(db):
    for aid in ("ws-1", "a", "b", "c", "d"):
        api.create_artifact(db, _entity(aid=aid))
    api.add_artifact_to_collection(db, "ws-1", "a")                       # unrestricted
    api.add_artifact_to_collection(db, "a", "b", propagate=["read"])      # read flows
    api.add_artifact_to_collection(db, "a", "c", propagate=["update"])    # read blocked
    api.add_artifact_to_collection(db, "b", "d")                          # behind b
    cone = api.list_origin_descendants(db, ["ws-1"], "read")
    assert cone == {"a", "b", "d"}                 # c pruned, seeds excluded
    assert api.list_origin_descendants(db, ["ws-1"], "update") == {"a", "c"}
    # FAILURE MODE: a `max_depth=1` argument used to truncate the cone to {"a"}, so a grant
    # deeper than the cap produced a false deny. There is no depth parameter now — `seen`
    # terminates the BFS and "d" (two levels down, behind b) must be reached.
    assert "d" in api.list_origin_descendants(db, ["ws-1"], "read")
    assert api.list_origin_descendants(db, [], "read") == set()


def test_get_edge_and_relationship_target(db):
    api.create_artifact(db, _entity(aid="col-E"))
    api.create_artifact(db, _version("t1", "root-T"))
    api.add_artifact_to_collection(db, "col-E", "root-T", relationship="operator")
    e = api.get_edge(db, "col-E", "root-T")
    assert e is not None and e["relationship"] == "operator" and e["origin"] is True
    assert e["relation"] == "derivation"           # operator edge = derivation
    assert api.get_edge(db, "col-E", "nope") is None
    assert api.get_relationship_target(db, "col-E", "operator") == "root-T"
    assert api.get_relationship_target(db, "col-E", "missing") is None


def test_remove_all_edges_and_delete_by_root(db):
    api.create_artifact(db, _entity(aid="col-A"))
    api.create_artifact(db, _entity(aid="col-B"))
    api.create_artifact(db, _version("w1", "root-W"))
    api.create_artifact(db, _version("w2", "root-W"))
    api.add_artifact_to_collection(db, "col-A", "root-W")
    api.add_artifact_to_collection(db, "col-B", "root-W", origin=False)
    assert api.remove_all_edges_for_root(db, "root-W") == 2
    assert api.get_edge(db, "col-A", "root-W") is None
    assert sorted(api.delete_artifacts_by_root(db, "root-W")) == ["w1", "w2"]
    assert api.get_artifact(db, "w1") is None


def test_batch_commit_drafts(db):
    api.create_artifact(db, _version("bc1", "root-P", state="draft"))
    api.create_artifact(db, _version("bc2", "root-Q", state="draft"))
    api.create_artifact(db, _version("bc3", "root-R", state="draft", collection="col-OTHER"))
    api.create_artifact(db, _version("bc4", "root-S"))                    # already committed
    n = api.batch_commit_drafts(db, "col-1", ["bc1", "bc2", "bc3", "bc4", "ghost"],
                                "committer-1", "2026-07-22T12:00:00+00:00")
    assert n == 2                                  # only in-collection drafts flip
    got = api.get_artifact(db, "bc1")
    assert got.state == "committed" and got.modified_by == "committer-1"
    assert api.get_artifact(db, "bc3").state == "draft"


def test_archive_artifact_gated_on_delete_grant(db):
    api.create_artifact(db, _entity(aid="arch-1"))
    assert api.archive_artifact(db, "user-1", "arch-1") is False          # no grant
    api.create_grant(db, _grant("g-del", resource_id="col-1", can_delete=True))
    assert api.archive_artifact(db, "user-1", "arch-1") is True
    # head-only default read no longer surfaces it, but explicit state does
    assert all(d.get("id") != "arch-1"
               for d in db.artifacts.list_artifacts(content_type="text/markdown"))


# ── brick 6: commits + api keys + server credentials + JWKs ──────────────────
def test_commits_round_trip_and_collection_lookup(db):
    try:
        from mantle.entities.commit import Commit
        from mantle.entities.commit_item import CommitItem
    except ImportError:
        from mantle.entities.commit import Commit
        from mantle.entities.commit_item import CommitItem
    items = [CommitItem(id="ci-1", item_type="add", collection_id="col-1",
                        artifact_version_ids=["v1"]),
             CommitItem(id="ci-2", item_type="add", collection_id="col-OTHER",
                        artifact_version_ids=["v2"])]
    assert api.create_commit_items(db, items) == ["ci-1", "ci-2"]
    api.create_commit(db, Commit(id="cm-1", message="first", timestamp="2026-07-22T10:00:00Z",
                                 author_id="user-1", item_ids=["ci-1"]))
    api.create_commit(db, Commit(id="cm-2", message="second", timestamp="2026-07-22T11:00:00Z",
                                 author_id="user-1", item_ids=["ci-2"]))
    raw = api.get_commit_by_id(db, "cm-1")
    assert raw is not None and raw["id"] == "cm-1" and raw["message"] == "first"
    assert api.get_commit_by_id(db, "nope") is None
    got = api.get_commits_for_collection(db, "col-1")
    assert [c["id"] for c in got] == ["cm-1"]
    both = api.get_commits_for_collection(db, "col-OTHER")
    assert [c["id"] for c in both] == ["cm-2"]


def test_api_keys(db):
    try:
        from mantle.entities.api_key import APIKey
    except ImportError:
        from mantle.entities.api_key import APIKey
    api.create_api_key(db, APIKey(id="k-1", user_id="user-1", name="k", key_hash="h-live"))
    api.create_api_key(db, APIKey(id="k-2", user_id="user-1", name="k2", key_hash="h-off",
                                  is_active=False))
    api.create_api_key(db, APIKey(id="k-3", user_id="user-2", name="k3", key_hash="h-exp",
                                  expires_at="2001-01-01T00:00:00Z"))
    assert api.get_api_key_by_hash(db, "h-live").id == "k-1"
    assert api.get_api_key_by_hash(db, "h-off") is None          # inactive
    assert api.get_api_key_by_hash(db, "h-exp") is None          # expired
    assert sorted(k.id for k in api.get_api_keys_by_user(db, "user-1")) == ["k-1", "k-2"]
    assert api.update_api_key_last_used(db, "k-1", "2026-07-22T12:00:00Z") is True
    assert api.get_api_key_by_id(db, "k-1").last_used_at == "2026-07-22T12:00:00Z"
    assert api.delete_api_key(db, "k-1") is True
    assert api.get_api_key_by_id(db, "k-1") is None


def test_server_credentials_and_jwks(db):
    try:
        from mantle.entities.server_credential import ServerCredential
    except ImportError:
        from mantle.entities.server_credential import ServerCredential
    api.create_server_credential(db, ServerCredential(id="sc-1", user_id="user-1",
                                                      client_id="client-A"))
    assert api.get_server_credential_by_client_id(db, "client-A").id == "sc-1"
    assert api.get_server_credential_by_client_id(db, "client-Z") is None
    assert [c.id for c in api.get_server_credentials_by_user(db, "user-1")] == ["sc-1"]
    assert [c.id for c in api.get_all_server_credentials(db)] == ["sc-1"]
    assert api.update_server_credential_last_used(db, "sc-1", "2026-07-22T12:00:00Z") is True
    api.upsert_server_jwk(db, "client-A", {"kty": "OKP", "x": "abc"})
    assert api.get_server_jwk(db, "client-A") == {"kty": "OKP", "x": "abc"}
    api.upsert_server_jwk(db, "client-A", {"kty": "OKP", "x": "xyz"})    # upsert replaces
    assert api.get_server_jwk(db, "client-A") == {"kty": "OKP", "x": "xyz"}
    assert api.get_server_jwk(db, "sc-1") is None                        # scoping preserved
    assert api.delete_server_credential(db, "sc-1") is True


# ── brick 7: the shared persistence boundary (crypto + events) ───────────────
def test_content_is_ciphertext_at_rest_plaintext_on_read(db):
    import base64
    api.create_artifact(db, _entity())
    raw = db.artifacts.get_artifact("art-lattice-1")          # the STORED doc
    assert raw["content_encrypted"] is True
    assert base64.b64decode(raw["content"])[:4] == b"MEC1"    # envelope, not plaintext
    assert "hello lattice" not in raw["content"]
    assert raw["content_key_principal"] == "col-1"    # keyed on the COLLECTION's origin root
    got = api.get_artifact(db, "art-lattice-1")               # the read chokepoint decrypts
    assert got.content == "hello lattice"


def test_list_collection_artifacts_returns_plaintext(db):
    _mk_collection(db, "col-A")
    api.create_artifact(db, _version("p1", "root-P", collection="col-A", content="secret text"))
    api.add_artifact_to_collection(db, "col-A", "root-P")
    rows = api.list_collection_artifacts(db, "col-A")
    assert rows[0]["content"] == "secret text"


def test_grants_are_not_content_encrypted(db):
    api.create_grant(db, _grant("g-plain"))
    raw = db.artifacts.get_artifact("g-plain")
    assert not raw.get("content_encrypted")                   # no inline content, no envelope


def test_materialized_markers_namespaced(db):
    api.create_artifact(db, _entity(aid="mat-1"))
    assert api.is_materialized(db, "mat-1") is False
    api.mark_materialized(db, "mat-1")
    api.mark_materialized(db, "mat-1")                                   # idempotent
    assert api.is_materialized(db, "mat-1") is True
    assert api.get_artifact(db, "mat-1").content == "hello lattice"      # artifact untouched
    assert api.is_materialized(db, "materialized:mat-1") is False        # no double-prefix leak


def test_a_db_fault_is_not_served_as_no_children(tmp_path):
    """FAILURE MODE FIRST: `has_children` caught every exception and returned False; the router
    serves that straight to clients as `doc["has_children"]`, then forces `child_count` to 0 without
    attempting the count. A transient DB fault made a POPULATED container look empty to every caller,
    indistinguishable from a genuinely childless artifact. [John, 2026-07-31: fail loudly.]
    """
    import pytest

    from mantle.db import lattice_api as api

    class _BrokenGraph:
        def edges_of(self, *a, **kw):
            raise sqlite3.OperationalError("database is locked")

    class _BrokenDB:
        graph = _BrokenGraph()

    db = _BrokenDB()
    with pytest.raises(sqlite3.OperationalError):
        api.has_children(db, "parent-1")
    with pytest.raises(sqlite3.OperationalError):
        api.count_children(db, "parent-1")
