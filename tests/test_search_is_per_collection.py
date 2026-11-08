"""Search cells and postings are keyed under each collection's own key, so a caller who reaches a
collection for read can search it and finds nothing in a sibling, and the master key, released
only to its principal, is not needed to search at all.

A query spanning several collections opens only the collection keys the caller reaches. The owner
of a workspace, who is not its key principal, searches everything they reach. The cutover pass
(`rekey_collections.rekey_index`) moves an existing index (cells re-encrypted in place, postings
filed again under the collection key and removed from under the principal-wide key) and survives an
interruption. A collection's key is minted as the collection is created. Small fixtures only.
"""
from __future__ import annotations

import pytest

from mantle.attenuation import ACTIONS
from mantle.db import lattice_api as la
from mantle.entities.grant import Grant
from mantle.search.mantle import cell as cell_mod
from mantle.search.mantle import oracle as o
from mantle.search.mantle.sse.indexer import SseIndexer
from mantle.search.mantle.sse.narrowing import TokenNarrower
from mantle.search.mantle.sse.posting import InMemoryPostingStore
from mantle.search.mantle.stores import InMemoryCellStore
from mantle.services.acting_principal import acting_as
from mantle.system import rekey_collections as rk

from .test_a_split_master_key_needs_two_custodians import _FakeKek

W = "W"                       # the workspace: the key principal
DOCS = {"c-1": ("C", "encryption library"), "c3-1": ("C3", "encryption keys")}


def _grant(gid, who, resource, actions):
    return Grant(id=gid, resource_id=resource, grantee_type="user", grantee_id=who,
                 granted_by=W, **{f"can_{a}": a in actions for a in ACTIONS})


@pytest.fixture
def node(tmp_path, monkeypatch):
    """`W` holds `C` and `C3`. Alice owns the workspace (every right on `W`) and is not its
    principal; Bob reads `C`; Carol reads nothing."""
    from mantle.search.mantle import wiring

    db = la.LatticeDatabase(str(tmp_path / "s.db"), origin="s")
    for coll in ("C", "C3"):
        la.add_artifact_to_collection(db, W, coll, origin=True)
    la.create_grant(db, _grant("g-aw", "alice", W, ACTIONS))
    la.create_grant(db, _grant("g-bc", "bob", "C", ("read",)))
    oracle = o.OracleService(o.LatticeMasterKeyStore(_FakeKek(), lambda: db),
                             grant_verifier=o.LightConeGrantVerifier(db, ttl_s=0))
    monkeypatch.setattr(wiring, "_build_oracle", lambda: oracle)
    postings = InMemoryPostingStore()
    return db, oracle, postings


def _req(who, action="read"):
    return o.KeyRequest(requester_id=who, purpose=o.KeyPurpose.GRANT, action=action)


def _index(oracle, postings):
    ix = SseIndexer(oracle, postings)
    with acting_as(W, principal_type="user"):
        for aid, (coll, title) in DOCS.items():
            ix.index_artifact(W, coll, aid, {"title": title}, _req(W, "create"))
    return ix


def _search(oracle, postings, who, term="encryption"):
    with acting_as(who, principal_type="user"):
        return set(TokenNarrower(oracle, postings).lookup_for(term, _req(who))(
            [(W, "C"), (W, "C3")]))


def test_a_grantee_searches_their_collection_and_sees_nothing_from_a_sibling(node):
    db, oracle, postings = node
    _index(oracle, postings)
    assert _search(oracle, postings, "bob") == {"c-1"}
    assert _search(oracle, postings, "carol") == set()


def test_the_owner_who_is_not_the_principal_searches_everything_they_reach(node):
    db, oracle, postings = node
    _index(oracle, postings)
    assert _search(oracle, postings, "alice") == {"c-1", "c3-1"}


def test_a_collection_origin_refuses_is_skipped_and_the_rest_are_searched(node, monkeypatch):
    """Under threshold custody Origin may refuse one collection's share: that collection
    contributes nothing, as an ungranted one does, and the search goes on."""
    from mantle.search.mantle.threshold_custody import ShareRefused

    db, oracle, postings = node
    _index(oracle, postings)
    real = oracle.get_or_create_collection_key

    def refuse_c3(principal, collection, request):
        if collection == "C3":
            raise ShareRefused("Origin refused the share for ck:C3")
        return real(principal, collection, request)

    monkeypatch.setattr(oracle, "get_or_create_collection_key", refuse_c3)
    assert _search(oracle, postings, "alice") == {"c-1"}


def test_searching_never_asks_for_the_master_key(node, monkeypatch):
    """Under threshold custody the master key is released to its principal alone; search keys
    come from collection keys, so a search never asks for it."""
    db, oracle, postings = node
    _index(oracle, postings)

    def refuse(*a, **k):
        raise AssertionError("search asked for the master key")

    monkeypatch.setattr(oracle, "get_or_create_master_key", refuse)
    assert _search(oracle, postings, "alice") == {"c-1", "c3-1"}


def test_a_cell_key_opens_for_its_collection_s_readers_alone(node):
    db, oracle, _p = node
    with acting_as("alice", principal_type="user"):
        k_c = oracle.derive_cell_key(W, "C", "k1", _req("alice", "create"))
        k_c3 = oracle.derive_cell_key(W, "C3", "k1", _req("alice", "create"))
    assert k_c != k_c3
    with acting_as("bob", principal_type="user"):
        assert oracle.derive_cell_key(W, "C", "k1", _req("bob")) == k_c
        with pytest.raises(o.GrantDenied):
            oracle.derive_cell_key(W, "C3", "k1", _req("bob"))


def test_a_collection_s_key_is_minted_as_it_is_created(node, monkeypatch):
    from mantle.services import collection_service as cs

    db, oracle, _p = node
    with acting_as("alice", principal_type="user"):
        created = cs.create_new_collection(db, owner_id="alice", name="New", description="",
                                           container_id=W)
    assert oracle._store.get(o.collection_key_id(created.id)) is not None


# -- the cutover pass, for the index ----------------------------------------------------------------

class _LegacyProvider:
    """How postings were keyed before collections had keys: one key per principal."""

    def __init__(self, oracle):
        self.oracle = oracle

    def derive_sse_key(self, principal_id, request):
        return self.oracle.derive_legacy_sse_key(principal_id, request)


def _legacy_index(db, oracle, postings, cells):
    """An index written before collections had keys: cells under master-derived keys, postings
    under the principal-wide SSE key."""
    with acting_as(W, principal_type="user"):
        oracle.get_or_create_master_key(W, _req(W, "create"))
        old = SseIndexer(_LegacyProvider(oracle), postings)
        for aid, (coll, title) in DOCS.items():
            old.index_artifact(W, coll, aid, {"title": title}, _req(W, "create"))
            db.artifacts.put_artifact({"id": aid, "collection_id": coll, "state": "live"})
            key = oracle.derive_legacy_cell_key(W, coll, "k1", _req(W, "create"))
            cells.put(W, coll, cell_mod.pack_cell([{"artifact_id": aid, "chunk_index": 0}], key,
                                                  collection_id=cell_mod.cell_aad(coll, "k1")),
                      "k1")


@pytest.fixture
def legacy(node, monkeypatch):
    db, oracle, postings = node
    cells = InMemoryCellStore()
    _legacy_index(db, oracle, postings, cells)
    monkeypatch.setattr("mantle.search.ingest.pipeline_unified._extract_artifact_fields",
                        lambda a: {"title": DOCS[a.id][1]})
    return db, oracle, postings, cells


def _cells_open(oracle, cells):
    with acting_as(W, principal_type="user"):
        for coll in ("C", "C3"):
            key = oracle.derive_cell_key(W, coll, "k1", _req(W, "create"))
            cell_mod.unpack_cell(cells.get(W, coll, "k1"), key,
                                 collection_id=cell_mod.cell_aad(coll, "k1"))
    return True


def test_the_pass_moves_the_existing_index_onto_collection_keys(legacy):
    db, oracle, postings, cells = legacy
    assert _search(oracle, postings, "bob") == set()          # before: under the old keys
    report = rk.rekey_index(db, oracle=oracle, cells=cells, indexer=SseIndexer(oracle, postings))
    assert report.cells_moved == 2 and report.postings_moved == 2 and report.failed == 0
    assert _cells_open(oracle, cells)
    assert _search(oracle, postings, "bob") == {"c-1"}
    assert _search(oracle, postings, "alice") == {"c-1", "c3-1"}
    assert all(postings.get_manifest(W, aid) is None for aid in DOCS)   # the old copies are gone
    again = rk.rekey_index(db, oracle=oracle, cells=cells, indexer=SseIndexer(oracle, postings))
    assert again.cells_already == 2 and again.postings_already == 2 and again.cells_moved == 0


@pytest.mark.parametrize("step", ["a cell write", "a postings write", "an old-postings removal"])
def test_the_index_pass_survives_an_interruption(legacy, monkeypatch, step):
    db, oracle, postings, cells = legacy
    ix = SseIndexer(oracle, postings)
    with monkeypatch.context() as m:
        boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("interrupted"))  # noqa: E731
        if step == "a cell write":
            m.setattr(cells, "put", boom)
        elif step == "a postings write":
            m.setattr(ix, "index_artifact", boom)
        else:
            m.setattr(ix, "remove_artifact", boom)
        report = rk.rekey_index(db, oracle=oracle, cells=cells, indexer=ix)
        assert report.failed >= 1
    # The old index still opens under the old keys where it was not moved.
    with acting_as(W, principal_type="user"):
        for coll in ("C", "C3"):
            blob = cells.get(W, coll, "k1")
            opened = False
            for key in (oracle.derive_cell_key(W, coll, "k1", _req(W, "create")),
                        oracle.derive_legacy_cell_key(W, coll, "k1", _req(W, "create"))):
                try:
                    cell_mod.unpack_cell(blob, key, collection_id=cell_mod.cell_aad(coll, "k1"))
                    opened = True
                except cell_mod.CellTampered:
                    pass
            assert opened, coll
    report = rk.rekey_index(db, oracle=oracle, cells=cells, indexer=SseIndexer(oracle, postings))
    assert report.failed == 0
    assert _cells_open(oracle, cells)
    assert _search(oracle, postings, "bob") == {"c-1"}
    assert all(postings.get_manifest(W, aid) is None for aid in DOCS)


def test_a_cell_key_never_asks_for_the_master_key(node, monkeypatch):
    db, oracle, _p = node

    def refuse(*a, **k):
        raise AssertionError("a cell key asked for the master key")

    monkeypatch.setattr(oracle, "get_or_create_master_key", refuse)
    with acting_as("alice", principal_type="user"):
        assert len(oracle.derive_cell_key(W, "C", "k1", _req("alice", "create"))) == 32


def test_an_archived_head_s_postings_move_and_a_prior_version_files_none(legacy):
    """Postings are filed by root: an archived head's are moved like a live one's, and a prior
    version (a row whose id is not its root) files nothing of its own."""
    db, oracle, postings, cells = legacy
    head = {k: v for k, v in db.artifacts.get_artifact("c-1").items() if not k.startswith("_")}
    db.artifacts.put_artifact({**head, "state": "archived"}, snapshot=False)
    db.artifacts.put_artifact({**head, "id": "c-1.v0", "root_id": "c-1", "state": "archived"},
                              snapshot=False)
    report = rk.rekey_index(db, oracle=oracle, cells=cells, indexer=SseIndexer(oracle, postings))
    assert report.postings_moved == 2 and report.failed == 0
    assert report.postings_seen == 2                          # the prior version is not visited
