"""A canonical AnchorSet loads with its ids intact, and a mismatched one does not load at all.

An anchor id is `uuid5(_ANCHOR_NS, sha256(label ‖ model_id ‖ embedding))`, and that id IS the
cluster id: it names the cell storage path, the HKDF key `info`, the AEAD associated data and the
mesh region. Every one of those accepts an arbitrary string, so an id that does not match its own
content produces a node that routes confidently into regions no peer computes — queries miss,
cells never hit, the semantic arm returns nothing, the request 200s on lexical results, and mesh
sync transfers nothing and reports success. Nothing downstream can notice. These tests hold the
two places where it CAN be noticed: the file that carries the set, and the store that holds it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mantle.search.anchors.anchorset import (
    Anchor,
    AnchorSet,
    AnchorSetCorrupt,
    anchor_content_hash,
    anchor_id_for,
    anchorset_fingerprint,
    l2norm,
    verify_anchor_id,
)
from mantle.search.anchors.repo import InMemoryAnchorRepo, _build_anchorset

D = 16
MODEL = "hf:test@1.0"


def _canonical(n: int = 24, dim: int = D, model: str = MODEL, seed: int = 5) -> AnchorSet:
    rng = np.random.default_rng(seed)
    s = AnchorSet(model, dim)
    for i in range(n):
        s.add_text(f"anchor-{i}", rng.standard_normal(dim))
    return s


# ── the round trip ──────────────────────────────────────────────────────────────────────────────

def test_ids_survive_a_file_round_trip_into_a_fresh_store(tmp_path):
    """Build, save, load into a store that has never held an anchor — same ids, exactly."""
    src = _canonical()
    path = tmp_path / "anchors.json"
    src.save(path)

    stated = [a["anchor_id"] for a in json.loads(path.read_text(encoding="utf-8"))["anchors"]]

    repo = InMemoryAnchorRepo()          # a node that has never provisioned
    assert repo.load() is None
    repo.bulk_add(AnchorSet.load(path).anchors)

    stored = repo.load()
    assert stored is not None
    assert {a.anchor_id for a in stored.anchors} == set(stated)
    assert {a.anchor_id for a in stored.anchors} == {a.anchor_id for a in src.anchors}
    # The fingerprint is the whole claim in one value: same coordinate system, same regions.
    assert anchorset_fingerprint(stored) == anchorset_fingerprint(src)


def test_loading_the_same_file_twice_is_a_no_op(tmp_path):
    """`add` is idempotent on `anchor_id`, so re-running the load must not duplicate."""
    path = tmp_path / "anchors.json"
    _canonical().save(path)

    repo = InMemoryAnchorRepo()
    repo.bulk_add(AnchorSet.load(path).anchors)
    first = repo.count()
    repo.bulk_add(AnchorSet.load(path).anchors)
    assert repo.count() == first == 24


def test_embeddings_are_preserved_bit_for_bit_so_the_ids_stay_derivable(tmp_path):
    """The id is a hash of the embedding's bytes, so reading must not move them.

    `l2norm` is not bitwise idempotent in float32 — re-normalising an already-unit vector shifts
    roughly a third of them by an ulp — so a reader that normalises on the way in hands back
    anchors whose bytes do not produce the ids they arrived with, and save→load→save emits a file
    that its own verification rejects.
    """
    src = _canonical()
    p, q = tmp_path / "a.json", tmp_path / "b.json"
    src.save(p)
    mid = AnchorSet.load(p)
    for a, b in zip(src.anchors, mid.anchors):
        assert a.embedding.tobytes() == b.embedding.tobytes()
    mid.save(q)
    assert p.read_text(encoding="utf-8") == q.read_text(encoding="utf-8")


# ── refusals ────────────────────────────────────────────────────────────────────────────────────

def test_a_stated_id_that_disagrees_with_its_content_refuses_the_whole_file(tmp_path):
    src = _canonical()
    path = tmp_path / "anchors.json"
    src.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["anchors"][7]["anchor_id"] = "00000000-0000-0000-0000-000000000000"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(AnchorSetCorrupt) as e:
        AnchorSet.load(path)
    assert "anchor-7" in str(e.value)


def test_an_edited_embedding_refuses_too(tmp_path):
    """The id is over the vector as well as the label — moving the geometry invalidates it."""
    src = _canonical()
    path = tmp_path / "anchors.json"
    src.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["anchors"][2]["embedding"][0] += 0.25
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(AnchorSetCorrupt):
        AnchorSet.load(path)


def test_a_refused_file_loads_nothing_at_all(tmp_path):
    """Refused whole, not thinned: a partial canonical set is a different set."""
    src = _canonical()
    path = tmp_path / "anchors.json"
    src.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["anchors"][0]["label"] = "relabelled"      # the id no longer follows from the content
    path.write_text(json.dumps(raw), encoding="utf-8")

    repo = InMemoryAnchorRepo()
    with pytest.raises(AnchorSetCorrupt):
        repo.bulk_add(AnchorSet.load(path).anchors)
    assert repo.count() == 0


def test_a_mixed_width_file_refuses_rather_than_loading_the_majority(tmp_path):
    src = _canonical()
    path = tmp_path / "anchors.json"
    src.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    wide = Anchor.make("wide", np.arange(1, D * 2 + 1, dtype=np.float32), MODEL)
    raw["anchors"].append({
        "anchor_id": wide.anchor_id, "label": wide.label, "model_id": wide.model_id,
        "type_id": wide.type_id, "tier": wide.tier, "placed_frame": 0, "status": "active",
        "embedding": wide.embedding.astype(float).tolist(),
    })
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(AnchorSetCorrupt) as e:
        AnchorSet.load(path)
    assert "wide" in str(e.value)


def test_a_mixed_model_file_refuses(tmp_path):
    src = _canonical()
    path = tmp_path / "anchors.json"
    src.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    foreign = Anchor.make("foreign", np.eye(D)[0], "hf:other@2.0")
    raw["anchors"].append({
        "anchor_id": foreign.anchor_id, "label": foreign.label, "model_id": foreign.model_id,
        "type_id": foreign.type_id, "tier": foreign.tier, "placed_frame": 0, "status": "active",
        "embedding": foreign.embedding.astype(float).tolist(),
    })
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(AnchorSetCorrupt):
        AnchorSet.load(path)


# ── the store's own assembly ────────────────────────────────────────────────────────────────────

def test_a_mixed_set_is_refused_by_the_repo_not_thinned():
    """`_build_anchorset` used to take dim/model from whichever anchor it read first and
    `logger.debug`-skip the rest — which resolves a two-vocabulary store into whichever the
    listing happened to order first, differently on a node whose listing order differs."""
    good = _canonical(n=3)
    foreign = Anchor.make("foreign", np.eye(D * 2)[0], MODEL)
    with pytest.raises(AnchorSetCorrupt) as e:
        _build_anchorset(good.anchors + [foreign])
    assert "foreign" in str(e.value)


def test_a_store_whose_anchor_states_a_wrong_id_refuses_to_load(monkeypatch):
    """The documented POST /artifacts route assigns a fresh uuid4, which is exactly this."""
    from mantle.db import backend as db_store
    from mantle.search.anchors.repo import StoreAnchorRepo
    from mantle.services import platform_topology

    docs: dict = {}
    monkeypatch.setattr(platform_topology, "get_id_optional", lambda slug: "anchorset-col")
    monkeypatch.setattr(db_store, "get_collection_by_id", lambda _db, cid: object())
    monkeypatch.setattr(db_store, "get_artifact", lambda _db, aid: docs.get(aid))
    monkeypatch.setattr(db_store, "add_artifact_to_collection", lambda *a, **k: True)
    monkeypatch.setattr(db_store, "get_raw_artifacts", lambda _db, ids: {})
    monkeypatch.setattr(db_store, "list_collection_artifacts",
                        lambda _db, cid, **kw: list(docs.values()))

    def _create(_db, entity):
        d = entity.to_dict()
        d["content_type"] = entity.content_type
        docs[entity.id] = d
        return entity
    monkeypatch.setattr(db_store, "create_artifact", _create)

    repo = StoreAnchorRepo(object())
    repo.bulk_add(_canonical(n=4).anchors)
    assert repo.load() is not None                       # ids match their content: fine

    # Now the shape POST /artifacts produces: the same anchor record under a minted uuid4.
    victim = next(iter(docs.values()))
    minted = dict(victim)
    minted["id"] = minted["root_id"] = "11111111-2222-3333-4444-555555555555"
    docs[minted["id"]] = minted

    with pytest.raises(AnchorSetCorrupt) as e:
        repo.load()
    assert "manage_anchors --action load" in str(e.value)


# ── the primitives ──────────────────────────────────────────────────────────────────────────────

def test_verify_anchor_id_accepts_what_make_produced_and_nothing_else():
    a = Anchor.make("x", l2norm(np.eye(D)[3]), MODEL)
    assert verify_anchor_id(a.anchor_id, a.label, a.model_id, a.embedding) is None
    assert verify_anchor_id(a.anchor_id, "y", a.model_id, a.embedding) is not None
    assert verify_anchor_id(a.anchor_id, a.label, "hf:other@1", a.embedding) is not None
    assert anchor_id_for(anchor_content_hash(a.label, a.model_id, a.embedding)) == a.anchor_id


def test_verification_must_hash_the_stored_bytes_not_a_renormalised_copy():
    """The reason `anchor_content_hash` refuses to normalise, stated as a measurement.

    Dividing an already-unit float32 vector by its recomputed norm is not the identity: it moves
    a large minority of vectors by an ulp. A verifier that re-normalised before hashing would
    therefore reject that same fraction of a perfectly good canonical set — the check would fire
    on the honest case and teach operators to route around it.
    """
    rng = np.random.default_rng(3)
    anchors = [Anchor.make(f"a{i}", rng.standard_normal(384), MODEL) for i in range(300)]

    assert all(verify_anchor_id(a.anchor_id, a.label, a.model_id, a.embedding) is None
               for a in anchors)

    renormalised_would_reject = sum(
        1 for a in anchors
        if verify_anchor_id(a.anchor_id, a.label, a.model_id, l2norm(a.embedding)) is not None
    )
    assert renormalised_would_reject > 0, (
        "the float32 double-normalisation drift this guard is about did not occur, so the guard "
        "is no longer measuring anything — re-derive the tolerance before relaxing the rule"
    )


# ── the load command ────────────────────────────────────────────────────────────────────────────

def test_the_load_action_preserves_ids_and_is_re_runnable(tmp_path, monkeypatch, caplog):
    """`manage_anchors --action load` is the id-preserving route.

    It exists because the documented one cannot be: `CreateArtifactRequest` has no `id` field and
    `workspace_service` assigns `str(uuid.uuid4())`, so POST /artifacts replaces exactly the value
    that makes two nodes' cells comparable.
    """
    from mantle.search.anchors import store as anchor_store
    from mantle.system import manage_anchors as ma

    src = _canonical(n=30)
    path = tmp_path / "anchors.json"
    src.save(path)

    repo = InMemoryAnchorRepo()
    monkeypatch.setattr(anchor_store, "_repo_override", repo)

    with caplog.at_level("INFO"):
        ma.action_load(str(path), dry_run=False)
    assert {a.anchor_id for a in repo.load().anchors} == {a.anchor_id for a in src.anchors}
    assert anchorset_fingerprint(src)[:16] in caplog.text     # comparable without exporting data

    ma.action_load(str(path), dry_run=False)                  # re-runnable
    assert repo.count() == 30


def test_the_load_action_refuses_a_corrupt_file_and_writes_nothing(tmp_path, monkeypatch):
    from mantle.search.anchors import store as anchor_store
    from mantle.system import manage_anchors as ma

    path = tmp_path / "anchors.json"
    _canonical(n=8).save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["anchors"][4]["anchor_id"] = str(np.random.default_rng(0).integers(0, 9)) + \
        "0000000-0000-0000-0000-000000000000"
    path.write_text(json.dumps(raw), encoding="utf-8")

    repo = InMemoryAnchorRepo()
    monkeypatch.setattr(anchor_store, "_repo_override", repo)

    with pytest.raises(SystemExit) as e:
        ma.action_load(str(path), dry_run=False)
    assert "REFUSED" in str(e.value)
    assert repo.count() == 0


def test_the_load_action_dry_run_writes_nothing(tmp_path, monkeypatch):
    from mantle.search.anchors import store as anchor_store
    from mantle.system import manage_anchors as ma

    path = tmp_path / "anchors.json"
    _canonical(n=6).save(path)
    repo = InMemoryAnchorRepo()
    monkeypatch.setattr(anchor_store, "_repo_override", repo)

    ma.action_load(str(path), dry_run=True)
    assert repo.count() == 0


def test_the_load_action_reports_progress_for_large_sets(tmp_path, monkeypatch, caplog):
    from mantle.search.anchors import store as anchor_store
    from mantle.search.anchors.repo import _PROGRESS_EVERY
    from mantle.system import manage_anchors as ma

    path = tmp_path / "anchors.json"
    _canonical(n=_PROGRESS_EVERY * 2 + 3).save(path)
    monkeypatch.setattr(anchor_store, "_repo_override", InMemoryAnchorRepo())

    with caplog.at_level("INFO"):
        ma.action_load(str(path), dry_run=False)
    assert f"{_PROGRESS_EVERY}/{_PROGRESS_EVERY * 2 + 3} anchors" in caplog.text


def test_a_corrupt_store_is_not_reported_as_an_unprovisioned_one(monkeypatch):
    """`None` from the repo means "nobody provisioned this node", and `require_live_anchorset`
    says so at length. That is the wrong answer — and an actively misleading one — for a node
    somebody DID provision, incorrectly. The corruption propagates instead of being flattened."""
    from mantle.search.anchors import store as anchor_store

    class _Corrupt:
        def load(self):
            raise AnchorSetCorrupt("anchor 'x' states an id its content does not produce")

        def add(self, anchor): ...
        def bulk_add(self, anchors, *, progress=None): ...
        def count(self): return 1

    anchor_store.set_anchor_repo(_Corrupt())
    try:
        with pytest.raises(AnchorSetCorrupt):
            anchor_store.get_live_anchorset()
        with pytest.raises(AnchorSetCorrupt):
            anchor_store.require_live_anchorset()
    finally:
        anchor_store.set_anchor_repo(None)
