"""Every collection's content is keyed under its own random collection key (John, 2026-10-31), and
existing content is moved onto it by a one-time pass at the cutover.

A grantee on one collection opens that collection's key and no sibling's, and content keyed under
a collection does not open under the principal's master key. The key is cached under its own id
for at most 30 seconds. The pass (`mantle.system.rekey_collections`) re-encrypts each row's body,
sealed fields and content-tier copy under its collection's key; it survives an interruption at
every step with the content readable throughout, leaves a row someone wrote during the pass as
they wrote it, and does nothing a second time. Small fixtures only.
"""
from __future__ import annotations

import base64
import json

import pytest
from cryptography.exceptions import InvalidTag

from mantle.attenuation import ACTIONS
from mantle.db import doc_boundary as b
from mantle.db import lattice_api as la
from mantle.entities.grant import Grant
from mantle.search.mantle import oracle as o
from mantle.services import content_crypto as cc
from mantle.services.acting_principal import acting_as
from mantle.system import rekey_collections as rk

from .test_a_split_master_key_needs_two_custodians import _FakeKek

OWNER = "W"


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def node(tmp_path, monkeypatch):
    """A workspace `W` (the principal) holding collections `C` and `C3`. Bob reads `C`."""
    from mantle.search.mantle import wiring

    db = la.LatticeDatabase(str(tmp_path / "k.db"), origin="k")
    for coll in ("C", "C3"):
        la.add_artifact_to_collection(db, OWNER, coll, origin=True)
    la.create_grant(db, Grant(id="g-bc", resource_id="C", grantee_type="user", grantee_id="bob",
                              granted_by=OWNER, **{f"can_{a}": a == "read" for a in ACTIONS}))
    clock = Clock()
    store = o.LatticeMasterKeyStore(_FakeKek(), lambda: db)
    oracle = o.OracleService(store, grant_verifier=o.LightConeGrantVerifier(db, ttl_s=0),
                             clock=clock)
    monkeypatch.setattr(wiring, "_build_oracle", lambda: oracle)
    with acting_as(OWNER, principal_type="user"):
        master = oracle.get_or_create_master_key(OWNER, o.KeyRequest(
            requester_id=OWNER, purpose=o.KeyPurpose.SELF, action="create"))
    return db, oracle, master, clock


def _as(who):
    return acting_as(who, principal_type="user")


def _write(coll, text, who=OWNER):
    with _as(who):
        return cc.encrypt_content(OWNER, text.encode(), collection_id=coll, creator_id=OWNER)


def _read(blob, coll, who=OWNER):
    with _as(who):
        return cc.decrypt_content(OWNER, blob, collection_id=coll).decode()


# -- the keys -------------------------------------------------------------------------------------

def test_content_is_keyed_under_its_collection_s_own_key(node):
    blob = _write("C", "in C")
    assert blob[:4] == b"MEC3" and _read(blob, "C") == "in C"


def test_a_grantee_on_one_collection_cannot_open_a_sibling_or_the_master(node):
    db, oracle, master, _ = node
    in_c, in_c3 = _write("C", "for bob"), _write("C3", "not for bob")
    assert _read(in_c, "C", "bob") == "for bob"
    with pytest.raises(o.GrantDenied):
        _read(in_c3, "C3", "bob")
    with _as("bob"):                              # even holding C's key, C3 does not open
        ck = oracle.get_or_create_collection_key(OWNER, "C", o.KeyRequest(
            requester_id="bob", purpose=o.KeyPurpose.GRANT))
    with pytest.raises(InvalidTag):
        cc.decrypt_content(OWNER, in_c3, collection_id="C3",
                           collection_key_provider=lambda p, c: ck)
    with pytest.raises(InvalidTag):               # nor does the master open collection content
        cc.decrypt_content(OWNER, in_c, collection_id="C", master_key_provider=lambda p: master,
                           collection_key_provider=lambda p, c: master)


def test_each_collection_s_key_is_its_own(node):
    db, oracle, _m, _ = node
    with _as(OWNER):
        keys = {c: oracle.get_or_create_collection_key(OWNER, c, o.KeyRequest(
            requester_id=OWNER, purpose=o.KeyPurpose.SELF, action="create")) for c in ("C", "C3")}
    assert keys["C"] != keys["C3"] and len(set(keys.values())) == 2


def test_a_collection_key_is_cached_under_its_own_id_for_30_seconds(node, monkeypatch):
    db, oracle, _m, clock = node
    _write("C", "x")
    reads = []
    real = oracle._store.get
    monkeypatch.setattr(oracle._store, "get", lambda kid: reads.append(kid) or real(kid))
    oracle.evict()
    for _ in range(3):
        _read(_write("C", "y"), "C")
    clock.t += 31
    _read(_write("C", "z"), "C")
    assert reads == ["ck:C", "ck:C"]


def test_a_moved_artifact_stays_under_its_creation_parent_s_key(node):
    """The content key follows the creation parent, as reach does; a move is secondary
    containment and does not re-key. Readers of the move target open it only if they reach the
    artifact; the creation parent's readers and the artifact's own grantees do."""
    from mantle.entities.artifact import Artifact

    db, _o, _m, _ = node
    for coll in ("C", "C3"):                                   # the collections, as vertices
        db.artifacts.put_artifact({"id": coll, "state": "live"})
    la.create_grant(db, Grant(id="g-dc3", resource_id="C3", grantee_type="user", grantee_id="dave",
                              granted_by=OWNER, **{f"can_{a}": a == "read" for a in ACTIONS}))
    with _as(OWNER):
        la.create_artifact(db, Artifact(id="X", collection_id="C", content="in X",
                                        created_by=OWNER, origin_root=OWNER,
                                        content_key_scope="C3"))     # the caller's is ignored
        moved = la.get_artifact(db, "X")
        moved.collection_id = "C3"
        la.update_artifact(db, moved)
    raw = db.artifacts.get_artifact("X")
    assert raw["collection_id"] == "C3" and b.content_key_scope(raw) == "C"
    assert la.get_origin_parent(db, "X")[0] == "C"

    def opens(who):
        doc = dict(raw)
        with _as(who):
            try:
                b.decrypt_artifact_content(doc)
                return doc["content"] == "in X"
            except Exception:  # noqa: BLE001 — refused
                return False

    assert opens(OWNER) and opens("bob")                       # C's reader: the creation parent
    assert not opens("dave")                                   # C3's reader: the move target
    la.create_grant(db, Grant(id="g-dx", resource_id="X", grantee_type="user", grantee_id="dave",
                              granted_by=OWNER, **{f"can_{a}": a == "read" for a in ACTIONS}))
    assert opens("dave")                                       # reaching X itself, it opens


# -- the pass ---------------------------------------------------------------------------------------

class FakeCas:
    """The content tier: blobs by ref."""

    def __init__(self):
        self.blobs, self.writes = {}, 0

    def as_cas(self, fail_after_write=False):
        def read(doc, principal, scope):
            return cc.decrypt_content(principal, self.blobs[doc["content_ref"]],
                                      collection_id=scope)

        def raw(doc):
            return self.blobs.get(doc.get("content_ref"))

        def write(doc, principal, scope, data):
            blob = cc.encrypt_content(principal, data, collection_id=scope)
            self.writes += 1
            ref = f"cas/new-{self.writes}"
            self.blobs[ref] = blob
            if fail_after_write:
                raise RuntimeError("interrupted after the content tier write")
            return ref

        def drop(ref):
            self.blobs.pop(ref, None)

        return rk.Cas(read=read, raw=raw, write=write, drop=drop)


def _legacy_row(db, master, rid, coll, text, cas=None):
    """A row written before collections had keys: body, sealed fields and a content-tier copy
    under the master key (MEC1)."""
    def mec1(data):
        return cc.encrypt_content(OWNER, data, collection_id=coll,
                                  master_key_provider=lambda p: master)

    sealed = json.dumps({"v": b._PAYLOAD_VERSION, "context": {"title": f"t-{rid}"},
                         "doc": {}}).encode()
    doc = {"id": rid, "collection_id": coll, "origin_root": OWNER,
           b.CONTENT_KEY_PRINCIPAL: OWNER, "created_by": OWNER, "state": "live",
           "content": base64.b64encode(mec1(text.encode())).decode(), "content_encrypted": True,
           "context": {b.SEALED: {"p": OWNER, "blob": base64.b64encode(mec1(sealed)).decode()}}}
    if cas is not None:
        ref = f"cas/old-{rid}"
        cas.blobs[ref] = mec1(f"body of {rid}".encode())
        doc["content_ref"] = ref
    db.artifacts.put_artifact(doc)


def _readable(db, rid, cas=None):
    """Everything in the row opens, under whichever key it is now under."""
    raw = db.artifacts.get_artifact(rid)
    with _as(OWNER):
        d = dict(raw)
        b._decrypt_body(d)
        title = b._open(OWNER, raw["context"][b.SEALED]["blob"], raw)["context"]["title"]
        body = (cc.decrypt_content(OWNER, cas.blobs[raw["content_ref"]],
                                   collection_id=raw["collection_id"]).decode()
                if cas is not None else None)
    return d["content"], title, body


def _magics(db, rid):
    raw = db.artifacts.get_artifact(rid)
    return (base64.b64decode(raw["content"])[:4],
            base64.b64decode(raw["context"][b.SEALED]["blob"])[:4])


def test_the_pass_moves_every_row_onto_its_collection_s_key_and_reports(node):
    db, _o, master, _ = node
    cas = FakeCas()
    _legacy_row(db, master, "a1", "C", "one", cas)
    _legacy_row(db, master, "a2", "C3", "two", cas)
    before = {r: _readable(db, r, cas) for r in ("a1", "a2")}
    seen = []
    report = rk.rekey(db, cas=cas.as_cas(), every=1, progress=lambda r: seen.append(r.seen))
    assert report.as_dict()["rekeyed"] == 2 and report.as_dict()["collections"] == 2
    assert seen and seen[-1] == report.seen
    for r in ("a1", "a2"):
        assert _magics(db, r) == (b"MEC3", b"MEC3")
        assert db.artifacts.get_artifact(r)[rk.CAS_MARK] is True
        assert _readable(db, r, cas) == before[r]


def test_a_second_run_does_nothing(node):
    db, _o, master, _ = node
    _legacy_row(db, master, "a1", "C", "one")
    rk.rekey(db, cas=FakeCas().as_cas())
    report = rk.rekey(db, cas=FakeCas().as_cas())
    assert report.rekeyed == 0 and report.already >= 1


def _versions(db, rid):
    return sorted((d for d in db.artifacts.list_artifacts(include_archived=True)
                   if str(d.get("root_id") or d.get("id")) == rid), key=lambda d: d["id"])


def test_prior_versions_and_content_tier_copies_move_and_a_rerun_writes_nothing(node):
    """A content-tier body and a prior version: the pass rewrites both in place, removes the old
    copies once no row names them, and a second run changes nothing, adding no version."""
    db, _o, master, _ = node
    cas = FakeCas()
    _legacy_row(db, master, "a1", "C", "one", cas)
    head = {k: v for k, v in db.artifacts.get_artifact("a1").items() if not k.startswith("_")}
    cas.blobs["cas/old2-a1"] = cc.encrypt_content(OWNER, b"body two", collection_id="C",
                                                  master_key_provider=lambda p: master)
    db.artifacts.put_artifact({**head, "content_ref": "cas/old2-a1"})      # a new version
    rows = _versions(db, "a1")
    assert len(rows) == 2                                                  # head + prior
    db.artifacts.put_artifact({"id": "loose", "state": "live", "created_by": OWNER,
                               "content": base64.b64encode(cc.encrypt_content(
                                   OWNER, b"x", master_key_provider=lambda p: master)).decode(),
                               "content_encrypted": True})                 # no collection
    first = rk.rekey(db, cas=cas.as_cas())
    assert first.rekeyed == 2 and first.old_copies_removed == 2
    assert first.as_dict()["left"] == {"loose": "unscoped"}
    after = _versions(db, "a1")
    assert [d["id"] for d in after] == [d["id"] for d in rows]             # no new version
    for d in after:
        assert base64.b64decode(d["content"])[:4] == b"MEC3"
        assert cas.blobs[d["content_ref"]][:4] == b"MEC3"
    assert not any(ref.startswith("cas/old") for ref in cas.blobs)          # old copies gone
    writes = cas.writes
    second = rk.rekey(db, cas=cas.as_cas())
    assert second.rekeyed == 0 and cas.writes == writes
    assert [d["_seq"] for d in _versions(db, "a1")] == [d["_seq"] for d in after]
    assert second.as_dict()["left"] == {"loose": "unscoped"}


def test_a_content_tier_that_takes_no_copy_leaves_the_row_to_do(node):
    db, _o, master, _ = node
    cas = FakeCas()
    _legacy_row(db, master, "a1", "C", "one", cas)
    c = cas.as_cas()
    c.write = lambda doc, principal, scope, data: None              # not landed
    report = rk.rekey(db, cas=c)
    assert report.failed == 1 and report.as_dict()["left"] == {"a1": "failed"}
    assert not db.artifacts.get_artifact("a1").get(rk.CAS_MARK)
    assert rk.rekey(db, cas=cas.as_cas()).rekeyed == 1                # a rerun completes it


def test_a_row_changed_mid_pass_leaves_no_orphan_copy(node, monkeypatch):
    db, _o, master, _ = node
    cas = FakeCas()
    _legacy_row(db, master, "a1", "C", "one", cas)
    real = rk._plan

    def plan_then_someone_writes(doc, c):
        todo = real(doc, c)
        fresh = {k: v for k, v in db.artifacts.get_artifact("a1").items() if not k.startswith("_")}
        db.artifacts.put_artifact({**fresh, "state": "live", "name": "edited"})
        return todo

    monkeypatch.setattr(rk, "_plan", plan_then_someone_writes)
    report = rk.rekey(db, cas=cas.as_cas())
    assert report.changed == 1
    assert not [r for r in cas.blobs if r.startswith("cas/new")]       # the new copy is gone


def test_a_dry_run_changes_nothing(node):
    db, _o, master, _ = node
    _legacy_row(db, master, "a1", "C", "one")
    assert rk.rekey(db, dry_run=True, cas=FakeCas().as_cas()).rekeyed == 1
    assert _magics(db, "a1") == (b"MEC1", b"MEC1")


@pytest.mark.parametrize("step", ["after the key is minted", "after the content tier write",
                                  "at the row write", "at the read-back check"])
def test_the_pass_survives_an_interruption_at_every_step(node, monkeypatch, step):
    """Interrupted at each step, the row reads exactly as before; run again, it completes."""
    db, _o, master, _ = node
    cas = FakeCas()
    _legacy_row(db, master, "a1", "C", "one", cas)
    before = _readable(db, "a1", cas)
    with monkeypatch.context() as m:
        if step == "after the key is minted":
            real = cc.encrypt_content
            m.setattr(cc, "encrypt_content", lambda *a, **k: (real(*a, **k), (_ for _ in ()).throw(
                RuntimeError("interrupted")))[0])
            c = cas.as_cas()
        elif step == "after the content tier write":
            c = cas.as_cas(fail_after_write=True)
        elif step == "at the row write":
            m.setattr(db.artifacts, "put_artifact",
                      lambda *a, **k: (_ for _ in ()).throw(RuntimeError("interrupted")))
            c = cas.as_cas()
        else:
            real_dec = cc.decrypt_content
            calls = []

            def flaky(*a, **k):
                calls.append(1)
                out = real_dec(*a, **k)
                return out if len(calls) % 2 else out + b"!"      # the read-back differs
            m.setattr(cc, "decrypt_content", flaky)
            c = cas.as_cas()
        report = rk.rekey(db, cas=c)
        assert report.failed == 1 and report.rekeyed == 0
    assert _readable(db, "a1", cas) == before                   # readable, as it was
    assert rk.rekey(db, cas=cas.as_cas()).rekeyed == 1          # and a rerun completes it
    assert _magics(db, "a1") == (b"MEC3", b"MEC3")
    assert _readable(db, "a1", cas) == before


def test_content_written_during_the_pass_stays_as_its_writer_wrote_it(node, monkeypatch):
    """A row written between the pass reading it and writing it back is left alone, and the new
    content, already under the collection key, reads; the next run finds nothing to do."""
    db, _o, master, _ = node
    _legacy_row(db, master, "a1", "C", "old")
    real = rk._plan

    def plan_then_someone_writes(doc, cas):
        todo = real(doc, cas)
        fresh = db.artifacts.get_artifact("a1")
        fresh = {k: v for k, v in fresh.items() if not k.startswith("_")}
        fresh["content"] = base64.b64encode(_write("C", "new")).decode()
        db.artifacts.put_artifact(fresh)
        return todo

    with monkeypatch.context() as m:
        m.setattr(rk, "_plan", plan_then_someone_writes)
        report = rk.rekey(db, cas=FakeCas().as_cas())
    assert report.changed == 1 and report.rekeyed == 0
    raw = dict(db.artifacts.get_artifact("a1"))
    with _as(OWNER):
        b._decrypt_body(raw)
    assert raw["content"] == "new"
    assert rk.rekey(db, cas=FakeCas().as_cas()).rekeyed == 1      # its sealed fields, still old


def test_the_head_s_object_store_copy_is_always_the_head(node, monkeypatch):
    """A prior version shares its head's content key. Each rewrite stores under a key of its own
    and the row names it, so no version writes over the head's object-store copy; the copy no row
    names any more is removed from the object store, not only from the local tier."""
    from mantle.services import content_service as cs

    db, _o, master, _ = node
    cas = FakeCas()
    _legacy_row(db, master, "a1", "C", "one", cas)
    head = {k: v for k, v in db.artifacts.get_artifact("a1").items() if not k.startswith("_")}
    head["context"] = {**head["context"], "content_key": "artifacts/a1.content"}
    db.artifacts.put_artifact(head)
    head = {k: v for k, v in db.artifacts.get_artifact("a1").items() if not k.startswith("_")}
    cas.blobs["cas/old2-a1"] = cc.encrypt_content(OWNER, b"NEWEST body", collection_id="C",
                                                  master_key_provider=lambda p: master)
    db.artifacts.put_artifact({**head, "content_ref": "cas/old2-a1"})      # a new version
    bucket = {"artifacts/a1.content": b"old object"}

    def put_bytes_encrypted(content_key, data, content_type, owner_id, *, collection_id=None,
                            creator_id=None, cas=True, on_mirror_deferred=None):
        blob = cc.encrypt_content(owner_id, data, collection_id=collection_id,
                                  creator_id=creator_id)
        bucket[content_key] = blob
        ref = f"cas/new-{len(bucket)}"
        fake.blobs[ref] = blob
        return ref

    fake = cas
    monkeypatch.setattr(cs, "put_bytes_encrypted", put_bytes_encrypted)
    monkeypatch.setattr(cs, "delete_object", lambda key: bucket.pop(key, None) is not None)
    default = rk.Cas.default()
    use = rk.Cas(read=cas.as_cas().read, raw=cas.as_cas().raw, write=default.write,
                 drop=cas.as_cas().drop, drop_object=default.drop_object)
    rk.rekey(db, cas=use)
    live = db.artifacts.get_artifact("a1")
    key = live["context"]["content_key"]
    with _as(OWNER):
        assert cc.decrypt_content(OWNER, bucket[key], collection_id="C") == b"NEWEST body"
    assert "artifacts/a1.content" not in bucket                       # the old copy is gone
    keys = {d["context"]["content_key"] for d in _versions(db, "a1")}
    assert len(keys) == 2 and keys <= set(bucket)                    # each version its own


def test_an_old_object_copy_named_by_a_string_context_is_kept_until_no_version_names_it(
        node, monkeypatch):
    """References are read from each row's parsed context, so a context stored as a JSON string
    counts: the old object-store copy is removed only when no version names it."""
    from mantle.services import content_service as cs

    db, _o, master, _ = node
    cas = FakeCas()
    _legacy_row(db, master, "a1", "C", "one", cas)
    head = {k: v for k, v in db.artifacts.get_artifact("a1").items() if not k.startswith("_")}
    ctx = dict(head["context"])
    ctx["content_key"] = "artifacts/a1.content"
    head["context"] = json.dumps(ctx)                               # a string context
    db.artifacts.put_artifact(head)
    head = {k: v for k, v in db.artifacts.get_artifact("a1").items() if not k.startswith("_")}
    cas.blobs["cas/old2-a1"] = cc.encrypt_content(OWNER, b"newer", collection_id="C",
                                                  master_key_provider=lambda p: master)
    db.artifacts.put_artifact({**head, "content_ref": "cas/old2-a1"})      # a prior version
    assert len(_versions(db, "a1")) == 2
    bucket = {"artifacts/a1.content": b"old"}
    removed = []

    def put_bytes_encrypted(content_key, data, content_type, owner_id, **kw):
        blob = cc.encrypt_content(owner_id, data, collection_id=kw.get("collection_id"))
        bucket[content_key] = blob
        ref = f"cas/new-{len(bucket)}"
        cas.blobs[ref] = blob
        return ref

    monkeypatch.setattr(cs, "put_bytes_encrypted", put_bytes_encrypted)
    monkeypatch.setattr(cs, "delete_object", lambda k: removed.append(k) or bucket.pop(k, None))
    default = rk.Cas.default()
    use = rk.Cas(read=cas.as_cas().read, raw=cas.as_cas().raw, write=default.write,
                 drop=cas.as_cas().drop, drop_object=default.drop_object)
    first = []
    real_one = rk.rekey_one

    def one_then_look(db_, doc, cas_, report=None, **kw):
        out = real_one(db_, doc, cas_, report, **kw)
        first.append(list(removed))
        return out

    monkeypatch.setattr(rk, "rekey_one", one_then_look)
    rk.rekey(db, cas=use)
    assert first[0] == []                     # the other version still named it after the first
    assert removed == ["artifacts/a1.content"]   # and once neither does, it goes


def test_a_stale_reference_count_never_removes_a_copy_a_version_still_names(node, monkeypatch):
    """F7: the pass counts references once, at its start. A version written after that count
    and naming the same object-store key must keep the copy: the count is checked against the
    artifact's own versions before anything is removed."""
    from mantle.services import content_service as cs

    db, _o, master, _ = node
    cas = FakeCas()
    _legacy_row(db, master, "a1", "C", "one", cas)
    head = {k: v for k, v in db.artifacts.get_artifact("a1").items() if not k.startswith("_")}
    head["context"] = {**head["context"], "content_key": "artifacts/a1.content"}
    db.artifacts.put_artifact(head)
    stale = {"artifacts/a1.content": 1}                     # counted before the next version
    head = {k: v for k, v in db.artifacts.get_artifact("a1").items() if not k.startswith("_")}
    cas.blobs["cas/old2-a1"] = cc.encrypt_content(OWNER, b"newer", collection_id="C",
                                                  master_key_provider=lambda p: master)
    db.artifacts.put_artifact({**head, "content_ref": "cas/old2-a1"})  # a version naming it too
    assert len(_versions(db, "a1")) >= 2
    removed = []

    def put_bytes_encrypted(content_key, data, content_type, owner_id, **kw):
        blob = cc.encrypt_content(owner_id, data, collection_id=kw.get("collection_id"))
        ref = f"cas/new-{len(cas.blobs)}"
        cas.blobs[ref] = blob
        return ref

    monkeypatch.setattr(cs, "put_bytes_encrypted", put_bytes_encrypted)
    monkeypatch.setattr(cs, "delete_object", lambda k: removed.append(k))
    default = rk.Cas.default()
    use = rk.Cas(read=cas.as_cas().read, raw=cas.as_cas().raw, write=default.write,
                 drop=cas.as_cas().drop, drop_object=default.drop_object)
    live = {k: v for k, v in db.artifacts.get_artifact("a1").items()}
    rk.rekey_one(db, live, use, refs=stale)
    assert "artifacts/a1.content" not in removed           # a prior version still names it


def test_the_pass_leaves_no_master_keyed_bytes_in_the_store_file(node, tmp_path):
    """Review 7: the inline body the pass re-keys must not survive under the master key in the
    store file (SQLite keeps a replaced row's bytes until the page is reused)."""
    db, _o, master, _ = node
    _legacy_row(db, master, "a1", "C", "one")
    old = db.artifacts.get_artifact("a1")["content"]
    assert rk.rekey(db, cas=FakeCas().as_cas()).rekeyed == 1
    raw = b"".join(p.read_bytes() for p in tmp_path.glob("k.db*") if p.is_file())
    assert old.encode() not in raw
