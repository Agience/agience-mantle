"""`mantle.system.rewrap_kek`: rotate the KEK (the node's `encryption.key`), review 10.

John, 2026-11-02: every key from the root era rotates at cutover. The KEK wraps every stored key
(whole master and collection keys, and this node's share of each split key) and seals the node's own
fields (grants' names and notes, link names and their keyed tokens, secret platform settings). The
pass re-wraps and re-seals each under the new KEK, verifies each by opening it again, and then
scrubs the old wraps from the store file. `--dry-run` counts and writes nothing.
"""
from __future__ import annotations

import json
import os

import pytest
from cryptography.fernet import Fernet

from mantle.db import lattice_api as la
from mantle.db import node_seal
from mantle.entities.grant import Grant
from mantle.search.mantle.key_provider import LocalKeyProvider
from mantle.search.mantle.oracle import SHARE_CONTENT_TYPE, LatticeMasterKeyStore

from .test_there_is_no_promotion_state import USER, real_node

OLD, NEW = Fernet.generate_key().decode(), Fernet.generate_key().decode()


@pytest.fixture
def db(tmp_path, monkeypatch):
    import prism.trust.key_manager as km
    monkeypatch.setattr(km, "_encryption_key", OLD)
    database = real_node(tmp_path, monkeypatch)
    old = LocalKeyProvider(Fernet(OLD))
    LatticeMasterKeyStore(old, lambda: database).put("alice", b"A" * 32)
    LatticeMasterKeyStore(old, lambda: database).put("ck:c-1", b"C" * 32)
    database.artifacts.put_artifact({"id": "master-key:bob", "content_type": SHARE_CONTENT_TYPE,
                                     "state": "active", "kcv": "k", "share": old.wrap(b"S" * 34)})
    la.create_grant(database, Grant(id="g-1", resource_id="c-1", grantee_type="user",
                                    grantee_id=USER, granted_by=USER, can_read=True,
                                    name="Shared marrowfinch"))
    la.add_artifact_to_collection(database, "c-1", "a-1", relationship="step1.marrowfinch")
    return database


def _run(db, tmp_path, **kw):
    from mantle.system.rewrap_kek import rewrap_kek
    p = tmp_path / "new.key"
    p.write_text(NEW)
    return rewrap_kek(db, new_key=p.read_text(), **kw)


def _file_bytes(tmp_path):
    out = b""
    for name in os.listdir(tmp_path):
        if name.startswith("live.db"):
            out += (tmp_path / name).read_bytes()
    return out


def test_a_dry_run_counts_and_writes_nothing(db, tmp_path):
    before = json.dumps(sorted((d["id"], json.dumps(d, sort_keys=True))
                               for d in db.artifacts.list_artifacts()))
    counts = _run(db, tmp_path, dry_run=True)
    assert counts["keys"] == 3 and counts["sealed_rows"] >= 1 and counts["labels"] == 1
    assert json.dumps(sorted((d["id"], json.dumps(d, sort_keys=True))
                             for d in db.artifacts.list_artifacts())) == before


def test_every_stored_key_opens_under_the_new_kek_alone(db, tmp_path, monkeypatch):
    import prism.trust.key_manager as km

    raw_before = {i: db.artifacts.get_artifact(i) for i in ("master-key:alice",
                                                            "master-key:ck:c-1",
                                                            "master-key:bob")}
    counts = _run(db, tmp_path)
    assert counts["keys"] == 3 and counts["failed"] == 0
    new = LocalKeyProvider(Fernet(NEW))
    assert LatticeMasterKeyStore(new, lambda: db).get("alice") == b"A" * 32
    assert LatticeMasterKeyStore(new, lambda: db).get("ck:c-1") == b"C" * 32
    assert new.unwrap(db.artifacts.get_artifact("master-key:bob")["share"]) == b"S" * 34
    with pytest.raises(Exception):
        LatticeMasterKeyStore(LocalKeyProvider(Fernet(OLD)), lambda: db).get("alice")
    # The node's own fields, under the new key.
    monkeypatch.setattr(km, "_encryption_key", NEW)
    assert la.get_grant_by_id(db, "g-1").name == "Shared marrowfinch"
    edges = [e for e in db.graph.db.read().execute("SELECT label, props FROM edge "
                                                   "WHERE src='c-1' AND dst='a-1'").fetchall()
             if node_seal.is_token(e["label"])]
    assert len(edges) == 1 and edges[0]["label"] == la._stored_label("step1.marrowfinch")
    # The old wraps are gone from the store file.
    data = _file_bytes(tmp_path)
    for doc in raw_before.values():
        tok = (doc.get("token") or doc.get("share") or "").encode()
        assert tok and tok not in data


def test_a_second_run_finds_everything_already_under_the_new_kek(db, tmp_path, monkeypatch):
    import prism.trust.key_manager as km

    _run(db, tmp_path)
    monkeypatch.setattr(km, "_encryption_key", OLD)          # the old key file is still in place
    counts = _run(db, tmp_path)
    assert (counts["keys"], counts["sealed_rows"], counts["labels"], counts["failed"]) == (0, 0, 0, 0)
    assert counts["already"] == 5                    # three keys, a grant and a link name
