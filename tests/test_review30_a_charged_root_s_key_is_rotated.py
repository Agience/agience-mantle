"""Review 30, F4 (John, 2026-11-08): a charged root's master key is rotated at step 21, not
confirmed.

A root a part of which was removed is charged at Origin for good: its master key opens for its
own principal alone. At step 21, under lattice custody (where this node holds every key anyway),
`rotate_charged_roots` takes what is still under the root's old key:
  * a row whose root, by this store's own creation edges (the edges Origin co-signed), is still the
    root is re-encrypted under a fresh key for the root;
  * a row whose root is no longer the root (it belonged to the removed part) is left under the old
    key, and the old key is then deleted, so the row is shredded;
  * the new key and the rewritten rows are written in one transaction, and the old key's bytes
    settled out of the store file;
  * a root with nothing left under it has its whole key removed.
No one ever opens the old key under threshold custody, because it no longer exists then.
"""
from __future__ import annotations

import base64

import pytest

from mantle.db import doc_boundary as b
from mantle.db import lattice_api as la
from mantle.search.mantle import oracle as o
from mantle.services import content_crypto as cc
from mantle.system import rekey_collections as rk

from .test_a_split_master_key_needs_two_custodians import _FakeKek

ROOT, PART = "r-team", "c-removed"
OLD = bytes(range(32))


def _mec1(key, data):
    return base64.b64encode(cc.encrypt_content(ROOT, data,
                                               master_key_provider=lambda p: key)).decode()


def _opens(db, rid, key):
    raw = base64.b64decode(db.artifacts.get_artifact(rid)["content"])
    try:
        return cc.decrypt_content(ROOT, raw, master_key_provider=lambda p: key)
    except Exception:  # noqa: BLE001 — the wrong key
        return None


@pytest.fixture()
def charged(tmp_path):
    db = la.LatticeDatabase(str(tmp_path / "k.db"), origin="k")
    kek = _FakeKek()
    db.artifacts.put_artifact({"id": ROOT, "content_type": "application/vnd.agience.collection+json",
                               "origin_root": ROOT, "root_id": ROOT})
    db.artifacts.put_artifact({"id": PART, "content_type": "application/vnd.agience.collection+json",
                               "origin_root": ROOT, "root_id": PART})     # its edge removed
    for rid, body in (("own", b"the root's own"), ("gone", b"the removed part's")):
        db.artifacts.put_artifact({"id": rid, "root_id": rid, "origin_root": ROOT,
                                   b.CONTENT_KEY_PRINCIPAL: ROOT, "created_by": "olga",
                                   "state": "live", "content_encrypted": True,
                                   "content": _mec1(OLD, body)})
    la.add_artifact_to_collection(db, ROOT, "own", origin=True)
    la.add_artifact_to_collection(db, PART, "gone", origin=True)
    o.LatticeMasterKeyStore(kek, lambda: db).put(ROOT, OLD)
    return db, kek, tmp_path


def test_the_root_s_leftovers_move_to_a_new_key_and_the_removed_part_is_shredded(charged):
    db, kek, tmp_path = charged
    old_token = db.artifacts.get_artifact("master-key:" + ROOT)["token"]
    report = rk.rotate_charged_roots(db, kek, [ROOT])
    assert report[ROOT]["rotated"] == 1 and report[ROOT]["shredded"] == 1
    new = o.LatticeMasterKeyStore(kek, lambda: db).get(ROOT)
    assert new is not None and new != OLD
    assert _opens(db, "own", new) == b"the root's own"
    assert _opens(db, "gone", new) is None                    # only the old key opened it
    raw = b"".join(p.read_bytes() for p in tmp_path.glob("k.db*") if p.name in
                   ("k.db", "k.db-wal"))
    assert old_token.encode() not in raw                      # the old key left the file


def test_a_root_with_nothing_left_has_its_key_removed(charged):
    db, kek, _ = charged
    db.artifacts.delete_artifact("own")
    report = rk.rotate_charged_roots(db, kek, [ROOT])
    assert report[ROOT]["removed"] is True and report[ROOT]["rotated"] == 0
    assert db.artifacts.get_artifact("master-key:" + ROOT) is None


def test_a_split_key_is_not_rotated_here(charged):
    db, kek, _ = charged
    doc = db.artifacts.get_artifact("master-key:" + ROOT)
    db.artifacts.put_artifact({"id": doc["id"], "content_type": o.SHARE_CONTENT_TYPE,
                               "state": "active", "share": "x", "kcv": "k"})
    report = rk.rotate_charged_roots(db, kek, [ROOT])
    assert report[ROOT] == {"rotated": 0, "shredded": 0, "removed": False,
                            "left": "split: rotated only under lattice custody, before step 23"}


def test_the_client_reads_the_charged_roots_origin_names():
    import httpx

    from mantle.clients.origin_key_share_client import OriginShareCustodian

    def answer(request):
        return httpx.Response(200, json={"kinds": {ROOT: "root", "x": "root"},
                                         "charged": [ROOT]})

    c = OriginShareCustodian("http://origin.invalid",
                             client=httpx.Client(transport=httpx.MockTransport(answer)))
    c._token = lambda *a, **k: "t"
    assert c.charged_roots([ROOT, "x"]) == [ROOT]
