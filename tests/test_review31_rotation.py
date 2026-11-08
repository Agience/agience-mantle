"""Review 31: step 21's rotation of a charged root's key, and the split that follows it.

M1. A write to a kept row between the rotation's read and its transaction is not overwritten: each
    row's `_seq`/`_origin` is checked inside the write, and the root is left on any change.
L-M. One row that will not move leaves the other roots alone: a root that fails is counted and
    left, and the run goes on. A drop (MED1) is under no master key, so it is not counted as one.
H1. The new key row carries the rotation mark (the rotated key's kcv) in the same transaction;
    the split refuses a charged root whose key carries none; and step 21 exits non-zero when it
    has no kinds from Origin while key rows exist, when the feed is not at Origin's head, when
    Mantle is not stated stopped, or when any charged root is left.
I1. Step 21 runs the keep/shred test over every root, so a removal from before the ledger is
    caught though Origin names the root as no charged one.
"""
from __future__ import annotations

import base64

import httpx

from mantle.db import doc_boundary as b
from mantle.db import lattice_api as la
from mantle.search.mantle import oracle as o
from mantle.search.mantle import threshold_custody as tc
from mantle.services import content_crypto as cc
from mantle.system import rekey_collections as rk
from mantle.system import split_master_keys as smk

from . import test_review30_a_charged_root_s_key_is_rotated as r30

OLD, ROOT, _mec1, _opens = r30.OLD, r30.ROOT, r30._mec1, r30._opens
charged = r30.charged                     # the fixture review 30's own test uses


def _store(kek, db):
    return o.LatticeMasterKeyStore(kek, lambda: db)


def test_a_write_during_the_rotation_is_not_overwritten(charged, monkeypatch):
    db, kek, _ = charged
    real = cc.encrypt_content
    done = {"n": 0}

    def racing(principal, plain, **kw):
        if done["n"] == 0:
            done["n"] = 1
            # Another writer (a server still holding the old key) updates "own" meanwhile.
            doc = {k: v for k, v in db.artifacts.get_artifact("own").items()
                   if not k.startswith("_")}
            doc["content"] = _mec1(OLD, b"an edit made meanwhile")
            db.artifacts.put_artifact(doc, snapshot=False)
        return real(principal, plain, **kw)

    monkeypatch.setattr(cc, "encrypt_content", racing)
    report = rk.rotate_charged_roots(db, kek, [ROOT])
    assert report[ROOT]["left"].startswith("changed")
    assert _store(kek, db).get(ROOT) == OLD                # left as it was, for a re-run
    assert _opens(db, "own", OLD) == b"an edit made meanwhile"
    monkeypatch.setattr(cc, "encrypt_content", real)
    again = rk.rotate_charged_roots(db, kek, [ROOT])       # the re-run takes it
    assert again[ROOT]["rotated"] == 1 and "left" not in again[ROOT]
    assert _opens(db, "own", _store(kek, db).get(ROOT)) == b"an edit made meanwhile"


def _first_root(db, kek, content):
    first = "a-first"
    db.artifacts.put_artifact({"id": first, "content_type":
                               "application/vnd.agience.collection+json",
                               "origin_root": first, "root_id": first})
    db.artifacts.put_artifact({"id": "row-1", "root_id": "row-1", "origin_root": first,
                               b.CONTENT_KEY_PRINCIPAL: first, "created_by": "olga",
                               "state": "live", "content_encrypted": True, "content": content})
    la.add_artifact_to_collection(db, first, "row-1", origin=True)
    _store(kek, db).put(first, bytes(32))
    return first


def test_a_drop_is_under_no_master_key(charged):
    db, kek, _ = charged
    first = _first_root(db, kek, base64.b64encode(b"MED1" + bytes(80)).decode())
    from mantle.services.owner_rekey import rows_under_master_key
    assert [d["id"] for d, _ in rows_under_master_key(db, first)] == []
    report = rk.rotate_charged_roots(db, kek, [first, ROOT])
    assert report[first]["removed"] is True                 # nothing of its own under its key
    assert db.artifacts.get_artifact("row-1")["content"]   # the drop itself is untouched
    assert _store(kek, db).get(ROOT) not in (None, OLD)


def test_one_row_that_will_not_move_leaves_the_other_roots_alone(charged):
    db, kek, _ = charged
    # A first charged root ("a-first", sorted before ROOT) whose row will not open.
    first = _first_root(db, kek, base64.b64encode(b"MEC1" + bytes(80)).decode())
    report = rk.rotate_charged_roots(db, kek, [first, ROOT])
    assert report[first]["left"].startswith("failed")
    assert _store(kek, db).get(first) == bytes(32)         # left as it was
    assert report[ROOT]["rotated"] == 1
    assert _store(kek, db).get(ROOT) not in (None, OLD)


def test_the_new_key_row_carries_the_rotation_mark(charged):
    db, kek, _ = charged
    report = rk.rotate_charged_roots(db, kek, [ROOT])
    new = _store(kek, db).get(ROOT)
    row = db.artifacts.get_artifact("master-key:" + ROOT)
    assert row["rotated"]["kcv"] == tc.key_check_value(ROOT, new) == report[ROOT]["kcv"]
    assert rk.rotation_mark(row, ROOT, new)
    assert not rk.rotation_mark(row, ROOT, OLD)


def test_the_split_refuses_a_charged_root_without_the_rotation_mark(tmp_path):
    from mantle.services.acting_principal import acting_as

    from .test_a_split_master_key_needs_two_custodians import make_env

    e = make_env(tmp_path)
    _store(e.kek, e.db).put(ROOT, OLD)
    answer = {"charged": {ROOT}}

    def charged_roots(ids):
        if answer["charged"] is None:
            raise RuntimeError("Origin did not answer")
        return [i for i in ids if i in answer["charged"]]

    e.origin.charged_roots = charged_roots

    def split():
        with acting_as("system", source="system", scope="platform.custody"):
            return smk.split_master_keys(e.db, e.kek, e.origin, e.recipient, e.escrow, e.lock)

    counts = split()
    assert counts["split"] == 0 and counts["refused_unrotated"] == 1
    assert e.db.artifacts.get_artifact("master-key:" + ROOT)["content_type"] == \
        o.LatticeMasterKeyStore.CONTENT_TYPE                # still whole, still the old key
    # No answer from Origin about which roots are charged: no master key is split.
    answer["charged"] = None
    assert split()["split"] == 0
    # A mark for another key is no mark for this one.
    answer["charged"] = {ROOT}
    e.db.artifacts.put_artifact(rk.rotated_key_row(e.kek, ROOT, OLD,
                                                   kcv=tc.key_check_value(ROOT, bytes(32))))
    assert split()["refused_unrotated"] == 1
    # The rotated key, marked: split.
    e.db.artifacts.put_artifact(rk.rotated_key_row(e.kek, ROOT, bytes(range(5, 37))))
    counts = split()
    assert counts["split"] == 1 and counts["refused_unrotated"] == 0


# -- step 21's exit, through Mantle's real Origin client over a mock transport -------------------

def _client(kinds, charged, *, head=0, refuse_kinds=False):
    from mantle.clients.origin_key_share_client import OriginShareCustodian

    def answer(request):
        if request.url.path.endswith("/principal-kinds"):
            if refuse_kinds:
                return httpx.Response(503, json={"detail": "down"})
            return httpx.Response(200, json={"kinds": kinds, "charged": charged})
        if request.url.path.endswith("/feed"):
            return httpx.Response(200, json={"entries": [], "head": head})
        return httpx.Response(404)

    c = OriginShareCustodian("http://origin.invalid",
                             client=httpx.Client(transport=httpx.MockTransport(answer)))
    c._token = lambda *a, **k: "t"
    return c


def test_step_21_fails_with_no_kinds_while_key_rows_exist(charged):
    db, kek, _ = charged
    out, failed = rk.step21(db, kek, _client({}, [], refuse_kinds=True), mantle_stopped=True)
    assert failed and "no_kinds" in out["refused"]
    assert _store(kek, db).get(ROOT) == OLD                # nothing rotated on no answer


def test_step_21_fails_until_the_feed_is_at_origin_s_head(charged):
    db, kek, _ = charged
    out, failed = rk.step21(db, kek, _client({ROOT: "root"}, [ROOT], head=7),
                            mantle_stopped=True)
    assert failed and "feed_behind" in out["refused"]
    assert _store(kek, db).get(ROOT) == OLD


def test_step_21_rotates_only_with_mantle_stopped(charged):
    db, kek, _ = charged
    out, failed = rk.step21(db, kek, _client({ROOT: "root"}, [ROOT]), mantle_stopped=False)
    assert failed and "mantle_running" in out["refused"]
    assert _store(kek, db).get(ROOT) == OLD
    out, failed = rk.step21(db, kek, _client({ROOT: "root"}, [ROOT]), mantle_stopped=True)
    assert not failed, out
    assert out["charged_roots"][ROOT]["kcv"] == tc.key_check_value(ROOT, _store(kek, db).get(ROOT))


def test_step_21_fails_when_a_charged_root_is_left(charged, monkeypatch):
    db, kek, _ = charged
    row = db.artifacts.get_artifact("own")
    db.artifacts.put_artifact({**{k: v for k, v in row.items() if not k.startswith("_")},
                               "content_ref": "cas/x"}, snapshot=False)     # a content-tier copy
    out, failed = rk.step21(db, kek, _client({ROOT: "root"}, [ROOT]), mantle_stopped=True)
    assert failed and "left" in out["charged_roots"][ROOT]


def test_step_21_rotates_a_root_with_a_removal_from_before_the_ledger(charged):
    """Origin names ROOT as no charged root (its removal predates the ledger), but a row under its
    key is no longer under it: Mantle's own keep/shred test rotates it."""
    db, kek, _ = charged
    out, failed = rk.step21(db, kek, _client({ROOT: "root"}, []), mantle_stopped=True)
    assert not failed, out
    assert out["charged_roots"][ROOT]["shredded"] == 1
    assert _store(kek, db).get(ROOT) not in (None, OLD)
