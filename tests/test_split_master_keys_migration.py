"""`mantle.system.split_master_keys` splits every whole master key and can combine them back.

The key does not change, so whatever it encrypted still opens: each test reads the key back through
the store that will serve it and compares bytes. A split replaces the whole row only after its
shares, including Origin's stored copy, are shown to recombine; a key that does not read back is
restored whole. Each row is re-read under the custody lock before it is acted on, and left alone
if it changed since the pass first read it.
"""
from __future__ import annotations

import base64
import json

import pyrage
import pytest

from mantle.search.mantle import threshold_custody as tc
from mantle.search.mantle.oracle import SHARE_CONTENT_TYPE, LatticeMasterKeyStore
from mantle.system import split_master_keys as smk
from prism.trust import shamir

from .test_a_split_master_key_needs_two_custodians import make_env

KEYS = {"alice": bytes(range(32)), "bob": bytes(range(1, 33)), "carol": bytes(range(2, 34))}


@pytest.fixture()
def node(tmp_path):
    e = make_env(tmp_path)
    whole = LatticeMasterKeyStore(e.kek, lambda: e.db)
    for pid, key in KEYS.items():
        whole.put(pid, key)
    e.db.artifacts.put_artifact({"id": "not-a-key", "content_type": "text/plain"})
    return e


def _split(e, **kw):
    exported = []
    counts = smk.split_master_keys(e.db, e.kek, e.origin, e.recipient, e.escrow, e.lock,
                                   write_escrow=exported.extend, **kw)
    return counts, exported


def _combine(e, store=None):
    return smk.combine_master_keys(e.db, e.kek, store or e.store, e.lock)


def _whole_keys(e):
    whole = LatticeMasterKeyStore(e.kek, lambda: e.db)
    return {pid: whole.get(pid) for pid in KEYS}


def test_dry_run_changes_nothing(node):
    before = json.dumps(node.db.artifacts.docs, sort_keys=True)
    counts, exported = _split(node, dry_run=True)
    assert counts["whole"] == 3 and counts["split"] == 0
    assert json.dumps(node.db.artifacts.docs, sort_keys=True) == before
    assert not node.origin.shares and not exported and not node.escrow.objects


def test_every_key_is_split_and_reads_back_unchanged(node):
    counts, exported = _split(node)
    assert counts == {"whole": 3, "already_split": 0, "split": 3, "changed": 0, "failed": 0}
    for pid, key in KEYS.items():
        row = node.db.artifacts.docs["master-key:" + pid]
        assert row["content_type"] == SHARE_CONTENT_TYPE and "token" not in row
        assert "escrow" not in row
        assert node.store.get(pid) == key
        assert node.escrow.for_row(row) is not None
    assert node.db.artifacts.docs["not-a-key"]["content_type"] == "text/plain"
    assert sorted(e["principal_id"] for e in exported) == sorted(KEYS)
    for e in exported:
        o = node.origin.shares[e["principal_id"]]
        assert shamir.combine([(e["x"], base64.b64decode(e["y"])),
                               (o["x"], base64.b64decode(o["y"]))]) == KEYS[e["principal_id"]]


def test_a_second_run_splits_nothing(node):
    _split(node)
    counts, exported = _split(node)
    assert counts == {"whole": 0, "already_split": 3, "split": 0, "changed": 0, "failed": 0}
    assert exported == []


def test_origin_down_leaves_every_key_whole(node):
    node.origin.down = True
    counts, _ = _split(node)
    assert counts["split"] == 0 and counts["failed"] == 3
    assert _whole_keys(node) == KEYS


def test_a_durable_store_that_refuses_leaves_every_key_whole(node):
    node.escrow.down = True
    counts, _ = _split(node)
    assert counts["split"] == 0 and counts["failed"] == 3
    assert _whole_keys(node) == KEYS and not node.origin.shares


def test_origin_storing_a_wrong_share_leaves_the_key_whole(node):
    real_get = node.origin.get_share
    node.origin.get_share = lambda pid: (lambda s: s and {
        **s, "y": base64.b64encode(bytes(32)).decode()})(real_get(pid))
    counts, _ = _split(node)
    assert counts["split"] == 0 and counts["failed"] == 3
    assert _whole_keys(node)["alice"] == KEYS["alice"]


def test_a_row_changed_after_it_was_read_is_left_alone(node):
    """The pass reads every row, then exports, then writes. A row rewritten in between (here, by
    the export hook) is not replaced from the stale read."""
    def rewrite(entries):
        LatticeMasterKeyStore(node.kek, lambda: node.db).put("bob", KEYS["alice"])

    counts = smk.split_master_keys(node.db, node.kek, node.origin, node.recipient, node.escrow,
                                   node.lock, write_escrow=rewrite)
    assert counts["changed"] == 1 and counts["split"] == 2
    assert LatticeMasterKeyStore(node.kek, lambda: node.db).get("bob") == KEYS["alice"]


def test_the_split_waits_for_the_lock_the_server_holds(node):
    with tc.CustodyLock(node.lock.path)():
        with pytest.raises(Exception, match="custody lock"):
            _split(node)
    assert _whole_keys(node) == KEYS


def test_the_escrow_is_exported_before_any_row_is_replaced(node):
    seen = []

    def write(entries):
        seen.append([node.db.artifacts.docs["master-key:" + e["principal_id"]]["content_type"]
                     for e in entries])

    smk.split_master_keys(node.db, node.kek, node.origin, node.recipient, node.escrow, node.lock,
                          write_escrow=write)
    assert seen == [[LatticeMasterKeyStore.CONTENT_TYPE] * 3]


def test_the_sealed_export_opens_with_the_escrow_identity(node, tmp_path):
    path = tmp_path / "escrow.age"
    smk.split_master_keys(node.db, node.kek, node.origin, node.recipient, node.escrow, node.lock,
                          write_escrow=smk.escrow_writer(str(path), node.recipient))
    doc = json.loads(pyrage.decrypt(path.read_bytes(),
                                    [pyrage.x25519.Identity.from_str(node.identity)]))
    assert doc["format"] == "agience-mantle-escrow-v1"
    assert sorted(s["principal_id"] for s in doc["shares"]) == sorted(KEYS)
    with pytest.raises(FileExistsError):
        smk.escrow_writer(str(path), node.recipient)([])


def test_combine_writes_every_key_back_whole(node):
    _split(node)
    counts = _combine(node)
    assert counts["combined"] == 3 and counts["failed"] == 0
    assert _whole_keys(node) == KEYS
    counts, _ = _split(node)                                    # and splits again over Origin's
    assert counts["split"] == 3


def test_combine_leaves_a_key_it_cannot_combine_split(node):
    _split(node)
    node.origin.shares.pop("bob")
    counts = _combine(node)
    assert counts["combined"] == 2 and counts["failed"] == 1
    assert node.db.artifacts.docs["master-key:bob"]["content_type"] == SHARE_CONTENT_TYPE


def test_combine_in_recovery_mode_uses_the_escrow(node):
    _split(node)
    node.origin.shares.clear()
    assert _combine(node, node.make(recovery_identity=node.identity))["combined"] == 3
    assert _whole_keys(node) == KEYS


def test_combine_drops_a_pending_row_only_if_it_is_still_pending(node):
    row, *_ = tc.split_key("dave", bytes(32), node.kek, node.recipient, state="pending")
    node.db.artifacts.put_artifact(row)
    row2, *_ = tc.split_key("erin", bytes(32), node.kek, node.recipient, state="pending")
    node.db.artifacts.put_artifact(row2)
    real_list = node.db.artifacts.list_artifacts

    def list_then_activate(**kw):
        rows = real_list(**kw)
        node.db.artifacts.put_artifact({**row2, "state": "active"})   # a server finished its put
        return rows

    node.db.artifacts.list_artifacts = list_then_activate
    counts = _combine(node)
    assert counts["pending_dropped"] == 1 and "master-key:dave" not in node.db.artifacts.docs
    assert node.db.artifacts.docs["master-key:erin"]["state"] == "active"
    assert counts["changed"] == 1


def test_sealed_escrow_rows_lists_every_active_split_key(node):
    _split(node)
    entries = smk.sealed_escrow_rows(node.db, node.escrow)
    assert sorted(e["principal_id"] for e in entries) == sorted(KEYS)
    assert all(e["escrow"] and e["kcv"] for e in entries)


def test_a_real_split_needs_an_export_path_and_refuses_to_replace_one(tmp_path, monkeypatch):
    """Checked before any key is loaded: `main` exits on the arguments alone."""
    pub = tmp_path / "escrow.pub"
    pub.write_text(str(pyrage.x25519.Identity.generate().to_public()))
    monkeypatch.setattr("prism.trust.key_manager.init_encryption_key",
                        lambda: pytest.fail("keys loaded before the arguments were checked"))
    with pytest.raises(SystemExit):
        smk.main(["--recipient-file", str(pub)])
    existing = tmp_path / "old.age"
    existing.write_bytes(b"an earlier export")
    with pytest.raises(SystemExit):
        smk.main(["--recipient-file", str(pub), "--escrow-out", str(existing)])
    monkeypatch.delenv("MANTLE_ESCROW_RECIPIENT_FILE", raising=False)
    with pytest.raises(SystemExit):
        smk.main(["--escrow-out", str(tmp_path / "new.age"), "--recipient-file", ""])
