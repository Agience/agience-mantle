"""Review 17, M1: the Add seal is bound to its collection key. A reader's open of the collection
writes it as `add-seal:<collection>`: the public half and a MAC over the collection and that half,
under a key derived from the collection key. An Add-only write seals only to a seal it verifies
against the collection key this node holds (in its key cache after a reader's open, or in a store
that keeps keys locally). A row it cannot verify, one planted with a single store write included,
counts as no seal, and the write is refused until a reader opens the collection (John,
2026-11-03). An id the operator retires loses its seal.

What the MAC does not cover: a Mantle that holds a reader's token during a compromise can open the
collection key and mint a bound seal of its own; the next reader's open repairs it.
"""
# ruff: noqa: F811
from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from mantle.search.mantle import oracle as o
from mantle.services import content_crypto as cc
from mantle.services.acting_principal import KeyCustodyDenied, acting_as

from .test_add_does_not_carry_read import OWNER, _inbox, _send, node  # noqa: F401


def _plant(db, collection_id, *, row="add-seal", mac=None):
    mine = X25519PrivateKey.generate()
    pub = mine.public_key().public_bytes(serialization.Encoding.Raw,
                                         serialization.PublicFormat.Raw)
    ctx = {"public": base64.b64encode(pub).decode("ascii"), "collection_id": collection_id}
    if mac is not None:
        ctx["mac"] = mac
    db.artifacts.put_artifact({"id": f"{row}:{collection_id}", "state": "live",
                               "content_type": o.OracleService.ADD_SEAL_CONTENT_TYPE,
                               "context": ctx})
    return pub


@pytest.mark.parametrize("row,mac", [("add-seal", None), ("add-seal", "AAAA"),
                                     ("drop-key", None)])
def test_a_planted_seal_takes_no_add_only_write(node, row, mac):
    db, oracle = node
    inbox = _inbox(db)
    db.artifacts.delete_artifact(f"add-seal:{inbox}")          # no reader's seal yet
    pub = _plant(db, inbox, row=row, mac=mac)
    oracle.evict()
    with pytest.raises(KeyCustodyDenied):
        _send(db, inbox, "the board minutes")
    assert oracle.add_seal_public(inbox) != pub


def test_a_reader_s_open_writes_a_bound_seal_that_the_writer_seals_to(node):
    db, oracle = node
    inbox = _inbox(db)
    oracle.evict()
    with acting_as(OWNER, principal_type="user"):
        ck = oracle.get_or_create_collection_key(inbox, inbox, o.KeyRequest(
            requester_id=OWNER, purpose=o.KeyPurpose.GRANT, action="read"))
    doc = db.artifacts.get_artifact(f"add-seal:{inbox}")
    assert doc and doc["context"]["mac"]
    assert oracle.add_seal_public(inbox) == cc.drop_public_key(ck)
    sent = _send(db, inbox, "a note")
    assert base64.b64decode(db.artifacts.get_artifact(sent.id)["content"])[:4] == b"MED1"


def test_a_planted_seal_is_repaired_by_the_next_reader(node, caplog):
    db, oracle = node
    inbox = _inbox(db)
    _plant(db, inbox, mac="AAAA")
    oracle.evict()
    with acting_as(OWNER, principal_type="user"):
        ck = oracle.get_or_create_collection_key(inbox, inbox, o.KeyRequest(
            requester_id=OWNER, purpose=o.KeyPurpose.GRANT, action="read"))
    assert any("ALARM" in r.getMessage() for r in caplog.records)
    assert oracle.add_seal_public(inbox) == cc.drop_public_key(ck)


def test_an_id_the_operator_retires_loses_its_seal(node):
    from mantle.services import authority_ledger as mantle_ledger
    db, oracle = node
    inbox = _inbox(db)
    assert db.artifacts.get_artifact(f"add-seal:{inbox}") is not None
    entry = {"seq": 1, "kind": "root.retire", "credential_kind": "operator",
             "body": {"kind": "root.retire", "principal_id": inbox}}
    assert mantle_ledger._apply_one(db, entry) == 1
    assert db.artifacts.get_artifact(f"add-seal:{inbox}") is None
    assert oracle.add_seal_public(inbox) is None


def test_the_sender_is_told_to_try_again_later(node):
    """John, 2026-11-03: refused until a reader opens it, with a clear "try again later"."""
    from fastapi.testclient import TestClient

    from mantle import main
    from mantle.services.acting_principal import AddSealPending

    db, oracle = node
    inbox = _inbox(db)
    db.artifacts.delete_artifact(f"add-seal:{inbox}")
    oracle.evict()
    with pytest.raises(AddSealPending, match="try again later"):
        _send(db, inbox, "too early")

    from fastapi import FastAPI
    app = FastAPI()
    app.add_exception_handler(AddSealPending, main.add_seal_pending)   # Mantle's own handler

    @app.get("/pending")
    def _pending():
        raise AddSealPending(f"the collection {inbox} has no Add seal yet: try again later")
    r = TestClient(app, raise_server_exceptions=False).get("/pending")
    assert r.status_code == 503 and "try again later" in r.text and r.headers.get("Retry-After")
