"""An Add seal is trusted on Mantle's own signature when no collection key is at hand (John,
2026-11-12). Under custody a collection key stays cached 30 s after a reader's open, and a timer
fires minutes or hours later: it must still seal its run to the collection. A reader's open writes
the seal with the MAC and a signature by Mantle's key; without the key, only the signature
decides. A seal planted by a store write alone carries no valid signature and takes no write.
"""
# ruff: noqa: F811
from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from mantle.search.mantle import oracle as o
from mantle.services import content_crypto as cc
from mantle.services import peer_signing
from mantle.services.acting_principal import acting_as

from .test_add_does_not_carry_read import OWNER, _inbox, node  # noqa: F401
from .test_review17_the_add_seal_is_bound_to_its_collection_key import _plant


@pytest.fixture
def mantle_key(tmp_path, monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode("ascii")
    monkeypatch.setattr(peer_signing, "_priv_pem", pem)
    return pem


def _open(oracle, inbox):
    with acting_as(OWNER, principal_type="user"):
        return oracle.get_or_create_collection_key(inbox, inbox, o.KeyRequest(
            requester_id=OWNER, purpose=o.KeyPurpose.GRANT, action="read"))


def test_a_reader_s_seal_is_trusted_on_its_signature_once_the_key_is_gone(node, mantle_key, monkeypatch):
    db, oracle = node
    inbox = _inbox(db)
    oracle.evict()
    ck = _open(oracle, inbox)
    assert db.artifacts.get_artifact(f"add-seal:{inbox}")["context"]["sig"]
    monkeypatch.setattr(oracle, "_key_for_seal", lambda cid: None)    # the cache has lapsed
    assert oracle.add_seal_public(inbox) == cc.drop_public_key(ck)


def test_a_planted_or_altered_seal_is_not_trusted_without_the_key(node, mantle_key, monkeypatch):
    db, oracle = node
    inbox = _inbox(db)
    oracle.evict()
    _open(oracle, inbox)
    signed = dict(db.artifacts.get_artifact(f"add-seal:{inbox}")["context"])
    monkeypatch.setattr(oracle, "_key_for_seal", lambda cid: None)
    db.artifacts.delete_artifact(f"add-seal:{inbox}")
    pub = _plant(db, inbox)                                          # no signature
    assert oracle.add_seal_public(inbox) is None
    ctx = db.artifacts.get_artifact(f"add-seal:{inbox}")["context"]
    ctx["sig"] = signed["sig"]                                      # a real signature, other half
    db.artifacts.put_artifact({"id": f"add-seal:{inbox}", "state": "live",
                               "content_type": o.OracleService.ADD_SEAL_CONTENT_TYPE, "context": ctx})
    assert oracle.add_seal_public(inbox) is None and pub
