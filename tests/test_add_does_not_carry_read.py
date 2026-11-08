"""CRUDEASIO: each action stands alone, and no action but Read releases a content key
(architecture §51; John, 2026-11-01).

A sender holding Add (intake) or Create on an inbox, and not Read, writes into it without ever
being given its collection key, in Mantle's process or at Origin: what lets them is their Add grant
(John, 2026-11-03: "It's not a drop key. It's a grant with Add rights and nothing else."), and what
they write is sealed to the inbox's public Add seal, which this store derives from the collection
key when a reader opens it. They cannot read their own drop afterwards. The owner reads it, which
re-keys it under the collection key and indexes it as the owner, and then finds it by search.
The other actions (Update, Delete, Evict, Invoke, Share, Admin) open no content key either.
Small fixtures: a temporary store and a fake KEK.
"""
from __future__ import annotations

import base64
import json

import pytest

from mantle.attenuation import ACTIONS
from mantle.db import lattice_api as la
from mantle.entities.grant import Grant
from mantle.search.mantle import oracle as o
from mantle.services import content_crypto as cc
from mantle.services.acting_principal import acting_as

from .test_a_split_master_key_needs_two_custodians import _FakeKek

OWNER, SAM = "owner-1", "sam-1"


@pytest.fixture
def node(tmp_path, monkeypatch):
    from mantle.search.mantle import wiring

    db = la.LatticeDatabase(str(tmp_path / "k.db"), origin="k")
    store = o.LatticeMasterKeyStore(_FakeKek(), lambda: db)
    oracle = o.OracleService(store, grant_verifier=o.LightConeGrantVerifier(db, ttl_s=0))
    monkeypatch.setattr(wiring, "_build_oracle", lambda: oracle)
    from mantle.services import workspace_service as ws
    # No content tier on this node: the body stays in the row, sealed there by the same
    # `encrypt_content` a content-tier copy goes through.
    monkeypatch.setattr(ws, "_store_content_in_s3",
                        lambda artifact_id, content, context_str, **_: (
                            f"artifacts/{artifact_id}.content", content, None))
    return db, oracle


def _inbox(db, actions=("add",)):
    from mantle.services import workspace_service as ws

    with acting_as(OWNER, principal_type="user"):
        inbox = ws.create_container(db, OWNER, content_type="application/vnd.agience.collection+json",
                                    name="Inbox").id
        la.create_grant(db, Grant(id="g-sam", resource_id=inbox, grantee_type="user",
                                  grantee_id=SAM, granted_by=OWNER,
                                  **{f"can_{a}": a in actions for a in ACTIONS}))
    return inbox


def _send(db, inbox, text="a note for the host"):
    from mantle.services import workspace_service as ws

    with acting_as(SAM, principal_type="user"):
        return ws.create_workspace_artifact(db, SAM, inbox, context=json.dumps({"title": "s"}),
                                            content=text)


@pytest.mark.parametrize("actions", [("add",), ("create",)])
def test_a_sender_without_read_writes_with_no_key_and_cannot_read_it_back(node, actions):
    db, oracle = node
    inbox = _inbox(db, actions)
    oracle.evict()                                            # a cold cache
    sent = _send(db, inbox)
    raw = db.artifacts.get_artifact(sent.id)
    assert raw.get("content_dropped") is True
    assert base64.b64decode(raw["content"])[:4] == b"MED1"          # sealed to the drop key
    with acting_as(SAM, principal_type="user"):
        with pytest.raises(o.GrantDenied):                    # Mantle's oracle gives no key
            oracle.get_or_create_collection_key(inbox, inbox, o.KeyRequest(
                requester_id=SAM, purpose=o.KeyPurpose.GRANT, action="read", may_create=True,
                creator_id=SAM))
        with pytest.raises(Exception):                        # nor can they open their drop
            la.get_artifact(db, sent.id)
    assert db.artifacts.get_artifact(sent.id).get("content_dropped") is True


def test_the_owner_reads_a_drop_which_re_keys_and_indexes_it_as_the_owner(node, monkeypatch):
    from mantle.search.ingest import pipeline_unified
    from mantle.search.mantle.sse.indexer import SseIndexer
    from mantle.search.mantle.sse.narrowing import TokenNarrower
    from mantle.search.mantle.sse.posting import InMemoryPostingStore

    db, oracle = node
    inbox = _inbox(db)
    sent = _send(db, inbox, "encryption library notes")
    postings = InMemoryPostingStore()
    indexed = []

    def index_as_reader(entity, collection_id, *, tenant_id=None, **kw):
        from mantle.services.acting_principal import current_acting_principal
        indexed.append(tenant_id)
        req = o.KeyRequest(requester_id=current_acting_principal().principal_id,
                           purpose=o.KeyPurpose.GRANT, action="read")
        SseIndexer(oracle, postings).index_artifact(inbox, collection_id, entity.root_id,
                                                    {"title": entity.content}, req)

    monkeypatch.setattr(pipeline_unified, "enqueue_index_artifact", index_as_reader)
    with acting_as(OWNER, principal_type="user"):
        read = la.get_artifact(db, sent.id)
    assert read.content == "encryption library notes"
    raw = db.artifacts.get_artifact(sent.id)
    assert not raw.get("content_dropped") and indexed == [OWNER]
    assert base64.b64decode(raw["content"])[:4] == b"MEC3"          # under the collection key
    with acting_as(OWNER, principal_type="user"):
        found = set(TokenNarrower(oracle, postings).lookup_for("encryption", o.KeyRequest(
            requester_id=OWNER, purpose=o.KeyPurpose.GRANT, action="read"))([(inbox, inbox)]))
    assert sent.root_id in found


@pytest.mark.parametrize("action", ["create", "update", "delete", "evict", "invoke", "add",
                                    "share", "admin"])
def test_no_action_but_read_releases_a_content_key(node, action):
    """Holding any one action other than Read opens no content key, whatever the request names."""
    db, oracle = node
    inbox = _inbox(db, (action,))
    oracle.evict()
    for asked in ("read", action):
        with acting_as(SAM, principal_type="user"):
            with pytest.raises(o.GrantDenied):
                oracle.get_or_create_collection_key(inbox, inbox, o.KeyRequest(
                    requester_id=SAM, purpose=o.KeyPurpose.GRANT, action=asked))


def test_a_drop_is_bound_to_its_collection_and_artifact():
    key = b"k" * 32
    pub = cc.drop_public_key(key)
    blob = cc.encrypt_drop("P", b"hello", collection_id="C", artifact_id="A", public_key=pub)
    assert blob[:4] == b"MED1"
    assert cc._decrypt_drop("P", blob, "C", "A", key) == b"hello"
    for wrong in (("P", "C2", "A"), ("P", "C", "A2"), ("Q", "C", "A")):
        with pytest.raises(Exception):
            cc._decrypt_drop(wrong[0], blob, wrong[1], wrong[2], key)
    assert base64.b64encode(pub)



def test_a_substituted_add_seal_is_repaired_on_the_next_key_open(node, caplog):
    """The cached Add seal is checked against the one the collection key derives on every Read
    open; a mismatch is repaired and reported, so a writer seals to the collection's readers."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    db, oracle = node
    inbox = _inbox(db)
    attacker = X25519PrivateKey.generate()
    pub = attacker.public_key().public_bytes(serialization.Encoding.Raw,
                                             serialization.PublicFormat.Raw)
    db.artifacts.put_artifact({"id": f"add-seal:{inbox}", "state": "live",
                               "content_type": o.OracleService.ADD_SEAL_CONTENT_TYPE,
                               "context": {"public": base64.b64encode(pub).decode(),
                                           "collection_id": inbox}})
    oracle.evict()
    with acting_as(OWNER, principal_type="user"):
        ck = oracle.get_or_create_collection_key(inbox, inbox, o.KeyRequest(
            requester_id=OWNER, purpose=o.KeyPurpose.GRANT, action="read"))
    assert oracle.add_seal_public(inbox) == cc.drop_public_key(ck)
    assert any("ALARM" in r.getMessage() for r in caplog.records)
    sent = _send(db, inbox, "secret from sam")
    with acting_as(OWNER, principal_type="user"):
        assert la.get_artifact(db, sent.id).content == "secret from sam"
