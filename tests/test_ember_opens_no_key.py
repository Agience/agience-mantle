"""Ember opens no person's key (John, 2026-11-02, review 10).

Under custody Ember does everything except open keys. On an Ember delegation, Mantle asks for no
content key, master or collection, whatever the person's grants: the person holds Read on their
own memory collection, and still nothing is opened for Ember. A memory Ember writes there is
sealed to the collection's drop key, written without Read, and indexed by no one until the person
reads it. The person's own token opens their own key as before.
"""
from __future__ import annotations

import json

import pytest

from mantle.services import content_crypto
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import KeyCustodyDenied, acting_as
from mantle.services.artifact_identity import derive_artifact_id
from mantle.services.dependencies import EMBER_ACTOR

from .test_there_is_no_promotion_state import USER, real_node

COLLECTION = "application/vnd.agience.collection+json"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def test_an_ember_delegation_asks_for_no_key(db, monkeypatch):
    asked = []
    from mantle.search.mantle import wiring
    real = wiring._build_oracle()
    monkeypatch.setattr(real, "get_or_create_collection_key",
                        lambda *a, **k: asked.append("ck") or b"k" * 32)
    monkeypatch.setattr(real, "get_or_create_master_key",
                        lambda *a, **k: asked.append("mk") or b"k" * 32)
    with acting_as(USER, principal_type="user", actor=EMBER_ACTOR, narrowed=EMBER_ACTOR, source="request"):
        with pytest.raises(KeyCustodyDenied, match="Ember"):
            content_crypto._default_collection_key(USER, "c-1")
        with pytest.raises(KeyCustodyDenied, match="Ember"):
            content_crypto._default_master_key(USER, "c-1")
    assert asked == []
    with acting_as(USER, principal_type="user", source="request"):
        content_crypto._default_collection_key(USER, "c-1")
    assert asked == ["ck"]


def test_a_memory_ember_writes_is_a_drop_and_opens_nothing(db, monkeypatch):
    from mantle.search.mantle import wiring
    oracle = wiring._build_oracle()
    mem_id = derive_artifact_id(USER, "ember:memory")
    with acting_as(USER, principal_type="user"):
        ws.create_container(db, USER, content_type=COLLECTION, name="Memory",
                            artifact_id=mem_id)
        oracle.get_or_create_collection_key(USER, mem_id, __import__(
            "mantle.search.mantle.oracle", fromlist=["KeyRequest"]).KeyRequest(
                requester_id=USER, purpose=__import__(
                    "mantle.search.mantle.oracle", fromlist=["KeyPurpose"]).KeyPurpose.GRANT,
                requester_type="user", action="read", may_create=True))
    opened = []
    real_ck = oracle.get_or_create_collection_key
    monkeypatch.setattr(oracle, "get_or_create_collection_key",
                        lambda *a, **k: opened.append(a) or real_ck(*a, **k))
    with acting_as(USER, principal_type="user", actor=EMBER_ACTOR, narrowed=EMBER_ACTOR, source="request"):
        made = ws.create_workspace_artifact(db, USER, mem_id, context=json.dumps({"via": "x"}),
                                            content="ships Friday", content_type="text/markdown")
    assert opened == []
    raw = db.artifacts.get_artifact(made.id)
    assert raw.get("content_dropped") is True
    assert "ships Friday" not in json.dumps(raw)
