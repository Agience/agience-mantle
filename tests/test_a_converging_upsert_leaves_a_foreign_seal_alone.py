"""On the release candidate's copy, converging a grant whose name will not open changes its rights and
leaves the name sealed; anywhere else such a seal is a fault, and the write still fails on it.

The release candidate (`AGIENCE_PLANE=rc`) runs Mantle's migrations against a copy of production's
store under a key of its own: it holds no production key material. A grant's name, sealed under
production's key, reads as no name there, so `upsert_user_collection_grant` saw a difference that is
not there, set the name, and the write was refused: `platform_operators` stopped the release
candidate (2026-11-09). On production, or any node that should hold the key, a seal that will not
open is a corrupt row or a rotation gone wrong, never something to step around.
"""
from __future__ import annotations

import pytest
from cryptography.fernet import Fernet, InvalidToken

from mantle.db import lattice_api as la
from mantle.db import node_seal
from mantle.entities.grant import Grant

from .test_there_is_no_promotion_state import USER, real_node

THEIRS, OURS = Fernet.generate_key().decode(), Fernet.generate_key().decode()


@pytest.fixture
def copied(tmp_path, monkeypatch):
    """A store whose grant was written under THEIRS, now opened under OURS."""
    import prism.trust.key_manager as km
    monkeypatch.setattr(km, "_encryption_key", THEIRS)
    db = real_node(tmp_path, monkeypatch)
    la.create_grant(db, Grant(id="g-1", resource_id="c-1", grantee_type="user", grantee_id=USER,
                              granted_by=USER, can_read=True, name="Owner: Platform operators"))
    monkeypatch.setattr(km, "_encryption_key", OURS)
    return db


def _raw(db):
    return next(d for d in la._grant_docs(db, state="active") if d.get("id") == "g-1")


def _converge(db, **rights):
    return la.upsert_user_collection_grant(db, user_id=USER, collection_id="c-1", granted_by=USER,
                                           can_read=True, name="Owner: Platform operators", **rights)


def test_on_the_copy_converging_the_same_rights_writes_nothing(copied, monkeypatch):
    monkeypatch.setenv("AGIENCE_PLANE", "rc")
    sealed = _raw(copied)[node_seal.SEALED]
    grant, changed = _converge(copied)
    assert changed is False
    assert _raw(copied)[node_seal.SEALED] == sealed


def test_on_the_copy_new_rights_converge_and_the_name_stays_sealed(copied, monkeypatch):
    monkeypatch.setenv("AGIENCE_PLANE", "rc")
    sealed = _raw(copied)[node_seal.SEALED]
    grant, changed = _converge(copied, can_invoke=True)
    assert changed is True
    raw = _raw(copied)
    assert raw["can_invoke"] is True
    assert raw[node_seal.SEALED] == sealed                    # untouched: not ours to rewrite
    import prism.trust.key_manager as km
    monkeypatch.setattr(km, "_encryption_key", THEIRS)        # still opens where it was sealed
    assert la.get_grant_by_id(copied, "g-1").name == "Owner: Platform operators"


@pytest.mark.parametrize("plane", ["production", "home", None])
def test_anywhere_else_a_seal_that_will_not_open_fails_the_write(copied, monkeypatch, plane):
    """The own-node case: on production, a node at home, or one with no plane set, the key should
    open every seal in its store, so one that does not is a fault and the write fails on it."""
    if plane is None:
        monkeypatch.delenv("AGIENCE_PLANE", raising=False)
    else:
        monkeypatch.setenv("AGIENCE_PLANE", plane)
    sealed = _raw(copied)[node_seal.SEALED]
    with pytest.raises(InvalidToken):
        _converge(copied)
    assert _raw(copied)[node_seal.SEALED] == sealed           # and nothing was written


def test_on_its_own_node_a_new_name_is_still_written(copied, monkeypatch):
    import prism.trust.key_manager as km
    monkeypatch.setattr(km, "_encryption_key", THEIRS)
    monkeypatch.setenv("AGIENCE_PLANE", "production")
    grant, changed = la.upsert_user_collection_grant(
        copied, user_id=USER, collection_id="c-1", granted_by=USER, can_read=True,
        name="Renamed")
    assert changed is True
    assert la.get_grant_by_id(copied, "g-1").name == "Renamed"
