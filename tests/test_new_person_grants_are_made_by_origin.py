"""Where Origin keeps the ledger, the grants every new person receives are Origin's to make:
Mantle names the person and applies Origin's own entries from the feed. Mantle links what "New
accounts receive" lists, and writes no grant to the person itself."""
from __future__ import annotations

import pytest

from mantle.db import lattice_api as la
from mantle.db import reach_journal
from mantle.services import authority_ledger
from mantle.services.seed_provisioning import platform_operators as P


class _Origin:
    def __init__(self, granted=2):
        self.asked, self.granted = [], granted

    def new_person_grants(self, person_id):
        self.asked.append(person_id)
        return {"granted": self.granted, "seqs": list(range(self.granted))}


@pytest.fixture
def db(tmp_path):
    return la.LatticeDatabase(str(tmp_path / "n.db"), origin="n")


@pytest.fixture
def origin(monkeypatch):
    o = _Origin()
    reach_journal.install(authority_ledger.Submitter(o, "record"))
    monkeypatch.setattr(P, "apply_remote", lambda db: 0)
    yield o
    reach_journal.uninstall()


def test_with_origin_s_ledger_mantle_asks_origin_and_writes_none_itself(db, origin):
    assert P.grant_to_account(db, "alice") == 2
    assert origin.asked == ["alice"]
    assert [d for d in la._grant_docs_by(db, "grantee_id", "alice")] == []


def _entry(state="active"):
    return {"credential_kind": "operator", "actor": "op-1", "body": {
        "kind": "grant", "grant_id": "npg-1", "grantee_id": "alice", "grantee_type": "user",
        "resource_id": "op-read-an-item", "actions": ["invoke", "read"], "effect": "allow",
        "state": state, "expires_at": None, "target": None, "max_claims": None}}


def test_an_operator_grant_origin_made_is_applied_from_the_feed(db):
    assert authority_ledger._apply_one(db, _entry()) == 1
    g = la.get_grant_by_id(db, "npg-1")
    assert g is not None and g.grantee_id == "alice" and g.can_read and g.can_invoke
    assert not g.can_admin and not g.can_share


def test_its_revocation_is_applied_too(db):
    authority_ledger._apply_one(db, _entry())
    assert authority_ledger._apply_one(db, _entry("revoked")) == 1
    assert la.get_grant_by_id(db, "npg-1").state == "revoked"


def test_new_accounts_receive_has_its_id_from_the_operator_alone():
    import uuid
    assert P.new_accounts_receive_id("op-1") == str(
        uuid.uuid5(uuid.NAMESPACE_URL, "agience://new-accounts-receive/op-1"))
    assert P.new_accounts_receive_id("op-1") != P.new_accounts_receive_id("op-2")
