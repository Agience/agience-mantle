"""`mantle.system.cosign_authority` co-signs the authority this node already holds at Origin, once,
on the operator's bootstrap credential, and keeps each receipt beside its row.

Origin's replica holds only co-signed entries, so what was granted before the ledger existed is
invisible to Origin's release check until this runs. It submits every containment edge, then every
key principal's root, then every grant, as bootstrap entries; a second run appends nothing. A
member with more than one creation edge (a move, before the store kept one) does not stop it: the
edge Mantle reads as the creation edge goes as one, the others as secondary containment.
"""
from __future__ import annotations

import json

import pytest

from mantle.attenuation import ACTIONS
from mantle.db import lattice_api as la
from mantle.db import reach_journal
from mantle.entities.grant import Grant
from mantle.system import cosign_authority as ca


class FakeOrigin:
    """Holds what it was sent; answers `unchanged` for an entry it already holds."""

    def __init__(self):
        self.held, self.calls = {}, []

    def submit_entry(self, entry, *, credential, bootstrap=False, proof=None, compensates=None):
        self.calls.append((entry["kind"], credential, bootstrap))
        key = json.dumps(entry, sort_keys=True)
        if key in self.held:
            return {**self.held[key], "outcome": "unchanged"}
        seq = len(self.held) + 1
        self.held[key] = {"seq": seq, "entry_hash": f"h{seq}", "signature": "s", "kid": "k"}
        return {**self.held[key], "outcome": "appended"}


@pytest.fixture
def db(tmp_path):
    d = la.LatticeDatabase(str(tmp_path / "c.db"), origin="c")
    la.add_artifact_to_collection(d, "W", "C", origin=True)
    la.add_artifact_to_collection(d, "C", "card", origin=True, propagate=["read"])
    la.add_artifact_to_collection(d, "W", "link", origin=False, relationship="cites")
    for gid, who, res in (("g1", "alice", "W"), ("g2", "bob", "C")):
        la.create_grant(d, Grant(id=gid, resource_id=res, grantee_type="user", grantee_id=who,
                                 granted_by="alice", name="named", **{f"can_{a}": a == "read"
                                                                       for a in ACTIONS}))
    d.artifacts.put_artifact({"id": "master-key:W", "content_type":
                              "application/vnd.agience.master-key+json", "token": "t"})
    return d


def _receipt(db, key):
    return reach_journal.receipt_of_row(db.graph.db.read(), key)


def test_a_dry_run_plans_edges_roots_and_grants_and_sends_nothing(db):
    p = ca.plan(db)
    assert sorted((e["src"], e["dst"], e["propagate"]) for e in p["edges"]) == [
        ("C", "card", ["read"]), ("W", "C", None)]                 # the link is not containment
    assert p["roots"] == [{"kind": "root", "principal_id": "W", "root_id": "W"}]
    assert sorted(e["grant_id"] for e in p["grants"]) == ["g1", "g2"]
    assert all("name" not in e for e in p["grants"]) and p["secondary"] == 0


def test_it_submits_in_order_as_bootstrap_on_the_bootstrap_credential(db):
    origin = FakeOrigin()
    counts = ca.cosign(db, origin, "bootstrap-token")
    kinds = [k for k, _, _ in origin.calls]
    assert kinds == ["edge.add", "edge.add", "root", "grant", "grant"]
    assert all(c == "bootstrap-token" and b is True for _, c, b in origin.calls)
    assert counts["appended"] == 5 and counts["unchanged"] == 0


def test_receipts_are_kept_beside_the_rows(db):
    ca.cosign(db, FakeOrigin(), "bootstrap-token")
    assert _receipt(db, "grant:g1")["entry_hash"]
    assert _receipt(db, "edge:C card")["entry_hash"]
    edge = next(e for e in db.graph.edges_of("card", direction="in") if e["src"] == "C")
    assert la._prop_mask(edge) == ["read"]                         # the row is untouched


def test_a_second_run_appends_nothing(db):
    origin = FakeOrigin()
    ca.cosign(db, origin, "bootstrap-token")
    counts = ca.cosign(db, origin, "bootstrap-token")
    assert counts["appended"] == 0 and counts["unchanged"] == 5


def test_a_member_with_two_creation_edges_goes_with_one(db):
    """A move written before the store kept one creation edge per member."""
    db.graph.add_edges([("W", "card", "contains", {"is_origin": 1})])
    p = ca.plan(db)
    card = [e for e in p["edges"] if e["dst"] == "card"]
    chosen = la.get_origin_parent(db, "card")[0]          # what Mantle reads as its creation
    other = ({"C", "W"} - {chosen}).pop()
    assert sorted((e["src"], e["origin"]) for e in card) == sorted([(chosen, True),
                                                                     (other, False)])
    assert p["secondary"] == 1
    assert ca.cosign(db, FakeOrigin(), "b")["secondary"] == 1


def test_the_cli_needs_the_bootstrap_credential_unless_it_is_a_dry_run():
    with pytest.raises(SystemExit):
        ca.main([])
