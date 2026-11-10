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


def test_a_second_run_sends_nothing_origin_co_signed(db):
    """Each receipt kept says Origin co-signed that row: an edge or grant with one is not stated
    again. A key root keeps no receipt; Origin answers it `unchanged`."""
    origin = FakeOrigin()
    ca.cosign(db, origin, "bootstrap-token")
    origin.calls.clear()
    counts = ca.cosign(db, origin, "bootstrap-token")
    assert [k for k, _, _ in origin.calls] == ["root"]
    assert counts["held"] == 4 and counts["appended"] == 0 and counts["unchanged"] == 1


def test_the_pending_ids_are_what_origin_filters_and_only_those_are_stated(db):
    """Once Origin has ordinary entries it accepts a bootstrap only for ids it has not seen and
    the window lists: the run states only entries all of whose ids it was given."""
    assert ca.pending_ids(db) == ["C", "W", "card", "g1", "g2"]
    origin = FakeOrigin()
    counts = ca.cosign(db, origin, "b", only={"W", "C", "g1"})
    sent = sorted(k for k, _, _ in origin.calls)
    assert sent == ["edge.add", "grant", "root"]             # W->C, g1 on W, W's root
    assert counts["left_out"] == 2                           # C->card (card is not given), g2
    assert counts["left_out_by_kind"] == {"edges": 1, "roots": 0, "grants": 1}
    assert counts["left_out_grantees"] == {"bob": 1}       # g2 is bob's


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


def test_a_malformed_row_is_counted_by_reason_and_the_run_goes_on(db):
    """2026-11-09, the RC's seed rehearsal: production holds a grant with no grantee, which Origin
    refuses as malformed (400). It reaches no one; the run counts it and states the rest. Any other
    refusal still stops the run."""
    from mantle.clients.origin_key_share_client import AuthorityRefused

    class Picky(FakeOrigin):
        def submit_entry(self, entry, **kw):
            if entry.get("grant_id") == "g1":
                raise AuthorityRefused(400, "grantee_id must be a non-empty string of at most 128")
            return super().submit_entry(entry, **kw)

    origin = Picky()
    counts = ca.cosign(db, origin, "b")
    assert counts["malformed"] == 1 and counts["malformed_by_kind"]["grants"] == 1
    assert counts["malformed_by_reason"] == {"grantee_id must be a non-empty string of at most 128": 1}
    assert counts["appended"] == 4

    class Refusing(FakeOrigin):
        def submit_entry(self, entry, **kw):
            raise AuthorityRefused(403, "an id the ledger already holds")

    with pytest.raises(AuthorityRefused):
        ca.cosign(db, Refusing(), "b")
