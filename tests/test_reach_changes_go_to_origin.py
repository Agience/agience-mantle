"""Every change to reach is sent to Origin to be co-signed in the write transaction that makes it,
at the store layer every writer goes through, and Origin's receipt is kept beside the row.

The store records grants (written or deleted), `contains` edges (added, changed, retired, and the
creation edge an artifact placed in a collection gets) as it writes them (`vertex._write_row`,
`vertex.delete_artifact`, `edge._add_one`, `edge._retire_edge_row`), and submits them just before
the transaction commits (`seq.LatticeConn.write` → `reach_journal.flush`). A refusal rolls the
transaction back. An entry co-signed for a transaction that then fails is withdrawn. Under
threshold custody a change with no ledger installed is refused at the store, whoever wrote it.
"""
from __future__ import annotations

import logging
import uuid

import httpx
import pytest

from mantle.attenuation import ACTIONS
from mantle.db import lattice_api as la
from mantle.db import reach_journal
from mantle.entities.grant import Grant
from mantle.services import authority_ledger
from mantle.services.acting_principal import ForwardedCredential, acting_as

RECEIPT = {"outcome": "appended", "entry_hash": "h" * 64, "prev_hash": "p" * 64,
           "signature": "sig", "kid": "origin-1"}


class FakeOrigin:
    """Origin's `/internal/authority/entries`: refuses what `refuse` says, else co-signs."""

    def __init__(self, refuse=None, down=False):
        self.entries, self.credentials, self.minted, self.proofs = [], [], [], []
        self.compensations, self.seeds_done, self.compensation_credentials = [], [], []
        self.batches, self.receipt_batches = [], []
        self.refuse, self.down = refuse, down
        self.head = None                 # the ledger head to report; None reports each seq
        self.seq = 0
        self.compensation_down = False

    def seed_done(self, credential):
        self.seeds_done.append(credential)

    def submit_entry(self, entry, *, credential, bootstrap=False, proof=None, compensates=None,
                     batch=None):
        from mantle.clients.origin_key_share_client import AuthorityRefused, CustodianUnavailable

        self.batches.append(batch)
        if batch is None and compensates is None:
            self.issued = getattr(self, "issued", 0) + 1
            batch = f"batch-{self.issued}"           # Origin issues the batch
        if self.down:
            raise CustodianUnavailable("Origin is unreachable (ConnectError)")
        if compensates is not None:
            if self.compensation_down:
                raise CustodianUnavailable("Origin is unreachable (ConnectError)")
            self.compensations.append((entry, compensates))
            self.compensation_credentials.append(credential)
            return {**RECEIPT, "seq": 1000 + compensates}
        if self.refuse and self.refuse(entry):
            raise AuthorityRefused(403, "the actor holds neither share nor admin on the resource")
        self.entries.append(entry)
        self.credentials.append(credential)
        self.proofs.append(proof)
        self.seq += 1
        self.receipt_batches.append(batch)
        return {**RECEIPT, "seq": self.seq, "batch": batch,
                "head_seq": self.seq if self.head is None else self.head}

    def background_delegation(self, principal_id, job=""):
        self.minted.append(principal_id)
        return f"background:{principal_id}"


@pytest.fixture
def db(tmp_path):
    return la.LatticeDatabase(str(tmp_path / "j.db"), origin="j")


@pytest.fixture
def person():
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        yield


def _install(origin, mode):
    reach_journal.install(authority_ledger.Submitter(origin, mode))
    return origin


def _grant(gid="g-1", resource="C1", **kw):
    actions = kw.pop("actions", ("read",))
    flags = {f"can_{a}": a in actions for a in ACTIONS}
    return Grant(id=gid, resource_id=resource, grantee_type=kw.pop("grantee_type", "user"),
                 grantee_id=kw.pop("grantee_id", "bob"), granted_by="alice", name="Bob's access",
                 notes="private", **flags, **kw)


def _edge(db, src, dst):
    return next((e for e in db.graph.edges_of(dst, direction="in") or []
                 if e.get("src") == src), None)


def _receipt(db, key):
    return reach_journal.receipt_of_row(db.graph.db.read(), key)


# -- what is journaled, and the receipt kept beside the row --------------------------------------

def test_a_grant_is_co_signed_in_its_write_and_keeps_the_receipt(db, person):
    origin = _install(FakeOrigin(), "enforce")
    la.create_grant(db, _grant())
    assert origin.entries == [{"kind": "grant", "grant_id": "g-1", "grantee_id": "bob",
                               "grantee_type": "user", "resource_id": "C1", "actions": ["read"],
                               "effect": "allow", "state": "active", "expires_at": None,
                               "target": None, "max_claims": None}]
    assert origin.credentials == ["alice-token"]
    assert _receipt(db, "grant:g-1")["seq"] == 1


def test_a_revocation_is_journaled_as_the_grant_s_new_state(db, person):
    origin = _install(FakeOrigin(), "enforce")
    g = _grant()
    la.create_grant(db, g)
    g.state = Grant.STATE_REVOKED
    la.update_grant(db, g)
    assert [e["state"] for e in origin.entries] == ["active", "revoked"]


def test_a_rewrite_that_changes_no_structure_sends_nothing(db, person):
    origin = _install(FakeOrigin(), "enforce")
    g = _grant()
    la.create_grant(db, g)
    g.claims_count = 3
    la.update_grant(db, g)
    assert len(origin.entries) == 1


def test_an_invite_carries_its_target_as_a_digest_and_its_limit(db, person):
    origin = _install(FakeOrigin(), "enforce")
    la.create_grant(db, _grant("inv-1", grantee_type="invite", grantee_id="token-hash",
                               target_entity="dan@team.test", target_entity_type="email",
                               max_claims=1))
    e = origin.entries[0]
    assert e["target"] == {"kind": "email",
                           "digest": reach_journal.target_digest("email", "dan@team.test")}
    assert e["max_claims"] == 1 and "dan@team.test" not in str(e)


def test_containment_edges_are_journaled_and_links_are_not(db, person):
    origin = _install(FakeOrigin(), "enforce")
    la.add_artifact_to_collection(db, "C1", "m-1", origin=True, propagate=["read", "update"])
    la.add_artifact_to_collection(db, "C1", "m-2", origin=False, relationship="cites")
    la.remove_artifact_from_collection(db, "C1", "m-1")
    assert origin.entries == [
        {"kind": "edge.add", "src": "C1", "dst": "m-1", "origin": True,
         "propagate": ["read", "update"]},
        {"kind": "edge.remove", "src": "C1", "dst": "m-1"}]


def test_an_edge_keeps_its_receipt_and_loses_it_when_removed(db, person):
    _install(FakeOrigin(), "enforce")
    la.add_artifact_to_collection(db, "C1", "m-1")
    assert _receipt(db, "edge:C1 m-1")["seq"] == 1
    la.remove_artifact_from_collection(db, "C1", "m-1")
    assert _receipt(db, "edge:C1 m-1") is None


def test_the_creation_edge_the_store_writes_is_journaled_once(db, person):
    """`vertex._place` writes the creation edge for an artifact naming a collection; a later put
    of the same artifact rewrites nothing and sends nothing."""
    origin = _install(FakeOrigin(), "enforce")
    db.artifacts.put_artifact({"id": "C1", "state": "live"})
    db.artifacts.put_artifact({"id": "card-1", "collection_id": "C1", "state": "live"})
    db.artifacts.put_artifact({"id": "card-1", "collection_id": "C1", "state": "live", "x": 1})
    assert [e for e in origin.entries if e["kind"] == "edge.add"] == [
        {"kind": "edge.add", "src": "C1", "dst": "card-1", "origin": True, "propagate": None}]


def test_a_move_is_secondary_containment_and_never_re_roots(db, person):
    origin = _install(FakeOrigin(), "enforce")
    for cid in ("C1", "C2"):
        db.artifacts.put_artifact({"id": cid, "state": "live"})
    db.artifacts.put_artifact({"id": "card-1", "collection_id": "C1", "state": "live"})
    db.artifacts.put_artifact({"id": "card-1", "collection_id": "C2", "state": "live"})
    assert la.get_origin_parent(db, "card-1")[0] == "C1"
    assert origin.entries[-1] == {"kind": "edge.add", "src": "C2", "dst": "card-1",
                                  "origin": False, "propagate": None}


def test_a_re_placement_does_not_reset_a_mask(db, person):
    _install(FakeOrigin(), "enforce")
    db.artifacts.put_artifact({"id": "C1", "state": "live"})
    la.add_artifact_to_collection(db, "C1", "card-1", origin=True, propagate=["read"])
    db.artifacts.put_artifact({"id": "card-1", "collection_id": "C1", "state": "live"})
    assert la._prop_mask(_edge(db, "C1", "card-1")) == ["read"]


def test_a_new_master_key_registers_its_root_first(person):
    from cryptography.fernet import Fernet

    from mantle.search.mantle.oracle import FernetMasterKeyStore, KeyPurpose, KeyRequest, OracleService

    origin = _install(FakeOrigin(), "enforce")
    oracle = OracleService(FernetMasterKeyStore(Fernet(Fernet.generate_key())),
                           grant_verifier=type("V", (), {"authorized": lambda s, **k: True})())
    oracle.get_or_create_master_key("alice", KeyRequest(requester_id="alice",
                                                        purpose=KeyPurpose.SELF, action="create"))
    assert origin.entries == [{"kind": "root", "principal_id": "alice", "root_id": "alice"}]
    # With it, the creation the store records: by whom, and when (Origin checks it is fresh).
    assert origin.proofs[0]["created_by"] == "alice" and origin.proofs[0]["created_at"]


def test_a_ledger_head_that_goes_backwards_is_refused(db, person):
    """Each receipt names Origin's ledger head; one lower than a head already seen means Origin's
    ledger was truncated or restored from an older copy, and the change is not written."""
    origin = _install(FakeOrigin(), "enforce")
    la.create_grant(db, _grant("g-1"))
    la.create_grant(db, _grant("g-2"))
    origin.head = 1                                             # Origin now says its head is 1
    with pytest.raises(reach_journal.ReachChangeRefused, match="went backwards"):
        la.create_grant(db, _grant("g-3"))
    assert la.get_grant_by_id(db, "g-3") is None
    with pytest.raises(reach_journal.ReachChangeRefused, match="went backwards"):
        reach_journal.submit(reach_journal.root_entry("p-9"))  # outside a transaction, too
    _install(origin, "enforce")                                 # a restart: the store remembers
    with pytest.raises(reach_journal.ReachChangeRefused, match="went backwards"):
        la.create_grant(db, _grant("g-4"))
    assert origin.compensations[-1][0]["grant_id"] == "g-4"    # appended, so withdrawn


def test_an_entry_refused_for_its_head_is_withdrawn(db, person):
    """Origin appended it, then its receipt named a lower head: the write is refused and the
    co-signed entry withdrawn, not left at Origin."""
    origin = _install(FakeOrigin(), "enforce")
    la.create_grant(db, _grant("g-1"))
    la.create_grant(db, _grant("g-2"))
    origin.head = 1
    with pytest.raises(reach_journal.ReachChangeRefused):
        la.create_grant(db, _grant("g-3"))
    assert origin.compensations and origin.compensations[-1][0]["grant_id"] == "g-3"


def test_shortening_a_deny_is_a_widening():
    deny = {"kind": "grant", "actions": ["read"], "effect": "deny", "state": "active",
            "expires_at": "2030-01-01T00:00:00Z"}
    assert reach_journal.grant_widens(deny, {**deny, "expires_at": "2029-01-01T00:00:00Z"})
    assert not reach_journal.grant_widens(deny, {**deny, "expires_at": "2031-01-01T00:00:00Z"})


def test_a_longer_expiry_is_a_widening_compared_as_time():
    """11:00 at +05:00 is 06:00 UTC: earlier than 10:00 UTC, though it sorts later as text."""
    def g(expires):
        return {"kind": "grant", "actions": ["read"], "effect": "allow", "state": "active",
                "expires_at": expires}

    assert not reach_journal.grant_widens(g("2030-01-01T10:00:00+00:00"),
                                          g("2030-01-01T11:00:00+05:00"))
    assert reach_journal.grant_widens(g("2030-01-01T11:00:00+05:00"),
                                      g("2030-01-01T10:00:00+00:00"))


def test_a_withdrawal_origin_does_not_take_is_kept_and_retried(db, person, monkeypatch):
    """Not only logged: recorded, retried before the next change and at startup as the system
    principal's narrowing, and under threshold custody nothing more is written until it clears."""
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    origin = _install(FakeOrigin(), "enforce")
    origin.compensation_down = True
    real = reach_journal.write_receipts

    def flush_then_fail(cur, done):
        real(cur, done)
        raise RuntimeError("disk full at commit")

    with monkeypatch.context() as m:
        m.setattr(reach_journal, "write_receipts", flush_then_fail)
        with pytest.raises(RuntimeError):
            db.graph.add_edges([("C1", "m-1", "contains", {"is_origin": 1})])
    conn = db.graph.db.read()
    assert reach_journal.due(conn) == [(1, {"kind": "edge.remove", "src": "C1", "dst": "m-1"})]
    origin.down = True                                          # retried, and still not taken
    with pytest.raises(reach_journal.ReachChangeRefused, match="outstanding"):
        la.create_grant(db, _grant("g-1"))
    origin.down = False
    origin.compensation_down = False
    assert authority_ledger.retry_outstanding(db) == 0          # at startup: as the system
    assert origin.compensations[-1] == ({"kind": "edge.remove", "src": "C1", "dst": "m-1"}, 1)
    assert origin.compensation_credentials[-1] == "background:sys-1"   # naming entry 1
    la.create_grant(db, _grant("g-1"))                          # cleared: writes go on


# -- deletes ---------------------------------------------------------------------------------------

def test_deleting_a_collection_journals_the_edges_and_grants_it_takes(db, person):
    origin = _install(FakeOrigin(), "enforce")
    db.artifacts.put_artifact({"id": "W", "state": "live"})
    db.artifacts.put_artifact({"id": "C1", "collection_id": "W", "state": "live"})
    la.add_artifact_to_collection(db, "C1", "m-1")
    la.create_grant(db, _grant("g-c", resource="C1"))
    origin.entries.clear()
    assert la.delete_collection(db, "C1")
    removed = sorted((e["src"], e["dst"]) for e in origin.entries if e["kind"] == "edge.remove")
    assert removed == [("C1", "m-1"), ("W", "C1")]
    db.artifacts.delete_artifact("g-c")
    assert origin.entries[-1]["kind"] == "grant" and origin.entries[-1]["state"] == "deleted"


def test_a_refused_delete_removes_nothing(db, person):
    origin = _install(FakeOrigin(), "enforce")
    db.artifacts.put_artifact({"id": "C1", "state": "live"})
    la.add_artifact_to_collection(db, "C1", "m-1")
    origin.refuse = lambda e: e["kind"] == "edge.remove"
    with pytest.raises(reach_journal.ReachChangeRefused):
        la.delete_collection(db, "C1")
    with pytest.raises(reach_journal.ReachChangeRefused):
        la.remove_all_edges_for_root(db, "m-1")             # not swallowed
    assert _edge(db, "C1", "m-1") is not None and db.artifacts.get_artifact("C1")


# -- enforce: fail closed, at the store ------------------------------------------------------------

def test_under_enforcement_a_refused_grant_is_not_written(db, person):
    _install(FakeOrigin(refuse=lambda e: True), "enforce")
    with pytest.raises(reach_journal.ReachChangeRefused, match="neither share nor admin"):
        la.create_grant(db, _grant())
    assert db.artifacts.get_artifact("g-1") is None


def test_under_enforcement_an_unreachable_origin_writes_nothing(db, person):
    _install(FakeOrigin(down=True), "enforce")
    with pytest.raises(reach_journal.ReachChangeRefused, match="unreachable"):
        la.add_artifact_to_collection(db, "C1", "m-1")
    assert _edge(db, "C1", "m-1") is None


def test_a_refusal_rolls_back_the_whole_transaction_and_withdraws_what_was_co_signed(db, person):
    """Two edges in one transaction: the first co-signed, the second refused. Neither is
    written, and the first is withdrawn at Origin."""
    origin = _install(FakeOrigin(refuse=lambda e: e.get("dst") == "m-2"), "enforce")
    with pytest.raises(reach_journal.ReachChangeRefused):
        db.graph.add_edges([("C1", "m-1", "contains", {"is_origin": 1}),
                            ("C1", "m-2", "contains", {"is_origin": 1})])
    assert _edge(db, "C1", "m-1") is None and _edge(db, "C1", "m-2") is None
    assert origin.compensations == [({"kind": "edge.remove", "src": "C1", "dst": "m-1"}, 1)]


def test_a_commit_that_fails_after_co_signing_withdraws_the_entry(db, person, monkeypatch):
    origin = _install(FakeOrigin(), "enforce")
    real = reach_journal.write_receipts

    def flush_then_fail(cur, done):
        real(cur, done)
        raise RuntimeError("disk full at commit")

    monkeypatch.setattr(reach_journal, "write_receipts", flush_then_fail)
    with pytest.raises(RuntimeError):
        db.graph.add_edges([("C1", "m-1", "contains", {"is_origin": 1})])
    monkeypatch.undo()
    assert _edge(db, "C1", "m-1") is None
    assert origin.compensations and origin.compensations[0][0]["kind"] == "edge.remove"


def test_under_threshold_custody_a_store_with_no_ledger_refuses(db, person, monkeypatch):
    """Whatever writes it: a CLI script, mesh consume, a direct store call."""
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    reach_journal.uninstall()
    with pytest.raises(reach_journal.ReachChangeRefused, match="none is installed"):
        db.graph.add_edges([("C1", "m-1", "contains", {"is_origin": 1})])
    with pytest.raises(reach_journal.ReachChangeRefused):
        la.create_grant(db, _grant())
    db.artifacts.put_artifact({"id": "plain", "state": "live"})          # no reach: written


def test_under_enforcement_a_grant_key_cannot_change_reach(db):
    _install(FakeOrigin(), "enforce")
    with acting_as("gk-1", principal_type="grant_key", source="request"):
        with pytest.raises(reach_journal.ReachChangeRefused, match="grant key"):
            la.create_grant(db, _grant())


def test_background_work_submits_on_a_delegation_minted_for_the_system_principal(db, monkeypatch):
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    origin = _install(FakeOrigin(), "enforce")
    with acting_as("sys-1", source="system", scope="platform.manage-seed"):
        la.create_grant(db, _grant())
    assert origin.minted == ["sys-1"] and origin.credentials == ["background:sys-1"]


def test_work_run_as_the_signed_in_person_keeps_their_credential(db):
    """Provisioning at first sign-in runs `acting_as(user)`: it acts on the person's own token,
    not as background work."""
    origin = _install(FakeOrigin(), "enforce")
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        with acting_as("alice", principal_type="user"):
            la.create_grant(db, _grant())
    assert origin.credentials == ["alice-token"] and origin.minted == []


def test_an_invite_claim_carries_its_token_as_proof(db, person):
    origin = _install(FakeOrigin(), "enforce")
    with reach_journal.proof(invite_id="inv-1", token="raw"):
        la.create_grant(db, _grant())
    assert origin.proofs == [{"invite_id": "inv-1", "token": "raw"}]


# -- record: written, not enforced ------------------------------------------------------------------

def test_recorded_a_refused_change_is_written_without_a_receipt_and_logged(db, person, caplog):
    _install(FakeOrigin(refuse=lambda e: True), "record")
    with caplog.at_level(logging.WARNING, logger="mantle.services.authority_ledger"):
        la.create_grant(db, _grant())
    assert db.artifacts.get_artifact("g-1") is not None and _receipt(db, "grant:g-1") is None
    assert "not co-signed" in caplog.text


def test_recorded_an_unreachable_origin_does_not_stop_a_write(db, person):
    _install(FakeOrigin(down=True), "record")
    assert la.add_artifact_to_collection(db, "C1", "m-1")
    assert _edge(db, "C1", "m-1") is not None


def test_with_no_authority_installed_nothing_is_journaled(db, person):
    la.create_grant(db, _grant())
    assert _receipt(db, "grant:g-1") is None


# -- the mode ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("custody,setting,expected", [
    ("threshold", "", "enforce"), ("threshold", "record", "enforce"),
    ("lattice", "", "record"), ("lattice", "record", "record"), ("lattice", "off", "off"),
])
def test_the_mode_follows_the_custody(monkeypatch, custody, setting, expected):
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", custody)
    monkeypatch.setenv(authority_ledger.MODE_SETTING, setting)
    assert authority_ledger.mode() == expected


def test_threshold_custody_refuses_to_turn_the_ledger_off(monkeypatch):
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    monkeypatch.setenv(authority_ledger.MODE_SETTING, "off")
    with pytest.raises(ValueError, match="refused under threshold"):
        authority_ledger.mode()


# -- the wire ---------------------------------------------------------------------------------------

def test_the_client_binds_its_token_to_the_entry_and_the_proof(monkeypatch):
    from mantle.clients import origin_key_share_client as oc

    signed, seen = [], {}
    monkeypatch.setattr(oc.peer_signing, "sign_service_jwt",
                        lambda **kw: signed.append(kw) or "svc")

    def handler(req):
        seen.update(req.headers)
        seen["path"] = req.url.path
        seen["body"] = req.content
        return httpx.Response(200, json={**RECEIPT, "seq": 7})

    c = oc.OriginShareCustodian("http://origin.test",
                                client=httpx.Client(transport=httpx.MockTransport(handler)))
    entry = reach_journal.unedge_entry("C1", "m-1")
    proof = {"invite_id": "i", "token": "t"}
    assert c.submit_entry(entry, credential="alice-token", proof=proof)["seq"] == 7
    claims = signed[0]["additional_claims"]
    assert claims["htm"] == "POST" and claims["htu"] == "/internal/authority/entries"
    assert claims["body_sha256"] == oc.entry_digest(entry) and "bootstrap" not in claims
    assert claims["proof_sha256"] == oc.entry_digest(proof)
    assert seen["x-acting-credential"] == "alice-token" and b'"proof"' in seen["body"]


def test_origin_s_refusal_reaches_the_caller_with_its_reason(monkeypatch):
    from mantle.clients import origin_key_share_client as oc

    monkeypatch.setattr(oc.peer_signing, "sign_service_jwt", lambda **kw: "svc")
    c = oc.OriginShareCustodian("http://origin.test", client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(403, json={"detail": "no admin"}))))
    with pytest.raises(oc.AuthorityRefused) as ei:
        c.submit_entry(reach_journal.root_entry("p"), credential="t")
    assert ei.value.status == 403 and ei.value.detail == "no admin"


def test_a_minted_delegation_is_reused_until_near_its_expiry(monkeypatch):
    import base64
    import json
    import time

    from mantle.clients import origin_key_share_client as oc

    monkeypatch.setattr(oc.peer_signing, "sign_service_jwt", lambda **kw: "svc")
    calls = []

    def tok(ttl):
        seg = base64.urlsafe_b64encode(json.dumps({"exp": time.time() + ttl}).encode()).decode()
        return f"h.{seg.rstrip('=')}.s"

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, json={"token": tok(300), "jti": uuid.uuid4().hex})

    c = oc.OriginShareCustodian("http://origin.test",
                                client=httpx.Client(transport=httpx.MockTransport(handler)))
    a = c.background_delegation("p-1", "platform.reindex")
    assert c.background_delegation("p-1", "platform.reindex") == a and len(calls) == 1
    c.background_delegation("p-2", "platform.reindex")
    assert calls == ["/internal/custody-delegations/p-1", "/internal/custody-delegations/p-2"]


def test_claim_invite_sends_the_claim_token_as_proof(monkeypatch):
    """`grant_service.claim_invite` writes the claimant's grant and the invite inside
    `reach_journal.proof`, so Origin is shown the token for both."""
    from types import SimpleNamespace

    from mantle.services import grant_service

    seen = []
    invite = SimpleNamespace(id="inv-1", resource_id="C1", target_entity=None,
                             target_entity_type=None, max_claims=None, claims_count=0,
                             granted_by="alice", name=None, expires_at=None, state="active",
                             **{f"can_{a}": a == "read" for a in ACTIONS})
    monkeypatch.setattr(grant_service, "_lookup_active_invite", lambda db, tok: invite)
    monkeypatch.setattr(grant_service, "create_grant",
                        lambda db, g: seen.append(reach_journal._proof.get()) or g)
    monkeypatch.setattr(grant_service, "update_grant",
                        lambda db, g: seen.append(reach_journal._proof.get()) or g)
    monkeypatch.setattr(grant_service, "_emit_invite_event", lambda *a, **k: None)
    grant_service.claim_invite(None, "carol", "raw-token")
    assert seen == [{"invite_id": "inv-1", "token": "raw-token"}] * 2


# -- operator-signed seeds -------------------------------------------------------------------------

def test_a_seed_pass_grants_on_the_operator_s_seed_credential(db, monkeypatch, tmp_path):
    """A seed scope runs on `MANTLE_SEED_CREDENTIAL_FILE`, the credential the operator minted at
    Origin for this pass; without it the pass is background work, which only narrows."""
    from mantle.services.system_identity import system_acting_context

    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    origin = _install(FakeOrigin(), "enforce")
    path = tmp_path / "seed.token"
    path.write_text("seed-token\n")
    monkeypatch.setenv("MANTLE_SEED_CREDENTIAL_FILE", str(path))
    with system_acting_context(scope="platform.operators-seed"):
        la.create_grant(db, _grant("g-seed"))
    assert origin.credentials == ["seed-token"] and origin.minted == []
    # One pass: Origin is told it is over (the credential is revoked) and the file is gone.
    assert origin.seeds_done == ["seed-token"] and not path.exists()
    with system_acting_context(scope="platform.operators-seed"):
        la.create_grant(db, _grant("g-bg"))
    assert origin.credentials[-1] == "background:sys-1"


def test_a_seed_pass_serves_its_scopes_and_ends_even_when_it_fails(db, monkeypatch, tmp_path):
    from mantle.services.system_identity import seed_pass, system_acting_context

    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    origin = _install(FakeOrigin(), "enforce")
    path = tmp_path / "seed.token"
    path.write_text("seed-token")
    monkeypatch.setenv("MANTLE_SEED_CREDENTIAL_FILE", str(path))
    with pytest.raises(RuntimeError):
        with seed_pass():
            for gid, scope in (("g-a", "platform.issuer-seed"), ("g-b", "platform.email-seed")):
                with system_acting_context(scope=scope):
                    la.create_grant(db, _grant(gid))
            raise RuntimeError("the pass failed")
    assert origin.credentials == ["seed-token", "seed-token"]
    assert origin.seeds_done == ["seed-token"] and not path.exists()


def test_only_a_seed_scope_takes_the_seed_credential(db, monkeypatch, tmp_path):
    from mantle.services.system_identity import system_acting_context

    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    origin = _install(FakeOrigin(), "enforce")
    path = tmp_path / "seed.token"
    path.write_text("seed-token")
    monkeypatch.setenv("MANTLE_SEED_CREDENTIAL_FILE", str(path))
    with system_acting_context(scope="platform.reindex"):
        la.create_grant(db, _grant("g-r"))
    assert origin.credentials == ["background:sys-1"]


# -- a peer's widening is held until a person here accepts it ----------------------------------------

@pytest.fixture
def threshold(monkeypatch):
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")


def _peer_edge(src, dst, seq=1, **props):
    return (src, dst, "contains", {"is_origin": 1, "_origin": "peer-node", "_seq": seq, **props})


def test_a_peer_s_widening_is_held_unwritten(db, threshold):
    from mantle.services import held_reach

    origin = _install(FakeOrigin(), "enforce")
    with acting_as("sys-1", source="system", scope="platform.manage-seed"):
        assert db.graph.add_edges([_peer_edge("C1", "m-1")], stamp_rev=False) == 1
    assert _edge(db, "C1", "m-1") is None and origin.entries == []
    held = held_reach.list_held(db)
    assert [(h["kind"], h["subject"], h["received_from"]) for h in held] == [
        ("edge", "edge:C1 m-1", "peer-node")]


def test_accepting_a_held_change_is_the_co_signed_change(db, threshold, person):
    from mantle.services import held_reach

    origin = _install(FakeOrigin(), "enforce")
    db.graph.add_edges([_peer_edge("C1", "m-1")], stamp_rev=False)
    held_reach.accept(db, held_reach.list_held(db)[0]["id"])
    assert _edge(db, "C1", "m-1") is not None and held_reach.list_held(db) == []
    assert origin.credentials == ["alice-token"]
    assert origin.entries == [{"kind": "edge.add", "src": "C1", "dst": "m-1", "origin": True,
                               "propagate": None}]


def test_a_refused_acceptance_stays_held(db, threshold, person):
    from mantle.services import held_reach

    _install(FakeOrigin(refuse=lambda e: True), "enforce")
    db.graph.add_edges([_peer_edge("C1", "m-1")], stamp_rev=False)
    with pytest.raises(reach_journal.ReachChangeRefused):
        held_reach.accept(db, held_reach.list_held(db)[0]["id"])
    assert _edge(db, "C1", "m-1") is None and len(held_reach.list_held(db)) == 1


def test_a_peer_s_widening_grant_is_held_and_its_narrowing_is_written(db, threshold, person):
    from mantle.services import held_reach

    origin = _install(FakeOrigin(), "enforce")
    la.create_grant(db, _grant("g-1"))
    raw = db.artifacts.get_artifact("g-1")
    wider = {**raw, "can_update": True, "_origin": "peer-node", "_seq": 99}
    db.artifacts.put_many([wider], stamp_rev=False, on_unordered="take_remote")
    assert db.artifacts.get_artifact("g-1").get("can_update") in (None, False)
    assert [h["kind"] for h in held_reach.list_held(db)] == ["grant"]
    narrower = {**raw, "state": "revoked", "_origin": "peer-node", "_seq": 100}
    db.artifacts.put_many([narrower], stamp_rev=False, on_unordered="take_remote")
    assert db.artifacts.get_artifact("g-1")["state"] == "revoked"
    assert origin.entries[-1]["state"] == "revoked"


def test_outside_threshold_custody_a_peer_s_change_is_written_as_before(db):
    from mantle.services import held_reach

    db.graph.add_edges([_peer_edge("C1", "m-1")], stamp_rev=False)
    assert _edge(db, "C1", "m-1") is not None and held_reach.list_held(db) == []


def test_a_person_who_manages_the_resource_lists_and_accepts_a_held_change(db, threshold, person):
    """`/held-reach`: listed only to whoever manages what it would widen; accepting it there is
    the co-signed write, on that person's credential."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mantle.routers.held_reach_router import router
    from mantle.services.dependencies import AuthContext, get_auth, get_store_db

    origin = _install(FakeOrigin(), "enforce")
    db.artifacts.put_artifact({"id": "C1", "state": "live"})
    la.create_grant(db, _grant("g-a", resource="C1", grantee_id="alice", actions=ACTIONS))
    db.graph.add_edges([_peer_edge("C1", "m-1")], stamp_rev=False)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_store_db] = lambda: db
    who = {"id": "carol"}
    app.dependency_overrides[get_auth] = lambda: AuthContext(
        principal_id=who["id"], principal_type="user", user_id=who["id"])
    client = TestClient(app)
    assert client.get("/held-reach").json() == {"items": []}           # carol manages nothing
    held_id = __import__("mantle.services.held_reach", fromlist=["x"]).list_held(db)[0]["id"]
    assert client.post(f"/held-reach/{held_id}/accept").status_code == 404
    who["id"] = "alice"
    items = client.get("/held-reach").json()["items"]
    assert [(i["kind"], i["resource_id"], i["member_id"]) for i in items] == [
        ("edge", "C1", "m-1")]
    r = client.post(f"/held-reach/{held_id}/accept")
    assert r.status_code == 200, r.text
    assert _edge(db, "C1", "m-1") is not None and client.get("/held-reach").json()["items"] == []
    assert origin.entries[-1]["kind"] == "edge.add" and origin.credentials[-1] == "alice-token"


# -- co-signed outside the write lock (John, 2026-11-01) --------------------------------------------

class OriginWithAWriterMeanwhile(FakeOrigin):
    """While Origin is asked, another writer on this node writes (`meanwhile`, in a thread). It
    must not wait for us: it runs to completion inside the call."""

    def __init__(self, db, meanwhile, times=1):
        import threading
        super().__init__()
        self.db, self.meanwhile, self.times, self.ran = db, meanwhile, times, 0
        self.asker = threading.get_ident()          # only our own asks have a writer meanwhile

    def submit_entry(self, entry, **kw):
        import threading
        if (kw.get("compensates") is None and self.ran < self.times
                and threading.get_ident() == self.asker):
            self.ran += 1
            t = threading.Thread(target=self.meanwhile, args=(self.ran,))
            t.start()
            t.join(timeout=10)
            assert not t.is_alive(), "an ordinary writer queued behind Origin"
        return super().submit_entry(entry, **kw)


def test_an_unrelated_write_while_origin_is_asked_neither_waits_nor_conflicts(db, person):
    db.artifacts.put_artifact({"id": "C1", "state": "live"})
    origin = _install(OriginWithAWriterMeanwhile(db, lambda n: db.artifacts.put_artifact(
        {"id": f"plain-{n}", "state": "live"})), "enforce")
    la.create_grant(db, _grant("g-1"))
    assert db.artifacts.get_artifact("plain-1") and la.get_grant_by_id(db, "g-1") is not None
    assert len(origin.entries) == 1 and origin.compensations == []
    assert reach_journal.receipt_of_row(db.graph.db.read(), "grant:g-1")["seq"] == 1


def test_a_conflicting_change_between_ask_and_commit_is_caught(db, person):
    """Someone changes the same grant while Origin co-signs ours: ours is withdrawn and made again
    after theirs, so the store and Origin's ledger end in the same state, in the same order."""
    db.artifacts.put_artifact({"id": "C1", "state": "live"})
    la.create_grant(db, _grant("g-1"))                                  # no ledger yet

    def change_the_same_grant(n):
        with acting_as("alice", principal_type="user", source="request",
                       credential=ForwardedCredential("alice-token")):
            g = la.get_grant_by_id(db, "g-1")
            g.can_update = True
            la.update_grant(db, g)

    origin = _install(OriginWithAWriterMeanwhile(db, change_the_same_grant), "enforce")
    g = la.get_grant_by_id(db, "g-1")
    g.can_share = True
    la.update_grant(db, g)
    final = la.get_grant_by_id(db, "g-1")
    assert final.can_share and not final.can_update                   # ours, written last
    assert [c[1] for c in origin.compensations] == [2]                # our first, withdrawn
    assert [e["actions"] for e in origin.entries] == [
        ["read", "update"], ["read", "share"], ["read", "share"]]      # theirs, ours, ours again
    assert reach_journal.receipt_of_row(db.graph.db.read(), "grant:g-1")["seq"] == 3


def test_a_change_that_keeps_conflicting_is_refused_with_nothing_written(db, person):
    from mantle.db import seq

    db.artifacts.put_artifact({"id": "C1", "state": "live"})
    la.create_grant(db, _grant("g-1"))

    def change_it(n):
        with acting_as("alice", principal_type="user", source="request",
                       credential=ForwardedCredential("alice-token")):
            g = la.get_grant_by_id(db, "g-1")
            g.notes = f"edit {n}"
            g.can_update = not g.can_update
            la.update_grant(db, g)

    origin = _install(OriginWithAWriterMeanwhile(db, change_it, times=99), "enforce")
    g = la.get_grant_by_id(db, "g-1")
    g.can_share = True
    with pytest.raises(seq.ReachChangeConflict):
        la.update_grant(db, g)
    assert not la.get_grant_by_id(db, "g-1").can_share
    assert len(origin.compensations) >= seq.COSIGN_ATTEMPTS


def test_a_write_block_is_replayed_only_if_nothing_else_committed(db, person):
    """A caller's own write block that changes reach: its statements are replayed after Origin
    co-signs, and only if no other connection committed meanwhile."""
    from mantle.db.seq import ReachChangeConflict

    db.artifacts.put_artifact({"id": "C1", "state": "live"})
    origin = _install(OriginWithAWriterMeanwhile(db, lambda n: db.artifacts.put_artifact(
        {"id": f"plain-{n}", "state": "live"})), "enforce")
    with pytest.raises(ReachChangeConflict):
        with db.graph.db.write():
            db.graph.add_edges([("C1", "m-1", "contains", {"is_origin": 1})])
    assert _edge(db, "C1", "m-1") is None and origin.compensations
    with db.graph.db.write():                                       # nothing meanwhile now
        db.graph.add_edges([("C1", "m-2", "contains", {"is_origin": 1})])
    assert _edge(db, "C1", "m-2") is not None
    assert reach_journal.receipt_of_row(db.graph.db.read(), "edge:C1 m-2") is not None


# -- third review: confirmation, same-connection writes, a 409 that is not ours ------------------

class ConfirmingOrigin(FakeOrigin):
    def __init__(self):
        super().__init__()
        self.confirmed, self.confirm_down = [], False

    def confirm_entries(self, seqs):
        from mantle.clients.origin_key_share_client import CustodianUnavailable
        if self.confirm_down:
            raise CustodianUnavailable("Origin is unreachable (ConnectError)")
        self.confirmed.extend(seqs)


def test_a_committed_change_is_confirmed_and_one_not_heard_is_told_again(db, person):
    origin = _install(ConfirmingOrigin(), "enforce")
    db.artifacts.put_artifact({"id": "C1", "state": "live"})
    la.create_grant(db, _grant("g-1"))
    assert origin.confirmed == [1]
    origin.confirm_down = True
    la.create_grant(db, _grant("g-2"))
    assert origin.confirmed == [1]
    assert [r[0] for r in db.graph.db.read().execute(
        "SELECT seq FROM authority_confirm_due").fetchall()] == [2]
    origin.confirm_down = False
    assert authority_ledger.retry_outstanding(db) == 0
    assert origin.confirmed == [1, 2]


def test_a_same_connection_write_while_origin_is_asked_is_caught(db, person):
    """A write on this very connection between the reservation and the commit moves the store's
    write mark, so a replayed block is refused rather than written over it."""
    from mantle.db.seq import ReachChangeConflict

    db.artifacts.put_artifact({"id": "C1", "state": "live"})

    class SameThread(FakeOrigin):
        def submit_entry(self, entry, **kw):
            if kw.get("compensates") is None and not self.entries:
                db.artifacts.put_artifact({"id": "same-thread", "state": "live"})
            return super().submit_entry(entry, **kw)

    origin = _install(SameThread(), "enforce")
    with pytest.raises(ReachChangeConflict):
        with db.graph.db.write():
            db.graph.add_edges([("C1", "m-1", "contains", {"is_origin": 1})])
    assert _edge(db, "C1", "m-1") is None and db.artifacts.get_artifact("same-thread")
    assert origin.compensations


class OriginAnswering409(FakeOrigin):
    """A withdrawal is answered 409, naming the row's latest entry."""

    def __init__(self, latest):
        super().__init__()
        self.latest, self.compensation_down = latest, False

    def submit_entry(self, entry, *, credential, bootstrap=False, proof=None, compensates=None):
        from mantle.clients.origin_key_share_client import AuthorityRefused
        if compensates is not None and self.latest is not None:
            raise AuthorityRefused(409, "the row has changed since; it is not withdrawn "
                                        f"(latest entry {self.latest})")
        return super().submit_entry(entry, credential=credential, compensates=compensates)


def test_a_withdrawal_answered_409_stays_due_unless_the_later_change_is_ours(db, person,
                                                                           monkeypatch):
    """A write Mantle rolled back leaves its entry at Origin, and Origin says the row has changed
    since (entry 7) to a widening Mantle never wrote. The withdrawal stays due, and under threshold
    custody no change is written. It clears only once entry 7 is a change this store committed."""
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    _install(OriginAnswering409(latest=7), "enforce")
    real = reach_journal.write_receipts

    def then_fail(cur, done):
        real(cur, done)
        raise RuntimeError("disk full at commit")

    with monkeypatch.context() as m:
        m.setattr(reach_journal, "write_receipts", then_fail)
        with pytest.raises(RuntimeError):
            db.graph.add_edges([("C1", "m-1", "contains", {"is_origin": 1})])
    conn = db.graph.db.read()
    assert len(reach_journal.due(conn)) == 1
    assert authority_ledger.retry_outstanding(db) == 1                 # still due
    with pytest.raises(reach_journal.ReachChangeRefused, match="outstanding"):
        la.create_grant(db, _grant("g-1"))
    with db.graph.db.write() as cur:                                    # entry 7 is ours
        cur.execute("INSERT OR REPLACE INTO authority_receipt(subject, kind, seq) "
                    "VALUES('edge:C1 m-1', 'edge.add', 7)")
    assert authority_ledger.retry_outstanding(db) == 0


def test_the_held_listing_asks_who_manages_each_resource_once(db, threshold, person, monkeypatch):
    from mantle.routers import held_reach_router as hr
    from mantle.services import grant_service
    from mantle.services.dependencies import AuthContext

    _install(FakeOrigin(), "enforce")
    db.artifacts.put_artifact({"id": "C1", "state": "live"})
    la.create_grant(db, _grant("g-a", resource="C1", grantee_id="alice", actions=ACTIONS))
    db.graph.add_edges([_peer_edge("C1", f"m-{i}", seq=i + 1) for i in range(5)],
                       stamp_rev=False)
    asked = []
    real = grant_service.can_admin
    monkeypatch.setattr(grant_service, "can_admin",
                        lambda d, u, r: asked.append(r) or real(d, u, r))
    held = hr._visible(db, AuthContext(principal_id="alice", principal_type="user",
                                       user_id="alice"))
    assert len(held) == 5 and asked == ["C1"]


# -- fourth review: memberships and identity --------------------------------------------------------

def test_a_membership_is_secondary_unless_the_root_has_no_creation_edge(db, person):
    origin = _install(FakeOrigin(), "enforce")
    for cid in ("C1", "C2"):
        db.artifacts.put_artifact({"id": cid, "state": "live"})
    db.artifacts.put_artifact({"id": "card-1", "collection_id": "C1", "state": "live"})
    origin.entries.clear()
    la.add_artifact_to_collection(db, "C2", "card-1")                # a link: secondary
    la.add_artifact_to_collection(db, "C1", "card-1", order_key="a5")  # re-placed: unchanged
    assert la.get_origin_parent(db, "card-1")[0] == "C1"
    assert [e for e in origin.entries if e["kind"] == "edge.add"] == [
        {"kind": "edge.add", "src": "C2", "dst": "card-1", "origin": False, "propagate": None}]


def test_an_identity_in_another_container_is_refused_and_writes_no_version(db, person):
    from fastapi import HTTPException
    from mantle.services import workspace_service as ws

    db.artifacts.put_artifact({"id": "A", "state": "live", "content_type":
                               "application/vnd.agience.collection+json"})
    db.artifacts.put_artifact({"id": "B", "state": "live", "content_type":
                               "application/vnd.agience.collection+json"})
    for cid in ("A", "B"):
        la.create_grant(db, _grant(f"g-{cid}", resource=cid, grantee_id="alice", actions=ACTIONS))
    db.artifacts.put_artifact({"id": "mirror-1", "root_id": "mirror-1", "collection_id": "A",
                               "state": "live"})
    before = len(db.artifacts.versions_of("mirror-1") or [])
    with pytest.raises(HTTPException) as e:
        ws.upsert_identity_member(db, "alice", "B", "mirror-1", content="x")
    assert e.value.status_code == 409
    assert len(db.artifacts.versions_of("mirror-1") or []) == before


# -- fourth review: a failed deny, an unchanged receipt, a resolved entry, one batch -------------

def _failing_commit(monkeypatch):
    real = reach_journal.write_receipts

    def then_fail(cur, done):
        real(cur, done)
        raise RuntimeError("database is locked at commit")
    return monkeypatch.context(), then_fail


def test_a_failed_deny_write_is_withdrawn_and_blocks_nothing(db, person, monkeypatch):
    """The deny of a write that did not commit was never in force in this store; Origin takes its
    withdrawal (it is unconfirmed), so nothing is left due and later changes go on."""
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    origin = _install(FakeOrigin(), "enforce")
    ctx, then_fail = _failing_commit(monkeypatch)
    with ctx as m:
        m.setattr(reach_journal, "write_receipts", then_fail)
        with pytest.raises(RuntimeError):
            la.create_grant(db, _grant("g-deny", effect="deny"))
    assert origin.compensations[-1][0]["effect"] == "deny"
    assert reach_journal.due(db.graph.db.read()) == []
    la.create_grant(db, _grant("g-next", resource="C9"))


def test_an_unchanged_receipt_names_no_entry_to_withdraw(db, person, monkeypatch):
    """Origin answers `unchanged` with the receipt of the older entry that set that state. A
    write that then fails withdraws nothing: that entry is not this write's."""
    class Unchanged(FakeOrigin):
        def submit_entry(self, entry, **kw):
            if kw.get("compensates") is None and self.entries:
                return {"outcome": "unchanged", "seq": 1, "entry_hash": "h" * 64,
                        "signature": "s", "kid": "k", "head_seq": self.seq}
            return super().submit_entry(entry, **kw)

    origin = _install(Unchanged(), "enforce")
    la.create_grant(db, _grant("g-x"))
    g = _grant("g-x")
    g.can_update = True
    ctx, then_fail = _failing_commit(monkeypatch)
    with ctx as m:
        m.setattr(reach_journal, "write_receipts", then_fail)
        with pytest.raises(RuntimeError):
            la.update_grant(db, g)
    assert origin.compensations == []


def test_a_withdrawal_the_operator_resolved_clears(db, person, monkeypatch):
    from mantle.clients.origin_key_share_client import AuthorityRefused

    class Resolved(FakeOrigin):
        def submit_entry(self, entry, **kw):
            if kw.get("compensates") is not None and self.resolved:
                raise AuthorityRefused(410, "the operator resolved entry 1")
            return super().submit_entry(entry, **kw)

    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    origin = _install(Resolved(), "enforce")
    origin.resolved, origin.compensation_down = False, True
    ctx, then_fail = _failing_commit(monkeypatch)
    with ctx as m:
        m.setattr(reach_journal, "write_receipts", then_fail)
        with pytest.raises(RuntimeError):
            db.graph.add_edges([("C1", "m-1", "contains", {"is_origin": 1})])
    assert len(reach_journal.due(db.graph.db.read())) == 1
    origin.compensation_down, origin.resolved = False, True
    assert authority_ledger.retry_outstanding(db) == 0


def test_a_container_s_root_and_its_creation_grant_go_in_one_batch(db, person):
    from mantle.services import workspace_service as ws

    origin = _install(FakeOrigin(), "enforce")
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        cid = ws.create_container(db, "alice", content_type="application/vnd.agience.collection+json",
                                  name="Notes").id
    pairs = [(e, b) for e, b in zip(origin.entries, origin.receipt_batches)
             if e["kind"] != "drop_key"]
    root = [b for e, b in pairs if e["kind"] == "root"]
    grant = [b for e, b in pairs if e["kind"] == "grant" and e["resource_id"] == cid]
    assert root and grant and root[0] == grant[0] and root[0]
    assert {"kind": "root", "principal_id": cid, "root_id": cid} in origin.entries


def test_a_seeded_platform_collection_comes_with_its_root_and_creation_grant(db, monkeypatch):
    from mantle.entities.collection import Collection
    from mantle.services.seed_provisioning import loader

    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    origin = _install(FakeOrigin(), "enforce")
    with acting_as("sys-1", source="system", scope="platform.manage-seed"):
        loader._create_platform_collection(db, Collection(id="plat-1", name="Platform",
                                                          created_by="sys-1",
                                                          content_type="application/vnd.agience.collection+json"))
    kinds = [e["kind"] for e in origin.entries]
    assert kinds[0] == "root" and "grant" in kinds
    assert len(set(origin.receipt_batches)) == 1 and origin.receipt_batches[0]   # one batch
    grant = next(e for e in origin.entries if e["kind"] == "grant")
    assert grant["grantee_id"] == "sys-1" and grant["resource_id"] == "plat-1"


def test_the_creators_memoized_grants_are_dropped_once_the_creation_grant_is_written(db, monkeypatch):
    # 71/test, 2026-11-11: the admin's grants were memoized empty at first sign-in, seconds before
    # the seed; every write into the new collections was then refused for the memo's 30 s.
    from mantle.entities.collection import Collection
    from mantle.search.mantle import wiring
    from mantle.services.seed_provisioning import loader

    dropped = []
    monkeypatch.setattr(wiring, "invalidate_grant_cache", lambda rid=None: dropped.append(rid))
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    _install(FakeOrigin(), "enforce")
    with acting_as("sys-1", source="system", scope="platform.manage-seed"):
        loader._create_platform_collection(db, Collection(id="plat-2", name="Platform",
                                                          created_by="sys-1",
                                                          content_type="application/vnd.agience.collection+json"))
    assert "sys-1" in dropped


# -- fifth review: the operator's changes at Origin reach this store -------------------------------

def test_an_operator_revocation_at_origin_is_applied_to_this_store(db, person, monkeypatch):
    # What is applied, given entries that verify (`verify_entry`'s own tests cover the check).
    monkeypatch.setattr(authority_ledger, "verify_entry", lambda e, prev, keys=None: "h" * 64)
    origin = _install(FakeOrigin(), "enforce")
    db.artifacts.put_artifact({"id": "C1", "state": "live"})
    la.create_grant(db, _grant("g-1"))
    db.graph.add_edges([("C1", "m-1", "contains", {"is_origin": 1})])
    feed = {"entries": [
        {"seq": 7, "kind": "grant", "credential_kind": "operator", "actor": "op",
         "body": {"kind": "grant", "grant_id": "g-1", "state": "revoked"}},
        {"seq": 8, "kind": "edge.remove", "credential_kind": "operator", "actor": "op",
         "body": {"kind": "edge.remove", "src": "C1", "dst": "m-1"}},
        {"seq": 9, "kind": "grant", "credential_kind": "access", "actor": "alice",
         "body": {"kind": "grant", "grant_id": "g-1", "state": "active"}}], "head": 9}
    asked = []
    origin.ledger_feed = lambda after: asked.append(after) or feed
    sent = len(origin.entries)
    assert authority_ledger.apply_remote(db, origin) == 2
    assert la.get_grant_by_id(db, "g-1").state == "revoked" and _edge(db, "C1", "m-1") is None
    assert len(origin.entries) == sent                       # not submitted back
    feed["entries"] = []
    authority_ledger.apply_remote(db, origin)
    assert asked == [0, 9]                                   # read from where it stopped


def test_a_first_sign_in_writes_the_persons_own_card_on_their_own_credential(tmp_path,
                                                                           monkeypatch):
    """John, 2026-11-01: the person's card belongs to themselves. Its root and its creation grant
    go to Origin on the person's own credential, in one Origin-issued batch; it is placed in no
    collection, and no one else is granted anything on it."""
    from .test_there_is_no_promotion_state import real_node
    db = real_node(tmp_path, monkeypatch)
    from mantle.services.seed_provisioning import user_provisioning as up

    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    origin = _install(FakeOrigin(), "enforce")
    ctx = up.UserContext(id="u-1", email="u1@x.test", name="U One")
    with acting_as("u-1", principal_type="user", source="request",
                   credential=ForwardedCredential("u1-token")):
        up._ensure_person_artifact(db, ctx)
    pid = up.person_artifact_id("u-1")
    mine = [(e, cred, b) for e, cred, b in zip(origin.entries, origin.credentials,
                                               origin.receipt_batches)
            if pid in (e.get("dst"), e.get("resource_id"), e.get("principal_id"))]
    kinds = [e["kind"] for e, _c, _b in mine]
    assert kinds.count("root") == 1 and "edge.add" not in kinds         # its own root, homed nowhere
    grants = [e for e, _c, _b in mine if e["kind"] == "grant"]
    assert [(g["grantee_id"], g["grantee_type"]) for g in grants] == [("u-1", "user")]
    assert {c for _e, c, _b in mine} == {"u1-token"}                    # the person's credential
    assert len({b for _e, _c, b in mine}) == 1                          # one batch
    assert not [e for e in origin.entries if e.get("kind") == "edge.add" and e.get("dst") == pid]
    assert la.get_artifact(db, pid) is not None


def test_under_threshold_a_card_origin_refuses_is_not_swallowed(tmp_path, monkeypatch):
    from .test_there_is_no_promotion_state import real_node
    db = real_node(tmp_path, monkeypatch)
    from mantle.services.seed_provisioning import user_provisioning as up

    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    _install(FakeOrigin(refuse=lambda e: e["kind"] == "root"), "enforce")
    ctx = up.UserContext(id="u-2", email="u2@x.test", name="U Two")
    with acting_as("u-2", principal_type="user", source="request",
                   credential=ForwardedCredential("u2-token")):
        with pytest.raises(Exception):
            up._ensure_person_artifact(db, ctx)


def test_a_submitted_entry_whose_confirmation_fails_is_recorded_as_owed(db):
    """L3: an entry submitted outside a store transaction and then not confirmed is recorded in
    `authority_confirm_due`, and told again by the next retry, as a transaction's would be."""
    class Unconfirming(FakeOrigin):
        def __init__(self):
            super().__init__()
            self.told = []
            self.fail = True

        def confirm_entries(self, seqs):
            if self.fail:
                raise RuntimeError("Origin is unreachable")
            self.told.extend(seqs)

    origin = Unconfirming()
    reach_journal.install(authority_ledger.Submitter(origin, "enforce"))
    entry = reach_journal.root_entry("p-1")
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        receipt = reach_journal.submit(entry)
    reach_journal.confirm_submitted(entry, receipt, db=db)
    conn = db.artifacts.db.conn()
    owed = [r[0] for r in conn.execute("SELECT seq FROM authority_confirm_due").fetchall()]
    assert owed == [receipt["seq"]]
    origin.fail = False
    reach_journal.retry_due(conn)
    assert origin.told == [receipt["seq"]]
    assert conn.execute("SELECT COUNT(*) FROM authority_confirm_due").fetchone()[0] == 0


def test_a_failed_confirmation_with_no_store_at_hand_is_kept_until_the_next_retry(db):
    class Unconfirming(FakeOrigin):
        fail = True
        told: list = []

        def confirm_entries(self, seqs):
            if self.fail:
                raise RuntimeError("Origin is unreachable")
            self.told.extend(seqs)

    origin = Unconfirming()
    reach_journal.install(authority_ledger.Submitter(origin, "enforce"))
    entry = reach_journal.root_entry("p-2")
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        receipt = reach_journal.submit(entry)
    reach_journal.confirm_submitted(entry, receipt)
    conn = db.artifacts.db.conn()
    reach_journal.retry_due(conn)                 # still failing: now recorded in the store
    owed = [r[0] for r in conn.execute("SELECT seq FROM authority_confirm_due").fetchall()]
    assert owed == [receipt["seq"]]


# -- sixth review (M3): the feed's cursor, and the store reconciled with Origin's replica ----------

def test_the_feed_cursor_stops_before_an_entry_it_fails_to_apply(db, person, monkeypatch):
    monkeypatch.setattr(authority_ledger, "verify_entry", lambda e, prev, keys=None: "h" * 64)
    origin = _install(FakeOrigin(), "enforce")
    for gid in ("g-1", "g-2", "g-3"):
        la.create_grant(db, _grant(gid))
    feed = {"entries": [
        {"seq": s, "kind": "grant", "credential_kind": "operator", "actor": "op",
         "body": {"kind": "grant", "grant_id": gid, "state": "revoked"}}
        for s, gid in ((5, "g-1"), (6, "g-2"), (7, "g-3"))], "head": 7}
    asked = []
    origin.ledger_feed = lambda after: asked.append(after) or {
        **feed, "entries": [e for e in feed["entries"] if e["seq"] > after]}
    real_put = db.artifacts.put_artifact

    def put(doc, *a, **k):
        if doc.get("id") == "g-2":
            raise RuntimeError("the store refused the write")
        return real_put(doc, *a, **k)

    monkeypatch.setattr(db.artifacts, "put_artifact", put)
    authority_ledger.apply_remote(db, origin)
    assert la.get_grant_by_id(db, "g-1").state == "revoked"
    assert la.get_grant_by_id(db, "g-2").state == "active"
    monkeypatch.setattr(db.artifacts, "put_artifact", real_put)
    authority_ledger.apply_remote(db, origin)
    assert asked == [0, 5]                                   # resumed at the entry that failed
    assert la.get_grant_by_id(db, "g-2").state == "revoked"
    assert la.get_grant_by_id(db, "g-3").state == "revoked"


def test_the_store_s_digest_of_reach_is_canonical(db, person):
    """The digest Mantle sends Origin is of its active grants and its `contains` edges, in a form
    that does not depend on row order or on how a mask is spelled."""
    _install(FakeOrigin(), "enforce")
    la.create_grant(db, _grant("g-b"))
    la.create_grant(db, _grant("g-a"))
    db.graph.add_edges([("C1", "m-1", "contains", {"is_origin": 1})])
    one = authority_ledger.reach_digest(db)
    assert one["grants"] == 2 and one["edges"] == 1 and len(one["digest"]) == 64
    la.create_grant(db, _grant("g-c"))
    assert authority_ledger.reach_digest(db)["digest"] != one["digest"]


def test_a_divergence_from_origin_s_replica_is_an_alarm(db, person, caplog):
    origin = _install(FakeOrigin(), "enforce")
    la.create_grant(db, _grant("g-1"))
    seen = {}

    def verify(digest):
        seen.update(digest)
        return {"match": False, "replica": {"digest": "0" * 64, "grants": 0, "edges": 0}}

    origin.verify_reach = verify
    with caplog.at_level(logging.ERROR):
        assert authority_ledger.verify_with_origin(db, origin) is False
    assert seen["grants"] == 1
    assert any("diverges" in r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    origin.verify_reach = lambda digest: {"match": True, "replica": digest}
    assert authority_ledger.verify_with_origin(db, origin) is True
