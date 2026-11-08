"""Review 12, decision 1 and L13: authority changes are made at Origin, and Mantle applies them from
Origin's feed, verified.

* With Origin's ledger installed, Mantle's own grant routes change no authority: a grant, an invite,
  a deny, a claim, a grant key or its members is made in the person's Origin session, and Mantle
  answers 409 naming that. A creation grant for what a person creates through Mantle is unaffected.
* Mantle applies every entry made in a person's Origin session (`origin_session`, the kind
  Origin writes for it; review 17) from the feed:
  grants (denies and invites included) and containment edges, with their masks.
* Before applying anything, Mantle checks each entry: its hash recomputed from its body, its
  co-signature by the kid's key from the authority manifest's Origin anchor (never the feed's own
  key list), and the chain from the last entry it applied. An entry that fails stops the pass.
"""
from __future__ import annotations

import base64
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from jose import jwk

from mantle.db import lattice_api as la
from mantle.db import reach_journal
from mantle.services import authority_ledger

GENESIS = "0" * 64
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwk(key, kid):
    pub = key.public_key().public_bytes(serialization.Encoding.PEM,
                                        serialization.PublicFormat.SubjectPublicKeyInfo)
    j = jwk.construct(pub, "RS256").to_dict()
    j = {a: (b.decode() if isinstance(b, bytes) else b) for a, b in j.items()}
    j.update({"kid": kid, "use": "sig"})
    return j


@pytest.fixture(autouse=True)
def manifest(tmp_path, monkeypatch):
    (tmp_path / "authority.manifest.json").write_text(json.dumps(
        {"issuer": "https://origin.test", "artifact_id": "m",
         "trust_anchors": {"origin": {"jwks": {"keys": [_jwk(KEY, "origin-1")]}}}}))
    monkeypatch.setenv("KEYS_DIR", str(tmp_path))


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _entry(seq, prev, body, *, kind_of="origin_session", actor="alice", key=KEY, kid="origin-1",
           tamper=False):
    env = {"v": 2, "seq": seq, "prev": prev, "body": body, "actor": actor,
           "credential_kind": kind_of, "credential_ref": "session:alice", "bootstrap": False,
           "compensates": None, "kid": kid, "claimed_from": None}
    digest = hashlib.sha256(_canonical(env).encode()).hexdigest()
    sig = base64.urlsafe_b64encode(key.sign(digest.encode("ascii"), padding.PKCS1v15(),
                                            hashes.SHA256())).rstrip(b"=").decode()
    b = dict(body)
    if tamper:
        b["actions"] = ["read", "admin"]
    return {"seq": seq, "kind": body["kind"], "body": b, "actor": actor,
            "credential_kind": kind_of, "credential_ref": "session:alice", "bootstrap": False,
            "compensates": None, "claimed_from": None, "submitted_by": None, "prev_hash": prev,
            "entry_hash": digest, "signature": sig, "kid": kid}


def _grant_body(gid="g-1", actions=("read",), effect="allow"):
    return {"kind": "grant", "grant_id": gid, "grantee_id": "bob", "grantee_type": "user",
            "resource_id": "C1", "actions": list(actions), "effect": effect, "state": "active",
            "expires_at": None, "target": None, "max_claims": None}


class Feed:
    def __init__(self, entries):
        self.entries = entries

    def ledger_feed(self, after):
        return {"entries": [e for e in self.entries if e["seq"] > after],
                "keys": {"origin-1": "forged"}}


@pytest.fixture
def db(tmp_path):
    return la.LatticeDatabase(str(tmp_path / "f.db"), origin="f")


def _grant(db, gid="g-1"):
    return la.get_grant_by_id(db, gid)


def test_a_session_grant_from_the_feed_is_applied(db):
    e1 = _entry(1, GENESIS, _grant_body())
    assert authority_ledger.apply_remote(db, Feed([e1])) == 1
    g = _grant(db)
    assert g is not None and g.grantee_id == "bob" and g.can_read and not g.can_admin


def test_a_session_deny_and_an_edge_with_its_mask_are_applied(db):
    e1 = _entry(1, GENESIS, _grant_body("g-d", effect="deny"))
    e2 = _entry(2, e1["entry_hash"], {"kind": "edge.add", "src": "C1", "dst": "A1",
                                      "origin": False, "propagate": ["read"]})
    assert authority_ledger.apply_remote(db, Feed([e1, e2])) == 2
    assert _grant(db, "g-d").effect == "deny"
    assert la.get_edge(db, "C1", "A1")


def test_an_entry_whose_body_does_not_match_its_hash_is_refused(db):
    e1 = _entry(1, GENESIS, _grant_body(), tamper=True)
    assert authority_ledger.apply_remote(db, Feed([e1])) == 0
    assert _grant(db) is None


def test_an_entry_signed_by_a_key_the_manifest_does_not_anchor_is_refused(db):
    e1 = _entry(1, GENESIS, _grant_body(), key=OTHER)
    assert authority_ledger.apply_remote(db, Feed([e1])) == 0
    assert _grant(db) is None


def test_a_broken_chain_stops_the_pass_before_the_break(db):
    e1 = _entry(1, GENESIS, _grant_body("g-1"))
    e2 = _entry(2, "f" * 64, _grant_body("g-2"))
    assert authority_ledger.apply_remote(db, Feed([e1, e2])) == 1
    assert _grant(db, "g-1") is not None and _grant(db, "g-2") is None
    # The next pass resumes at entry 2 and still refuses it.
    assert authority_ledger.apply_remote(db, Feed([e1, e2])) == 0


def test_mantle_s_own_entries_are_verified_and_passed_over(db):
    e1 = _entry(1, GENESIS, _grant_body("g-own"), kind_of="access")
    e2 = _entry(2, e1["entry_hash"], _grant_body("g-2"))
    assert authority_ledger.apply_remote(db, Feed([e1, e2])) == 1
    assert _grant(db, "g-own") is None and _grant(db, "g-2") is not None


def test_mantle_s_grant_routes_change_no_authority_while_the_ledger_is_installed():
    from fastapi import HTTPException

    from mantle.routers import grants_router

    reach_journal.install(lambda *a, **k: None)
    try:
        with pytest.raises(HTTPException) as refused:
            grants_router.authority_changes_are_made_at_origin()
        assert refused.value.status_code == 409 and "Origin" in refused.value.detail
    finally:
        reach_journal.uninstall()
    grants_router.authority_changes_are_made_at_origin()     # no ledger: Mantle alone


@pytest.mark.parametrize("method,path", [
    ("POST", "/grants"), ("POST", "/grants/claim"), ("POST", "/grants/keys"),
    ("DELETE", "/grants/keys/{key_id}"), ("POST", "/grants/keys/{key_id}/members"),
    ("DELETE", "/grants/keys/{key_id}/members/{member_id}"), ("DELETE", "/grants/{grant_id}"),
    ("POST", "/grants/{grant_id}/accept"),
])
def test_every_grant_route_that_changes_authority_carries_the_rule(method, path):
    from mantle.routers import grants_router

    def calls(dep):
        yield dep.call
        for d in dep.dependencies:
            yield from calls(d)

    route = next(r for r in grants_router.router.routes
                 if r.path == path and method in r.methods)
    assert grants_router.authority_changes_are_made_at_origin in set(calls(route.dependant))


def test_mantle_s_entry_hash_is_origin_s():
    origin_ledger = pytest.importorskip("origin.services.authority_ledger")
    e = _entry(7, "a" * 64, _grant_body("g-p"))
    assert authority_ledger.entry_hash(e) == origin_ledger.entry_hash(
        seq=7, prev_hash="a" * 64, body=e["body"], actor=e["actor"],
        credential_kind=e["credential_kind"], credential_ref=e["credential_ref"],
        bootstrap=False, compensates=None, kid=e["kid"], claimed_from=None)
