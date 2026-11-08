"""Under threshold custody, Mantle shows Origin the acting caller's own credential for each share.

Origin releases share 2 only on a live credential it issued itself (Origin's side is in
`agience-origin/.../test_a_share_release_needs_the_callers_credential.py`). Mantle's part is to
carry the right one: the bearer a person presented, or, for background work and the platform
system principal, a delegation Origin mints for the one principal being opened and the job the
block names (only under a live job grant for both at Origin); to name the collection or artifact the key
is wanted for; and to refuse, by name, when there is no credential, including for a grant key,
which Origin cannot verify. Lattice custody is untouched.
"""
from __future__ import annotations

import base64
import json

import httpx
import pytest

from mantle.search.mantle import threshold_custody as tc
from mantle.search.mantle.oracle import (
    KeyPurpose, KeyRequest, LatticeMasterKeyStore, MasterKeyUnavailable, OracleService,
)
from mantle.services.acting_principal import (
    ActingPrincipal, ForwardedCredential, acting_as, forwardable_credential,
)

from .test_a_split_master_key_needs_two_custodians import ALICE_KEY, _SelfOnly, make_env

ISSUER = "http://origin.test"


def _jwt(claims: dict) -> str:
    seg = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()  # noqa: E731
    return f"{seg({'alg': 'RS256'})}.{seg(claims)}.sig"


class _Auth:
    def __init__(self, principal_type="user"):
        self.principal_type = principal_type


# -- which bearer is carried ---------------------------------------------------------------------

def test_an_origin_person_token_and_delegation_are_carried():
    for claims in ({"iss": ISSUER, "sub": "alice"},
                   {"iss": ISSUER, "sub": "alice", "principal_type": "delegation",
                    "act": {"sub": "seraph"}}):
        tok = _jwt(claims)
        cred = forwardable_credential(f"Bearer {tok}", _Auth(), ISSUER)
        assert cred is not None and cred.token == tok


@pytest.mark.parametrize("token,auth", [
    ("agk_abc.def.ghi", _Auth("grant_key")),
    (_jwt({"iss": "https://other-idp.test", "sub": "x"}), _Auth()),
    (_jwt({"iss": ISSUER, "sub": "x", "aud": "client"}), _Auth("mcp_client")),
    (_jwt({"iss": "mantle", "sub": "mantle", "principal_type": "service"}), _Auth("service")),
    ("not-a-jwt", _Auth()),
])
def test_anything_else_is_not_carried(token, auth):
    assert forwardable_credential(token, auth, ISSUER) is None


def test_the_credential_never_shows_in_a_repr():
    tok = _jwt({"iss": ISSUER, "sub": "alice"})
    p = ActingPrincipal(principal_id="alice", credential=ForwardedCredential(tok))
    assert tok not in repr(p) and tok not in str(p.credential)
    assert p == ActingPrincipal(principal_id="alice")       # not part of equality either


# -- which credential Origin is shown ------------------------------------------------------------

class SeeingOrigin:
    """Records the credential each release was asked with, and refuses none."""

    def __init__(self, inner):
        self.inner, self.seen, self.scopes = inner, [], []

    def get_share(self, principal_id, *, credential, collection_id=None, artifact_id=None):
        if not credential:
            raise RuntimeError("401 the acting caller's credential is required")
        self.seen.append(credential)
        self.scopes.append((collection_id, artifact_id))
        return self.inner.get_share(principal_id)

    def background_delegation(self, principal_id, job=""):
        return self.inner.background_delegation(principal_id, job)

    def put_share(self, *a, **kw):
        return self.inner.put_share(*a, **kw)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    e = make_env(tmp_path)
    e.origin = SeeingOrigin(e.origin)
    e.store = e.make()
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        e.store.put("alice", ALICE_KEY)
    e.origin.seen.clear()
    e.origin.scopes.clear()
    return e


def test_a_request_shows_origin_its_own_bearer(env):
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        assert env.store.get("alice") == ALICE_KEY
    assert env.origin.seen == ["alice-token"]


def test_a_request_with_no_origin_credential_refuses_and_says_why(env):
    with acting_as("alice", principal_type="user", source="request"):
        with pytest.raises(MasterKeyUnavailable, match="carries no credential Origin issued"):
            env.store.get("alice")
    assert env.origin.seen == []


def test_a_grant_key_refuses_cleanly(env):
    with acting_as("grant-1", principal_type="grant_key", source="request"):
        with pytest.raises(tc.GrantKeyCannotOpenShare, match="grant key"):
            env.store.get("alice")
    assert env.origin.seen == []


def test_background_work_shows_a_delegation_minted_for_the_principal_it_opens(env):
    with acting_as("alice", source="system", scope="platform.reindex"):   # a per-owner pass
        assert env.store.get("alice") == ALICE_KEY
    assert env.origin.seen == ["background:alice"] and env.origin.inner.minted == ["alice"]


def test_background_work_without_a_job_grant_refuses_and_says_why(env):
    env.origin.inner.background = False
    with acting_as("alice", source="system", scope="platform.reindex"):
        with pytest.raises(MasterKeyUnavailable, match="job grant"):
            env.store.get("alice")
    assert env.origin.seen == []


def test_the_system_principal_on_a_request_uses_a_minted_delegation(env, monkeypatch):
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    with acting_as("sys-1", principal_type="user", source="request", scope="platform.describe",
                   credential=ForwardedCredential("describe-delegation")):
        env.store.get("alice")
    assert env.origin.seen == ["background:alice"]


def test_nothing_long_lived_is_read_from_this_node_s_disk():
    """The year-long platform delegation file is gone: no setting, no reader."""
    assert not hasattr(tc, "platform_delegation") and not hasattr(tc, "custody_delegation_path")


def test_a_release_names_the_collection_and_the_artifact_s_root(env):
    env.db.artifacts.put_artifact({"id": "v-2", "root_id": "card-1"})
    oracle = OracleService(env.store, grant_verifier=_SelfOnly())
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        oracle.get_or_create_master_key("alice", KeyRequest(
            requester_id="alice", purpose=KeyPurpose.SELF, artifact_id="v-2"),
            collection_id="coll-1")
    assert env.origin.scopes == [("coll-1", "card-1")]


def test_a_cached_key_is_not_served_to_a_caller_that_could_not_open_it(env):
    oracle = OracleService(env.store, grant_verifier=_SelfOnly())
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        oracle.get_or_create_master_key("alice", KeyRequest(requester_id="alice",
                                                            purpose=KeyPurpose.SELF))
    with acting_as("alice", principal_type="grant_key", source="request"):
        with pytest.raises(tc.GrantKeyCannotOpenShare):
            oracle.get_or_create_master_key("alice", KeyRequest(requester_id="alice",
                                                                purpose=KeyPurpose.SELF))


def test_lattice_custody_serves_a_grant_key_as_before(env):
    whole = LatticeMasterKeyStore(env.kek, lambda: env.db)
    whole.put("bob", ALICE_KEY)
    oracle = OracleService(whole, grant_verifier=_SelfOnly())
    with acting_as("bob", principal_type="grant_key", source="request"):
        assert oracle.get_or_create_master_key(
            "bob", KeyRequest(requester_id="bob", purpose=KeyPurpose.SELF)) == ALICE_KEY


# -- the wire ------------------------------------------------------------------------------------

def test_the_client_sends_the_credential_beside_the_service_token(monkeypatch):
    from mantle.clients import origin_key_share_client as oc

    monkeypatch.setattr(oc.peer_signing, "sign_service_jwt", lambda **_k: "svc")
    seen = {}

    def handler(req):
        seen.update(req.headers)
        return httpx.Response(200, json={"x": 2, "y": "AA==", "kcv": "k"})

    c = oc.OriginShareCustodian("http://origin.test",
                                client=httpx.Client(transport=httpx.MockTransport(handler)))
    c.get_share("p", credential="alice-token")
    assert seen["authorization"] == "Bearer svc" and seen["x-acting-credential"] == "alice-token"
    with pytest.raises(oc.CustodianUnavailable):
        c.get_share("p", credential="")


def test_origin_s_refusal_reason_reaches_the_caller(monkeypatch):
    from mantle.clients import origin_key_share_client as oc

    monkeypatch.setattr(oc.peer_signing, "sign_service_jwt", lambda **_k: "svc")
    c = oc.OriginShareCustodian("http://origin.test", client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(
            403, json={"detail": "acting credential refused: a revoked background delegation"}))))
    with pytest.raises(oc.CustodianUnavailable, match="revoked background delegation"):
        c.get_share("p", credential="tok")


@pytest.mark.parametrize("ptype,carried", [("user", True), ("grant_key", False)])
async def test_the_auth_dependency_carries_the_bearer_into_the_acting_principal(
        monkeypatch, ptype, carried):
    from mantle import config
    from mantle.services import dependencies as d
    from mantle.services.acting_principal import current_acting_principal

    monkeypatch.setattr(d, "resolve_auth", lambda **_kw: d.AuthContext(
        principal_id="alice", principal_type=ptype, user_id="alice"))
    monkeypatch.setattr(d, "enforce_token_bounds", lambda *a, **k: None)
    monkeypatch.setattr(config, "AUTHORITY_ISSUER", ISSUER)
    tok = _jwt({"iss": ISSUER, "sub": "alice"})
    await d.get_auth(token=tok, store_db=None, request=None)
    cred = current_acting_principal().credential
    assert (cred.token if cred else None) == (tok if carried else None)


def test_a_refusal_from_origin_never_falls_through_to_escrow_recovery(tmp_path):
    """A refusal is an answer, about who may open the key: a node without the escrow identity
    refuses on it. Offline recovery holds the identity and asks Origin nothing (review 33, F1,
    John's decision 2026-11-08): it combines this node's share with the escrow share whether
    Origin refuses or is down. The identity never sits on a running node."""
    from mantle.clients.origin_key_share_client import AuthorityRefused, CustodianUnavailable

    e = make_env(tmp_path)
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        e.store.put("alice", ALICE_KEY)
    recovering = e.make(recovery_identity=e.identity)

    def refuse(*a, **kw):
        raise AuthorityRefused(403, "not released: the subject does not reach this key")

    e.origin.get_share = refuse
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        with pytest.raises(tc.ShareRefused, match="does not reach"):
            e.store.get("alice")                              # a running node: refused
        assert recovering.get("alice") == ALICE_KEY          # offline recovery: Origin not asked

    def down(*a, **kw):
        raise CustodianUnavailable("Origin is unreachable (ConnectError)")

    e.origin.get_share = down
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        assert recovering.get("alice") == ALICE_KEY          # an outage: recovery applies


def test_a_first_share_is_stored_only_on_a_credential(env):
    """Origin stores a first share only on its principal's or creator's credential, so a store
    with none refuses here, before anything is sent."""
    with acting_as("bob", principal_type="user", source="request"):
        with pytest.raises(MasterKeyUnavailable, match="carries no credential Origin issued"):
            env.store.put("bob", b"b" * 32)
    assert env.store._row("bob") is None                         # nothing written


def test_a_cached_key_is_used_for_each_requester_only_on_their_own_verdict(env):
    """Architecture §51: the verdict memo (30 s) bounds post-revocation validity. A key one
    requester opened is not served warm to another until Origin has decided for that one, and
    a refusal for them stands while the key stays cached for the first."""
    from mantle.search.mantle import oracle as o

    class Yes:
        def authorized(self, **kw):
            return True

    refused = set()
    real = env.origin.get_share

    def get_share(principal_id, *, credential, **kw):
        if credential in refused:
            raise MasterKeyUnavailable("Origin refused")
        return real(principal_id, credential=credential, **kw)

    env.origin.get_share = get_share
    oracle = o.OracleService(env.store, grant_verifier=Yes())
    req = lambda who: o.KeyRequest(requester_id=who, purpose=o.KeyPurpose.GRANT)   # noqa: E731
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        assert oracle.get_or_create_master_key("alice", req("alice")) == ALICE_KEY
    asked = len(env.origin.seen)
    refused.add("bob-token")
    with acting_as("bob", principal_type="user", source="request",
                   credential=ForwardedCredential("bob-token")):
        with pytest.raises(Exception):
            oracle.get_or_create_master_key("alice", req("bob"))
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        assert oracle.get_or_create_master_key("alice", req("alice")) == ALICE_KEY
    assert len(env.origin.seen) == asked                         # alice's verdict is memoized


def test_an_answer_over_the_cap_is_refused_on_every_call(monkeypatch):
    """Review 19: every call to Origin reads at most `MAX_ANSWER_BYTES`; a larger answer is
    Origin unreachable, never parsed."""
    from mantle.clients import origin_key_share_client as oc

    monkeypatch.setattr(oc.peer_signing, "sign_service_jwt", lambda **_k: "svc")
    big = b'{"x": 2, "pad": "' + b"a" * (oc.MAX_ANSWER_BYTES + 10) + b'"}'
    c = oc.OriginShareCustodian("http://origin.test", client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=big))))
    with pytest.raises(oc.CustodianUnavailable, match="too large"):
        c.get_share("p", credential="tok")
    with pytest.raises(oc.CustodianUnavailable, match="too large"):
        c.ledger_feed(0)
    small = oc.OriginShareCustodian("http://origin.test", client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(
            200, json={"x": 2, "y": "AA==", "kcv": "k"}))))
    assert small.get_share("p", credential="tok")["x"] == 2
