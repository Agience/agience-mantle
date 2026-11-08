"""First sign-in under enforcement, against a real Origin app: Mantle's provisioning sends its reach
changes through Mantle's own Origin client to Origin's own routes, and Origin decides them by its
own rules on its own replica.

John, 2026-11-01: "We don't need a shared people directory. Their own card belongs to themselves,
and they can share it in a collection they can add to. But a signup could also produce a card that
goes into a platform collection that is separate, belonging to the platform."

The person's card is their own root, created on their own credential with their own creation grant
in one Origin-issued batch, and no one else holds anything on it. The platform's sign-up card is
not written: no credential either side holds today lets it into a platform collection (the person
holds no Add there, and the platform's background delegation adds no edge). The last tests here
hold that, on Origin's own rules, until John decides how a platform collection receives it.
"""
from __future__ import annotations

import uuid

import pytest

pytest.importorskip("origin")

from mantle.db import lattice_api as la  # noqa: E402
from mantle.db import reach_journal  # noqa: E402
from mantle.services import authority_ledger  # noqa: E402
from mantle.services.acting_principal import ForwardedCredential, acting_as  # noqa: E402


@pytest.fixture
def origin_client(monkeypatch, tmp_path_factory):
    """Origin's own app on an in-memory database, with a manifest anchoring Mantle's test key:
    Origin's own fixtures, called here so Mantle's install-root guard stays in force."""
    from origin.tests import conftest as origin_conftest
    from origin.tests import test_a_key_share_is_released_only_to_mantle as shares

    for name in origin_conftest._CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)
    from origin.services import custody_plane
    monkeypatch.setattr(custody_plane, "TEST_OVERRIDE", True)   # stands in for isolated production
    keys = origin_conftest.jwt_keypair.__wrapped__(tmp_path_factory)
    app = origin_conftest.origin_app.__wrapped__(monkeypatch, keys)
    made = shares.custody_client.__wrapped__(app, monkeypatch)
    c = next(made)
    try:
        yield c
    finally:
        made.close()


@pytest.fixture
def custodian(origin_client, monkeypatch):
    """Mantle's own Origin client, its service tokens signed with the key Origin's manifest
    anchors for Mantle."""
    from mantle.clients import origin_key_share_client as oc
    from origin.tests import test_a_key_share_is_released_only_to_mantle as shares

    def sign(*, audience, ttl_seconds, additional_claims):
        claims = dict(additional_claims)
        return shares._token(claims.pop("htm"), claims.pop("principal_id", "-"), aud=audience,
                             ttl=ttl_seconds, **claims)

    monkeypatch.setattr(oc.peer_signing, "sign_service_jwt", sign)
    custodian = oc.OriginShareCustodian("http://testserver", client=origin_client)
    reach_journal.install(authority_ledger.Submitter(custodian, "enforce"))
    try:
        yield custodian
    finally:
        reach_journal.uninstall()


@pytest.fixture
def node(tmp_path, monkeypatch):
    from .test_there_is_no_promotion_state import real_node
    return real_node(tmp_path, monkeypatch)


def _replica():
    from origin.db.session import SessionLocal
    return SessionLocal()


def _grants_on(resource):
    from origin.models.authority_ledger import AuthorityGrant
    with _replica() as db:
        return [(g.grantee_id, g.grantee_type, g.actions, g.state)
                for g in db.query(AuthorityGrant).filter(AuthorityGrant.resource_id == resource)]


def _edges_to(node_id):
    from origin.models.authority_ledger import AuthorityEdge
    with _replica() as db:
        return [(e.src, e.dst) for e in db.query(AuthorityEdge).filter(AuthorityEdge.dst == node_id)]


def _sign_in(db, person_id, token, *, name="P"):
    from mantle.services.seed_provisioning import user_provisioning as up
    ctx = up.UserContext(id=person_id, email=f"{name.lower()}@x.test", name=name)
    with acting_as(person_id, principal_type="user", source="request",
                   credential=ForwardedCredential(token)):
        up._ensure_person_artifact(db, ctx)
    return up.person_artifact_id(person_id)


def test_first_sign_in_writes_the_persons_own_card_at_a_real_origin(custodian, node):
    from origin.models.authority_ledger import AuthorityEntry, AuthorityRoot
    from origin.tests import authority_fixtures as af

    alice, alice_token = af.person("Alice")
    card = _sign_in(node, alice, alice_token, name="Alice")

    assert la.get_artifact(node, card) is not None
    with _replica() as db:
        root = db.get(AuthorityRoot, card)
        assert root is not None and root.root_id == card            # its own root
        made = db.query(AuthorityEntry).filter(AuthorityEntry.seq > 0).all()
        mine = [e for e in made if card in (e.body or "")]
        assert mine and len({e.batch for e in mine}) == 1          # one Origin-issued batch
        assert all(e.confirmed for e in mine)                       # Mantle's write committed
    grants = _grants_on(card)
    assert [(g, t, s) for g, t, _a, s in grants] == [(alice, "user", "active")]
    assert _edges_to(card) == []                                    # placed in no collection


def test_the_card_is_the_persons_and_no_one_elses_at_a_real_origin(custodian, node,
                                                                   origin_client):
    from origin.tests import authority_fixtures as af

    alice, alice_token = af.person("Alice")
    bob, bob_token = af.person("Bob")
    card = _sign_in(node, alice, alice_token, name="Alice")
    principal = f"ck:{card}"
    assert af.store_share(origin_client, principal, credential=alice_token).status_code == 200
    mine = af.release(origin_client, principal, alice_token, collection=card)
    assert mine.status_code == 200, mine.text                      # she reaches her own card
    theirs = af.release(origin_client, principal, bob_token, collection=card)
    assert theirs.status_code in (403, 404) and "y" not in theirs.text   # nobody else does


@pytest.fixture
def signups(origin_client, monkeypatch):
    """A platform collection, made at Origin by the operator: its root and its creation grant in
    one batch, on the operator's own credential."""
    from origin.tests import authority_fixtures as af

    op = af.operator(monkeypatch)
    sid = f"signups-{uuid.uuid4().hex[:8]}"
    af.ok(af.submit(origin_client, op, af.root(sid), batch="signups"))
    af.ok(af.submit(origin_client, op, af.grant(f"g-{sid}", af.OPERATOR, sid,
                                                 actions=af_all_actions()), batch="signups"))
    return sid


def af_all_actions():
    from mantle.attenuation import ACTIONS
    return tuple(ACTIONS)


def test_a_person_holds_no_add_on_a_platform_collection_at_a_real_origin(custodian, node,
                                                                         origin_client, signups):
    """The person cannot write a sign-up card into the platform's collection: they hold no Add on
    it, and Origin refuses the creation edge on their credential."""
    from origin.tests import authority_fixtures as af

    alice, alice_token = af.person("Alice")
    _sign_in(node, alice, alice_token, name="Alice")
    r = af.submit(origin_client, alice_token, af.edge(signups, f"card-{uuid.uuid4().hex[:8]}"),
                  batch=None)
    assert r.status_code == 403 and "neither create nor add" in r.text


def test_the_platforms_background_credential_adds_no_card_at_a_real_origin(origin_client,
                                                                           signups, monkeypatch):
    """Nor can the platform write it on the credential it holds at run time: a background
    delegation, the platform system principal's own, adds no edge."""
    from origin import service_identity
    from origin.tests import authority_fixtures as af
    from origin.tests import test_a_key_share_is_released_only_to_mantle as shares

    system = "00000000-0000-0000-0000-00000000005e"
    monkeypatch.setattr(service_identity, "get_system_principal_id", lambda: system)
    background = shares.own_credential(system)
    r = af.submit(origin_client, background, af.edge(signups, f"card-{uuid.uuid4().hex[:8]}"),
                  batch=None)
    assert r.status_code == 403 and "background delegation does not add" in r.text


def test_a_person_cannot_read_a_platform_collection_at_a_real_origin(custodian, node,
                                                                     origin_client, signups,
                                                                     monkeypatch):
    from origin.tests import authority_fixtures as af

    op = af.operator(monkeypatch)
    alice, alice_token = af.person("Alice")
    _sign_in(node, alice, alice_token, name="Alice")
    principal = f"ck:{signups}"
    assert af.store_share(origin_client, principal, credential=op).status_code == 200
    assert af.release(origin_client, principal, op, collection=signups).status_code == 200
    r = af.release(origin_client, principal, alice_token, collection=signups)
    assert r.status_code in (403, 404) and "y" not in r.text


def test_the_whole_first_sign_in_is_co_signed_at_a_real_origin(custodian, node, monkeypatch):
    """The whole first sign-in (Inbox, observations container, card), on the person's own
    credential: every change it makes to reach, Origin co-signs and Mantle confirms."""
    from origin.models.authority_ledger import AuthorityEntry
    from origin.tests import authority_fixtures as af

    from mantle.services.seed_provisioning import user_provisioning as up

    monkeypatch.delenv("AGIENCE_SEEDS_ROOT", raising=False)
    alice, alice_token = af.person("Alice")
    with acting_as(alice, principal_type="user", source="request",
                   credential=ForwardedCredential(alice_token)):
        up.provision_user(node, alice, email="alice@x.test", name="Alice")
    card = up.person_artifact_id(alice)
    assert la.get_artifact(node, card) is not None
    with _replica() as db:
        made = db.query(AuthorityEntry).all()
        assert made and {e.actor for e in made} == {alice}
        assert all(e.confirmed for e in made)
    assert [(g, t) for g, t, _a, _s in _grants_on(card)] == [(alice, "user")]


# -- the platform's user list: Origin's own records (John, 2026-11-02) ----------------------------

@pytest.fixture
def pinned_authorities(origin_client, monkeypatch):
    """The Authorities collection at Origin, founded by Ada with Admin, and pinned there."""
    from origin.services.platform_settings_service import settings as platform_settings
    from origin.tests import authority_fixtures as af

    af.operator(monkeypatch)
    ada, ada_token = af.person("Ada", email="ada@x.test")
    aid = f"authorities-{uuid.uuid4().hex[:8]}"
    af.ok(af.submit(origin_client, ada_token, af.root(aid), batch=aid))
    af.ok(af.submit(origin_client, ada_token, af.grant(f"g-{aid}", ada, aid,
                                                        actions=("read", "admin")), batch=aid))
    real_get = platform_settings.get
    monkeypatch.setattr(platform_settings, "get", lambda k, *a, **kw: aid
                        if k == "platform.authority_collection_id" else real_get(k, *a, **kw))
    return ada, ada_token


def test_the_admin_user_list_is_origins_own_records(custodian, node, pinned_authorities):
    """The list is Origin's person records, not cards in Mantle: a card left in the old People
    directory for someone Origin does not know is not listed, and everyone Origin knows is."""
    from origin.tests import authority_fixtures as af

    from mantle.routers import system_router

    ada, ada_token = pinned_authorities
    bo, _ = af.person("Bo", email="bo@x.test")
    users, total = system_router._platform_user_page(node, 100, 0, credential=ada_token,
                                                     custodian=custodian)
    ids = {u.id for u in users}
    assert {ada, bo} <= ids and total == len(users)
    assert next(u for u in users if u.id == bo).email == "bo@x.test"


def test_a_member_who_is_no_admin_is_refused_the_list_by_origin(custodian, node,
                                                                 pinned_authorities):
    from fastapi import HTTPException
    from origin.tests import authority_fixtures as af

    from mantle.routers import system_router

    _bo, bo_token = af.person("Bo")
    with pytest.raises(HTTPException) as refused:
        system_router._platform_user_page(node, 100, 0, credential=bo_token, custodian=custodian)
    assert refused.value.status_code == 403


# -- M3: Origin's verdict per action, and the store reconciled with the replica ------------------

def test_origin_s_verdict_for_an_action_is_its_own_walk(custodian, node):
    from origin.tests import authority_fixtures as af

    alice, alice_token = af.person("Alice")
    _bob, bob_token = af.person("Bob")
    card = _sign_in(node, alice, alice_token, name="Alice")
    assert custodian.reach_verdict(card, "invoke", credential=alice_token) is True
    assert custodian.reach_verdict(card, "invoke", credential=bob_token) is False


def test_the_store_and_the_replica_agree_after_a_sign_in(custodian, node):
    from origin.tests import authority_fixtures as af

    from mantle.services import authority_ledger as al

    alice, alice_token = af.person("Alice")
    _sign_in(node, alice, alice_token, name="Alice")
    assert al.verify_with_origin(node, custodian) is True


def test_a_grant_written_past_origin_is_a_divergence(custodian, node, caplog):
    import logging

    from origin.tests import authority_fixtures as af

    from mantle.entities.grant import Grant
    from mantle.services import authority_ledger as al

    alice, alice_token = af.person("Alice")
    card = _sign_in(node, alice, alice_token, name="Alice")
    with reach_journal.suspended():                          # as a drifted store would hold it
        la.create_grant(node, Grant(id="g-drift", resource_id=card, grantee_type="user",
                                    grantee_id="mal", granted_by=alice, can_read=True))
    with caplog.at_level(logging.ERROR):
        assert al.verify_with_origin(node, custodian) is False
    assert any("diverges" in r.getMessage() for r in caplog.records)
