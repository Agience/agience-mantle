"""`mantle.system.people_to_signups`: the old People directory becomes the platform's sign-up
collection (John, 2026-11-02).

It renames the collection and revokes each member's owner grant on their old card, so the platform
owns those cards and members hold nothing on them. A revocation is a narrowing, so Origin takes it
from the platform's system principal. Nothing is deleted: cards, edges and the revoked grants all
stay. It is resumable and idempotent, and `--dry-run` changes nothing. Nothing adds a card to the
collection afterwards: a member signing in gets their own card, at an id of its own.
"""
from __future__ import annotations

import json
import uuid

import pytest

from mantle.db import lattice_api as la
from mantle.entities.artifact import Artifact
from mantle.entities.collection import COLLECTION_CONTENT_TYPE, Collection
from mantle.entities.grant import Grant
from mantle.services.acting_principal import ForwardedCredential, acting_as
from mantle.services.seed_provisioning import user_provisioning as up

PEOPLE = str(uuid.uuid4())
OPERATOR = "op-1"
SYSTEM = "sys-1"
MEMBERS = ("m-ada", "m-bo")


@pytest.fixture
def db(tmp_path, monkeypatch):
    from .test_there_is_no_promotion_state import real_node
    database = real_node(tmp_path, monkeypatch)
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: SYSTEM)
    _legacy(database)
    return database


def _legacy(db):
    """The People directory as provisioning wrote it before 2026-11-01."""
    la.create_collection(db, Collection(id=PEOPLE, name="People", description="directory",
                                        created_by=up.AGIENCE_PLATFORM_USER_ID,
                                        content_type=COLLECTION_CONTENT_TYPE, state="live"))
    la.create_grant(db, Grant(id="g-op", resource_id=PEOPLE, grantee_type="user",
                              grantee_id=OPERATOR, granted_by=up.AGIENCE_PLATFORM_USER_ID,
                              can_read=True, can_admin=True, name="People (admin management)"))
    la.create_grant(db, Grant(id="g-sys", resource_id=PEOPLE, grantee_type="user",
                              grantee_id=SYSTEM, granted_by=up.AGIENCE_PLATFORM_USER_ID,
                              can_read=True, can_update=True, name="Platform rows"))
    for m in MEMBERS:
        card = up.person_artifact_id(m)
        la.create_artifact(db, Artifact(
            id=card, root_id=card, collection_id=PEOPLE, state="live",
            content_type=up.PERSON_CONTENT_TYPE, name=f"Name {m}", created_by=m,
            context=json.dumps({"identity": {"agience_root_id": m}, "email": f"{m}@x.test",
                                "display_name": f"Name {m}"})))
        if not la.get_edge(db, PEOPLE, card):
            la.add_artifact_to_collection(db, PEOPLE, card, origin=True)
        la.create_grant(db, Grant(id=f"g-own-{m}", resource_id=card, grantee_type="user",
                                  grantee_id=m, granted_by=m, can_read=True, can_update=True,
                                  can_delete=True, name="Own profile (read + edit + delete)"))
        # What the member did with their card: shared it with a friend, and left an invite open.
        la.create_grant(db, Grant(id=f"g-friend-{m}", resource_id=card, grantee_type="user",
                                  grantee_id=f"friend-of-{m}", granted_by=m, can_read=True,
                                  name="shared"))
        la.create_grant(db, Grant(id=f"g-invite-{m}", resource_id=card, grantee_type="invite",
                                  grantee_id=f"hash-{m}", granted_by=m, can_read=True,
                                  name="invite"))


def _run(db, **kw):
    from mantle.system.people_to_signups import people_to_signups
    return people_to_signups(db, people_id=PEOPLE, **kw)


def _held(db, member):
    return la.get_active_grants_for_principal_resource(db, grantee_id=member,
                                                       resource_id=up.person_artifact_id(member))


def test_a_dry_run_counts_and_changes_nothing(db):
    before = {d["id"]: dict(d) for d in db.artifacts.list_artifacts()}
    counts = _run(db, dry_run=True)
    assert counts == {"collection": PEOPLE, "renamed": 1, "cards": 2, "revoked": 6,
                      "already_revoked": 0}
    assert {d["id"]: dict(d) for d in db.artifacts.list_artifacts()} == before


def test_the_directory_is_renamed_and_members_hold_nothing_on_their_old_cards(db):
    counts = _run(db)
    assert counts["renamed"] == 1 and counts["revoked"] == 6
    assert la.get_collection_by_id(db, PEOPLE).name == "Sign-ups"
    for m in MEMBERS:
        assert _held(db, m) == []
        old = la.get_grant_by_id(db, f"g-own-{m}")
        assert old is not None and old.state == Grant.STATE_REVOKED     # kept, not deleted
        assert la.get_artifact(db, up.person_artifact_id(m)) is not None   # the card stays
        assert la.get_edge(db, PEOPLE, up.person_artifact_id(m))           # and its edge
    held = {g.grantee_id for g in la.get_grants_for_collection(db, PEOPLE)
            if g.state == Grant.STATE_ACTIVE}
    assert held == {OPERATOR, SYSTEM}                                     # the platform's


def test_a_second_run_does_nothing(db):
    _run(db)
    assert _run(db) == {"collection": PEOPLE, "renamed": 0, "cards": 2, "revoked": 0,
                        "already_revoked": 6}


def test_an_interrupted_run_is_finished_by_the_next(db, monkeypatch):
    real = la.update_grant
    calls = []

    def once(database, entity):
        calls.append(entity.id)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return real(database, entity)

    monkeypatch.setattr(la, "update_grant", once)
    with pytest.raises(RuntimeError):
        _run(db)
    monkeypatch.setattr(la, "update_grant", real)
    counts = _run(db)
    assert counts["revoked"] == 5 and counts["already_revoked"] == 1
    assert all(_held(db, m) == [] for m in MEMBERS)


def test_the_revocations_go_to_origin_as_the_platforms_narrowing(db):
    from mantle.db import reach_journal
    from mantle.services import authority_ledger

    from .test_reach_changes_go_to_origin import FakeOrigin

    origin = FakeOrigin()
    reach_journal.install(authority_ledger.Submitter(origin, "enforce"))
    try:
        _run(db)
    finally:
        reach_journal.uninstall()
    revoked = [e for e in origin.entries if e.get("kind") == "grant"]
    assert {e["grant_id"] for e in revoked} == {f"g-{k}-{m}" for m in MEMBERS
                                                for k in ("own", "friend", "invite")}
    assert all(e["state"] == "revoked" for e in revoked)
    assert set(origin.credentials) == {f"background:{SYSTEM}"}


def test_a_member_signing_in_afterwards_gets_their_own_card_and_nothing_on_the_old(db):
    _run(db)
    ada = MEMBERS[0]
    ctx = up.UserContext(id=ada, email=f"{ada}@x.test", name="Ada")
    with acting_as(ada, principal_type="user", source="request",
                   credential=ForwardedCredential("ada-token")):
        assert up._ensure_person_artifact(db, ctx) is True
        assert up._ensure_person_artifact(db, ctx) is False             # idempotent
    own = up.own_card_id(db, ada)
    assert own != up.person_artifact_id(ada)
    card = la.get_artifact(db, own)
    assert card is not None and getattr(card, "collection_id", None) in (None, "", own)
    assert any(g.can_admin for g in la.get_active_grants_for_principal_resource(
        db, grantee_id=ada, resource_id=own))
    assert _held(db, ada) == []                                          # still nothing there
    members = {r.get("root_id") or r.get("id") for r in la.list_collection_artifacts(db, PEOPLE)}
    assert own not in members and len(members) == 2                      # nothing added


def test_a_new_member_keeps_the_card_id_they_always_had(db):
    """Someone who never had a People card gets their own card at `person_artifact_id`."""
    ctx = up.UserContext(id="m-new", email="new@x.test", name="New")
    with acting_as("m-new", principal_type="user", source="request",
                   credential=ForwardedCredential("new-token")):
        up._ensure_person_artifact(db, ctx)
    assert up.own_card_id(db, "m-new") == up.person_artifact_id("m-new")


def test_grants_a_member_issued_on_their_old_card_are_revoked_and_invites_closed(db):
    """L2: what a member handed out on their old card ends with their hold on it."""
    _run(db)
    for m in MEMBERS:
        for gid in (f"g-friend-{m}", f"g-invite-{m}"):
            assert la.get_grant_by_id(db, gid).state == Grant.STATE_REVOKED, gid
        assert la.get_active_grants_for_principal_resource(
            db, grantee_id=f"friend-of-{m}", resource_id=up.person_artifact_id(m)) == []


def test_the_member_is_named_by_the_card_s_derived_id_not_its_context(db):
    """L2: the card's context is editable; its id is derived from its member's. A context that
    names someone else changes nothing about whose card it is."""
    ada = MEMBERS[0]
    card = la.get_artifact(db, up.person_artifact_id(ada))
    card.context = json.dumps({"identity": {"agience_root_id": "m-bo"}, "email": "x@x.test"})
    card.created_by = "m-bo"
    la.update_artifact(db, card)
    _run(db)
    assert _held(db, ada) == []
    assert la.get_grant_by_id(db, f"g-own-{ada}").state == Grant.STATE_REVOKED
