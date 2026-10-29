"""An invite by email is mailed, its member is listed under the address it went to, and the
owner can resend it or revoke it.

2026-10-23, a tester: Share → invite by email returned 201 and no mail arrived; the owner found no
member list, no pending invites and no revoke or resend. The mail helper existed and was tested,
and the invite route never called it.
"""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from mantle.routers import grants_router as grants
from mantle.services import grant_service
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext, check_access
from mantle.services import workspace_service as ws

from .test_there_is_no_promotion_state import OTHER, USER, real_node

WORKSPACE_CT = "application/vnd.agience.workspace+json"
INVITEE = "invitee@example.invalid"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


@pytest.fixture
def outbox(tmp_path, monkeypatch):
    monkeypatch.setenv("EMAIL_PROVIDER", "outbox")
    monkeypatch.setenv("EMAIL_OUTBOX_DIR", str(tmp_path / "outbox"))
    return tmp_path / "outbox"


def _mails(folder):
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(folder.glob("*.json"))]


def _auth(user):
    return AuthContext(principal_id=user, principal_type="user", user_id=user)


async def _invite(db, home):
    with acting_as(USER, principal_type="user"):
        return await grants.create_grant_endpoint(
            body=grants.CreateGrantRequest(resource_id=home, grantee_type="invite", role="viewer",
                                           target_entity=INVITEE, target_entity_type="email",
                                           max_claims=1, message="See you there"),
            auth=_auth(USER), store_db=db)


def _workspace(db):
    with acting_as(USER, principal_type="user"):
        return ws.create_container(db, USER, content_type=WORKSPACE_CT, name="Plans").id


async def test_an_invite_by_email_is_mailed_with_its_claim_link(db, outbox):
    made = await _invite(db, _workspace(db))
    assert made["email_sent"] is True
    (mail,) = _mails(outbox)
    assert mail["to"] == INVITEE
    assert made["claim_url"] in mail["text"]
    assert "See you there" in mail["text"]


async def test_without_mail_the_invite_says_it_was_not_sent(db, monkeypatch):
    monkeypatch.delenv("EMAIL_PROVIDER", raising=False)
    made = await _invite(db, _workspace(db))
    assert made["email_sent"] is False
    assert made["claim_url"]


async def test_a_resend_mails_a_new_link_and_the_old_one_stops_working(db, outbox):
    home = _workspace(db)
    made = await _invite(db, home)
    with acting_as(USER, principal_type="user"):
        again = await grants.resend_invite(grant_id=made["id"], auth=_auth(USER), store_db=db)
    assert again["email_sent"] is True and again["claim_token"] != made["claim_token"]
    assert again["claim_url"] in _mails(outbox)[-1]["text"]
    with pytest.raises(grant_service.InviteNotFound):
        grant_service.claim_invite(db, OTHER, made["claim_token"], INVITEE)
    grant_service.claim_invite(db, OTHER, again["claim_token"], INVITEE)
    with acting_as(OTHER, principal_type="user"):
        check_access(_auth(OTHER), home, "read", db)


async def test_only_its_granter_or_an_admin_may_resend_an_invite(db, outbox):
    made = await _invite(db, _workspace(db))
    with acting_as(OTHER, principal_type="user"), pytest.raises(HTTPException) as refused:
        await grants.resend_invite(grant_id=made["id"], auth=_auth(OTHER), store_db=db)
    assert refused.value.status_code == 404


async def test_the_member_is_listed_by_address_and_a_revoke_ends_their_access(db, outbox):
    home = _workspace(db)
    made = await _invite(db, home)
    member = grant_service.claim_invite(db, OTHER, made["claim_token"], INVITEE)
    with acting_as(USER, principal_type="user"):
        listed = await grants.list_grants_endpoint(resource_id=home, auth=_auth(USER), store_db=db)
    (row,) = [g for g in listed if g.get("grantee_id") == OTHER]
    assert row["name"] == INVITEE and row["state"] == "active"
    with acting_as(USER, principal_type="user"):
        await grants.revoke_grant(grant_id=member.id, auth=_auth(USER), store_db=db)
    with acting_as(OTHER, principal_type="user"), pytest.raises(HTTPException):
        check_access(_auth(OTHER), home, "read", db)
