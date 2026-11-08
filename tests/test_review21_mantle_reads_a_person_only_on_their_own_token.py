"""Mantle reads a person record at Origin only on that person's own token (review 21, F4).

`GET /internal/persons/{id}` answered Mantle's service token for any person. Two callers sent no
subject token: the invite mail (the inviter's name for a salutation) and first-sign-in
provisioning (the new member's address and name). Each now sends the person's own access token
when the request it runs in carries one, and otherwise asks Origin nothing, so production can
require the subject token (`PERSON_LOOKUP_SUBJECT_TOKEN_REQUIRED`). A delegation or a grant key
is never sent as one.
"""
from __future__ import annotations

import pytest

from mantle.services import person_service
from mantle.services.acting_principal import ForwardedCredential, acting_as
from mantle.services.person_service import SUBJECT_TOKEN_HEADER

from .test_the_person_lookup_names_its_subject import SUBJECT, _Origin

OWN = "ey.the-person-s-own.token"


@pytest.fixture()
def origin(monkeypatch):
    o = _Origin()
    monkeypatch.setattr(person_service, "get_origin_client", lambda: o)
    monkeypatch.setattr(person_service, "_claims_of", lambda tok: {
        OWN: {"iss": person_service._authority_issuer(), "sub": SUBJECT},
        "ey.a-delegation": {"iss": person_service._authority_issuer(), "sub": SUBJECT,
                            "principal_type": "delegation", "act": {"sub": "agience-server-x"}},
    }.get(tok, {}))
    return o


def _profile():
    from mantle.services.seed_provisioning import user_provisioning
    return user_provisioning._resolve_profile(SUBJECT, None, None)


def test_provisioning_with_no_token_asks_origin_nothing(origin):
    assert _profile() == (None, None)
    assert origin.seen == []


def test_provisioning_on_the_person_s_own_token_sends_it(origin):
    with acting_as(SUBJECT, principal_type="user", credential=ForwardedCredential(OWN)):
        assert _profile() == ("u@example.com", "A Person")
    assert origin.seen[0]["headers"][SUBJECT_TOKEN_HEADER] == OWN


@pytest.mark.parametrize("token", ["ey.a-delegation", "agk_a-grant-key"])
def test_a_delegation_or_a_grant_key_is_never_sent(origin, token):
    with acting_as(SUBJECT, principal_type="user", credential=ForwardedCredential(token)):
        assert _profile() == (None, None)
    assert origin.seen == []


def test_another_person_s_token_is_never_sent(origin):
    with acting_as("someone-else", principal_type="user", credential=ForwardedCredential(OWN)):
        assert person_service.own_subject_token(SUBJECT) is None


def test_the_invite_mail_asks_origin_nothing_without_the_inviter_s_token(origin, monkeypatch):
    from mantle.services import email_service, grant_service

    sent = []

    async def _send(target, from_name, resource, url, message):
        sent.append(from_name)
        return True

    monkeypatch.setattr(email_service, "send_invite", _send)
    assert grant_service.send_invite_email(None, SUBJECT, "res-1", "t@x.test", "tok", None)
    assert origin.seen == [] and sent == ["Someone"]
    with acting_as(SUBJECT, principal_type="user", credential=ForwardedCredential(OWN)):
        grant_service.send_invite_email(None, SUBJECT, "res-1", "t@x.test", "tok", None)
    assert sent[-1] == "A Person"
    assert origin.seen[-1]["headers"][SUBJECT_TOKEN_HEADER] == OWN
