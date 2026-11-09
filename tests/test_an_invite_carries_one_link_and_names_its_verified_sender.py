"""An invite mail carries one link, its claim link, and says who sent it by their verified address.

2026-11-09: an account on my.agience.ai sent the owner a real Agience invite whose personal message
was phishing. What an inviter chooses (the personal message, their display name, the name of what
they share) reaches the invitee's inbox, so it arrives inert in subject, text and HTML: no URL-like
thing in it stays clickable, by the rules the website applies to lead mail and Origin to its own
invites (`mantle.services.defang`). The message is at most 500 characters. The invitee can tell who
sent it: the mail names the sender and their address, and only a person whose address the issuer
verified sends one.

`MAIL_CLIENT_LINK` is a pattern of the kind mail clients use to find links, so these tests check
what the mail says, not only the input's exact spelling. Each test goes through the route, the way
a person's request does; the mail lands in a local outbox and nothing touches the network.
"""
from __future__ import annotations

import html
import json
import re
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mantle.routers import grants_router as grants
from mantle.services import grant_service
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext

from .test_there_is_no_promotion_state import USER, real_node

WORKSPACE_CT = "application/vnd.agience.workspace+json"
INVITEE = "bob@example.invalid"
SENDER = "ann@team.test"
APP = "https://app.example"

_TLDS = ("com|net|org|io|ai|co|cafe|example|invalid|info|biz|us|uk|ca|de|ru|cn|app|dev|xyz|top|"
         "online|site|link|click|me|tv|ly|gl|test|рф|xn--[a-z0-9-]+")
_DOTS = "[.。．｡]"
MAIL_CLIENT_LINK = re.compile(
    r"(?<![\w\[\]])(?:https?|ftps?|file|javascript|vbscript|data|mailto)\s*:"
    r"|\bwww" + _DOTS +
    r"|[\w-]+(?:" + _DOTS + r"[\w-]+)*" + _DOTS + r"(?:" + _TLDS + r")\b"
    r"|\b\d{1,3}(?:\.\d{1,3}){3}\b"
    r"|\]\("
    r"|<a\s[^>]*href",
    re.IGNORECASE,
)

#: What an attacker types, and a fragment of what the mail must say instead.
PHISH = {
    "plain url": ("verify at https://evil.example/login", "hxxps://evil[.]example/login"),
    "markdown link": ("[Agience support](https://evil.example/x)",
                      "Agience support (hxxps://evil[.]example/x)"),
    "html anchor": ('<a href="http://evil.example/x">your account</a>',
                    "your account (hxxp://evil[.]example/x)"),
    "bare domain": ("go to agience-security.com now", "agience-security[.]com"),
    "idn": ("visit пример.рф", "пример[.]рф"),
    "lookalike": ("visit аgience.ai", "аgience[.]ai"),
    "ideographic dot": ("evil。example", "evil[.]example"),
    "ipv4": ("http://203.0.113.9/login", "hxxp://203[.]0[.]113[.]9/login"),
}


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


@pytest.fixture
def outbox(tmp_path, monkeypatch):
    monkeypatch.setenv("EMAIL_PROVIDER", "outbox")
    monkeypatch.setenv("EMAIL_OUTBOX_DIR", str(tmp_path / "outbox"))
    monkeypatch.setattr("mantle.config.FACET_URI", APP)
    return tmp_path / "outbox"


@pytest.fixture
def named(monkeypatch):
    """Set what the mail calls the inviter and the resource: `named(from_name=…, resource_name=…)`."""
    names = {"from_name": "Ann Lee", "resource_name": "Notes"}
    monkeypatch.setattr("mantle.services.person_service.own_subject_token", lambda pid: "own")
    monkeypatch.setattr("mantle.services.person_service.get_user_by_id",
                        lambda db, id, subject_token=None: SimpleNamespace(name=names["from_name"]))
    monkeypatch.setattr(grant_service, "_resource_name", lambda db, rid: names["resource_name"])
    return names.update


def _mails(folder):
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(folder.glob("*.json"))]


def _auth(email=SENDER, verified=True):
    return AuthContext(principal_id=USER, principal_type="user", user_id=USER,
                       email=email, email_verified=verified)


def _workspace(db):
    with acting_as(USER, principal_type="user"):
        return ws.create_container(db, USER, content_type=WORKSPACE_CT, name="Plans").id


async def _invite(db, message=None, auth=None):
    home = _workspace(db)
    with acting_as(USER, principal_type="user"):
        return await grants.create_grant_endpoint(
            body=grants.CreateGrantRequest(resource_id=home, grantee_type="invite", role="viewer",
                                           target_entity=INVITEE, target_entity_type="email",
                                           max_claims=1, message=message),
            auth=auth or _auth(), store_db=db)


def _links_outside(text: str, claim_url: str) -> list[str]:
    return [m.group(0) for m in MAIL_CLIENT_LINK.finditer(text.replace(claim_url, ""))]


def _invites_sent(db):
    return grant_service.list_invites_sent(db, USER, include_revoked=True)


@pytest.mark.parametrize("case", sorted(PHISH))
@pytest.mark.parametrize("field", ["message", "from_name", "resource_name"])
async def test_what_the_inviter_types_arrives_inert_in_every_part(db, outbox, named, field, case):
    typed, fragment = PHISH[case]
    if field != "message":
        named(**{field: typed})
    made = await _invite(db, message=typed if field == "message" else "See you there")
    (mail,) = _mails(outbox)
    claim = made["claim_url"]
    assert fragment in mail["text"]
    assert _links_outside(mail["subject"] + "\n" + mail["text"], claim) == []
    # The HTML says the same, escaped; nothing the inviter typed is markup or a link.
    assert html.escape(fragment, quote=False) in mail["html"]
    assert _links_outside(html.unescape(re.sub(r"<[^>]+>", " ", mail["html"])), claim) == []


async def test_the_invite_link_is_the_only_link(db, outbox, named):
    named(from_name='<a href="https://evil.example">Ann</a>',
          resource_name="[Notes](https://evil.example/n)")
    made = await _invite(db, message='<a href="https://evil.example/m">open</a> https://evil.example')
    (mail,) = _mails(outbox)
    claim = made["claim_url"]
    assert re.findall(r'href="([^"]*)"', mail["html"]) == [claim]
    assert mail["html"].count("<a ") == 1
    assert claim in mail["text"] and _links_outside(mail["text"], claim) == []


async def test_a_message_over_500_characters_is_refused_and_nothing_is_made(db, outbox, named):
    with pytest.raises(HTTPException) as refused:
        await _invite(db, message="x" * 501)
    assert refused.value.status_code == 400
    assert "500" in refused.value.detail
    assert _mails(outbox) == [] and _invites_sent(db) == []


async def test_a_message_of_500_characters_is_sent(db, outbox, named):
    made = await _invite(db, message="x" * 500)
    assert made["email_sent"] is True and "x" * 500 in _mails(outbox)[0]["text"]


async def test_the_mail_names_the_sender_and_their_verified_address(db, outbox, named):
    await _invite(db, message="See you there")
    (mail,) = _mails(outbox)
    assert "Ann Lee (ann@team[.]test) invited you to" in mail["text"]
    assert "Ann Lee (ann@team[.]test) invited you to" in mail["subject"]
    assert "Ann Lee (ann@team[.]test) invited you to" in mail["html"]


@pytest.mark.parametrize("email,verified", [(SENDER, False), (None, False), ("", True)])
async def test_an_inviter_without_a_verified_address_is_refused_before_anything_is_made(
        db, outbox, named, email, verified):
    with pytest.raises(HTTPException) as refused:
        await _invite(db, message="See you there", auth=_auth(email=email, verified=verified))
    assert refused.value.status_code == 403
    assert refused.value.detail == ("verify your email address first: an invite names its sender "
                                    "by their verified address")
    assert _mails(outbox) == [] and _invites_sent(db) == []


async def test_a_resend_names_the_sender_and_is_refused_to_an_unverified_one(db, outbox, named):
    made = await _invite(db)
    with acting_as(USER, principal_type="user"), pytest.raises(HTTPException) as refused:
        await grants.resend_invite(grant_id=made["id"], auth=_auth(verified=False), store_db=db)
    assert refused.value.status_code == 403 and len(_mails(outbox)) == 1
    with acting_as(USER, principal_type="user"):
        again = await grants.resend_invite(grant_id=made["id"], auth=_auth(), store_db=db)
    assert again["email_sent"] is True
    assert "Ann Lee (ann@team[.]test) invited you to" in _mails(outbox)[-1]["text"]
