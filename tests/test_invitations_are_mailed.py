"""An invitation is mailed to the invitee, with a claim link to the hosted app.

2026-10-13, a tester: granting Viewer succeeded and no mail arrived, in Inbox or Spam, because the
invite path imported a mail service Mantle did not have; and the claim link pointed at localhost.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from mantle.services import email_service, grant_service


@pytest.fixture
def outbox(tmp_path, monkeypatch):
    monkeypatch.setenv("EMAIL_PROVIDER", "outbox")
    monkeypatch.setenv("EMAIL_OUTBOX_DIR", str(tmp_path / "outbox"))
    return tmp_path / "outbox"


def _mails(folder):
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(folder.glob("*.json"))]


def test_mail_is_configured_only_with_a_provider_and_its_settings(monkeypatch):
    for key in ("EMAIL_PROVIDER", "EMAIL_OUTBOX_DIR", "PLATFORM_EMAIL_ADDRESS",
                "GMAIL_OAUTH_CLIENT_ID", "GMAIL_OAUTH_CLIENT_SECRET", "GMAIL_OAUTH_REFRESH_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    assert not email_service.is_configured()
    monkeypatch.setenv("EMAIL_PROVIDER", "gmail")
    assert not email_service.is_configured(), "gmail without its credentials sends nothing"
    for key, value in (("PLATFORM_EMAIL_ADDRESS", "connect@example.invalid"),
                       ("GMAIL_OAUTH_CLIENT_ID", "id"), ("GMAIL_OAUTH_CLIENT_SECRET", "s"),
                       ("GMAIL_OAUTH_REFRESH_TOKEN", "r")):
        monkeypatch.setenv(key, value)
    assert email_service.is_configured()


def test_an_invitation_names_the_inviter_and_resource_and_escapes_them(outbox):
    ok = asyncio.run(email_service.send_invite(
        "invitee@example.invalid", "Ann <b>", "Q4 <plans>", "https://my.agience.ai/invite/tok", "see <you>"))
    assert ok
    (mail,) = _mails(outbox)
    assert mail["to"] == "invitee@example.invalid"
    assert mail["subject"] == "Ann <b> invited you to Q4 <plans> on Agience"
    assert "https://my.agience.ai/invite/tok" in mail["text"]
    assert "<b>" not in mail["html"] and "&lt;plans&gt;" in mail["html"] and "&lt;you&gt;" in mail["html"]


def test_the_invite_path_mails_the_claim_link_to_the_hosted_app(outbox, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("mantle.services.person_service.get_user_by_id",
                        lambda db, id: SimpleNamespace(name="Ann"))
    monkeypatch.setattr(grant_service, "_resource_name", lambda db, rid: "Shared Plans")
    monkeypatch.setattr("mantle.config.FACET_URI", "https://my.agience.ai")

    sent = grant_service.send_invite_email(None, "u-1", "ws-9", "invitee@example.invalid", "tok123", None)

    assert sent is True
    (mail,) = _mails(outbox)
    assert mail["subject"] == "Ann invited you to Shared Plans on Agience"
    assert "https://my.agience.ai/invite/tok123" in mail["text"]
