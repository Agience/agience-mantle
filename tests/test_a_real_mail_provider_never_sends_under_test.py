"""The operator's shell may export a real mail sender (`EMAIL_PROVIDER=gmail`). Under pytest no
send reaches Gmail: the suite starts without it, and a send that still selects it raises."""
from __future__ import annotations

import asyncio
import os

import pytest

from mantle.services import email_service


def test_the_suite_starts_without_the_inherited_sender():
    for name in ("EMAIL_PROVIDER", "PLATFORM_EMAIL_ADDRESS", "GMAIL_OAUTH_REFRESH_TOKEN"):
        assert name not in os.environ, name


def test_gmail_under_test_raises(monkeypatch):
    monkeypatch.setenv("EMAIL_PROVIDER", "gmail")
    monkeypatch.setenv("PLATFORM_EMAIL_ADDRESS", "sender@example.com")
    for key in ("CLIENT_ID", "CLIENT_SECRET", "REFRESH_TOKEN"):
        monkeypatch.setenv(f"GMAIL_OAUTH_{key}", "x")
    reached = []
    monkeypatch.setattr(email_service, "_send_gmail", lambda *a, **k: reached.append(a) or True)
    with pytest.raises(RuntimeError, match="real mail under test"):
        asyncio.run(email_service.send_email("someone@example.com", "s", "<p>h</p>", "t"))
    assert not reached


def test_the_outbox_still_sends_under_test(monkeypatch, tmp_path):
    monkeypatch.setenv("EMAIL_PROVIDER", "outbox")
    monkeypatch.setenv("EMAIL_OUTBOX_DIR", str(tmp_path))
    assert asyncio.run(email_service.send_email("someone@example.com", "s", "<p>h</p>", "t"))
    assert list(tmp_path.iterdir())
