"""Mantle's mail: the invitation to a workspace or collection.

Providers, from ``EMAIL_PROVIDER``:

- ``gmail``: the Gmail API, as the sending mailbox ``PLATFORM_EMAIL_ADDRESS``, with
  ``GMAIL_OAUTH_CLIENT_ID`` / ``_CLIENT_SECRET`` / ``_REFRESH_TOKEN``.
- ``outbox``: sends nothing and writes each message as one JSON file in ``EMAIL_OUTBOX_DIR``, for
  a test stack to read.

Anything else, or a provider missing its settings, is not configured, and nothing is sent. The
message carries the headers receiving filters expect of real mail (``Date``, a ``Message-ID`` on
the sender's domain, ``Reply-To``); without them Outlook files platform mail as junk.
"""
from __future__ import annotations

import base64
import html
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_GMAIL_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"


def _provider() -> str:
    return (os.getenv("EMAIL_PROVIDER", "") or "").strip().lower()


def _gmail() -> dict:
    return {k: (os.getenv(f"GMAIL_OAUTH_{k.upper()}", "") or "").strip()
            for k in ("client_id", "client_secret", "refresh_token")}


def _sender() -> str:
    return (os.getenv("PLATFORM_EMAIL_ADDRESS", "") or "").strip()


def is_configured() -> bool:
    provider = _provider()
    if provider == "outbox":
        return bool((os.getenv("EMAIL_OUTBOX_DIR", "") or "").strip())
    if provider == "gmail":
        return bool(_sender()) and all(_gmail().values())
    return False


def _mime(to: str, subject: str, html_body: str, text: str) -> MIMEMultipart:
    sender = _sender()
    msg = MIMEMultipart("alternative")
    msg["From"] = formataddr((os.getenv("PLATFORM_EMAIL_NAME", "") or "Agience", sender))
    msg["To"] = to
    msg["Subject"] = subject
    msg["Reply-To"] = sender
    msg["Date"] = formatdate(localtime=False)
    msg["Message-ID"] = make_msgid(domain=sender.rpartition("@")[2] or None)
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    return msg


async def _send_gmail(to: str, subject: str, html_body: str, text: str) -> bool:
    gm = _gmail()
    async with httpx.AsyncClient() as client:
        token = await client.post(_GMAIL_TOKEN_URL, timeout=15, data={
            "client_id": gm["client_id"], "client_secret": gm["client_secret"],
            "refresh_token": gm["refresh_token"], "grant_type": "refresh_token"})
        token.raise_for_status()
        access = (token.json() or {}).get("access_token")
        if not access:
            raise ValueError("Gmail token exchange returned no access_token")
        raw = base64.urlsafe_b64encode(_mime(to, subject, html_body, text).as_bytes()).decode("ascii")
        sent = await client.post(_GMAIL_SEND_URL, json={"raw": raw}, timeout=30,
                                 headers={"Authorization": f"Bearer {access}"})
        sent.raise_for_status()
    return True


def _send_outbox(to: str, subject: str, html_body: str, text: str) -> bool:
    folder = (os.getenv("EMAIL_OUTBOX_DIR", "") or "").strip()
    os.makedirs(folder, exist_ok=True)
    now = datetime.now(timezone.utc)
    name = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{secrets.token_hex(4)}.json"
    tmp = os.path.join(folder, "." + name)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"to": to, "subject": subject, "text": text, "html": html_body,
                   "sent_at": now.isoformat()}, f)
    os.replace(tmp, os.path.join(folder, name))
    return True


async def send_email(to: str, subject: str, html_body: str, text: str) -> bool:
    """Send one message; False, logged, when mail is not configured or sending fails."""
    if not is_configured():
        logger.info("mail not configured; nothing sent")
        return False
    try:
        if _provider() == "outbox":
            return _send_outbox(to, subject, html_body, text)
        return await _send_gmail(to, subject, html_body, text)
    except Exception:
        logger.exception("sending mail via %s failed", _provider())
        return False


def invite_message(from_name: str, resource_name: str, claim_url: str,
                   message: Optional[str] = None) -> tuple[str, str, str]:
    """`(subject, html, text)` of an invitation. Every caller-supplied value is escaped."""
    subject = f"{from_name} invited you to {resource_name} on Agience"
    note = (message or "").strip()
    text = (f"{from_name} invited you to {resource_name} on Agience.\n"
            + (f'\n"{note}"\n' if note else "")
            + f"\nOpen this link to accept:\n{claim_url}\n\n"
            "If you do not have an account yet, you can create one when you open the link. "
            "If you were not expecting this, you can ignore it.\n")
    e = html.escape
    body = (f"<p>{e(from_name)} invited you to <strong>{e(resource_name)}</strong> on Agience.</p>"
            + (f"<blockquote>{e(note)}</blockquote>" if note else "")
            + f'<p><a href="{e(claim_url, quote=True)}">Accept the invitation</a></p>'
            + f"<p>Or open this link: {e(claim_url)}</p>"
            "<p>If you were not expecting this, you can ignore it.</p>")
    return subject, f"<!doctype html><html><body>{body}</body></html>", text


async def send_invite(to: str, from_name: str, resource_name: str, claim_url: str,
                      message: Optional[str] = None) -> bool:
    subject, html_body, text = invite_message(from_name, resource_name, claim_url, message)
    return await send_email(to, subject, html_body, text)
