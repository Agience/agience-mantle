"""`GET {ORIGIN_URI}/internal/persons/{id}` says WHO is asking, not only WHICH service asks.

Mantle's service token is signed `iss=sub=mantle` and carries no subject — `peer_signing`'s own
words are that it "says Mantle is calling, and it can say nothing else". Presented alone at
Origin's person route it authenticates a machine and authorizes nothing: every enrolled platform
service can read every person's record, and the request records nobody on whose behalf it asked.

Origin already has the shape that fixes it, one route over. `POST /internal/delegation-token`
refuses a bare `user_id` and takes the user's own verified `subject_token`, so a service can only
act for a user whose token it actually holds. This file is Mantle's half: send that token, under
that name, wherever the call site has one — and stay working where it does not, because Origin's
requirement is gated off by default (`agience-chorus/src/ophan/server.py` is a live caller of the
same route that cannot be updated from here).

What is asserted, in the order it matters:

    the header is SENT, and it carries the subject's own bearer
    the header is NAMED as Origin names the field, checked against Origin's source
    the call still SUCCEEDS against an Origin that does not require it
    a caller with no subject token sends no header, and still succeeds
    a grant key is never forwarded as one
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mantle.services import person_service
from mantle.services.person_service import SUBJECT_TOKEN_HEADER

SUBJECT = "user-1"
SUBJECT_TOKEN = "ey.the-subjects-own.token"


class _Origin:
    """An Origin that answers, and remembers what it was asked with.

    `require_subject` is the GATE, modelled the way Origin will carry it: off by default, and a
    403 when it is on and the header is absent. Off is the configuration Mantle must not break.
    """

    _base = "http://origin.test"

    def __init__(self, *, require_subject: bool = False, person: dict | None = None):
        self.require_subject = require_subject
        self.person = person if person is not None else {
            "id": SUBJECT, "email": "u@example.com", "name": "A Person"}
        self.seen: list[dict] = []
        self._client = self

    def _service_token(self) -> str:
        return "mantle.service.jwt"

    # `_client.get`
    def get(self, url, headers=None, **_kw):
        headers = dict(headers or {})
        self.seen.append({"url": url, "headers": headers})

        class _Resp:
            def __init__(self, code, payload):
                self.status_code, self._payload = code, payload

            def json(self):
                return self._payload

        if self.require_subject and not headers.get(SUBJECT_TOKEN_HEADER):
            return _Resp(403, {"detail": "subject token required"})
        return _Resp(200, self.person)


@pytest.fixture()
def origin(monkeypatch):
    o = _Origin()
    monkeypatch.setattr(person_service, "get_origin_client", lambda: o)
    return o


# ---------------------------------------------------------------------------
# The header
# ---------------------------------------------------------------------------

def test_the_subjects_token_is_sent_alongside_the_service_token(origin):
    """Both, not either. The service token proves the caller is a platform service; the subject
    token is the only part of the request that says this node may read THIS person."""
    got = person_service.get_user_by_id(db=None, id=SUBJECT, subject_token=SUBJECT_TOKEN)

    assert got is not None and got.id == SUBJECT
    sent = origin.seen[0]["headers"]
    assert sent[SUBJECT_TOKEN_HEADER] == SUBJECT_TOKEN
    assert sent["Authorization"] == "Bearer mantle.service.jwt"
    assert origin.seen[0]["url"].endswith("/internal/persons/" + SUBJECT)


def test_no_subject_token_means_no_header_at_all(origin):
    """Absent, not empty. A header present and blank is a caller asserting it has a subject when
    it does not — the background paths have none, and they say so by saying nothing."""
    assert person_service.get_user_by_id(db=None, id=SUBJECT) is not None
    assert SUBJECT_TOKEN_HEADER not in origin.seen[0]["headers"]

    origin.seen.clear()
    assert person_service.get_user_by_id(db=None, id=SUBJECT, subject_token="") is not None
    assert SUBJECT_TOKEN_HEADER not in origin.seen[0]["headers"]


def test_the_header_is_named_as_origin_names_the_field():
    """One spelling of "here is the subject's token", derived rather than agreed.

    Read off Origin's own source: `_DelegationTokenRequest.subject_token` is the field, and
    `/internal/persons/{id}` is a GET with no body, so the field becomes a header under the
    casing Origin already uses for exactly that (`X-Setup-Token`). Two spellings is how these
    drift, and a copy of the string in this file would agree with itself forever.

    Skipped without the sibling checkout — this asserts a fact about a neighbour.
    """
    here = Path(__file__).resolve()
    candidates = [p / "agience-origin" / "src" / "origin" / "routers" / "auth_router.py"
                  for p in here.parents]
    src_path = next((c for c in candidates if c.exists()), None)
    if src_path is None:
        pytest.skip("no origin checkout alongside this one")
    src = src_path.read_text(encoding="utf-8")
    body = src.split("class _DelegationTokenRequest", 1)[1].split("\n@", 1)[0]
    field = next(ln.split(":", 1)[0].strip() for ln in body.splitlines()
                 if ln.strip().startswith("subject_token"))
    assert field == "subject_token"
    assert SUBJECT_TOKEN_HEADER == "X-" + field.replace("_", "-").title()


# ---------------------------------------------------------------------------
# The gate is off by default, and Mantle must not need it on
# ---------------------------------------------------------------------------

def test_the_call_still_succeeds_against_an_origin_that_does_not_require_it(origin):
    """Additive. An Origin that does not read this header ignores it, and the answer is the one
    Mantle has always got — which is what lets Mantle land ahead of Origin's gate, and lets
    Chorus keep calling the same route without one."""
    assert origin.require_subject is False
    assert person_service.get_user_by_id(db=None, id=SUBJECT, subject_token=SUBJECT_TOKEN) is not None
    assert person_service.get_user_by_id(db=None, id=SUBJECT) is not None
    assert len(origin.seen) == 2


def test_a_gated_origin_is_satisfied_by_the_header_and_refuses_without_it(monkeypatch):
    """The other side of the same switch: once Origin requires it, sending it is what keeps the
    request working. A refusal is reported as "no person", which every caller already handles —
    this lookup has never been allowed to raise into its callers."""
    gated = _Origin(require_subject=True)
    monkeypatch.setattr(person_service, "get_origin_client", lambda: gated)

    assert person_service.get_user_by_id(db=None, id=SUBJECT, subject_token=SUBJECT_TOKEN) is not None
    assert person_service.get_user_by_id(db=None, id=SUBJECT) is None


# ---------------------------------------------------------------------------
# The call site
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_person_forwards_the_bearer_it_was_authenticated_with(origin):
    """The one call site that HAS a subject token, driven end to end.

    `get_person` reads `auth.user_id` — the subject `resolve_auth` derived from this very token —
    so the bearer it forwards is the token of the person it is reading, by construction.
    """
    from mantle.services.dependencies import AuthContext, get_person

    person = await get_person(
        auth=AuthContext(principal_id=SUBJECT, principal_type="user", user_id=SUBJECT),
        store_db=None, token=SUBJECT_TOKEN)

    assert person.id == SUBJECT
    assert origin.seen[0]["headers"][SUBJECT_TOKEN_HEADER] == SUBJECT_TOKEN


@pytest.mark.asyncio
async def test_a_grant_key_is_never_forwarded_as_a_subject_token(origin):
    """`resolve_auth` gives a grant-key context no `user_id` ("a key is not a person"), so the
    401 lands before the lookup. An opaque `agk_` credential of this node's never leaves it as
    somebody's claimed identity, and Origin is never asked to verify one it cannot."""
    from fastapi import HTTPException

    from mantle.services.dependencies import AuthContext, get_person

    with pytest.raises(HTTPException) as caught:
        await get_person(
            auth=AuthContext(principal_id="grant-1", principal_type="grant_key", user_id=None),
            store_db=None, token="agk_secret")

    assert caught.value.status_code == 401
    assert origin.seen == []
