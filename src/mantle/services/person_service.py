"""Mantle-side person service — thin HTTP shim to Origin.

Person records live in Origin's Postgres. Mantle only needs `get_user_by_id`
(used by `services.dependencies.get_person`); `record_person_event` is a no-op
stub kept so imports resolve — the auth flows that log person events live in
Origin.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from mantle.clients.origin_client import get_origin_client
from mantle.entities.person import Person as PersonEntity

logger = logging.getLogger(__name__)

#: The header carrying the SUBJECT'S OWN token alongside Mantle's service token.
#:
#: Spelled to match the two conventions Origin already has, rather than to introduce a third.
#: The NAME is `/internal/delegation-token`'s request field — `subject_token`
#: (`origin/routers/auth_router.py::_DelegationTokenRequest`), the route that already refuses a
#: bare `user_id` and takes the user's own verified token instead. The CASING is Origin's own for
#: a token that cannot ride in a body: `X-Setup-Token` (`origin/routers/setup_router.py`, read
#: with `Header(..., alias=...)`). `/internal/persons/{id}` is a GET and has no body, so the
#: field has to become a header; this is that field, spelled as a header, and nothing else.
SUBJECT_TOKEN_HEADER = "X-Subject-Token"


def get_user_by_id(db, id: str,  # noqa: A002 — keep `id` for compat
                   *, subject_token: Optional[str] = None) -> Optional[PersonEntity]:
    """Resolve a Person by ID via Origin's `/internal/persons/{id}` endpoint.

    `db` is unused — Origin owns identity. Kept in the signature so callers
    that pass a db handle don't need a special case.

    **`subject_token` is the caller's own verified bearer, and it is what narrows this call.**
    The service token says *"I am Mantle"* and can say nothing else — `peer_signing`
    signs `iss=sub=mantle` with no subject — so on its own it asks Origin "give me person X"
    with no statement that this node may read X, and nothing in the request records on whose
    behalf it asked. Presenting the subject's token alongside it makes the request say what it
    means: *this is X's own token, and X is who wants X's record.* It is the same constraint
    `/internal/delegation-token` already applies, which is why it is sent under that route's own
    field name (:data:`SUBJECT_TOKEN_HEADER`).

    Every caller sends one (review 21, F4), and a caller with none asks Origin nothing, so
    production requires it (`PERSON_LOOKUP_SUBJECT_TOKEN_REQUIRED`):

      * ``services/dependencies.get_person``: the person being read IS the authenticated caller
        (`auth.user_id`), so its bearer is by construction the subject's own token.
      * ``services/grant_service.send_invite_email`` and first-sign-in provisioning
        (``seed_provisioning/user_provisioning._resolve_profile``) take it from the request they
        run in (:func:`own_subject_token`), and skip the lookup without one.

    A token is never taken from anywhere but the call site's own request, and never minted for a
    background path.
    """
    del db
    if not id:
        return None
    client = get_origin_client()
    if not getattr(client, "_base", ""):  # noqa: SLF001
        # No Origin URI configured: a standalone node owns no person records to fetch.
        logger.debug("No ORIGIN_URI configured; get_user_by_id(%s) resolves to None", id)
        return None
    headers = {"Authorization": f"Bearer {client._service_token()}"}  # noqa: SLF001
    if subject_token:
        headers[SUBJECT_TOKEN_HEADER] = subject_token
    try:
        resp = client._client.get(  # noqa: SLF001
            f"{client._base}/internal/persons/{id}",
            headers=headers,
        )
    except httpx.HTTPError as exc:
        # An absent optional peer is an expected state on a standalone node, and this
        # runs per request: the message names the condition, the stack trace does not.
        logger.warning(
            "Origin at %s is unreachable in get_user_by_id(%s) (%s)",
            client._base, id, type(exc).__name__, exc_info=False,  # noqa: SLF001
        )
        return None
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        logger.warning("Origin /internal/persons/%s returned %d", id, resp.status_code)
        return None
    try:
        return PersonEntity.from_dict(resp.json() or {})
    except (ValueError, KeyError):
        return None


def _authority_issuer() -> str:
    from mantle import config
    return getattr(config, "AUTHORITY_ISSUER", "") or ""


def _claims_of(token: str) -> dict:
    """A token's claims, unverified: the signature was verified where it was presented, and
    Origin verifies it again before it answers."""
    import base64
    import json

    try:
        seg = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))
    except (IndexError, ValueError, TypeError):
        return {}
    return claims if isinstance(claims, dict) else {}


def own_subject_token(person_id: str) -> Optional[str]:
    """The person's own access token, when the request this runs in was made on it, else None
    (review 21, F4): the acting principal's forwarded bearer, if it is Origin's access token for
    `person_id` itself. A delegation, a grant key, a program's token or another person's is
    never one, and nothing is minted to stand in for it."""
    from mantle.services.acting_principal import current_acting_principal

    acting = current_acting_principal()
    held = getattr(getattr(acting, "credential", None), "token", "") if acting else ""
    if not held or not person_id or getattr(acting, "principal_id", None) != str(person_id):
        return None
    claims = _claims_of(held)
    if (not claims or claims.get("iss") != _authority_issuer()
            or str(claims.get("sub") or "") != str(person_id)
            or claims.get("principal_type", "user") != "user"
            or claims.get("act") or claims.get("token_type")):
        return None
    return held


async def record_person_event(payload: dict, event_type: str = "person") -> None:
    """No-op shim. Origin owns the event-logging webhook (`auth_service.record_person_event`);
    this stub exists only so imports of this module resolve."""
    del payload, event_type
    return None
