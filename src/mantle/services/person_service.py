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

    **Additive, and it must stay that way.** Origin's requirement is gated off by default —
    `agience-chorus/src/ophan/server.py::_resolve_person_email` is a live caller of this same
    route that cannot be updated from here, and it sends service headers only. So this is a
    header an ungated Origin ignores, never a precondition Mantle enforces locally: sending it
    cannot make a call fail that would otherwise have succeeded, and omitting it leaves the call
    exactly as it was.

    **Not every caller has one, and none of them invents one.** `None` is the honest value for a
    path with no request-scoped bearer, and the call is then the unscoped one it has always been:

      * ``services/dependencies.get_person`` HAS one. The person being read IS the authenticated
        caller (`auth.user_id`), so its bearer is by construction the subject's own token.
      * ``services/grant_service._send_invite_email`` does NOT. It runs several frames below the
        route, off the invite-creation path, and is handed a `user_id` rather than a request.
      * ``services/grant_service._verify_target_match`` does NOT, on its fallback arm only — and
        that arm is already the one the caller works to avoid, by passing the claimant's verified
        email off the token so no lookup happens at all.
      * ``services/seed_provisioning/user_provisioning`` does NOT. It is bootstrap, running under
        no user's request.

    A token is never taken from anywhere but the call site's own request. `acting_principal` is
    deliberately not a carrier for one ("this is deliberately not a token — it holds an
    already-verified chain"), and manufacturing a token for a background path would be inventing
    the very authority this parameter exists to stop asserting.
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


async def record_person_event(payload: dict, event_type: str = "person") -> None:
    """No-op shim. Origin owns the event-logging webhook (`auth_service.record_person_event`);
    this stub exists only so imports of this module resolve."""
    del payload, event_type
    return None
