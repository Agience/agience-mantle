"""Unified /events WebSocket.

Single bidirectional channel for all real-time artifact and operation events.
Clients can hold multiple subscriptions on one connection, each with its own
server-side filter.

## Two delivery classes, chosen per subscription

The bus offers two guarantees (see `event_bus`'s module docstring) and this endpoint exposes both,
selected by what the `subscribe` message carries — the class is a field a client sets, never
something it has to infer:

| subscribe carries         | class                       | guarantee |
|---------------------------|-----------------------------|-----------|
| nothing extra             | **live**                    | best-effort, at-most-once. No replay, no ack. Overflow is dropped and reported. |
| `since` and/or `durable`  | **durable**                 | at-least-once. Replays from the cursor, then continues live; the client acks by cursor. |

A durable subscription is backed by a `vnd.agience.subscription+json` artifact
(`entities/subscription.py`), so its cursor survives this socket, this process, and this node. The
client names it with `durable: "<artifact-id>"`; the server reads the cursor off the artifact,
replays what is owed, and writes the cursor back on `ack`. Because that artifact is authorized like
any other, a client may only resume a subscription it can `read` and may only ack one it can
`update`.

Redelivery is possible on the durable path and is part of the contract: a consumer that acted on an
event but crashed before acking sees it again. Durable consumers must be idempotent on `event_id`.

## Protocol

### Client → Server (JSON messages)

```
{"op": "subscribe",   "id": "<client-sub-id>", "filter": {...},
                      "since": "<cursor>", "durable": "<subscription-artifact-id>",
                      "overflow": "drop_oldest" | "drop_newest" | "disconnect"}
{"op": "unsubscribe", "id": "<client-sub-id>"}
{"op": "ack",         "id": "<client-sub-id>", "cursor": "<cursor>"}
{"op": "ping"}
```

Filter shape (all fields optional; empty filter matches every event the
caller is authorized to see):

```
{
  "container_id": "workspace-or-collection-id",
  "artifact_id":  "artifact-id",
  "content_type": "application/vnd.agience.operator+json",
  "event_names":  ["artifact.invoke.*", "artifact.created"]
}
```

A `container_id` filter also matches events on artifacts *inside* that container — see
`event_bus.containers_of`. Watching a container is watching its contents.

### Server → Client (JSON messages)

```
{"ack": "<client-sub-id>", "delivery": "best_effort"|"at_least_once", "cursor": "<cursor>"}
{"unack": "<client-sub-id>"}
{"acked": "<client-sub-id>", "cursor": "<cursor>"}
{"pong": true}
{"event": "<name>", "payload": {...}, "sub_id": "<client-sub-id>", "ts": 1712345678.9,
 "event_id": "abc", "cursor": "<origin:seq>", "replay": true}
{"dropped": 17, "sub_id": "<client-sub-id>", "overflow": "drop_oldest"}
{"replay_truncated": true, "sub_id": "<client-sub-id>", "cursor": "<origin:seq>"}
```

`cursor` is present on an event only when the durable log is configured; `replay: true` marks an
event served from the log rather than from the live feed. `replay_truncated` says a resume hit
this pass's ceiling and names the position it reached, so the client can reconnect from there.

An event payload describes an artifact; it never carries the artifact's body. `content` and
`content_encrypted` are stripped at the publish seam (`event_bus.redact_content`), so a consumer
that needs the bytes reads them back through the authorized artifact route, where decryption is
gated on key custody.

### Backpressure

A live subscription's queue is bounded and its overflow policy is explicit (default: drop the
oldest). When events are dropped the server sends a `dropped` notice with the running count rather
than failing silently — a client that sees one should either resynchronize or move to a durable
subscription. The publish path never blocks on a slow socket.

### Auth

Bearer token in the `Authorization` header (same pattern as the rest of mantle).
Browser clients that cannot set WS headers may pass `?access_token=...` as a
query parameter.

### ACL

Each delivery is filtered server-side by the authenticated principal's grants — on the replay path
exactly as on the live path, from the same `_event_visible_to`. Events whose subject the caller
cannot `read` are silently dropped.

The question asked is `services.dependencies.check_access(..., "read", ...)`, the same one every
HTTP route asks about the same artifact: deny-first, direct grant then root then the origin light
cone with its per-edge propagation prune. It is asked about the **artifact the event is about**,
because that is the artifact the descriptor describes. The event path holds no rule of its own —
no grant of its own to read, no second question to ask — so it cannot answer more widely than the
route that serves the artifact. That is README invariant #2 (*authorization is decided only by the
light cone and grants*) on this surface, and `_event_visible_to`'s only standing exception to it is
the SYSTEM change-feed consumer named in `_SYSTEM_EVENT_CONSUMERS`.

Verdicts are cached per socket and expire (`_ACCESS_TTL_SECONDS`), permissions and refusals alike,
so a revocation lands on an open connection instead of waiting for it to close; the token behind
the socket is re-resolved on the same principle (`_AUTH_TTL_SECONDS`).

Being the event's actor confers nothing, on a user or on any other principal. `actor_id` is a
provenance column, and a principal whose grants are revoked stops seeing events about the artifacts
it once touched. The one thing an actor match buys is a *re-ask*: a creation and the owner grant
that authorizes it are two transactions and the announcement is made between them, so its creator's
socket re-asks `check_access` for `_CREATION_GRACE_SECONDS` rather than caching a verdict about a
state no reader will ever observe. The answer is still `check_access`'s.

Fan-out delivers one event to many sockets; this check is what each socket independently applies to
it. The two stay separate deliberately: the delivery path computes no authority, so it cannot widen
any — see `event_bus`'s note on visibility attenuating while delivery amplifies.

This endpoint is the only real-time event surface on the platform; every
client subscribes through `/events` rather than a per-container SSE stream.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from mantle.events import event_bus
from mantle.db.doc_boundary import CREATED as _CREATED
from mantle.entities.grant import grant_is_deny
from mantle.services.acting_principal import acting_from_auth, set_acting_principal
from mantle.services.dependencies import get_store_db
from mantle.services.dependencies import (
    AuthContext,
    _check_grant_permission,
    check_access,
    get_auth,
    resolve_auth,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Events"])

# Platform services allowed to consume the change-feed as SYSTEM consumers — they
# observe every data-change event regardless of per-user ACL. Observing the feed
# is infrastructure (e.g. the content-type gateway driving event-driven
# describers); any ACTION taken in response still runs under a delegation rooted
# to the change's actor (a person). Keep this list tight.
#
# **This is a standing exception to README invariant #2** ("authorization is decided only by the
# light cone and grants"), and the only one on this surface. A SYSTEM consumer's authority comes
# from being the platform rather than from a grant, so it is named here — a list an operator can
# read — instead of being expressible as a wider answer to the ordinary question. Everything else
# on this path goes through `check_access`.
_SYSTEM_EVENT_CONSUMERS = {"crystal"}

#: How long one read verdict stands before the store is asked again.
#:
#: A cached authorization is a revocation that has not landed yet, so the cache holds refusals and
#: permissions alike and both expire. Without an expiry a socket learns about grants and never
#: about revokes — the asymmetry that lets a principal keep reading a feed after its authority is
#: gone. Short enough that a revoke takes effect while the socket is still open; long enough that
#: a busy feed is not one authorization walk per event.
_ACCESS_TTL_SECONDS = float(os.getenv("MANTLE_EVENTS_ACL_TTL_SECONDS") or 30.0)

#: How long the socket's authentication stands before the presented token is resolved again.
#:
#: A WebSocket outlives the request that opened it, so resolving the token once and closing over
#: the result for the connection's lifetime makes the socket outlive the token as well. The token
#: is re-resolved on this clock and a socket whose token no longer resolves is closed.
_AUTH_TTL_SECONDS = float(os.getenv("MANTLE_EVENTS_AUTH_TTL_SECONDS") or 60.0)

#: How long a refusal about a principal's own brand-new artifact is treated as provisional.
#:
#: Creating an artifact writes two things in two transactions: the row, and the owner grant that
#: authorizes its creator to read it. The announcement is made between them (the emit sits at the
#: persistence chokepoint, `db.doc_boundary.emit_artifact_change`), so a verdict taken at that
#: instant describes a state no reader will ever observe — and a change feed that answers from it
#: drops the creation of every artifact, silently, along with everything about that artifact until
#: the verdict expires.
#:
#: Within this window a refusal on a principal's own creation is re-asked rather than believed,
#: and never cached. What is bought is a re-ask: `check_access` still decides, so a creation whose
#: grant never lands stays unreadable and no other principal's creation is reached at all. Bounded
#: by the event's own age, so a relayed or long-queued creation is a settled refusal rather than a
#: race. Long enough to cover a write transaction, short enough that a socket which sees one is
#: not held up in any way a client would notice.
_CREATION_GRACE_SECONDS = float(os.getenv("MANTLE_EVENTS_CREATION_GRACE_SECONDS") or 2.0)

#: How often the creation grace re-asks. One authorization walk per interval, for one artifact, on
#: the one socket whose principal just created it.
_CREATION_RECHECK_SECONDS = float(os.getenv("MANTLE_EVENTS_CREATION_RECHECK_SECONDS") or 0.05)

#: Log rows one replay pass reads, and the ceiling on how many it reads in total.
#:
#: The ceiling is what makes "bounded per call" true rather than aspirational: a consumer resuming
#: from an empty cursor against a long log would otherwise drain the whole of it in one pass,
#: holding the send lock and the store the entire time. Past the ceiling the client is told where
#: replay stopped and can resume from there; live delivery is already attached, so a truncated
#: replay leaves a gap in the middle of the stream, never at its head.
_REPLAY_BATCH = 200
_REPLAY_MAX_EVENTS = int(os.getenv("MANTLE_EVENTS_REPLAY_MAX") or 5000)


# ---------------------------------------------------------------------------
# REST endpoint for servers to push events into the bus
# ---------------------------------------------------------------------------

# There is no inbound app-event relay endpoint here. Mantle emits its OWN
# data-change events from the write path (artifact.created/updated/deleted via
# collection_service → event_bus); the outbound WS /events subscribe stream
# below is the database change feed. App-to-app event relaying is an application
# concern, not the database layer.


# ---------------------------------------------------------------------------
# Per-subscription state
# ---------------------------------------------------------------------------

class _Subscription:
    __slots__ = ("client_id", "filter", "queue", "task", "live", "durable_id", "cursor",
                 "reported_drops")

    def __init__(
        self,
        client_id: str,
        event_filter: event_bus.EventFilter,
        live: event_bus.LiveSubscription,
        *,
        durable_id: Optional[str] = None,
        cursor: Optional[event_bus.Cursor] = None,
    ):
        self.client_id = client_id
        self.filter = event_filter
        self.live = live
        self.queue = live.queue
        self.task: Optional[asyncio.Task] = None
        #: The subscription artifact this socket is resuming, if any. Its presence is what makes
        #: this subscription durable — there is no separate flag to fall out of step with it.
        self.durable_id = durable_id
        #: How far this subscription has been served. Held per socket so the live pump can skip
        #: what replay already delivered; the acked position lives on the artifact, not here.
        self.cursor = cursor or event_bus.Cursor()
        self.reported_drops = 0

    @property
    def delivery(self) -> str:
        return event_bus.DELIVERY_DURABLE if self.durable_id or self.cursor.marks \
            else event_bus.DELIVERY_LIVE


def _parse_filter(raw: Any) -> event_bus.EventFilter:
    if not isinstance(raw, dict):
        return event_bus.EventFilter()

    event_names_raw = raw.get("event_names")
    event_names: Optional[List[str]] = None
    if isinstance(event_names_raw, list):
        event_names = [str(n) for n in event_names_raw if isinstance(n, str)]

    return event_bus.EventFilter(
        container_id=raw.get("container_id") or None,
        artifact_id=raw.get("artifact_id") or None,
        content_type=raw.get("content_type") or None,
        event_names=event_names,
    )


def _artifact_known(store_db: Any, resource_id: str) -> bool:
    """Does the store hold a row for this id?

    Asked before `check_access` rather than inferred from it, because `check_access` answers 404
    for "no such artifact" and for "you may not read it" alike — the non-oracle shape the routes
    want, and the one shape this path has to tell apart. An id with no row is an id `check_access`
    can say nothing about: it is not a refusal, it is the absence of a question.

    A read that fails answers `False`, so an unreachable store degrades toward the narrow branch
    rather than toward the wide one.
    """
    if not store_db or not resource_id:
        return False
    try:
        from mantle.db.backend import get_raw_artifact
        return get_raw_artifact(store_db, resource_id) is not None
    except Exception:
        logger.debug("event-path existence probe failed for %s", resource_id, exc_info=True)
        return False


def _read_authorized(auth: AuthContext, resource_id: str, store_db: Any) -> bool:
    """Does this principal hold `read` on this resource, right now?

    The question is `services.dependencies.check_access(..., "read", ...)` and nothing else. Every
    HTTP route that serves an artifact asks it there; asking it there here too is what stops the
    event path from being a second, weaker answer to the same question. Deny-first, the direct
    grant, then the artifact's root, then the origin light cone with its per-edge propagation
    prune — all of it borrowed whole, none of it restated.

    A refusal arrives as `HTTPException`: 404 for both "no grant" and "denied", which is the
    non-oracle shape the routes already use, and 500 for a lattice whose origin chain does not
    terminate. All of them read as False here, and so does any other failure — an authorization
    that could not be computed is not an authorization.
    """
    if not store_db or not resource_id:
        return False
    # Neither an identity to look grants up by nor grants already in hand: nothing to authorize
    # from. `check_access` would answer the same, at the cost of a store read per event.
    if not getattr(auth, "user_id", None) and not getattr(auth, "grants", None):
        return False
    try:
        check_access(auth, resource_id, "read", store_db)
        return True
    except HTTPException:
        return False
    except Exception:
        logger.debug("event-path read check failed for %s", resource_id, exc_info=True)
        return False


def _read_denied(auth: AuthContext, resource_id: str, store_db: Any) -> bool:
    """Is there an active `read` **deny** naming this resource for this principal?

    Deny-only, never allow, so this probe can widen nothing: the worst a bug in it can do is
    withhold an event.

    Its whole scope is the id `check_access` cannot be asked about — an artifact with no row, which
    is how a hard delete announces itself (the container context lives at the service, and by the
    time the event exists the doc it names is gone). Such an event is answered from its container,
    and this is what keeps that answer from being an alternative grant: a deny naming the vanished
    artifact still withholds it. Where the artifact IS in the store nothing calls this, because
    `check_access` already applies deny-first at every level of its own walk — the direct grant,
    the artifact's `root_id`, and every ancestor the light cone reaches.

    A grant-key principal is answered from the grants resolved at authentication — the same source
    `check_access` uses for it, so a bundle's deny members are honoured here too.
    """
    if not resource_id:
        return False
    grants = list(getattr(auth, "grants", None) or [])
    if grants:
        return any(grant_is_deny(g) and getattr(g, "can_read", False)
                   and str(getattr(g, "resource_id", "") or "") == str(resource_id)
                   for g in grants)
    if not store_db or not getattr(auth, "user_id", None):
        return False
    try:
        from mantle.db.backend import get_active_grants_for_principal_resource
        return any(grant_is_deny(g) and getattr(g, "can_read", False)
                   for g in get_active_grants_for_principal_resource(
                       store_db, grantee_id=auth.user_id, resource_id=resource_id))
    except Exception:
        logger.debug("event-path deny probe failed for %s", resource_id, exc_info=True)
        return False


def _event_visible_to(
    auth: AuthContext,
    event: event_bus.Event,
    access: Optional["_Access"] = None,
) -> bool:
    """ACL check: would the ordinary read path serve the artifact this event is about?

    One question, asked once, of one authority: `check_access(..., "read", ...)` behind `access`'s
    short-lived cache. There is no rule here — no grant read, no shortcut, no second subject to try
    when the first says no — which is what makes README invariant #2 true of this surface rather
    than merely intended for it. An event is visible exactly where its subject is readable.

    **The container is a fallback for an absent artifact, not an alternative grant.** It is
    consulted only when the store holds no row for `event.artifact_id` — which is how a hard delete
    arrives, announced after the doc it names is gone, and how an event that names no artifact at
    all arrives. There `check_access` has nothing to answer about, so the container is the only
    subject left; an explicit deny naming the vanished artifact still withholds it. Where the
    artifact IS in the store the container is never asked, because "may this principal read the
    container" is a **wider** question than "may it read the artifact" — a `propagate` mask that
    prunes `read` at the origin edge, or a deny on the artifact's root, is exactly the gap between
    the two.

    **There is deliberately no actor shortcut, for any principal.** "I wrote this" is not a grant:
    `actor_id` is the doc's `modified_by`/`created_by`, provenance columns that record who touched
    an artifact and never change when the authority to see it is withdrawn. A match earns a re-ask
    of this same question while a creation's owner grant is still landing (see
    `_races_its_own_authorization`) and never an answer to it.

    The SYSTEM change-feed consumer is the one standing exception — see `_SYSTEM_EVENT_CONSUMERS`.

    `access` is absent only for a caller with no store to ask (unit callers). Authorization lives
    in the store, so with no store there is no authorization: `False`.
    """
    # System change-feed consumer (the content-type gateway) — sees all events. The documented
    # exception to invariant #2, and the only branch here that is not `check_access`.
    if getattr(auth, "principal_type", None) == "service" and \
            getattr(auth, "authority", None) in _SYSTEM_EVENT_CONSUMERS:
        return True

    if access is None:
        return False

    subject = event.artifact_id
    if subject and access.known(subject):
        return access.may_read(subject)
    if subject and access.denies(subject):
        return False
    return access.may_read(event.container_id)


class _Access:
    """A socket's read verdicts, cached and expiring.

    Three facts about a resource behind one cache — whether the store holds it, whether this
    principal may read it, and whether it is explicitly denied — because they are asked about the
    same resources on the same events and a cache for one of them would leave the others running
    per event. The deny probe is filled lazily: it is needed only for an id the store does not
    hold, so the ordinary event never pays for it.

    Every entry expires, and refusals are cached as readily as permissions. A positive-only cache
    with no expiry is not a cache but a decision: it makes granting take effect live and revoking
    take effect never, for however long the socket stays open. `_ACCESS_TTL_SECONDS` is the whole
    of how far behind the store a verdict is allowed to be. Verdicts are read against
    `session.auth`, so a re-authentication that narrows the principal narrows what is checked from
    the next expiry onward.

    **A refusal about an id the store does not hold is not cached at all.** "You may not read this"
    and "there is nothing here to read" are different facts with different lifetimes: the first
    stands until a grant changes, the second stops being true the moment a row is written. Holding
    the second for a TTL is what turns one mistimed check into a blackout on an artifact that has
    since come into existence.
    """

    __slots__ = ("_session", "verdicts", "denials")

    def __init__(self, session: "_Session"):
        self._session = session
        #: resource_id -> (allowed, known, expires_at)
        self.verdicts: Dict[str, tuple] = {}
        #: resource_id -> (denied, expires_at); filled only for ids the store does not hold
        self.denials: Dict[str, tuple] = {}

    def _verdict(self, resource_id: str) -> tuple:
        now = time.monotonic()
        cached = self.verdicts.get(resource_id)
        if cached is not None and now < cached[2]:
            return cached
        auth, store_db = self._session.auth, self._session.store_db
        # Existence first: an id with no row is one `check_access` can only 404 on, so asking it
        # would spend a light-cone walk to learn what the seek already said.
        known = _artifact_known(store_db, resource_id)
        entry = (known and _read_authorized(auth, resource_id, store_db), known,
                 now + _ACCESS_TTL_SECONDS)
        if known:
            self.verdicts[resource_id] = entry
        else:
            self.verdicts.pop(resource_id, None)
        return entry

    def may_read(self, resource_id: str) -> bool:
        return bool(resource_id) and self._verdict(resource_id)[0]

    def known(self, resource_id: str) -> bool:
        return bool(resource_id) and self._verdict(resource_id)[1]

    def denies(self, resource_id: str) -> bool:
        if not resource_id:
            return False
        now = time.monotonic()
        cached = self.denials.get(resource_id)
        if cached is not None and now < cached[1]:
            return cached[0]
        entry = (_read_denied(self._session.auth, resource_id, self._session.store_db),
                 now + _ACCESS_TTL_SECONDS)
        self.denials[resource_id] = entry
        return entry[0]

    def invalidate(self, resource_id: str) -> None:
        """Drop what is cached about one resource, so the next question reaches the store.

        The re-ask half of the creation grace: a verdict taken mid-write is not evidence about the
        state that follows it, and re-asking a cache would only repeat the answer it is trying to
        get past.
        """
        if resource_id:
            self.verdicts.pop(resource_id, None)
            self.denials.pop(resource_id, None)


def _container_access(session: "_Session"):
    """The socket's `_Access`, and the verdict map behind it.

    One implementation shared by the live pump and the replay pass, because a replayed event must
    be visible on exactly the same terms as a live one. Two checks would be two chances to disagree
    about who may see what, and the disagreement would show up as a stored event being visible
    where its live twin was not.
    """
    access = _Access(session)
    return access.verdicts, access


def _races_its_own_authorization(auth: AuthContext, event: event_bus.Event) -> bool:
    """Could this refusal be a verdict taken between a creation and the grant that authorizes it?

    Three conditions, all narrow, because the widest thing a wrong `True` here can cost is a short
    delay on one subscription: the event announces a **creation**, its actor is **this principal**,
    and it is **young**. Together they describe exactly one situation — this socket's own principal
    has just created an artifact, and the owner grant is a second transaction that has not landed.
    Another principal's creation is a settled refusal (the grants that would reach it already
    exist), and a creation older than the grace was not written just now.

    This authorizes nothing. It says only that the question is worth asking again — `check_access`
    gives the same answer it would have given anyway, one transaction later.
    """
    if event.name != _CREATED:
        return False
    if time.time() - float(event.ts or 0.0) > _CREATION_GRACE_SECONDS:
        return False
    principal = getattr(auth, "user_id", None) or getattr(auth, "principal_id", None)
    return bool(principal and event.actor_id and str(principal) == str(event.actor_id))


async def _visible_after_grace(session: "_Session", event: event_bus.Event,
                               access: "_Access") -> bool:
    """Re-ask `_event_visible_to` until the grace runs out, against the store rather than the cache.

    Only the subject of this one event is invalidated, so the socket's other verdicts keep standing
    and a busy feed does not turn into one authorization walk per event. Reads `session.auth` each
    pass, like everything else here, so a principal narrowed mid-wait is the one asked about.
    """
    deadline = time.monotonic() + _CREATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        await asyncio.sleep(_CREATION_RECHECK_SECONDS)
        access.invalidate(event.artifact_id)
        access.invalidate(event.container_id)
        if _event_visible_to(session.auth, event, access):
            return True
    return False


def _event_message(sub: _Subscription, event: event_bus.Event, *,
                   replay: bool = False) -> Dict[str, Any]:
    """The wire form of one delivered event.

    `cursor` rides along on every event that has one, so a durable client can ack without knowing
    how the log names positions — it echoes back the last cursor it processed. That is what keeps
    the resume contract free of client-side arithmetic over `(origin, seq)`.
    """
    msg: Dict[str, Any] = {
        "event": event.name,
        "payload": event.payload,
        "sub_id": sub.client_id,
        "ts": event.ts,
        "event_id": event.event_id,
    }
    if event.origin and isinstance(event.seq, int):
        msg["cursor"] = str(event_bus.Cursor.of({event.origin: event.seq}))
    if replay:
        msg["replay"] = True
    return msg


def _already_served(sub: _Subscription, event: event_bus.Event) -> bool:
    """Has replay already delivered this event on this socket?

    Live subscription is established BEFORE replay reads the log, so the window between the two is
    covered rather than lost — at the cost of the overlap arriving twice. This is where the overlap
    is removed, exactly, using the same `(origin, seq)` the log ordered it by. An event with no
    proper time was never logged and therefore cannot be a duplicate of anything replayed.
    """
    if not event.origin or not isinstance(event.seq, int):
        return False
    return event.seq <= sub.cursor.to_dict().get(event.origin, 0)


class _Session:
    """The socket's authorization, revalidated on a clock rather than frozen at connect.

    Two things go stale on a connection that can stay open for days, and closing over the result
    of a single `resolve_auth` freezes both: the token, which expires, and the grants behind it,
    which are revoked. This holds the credential that opened the socket so the first can be
    re-resolved, and hands `auth` out through an attribute rather than by value so every reader
    sees the current principal instead of the one that connected.

    The second — revocation — is `_container_access`'s expiring verdict cache, which reads
    `self.auth` each time it refills.
    """

    __slots__ = ("token", "store_db", "auth", "_deadline")

    def __init__(self, token: str, auth: AuthContext, store_db: Any):
        self.token = token
        self.store_db = store_db
        self.auth = auth
        self._deadline = time.monotonic() + _AUTH_TTL_SECONDS

    def valid(self) -> bool:
        """Whether the socket's authentication still stands; `False` means it must be closed.

        Re-resolves the presented token once its lease has run out. A token that no longer
        resolves — expired, or issued by a key that has since been revoked — ends the connection
        rather than being tolerated until the client happens to disconnect.
        """
        if time.monotonic() < self._deadline:
            return True
        try:
            self.auth = resolve_auth(self.token, self.store_db, request=None)
        except Exception as exc:
            logger.info("events WS re-authentication failed; closing: %s", exc)
            return False
        self._deadline = time.monotonic() + _AUTH_TTL_SECONDS
        if self.auth.principal_id or self.auth.user_id:
            set_acting_principal(acting_from_auth(self.auth))
        return True


async def _pump_subscription(
    sub: _Subscription,
    ws: WebSocket,
    session: "_Session",
    send_lock: asyncio.Lock,
) -> None:
    """Forward events from a subscription's queue to the WebSocket.

    Applies the per-user ACL check before sending. Exits cleanly when the
    socket closes or the task is cancelled.

    Read verdicts are cached per WebSocket session and expire, so a container costs one
    authorization walk per TTL rather than one per event — and one per TTL rather than one ever.
    The socket's own authentication is rechecked on the same loop; a token that has expired ends
    delivery here rather than at the client's convenience.

    A refusal on this principal's own fresh creation is re-asked rather than believed, because the
    row and the grant that authorizes it are written in that order and the announcement sits
    between them (`_races_its_own_authorization`). The wait is this subscription's alone and it is
    taken before anything is sent, so delivery order is unaffected: what arrives meanwhile waits in
    the queue behind it, under the same bound and the same overflow policy as always.

    Backpressure surfaces here: the bus drops into the subscription's own counter, and this loop
    reports the running total to the client rather than letting the loss be invisible. Under the
    `disconnect` policy the subscription is closed instead, which is what that policy is for.
    """
    _verdicts, access = _container_access(session)

    async def _report_drops() -> bool:
        """Tell the client what it lost. Returns False when the subscription must be closed."""
        if sub.live.dropped == sub.reported_drops:
            return True
        sub.reported_drops = sub.live.dropped
        async with send_lock:
            try:
                await ws.send_json({"dropped": sub.live.dropped, "sub_id": sub.client_id,
                                    "overflow": sub.live.overflow})
            except Exception:
                return False
        return not sub.live.overflowed

    try:
        while True:
            event = await sub.queue.get()
            if ws.client_state != WebSocketState.CONNECTED:
                return
            if not session.valid():
                try:
                    await ws.close(code=4401, reason="Token no longer valid")
                except Exception:
                    pass
                return
            if not await _report_drops():
                return
            if _already_served(sub, event):
                continue
            if not _event_visible_to(session.auth, event, access):
                # A creation is announced from the persistence chokepoint, between the row write
                # and the owner grant. A refusal there describes a state that no longer holds by
                # the time it is acted on, so this principal's own fresh creation is re-asked
                # rather than dropped — the same question, once the transaction behind it lands.
                if not _races_its_own_authorization(session.auth, event):
                    continue
                if not await _visible_after_grace(session, event, access):
                    continue
                if ws.client_state != WebSocketState.CONNECTED:
                    return
            async with send_lock:
                try:
                    await ws.send_json(_event_message(sub, event))
                except Exception as exc:
                    logger.debug("events WS send failed (sub=%s): %s", sub.client_id, exc)
                    return
            if event.origin and isinstance(event.seq, int):
                sub.cursor = sub.cursor.advanced(event.origin, event.seq)
    except asyncio.CancelledError:
        return


async def _replay(
    sub: _Subscription,
    ws: WebSocket,
    session: "_Session",
    send_lock: asyncio.Lock,
) -> int:
    """Serve what the durable cursor is owed, oldest first. Returns the count sent.

    Bounded per call: at most `_REPLAY_MAX_EVENTS` events, read `_REPLAY_BATCH` at a time. Past
    the ceiling the client is sent a `replay_truncated` notice carrying the cursor replay reached,
    so it can resume from there — a consumer that has been away a very long time drains across
    several connections rather than pinning this one until the log runs out.

    **The same ACL runs here as on the live path**, from the same `_container_access`: a replayed
    event is not privileged by having been stored, and the log row is not evidence of anything but
    what happened. The creation grace the live pump applies has nothing to do here — it exists to
    cover the moment between a write and the grant that authorizes it, and by the time a row is
    read back out of the log that moment is long over, so the verdict is simply the answer. This is what a client-supplied `since` is answered by. The cursor names a
    position in the log, never an authority over it, so resuming from a position someone else
    reached reveals only what this caller may read anyway — which is nothing more than the live
    feed would have given it.
    """
    log = event_bus.event_log()
    if log is None:
        return 0
    _verdicts, access = _container_access(session)
    sent = 0
    served = 0
    while served < _REPLAY_MAX_EVENTS:
        try:
            batch = log.read_since(sub.cursor, event_filter=sub.filter,
                                   limit=min(_REPLAY_BATCH, _REPLAY_MAX_EVENTS - served))
        except Exception:
            logger.debug("durable replay read failed (sub=%s)", sub.client_id, exc_info=True)
            return sent
        if not batch:
            return sent
        for event in batch:
            if ws.client_state != WebSocketState.CONNECTED:
                return sent
            served += 1
            if _event_visible_to(session.auth, event, access):
                async with send_lock:
                    try:
                        await ws.send_json(_event_message(sub, event, replay=True))
                    except Exception:
                        return sent
                sent += 1
            if event.origin and isinstance(event.seq, int):
                sub.cursor = sub.cursor.advanced(event.origin, event.seq)

    async with send_lock:
        try:
            await ws.send_json({"replay_truncated": True, "sub_id": sub.client_id,
                                "cursor": str(sub.cursor)})
        except Exception:
            pass
    return sent


# ---------------------------------------------------------------------------
# Durable subscriptions — the cursor lives on an artifact, so it is authorized like one
# ---------------------------------------------------------------------------

def _may_touch_subscription(auth: AuthContext, subscription: Any, action: str) -> bool:
    """May this caller `read` / `update` its own subscription artifact?

    The subscription is an ordinary artifact, so this asks the ordinary question and asks it of the
    ordinary helper. Owner-or-grant: the creator of a subscription always reaches it, and anyone
    else needs a grant naming it — which is how a subscription can be shared to a second consumer
    without a second mechanism.
    """
    owner = getattr(subscription, "owner_id", None)
    user_id = getattr(auth, "user_id", None) or getattr(auth, "principal_id", None)
    if owner and user_id and str(owner) == str(user_id):
        return True
    grants = list(getattr(auth, "grants", None) or [])
    return bool(grants) and _check_grant_permission(
        grants, action, resource_id=getattr(subscription, "id", None))


def _load_durable(store_db: Any, subscription_id: str, auth: AuthContext) -> Optional[Any]:
    """The stored subscription, if the caller may read it. `None` otherwise.

    One `None` for absent and for unauthorized, deliberately: distinguishing them here would make
    this socket an oracle for the existence of subscriptions the caller cannot see, which is the
    same disclosure the artifact routes already refuse by 404-ing an unauthorized read.
    """
    if not store_db:
        return None
    try:
        from mantle.entities.subscription import load_subscription
        subscription = load_subscription(store_db, subscription_id)
    except Exception:
        logger.debug("durable subscription load failed (%s)", subscription_id, exc_info=True)
        return None
    if subscription is None or not _may_touch_subscription(auth, subscription, "read"):
        return None
    return subscription


def _persist_cursor(store_db: Any, sub: "_Subscription", cursor: Any,
                    auth: AuthContext) -> "event_bus.Cursor":
    """Write an acknowledged cursor back onto the subscription artifact.

    Acking is a write, so it is gated on `update`, not on the `read` that let the socket resume.
    The two are separate permissions on the artifact and this is where the difference bites: a
    consumer shared a read-only subscription may follow it, and may not move anyone else's
    position. The check is re-run against the stored artifact rather than trusted from subscribe
    time, since a grant can be revoked while a socket is open.

    Falls back to the socket's own served position when the client acks without naming one, since
    "everything you have sent me" is the common case and making the client echo a cursor it has
    just received would add a way to get it wrong.

    A failed persist returns the merged in-memory cursor rather than raising: the socket keeps
    working and continues from the right place, and the cost of the failure is redelivery after a
    reconnect — which at-least-once already permits.
    """
    merged = sub.cursor if cursor is None else event_bus.Cursor.parse(cursor)
    for origin, seq in sub.cursor.marks:
        merged = merged.advanced(origin, seq)
    try:
        from mantle.entities.subscription import advance_cursor, load_subscription
        stored = load_subscription(store_db, sub.durable_id)
        if stored is None or not _may_touch_subscription(auth, stored, "update"):
            return merged
        written = advance_cursor(store_db, sub.durable_id, merged)
        if written is not None:
            return written.cursor
    except Exception:
        logger.debug("durable cursor persist failed (%s)", sub.durable_id, exc_info=True)
    return merged


async def _authenticate_ws(ws: WebSocket) -> Optional["_Session"]:
    """Authenticate a WebSocket connection via Bearer header or ?access_token query param.

    Returns the `_Session` on success, or `None` if auth fails (the socket is closed with an
    appropriate code). The token is kept on the session rather than discarded, because this
    connection has to be able to ask again later — see `_Session.valid`.
    """
    token: Optional[str] = None
    authorization = ws.headers.get("authorization") or ""
    if authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "", 1).strip()
    if not token:
        token = ws.query_params.get("access_token")
    if not token:
        await ws.close(code=4401, reason="Missing bearer token")
        return None

    try:
        store_db = next(get_store_db())
    except Exception as exc:
        logger.error("events WS could not acquire db session: %s", exc)
        await ws.close(code=1011, reason="Database unavailable")
        return None

    try:
        auth = resolve_auth(token, store_db, request=None)
    except Exception as exc:
        logger.info("events WS auth rejected: %s", exc)
        await ws.close(code=4401, reason="Invalid token")
        return None

    # Publish the acting principal, exactly as `get_auth` does for HTTP routes.
    # This socket authenticates by calling `resolve_auth` DIRECTLY rather than
    # through `Depends(get_auth)`, so it does not inherit that wiring — without this
    # the connection is authenticated yet has no principal in scope, and anything it
    # does that touches artifact content or the index fails closed. Set on the
    # connection's own task, so it covers the whole socket lifetime including the
    # subscription pump spawned from here.
    if auth.principal_id or auth.user_id:
        set_acting_principal(acting_from_auth(auth))

    return _Session(token, auth, store_db)


@router.websocket("/events")
async def events_ws(websocket: WebSocket) -> None:
    """Unified bidirectional event stream.

    Accepts JSON messages matching the protocol documented at the top of
    this module.
    """
    session = await _authenticate_ws(websocket)
    if session is None:
        return
    store_db = session.store_db

    await websocket.accept()

    subscriptions: Dict[str, _Subscription] = {}
    send_lock = asyncio.Lock()

    async def close_all() -> None:
        for sub in list(subscriptions.values()):
            if sub.task is not None:
                sub.task.cancel()
            try:
                await event_bus.unsubscribe_filtered(sub.queue)
            except Exception:
                pass
        subscriptions.clear()

    try:
        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                continue

            op = message.get("op")

            if op == "ping":
                async with send_lock:
                    await websocket.send_json({"pong": True})
                continue

            if op == "subscribe":
                client_id = str(message.get("id") or "")
                if not client_id:
                    async with send_lock:
                        await websocket.send_json({"error": "subscribe requires id"})
                    continue
                if client_id in subscriptions:
                    async with send_lock:
                        await websocket.send_json({"error": f"sub {client_id} already exists"})
                    continue

                event_filter = _parse_filter(message.get("filter"))
                durable_id = str(message.get("durable") or "") or None
                cursor = event_bus.Cursor.parse(message.get("since"))

                # A named subscription artifact is authoritative for both filter and cursor: it is
                # the record that survived the disconnect, and honouring a client-supplied filter
                # over it would let a resume quietly widen what the stored subscription selects.
                if durable_id:
                    stored = _load_durable(store_db, durable_id, session.auth)
                    if stored is None:
                        async with send_lock:
                            await websocket.send_json(
                                {"error": f"durable subscription {durable_id} not readable"})
                        continue
                    event_filter = stored.filter
                    cursor = stored.cursor

                live = await event_bus.subscribe_live(
                    event_filter,
                    overflow=str(message.get("overflow") or event_bus.Overflow.DROP_OLDEST),
                )
                sub = _Subscription(client_id, event_filter, live,
                                    durable_id=durable_id, cursor=cursor)

                # Live first, then replay. The reverse order loses every event published between
                # the last log row read and the moment the subscriber attaches; this order
                # duplicates that window instead, and `_already_served` removes the duplicates
                # exactly, using the log's own ordering.
                #
                # A named subscription artifact has already been authorized as an artifact, above.
                # A bare `since` has not, and does not need to be: replay applies the same
                # per-event read check the live pump applies, from the same `_container_access`,
                # so a cursor selects a position and never an authority. What the client gets back
                # is what the live feed would have given it, only earlier.
                if durable_id or cursor.marks:
                    await _replay(sub, websocket, session, send_lock)

                sub.task = asyncio.create_task(
                    _pump_subscription(sub, websocket, session, send_lock)
                )
                subscriptions[client_id] = sub

                async with send_lock:
                    await websocket.send_json({"ack": client_id, "delivery": sub.delivery,
                                               "cursor": str(sub.cursor)})
                continue

            if op == "ack":
                client_id = str(message.get("id") or "")
                sub = subscriptions.get(client_id)
                if sub is None or not sub.durable_id:
                    async with send_lock:
                        await websocket.send_json(
                            {"error": f"ack requires a durable subscription; {client_id!r} is not one"})
                    continue
                # `session.auth`, not the connect-time value: an ack is a write, and a socket that
                # has outlived the grant behind it must not still be able to make one.
                acked = _persist_cursor(store_db, sub, message.get("cursor"), session.auth)
                async with send_lock:
                    await websocket.send_json({"acked": client_id, "cursor": str(acked)})
                continue

            if op == "unsubscribe":
                client_id = str(message.get("id") or "")
                sub = subscriptions.pop(client_id, None)
                if sub is not None:
                    if sub.task is not None:
                        sub.task.cancel()
                    try:
                        await event_bus.unsubscribe_filtered(sub.queue)
                    except Exception:
                        pass
                async with send_lock:
                    await websocket.send_json({"unack": client_id})
                continue

            async with send_lock:
                await websocket.send_json({"error": f"unknown op {op!r}"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("events WS errored: %s", exc)
    finally:
        await close_all()

