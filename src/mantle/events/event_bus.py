"""The event bus — one change feed, two subscriber classes, one durable order.

Everything is an artifact and every artifact write deserves an event, so this is a product
surface rather than an internal notification helper. The emission point is the persistence
chokepoint (`db.doc_boundary.emit_artifact_change`), which is what makes CRUD coverage a property
of the write path rather than of anyone's discipline;
`tests/test_change_feed_is_complete_by_construction.py` holds that claim to the tree.

Delivery semantics, per subscriber class
----------------------------------------
The two classes are different products with different guarantees, and the distinction is in the
API rather than in a comment — a subscriber picks its class by which function it calls.

**Live — best-effort, at-most-once.** :func:`subscribe_filtered` / :func:`subscribe_live`.
An in-memory bounded queue, fed by fan-out. It gets what is published while it is attached, in
publication order, with no replay and no acknowledgement. A subscriber that cannot keep up loses
the overflow (see *Backpressure* below) and is told how much it lost. This is the right class for
a UI: a dropped intermediate state is invisible once the next one arrives, and a stalled browser
tab must never be able to stall the writer.

**Durable — at-least-once, cursor and ack.** :class:`EventLog` +
:mod:`mantle.entities.subscription`. Events are appended to a per-store append-only log ordered by
`(origin, seq)`; a subscriber reads forward from a cursor and advances it only after it has taken
responsibility for what it read. A disconnect resumes exactly where the cursor stands, so nothing
between disconnect and reconnect is lost. Redelivery is possible and expected — a consumer that
crashes after acting but before acking sees the event again — so durable consumers must be
idempotent on `event_id`. Exactly-once is not offered, because it cannot be offered honestly
across a process boundary this side does not control.

Non-CRUD events use the same two classes. :func:`publish_event` (async) and
:func:`emit_artifact_event_sync` (thread-safe) are the supported public seam for them; a caller
with something to announce that is not an artifact write publishes it here and it acquires the
log, the filters, the ACL and the back-plane for free.

Propagation — visibility attenuates, delivery fans out
------------------------------------------------------
These are two different operations on one event and conflating them would be a security bug.

* **Visibility** — *may this subscriber see this event* — **attenuates**. It composes along a path
  and can only ever narrow. It is computed with :mod:`mantle.attenuation`, the single
  implementation of the CRUDEASIO meet, via :func:`visibility_mask`; this module does not contain
  an intersection of its own and must never grow one. The mask that matters for an event is
  `read`: an event is visible where its subject is readable.
* **Delivery** — *one write notifies many subscribers* — **fans out**, which is the amplifying
  operation, the opposite of the meet. :func:`_fanout` performs it, and it is deliberately unable
  to reach the visibility mask: it hands each subscriber the same event and each recipient's ACL
  is applied on its own side, from its own grants. Nothing on the delivery path can widen what a
  recipient may see, because nothing on the delivery path computes what a recipient may see.

The failure this separation prevents is concrete: a fan-out step that "merged" the masks of the
subscribers it was serving would hand every one of them the union of their authorities.

**Child → container propagation.** An event on a child artifact is delivered to subscribers
watching its container as well as to those watching the artifact, because a container subscription
that missed writes to the things it contains would be useless for the tree view that is the bus's
first consumer. Which containers an event reaches is answered by :func:`containers_of`, and the
default answer is the artifact's immediate container — the one relationship every artifact doc
already carries. That function is a seam: when context edges become the propagation structure, a
resolver that walks them plugs in through :func:`set_container_resolver` with no change here and
no change at any call site. Propagation upward is *addressing*, not authorization — reaching a
container's subscribers does not grant them anything, since each delivery is still ACL-checked
against that subscriber's own grants.

Backpressure
------------
Explicit, bounded, and reported — not the fire-and-forget an unbounded queue inherits.

Each live subscription has a bounded queue and a stated :class:`Overflow` policy. The default is
:data:`Overflow.DROP_OLDEST`: on a full queue the oldest queued event is discarded, the new one is
enqueued, and `dropped` is incremented. Newest-wins is the correct default for a change feed
because the newest event is the one closest to current state, and the subscriber is told the count
so it can resynchronize or move to a durable subscription. :data:`Overflow.DROP_NEWEST` and
:data:`Overflow.DISCONNECT` are available for consumers with different needs;
`DISCONNECT` is the honest choice for a consumer that must not silently miss events but has not
taken a durable subscription.

What is *not* on the menu is blocking the publisher. The publish path never awaits a subscriber's
queue: one stalled WebSocket must not be able to stall the write path that fed it, and an
unbounded queue only converts that stall into unbounded memory.

Multi-process
-------------
Cross-worker distribution is :mod:`mantle.events.event_backplane`, and it is optional in the same sense
`remote=None` is optional for the tiered content store: a single worker with no back-plane is a
complete, supported configuration. More than one worker without one is refused at boot rather than
degraded — see `event_backplane.require_backplane_for_workers`.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mantle.attenuation import Mask, compose

logger = logging.getLogger(__name__)

__all__ = [
    "Event", "EventFilter", "Overflow", "LiveSubscription",
    "set_event_loop", "subscribe_filtered", "subscribe_live", "unsubscribe_filtered",
    "publish_event", "publish_event_sync", "emit_artifact_event_sync",
    "visibility_mask", "may_see", "containers_of", "set_container_resolver",
    "Cursor", "EventLog", "set_event_log", "event_log", "EVENT_LOG_DDL",
    "set_backplane", "backplane", "deliver_from_backplane",
    "DELIVERY_LIVE", "DELIVERY_DURABLE",
    "FEED_BODY_FIELDS", "redacted_artifact", "redact_content",
]

#: The two subscriber classes, named so a subscription artifact can record which it is and a
#: reader can tell which guarantee it holds without inferring it from the code path.
DELIVERY_LIVE = "best_effort"
DELIVERY_DURABLE = "at_least_once"

#: Default bound on a live subscription's queue. A bound rather than no bound, because the
#: unbounded case does not remove backpressure — it relocates it into memory, where it is
#: unobservable until the process dies. Overridable per subscription and by environment for an
#: operator whose subscribers are known to burst.
DEFAULT_QUEUE_MAX = int(os.getenv("MANTLE_EVENTS_QUEUE_MAX") or 1000)


# ---------------------------------------------------------------------------
# The feed carries descriptors, never bodies
# ---------------------------------------------------------------------------

#: The artifact fields the change feed never carries.
#:
#: Storage gets ciphertext: `db.doc_boundary.encrypt_artifact_content` envelope-encrypts an
#: artifact's inline body on the way into the store, and `decrypt_artifact_content` is the one
#: read chokepoint that opens it, under key custody. An event is neither of those. It travels to
#: every subscriber the filter selects, and — where a durable log is installed — into an
#: `event_log` row that no ACL covers, in the same file whose `artifacts.content` column is
#: encrypted. A body on that path is a body outside both controls.
#:
#: So the feed says *what changed*, and a subscriber that needs the bytes reads them back through
#: the authorized path, where decryption is gated on custody. `content_encrypted` goes with
#: `content` because a flag describing an absent body is worse than no flag: a consumer that
#: round-trips the descriptor into a write would hand `encrypt_artifact_content` an empty body
#: already marked encrypted.
FEED_BODY_FIELDS = ("content", "content_encrypted")


def redacted_artifact(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """One artifact descriptor with its body removed — the form the feed carries.

    Identity, container, state, provenance and content type all survive, because those are what a
    subscriber acts on. Only :data:`FEED_BODY_FIELDS` is dropped.

    A copy, never an in-place edit: the caller is usually holding the entity dict of a live write,
    and the announcement must not be able to damage the thing it announces.
    """
    return {k: v for k, v in dict(doc).items() if k not in FEED_BODY_FIELDS}


def redact_content(payload: Any) -> Any:
    """An event payload with every artifact body removed.

    The enforcement point, applied at publish rather than at each emit site. Emits arrive from the
    persistence boundary, from the services, and from the shard tools; a rule enforced at those
    sites is a rule the next site can forget, and the durable log — which is fed here — would keep
    the omission forever.

    Returns the payload unchanged (the same object) when it carries no body, so the ordinary event
    costs one membership test rather than a copy on the hottest write in the system.
    """
    if not isinstance(payload, Mapping):
        return payload

    artifact = payload.get("artifact")
    artifacts = payload.get("artifacts")
    carries_body = any(field in payload for field in FEED_BODY_FIELDS)
    if isinstance(artifact, Mapping):
        carries_body = carries_body or any(f in artifact for f in FEED_BODY_FIELDS)
    if isinstance(artifacts, (list, tuple)):
        carries_body = carries_body or any(
            isinstance(a, Mapping) and any(f in a for f in FEED_BODY_FIELDS) for a in artifacts)
    if not carries_body:
        return payload

    out = {k: v for k, v in payload.items() if k not in FEED_BODY_FIELDS}
    if isinstance(artifact, Mapping):
        out["artifact"] = redacted_artifact(artifact)
    if isinstance(artifacts, (list, tuple)):
        out["artifacts"] = [redacted_artifact(a) if isinstance(a, Mapping) else a
                            for a in artifacts]
    return out


# ---------------------------------------------------------------------------
# Unified event model
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """A single event broadcast through the bus.

    `name` is the dotted event identifier (e.g. `artifact.created`,
    `artifact.invoke.completed`). `payload` is the free-form JSON body. The other fields are used
    for server-side filter evaluation and for the ACL check the delivery side performs.

    `origin` / `seq` are the durable log's proper-time coordinates, stamped by :meth:`EventLog.append`
    and absent on an event that was never logged. They carry the same meaning as the lattice's
    `(_origin, _seq)`: `origin` is the authoring observer and `seq` is that observer's monotonic,
    gap-free proper time, so `seq > cursor` is a correct resume with no tie-breaking.
    """

    name: str
    payload: Dict[str, Any]
    container_id: Optional[str] = None
    artifact_id: Optional[str] = None
    content_type: Optional[str] = None
    actor_id: Optional[str] = None
    ts: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    #: Every container this event addresses, immediate container first. Resolved at publish time
    #: by `containers_of`; see the module docstring on child → container propagation.
    containers: Tuple[str, ...] = ()
    origin: Optional[str] = None
    seq: Optional[int] = None

    # -- back-plane wire form ---------------------------------------------------------

    def to_wire(self) -> Dict[str, Any]:
        """The JSON-safe dict a back-plane carries. Carries no authority — see
        `event_backplane`: every receiving process re-derives visibility from its own grants."""
        return {
            "name": self.name, "payload": self.payload,
            "container_id": self.container_id, "artifact_id": self.artifact_id,
            "content_type": self.content_type, "actor_id": self.actor_id,
            "ts": self.ts, "event_id": self.event_id,
            "containers": list(self.containers),
            "origin": self.origin, "seq": self.seq,
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> "Event":
        """Rebuild from a back-plane message. Unknown keys are ignored and missing ones take their
        defaults, so a peer running a different build cannot break this one's fan-out.

        The payload is redacted again on arrival. The publishing process already did it, but "a
        peer running a different build" is the whole reason this decoder is forgiving, and a peer
        that predates the rule must not be able to inject a body into this node's subscribers.
        """
        payload = wire.get("payload")
        return cls(
            name=str(wire.get("name") or ""),
            payload=redact_content(payload) if isinstance(payload, dict) else {},
            container_id=wire.get("container_id"),
            artifact_id=wire.get("artifact_id"),
            content_type=wire.get("content_type"),
            actor_id=wire.get("actor_id"),
            ts=float(wire.get("ts") or time.time()),
            event_id=str(wire.get("event_id") or uuid.uuid4().hex),
            containers=tuple(str(c) for c in (wire.get("containers") or []) if c),
            origin=wire.get("origin"),
            seq=wire.get("seq") if isinstance(wire.get("seq"), int) else None,
        )


@dataclass
class EventFilter:
    """Server-side filter applied to events before they reach a subscriber.

    An event matches if every provided field matches. Empty fields match anything. `event_names`
    supports fnmatch globs (e.g. `artifact.invoke.*`).

    A filter is a *selection*, never an authorization. Matching decides what a subscriber asked
    for; `routers/events_router._event_visible_to` decides what it may have. A wide-open filter is
    therefore harmless — it selects everything and receives only what the caller's grants reach.
    """

    container_id: Optional[str] = None
    artifact_id: Optional[str] = None
    content_type: Optional[str] = None
    event_names: Optional[List[str]] = None

    def matches(self, event: Event) -> bool:
        if self.container_id:
            # Container match is the propagation rule in one line: an event matches a container
            # subscription if it is addressed to that container directly OR reaches it through
            # `containers_of`. Watching a container therefore covers writes to what it contains.
            if self.container_id != event.container_id and \
                    self.container_id not in event.containers:
                return False
        if self.artifact_id and self.artifact_id != event.artifact_id:
            return False
        if self.content_type and self.content_type != event.content_type:
            return False
        if self.event_names:
            if not any(fnmatch.fnmatchcase(event.name, pat) for pat in self.event_names):
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """The wire/stored form — only the fields that narrow, so a stored filter round-trips
        without acquiring explicit nulls that read as deliberate."""
        out: Dict[str, Any] = {}
        for key in ("container_id", "artifact_id", "content_type"):
            value = getattr(self, key)
            if value:
                out[key] = value
        if self.event_names:
            out["event_names"] = list(self.event_names)
        return out

    @classmethod
    def from_dict(cls, raw: Any) -> "EventFilter":
        """Decode a stored or client-supplied filter. A non-dict is the empty filter rather than an
        error: the empty filter is safe (it selects everything and authorizes nothing), so
        degrading to it never widens anything."""
        if not isinstance(raw, Mapping):
            return cls()
        names_raw = raw.get("event_names")
        names = [str(n) for n in names_raw if isinstance(n, str)] \
            if isinstance(names_raw, (list, tuple)) else None
        return cls(
            container_id=raw.get("container_id") or None,
            artifact_id=raw.get("artifact_id") or None,
            content_type=raw.get("content_type") or None,
            event_names=names or None,
        )


# ---------------------------------------------------------------------------
# Visibility — attenuation, borrowed whole from the one operator
# ---------------------------------------------------------------------------

def visibility_mask(edge_masks: Iterable[Any]) -> Mask:
    """The authority an event's subject carries along a path of edges.

    A thin, deliberate forwarder onto `attenuation.compose`. It exists so that "event visibility is
    attenuation" is a call rather than a claim, and so that the guard against a second intersection
    of permission bits has exactly one site to point at in this module. The empty path is `TOP`,
    which is the identity: a zero-hop walk narrows nothing.
    """
    return compose(Mask.from_propagate(m) for m in edge_masks)


def may_see(mask: Mask) -> bool:
    """Does this composed authority let its holder see an event about the subject?

    `read`, and only `read`. Nothing on the event path invents a verb of its own: an event is
    visible exactly where its subject is readable, which keeps the change feed from becoming a
    side channel that answers questions the ordinary read path would refuse.
    """
    return mask.allows("read")


# ---------------------------------------------------------------------------
# Propagation — which containers an event addresses
# ---------------------------------------------------------------------------

_container_resolver: Optional[Callable[[Event], Sequence[str]]] = None


def set_container_resolver(resolver: Optional[Callable[[Event], Sequence[str]]]) -> None:
    """Install the walk that decides which containers an event reaches.

    The seam exists so container propagation can become a traversal of the context lattice without
    this module learning about edges. A resolver receives the event and returns container ids,
    nearest first; returning `()` means the event addresses no container. Passing `None` restores
    the default — the artifact's immediate container, which is the one containment fact every
    artifact doc already carries and therefore the answer that needs no graph read.
    """
    global _container_resolver
    _container_resolver = resolver


def containers_of(event: Event) -> Tuple[str, ...]:
    """Every container this event is addressed to, nearest first, deduplicated.

    A resolver that raises is not allowed to fail the publish: propagation is addressing, and an
    event delivered to fewer containers than it might have reached is a smaller event, never a
    wider one. The fallback is the immediate container, so the pre-resolver behaviour is exactly
    what a failed walk degrades to.
    """
    direct = (event.container_id,) if event.container_id else ()
    if _container_resolver is None:
        return direct
    try:
        resolved = tuple(str(c) for c in (_container_resolver(event) or ()) if c)
    except Exception:
        logger.debug("container resolver failed for %s; addressing the immediate container only",
                     event.name, exc_info=True)
        return direct
    seen: Dict[str, None] = {}
    for cid in direct + resolved:
        seen.setdefault(cid, None)
    return tuple(seen)


# ---------------------------------------------------------------------------
# Live subscriptions — bounded, with a stated overflow policy
# ---------------------------------------------------------------------------

class Overflow:
    """What happens to a live subscriber that cannot keep up. Stated, never inherited."""

    #: Discard the oldest queued event and enqueue the new one. The default: on a change feed the
    #: newest event is the one closest to current state, so an overwhelmed subscriber should fall
    #: behind at the tail rather than at the head.
    DROP_OLDEST = "drop_oldest"
    #: Discard the arriving event and keep the backlog intact. For a consumer whose backlog is a
    #: work queue rather than a state feed.
    DROP_NEWEST = "drop_newest"
    #: Stop delivering and mark the subscription overflowed, so the transport can close it. The
    #: honest answer for a consumer that must not silently miss events but holds no cursor.
    DISCONNECT = "disconnect"

    ALL = (DROP_OLDEST, DROP_NEWEST, DISCONNECT)


@dataclass
class LiveSubscription:
    """One attached best-effort subscriber: its filter, its bounded queue, and its loss account.

    `dropped` and `overflowed` are the reportable half of the backpressure policy. A policy that
    dropped silently would be indistinguishable from a bug in the filter, so the count is kept per
    subscription and the transport is expected to surface it — `routers/events_router` sends it to
    the client, which is how a subscriber learns it should take a durable subscription instead.
    """

    filter: EventFilter
    queue: "asyncio.Queue[Event]"
    overflow: str = Overflow.DROP_OLDEST
    dropped: int = 0
    overflowed: bool = False

    def offer(self, event: Event) -> bool:
        """Enqueue *event* under the overflow policy. Never blocks, never awaits.

        Returns whether the event was enqueued. The publish path ignores the answer — it is here
        for tests and for a transport that wants to react to the first loss rather than to the
        counter.
        """
        if self.overflowed:
            return False
        try:
            self.queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            pass
        if self.overflow == Overflow.DROP_NEWEST:
            self.dropped += 1
            return False
        if self.overflow == Overflow.DISCONNECT:
            self.dropped += 1
            self.overflowed = True
            return False
        try:                                   # DROP_OLDEST: make room at the head
            self.queue.get_nowait()
            self.dropped += 1
        except asyncio.QueueEmpty:             # drained between the two calls — room now exists
            pass
        try:
            self.queue.put_nowait(event)
            return True
        except asyncio.QueueFull:              # another producer refilled it; the drop stands
            self.dropped += 1
            return False


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

#: All active live subscriptions (global list; filter evaluated per event).
_filtered_subscribers: list[LiveSubscription] = []

_loop: Optional[asyncio.AbstractEventLoop] = None

_log: Optional["EventLog"] = None

_backplane: Optional[Any] = None


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Store the running event loop so sync callers can schedule coroutines."""
    global _loop
    _loop = loop


# ---------------------------------------------------------------------------
# Subscription API
# ---------------------------------------------------------------------------

async def subscribe_live(event_filter: EventFilter, *,
                         maxsize: int = DEFAULT_QUEUE_MAX,
                         overflow: str = Overflow.DROP_OLDEST) -> LiveSubscription:
    """Attach a best-effort subscriber and return its handle.

    The handle rather than the bare queue, because the guarantee this class offers is only half
    describable by a queue: the other half is how much it lost, and a caller that cannot see
    `dropped` cannot tell a quiet feed from a discarded one.
    """
    if overflow not in Overflow.ALL:
        raise ValueError(f"unknown overflow policy {overflow!r}; expected one of {Overflow.ALL}")
    sub = LiveSubscription(filter=event_filter,
                           queue=asyncio.Queue(maxsize=max(1, int(maxsize))),
                           overflow=overflow)
    _filtered_subscribers.append(sub)
    return sub


async def subscribe_filtered(event_filter: EventFilter, *,
                             maxsize: int = DEFAULT_QUEUE_MAX,
                             overflow: str = Overflow.DROP_OLDEST) -> "asyncio.Queue[Event]":
    """Register a best-effort subscriber. Returns a queue of `Event` objects.

    The queue-only form, kept because it is what most call sites want. It is the same subscription
    :func:`subscribe_live` creates — reach for that one when the drop count matters.
    """
    sub = await subscribe_live(event_filter, maxsize=maxsize, overflow=overflow)
    return sub.queue


async def unsubscribe_filtered(q: "asyncio.Queue[Event]") -> None:
    """Remove a subscriber by its queue. Safe if already removed."""
    global _filtered_subscribers
    _filtered_subscribers = [s for s in _filtered_subscribers if s.queue is not q]


def subscription_for(q: "asyncio.Queue[Event]") -> Optional[LiveSubscription]:
    """The handle behind a queue, for a caller holding only the queue form."""
    return next((s for s in _filtered_subscribers if s.queue is q), None)


# ---------------------------------------------------------------------------
# Publish API — the public seam for non-CRUD events
# ---------------------------------------------------------------------------

async def publish_event(event: Event) -> None:
    """Publish an `Event`: address it, log it, distribute it, fan it out.

    The order is deliberate. Addressing and logging happen before any subscriber is touched, so a
    durable subscriber's cursor covers an event even if every live subscriber's queue is full; and
    the back-plane is fed before the local fan-out so cross-worker delivery is not gated on local
    consumers.

    This is the supported entry point for events the write chokepoint does not produce. A caller
    with something to announce publishes it here and inherits the filters, the log, the ACL and
    the back-plane; there is no second, lighter way in, because a second way in is how a feed
    stops being complete.

    Redaction happens first, before anything can observe the event: the log row, the back-plane
    message and every live subscriber then see the same descriptor, and none of the three can be
    the one that carries a body. Being the single way in is exactly what lets this be the single
    place the rule is enforced — see :func:`redact_content`.
    """
    event.payload = redact_content(event.payload)
    if not event.containers:
        event.containers = containers_of(event)
    _append_to_log(event)
    _to_backplane(event)
    await _fanout(event)


def publish_event_sync(event: Event) -> None:
    """Thread-safe `publish_event` for synchronous callers.

    Schedules the coroutine on the stored event loop; a no-op if the loop is not yet available
    (early bootstrap). Sync callers are the write path, which must never block on the bus.
    """
    if _loop is None or _loop.is_closed():
        return
    asyncio.run_coroutine_threadsafe(publish_event(event), _loop)


def _extract_artifact_fields(data: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Best-effort extraction of (artifact_id, content_type) from a service-layer
    payload of shape `{artifact: {...}}` or `{artifact_id: "..."}`."""
    artifact_obj = data.get("artifact")
    artifact_id: Optional[str] = None
    content_type: Optional[str] = None
    if isinstance(artifact_obj, dict):
        artifact_id = artifact_obj.get("id") or artifact_obj.get("_key")
        ctx = artifact_obj.get("context")
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except Exception:
                ctx = None
        if isinstance(ctx, dict):
            content_type = ctx.get("content_type")
        if content_type is None:
            content_type = artifact_obj.get("content_type")
    if artifact_id is None:
        artifact_id = data.get("artifact_id")
    return artifact_id, content_type


def emit_artifact_event_sync(
    container_id: str,
    event_name: str,
    data: Dict[str, Any],
    *,
    actor_id: Optional[str] = None,
) -> None:
    """Convenience helper for service-layer and write-path code.

    Constructs an `Event` with extracted artifact_id / content_type and publishes it through the
    thread-safe path. Together with :func:`publish_event` this is the public seam: both are
    supported API, both are covered by `tests/test_event_bus_semantics.py` § "Direct emit is the
    public seam", and neither is a shortcut around anything — they are the same publish.
    """
    artifact_id, content_type = _extract_artifact_fields(data)
    event = Event(
        name=event_name,
        payload=data,
        container_id=container_id,
        artifact_id=artifact_id,
        content_type=content_type,
        actor_id=actor_id,
    )
    publish_event_sync(event)


# ---------------------------------------------------------------------------
# Fan-out — the amplifying half, kept away from the attenuating one
# ---------------------------------------------------------------------------

async def _fanout(event: Event) -> None:
    """Offer the event to every live subscription whose filter selects it.

    One event in, many subscribers out — the amplifying operation. It computes no authority and
    holds no mask: each recipient's transport applies that recipient's own ACL before the event
    leaves the process. Keeping the two apart is what stops fan-out from being a path by which one
    subscriber's authority reaches another.

    Never awaits a subscriber's queue. `offer` is non-blocking by construction, so a stalled
    consumer costs itself its overflow and costs the publisher nothing.
    """
    for sub in list(_filtered_subscribers):
        if not sub.filter.matches(event):
            continue
        try:
            sub.offer(event)
        except Exception as exc:
            logger.warning("event bus offer failed for event %s: %s", event.name, exc)


# Retained name: `_fanout_filtered` is what the pre-hardening bus called this.
_fanout_filtered = _fanout


# ---------------------------------------------------------------------------
# Back-plane wiring
# ---------------------------------------------------------------------------

def set_backplane(bp: Optional[Any]) -> None:
    """Install (or clear) the cross-process signal carrier.

    `None` is the standalone configuration, not an unconfigured one — see
    `event_backplane`'s module docstring. Installing starts the carrier and points its inbound
    deliveries at :func:`deliver_from_backplane`.
    """
    global _backplane
    if _backplane is not None:
        try:
            _backplane.close()
        except Exception:
            logger.debug("closing the previous back-plane failed", exc_info=True)
    _backplane = bp
    if bp is not None:
        bp.start(deliver_from_backplane)


def backplane() -> Optional[Any]:
    """The installed back-plane, or `None` for standalone."""
    return _backplane


def _to_backplane(event: Event) -> None:
    if _backplane is None:
        return
    try:
        _backplane.publish(event.to_wire())
    except Exception:
        logger.debug("back-plane publish failed for %s", event.name, exc_info=True)


def deliver_from_backplane(wire: Mapping[str, Any]) -> None:
    """Inject an event that arrived from another process into this one's fan-out.

    Fan-out only: it is neither re-published to the back-plane (which would loop) nor appended to
    the log (the emitting process already logged it against its own origin, and a second append
    here would mint a duplicate under this node's proper time and break `seq > cursor` as an exact
    resume).

    Called from the carrier's own thread, so it hops onto the event loop rather than touching the
    subscriber list directly.
    """
    if _loop is None or _loop.is_closed():
        return
    try:
        event = Event.from_wire(wire)
    except Exception:
        logger.debug("undecodable back-plane message dropped", exc_info=True)
        return
    asyncio.run_coroutine_threadsafe(_fanout(event), _loop)


# ---------------------------------------------------------------------------
# The durable log — proper-time ordering, borrowed in shape from the lattice
# ---------------------------------------------------------------------------

#: The table the log needs. Kept here as a constant, and created by `EventLog` with
#: `IF NOT EXISTS`, so the log works against a store whose schema module has not adopted it yet.
#:
#: The shape is the lattice's: `(_origin, _seq)` primary key, `_origin` the authoring observer and
#: `_seq` its monotonic proper time. `WITHOUT ROWID` because the primary key IS the access path —
#: every read is a range scan over one origin's ordered seqs, and a second b-tree keyed on a rowid
#: nobody queries would be pure write amplification on the hottest write in the system.
EVENT_LOG_DDL = """
CREATE TABLE IF NOT EXISTS event_log (
    _origin      TEXT    NOT NULL,
    _seq         INTEGER NOT NULL,
    event_id     TEXT    NOT NULL,
    name         TEXT    NOT NULL,
    ts           REAL    NOT NULL,
    container_id TEXT,
    artifact_id  TEXT,
    content_type TEXT,
    actor_id     TEXT,
    containers   TEXT,
    payload      TEXT    NOT NULL,
    PRIMARY KEY (_origin, _seq)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_event_log_name      ON event_log(name, _origin, _seq);
CREATE INDEX IF NOT EXISTS ix_event_log_container ON event_log(container_id, _origin, _seq);
"""


@dataclass(frozen=True)
class Cursor:
    """A durable subscriber's position: the last `_seq` it has taken responsibility for, per origin.

    Per origin rather than a single number, for the same reason the lattice's proper time is per
    observer: seqs from two origins are not comparable, and collapsing them onto one axis would
    make a replicated event from a peer either starve a reader or skip past unread local events
    depending on which number happened to be larger.

    An origin absent from the marks means "nothing read from it", so a new subscriber's cursor is
    the empty one and reads the whole log — the honest default, since the alternative (start at the
    head) silently discards everything that happened before the subscription existed.
    """

    marks: Tuple[Tuple[str, int], ...] = ()

    @classmethod
    def of(cls, marks: Mapping[str, int]) -> "Cursor":
        return cls(tuple(sorted((str(k), int(v)) for k, v in (marks or {}).items())))

    @classmethod
    def parse(cls, text: Any) -> "Cursor":
        """Decode the compact `origin:seq,origin:seq` string form.

        Unparseable fragments are dropped rather than raising. A cursor is a resume hint held by a
        client; a malformed one degrades to reading from further back, which redelivers — allowed
        under at-least-once — whereas raising would strand the subscriber entirely.
        """
        if isinstance(text, Cursor):
            return text
        if isinstance(text, Mapping):
            return cls.of(text)
        marks: Dict[str, int] = {}
        for part in str(text or "").split(","):
            if ":" not in part:
                continue
            origin, _, seq = part.rpartition(":")
            try:
                marks[origin.strip()] = int(seq)
            except ValueError:
                continue
        return cls.of(marks)

    def __str__(self) -> str:
        return ",".join(f"{origin}:{seq}" for origin, seq in self.marks)

    def to_dict(self) -> Dict[str, int]:
        return {origin: seq for origin, seq in self.marks}

    def advanced(self, origin: str, seq: int) -> "Cursor":
        """This cursor moved forward over `(origin, seq)`. Never backwards: `MAX`, for the reason
        `SeqAllocator.flush` uses `MAX` — a position that can move backwards can redeliver without
        bound, and a monotone one cannot."""
        marks = self.to_dict()
        marks[str(origin)] = max(int(seq), marks.get(str(origin), 0))
        return Cursor.of(marks)


class EventLog:
    """The append-only change log, ordered by `(origin, seq)` — the durable half of the bus.

    **Why its own counter rather than `seq.SeqAllocator`.** The lattice's allocator is shared by
    `vertex` and `edge` and underwrites an exact accounting identity —
    `live_rows + vacated == last_seq` — which is how a row lost outside the write path is detected
    at all. Drawing event seqs from it would inflate `last_seq` by rows that live in neither table,
    so every store would report `unaccounted > 0` and the loss detector would be permanently
    saturated. The log therefore takes the *shape and the discipline* of proper time — per-origin,
    monotonic, allocated inside the writing transaction so a rollback consumes none — with a
    counter of its own, derived from `MAX(_seq)` within the write lock.

    **The log is a replication of the feed, not the system of record.** The artifacts are. An
    append that fails is logged and counted and never propagated into the write path, because the
    persistence boundary's contract is that a failed emit does not break a write. What a lost log
    row costs is a durable subscriber's ability to learn about that one change by cursor; it can
    still see the artifact, because the artifact is what was actually written.
    """

    def __init__(self, conn: Any, origin: str, *, ensure: bool = True):
        self.conn = conn
        self.origin = str(origin)
        self._lock = threading.Lock()
        self.appended = 0
        self.failed = 0
        if ensure:
            self.ensure_schema()

    def ensure_schema(self) -> None:
        """Create the table if the store's schema module has not. Defensive on purpose: the log
        must work against a store file that predates it, and `IF NOT EXISTS` makes adopting the
        DDL centrally a no-op rather than a conflict."""
        with self.conn.write() as cur:
            for statement in EVENT_LOG_DDL.strip().split(";"):
                if statement.strip():
                    cur.execute(statement)

    # -- append ------------------------------------------------------------------------

    def append(self, event: Event) -> Optional[Cursor]:
        """Append one event and stamp its `(origin, seq)` onto it. Returns the new head, or `None`
        if the append failed.

        Allocation and insert share one transaction, so a rolled-back append consumes no proper
        time and the next one reissues the same seq — the gap-freeness the cursor arithmetic rests
        on. The process-level lock serializes allocation within this process; the store's own write
        lock covers the cross-process case.
        """
        try:
            with self._lock, self.conn.write() as cur:
                row = cur.execute(
                    "SELECT _seq FROM event_log WHERE _origin = ? "
                    "ORDER BY _origin DESC, _seq DESC LIMIT 1", (self.origin,)).fetchone()
                seq = (int(row[0]) if row is not None else 0) + 1
                cur.execute(
                    "INSERT INTO event_log(_origin, _seq, event_id, name, ts, container_id, "
                    "artifact_id, content_type, actor_id, containers, payload) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (self.origin, seq, event.event_id, event.name, float(event.ts),
                     event.container_id, event.artifact_id, event.content_type, event.actor_id,
                     json.dumps(list(event.containers)),
                     json.dumps(event.payload, default=str)))
            event.origin = self.origin
            event.seq = seq
            self.appended += 1
            return Cursor.of({self.origin: seq})
        except Exception:
            self.failed += 1
            logger.error("durable event log append failed for %s — the write itself stands; a "
                         "durable subscriber will not learn of this change by cursor",
                         event.name, exc_info=True)
            return None

    # -- read --------------------------------------------------------------------------

    def read_since(self, cursor: Any, *, event_filter: Optional[EventFilter] = None,
                   limit: int = 500) -> List[Event]:
        """Events strictly after *cursor*, in `(origin, seq)` order, at most *limit* of them.

        Strictly after, so acking a cursor and resuming from it cannot redeliver the acked event —
        the one redelivery at-least-once still permits is the one where the consumer never got to
        ack. Filtering is applied after the read rather than pushed into SQL for the glob cases,
        since `event_names` is fnmatch and SQL has no equivalent; the indexed columns still narrow
        the scan.
        """
        cur = Cursor.parse(cursor)
        marks = cur.to_dict()
        clauses: List[str] = []
        params: List[Any] = []
        if marks:
            placeholders = ",".join("?" for _ in marks)
            clauses.append(f"_origin NOT IN ({placeholders})")
            params.extend(marks.keys())
            for origin, seq in marks.items():
                clauses.append("(_origin = ? AND _seq > ?)")
                params.extend((origin, int(seq)))
        where = (" WHERE " + " OR ".join(clauses)) if clauses else ""
        sql = ("SELECT _origin, _seq, event_id, name, ts, container_id, artifact_id, "
               "content_type, actor_id, containers, payload FROM event_log"
               f"{where} ORDER BY _origin, _seq LIMIT ?")
        rows = self.conn.read().execute(sql, (*params, max(1, int(limit)))).fetchall()

        out: List[Event] = []
        for row in rows:
            event = self._row_to_event(row)
            if event_filter is not None and not event_filter.matches(event):
                continue
            out.append(event)
        return out

    def head(self) -> Cursor:
        """The cursor a subscriber would hold if it had already read everything."""
        rows = self.conn.read().execute(
            "SELECT _origin, MAX(_seq) FROM event_log GROUP BY _origin").fetchall()
        return Cursor.of({str(r[0]): int(r[1]) for r in rows if r[1] is not None})

    @staticmethod
    def _row_to_event(row: Any) -> Event:
        try:
            payload = json.loads(row[10])
        except Exception:
            payload = {}
        try:
            containers = tuple(json.loads(row[9] or "[]"))
        except Exception:
            containers = ()
        return Event(
            name=row[3], payload=payload if isinstance(payload, dict) else {},
            container_id=row[5], artifact_id=row[6], content_type=row[7], actor_id=row[8],
            ts=float(row[4]), event_id=row[2],
            containers=tuple(str(c) for c in containers),
            origin=str(row[0]), seq=int(row[1]),
        )

    # -- retention ---------------------------------------------------------------------

    def prune(self, *, keep_last: int) -> int:
        """Drop all but the most recent *keep_last* events per origin. Returns rows removed.

        Uncalled by default, and deliberately so: retention is an operator's fact about their disk
        and their slowest consumer, not a constant this module can know. Trimming past a live
        subscriber's cursor turns at-least-once into at-most-once for that subscriber, which is why
        this is a call an operator makes rather than a default that makes it for them.
        """
        removed = 0
        with self._lock, self.conn.write() as cur:
            for row in cur.execute("SELECT DISTINCT _origin FROM event_log").fetchall():
                origin = str(row[0])
                keep = cur.execute(
                    "SELECT _seq FROM event_log WHERE _origin = ? "
                    "ORDER BY _seq DESC LIMIT 1 OFFSET ?",
                    (origin, max(0, int(keep_last)) - 1 if keep_last > 0 else 0)).fetchone()
                if keep is None:
                    continue
                result = cur.execute("DELETE FROM event_log WHERE _origin = ? AND _seq < ?",
                                     (origin, int(keep[0])))
                removed += int(result.rowcount or 0)
        return removed


def set_event_log(log: Optional["EventLog"]) -> None:
    """Install (or clear) the durable log. `None` means live-only, which is a legal configuration:
    the in-process bus is complete without it, exactly as it is without a back-plane."""
    global _log
    _log = log


def event_log() -> Optional["EventLog"]:
    """The installed durable log, or `None`."""
    return _log


def open_event_log(db: Any) -> "EventLog":
    """Build the log for a `LatticeDatabase`, taking its connection and its origin.

    The origin is the store's, not a fresh one: the log is that observer's account of what it did,
    so its proper time belongs on the same axis the store's own writes are stamped with.
    """
    return EventLog(db.conn, getattr(db, "origin", None) or "mantle")


def _append_to_log(event: Event) -> None:
    if _log is None:
        return
    if event.seq is not None:
        return          # already carries proper time (a replayed or back-plane event)
    _log.append(event)
