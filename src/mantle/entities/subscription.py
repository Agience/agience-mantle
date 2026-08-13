"""A durable subscription — an artifact carrying a filter and a cursor.

Everything is an artifact, and a subscription is no exception. It is not a row in a side table and
it has no endpoints of its own: it is created, read, updated and deleted through `/artifacts` like
anything else, which means it is light-cone authorized by the machinery that already exists and
inherits versioning, grants, sharing and the change feed without a line of bespoke code. A consumer
outside this repo can therefore subscribe and resume with zero coupling — it needs the artifact API
and nothing more.

The same reasoning as `entities/collection.py`'s `Collection = Artifact`: the data model is uniform,
so a new *kind* of thing is a new `content_type`, not a new plane. This module is the codec for
that content type — :class:`Subscription` is a typed view over an `Artifact`, never a second
storage shape.

    content_type = application/vnd.agience.subscription+json
    context      = {"content_type": …, "filter": {…}, "cursor": "origin:seq,…",
                    "delivery": "at_least_once"}

Why the body is `context` and not `content`
-------------------------------------------
`content` is envelope-encrypted at the persistence boundary, keyed to the collection's origin root.
That is exactly right for a document and exactly wrong for this: a subscription holds no secret —
a filter and a position — and putting it behind the content key would mean a node could not read
its own subscriptions without content-key custody, so a cursor could become unreadable while the
subscription it belongs to is perfectly authorized. `context` is the artifact's metadata plane, it
is not encrypted, and it is where the content-type discriminator already lives.

Why the cursor lives on the artifact
------------------------------------
Because the artifact is the only thing both sides can already reach. A cursor held only in the
subscriber's memory is lost on the crash it exists to survive; a cursor held in a server-side table
needs an endpoint, an authorization rule and a migration. On the artifact it is a normal update,
authorized by the normal `update` permission on the normal resource — a subscriber that may advance
its own cursor is exactly a subscriber that holds `update` on its own subscription.

Advancing the cursor is the **ack** of the at-least-once contract. A consumer reads forward from
the cursor, takes responsibility for what it read, and only then writes the cursor back. A crash
between those two steps redelivers, which is why durable consumers must be idempotent on
`event_id`; a crash before them delivers nothing twice and loses nothing.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional

from mantle.entities.artifact import Artifact
from mantle.events.event_bus import DELIVERY_DURABLE, DELIVERY_LIVE, Cursor, EventFilter

__all__ = [
    "SUBSCRIPTION_CONTENT_TYPE",
    "Subscription",
    "is_subscription",
    "load_subscription",
    "list_subscriptions",
    "save_subscription",
    "advance_cursor",
]

#: The one discriminator. A subscription IS an artifact; this is what makes it a subscription.
SUBSCRIPTION_CONTENT_TYPE = "application/vnd.agience.subscription+json"


class Subscription:
    """A durable subscription: a filter, a cursor, and a delivery class.

    Constructed from and rendered back to an `Artifact`; nothing here persists anything itself.
    The helpers at the bottom of the module do that, through the ordinary store functions, so a
    subscription write is indistinguishable from any other artifact write — including to the change
    feed, which sees `artifact.updated` when a cursor advances.
    """

    __slots__ = ("id", "name", "filter", "cursor", "delivery", "container_id",
                 "owner_id", "batch_limit")

    #: A batch bound rather than an unbounded drain. A durable read that returned everything since
    #: the cursor would hand a long-disconnected consumer an unbounded response; the consumer
    #: advances its cursor and asks again, which is the same total work in bounded pieces.
    DEFAULT_BATCH_LIMIT = 500

    def __init__(self, *, id: Optional[str] = None, name: Optional[str] = None,
                 filter: Optional[EventFilter] = None, cursor: Any = "",
                 delivery: str = DELIVERY_DURABLE, container_id: str = "",
                 owner_id: Optional[str] = None,
                 batch_limit: int = DEFAULT_BATCH_LIMIT):
        self.id = id
        self.name = name
        self.filter = filter if isinstance(filter, EventFilter) else EventFilter.from_dict(filter)
        self.cursor = Cursor.parse(cursor)
        # Only the two classes the bus actually implements. An unrecognized value reads as durable
        # rather than as live: a subscription artifact exists in order to survive a disconnect, so
        # the fail-safe direction for a typo is the stronger guarantee, not the weaker one.
        self.delivery = delivery if delivery in (DELIVERY_DURABLE, DELIVERY_LIVE) \
            else DELIVERY_DURABLE
        self.container_id = container_id
        self.owner_id = owner_id
        self.batch_limit = max(1, int(batch_limit or self.DEFAULT_BATCH_LIMIT))

    # -- codec ---------------------------------------------------------------------

    def to_context(self) -> str:
        """The artifact's `context` — the whole subscription, as one JSON document.

        `content_type` is repeated inside it because the store's typed reads look on both sides:
        `list_committed_artifacts_by_context_content_type` narrows on the indexed column and then
        confirms against the context, so a subscription that set only one of the two would be
        findable by exactly one of the two paths.
        """
        return json.dumps({
            "content_type": SUBSCRIPTION_CONTENT_TYPE,
            "filter": self.filter.to_dict(),
            "cursor": str(self.cursor),
            "delivery": self.delivery,
            "batch_limit": self.batch_limit,
        }, separators=(",", ":"), ensure_ascii=False)

    def to_artifact(self) -> Artifact:
        """Render to the artifact that will be stored."""
        return Artifact(
            id=self.id,
            collection_id=self.container_id or "",
            name=self.name,
            content_type=SUBSCRIPTION_CONTENT_TYPE,
            context=self.to_context(),
            content="",
            created_by=self.owner_id,
            state=Artifact.STATE_COMMITTED,
        )

    @classmethod
    def from_artifact(cls, artifact: Any) -> Optional["Subscription"]:
        """Read a subscription off an artifact, or `None` if it is not one.

        `None` rather than an exception, because callers ask this of artifacts they did not choose
        — a listing, a light-cone walk — and "not a subscription" is an ordinary answer there. A
        subscription whose body will not parse is also `None`: a filter that cannot be read must
        not degrade to the empty filter, which selects everything.
        """
        if not is_subscription(artifact):
            return None
        raw = artifact.get("context") if isinstance(artifact, dict) \
            else getattr(artifact, "context", None)
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        try:
            body = json.loads(raw or "{}")
        except Exception:
            return None
        if not isinstance(body, dict):
            return None
        get = artifact.get if isinstance(artifact, dict) else \
            (lambda key, default=None: getattr(artifact, key, default))
        return cls(
            id=get("id"),
            name=get("name"),
            filter=body.get("filter"),
            cursor=body.get("cursor") or "",
            delivery=str(body.get("delivery") or DELIVERY_DURABLE),
            container_id=get("collection_id", "") or "",
            owner_id=get("created_by"),
            batch_limit=int(body.get("batch_limit") or cls.DEFAULT_BATCH_LIMIT),
        )

    # -- reading forward ------------------------------------------------------------

    def read(self, log: Any, *, limit: Optional[int] = None) -> List[Any]:
        """The next batch this subscription is owed, from *log*, honouring its own filter.

        Does not advance the cursor. Reading and acking are separate calls because they are
        separate decisions: the consumer decides when it has taken responsibility, and only it
        knows.
        """
        return log.read_since(self.cursor, event_filter=self.filter,
                              limit=int(limit or self.batch_limit))

    def ack(self, events: Any) -> Cursor:
        """Advance this subscription's cursor over *events* and return the new position.

        Monotone by way of `Cursor.advanced`, so acking an out-of-order or replayed batch cannot
        walk the position backwards and set up unbounded redelivery.
        """
        cursor = self.cursor
        for event in events or ():
            origin, seq = getattr(event, "origin", None), getattr(event, "seq", None)
            if origin and isinstance(seq, int):
                cursor = cursor.advanced(origin, seq)
        self.cursor = cursor
        return cursor


def is_subscription(artifact: Any) -> bool:
    """Is this artifact a subscription? The content type, on the artifact or in its context.

    Duck-typed like the grant predicates, and for the same reason: artifact-shaped objects reach
    this from several producers (entities, raw docs, test doubles) and the answer must not depend
    on which one built it.
    """
    if artifact is None:
        return False
    get = artifact.get if isinstance(artifact, dict) else \
        (lambda key, default=None: getattr(artifact, key, default))
    if get("content_type") == SUBSCRIPTION_CONTENT_TYPE:
        return True
    ctx = get("context")
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except Exception:
            return False
    return isinstance(ctx, dict) and ctx.get("content_type") == SUBSCRIPTION_CONTENT_TYPE


# ---------------------------------------------------------------------------
# Store helpers — the ordinary artifact path, never a private one
# ---------------------------------------------------------------------------

def load_subscription(db: Any, subscription_id: str) -> Optional[Subscription]:
    """Read one subscription by id, or `None` if it is absent or is not a subscription.

    Goes through `db.backend.get_artifact`, so the read is the same read any artifact gets: same
    decryption boundary, same store adapter, same absence semantics. Authorization is the caller's
    — the router that already checks `read` on an artifact id has checked it on this one.
    """
    from mantle.db import backend as store

    artifact = store.get_artifact(db, subscription_id)
    return Subscription.from_artifact(artifact)


def list_subscriptions(db: Any, *, owner_id: Optional[str] = None) -> List[Subscription]:
    """Every subscription artifact this store holds, optionally narrowed to one creator.

    Provided for an operator view and for the resume path; it is not an authorization boundary and
    performs no light-cone walk. A router surfacing this must narrow it the way it narrows any
    other listing.
    """
    from mantle.db import backend as store

    out: List[Subscription] = []
    for doc in store.list_committed_artifacts_by_context_content_type(
            db, SUBSCRIPTION_CONTENT_TYPE, created_by=owner_id):
        sub = Subscription.from_artifact(doc)
        if sub is not None:
            out.append(sub)
    return out


def save_subscription(db: Any, subscription: Subscription) -> Subscription:
    """Create or update the subscription's artifact.

    One function for both, because the store already distinguishes them and a caller holding a
    `Subscription` should not have to. A subscription with no id is new; one with an id that the
    store does not hold is also new, which is what makes a client-chosen id work.
    """
    from mantle.db import backend as store

    artifact = subscription.to_artifact()
    existing = store.get_artifact(db, artifact.id) if subscription.id else None
    if existing is None:
        store.create_artifact(db, artifact)
    else:
        artifact.created_time = getattr(existing, "created_time", artifact.created_time)
        artifact.created_by = getattr(existing, "created_by", artifact.created_by)
        store.update_artifact(db, artifact)
    subscription.id = artifact.id
    return subscription


def advance_cursor(db: Any, subscription_id: str, cursor: Any) -> Optional[Subscription]:
    """Persist an acknowledged cursor. Returns the stored subscription, or `None` if it is gone.

    Read-modify-write through the ordinary path rather than a targeted field update, because a
    subscription is an artifact and artifacts are written whole. The cursor is merged rather than
    replaced (`Cursor.advanced`), so two acks racing cannot move the position backwards —
    the later write wins on value, and the value only ever increases.
    """
    subscription = load_subscription(db, subscription_id)
    if subscription is None:
        return None
    merged = subscription.cursor
    for origin, seq in Cursor.parse(cursor).marks:
        merged = merged.advanced(origin, seq)
    subscription.cursor = merged
    return save_subscription(db, subscription)
