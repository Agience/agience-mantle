"""Observation events — a read is an observation, and it is recorded as one.

⭐ A QUERY IS AN OBSERVATION [John, 2026-08-13]. `db/edge.py` already defines one: *"an edge is an
observation: it records that some observer looked and found a relation."* A `recall` is exactly
that — an observer looked, and found. This module does not invent an audit vocabulary beside the
artifact universe; it records the looking, in the universe's own terms.

Why this is an EVENT and not an edge
------------------------------------
The two halves of "observation" pull apart, and only one of them is an edge.

* **The standing fact** — *this observer has seen this artifact* — is idempotent. Seeing a thing
  twice does not make it twice-seen. That is an edge, and `db/edge.py` keys one by
  `blake2b(src || dst || label)` precisely so a replay collapses to one row.
* **The occurrence** — *at 13:44:58 this observer looked, having asked X, and got back Y* — must
  NOT collapse. Every occurrence is a distinct fact and the whole content of an audit.

`db/schema.py` states the same thing from the other side, as its reason for keeping `access_event`
out of the edge table: an upsert by `(src, dst, label)` "would collapse repeated accesses of the
same (principal, artifact, action) into one row and destroy the history an audit log exists to
keep."

So the occurrence goes to the durable `EventLog`, which is append-only and ordered by
`(origin, seq)` — occurrences are what it holds natively. Nothing here forecloses the standing
edge: when one is wanted, it is derived from these, and the two agree by construction because one
is the fold of the other.

Why the container is a per-principal `Observations`, and not the artifacts that matched
--------------------------------------------------------------------------------------
⛔ ADDRESSING THIS AT THE MATCHED ARTIFACTS WOULD LEAK THE QUERY. `routers/events_router.
_event_visible_to` decides visibility as *"would the ordinary read path serve the artifact this
event is about"*:

    subject = event.artifact_id
    if subject and access.known(subject):  return access.may_read(subject)
    if subject and access.denies(subject): return False
    return access.may_read(event.container_id)          # ← an event naming no artifact

An observation naming one of its own hits would therefore be visible to everyone holding a grant
on that hit — which publishes one agent's search terms to every other reader of whatever happened
to match. The query text is the observer's, not the observed's.

Naming no artifact at all drops to `container_id`, so THAT is the subject, and it is a container
the observer owns. `Observations` is provisioned per principal through
`workspace_service.create_container`, which "grants the creator full CRUDEASIO" — the same
mechanism and the same owner-grant side effect as the `Inbox` that
`seed_provisioning/user_provisioning._ensure_inbox_workspace` provisions, and provisioned in the
same place for the same reason. A separate container rather than the Inbox because an observation
is not mail: it is a log the observer owns, and filing it into the workspace a person actually
reads would bury the one in the other.

⛔ PROVISIONED THERE, NEVER HERE. The recording path only ever LOOKS the container up — see
:func:`_lookup_container` for why a write on the read path is the one thing this module must not
do.

The result is the property that was asked for: an agent's queries are visible to the principal who
made them and to anyone that principal grants, and to nobody else.

⚠ THE RESULT SET GOES UNDER `artifacts`, AND THE KEY IS LOAD-BEARING. `event_bus.redact_content`
strips `FEED_BODY_FIELDS = ("content", "content_encrypted")` by NAME, at the payload top level and
inside `artifact` / `artifacts` — nowhere else. `recall` hits carry a `content` field. Under any
other key (`hits`, `results`, …) the redaction does not reach them and artifact plaintext lands in
the durable log permanently, where the whole point of `redact_content` is that it cannot. The
descriptors built here carry no body anyway; the key name is the second line of that defence, and
the one that survives someone later passing raw hits through.

Best-effort, always. An observation that cannot be recorded must never turn a successful read into
a failed one — the read is the fact, this is the announcement of it.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: The container's content type. Opaque to the store, like every other content type — it exists so
#: a client can find "my observations" without walking every container the principal owns.
OBSERVATIONS_CONTENT_TYPE = "application/vnd.agience.observations+json"

#: The container's name. Human-facing only; lookup is by content type, never by this.
OBSERVATIONS_NAME = "Observations"

#: The event name. `artifact.observed` sits in the same dotted vocabulary as `artifact.created` /
#: `artifact.updated` / `artifact.deleted` deliberately: an observation is a thing that happened to
#: an artifact, and a subscriber filtering `artifact.*` should see it without being taught a second
#: namespace. `EventFilter` supports fnmatch globs, so `artifact.observed` is selectable alone.
OBSERVED = "artifact.observed"

#: principal_id -> container id. A lookup per query would put a store round-trip on the hot read
#: path to answer a question whose answer does not change. Losing this costs one extra lookup and
#: never a duplicate container, because nothing on this path creates one.
_container_cache: Dict[str, str] = {}
_cache_lock = threading.Lock()


def _lookup_container(store_db, principal_id: str) -> Optional[str]:
    """The principal's existing `Observations` container id. **Never creates one.**

    ⛔ THE OBSERVATION PATH DOES NOT WRITE, and that is the whole reason this is split from
    :func:`ensure_observations_container`. Creating the container here would put a store WRITE on
    the path of every read:

    * the first observation for each principal would pay a container create — measured at ~11s on
      this stack — and pay it inside the tool call the user is waiting on;
    * it would emit a spurious `artifact.created` on the change feed, announcing a side effect of
      looking rather than a thing anyone did;
    * and it entangles the audit path with the tools' own write path. `tests/
      test_mcp_can_write_and_find.py` caught exactly that: it asserts `create_artifact` reaches
      `workspace_service.create_container` with the caller's arguments, and an observation calling
      the same function during the same request made the *observation's* call the one the
      assertion saw.

    A read that cannot be recorded is a gap in the log; a read that writes is a gap in the design.
    So provisioning is done once, out of band, by
    `seed_provisioning/user_provisioning._ensure_observations_container`, and the hot path only
    ever looks.
    """
    with _cache_lock:
        hit = _container_cache.get(principal_id)
    if hit:
        return hit

    try:
        import mantle.db.backend as _store
        existing = _store.get_collections_by_owner_and_type(
            store_db, principal_id, OBSERVATIONS_CONTENT_TYPE)
    except Exception:
        logger.debug("observation container lookup failed for %s", principal_id, exc_info=True)
        return None
    if not existing:
        # Not provisioned yet. Nothing is recorded, and nothing is created to fix that here — the
        # next provisioning run does it. Logged at debug rather than warning because on a node
        # mid-rollout this is the ordinary state, not a fault.
        logger.debug("no observations container for %s; not recording", principal_id)
        return None

    # Oldest wins, on the same rule `_ensure_inbox_workspace` uses to pick a primary workspace:
    # whichever exists first is the one everything else already references.
    container_id = min(existing, key=lambda c: getattr(c, "created_time", "") or "").id
    with _cache_lock:
        _container_cache[principal_id] = container_id
    return container_id


def ensure_observations_container(store_db, principal_id: str) -> Optional[str]:
    """Provision the principal's `Observations` container, returning its id.

    Called from provisioning, never from a read. `create_container` "grants the creator full
    CRUDEASIO", which is the owner grant the visibility rule then depends on — see the module
    docstring on why the container, and not the matched artifacts, is what an observation names.

    Find-then-create rather than create-then-catch: two racing calls would otherwise leave one
    principal owning two containers of the same content type with nothing to say which is theirs.
    """
    existing = _lookup_container(store_db, principal_id)
    if existing:
        return existing
    try:
        from mantle.services import workspace_service
        container_id = workspace_service.create_container(
            store_db,
            user_id=principal_id,
            content_type=OBSERVATIONS_CONTENT_TYPE,
            name=OBSERVATIONS_NAME,
            description="Queries and reads made by this principal, newest last.",
        ).id
    except Exception:
        logger.debug("observations container not provisioned for %s", principal_id, exc_info=True)
        return None
    with _cache_lock:
        _container_cache[principal_id] = container_id
    return container_id


def descriptors(hits: Sequence[Any]) -> List[Dict[str, Any]]:
    """Result descriptors for the payload: identity and score, never a body.

    Takes hits in either shape the tools produce — a mapping (`recall`) or an entity with
    attributes — because this is called from the one funnel every tool passes through and that
    funnel sees both.
    """
    out: List[Dict[str, Any]] = []
    for h in hits or ():
        if isinstance(h, dict):
            get = h.get
        else:
            def get(k, _h=h):  # noqa: E306
                return getattr(_h, k, None)
        d = {"id": get("id")}
        for field in ("score", "title", "content_type", "collection_id"):
            v = get(field)
            if v is not None:
                d[field] = v
        if d["id"]:
            out.append(d)
    return out


def record_observation(
    *,
    store_db,
    principal_id: Optional[str],
    tool: str,
    result: str = "allowed",
    query_text: Optional[str] = None,
    hits: Optional[Sequence[Any]] = None,
    actor: Optional[str] = None,
    via: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Record one observation. Never raises, never blocks the read that caused it.

    `actor` is WHICH MACHINE looked, where `principal_id` is whose authority it looked with — the
    same distinction `services/dependencies` draws when it fills the audit context, and the one
    that makes "which agent" answerable at all. Recorded, never consulted: nothing here decides
    access.
    """
    if not principal_id:
        # An unauthenticated read has no observer to attribute and no container to address. The
        # HTTP layer has already refused or allowed it; there is no third thing to say here.
        return
    try:
        container_id = _lookup_container(store_db, principal_id)
        if not container_id:
            return

        payload: Dict[str, Any] = {
            "tool": tool,
            "result": result,
            "principal_id": principal_id,
            "actor": actor,
            "via": via,
            # See the module note: this key is what `redact_content` reaches into.
            "artifacts": descriptors(hits or ()),
        }
        payload["count"] = len(payload["artifacts"])
        if query_text is not None:
            payload["query"] = query_text
        if extra:
            payload.update(extra)

        from mantle.events import event_bus

        event_bus.publish_event_sync(event_bus.Event(
            name=OBSERVED,
            payload=payload,
            # ⛔ artifact_id STAYS None. See the module docstring: naming a matched artifact hands
            # this observer's query to everyone who can read that artifact.
            artifact_id=None,
            container_id=container_id,
            containers=(container_id,),
            actor_id=actor or principal_id,
        ))
    except Exception:  # observing must never break the read
        logger.debug("observation not recorded for tool=%s", tool, exc_info=True)


__all__ = [
    "OBSERVATIONS_CONTENT_TYPE", "OBSERVATIONS_NAME", "OBSERVED",
    "descriptors", "record_observation",
]
