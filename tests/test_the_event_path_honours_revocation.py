"""Revocation lands on the event path, live and on replay.

The `/events` socket serves the same artifacts the HTTP routes serve, so it has to answer the same
question about them. It did not. Its check was a short list of shortcuts:

* an **actor match** — `user_id == event.actor_id` — returned visible with no grant consulted, no
  expiry, no deny and no light cone. `actor_id` is the doc's `modified_by`/`created_by`, so a
  principal whose grants were entirely revoked kept receiving every event it had ever created or
  last touched;
* the container check was **direct-grant-only**, with no root hop and no origin walk, so an
  artifact-level deny that `check_access` honours was invisible to it;
* and its cache was **positive-only and never invalidated** for the socket's lifetime, which makes
  granting take effect live and revoking take effect never.

`_read_authorized` now asks `services.dependencies.check_access(..., "read", ...)` — the same
question, the same implementation — and `_Access` caches both verdicts with an expiry. These tests
pin the answers, not the plumbing: each one revokes or denies something and asserts the feed stops,
with a positive control beside it so "stops" cannot be satisfied by a feed that never started.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle.events import event_bus                                       # noqa: E402
from mantle.db import lattice_api                                  # noqa: E402
from mantle.entities.artifact import Artifact                      # noqa: E402
from mantle.entities.grant import Grant                            # noqa: E402
from mantle.routers import events_router as ev                     # noqa: E402
from mantle.services.dependencies import AuthContext               # noqa: E402

ARTIFACT = "a-1"
CONTAINER = "col-1"
OWNER = "u-1"


@pytest.fixture
def store(tmp_path):
    """A container holding one artifact, joined by the **origin edge**.

    The edge is not decoration: `check_access` reaches a child from a grant on its container by
    walking origin edges, so a `collection_id` with no edge behind it is containment the light cone
    cannot see. The event path asks that same walk, so the fixture has to model containment the way
    the store does — an unrestricted `propagate`, which is what every ordinary write creates.
    """
    db = lattice_api.LatticeDatabase(str(tmp_path / "revocation.db"), origin="node-a")
    lattice_api.create_artifact(db, Artifact(
        id=CONTAINER, collection_id="", name="quarterly", content="", created_by=OWNER))
    lattice_api.create_artifact(db, Artifact(
        id=ARTIFACT, root_id=ARTIFACT, collection_id=CONTAINER, name="memo", content="",
        created_by=OWNER, modified_by=OWNER))
    lattice_api.add_artifact_to_collection(db, CONTAINER, ARTIFACT)
    return db


@pytest.fixture(autouse=True)
def instant_expiry(monkeypatch):
    """No verdict outlives the assertion that follows it.

    Production holds a verdict for `_ACCESS_TTL_SECONDS`; these tests revoke and re-ask in the same
    millisecond, so the TTL is set to zero. What is under test is that the cache expires at all —
    how long it waits is an operator's tuning, not the security property.
    """
    monkeypatch.setattr(ev, "_ACCESS_TTL_SECONDS", 0.0)


def _auth(user_id=OWNER, grants=()):
    return AuthContext(principal_id=user_id, principal_type="user", user_id=user_id,
                       grants=list(grants))


def _session(store, auth):
    return ev._Session("test-token", auth, store)


def _access(store, auth):
    _verdicts, access = ev._container_access(_session(store, auth))
    return access


def _grant(store, gid, resource_id, *, effect="allow", read=True, grantee=OWNER, state="active"):
    return lattice_api.create_grant(store, Grant(
        id=gid, resource_id=resource_id, grantee_type="user", grantee_id=grantee,
        granted_by="admin", can_read=read, effect=effect, state=state))


def _revoke(store, grant):
    grant.state = "revoked"
    lattice_api.update_grant(store, grant)


def _event(*, artifact_id=ARTIFACT, container_id=CONTAINER, actor_id=OWNER, **kw):
    return event_bus.Event(name="artifact.updated", payload={"artifact": {"id": artifact_id}},
                           artifact_id=artifact_id, container_id=container_id,
                           actor_id=actor_id, **kw)


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 1 · A revoked principal receives nothing
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_a_granted_principal_receives_its_events(store):
    """The positive control. Everything below asserts an absence, and an absence proves nothing
    unless the same setup with the grant in place produces a delivery."""
    _grant(store, "g-1", CONTAINER)
    auth = _auth()
    assert ev._event_visible_to(auth, _event(), _access(store, auth)) is True


def test_revoking_the_grant_stops_delivery_on_the_same_socket(store):
    """The verdict expires, so the revoke lands without waiting for the client to reconnect."""
    grant = _grant(store, "g-1", CONTAINER)
    auth = _auth()
    access = _access(store, auth)
    assert ev._event_visible_to(auth, _event(), access) is True

    _revoke(store, grant)

    assert ev._event_visible_to(auth, _event(), access) is False, \
        ("the socket kept serving a principal whose grant is revoked — a positive verdict cached "
         "for the connection's lifetime is a revocation that never lands")


def test_being_the_actor_is_not_a_grant(store):
    """`actor_id` is `modified_by or created_by`: provenance, which does not change when authority
    is withdrawn. The revoked owner of an artifact is still its last writer forever."""
    grant = _grant(store, "g-1", CONTAINER)
    auth = _auth()
    _revoke(store, grant)

    event = _event(actor_id=OWNER)
    assert event.actor_id == auth.user_id, "the case under test needs the actor to match"
    assert ev._event_visible_to(auth, event, _access(store, auth)) is False, \
        "a fully revoked principal still received every event it had created or last touched"


def test_an_expired_grant_stops_delivery(store):
    """Expiry is revocation the store performs on its own; the check has to see it too."""
    _grant(store, "g-1", CONTAINER)
    auth = _auth()
    assert ev._event_visible_to(auth, _event(), _access(store, auth)) is True

    doc = store.artifacts.get_artifact("g-1")
    doc["expires_at"] = "2000-01-01T00:00:00+00:00"
    store.artifacts.put_artifact(doc)

    assert ev._event_visible_to(auth, _event(), _access(store, auth)) is False


def test_a_stranger_never_sees_the_container(store):
    _grant(store, "g-1", CONTAINER)
    stranger = _auth("u-2")
    assert ev._event_visible_to(stranger, _event(actor_id="u-2"), _access(store, stranger)) is False


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 2 · Deny is honoured, at the artifact as well as at the container
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_a_deny_on_the_artifact_withholds_its_events(store):
    """A grant on the container must not carry an artifact its holder is explicitly refused.
    `check_access` applies deny-first at every level of its own walk, so asking it about the
    artifact is what honours the deny — there is no second probe to keep in step with it."""
    _grant(store, "g-allow", CONTAINER)
    _grant(store, "g-deny", ARTIFACT, effect="deny")
    auth = _auth()

    assert ev._event_visible_to(auth, _event(), _access(store, auth)) is False, \
        "an artifact-level deny grant was invisible to the event path"


def test_a_deny_on_the_container_withholds_its_events(store):
    _grant(store, "g-allow", CONTAINER)
    _grant(store, "g-deny", CONTAINER, effect="deny")
    auth = _auth()
    assert ev._event_visible_to(auth, _event(), _access(store, auth)) is False


def test_a_deny_elsewhere_withholds_nothing(store):
    """The control on deny: it must narrow the artifact it names and nothing else."""
    _grant(store, "g-allow", CONTAINER)
    _grant(store, "g-deny", "some-other-artifact", effect="deny")
    auth = _auth()
    assert ev._event_visible_to(auth, _event(), _access(store, auth)) is True


def test_a_deny_carried_by_a_grant_key_bundle_is_honoured(store):
    """A grant key is answered from the grants resolved at authentication — the same source
    `check_access` reads for it — so a bundle's deny member binds here too."""
    from mantle.entities.grant import Grant as G
    deny = G(id="g-deny", resource_id=ARTIFACT, grantee_type="grant_key", grantee_id="k",
             granted_by="admin", can_read=True, effect="deny")
    allow = G(id="g-allow", resource_id=CONTAINER, grantee_type="grant_key", grantee_id="k",
              granted_by="admin", can_read=True)
    auth = AuthContext(principal_id="k", principal_type="grant_key", grants=[allow, deny])
    assert ev._event_visible_to(auth, _event(actor_id=None), _access(store, auth)) is False


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 3 · Replay is gated exactly as live is
# ═════════════════════════════════════════════════════════════════════════════════════════════

class _FakeSocket:
    """Enough WebSocket for `_replay`: a connected state and a place for messages to land."""

    def __init__(self):
        from starlette.websockets import WebSocketState
        self.client_state = WebSocketState.CONNECTED
        self.sent: list = []

    async def send_json(self, message):
        self.sent.append(message)


def _socket_sub(client_id="s1", cursor=None):
    live = event_bus.LiveSubscription(filter=event_bus.EventFilter(),
                                      queue=asyncio.Queue(maxsize=8))
    return ev._Subscription(client_id, live.filter, live, cursor=cursor or event_bus.Cursor())


async def _replayed(store, auth, *, cursor=None):
    ws = _FakeSocket()
    sub = _socket_sub(cursor=cursor)
    await ev._replay(sub, ws, _session(store, auth), asyncio.Lock())
    return [m for m in ws.sent if "event" in m]


@pytest.fixture
def logged(store):
    """A durable log on this store, holding one event about the artifact."""
    log = event_bus.EventLog(store.conn, "node-a")
    event_bus.set_event_log(log)
    log.append(_event())
    yield log
    event_bus.set_event_log(None)


@pytest.mark.asyncio
async def test_replay_serves_a_granted_principal(store, logged):
    """The positive control for the replay path."""
    _grant(store, "g-1", CONTAINER)
    assert len(await _replayed(store, _auth())) == 1


@pytest.mark.asyncio
async def test_replay_serves_a_revoked_principal_nothing(store, logged):
    """A stored event is not privileged by having been stored. Composed with the plaintext bodies
    the feed used to carry, a revoked principal that reconnected with an old cursor could drain the
    history of everything it had ever touched."""
    grant = _grant(store, "g-1", CONTAINER)
    _revoke(store, grant)
    assert await _replayed(store, _auth()) == []


@pytest.mark.asyncio
async def test_a_client_supplied_cursor_does_not_bypass_the_read_check(store, logged):
    """`since` names a position in the log, never an authority over it. A client that resumes from
    a cursor it did not earn gets what the live feed would have given it, and no more."""
    _grant(store, "g-1", CONTAINER)
    stranger = _auth("u-2")
    assert await _replayed(store, stranger, cursor=event_bus.Cursor()) == [], \
        "a bare `since` replayed events the caller may not read"


@pytest.mark.asyncio
async def test_a_deny_is_honoured_on_replay_too(store, logged):
    _grant(store, "g-allow", CONTAINER)
    _grant(store, "g-deny", ARTIFACT, effect="deny")
    assert await _replayed(store, _auth()) == []


@pytest.mark.asyncio
async def test_replay_stops_at_its_ceiling_and_says_so(store, logged, monkeypatch):
    """The docstring claimed replay was "bounded per call by the log's own limit" while the loop
    drained the whole log. It is bounded now, and a truncated resume is reported with the position
    it reached rather than looking like the end of the stream."""
    monkeypatch.setattr(ev, "_REPLAY_MAX_EVENTS", 2)
    monkeypatch.setattr(ev, "_REPLAY_BATCH", 1)
    _grant(store, "g-1", CONTAINER)
    for _ in range(5):
        logged.append(_event())

    ws = _FakeSocket()
    await ev._replay(_socket_sub(), ws, _session(store, _auth()), asyncio.Lock())

    assert len([m for m in ws.sent if "event" in m]) == 2, "the ceiling did not bound the pass"
    truncation = [m for m in ws.sent if m.get("replay_truncated")]
    assert truncation, "replay stopped silently, which a client cannot tell from the end of the log"
    assert truncation[0]["cursor"], "the notice carries no position to resume from"


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 4 · The socket does not outlive its token
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_a_session_within_its_lease_does_not_re_resolve(store, monkeypatch):
    calls = []
    monkeypatch.setattr(ev, "resolve_auth", lambda *a, **k: calls.append(1))
    session = _session(store, _auth())
    assert session.valid() is True
    assert calls == [], "the token was re-resolved before its lease ran out"


def test_a_session_past_its_lease_re_resolves_the_token(store, monkeypatch):
    monkeypatch.setattr(ev, "_AUTH_TTL_SECONDS", 0.0)
    fresh = _auth("u-2")
    monkeypatch.setattr(ev, "resolve_auth", lambda *a, **k: fresh)

    session = _session(store, _auth())
    session._deadline = 0.0
    assert session.valid() is True
    assert session.auth is fresh, \
        "the socket kept the principal it connected with rather than the one the token resolves to"


def test_a_token_that_no_longer_resolves_ends_the_session(store, monkeypatch):
    def expired(*_a, **_k):
        raise ValueError("token expired")

    monkeypatch.setattr(ev, "resolve_auth", expired)
    session = _session(store, _auth())
    session._deadline = 0.0
    assert session.valid() is False, "a socket outlived the token that opened it"


def test_the_verdict_cache_follows_the_session_principal(store, monkeypatch):
    """Re-authentication is not a rule of its own — the verdicts simply read `session.auth`, so a
    narrowed principal narrows what is checked from the next expiry onward."""
    _grant(store, "g-1", CONTAINER, grantee=OWNER)
    session = _session(store, _auth(OWNER))
    _verdicts, access = ev._container_access(session)
    assert access.may_read(CONTAINER) is True

    session.auth = _auth("u-2")
    assert access.may_read(CONTAINER) is False
