"""The event path asks `check_access` and nothing else — and asks it late enough to be right.

README invariant #2: *authorization is decided only by the light cone and grants*. An endpoint
that answers a **different** question about the same artifact is a second authorization rule, and
a second rule is a wider one the moment the two disagree. `/events` serves artifact descriptors,
so the question it must ask is the one every artifact route asks:
`services.dependencies.check_access(auth, artifact_id, "read", store_db)`.

Two properties are pinned here and they pull in opposite directions, which is why they are tested
together:

* **Agreement (never wider).** For every shape `check_access` refuses — a `propagate` mask that
  prunes `read` at an origin edge, a deny on an ancestor the event does not name, an unscoped
  grant, a bare actor match — the feed refuses too. Each negative sits beside the positive control
  that makes the same setup deliver, so "refused" cannot be satisfied by a feed that never started.

* **Completeness (never narrower than the store).** A creation and the owner grant that authorizes
  it are two transactions, and the announcement is made between them. A verdict taken in that
  window is a verdict about a state no reader will ever observe, so it is provisional: the socket
  re-asks for as long as the grace allows, and never caches the refusal in the meantime. The
  question is still `check_access` — the grace buys a re-ask, never an answer.
"""
from __future__ import annotations

import ast
import asyncio
import pathlib
import sys
import time

import pytest
from fastapi import HTTPException

BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle.db import lattice_api                                       # noqa: E402
from mantle.entities.artifact import Artifact                           # noqa: E402
from mantle.entities.grant import Grant                                 # noqa: E402
from mantle.events import event_bus                                     # noqa: E402
from mantle.routers import events_router as ev                          # noqa: E402
from mantle.services.dependencies import AuthContext, check_access      # noqa: E402

OWNER = "u-1"
#: `W` contains `A` through an origin edge — the containment the light cone actually walks.
WORKSPACE = "ws-1"
ARTIFACT = "a-1"
#: A second child of `W`, versioned: `V` is a version of the lineage rooted at `R`.
ROOT = "r-1"
VERSION = "v-1"


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Fixtures — a lattice with real origin edges, because the edge is what carries authority
# ═════════════════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def store(tmp_path):
    db = lattice_api.LatticeDatabase(str(tmp_path / "events-acl.db"), origin="node-a")
    lattice_api.create_artifact(db, Artifact(
        id=WORKSPACE, root_id=WORKSPACE, collection_id="", name="workspace", content="",
        created_by=OWNER, modified_by=OWNER))
    lattice_api.create_artifact(db, Artifact(
        id=ARTIFACT, root_id=ARTIFACT, collection_id=WORKSPACE, name="memo", content="",
        created_by=OWNER, modified_by=OWNER))
    lattice_api.create_artifact(db, Artifact(
        id=VERSION, root_id=ROOT, collection_id=WORKSPACE, name="report", content="",
        created_by=OWNER, modified_by=OWNER))
    lattice_api.add_artifact_to_collection(db, WORKSPACE, ARTIFACT)
    lattice_api.add_artifact_to_collection(db, WORKSPACE, ROOT)
    return db


def _auth(user_id=OWNER, **kw):
    return AuthContext(principal_id=user_id, principal_type="user", user_id=user_id, **kw)


def _session(store, auth):
    return ev._Session("test-token", auth, store)


def _access(store, auth):
    _verdicts, access = ev._container_access(_session(store, auth))
    return access


def _grant(store, gid, resource_id, *, effect="allow", grantee=OWNER):
    return lattice_api.create_grant(store, Grant(
        id=gid, resource_id=resource_id, grantee_type="user", grantee_id=grantee,
        granted_by="admin", can_read=True, effect=effect, state="active"))


def _event(*, artifact_id=ARTIFACT, container_id=WORKSPACE, actor_id=OWNER,
           name="artifact.updated", **kw):
    return event_bus.Event(name=name, payload={"artifact": {"id": artifact_id}},
                           artifact_id=artifact_id, container_id=container_id,
                           actor_id=actor_id, **kw)


def _route_would_serve(store, auth, artifact_id) -> bool:
    """What the ordinary read path answers about this artifact — the authority under test."""
    try:
        check_access(auth, artifact_id, "read", store)
        return True
    except HTTPException:
        return False


def _agree(store, auth, event) -> bool:
    """Assert the feed and the route give the same answer, and return it."""
    feed = ev._event_visible_to(auth, event, _access(store, auth))
    route = _route_would_serve(store, auth, event.artifact_id)
    assert feed is route, (
        f"the event path answered {feed} where check_access answers {route} for "
        f"{event.artifact_id!r} — the feed is a second authorization rule")
    return feed


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 1 · The demonstrated escalation: an origin edge whose `propagate` mask prunes `read`
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_a_container_grant_reaches_a_child_the_edge_lets_read_through(store):
    """The positive control. Every refusal below is the same setup with one thing changed, so
    without this one "refused" would be indistinguishable from a feed that never delivered."""
    _grant(store, "g-ws", WORKSPACE)
    assert _agree(store, _auth(), _event()) is True


def test_a_propagate_mask_that_prunes_read_prunes_the_event_too(store):
    """`propagate=["update"]` is a supported edge configuration (`db/test_lattice_api.py`,
    `test_light_cone_propagation_mask`). `check_access` breaks its upward walk at that edge, so the
    holder of a container grant cannot read the child — and must not receive its events either.

    The container fallback is what made these two disagree: it asked *may this principal read the
    container*, which is a wider question than *may it read the artifact*."""
    lattice_api.add_artifact_to_collection(store, WORKSPACE, ARTIFACT, propagate=["update"])
    _grant(store, "g-ws", WORKSPACE)
    auth = _auth()

    assert _route_would_serve(store, auth, ARTIFACT) is False, \
        "the case under test needs check_access to refuse; the prune did not take"
    assert _agree(store, auth, _event()) is False


def test_the_prune_is_per_action_and_read_still_flows_where_the_mask_allows_it(store):
    """The control on the prune: a mask that names `read` narrows nothing here."""
    lattice_api.add_artifact_to_collection(store, WORKSPACE, ARTIFACT, propagate=["read"])
    _grant(store, "g-ws", WORKSPACE)
    assert _agree(store, _auth(), _event()) is True


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 2 · A deny on an ancestor the event does not name
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_a_deny_on_the_artifacts_root_withholds_its_events(store):
    """`check_access` applies deny-first at every level of its walk, including the artifact's
    `root_id`. An event names only the artifact and its container, so a probe that asks about
    exactly those two ids cannot see a deny one level in — and a container grant then carries an
    artifact whose lineage its holder is explicitly refused."""
    _grant(store, "g-ws", WORKSPACE)
    _grant(store, "g-deny-root", ROOT, effect="deny")
    auth = _auth()

    assert _route_would_serve(store, auth, VERSION) is False, \
        "the case under test needs check_access to refuse at the root"
    assert _agree(store, auth, _event(artifact_id=VERSION)) is False


def test_a_deny_on_a_grandparent_container_withholds_its_events(store):
    """The same shape one hop further out: the deny sits on a container the event never names."""
    lattice_api.create_artifact(store, Artifact(
        id="ws-outer", root_id="ws-outer", collection_id="", name="outer", content="",
        created_by=OWNER, modified_by=OWNER))
    lattice_api.add_artifact_to_collection(store, "ws-outer", WORKSPACE)
    _grant(store, "g-outer", "ws-outer")
    _grant(store, "g-deny-ws", WORKSPACE, effect="deny")
    auth = _auth()

    assert _route_would_serve(store, auth, ARTIFACT) is False
    assert _agree(store, auth, _event()) is False


def test_a_grant_two_hops_up_still_reaches_the_event(store):
    """The control: the walk itself is not what was broken, so an unpruned two-hop chain delivers."""
    lattice_api.create_artifact(store, Artifact(
        id="ws-outer", root_id="ws-outer", collection_id="", name="outer", content="",
        created_by=OWNER, modified_by=OWNER))
    lattice_api.add_artifact_to_collection(store, "ws-outer", WORKSPACE)
    _grant(store, "g-outer", "ws-outer")
    assert _agree(store, _auth(), _event()) is True


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 3 · An unscoped grant is not a grant on anything
# ═════════════════════════════════════════════════════════════════════════════════════════════

def _key_auth(*grants):
    return AuthContext(principal_id="k-1", principal_type="grant_key", user_id=None,
                       grants=list(grants), grant_key_id="k-1")


def test_an_unscoped_read_grant_does_not_reach_an_event(store):
    """`check_access` filters a key's resolved bundle on `g.resource_id == resource_id`, so a grant
    naming no resource authorizes no resource. The feed used to fall through to a loop that
    returned True for any event on any unscoped `can_read` grant — a platform-wide viewer nobody
    granted."""
    unscoped = Grant(id="g-open", resource_id=None, grantee_type="grant_key", grantee_id="k-1",
                     granted_by="admin", can_read=True)
    auth = _key_auth(unscoped)

    assert _route_would_serve(store, auth, ARTIFACT) is False, \
        "the case under test needs check_access to refuse an unscoped grant"
    assert _agree(store, auth, _event(actor_id=None)) is False


def test_a_scoped_key_grant_still_reaches_its_own_artifact(store):
    """The control: a key bundle that names the resource is honoured, from the same source
    `check_access` reads it."""
    scoped = Grant(id="g-scoped", resource_id=ARTIFACT, grantee_type="grant_key", grantee_id="k-1",
                   granted_by="admin", can_read=True)
    assert _agree(store, _key_auth(scoped), _event(actor_id=None)) is True


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 4 · Being the actor is not a grant, for any principal
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_a_non_user_principal_is_not_authorized_by_being_the_actor(store):
    """The actor fast-path survived for principals with no `user_id` — reachable by
    `principal_type` "service" and "server", and "server" is a dead branch now that the registry
    and credential planes are gone. `actor_id` is a provenance column: it records who touched an
    artifact and does not change when the authority to see it is withdrawn."""
    actor = AuthContext(principal_id="svc-1", principal_type="service", user_id=None, grants=[])
    event = _event(actor_id="svc-1")

    assert event.actor_id == actor.principal_id, "the case under test needs the actor to match"
    assert _route_would_serve(store, actor, ARTIFACT) is False
    assert _agree(store, actor, event) is False


def test_the_system_consumer_exception_is_the_only_standing_one(store):
    """The one designed exception to invariant #2, named in `auth_service` and asserted here so
    the exception is a list rather than a habit."""
    assert ev._SYSTEM_EVENT_CONSUMERS == {"crystal"}
    gateway = AuthContext(principal_id="svc-1", principal_type="service", user_id=None,
                          grants=[], authority="crystal")
    assert ev._event_visible_to(gateway, _event(actor_id=None), _access(store, gateway)) is True

    other = AuthContext(principal_id="svc-2", principal_type="service", user_id=None,
                        grants=[], authority="something-else")
    assert ev._event_visible_to(other, _event(actor_id=None), _access(store, other)) is False


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 5 · The container fallback is narrow: only where there is no artifact to ask about
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_an_event_about_a_vanished_artifact_falls_back_to_its_container(store):
    """A hard delete is announced after the row is gone, so `check_access` has nothing to answer
    about. The fallback exists for exactly that and is reached only there."""
    _grant(store, "g-ws", WORKSPACE)
    gone = _event(artifact_id="a-deleted", name="artifact.deleted")
    assert ev._event_visible_to(_auth(), gone, _access(store, _auth())) is True


def test_a_deny_on_a_vanished_artifact_still_withholds_its_delete(store):
    """The fallback is not an alternative grant: an explicit deny naming the vanished artifact
    withholds it, since `check_access` can no longer apply deny-first on a row that is gone."""
    _grant(store, "g-ws", WORKSPACE)
    _grant(store, "g-deny", "a-deleted", effect="deny")
    gone = _event(artifact_id="a-deleted", name="artifact.deleted")
    assert ev._event_visible_to(_auth(), gone, _access(store, _auth())) is False


def test_the_fallback_never_answers_for_an_artifact_the_store_knows(store):
    """The narrowing, stated directly: with the artifact present, the container is not consulted
    at all — which is what stops the fallback from being a second grant."""
    lattice_api.add_artifact_to_collection(store, WORKSPACE, ARTIFACT, propagate=["update"])
    _grant(store, "g-ws", WORKSPACE)
    access = _access(store, _auth())
    assert access.may_read(WORKSPACE) is True, "the container is readable; only the child is not"
    assert ev._event_visible_to(_auth(), _event(), access) is False


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 6 · The rule is one call, held to the tree
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_the_visibility_check_holds_no_grant_logic_of_its_own():
    """Invariant #2 as a property of the source: the only thing `_event_visible_to` may consult is
    the verdict cache in front of `check_access`. A grant read here is a second rule by
    construction, whatever it currently happens to conclude."""
    source = pathlib.Path(ev.__file__).read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(source))
              if isinstance(n, ast.FunctionDef) and n.name == "_event_visible_to")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    forbidden = {"_check_grant_permission", "grants", "grant_is_deny", "can_read", "resource_id"}
    assert not (names & forbidden), (
        f"_event_visible_to consults {sorted(names & forbidden)} — the event path must ask "
        f"check_access and nothing else")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 7 · Completeness: the creation race, live
# ═════════════════════════════════════════════════════════════════════════════════════════════

class _FakeSocket:
    """Enough WebSocket for `_pump_subscription`: a connected state and somewhere to send."""

    def __init__(self):
        from starlette.websockets import WebSocketState
        self.client_state = WebSocketState.CONNECTED
        self.sent: list = []

    async def send_json(self, message):
        self.sent.append(message)

    async def close(self, **_kw):
        from starlette.websockets import WebSocketState
        self.client_state = WebSocketState.DISCONNECTED


def _socket_sub(client_id="s1"):
    live = event_bus.LiveSubscription(filter=event_bus.EventFilter(),
                                      queue=asyncio.Queue(maxsize=16))
    return ev._Subscription(client_id, live.filter, live)


async def _frames(ws, count, *, timeout=3.0):
    """Wait for *count* event frames, or as many as arrive before *timeout*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        received = [m for m in ws.sent if "event" in m]
        if len(received) >= count:
            return received
        await asyncio.sleep(0.02)
    return [m for m in ws.sent if "event" in m]


TOP = "top-1"


def _create_top_level_row(store):
    """The row write of a top-level create. Its owner grant is a *second* transaction, and the
    announcement is made between the two — which is the window under test."""
    lattice_api.create_artifact(store, Artifact(
        id=TOP, root_id=TOP, collection_id="", name="fresh", content="",
        created_by=OWNER, modified_by=OWNER))


@pytest.mark.asyncio
async def test_a_top_level_creation_is_delivered_live(store):
    """A client watching the whole feed must see a top-level artifact appear.

    For a top-level artifact the event's `container_id` *is* the artifact, so there is no wider
    subject to fall back to: the refusal taken between the row write and the owner grant was the
    whole answer, and the create was dropped with nothing logged."""
    _create_top_level_row(store)
    auth = _auth()
    ws, sub = _FakeSocket(), _socket_sub()
    task = asyncio.create_task(ev._pump_subscription(sub, ws, _session(store, auth),
                                                     asyncio.Lock()))
    try:
        sub.queue.put_nowait(event_bus.Event(
            name="artifact.created", payload={"artifact": {"id": TOP}},
            artifact_id=TOP, container_id=TOP, actor_id=OWNER))
        await asyncio.sleep(0.1)
        _grant(store, "g-top", TOP)                      # the owner grant lands, as it does live

        frames = await _frames(ws, 1)
        assert len(frames) == 1, \
            "the creation of a top-level artifact never reached its own creator's feed"
        assert frames[0]["event"] == "artifact.created"
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_a_write_that_follows_a_creation_within_the_ttl_is_not_lost(store):
    """The refusal must not outlive the state it was taken in.

    A verdict cached for `_ACCESS_TTL_SECONDS` turns one mistimed refusal into a blackout on that
    artifact for the whole TTL — every subsequent write about it dropped, silently."""
    _create_top_level_row(store)
    auth = _auth()
    ws, sub = _FakeSocket(), _socket_sub()
    task = asyncio.create_task(ev._pump_subscription(sub, ws, _session(store, auth),
                                                     asyncio.Lock()))
    try:
        sub.queue.put_nowait(event_bus.Event(
            name="artifact.created", payload={"artifact": {"id": TOP}},
            artifact_id=TOP, container_id=TOP, actor_id=OWNER))
        await asyncio.sleep(0.1)
        _grant(store, "g-top", TOP)
        await _frames(ws, 1)

        # Well inside `_ACCESS_TTL_SECONDS`, which is what made this the sticky half.
        assert ev._ACCESS_TTL_SECONDS >= 1.0, "the case under test needs a real cache lifetime"
        sub.queue.put_nowait(event_bus.Event(
            name="artifact.updated", payload={"artifact": {"id": TOP}},
            artifact_id=TOP, container_id=TOP, actor_id=OWNER))

        frames = await _frames(ws, 2)
        assert [f["event"] for f in frames] == ["artifact.created", "artifact.updated"], \
            "a cached refusal blacked the artifact out for the rest of the verdict TTL"
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_the_grace_buys_a_re_ask_and_never_an_answer(store):
    """The completeness fix must not widen anything: a creation whose grant never arrives stays
    unreadable, and a stranger's creation is never delivered on the strength of being fresh."""
    _create_top_level_row(store)
    stranger = _auth("u-2")
    ws, sub = _FakeSocket(), _socket_sub()
    task = asyncio.create_task(ev._pump_subscription(sub, ws, _session(store, stranger),
                                                     asyncio.Lock()))
    try:
        sub.queue.put_nowait(event_bus.Event(
            name="artifact.created", payload={"artifact": {"id": TOP}},
            artifact_id=TOP, container_id=TOP, actor_id="u-2"))
        await asyncio.sleep(ev._CREATION_GRACE_SECONDS + 0.3)
        assert [m for m in ws.sent if "event" in m] == [], \
            "the creation grace delivered an event no grant authorizes"
    finally:
        task.cancel()


def test_a_stale_creation_is_not_treated_as_racing_its_own_authorization():
    """The grace is bounded by the event's own age: a creation replayed or relayed long after the
    fact is a settled refusal, not a race."""
    fresh = event_bus.Event(name="artifact.created", payload={}, artifact_id=TOP,
                            container_id=TOP, actor_id=OWNER)
    stale = event_bus.Event(name="artifact.created", payload={}, artifact_id=TOP,
                            container_id=TOP, actor_id=OWNER,
                            ts=time.time() - (ev._CREATION_GRACE_SECONDS + 60))
    other = event_bus.Event(name="artifact.updated", payload={}, artifact_id=TOP,
                            container_id=TOP, actor_id=OWNER)
    auth = _auth()

    assert ev._races_its_own_authorization(auth, fresh) is True
    assert ev._races_its_own_authorization(auth, stale) is False
    assert ev._races_its_own_authorization(auth, other) is False
    assert ev._races_its_own_authorization(_auth("u-2"), fresh) is False
