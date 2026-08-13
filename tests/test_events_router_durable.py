"""The `/events` durable path: resume, ack, deduplication, and the permissions each one needs.

The WebSocket transport is awkward to drive through httpx/ASGITransport, so this exercises the
router's building blocks directly, the way `tests/test_events_router.py` already does for
`_parse_filter` and `_event_visible_to`.

What matters here is that the durable class does not quietly acquire authority the live class does
not have. Resuming a subscription is a `read` of an artifact; acking one is an `update` of it.
They are different permissions on the same resource and the split is the point — a consumer shared
a read-only subscription may follow it and may not move anyone else's position.
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle.events import event_bus                                            # noqa: E402
from mantle.db import lattice_api                                       # noqa: E402
from mantle.entities.subscription import (                              # noqa: E402
    Subscription, load_subscription, save_subscription,
)
from mantle.routers import events_router as ev                          # noqa: E402


@pytest.fixture
def store(tmp_path):
    return lattice_api.LatticeDatabase(str(tmp_path / "durable.db"), origin="node-a")


class _Grant:
    def __init__(self, resource_id=None, **flags):
        for action in ("create", "read", "update", "delete", "evict", "invoke",
                       "add", "share", "admin"):
            setattr(self, f"can_{action}", flags.get(action, False))
        self.resource_id = resource_id
        self.effect = "allow"


def _auth(user_id=None, grants=()):
    return SimpleNamespace(user_id=user_id, principal_id=user_id, grants=list(grants),
                           principal_type="user", authority=None)


def _sub(store, owner="u-1", **kw):
    return save_subscription(store, Subscription(name="w", owner_id=owner, **kw))


def _socket_sub(client_id="s1", durable_id=None, cursor=None):
    """A router-side subscription handle with no socket behind it."""
    live = event_bus.LiveSubscription(filter=event_bus.EventFilter(),
                                      queue=__import__("asyncio").Queue(maxsize=8))
    return ev._Subscription(client_id, live.filter, live, durable_id=durable_id,
                            cursor=cursor or event_bus.Cursor())


# ---------------------------------------------------------------------------
# Delivery class is visible, not inferred
# ---------------------------------------------------------------------------

def test_a_plain_subscription_is_live():
    assert _socket_sub().delivery == event_bus.DELIVERY_LIVE


def test_naming_a_subscription_artifact_makes_it_durable():
    assert _socket_sub(durable_id="sub-1").delivery == event_bus.DELIVERY_DURABLE


def test_supplying_a_cursor_alone_also_makes_it_durable():
    """Resume-by-cursor without a stored subscription is a legal, weaker durable shape: the client
    keeps its own position. It still replays, so it is at-least-once for as long as the client
    remembers where it was."""
    assert _socket_sub(cursor=event_bus.Cursor.of({"node-a": 3})).delivery == \
        event_bus.DELIVERY_DURABLE


# ---------------------------------------------------------------------------
# Resume needs `read`
# ---------------------------------------------------------------------------

def test_an_owner_may_resume_its_own_subscription(store):
    sub = _sub(store, owner="u-1")
    assert ev._load_durable(store, sub.id, _auth("u-1")) is not None


def test_a_stranger_may_not_resume_someone_elses_subscription(store):
    sub = _sub(store, owner="u-1")
    assert ev._load_durable(store, sub.id, _auth("u-2")) is None


def test_a_read_grant_on_the_subscription_is_enough_to_resume(store):
    """Sharing a subscription is sharing an artifact — no second mechanism."""
    sub = _sub(store, owner="u-1")
    shared = _auth("u-2", grants=[_Grant(resource_id=sub.id, read=True)])
    assert ev._load_durable(store, sub.id, shared) is not None


def test_an_absent_and_an_unauthorized_subscription_answer_identically(store):
    """Distinguishing them would make this socket an oracle for the existence of subscriptions the
    caller cannot see — the disclosure the artifact routes already refuse by 404-ing."""
    sub = _sub(store, owner="u-1")
    stranger = _auth("u-2")
    assert ev._load_durable(store, sub.id, stranger) is None
    assert ev._load_durable(store, "no-such-id", stranger) is None


# ---------------------------------------------------------------------------
# Ack needs `update`
# ---------------------------------------------------------------------------

def test_the_owner_can_ack(store):
    sub = _sub(store, owner="u-1")
    socket = _socket_sub(durable_id=sub.id, cursor=event_bus.Cursor.of({"node-a": 4}))
    ev._persist_cursor(store, socket, None, _auth("u-1"))
    assert load_subscription(store, sub.id).cursor.to_dict() == {"node-a": 4}


def test_a_read_only_follower_cannot_move_the_stored_cursor(store):
    """The security-relevant half of the split. A follower may see the stream and may not
    acknowledge on the owner's behalf, which would make the owner skip what it never received."""
    sub = _sub(store, owner="u-1")
    follower = _auth("u-2", grants=[_Grant(resource_id=sub.id, read=True)])
    socket = _socket_sub(durable_id=sub.id, cursor=event_bus.Cursor.of({"node-a": 4}))

    ev._persist_cursor(store, socket, None, follower)
    assert load_subscription(store, sub.id).cursor.to_dict() == {}, \
        "a read-only follower advanced the owner's cursor"


def test_an_update_grant_lets_a_second_consumer_ack(store):
    sub = _sub(store, owner="u-1")
    worker = _auth("u-2", grants=[_Grant(resource_id=sub.id, read=True, update=True)])
    socket = _socket_sub(durable_id=sub.id, cursor=event_bus.Cursor.of({"node-a": 6}))

    ev._persist_cursor(store, socket, None, worker)
    assert load_subscription(store, sub.id).cursor.to_dict() == {"node-a": 6}


def test_acking_without_naming_a_cursor_acks_what_was_sent(store):
    """"Everything you have sent me" is the common case; making the client echo a cursor it just
    received would only add a way to get it wrong."""
    sub = _sub(store, owner="u-1")
    socket = _socket_sub(durable_id=sub.id, cursor=event_bus.Cursor.of({"node-a": 9}))
    assert ev._persist_cursor(store, socket, None, _auth("u-1")).to_dict() == {"node-a": 9}


def test_an_ack_merges_rather_than_replaces(store):
    """A client acking one origin must not drop this socket's progress on another."""
    sub = _sub(store, owner="u-1")
    socket = _socket_sub(durable_id=sub.id, cursor=event_bus.Cursor.of({"node-a": 5}))
    merged = ev._persist_cursor(store, socket, "node-b:2", _auth("u-1"))
    assert merged.to_dict() == {"node-a": 5, "node-b": 2}


# ---------------------------------------------------------------------------
# The replay/live overlap
# ---------------------------------------------------------------------------

def test_replay_and_live_overlap_is_removed_exactly():
    """Live is attached BEFORE the log is read, so the window between them is duplicated rather
    than lost. This is where the duplicate goes, using the log's own ordering."""
    socket = _socket_sub(cursor=event_bus.Cursor.of({"node-a": 5}))
    already = event_bus.Event(name="artifact.created", payload={}, origin="node-a", seq=5)
    fresh = event_bus.Event(name="artifact.created", payload={}, origin="node-a", seq=6)

    assert ev._already_served(socket, already) is True
    assert ev._already_served(socket, fresh) is False


def test_an_unlogged_event_is_never_treated_as_a_duplicate():
    """No proper time means it was never in the log, so it cannot be a replay of anything."""
    socket = _socket_sub(cursor=event_bus.Cursor.of({"node-a": 5}))
    assert ev._already_served(
        socket, event_bus.Event(name="artifact.created", payload={})) is False


def test_a_peer_origin_is_not_suppressed_by_this_nodes_cursor():
    """Seqs from two observers are not comparable; a shared axis would silently drop a peer's
    events whose numbers happen to sit below the local mark."""
    socket = _socket_sub(cursor=event_bus.Cursor.of({"node-a": 5}))
    peer = event_bus.Event(name="artifact.created", payload={}, origin="node-b", seq=1)
    assert ev._already_served(socket, peer) is False


# ---------------------------------------------------------------------------
# The delivered message carries what a client needs to ack
# ---------------------------------------------------------------------------

def test_a_delivered_event_carries_its_cursor():
    """So a durable client acks by echoing, with no client-side arithmetic over (origin, seq)."""
    socket = _socket_sub()
    event = event_bus.Event(name="artifact.created", payload={}, origin="node-a", seq=12)
    assert ev._event_message(socket, event)["cursor"] == "node-a:12"


def test_an_unlogged_event_carries_no_cursor():
    socket = _socket_sub()
    assert "cursor" not in ev._event_message(
        socket, event_bus.Event(name="artifact.created", payload={}))


def test_a_replayed_event_is_marked_as_such():
    """A consumer that treats a replay differently from a live event can; one that does not, need
    not look."""
    socket = _socket_sub()
    event = event_bus.Event(name="artifact.created", payload={}, origin="node-a", seq=1)
    assert ev._event_message(socket, event, replay=True)["replay"] is True
    assert "replay" not in ev._event_message(socket, event)


# ---------------------------------------------------------------------------
# One ACL, both paths
# ---------------------------------------------------------------------------

def test_replay_and_live_share_one_container_access_check():
    """Two checks would be two chances to disagree about who may see what, and the disagreement
    would surface as a stored event being visible where its live twin was not."""
    import ast

    source = (BACKEND / "src" / "mantle" / "routers" / "events_router.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    callers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                and sub.func.id == "_container_access" for sub in ast.walk(node))
    }
    assert {"_pump_subscription", "_replay"} <= callers, (
        f"the live pump and the replay pass do not share one access check (callers: {callers}). "
        f"A replayed event is not privileged by having been stored.")
