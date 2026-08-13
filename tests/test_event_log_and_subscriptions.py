"""The durable half of the bus: an append-only log in proper time, and subscriptions as artifacts.

Live delivery is best-effort and that is fine for a UI, but it is not a product surface an external
consumer can build on: a consumer that disconnects must be able to come back and find out what it
missed. That is what these two pieces provide, and what this suite pins.

  §1  proper time — the log's ordering is monotonic, gap-free per origin, and survives rollback
  §2  replay — a cursor resumes exactly, filters apply, and nothing is served twice
  §3  subscriptions as artifacts — the codec round-trips, and the cursor persists through the
      ordinary artifact path rather than a private one
  §4  retention — trimming is an operator's decision and is never taken automatically

The load-bearing claim throughout is the at-least-once contract: read forward from a cursor, act,
then advance. Redelivery is permitted; loss is not.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle.events import event_bus                                            # noqa: E402
from mantle.db import lattice_api                                       # noqa: E402
from mantle.entities.subscription import (                              # noqa: E402
    SUBSCRIPTION_CONTENT_TYPE, Subscription, advance_cursor, is_subscription,
    load_subscription, save_subscription,
)


@pytest.fixture
def store(tmp_path):
    return lattice_api.LatticeDatabase(str(tmp_path / "log.db"), origin="node-a")


@pytest.fixture
def log(store):
    made = event_bus.open_event_log(store)
    yield made
    event_bus.set_event_log(None)


def _event(name: str, **kw) -> event_bus.Event:
    return event_bus.Event(name=name, payload=kw.pop("payload", {"n": name}), **kw)


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 1 · Proper time
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_the_log_allocates_monotonic_gap_free_seqs_for_its_origin(log):
    """`seq > cursor` is only a correct resume if seqs are injective and gap-free.

    A wall clock is neither — two events can tie, and a skewed clock can pin a cursor past every
    real row. This is the same reasoning `db/seq.py` gives for allocating from the store
    rather than reading a clock, applied to the feed.
    """
    seqs = []
    for i in range(5):
        cursor = log.append(_event("artifact.created", payload={"i": i}))
        assert cursor is not None
        seqs.append(dict(cursor.marks)["node-a"])
    assert seqs == [1, 2, 3, 4, 5], f"seqs are not gap-free and monotonic: {seqs}"


def test_appending_stamps_proper_time_onto_the_event(log):
    """The event carries its own coordinates afterwards, so a live subscriber that receives it can
    ack without a second lookup."""
    event = _event("artifact.updated")
    assert event.origin is None and event.seq is None
    log.append(event)
    assert event.origin == "node-a"
    assert event.seq == 1


def test_the_log_does_not_borrow_the_lattices_row_allocator(store, log):
    """The log has its own counter, and this is why.

    `seq.seq_accounting` underwrites `live_rows + vacated == last_seq` over `vertex ∪ edge` — the
    identity that makes a row lost outside the write path detectable at all. Event rows live in
    neither table, so drawing their seqs from the shared allocator would inflate `last_seq` and
    leave every store permanently reporting unaccounted rows. Here the store's accounting is
    untouched by a hundred events.
    """
    from mantle.db import seq as lattice_seq

    before = lattice_seq.seq_accounting(store.conn, "node-a", scan=True)
    for i in range(100):
        log.append(_event("artifact.created", payload={"i": i}))
    after = lattice_seq.seq_accounting(store.conn, "node-a", scan=True)

    assert after["last_seq"] == before["last_seq"], \
        "appending events moved the lattice's row allocator, which breaks loss detection"
    assert after["balanced"], f"the store's accounting is no longer balanced: {after}"


def test_a_cursor_never_moves_backwards():
    """`Cursor.advanced` takes the max, for the reason `SeqAllocator.flush` does: a position that
    can regress can redeliver without bound."""
    cursor = event_bus.Cursor.of({"node-a": 10})
    assert cursor.advanced("node-a", 4).to_dict() == {"node-a": 10}
    assert cursor.advanced("node-a", 11).to_dict() == {"node-a": 11}


def test_a_cursor_round_trips_through_its_compact_string():
    cursor = event_bus.Cursor.of({"node-a": 7, "node-b": 3})
    assert event_bus.Cursor.parse(str(cursor)) == cursor


def test_an_unparseable_cursor_reads_as_further_back_rather_than_raising():
    """A cursor is a client-held hint. Degrading to an earlier position redelivers, which
    at-least-once permits; raising would strand the subscriber with no way back."""
    assert event_bus.Cursor.parse("garbage").marks == ()
    assert event_bus.Cursor.parse(None).marks == ()
    assert event_bus.Cursor.parse("node-a:notanumber").marks == ()


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 2 · Replay
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_a_subscriber_resumes_from_its_cursor_and_loses_nothing(log):
    """The whole point of the durable class: disconnect, miss things, come back, get them."""
    for i in range(3):
        log.append(_event("artifact.created", payload={"i": i}))

    seen = log.read_since("")
    assert [e.payload["i"] for e in seen] == [0, 1, 2]

    cursor = event_bus.Cursor()
    for event in seen:
        cursor = cursor.advanced(event.origin, event.seq)

    # ... the subscriber is away while three more happen ...
    for i in range(3, 6):
        log.append(_event("artifact.created", payload={"i": i}))

    resumed = log.read_since(cursor)
    assert [e.payload["i"] for e in resumed] == [3, 4, 5], \
        "resuming from the cursor did not deliver exactly what was missed"


def test_reading_from_a_cursor_never_redelivers_what_it_covers(log):
    """Strictly after, not at-or-after. An acked event redelivered on every reconnect would make
    the cursor useless as an ack."""
    log.append(_event("artifact.created"))
    head = log.head()
    assert log.read_since(head) == []


def test_replay_honours_the_subscriptions_filter(log):
    """The filter is part of the subscription, so it applies to stored events exactly as it does to
    live ones — otherwise a resume would deliver a different stream than the one subscribed to."""
    log.append(_event("artifact.created", container_id="ws-1"))
    log.append(_event("artifact.deleted", container_id="ws-1"))
    log.append(_event("artifact.created", container_id="ws-2"))

    only_ws1 = log.read_since("", event_filter=event_bus.EventFilter(container_id="ws-1"))
    assert {e.name for e in only_ws1} == {"artifact.created", "artifact.deleted"}
    assert all(e.container_id == "ws-1" for e in only_ws1)

    only_creates = log.read_since(
        "", event_filter=event_bus.EventFilter(event_names=["artifact.created"]))
    assert [e.container_id for e in only_creates] == ["ws-1", "ws-2"]


def test_replay_is_bounded_so_a_long_absence_drains_in_batches(log):
    """An unbounded drain would hand a consumer that has been away for a month one enormous
    response. It advances and asks again — the same total work, in pieces it can survive."""
    for i in range(50):
        log.append(_event("artifact.created", payload={"i": i}))
    first = log.read_since("", limit=20)
    assert len(first) == 20
    assert [e.payload["i"] for e in first] == list(range(20))


def test_events_from_two_origins_keep_separate_proper_time(store, log):
    """Seqs from two observers are not comparable, so the cursor is per origin.

    A single global position would make a replicated peer event either starve a reader or skip past
    unread local ones, depending only on which integer happened to be larger.
    """
    peer = event_bus.EventLog(store.conn, "node-b", ensure=False)
    log.append(_event("artifact.created", payload={"who": "a1"}))
    peer.append(_event("artifact.created", payload={"who": "b1"}))
    log.append(_event("artifact.created", payload={"who": "a2"}))

    caught_up_on_a = event_bus.Cursor.of({"node-a": 2})
    remaining = log.read_since(caught_up_on_a)
    assert [e.payload["who"] for e in remaining] == ["b1"], \
        "a cursor for one origin suppressed or leaked the other origin's events"


def test_the_log_is_wired_into_publish_when_installed(log):
    """Installing the log is all it takes — `publish_event` appends before it fans out, so a
    durable subscriber's cursor covers an event even when every live queue is full."""
    event_bus.set_event_log(log)
    asyncio.run(event_bus.publish_event(_event("artifact.created", container_id="ws-9")))
    stored = log.read_since("")
    assert [e.name for e in stored] == ["artifact.created"]
    assert stored[0].container_id == "ws-9"


def test_no_log_installed_is_a_legal_configuration():
    """Live-only is a supported shape, exactly as `backplane=None` is. Publishing must not require
    a log to exist."""
    event_bus.set_event_log(None)
    assert event_bus.event_log() is None
    asyncio.run(event_bus.publish_event(_event("artifact.created")))


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 3 · Subscriptions are artifacts
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_a_subscription_round_trips_through_its_artifact():
    sub = Subscription(
        name="tree watcher",
        filter=event_bus.EventFilter(container_id="ws-1", event_names=["artifact.*"]),
        cursor="node-a:12",
        container_id="ws-1",
        owner_id="u-1",
    )
    back = Subscription.from_artifact(sub.to_artifact())
    assert back is not None
    assert back.filter.container_id == "ws-1"
    assert back.filter.event_names == ["artifact.*"]
    assert back.cursor.to_dict() == {"node-a": 12}
    assert back.delivery == event_bus.DELIVERY_DURABLE
    assert back.owner_id == "u-1"


def test_the_body_rides_in_context_so_it_needs_no_content_key():
    """A cursor behind the content key could become unreadable on a node with no key custody,
    while the subscription itself is perfectly authorized. Nothing here is secret."""
    artifact = Subscription(filter=event_bus.EventFilter(container_id="ws-1")).to_artifact()
    assert artifact.content == ""
    assert "ws-1" in artifact.context
    assert artifact.content_type == SUBSCRIPTION_CONTENT_TYPE


def test_an_ordinary_artifact_is_not_a_subscription():
    from mantle.entities.artifact import Artifact

    assert is_subscription(Artifact(content_type="text/plain")) is False
    assert Subscription.from_artifact(Artifact(content_type="text/plain")) is None
    assert is_subscription(None) is False


def test_an_unparseable_subscription_reads_as_none_not_as_the_empty_filter():
    """Fail-closed: the empty filter selects everything, so degrading to it on a parse failure
    would turn a corrupt narrow subscription into a firehose."""
    from mantle.entities.artifact import Artifact

    broken = Artifact(content_type=SUBSCRIPTION_CONTENT_TYPE, context="{not json")
    assert Subscription.from_artifact(broken) is None


def test_a_subscription_persists_and_reloads_through_the_ordinary_artifact_path(store):
    """No private table and no bespoke endpoint — `create_artifact` and `get_artifact`, which is
    what makes it light-cone authorized like anything else."""
    sub = Subscription(name="w", filter=event_bus.EventFilter(event_names=["artifact.created"]),
                       owner_id="u-1")
    saved = save_subscription(store, sub)
    assert saved.id

    loaded = load_subscription(store, saved.id)
    assert loaded is not None
    assert loaded.filter.event_names == ["artifact.created"]
    assert loaded.cursor.marks == ()


def test_acking_advances_the_stored_cursor(store, log):
    """The ack of the at-least-once contract. Read, act, then move the cursor — and the moved
    cursor has to outlive the socket, which is why it is on the artifact."""
    sub = save_subscription(store, Subscription(name="w", owner_id="u-1"))
    for i in range(3):
        log.append(_event("artifact.created", payload={"i": i}))

    batch = load_subscription(store, sub.id).read(log)
    assert len(batch) == 3

    reloaded = load_subscription(store, sub.id)
    reloaded.ack(batch)
    advance_cursor(store, sub.id, reloaded.cursor)

    resumed = load_subscription(store, sub.id)
    assert resumed.cursor.to_dict() == {"node-a": 3}
    assert resumed.read(log) == [], "the acked events came back after the ack"


def test_a_stale_ack_cannot_walk_the_stored_cursor_backwards(store):
    """Two consumers racing on one subscription must not be able to unwind each other's progress
    into unbounded redelivery."""
    sub = save_subscription(store, Subscription(name="w", owner_id="u-1"))
    advance_cursor(store, sub.id, "node-a:9")
    advance_cursor(store, sub.id, "node-a:2")
    assert load_subscription(store, sub.id).cursor.to_dict() == {"node-a": 9}


def test_acking_a_subscription_that_is_gone_returns_none(store):
    assert advance_cursor(store, "no-such-subscription", "node-a:1") is None


def test_a_delivery_class_typo_reads_as_durable_not_as_live():
    """A subscription artifact exists in order to survive a disconnect, so the fail-safe direction
    for an unrecognized delivery value is the stronger guarantee."""
    assert Subscription(delivery="best_efort").delivery == event_bus.DELIVERY_DURABLE
    assert Subscription(delivery=event_bus.DELIVERY_LIVE).delivery == event_bus.DELIVERY_LIVE


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 4 · Retention
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_the_log_never_trims_itself(log):
    """Retention is an operator's fact about their disk and their slowest consumer. Trimming past a
    live cursor turns at-least-once into at-most-once for that subscriber, so it is a call someone
    makes, never a default that makes it for them."""
    for i in range(200):
        log.append(_event("artifact.created", payload={"i": i}))
    assert len(log.read_since("", limit=1000)) == 200


def test_pruning_keeps_the_most_recent_events(log):
    for i in range(20):
        log.append(_event("artifact.created", payload={"i": i}))
    removed = log.prune(keep_last=5)
    assert removed == 15
    kept = log.read_since("", limit=1000)
    assert [e.payload["i"] for e in kept] == [15, 16, 17, 18, 19]
