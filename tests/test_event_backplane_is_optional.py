"""`backplane=None` is a working configuration, and more than one worker without one is refused.

The air-gap invariant, applied to signaling. `TieredContentStore` treats `remote=None` as
first-class — "a node whose working set is local answers with the backing unreachable, by design" —
and the bus takes the same shape: the in-process fan-out is the product, a back-plane is an extra
for multi-worker and multi-node deployments, and the base install imports neither client library.

Two properties, and they pull in opposite directions, which is why both are pinned here:

  §1  standalone is complete — nothing about the bus requires a broker, and no broker library is
      imported unless one is configured;
  §2  the one configuration that is silently wrong (N workers, no back-plane) is refused at boot
      rather than degraded, because from inside it looks perfectly healthy.

§3 covers the protocol shape and the echo suppression that keeps fan-out from becoming a loop.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle.events import event_backplane as bp                                # noqa: E402
from mantle.events import event_bus                                           # noqa: E402


@pytest.fixture(autouse=True)
def clean_bus():
    event_bus._filtered_subscribers.clear()
    yield
    event_bus.set_backplane(None)
    event_bus._filtered_subscribers.clear()


class _Recorder(bp.EventBackplane):
    """A back-plane that keeps what it was given, so the wiring can be checked without a broker."""

    name = "recorder"

    def __init__(self):
        self.published = []
        self.deliver = None
        self.closed = False

    def start(self, deliver):
        self.deliver = deliver

    def publish(self, wire):
        self.published.append(self.stamp(wire))

    def close(self):
        self.closed = True


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 1 · Standalone is complete
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_no_backplane_is_the_default():
    assert event_bus.backplane() is None


def test_none_off_and_empty_all_mean_standalone():
    """Three spellings of "I am one process", all legal answers rather than missing config."""
    for value in (None, "", "none", "NONE", " off ", "disabled"):
        assert bp.make_backplane(value) is None, f"{value!r} was not read as standalone"


def test_publishing_works_with_no_backplane():
    """The core claim. A node with no broker still fans out to its own subscribers."""
    async def scenario():
        queue = await event_bus.subscribe_filtered(event_bus.EventFilter())
        await event_bus.publish_event(event_bus.Event(name="artifact.created", payload={}))
        return queue.get_nowait()

    assert asyncio.run(scenario()).name == "artifact.created"


def test_neither_client_library_is_imported_by_the_bus():
    """The base install needs nothing but `cryptography`, and importing the bus must not quietly
    make that false. Both adapters import lazily, inside `start`."""
    assert "redis" not in sys.modules, "importing the bus pulled in redis"
    assert "paho" not in sys.modules, "importing the bus pulled in paho-mqtt"


def test_an_unknown_kind_is_refused_rather_than_defaulted_to_standalone():
    """A typo must not present as a working standalone node when the operator asked for
    distribution — that is the same silent degradation the boot refusal exists to prevent."""
    with pytest.raises(ValueError) as err:
        bp.make_backplane("readis", "redis://localhost")
    assert bp.KIND_SETTING in str(err.value)


def test_a_broker_kind_with_no_address_is_refused():
    with pytest.raises(ValueError) as err:
        bp.make_backplane("redis", None)
    assert bp.URI_SETTING in str(err.value)


@pytest.mark.parametrize("kind, uri, module, extra", [
    ("redis", "redis://localhost:6379/0", "redis", "agience-mantle[redis]"),
    ("mqtt", "mqtt://localhost:1883", "paho", "agience-mantle[mqtt]"),
])
def test_a_configured_kind_with_no_client_library_names_the_extra(
    monkeypatch, kind, uri, module, extra,
):
    """`shard/sqlite_store.py`'s discipline: refuse, and say what to install.

    The absent library is SIMULATED, not waited for. This used to `pytest.skip` whenever the client
    library turned out to be importable — and `redis` IS installed in this environment, so the skip
    always fired and neither assertion below ever ran: the refusal path and the message naming the
    extra were pinned by nothing. A guard that stops running because a dependency got installed is
    the same silent pass as a permanent skip.

    `sys.modules[name] = None` is what makes `import name` raise `ImportError` inside `start()`,
    which is the condition under test, on every machine — installed or not. Both shipped adapters
    are covered, because both make the same promise and only one of them was ever named here.
    """
    monkeypatch.setitem(sys.modules, module, None)
    plane = bp.make_backplane(kind, uri)
    with pytest.raises(bp.BackplaneUnavailable) as err:
        plane.start(lambda wire: None)
    message = str(err.value)
    assert extra in message, message
    assert bp.KIND_SETTING in message, message


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 2 · The boot refusal
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_one_worker_with_no_backplane_starts():
    """The standalone configuration, and the one every test in this suite runs under."""
    bp.require_backplane_for_workers(1, None)
    bp.require_backplane_for_workers("1", None)


def test_many_workers_with_a_backplane_start():
    bp.require_backplane_for_workers(4, _Recorder())


def test_many_workers_with_no_backplane_refuse_and_name_the_settings():
    """Refusing over degrading. With N workers and no back-plane an emit in one worker never
    reaches a subscriber on another, every worker reports healthy, and there is no metric a node
    can emit about events it was never told about."""
    with pytest.raises(RuntimeError) as err:
        bp.require_backplane_for_workers(4, None)
    message = str(err.value)
    assert bp.WORKERS_SETTING in message, "the refusal does not name the setting that caused it"
    assert bp.KIND_SETTING in message, "the refusal does not name the setting that fixes it"
    assert bp.URI_SETTING in message


def test_an_unreadable_worker_count_reads_as_one():
    """An unset or malformed count must not manufacture a refusal on a single-worker node."""
    bp.require_backplane_for_workers(None, None)
    bp.require_backplane_for_workers("", None)
    bp.require_backplane_for_workers("many", None)


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 3 · The protocol, and the loop it must not create
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_both_shipped_adapters_satisfy_the_protocol():
    """Redis for the datacenter, MQTT for the edge — different deployment shapes, one seam."""
    for cls in (bp.RedisBackplane, bp.MqttBackplane):
        for method in ("start", "publish", "close"):
            assert callable(getattr(cls, method)), f"{cls.__name__} has no {method}"
        assert cls.name in ("redis", "mqtt")


def test_publishing_forwards_to_the_installed_backplane():
    recorder = _Recorder()
    event_bus.set_backplane(recorder)
    asyncio.run(event_bus.publish_event(
        event_bus.Event(name="artifact.created", payload={"x": 1}, container_id="ws-1")))
    assert len(recorder.published) == 1
    assert recorder.published[0]["name"] == "artifact.created"
    assert recorder.published[0]["container_id"] == "ws-1"


def test_an_events_own_echo_is_dropped():
    """Without this, one emit re-enters its own bus on every broker round trip and the fan-out
    becomes a fan-out loop."""
    recorder = _Recorder()
    event_bus.set_backplane(recorder)
    asyncio.run(event_bus.publish_event(event_bus.Event(name="artifact.created", payload={})))
    assert recorder.is_echo(recorder.published[0]) is True


def test_an_unreadable_message_reads_as_an_echo():
    """Dropping what cannot be parsed is the fail-closed direction; the alternative is feeding a
    malformed payload into fan-out."""
    recorder = _Recorder()
    assert recorder.is_echo("not a dict") is True
    assert recorder.is_echo({"name": "artifact.created", "_node": "some-other-process"}) is False


def test_an_event_survives_the_wire_round_trip():
    original = event_bus.Event(name="artifact.updated", payload={"a": [1, 2]},
                               container_id="ws-1", artifact_id="a-1", actor_id="u-1",
                               containers=("ws-1", "ws-root"), origin="node-b", seq=7)
    back = event_bus.Event.from_wire(original.to_wire())
    for field in ("name", "payload", "container_id", "artifact_id", "actor_id",
                  "containers", "origin", "seq", "event_id"):
        assert getattr(back, field) == getattr(original, field), f"{field} did not survive"


def test_a_peer_build_sending_unknown_fields_cannot_break_fanout():
    """A node on a different build must not be able to stop this one's feed."""
    back = event_bus.Event.from_wire({"name": "artifact.created", "unknown": "field"})
    assert back.name == "artifact.created"
    assert back.payload == {}


def test_an_inbound_event_is_fanned_out_but_not_re_published(monkeypatch):
    """Cross-worker delivery, and the two things it must not do: echo back to the broker, and
    append to this node's log under this node's proper time."""
    async def scenario():
        recorder = _Recorder()
        event_bus.set_event_loop(asyncio.get_running_loop())
        event_bus.set_backplane(recorder)
        queue = await event_bus.subscribe_filtered(event_bus.EventFilter())

        wire = event_bus.Event(name="artifact.created", payload={},
                               origin="node-b", seq=3).to_wire()
        wire["_node"] = "another-process"
        event_bus.deliver_from_backplane(wire)

        received = await asyncio.wait_for(queue.get(), 2.0)
        return received, recorder.published

    received, published = asyncio.run(scenario())
    assert received.origin == "node-b" and received.seq == 3, \
        "the peer's proper time was rewritten on arrival"
    assert published == [], "an inbound event was echoed straight back onto the back-plane"


def test_installing_a_new_backplane_closes_the_old_one():
    first, second = _Recorder(), _Recorder()
    event_bus.set_backplane(first)
    event_bus.set_backplane(second)
    assert first.closed is True
    assert event_bus.backplane() is second
