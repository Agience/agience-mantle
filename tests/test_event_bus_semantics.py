"""The bus's stated semantics, held to the code: propagation, backpressure, and the public seam.

  §1  visibility attenuates, delivery fans out — the two must not be the same operation, and the
      module must not contain a second implementation of the meet
  §2  child → container propagation, and the seam that lets it become a context-edge walk later
  §3  backpressure — bounded, policied, counted, and never blocking the publisher
  §4  direct emit is the supported public seam for events the write path does not produce

§1 is the security-relevant one. Event *visibility* narrows along a path and composes with the one
attenuation operator; event *delivery* amplifies, one write to many subscribers. A fan-out step
that reached for the mask would be a way to widen authority, so what is asserted here is not only
that the operator is reused but that the delivery path cannot see it.
"""
from __future__ import annotations

import ast
import asyncio
import pathlib
import sys

import pytest

BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

SRC = BACKEND / "src" / "mantle"

from mantle import attenuation                                          # noqa: E402
from mantle.events import event_bus                                     # noqa: E402

#: The bus module's own file, resolved from the import rather than spelled as a path. These checks
#: read the source with `ast`, so a hardcoded path turns a module MOVE into three failing guards
#: that look like the property broke rather than like the file moved.
_BUS_SRC = pathlib.Path(event_bus.__file__)


@pytest.fixture(autouse=True)
def clean_bus():
    event_bus._filtered_subscribers.clear()
    yield
    event_bus._filtered_subscribers.clear()
    event_bus.set_container_resolver(None)
    event_bus.set_event_log(None)


def _event(name="artifact.created", **kw):
    return event_bus.Event(name=name, payload=kw.pop("payload", {}), **kw)


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 1 · Visibility attenuates; delivery fans out
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_visibility_composes_with_the_one_attenuation_operator():
    """`visibility_mask` is a forwarder onto `attenuation.compose`, not a second implementation.

    Checked by result rather than by inspection: for every path this asserts the composed mask is
    identical to what the operator produces, which is what "reuses it" has to mean.
    """
    paths = [
        [],
        [["read"]],
        [["read", "update"], ["read"]],
        [["read"], ["update"]],
        [None, ["read"]],
        [["read"], []],
    ]
    for path in paths:
        assert event_bus.visibility_mask(path) == attenuation.compose(
            attenuation.Mask.from_propagate(m) for m in path)


def test_visibility_can_only_narrow_along_a_path():
    """The security content of the light cone, restated for events: composing more edges never
    yields more authority. A propagation rule that widened would let an event escape the
    subtree its subject is confined to."""
    prefix = event_bus.visibility_mask([["read", "update"]])
    longer = event_bus.visibility_mask([["read", "update"], ["read"]])
    assert longer <= prefix
    assert not (prefix <= longer)


def test_the_empty_path_is_the_identity():
    """A zero-hop walk narrows nothing — the property that keeps `compose(p + q)` decomposable."""
    assert event_bus.visibility_mask([]) is attenuation.TOP


def test_a_deny_anywhere_on_the_path_absorbs():
    """Deny is the zero of the meet. The event path inherits that rather than re-deciding it."""
    composed = attenuation.compose([attenuation.TOP, attenuation.DENY, attenuation.TOP])
    assert composed is attenuation.DENY
    assert event_bus.may_see(composed) is False


def test_visibility_asks_only_about_read():
    """An event is visible exactly where its subject is readable. Inventing a verb here would make
    the change feed a side channel answering questions the read path would refuse."""
    write_only = attenuation.Mask.of(["update", "delete"])
    assert event_bus.may_see(write_only) is False
    assert event_bus.may_see(attenuation.Mask.of(["read"])) is True


def test_the_bus_contains_no_second_implementation_of_the_meet():
    """The single-source guard, scoped to this module.

    AST rather than grep, because the docstrings here discuss intersection and masks at length and
    a textual search would match the prose. What it looks for is bitwise `&` or set intersection
    over anything mask-shaped inside `event_bus` — the shape a hand-rolled second operator takes.
    A copy that agreed with `attenuation` on the day it was written would still be free to drift.
    """
    tree = ast.parse(_BUS_SRC.read_text(encoding="utf-8"))
    mask_words = {"mask", "masks", "grant", "grants", "permission", "permissions",
                  "propagate", "actions", "allow", "deny"}

    def mentions_mask(node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id.lower() in mask_words:
                return True
            if isinstance(sub, ast.Attribute) and sub.attr.lower() in mask_words:
                return True
        return False

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd) and mentions_mask(node):
            offenders.append(node.lineno)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "intersection" and mentions_mask(node):
            offenders.append(node.lineno)
    assert not offenders, (
        f"event_bus.py intersects permission bits itself at line(s) {offenders}. Call "
        f"`attenuation.meet` / `Mask.__and__` instead — two implementations of one algebra is "
        f"how the light-cone deny bug happened.")


def test_the_fanout_path_never_touches_a_mask():
    """Delivery amplifies. It must therefore be structurally unable to compute authority.

    Asserted about `_fanout`'s own body: no mask type, no visibility call, no attenuation import
    reachable from it. A fan-out that "merged" the masks of the subscribers it was serving would
    hand each of them the union of everyone's authority — the exact trap the plan flags.
    """
    tree = ast.parse(_BUS_SRC.read_text(encoding="utf-8"))
    fanout = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "_fanout")
    forbidden = {"Mask", "meet", "compose", "visibility_mask", "may_see", "TOP", "DENY"}
    used = {sub.id for sub in ast.walk(fanout) if isinstance(sub, ast.Name)} | \
           {sub.attr for sub in ast.walk(fanout) if isinstance(sub, ast.Attribute)}
    assert not (used & forbidden), (
        f"_fanout references {sorted(used & forbidden)}. Fan-out is the amplifying operation and "
        f"must not compute visibility; each recipient's ACL is applied on its own side.")


def test_a_filter_is_a_selection_and_never_an_authorization():
    """A wide-open filter is harmless — it selects everything and authorizes nothing. The check
    that decides what a caller may have lives in the router, against that caller's grants."""
    wide = event_bus.EventFilter()
    assert wide.matches(_event(container_id="anything", artifact_id="anything")) is True

    import mantle.routers.events_router as router
    from types import SimpleNamespace

    no_grants = SimpleNamespace(grants=[], principal_id=None, user_id="u-1")
    assert router._event_visible_to(no_grants, _event(container_id="ws-1")) is False


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 2 · Child → container propagation
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_by_default_an_event_addresses_its_immediate_container():
    """The one containment fact every artifact doc already carries, so the default needs no graph
    read at all."""
    assert event_bus.containers_of(_event(container_id="ws-1")) == ("ws-1",)
    assert event_bus.containers_of(_event()) == ()


def test_watching_a_container_delivers_writes_to_what_it_contains():
    """The rule that makes a container subscription useful for a tree view. A subscriber watching
    `ws-1` learns about `a-1` inside it without enumerating its children."""
    async def scenario():
        queue = await event_bus.subscribe_filtered(event_bus.EventFilter(container_id="ws-1"))
        await event_bus.publish_event(_event(container_id="ws-1", artifact_id="a-1"))
        return queue.get_nowait()

    assert asyncio.run(scenario()).artifact_id == "a-1"


def test_a_resolver_can_widen_addressing_up_a_containment_chain():
    """The seam for the context lattice. When context edges become the propagation structure, a
    resolver that walks them plugs in here and nothing else changes — no call site, no filter, no
    subscriber."""
    event_bus.set_container_resolver(lambda e: ["ws-root"] if e.container_id == "ws-1" else [])

    async def scenario():
        watching_root = await event_bus.subscribe_filtered(
            event_bus.EventFilter(container_id="ws-root"))
        await event_bus.publish_event(_event(container_id="ws-1", artifact_id="a-1"))
        return watching_root.get_nowait()

    received = asyncio.run(scenario())
    assert received.artifact_id == "a-1"
    assert received.containers == ("ws-1", "ws-root"), "nearest container first, deduplicated"


def test_propagation_upward_is_addressing_and_not_authorization():
    """Reaching a container's subscribers grants them nothing: each delivery is still checked
    against that subscriber's own grants, so widening the address set cannot widen authority."""
    from types import SimpleNamespace

    import mantle.routers.events_router as router

    event_bus.set_container_resolver(lambda e: ["ws-root"])
    event = _event(container_id="ws-1", artifact_id="a-1")
    event.containers = event_bus.containers_of(event)

    stranger = SimpleNamespace(grants=[], principal_id=None, user_id="u-2")
    assert "ws-root" in event.containers
    assert router._event_visible_to(stranger, event) is False, \
        "propagating to a container made the event visible to someone with no grant on it"


def test_a_failing_resolver_degrades_to_the_immediate_container():
    """Propagation is addressing: fewer containers is a smaller event, never a wider one. A walk
    that raises must not fail the publish or widen the address set."""
    def explode(_event):
        raise RuntimeError("graph unavailable")

    event_bus.set_container_resolver(explode)
    assert event_bus.containers_of(_event(container_id="ws-1")) == ("ws-1",)


def test_a_resolver_cannot_duplicate_the_immediate_container():
    event_bus.set_container_resolver(lambda e: ["ws-1", "ws-root", "ws-root"])
    assert event_bus.containers_of(_event(container_id="ws-1")) == ("ws-1", "ws-root")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 3 · Backpressure
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_a_live_queue_is_bounded():
    """Unbounded does not remove backpressure, it relocates it into memory where it is
    unobservable until the process dies."""
    async def scenario():
        return await event_bus.subscribe_live(event_bus.EventFilter(), maxsize=4)

    assert asyncio.run(scenario()).queue.maxsize == 4


def test_drop_oldest_keeps_the_newest_and_counts_what_it_lost():
    """The default. On a change feed the newest event is the one closest to current state, so an
    overwhelmed subscriber should fall behind at the tail."""
    async def scenario():
        sub = await event_bus.subscribe_live(event_bus.EventFilter(), maxsize=2)
        for i in range(5):
            await event_bus.publish_event(_event(payload={"i": i}))
        return sub

    sub = asyncio.run(scenario())
    kept = [sub.queue.get_nowait().payload["i"] for _ in range(sub.queue.qsize())]
    assert kept == [3, 4], f"drop_oldest did not keep the newest events: {kept}"
    assert sub.dropped == 3, "the loss was not counted, so it is indistinguishable from a quiet feed"


def test_drop_newest_keeps_the_backlog():
    """For a consumer whose queue is a work list rather than a state feed."""
    async def scenario():
        sub = await event_bus.subscribe_live(event_bus.EventFilter(), maxsize=2,
                                             overflow=event_bus.Overflow.DROP_NEWEST)
        for i in range(5):
            await event_bus.publish_event(_event(payload={"i": i}))
        return sub

    sub = asyncio.run(scenario())
    kept = [sub.queue.get_nowait().payload["i"] for _ in range(sub.queue.qsize())]
    assert kept == [0, 1]
    assert sub.dropped == 3


def test_disconnect_marks_the_subscription_rather_than_losing_events_quietly():
    """The honest answer for a consumer that must not silently miss events but holds no cursor:
    stop, and let the transport close it, so the gap is a disconnect and not a hole."""
    async def scenario():
        sub = await event_bus.subscribe_live(event_bus.EventFilter(), maxsize=2,
                                             overflow=event_bus.Overflow.DISCONNECT)
        for i in range(5):
            await event_bus.publish_event(_event(payload={"i": i}))
        return sub

    sub = asyncio.run(scenario())
    assert sub.overflowed is True
    assert sub.dropped >= 1


def test_an_unknown_overflow_policy_is_refused():
    """A policy nobody implements must not silently become the default one."""
    with pytest.raises(ValueError):
        asyncio.run(event_bus.subscribe_live(event_bus.EventFilter(), overflow="whatever"))


def test_a_full_subscriber_does_not_block_the_publisher():
    """One stalled WebSocket must not be able to stall the write path that fed it.

    Asserted structurally as well as behaviourally: `_fanout` must contain no `await` on a
    subscriber, because a bounded queue plus `await put` is head-of-line blocking wearing a
    backpressure policy's clothes.
    """
    tree = ast.parse(_BUS_SRC.read_text(encoding="utf-8"))
    fanout = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "_fanout")
    assert not [n for n in ast.walk(fanout) if isinstance(n, ast.Await)], \
        "_fanout awaits inside the fan-out loop, so a slow subscriber can stall every other one"

    async def scenario():
        slow = await event_bus.subscribe_live(event_bus.EventFilter(), maxsize=1)
        fast = await event_bus.subscribe_live(event_bus.EventFilter(), maxsize=100)
        for i in range(50):
            await asyncio.wait_for(event_bus.publish_event(_event(payload={"i": i})), 1.0)
        return slow, fast

    slow, fast = asyncio.run(scenario())
    assert fast.queue.qsize() == 50, "a full subscriber cost a healthy one its events"
    assert slow.dropped == 49


def test_the_two_delivery_classes_are_named_in_the_api():
    """A subscriber has to be able to tell which guarantee it holds without reading the code path
    that gave it one."""
    assert event_bus.DELIVERY_LIVE == "best_effort"
    assert event_bus.DELIVERY_DURABLE == "at_least_once"
    assert event_bus.DELIVERY_LIVE != event_bus.DELIVERY_DURABLE


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 4 · Direct emit is the public seam
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_publish_event_is_the_seam_for_a_non_crud_event():
    """Events the write chokepoint does not produce come in here, and inherit everything: the
    filters, the log, the ACL and the back-plane."""
    async def scenario():
        queue = await event_bus.subscribe_filtered(
            event_bus.EventFilter(event_names=["mesh.*"]))
        await event_bus.publish_event(_event("mesh.peer.reconciled", payload={"peer": "node-b"}))
        return queue.get_nowait()

    received = asyncio.run(scenario())
    assert received.name == "mesh.peer.reconciled"
    assert received.payload["peer"] == "node-b"


def test_emit_artifact_event_sync_is_the_seam_for_synchronous_callers():
    """The thread-safe half. The write path is synchronous and must never block on the bus, so it
    schedules rather than awaits — which is why a sync caller before the loop exists is a no-op
    rather than an error."""
    async def scenario():
        event_bus.set_event_loop(asyncio.get_running_loop())
        queue = await event_bus.subscribe_filtered(event_bus.EventFilter())
        event_bus.emit_artifact_event_sync(
            "ws-1", "artifact.invoke.completed", {"artifact_id": "a-1"}, actor_id="u-1")
        return await asyncio.wait_for(queue.get(), 2.0)

    received = asyncio.run(scenario())
    assert received.name == "artifact.invoke.completed"
    assert received.container_id == "ws-1"
    assert received.artifact_id == "a-1"
    assert received.actor_id == "u-1"


def test_a_sync_emit_before_the_loop_exists_is_a_no_op_not_a_crash():
    """Early bootstrap writes must not fail because the bus is not up yet — the write is the fact,
    the event is the announcement."""
    event_bus._loop = None
    event_bus.emit_artifact_event_sync("ws-1", "artifact.created", {"artifact_id": "a-1"})


def test_the_seam_extracts_the_content_type_a_filter_needs():
    """A `content_type` filter is only useful if the field is populated, and service payloads carry
    it in two shapes."""
    from_context = event_bus._extract_artifact_fields(
        {"artifact": {"id": "a-1", "context": '{"content_type": "text/plain"}'}})
    assert from_context == ("a-1", "text/plain")

    from_column = event_bus._extract_artifact_fields(
        {"artifact": {"id": "a-2", "content_type": "text/markdown"}})
    assert from_column == ("a-2", "text/markdown")

    bare = event_bus._extract_artifact_fields({"artifact_id": "a-3"})
    assert bare == ("a-3", None)


def test_both_seam_functions_are_exported():
    """Supported API, so it is named in `__all__` rather than reachable by accident."""
    assert "publish_event" in event_bus.__all__
    assert "emit_artifact_event_sync" in event_bus.__all__
