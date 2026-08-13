"""The change feed says *what* changed. It never carries the artifact's body.

"Storage gets ciphertext" is the claim `db.doc_boundary` makes and enforces: an artifact's inline
`content` is envelope-encrypted on the way into the store and opened again only at the single read
chokepoint, under key custody. An event is neither of those things. It fans out to every subscriber
whose filter selects it, and — wherever a durable log is installed — lands in an `event_log` row
that no ACL covers, inside the same file whose `artifacts.content` column is encrypted. A body on
that path is a body outside both controls.

The leak this pins was structural rather than incidental. `to_lattice_doc` encrypts a *fresh* dict
built from the entity, so the entity handed to `emit_artifact_change` a line later still held its
plaintext; the write path therefore encrypted one copy and announced the other. The stripping
helper existed and was wired only to the two paths that emit a *stored* doc — the two that would
otherwise have leaked ciphertext, which is an availability bug — while the two that leaked
plaintext went without.

So the assertions here are about every emit, not about one function: the entity writes, the
container writes, the batch commit, the archive, the durable log row, and the back-plane decoder.
Each is checked twice — the field is absent, and the plaintext does not appear anywhere in the
serialized payload — because a body can leave under a name nobody thought to strip.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

import pytest

BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle.events import event_bus                                  # noqa: E402
from mantle.db import lattice_api                             # noqa: E402
from mantle.entities.artifact import Artifact                 # noqa: E402
from mantle.entities.collection import Collection             # noqa: E402
from mantle.entities.grant import Grant                       # noqa: E402

#: The string that must never reach a subscriber. Distinctive so a substring search over the whole
#: serialized payload is meaningful — an absent key proves less than an absent value.
SECRET = "LAYOFF LIST: alice, bob"


@pytest.fixture
def store(tmp_path):
    return lattice_api.LatticeDatabase(str(tmp_path / "feed-body.db"), origin="node-a")


@pytest.fixture
def content_key(monkeypatch):
    """A deterministic content master key, so the envelope boundary runs for real.

    Without it the encrypt path has no key oracle and refuses the write — which would prove the
    feed carries no content only because there was no content to carry.
    """
    from mantle.services import content_crypto
    monkeypatch.setattr(
        content_crypto, "_default_master_key",
        lambda principal_id, collection_id=None, *, may_create=False, creator_id=None: b"\x01" * 32,
    )


@pytest.fixture
async def bus():
    """A clean bus bound to the running loop, restored afterwards.

    Async so it binds the loop the test actually runs on: `publish_event_sync` schedules onto
    whatever loop was registered, and a sync fixture would register a different one.
    """
    event_bus._filtered_subscribers.clear()
    event_bus.set_event_loop(asyncio.get_running_loop())
    yield event_bus
    event_bus._filtered_subscribers.clear()
    event_bus.set_event_log(None)
    event_bus.set_container_resolver(None)


async def _drain(queue, *, expect: int, timeout: float = 2.0) -> list:
    out = []
    for _ in range(expect):
        out.append(await asyncio.wait_for(queue.get(), timeout))
    return out


def _artifact_of(event) -> dict:
    return event.payload.get("artifact") or {}


def _assert_no_body(event, *, where: str) -> None:
    """The two halves of the claim: the fields are gone, and the value is nowhere."""
    artifact = _artifact_of(event)
    assert "content" not in artifact, f"{where}: the feed carried the artifact's `content`"
    assert "content_encrypted" not in artifact, \
        f"{where}: the feed carried `content_encrypted`, a flag describing a body that is not there"
    serialized = json.dumps(event.payload, default=str)
    assert SECRET not in serialized, \
        f"{where}: the plaintext body reached a subscriber ({serialized[:200]}...)"


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 1 · The entity write paths — where the plaintext was
# ═════════════════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_create_announces_the_artifact_and_not_its_body(store, bus, content_key):
    """`create_artifact` encrypts a copy and used to announce the original.

    The entity's own `content` is never touched by `to_lattice_doc` — it builds a fresh dict — so
    the encryption that protects the row does nothing for the object handed to the emit on the
    next line.
    """
    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    artifact = Artifact(id="a-1", collection_id="", name="q3", content=SECRET, created_by="u-1")
    lattice_api.create_artifact(store, artifact)

    (event,) = await _drain(queue, expect=1)
    assert event.name == "artifact.created"
    assert store.artifacts.get_artifact("a-1")["content_encrypted"] is True, \
        "the envelope did not run, so this case never had a body to leak"
    _assert_no_body(event, where="artifact.created")


@pytest.mark.asyncio
async def test_an_update_announces_the_artifact_and_not_its_body(store, bus, content_key):
    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    artifact = Artifact(id="a-2", collection_id="", name="q3", content=SECRET, created_by="u-1")
    lattice_api.create_artifact(store, artifact)
    await _drain(queue, expect=1)

    artifact.name = "q3-final"
    artifact.modified_by = "u-2"
    lattice_api.update_artifact(store, artifact)

    (event,) = await _drain(queue, expect=1)
    assert event.name == "artifact.updated"
    _assert_no_body(event, where="artifact.updated")


@pytest.mark.asyncio
async def test_a_container_write_announces_the_container_and_not_its_body(store, bus, content_key):
    """A container IS an artifact, so it goes through the same emit and needs the same answer."""
    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    container = Collection(id="ws-1", collection_id="", name="before", content=SECRET,
                           created_by="u-1")
    lattice_api.create_collection(store, container)
    (created,) = await _drain(queue, expect=1)
    _assert_no_body(created, where="create_collection")

    container.name = "after"
    lattice_api.update_collection(store, container)
    (updated,) = await _drain(queue, expect=1)
    _assert_no_body(updated, where="update_collection")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 2 · The stored-doc write paths — where the ciphertext was
# ═════════════════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_batch_commit_announces_neither_plaintext_nor_ciphertext(store, bus, content_key):
    """The commit writes the stored doc back, so its emit starts from ciphertext rather than
    plaintext. Neither belongs on the feed: a base64 blob no subscriber holds a key for is worse
    than an absent field, because it reads as content instead of as something to go and fetch."""
    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    lattice_api.create_artifact(store, Artifact(
        id="d-1", root_id="d-1", collection_id="col-1", name="draft", content=SECRET,
        created_by="u-1", state="draft"))
    await _drain(queue, expect=1)

    stored = store.artifacts.get_artifact("d-1")
    assert stored["content_encrypted"] is True

    assert lattice_api.batch_commit_drafts(
        store, "col-1", ["d-1"], "u-2", "2026-07-22T12:00:00+00:00") == 1

    (event,) = await _drain(queue, expect=1)
    _assert_no_body(event, where="batch_commit_drafts")
    assert stored["content"] not in json.dumps(event.payload), \
        "the commit event carried the stored ciphertext"


@pytest.mark.asyncio
async def test_an_archive_announces_neither_plaintext_nor_ciphertext(store, bus, content_key):
    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    lattice_api.create_artifact(store, Artifact(
        id="a-9", collection_id="col-1", name="doomed", content=SECRET,
        created_by="u-1", state="committed"))
    await _drain(queue, expect=1)
    lattice_api.create_grant(store, Grant(
        id="g-1", resource_id="col-1", grantee_type="user", grantee_id="u-9",
        granted_by="admin", can_read=True, can_delete=True))

    stored = store.artifacts.get_artifact("a-9")
    assert lattice_api.archive_artifact(store, "u-9", "a-9") is True

    (event,) = await _drain(queue, expect=1)
    assert event.name == "artifact.deleted"
    _assert_no_body(event, where="archive_artifact")
    assert stored["content"] not in json.dumps(event.payload)


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 3 · The durable log — the copy that outlives the socket
# ═════════════════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_the_durable_log_row_holds_no_body(store, bus, content_key):
    """`event_log` lives in the same SQLite file as `artifacts`, and its `payload` column is plain
    JSON that no grant covers. A body written there is a body sitting beside the encrypted column
    it was supposed to be protected by, for as long as the operator's retention keeps it.
    """
    event_bus.set_event_log(event_bus.open_event_log(store))
    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))

    lattice_api.create_artifact(store, Artifact(
        id="a-log", collection_id="col-1", name="q3", content=SECRET, created_by="u-1"))
    (event,) = await _drain(queue, expect=1)
    assert event.seq is not None, "the event was never logged, so this proves nothing"

    rows = store.conn.read().execute("SELECT payload FROM event_log").fetchall()
    assert rows, "no log row was written"
    for (payload,) in rows:
        assert SECRET not in payload, f"the durable log stored the plaintext body: {payload[:200]}"
        assert json.loads(payload).get("artifact", {}).get("content") is None


@pytest.mark.asyncio
async def test_a_logged_event_still_identifies_what_changed(store, bus, content_key):
    """The control on all of the above. Stripping the body must leave a usable descriptor —
    a feed that carried nothing would pass every assertion in this file and serve nobody."""
    event_bus.set_event_log(event_bus.open_event_log(store))
    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))

    lattice_api.create_artifact(store, Artifact(
        id="a-desc", collection_id="col-1", name="q3", content=SECRET, created_by="u-1",
        content_type="text/markdown", state="draft"))
    (event,) = await _drain(queue, expect=1)

    artifact = _artifact_of(event)
    assert event.artifact_id == "a-desc"
    assert event.container_id == "col-1"
    assert event.actor_id == "u-1"
    assert artifact["id"] == "a-desc"
    assert artifact["collection_id"] == "col-1"
    assert artifact["name"] == "q3"
    assert artifact["state"] == "draft"
    assert artifact["content_type"] == "text/markdown", \
        "the content TYPE is a descriptor, not a body — a subscriber filters on it"


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 4 · The rule itself
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_redaction_removes_the_body_and_keeps_everything_else():
    doc = {"id": "a-1", "collection_id": "c-1", "name": "q3", "content": SECRET,
           "content_encrypted": True, "content_type": "text/markdown"}
    out = event_bus.redacted_artifact(doc)
    assert out == {"id": "a-1", "collection_id": "c-1", "name": "q3",
                   "content_type": "text/markdown"}
    assert doc["content"] == SECRET, "redaction edited the caller's doc in place"


def test_redaction_reaches_the_shapes_the_feed_actually_publishes():
    single = event_bus.redact_content({"artifact": {"id": "a-1", "content": SECRET}})
    assert single == {"artifact": {"id": "a-1"}}

    plural = event_bus.redact_content({"artifacts": [{"id": "a-1", "content": SECRET},
                                                     {"id": "a-2"}]})
    assert plural == {"artifacts": [{"id": "a-1"}, {"id": "a-2"}]}

    flat = event_bus.redact_content({"id": "a-1", "content": SECRET, "content_encrypted": True})
    assert flat == {"id": "a-1"}


def test_a_payload_with_no_body_is_passed_through_untouched():
    """The hot path: the ordinary event pays a membership test, not a copy."""
    payload = {"artifact": {"id": "a-1", "name": "q3"}, "artifact_id": "a-1"}
    assert event_bus.redact_content(payload) is payload


@pytest.mark.asyncio
async def test_publishing_redacts_before_anything_can_observe_the_event(bus):
    """Enforced at the seam rather than at each emit site. Emits arrive from the persistence
    boundary, from the services and from the shard tools; a rule enforced at those sites is a rule
    the next site can forget."""
    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    await event_bus.publish_event(event_bus.Event(
        name="artifact.updated", payload={"artifact": {"id": "a-1", "content": SECRET}},
        container_id="col-1"))

    (event,) = await _drain(queue, expect=1)
    _assert_no_body(event, where="publish_event")


def test_a_peer_cannot_inject_a_body_through_the_back_plane():
    """The decoder is forgiving about unknown keys precisely because a peer may run a different
    build — so a peer predating this rule must not be able to hand this node's subscribers a body.
    """
    event = event_bus.Event.from_wire({
        "name": "artifact.updated", "container_id": "col-1",
        "payload": {"artifact": {"id": "a-1", "content": SECRET, "content_encrypted": True}},
    })
    assert event.payload == {"artifact": {"id": "a-1"}}
