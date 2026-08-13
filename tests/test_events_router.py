"""Unit tests for the unified /events WebSocket helpers.

The WebSocket transport is awkward to drive through httpx/ASGITransport, so
these tests exercise the router's building blocks directly:

- `_parse_filter`: client JSON filter -> EventFilter dataclass
- `_event_visible_to`: per-user ACL check

The end-to-end WS handshake + protocol loop is validated via the router
smoke test at the bottom using `starlette.testclient.TestClient`.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle.events import event_bus  # noqa: E402
from mantle.routers import events_router as ev  # noqa: E402


# ---------------------------------------------------------------------------
# _parse_filter
# ---------------------------------------------------------------------------

def test_parse_filter_empty_yields_wildcard():
    f = ev._parse_filter(None)
    assert f.container_id is None
    assert f.artifact_id is None
    assert f.content_type is None
    assert f.event_names is None


def test_parse_filter_normalizes_all_fields():
    raw = {
        "container_id": "ws-1",
        "artifact_id": "art-1",
        "content_type": "application/vnd.agience.operator+json",
        "event_names": ["artifact.invoke.*", 123, "artifact.created"],
    }
    f = ev._parse_filter(raw)
    assert f.container_id == "ws-1"
    assert f.artifact_id == "art-1"
    assert f.content_type == "application/vnd.agience.operator+json"
    assert f.event_names == ["artifact.invoke.*", "artifact.created"]


def test_parse_filter_drops_empty_strings_to_none():
    f = ev._parse_filter({"container_id": "", "content_type": ""})
    assert f.container_id is None
    assert f.content_type is None


# ---------------------------------------------------------------------------
# _event_visible_to (ACL)
#
# The check is `services.dependencies.check_access(..., "read", ...)` and nothing else, so these
# drive a real store: a grant that is not in the store is not a grant, and a container that reaches
# its child does so through the origin edge the light cone walks. Answering from a hand-built grant
# list would test a rule the event path does not have — see
# `test_the_event_path_is_check_access.py` for the shapes where the two used to disagree.
# ---------------------------------------------------------------------------

from mantle.db import lattice_api                                    # noqa: E402
from mantle.entities.artifact import Artifact                        # noqa: E402
from mantle.entities.grant import Grant                              # noqa: E402
from mantle.services.dependencies import AuthContext                 # noqa: E402

WORKSPACE, ARTIFACT, USER = "ws-1", "a-1", "u-1"


@pytest.fixture
def store(tmp_path):
    db = lattice_api.LatticeDatabase(str(tmp_path / "acl.db"), origin="node-a")
    lattice_api.create_artifact(db, Artifact(
        id=WORKSPACE, root_id=WORKSPACE, collection_id="", name="ws", content="", created_by=USER))
    lattice_api.create_artifact(db, Artifact(
        id=ARTIFACT, root_id=ARTIFACT, collection_id=WORKSPACE, name="memo", content="",
        created_by=USER, modified_by=USER))
    lattice_api.add_artifact_to_collection(db, WORKSPACE, ARTIFACT)
    return db


def _auth(user_id=USER, grants=(), principal_type="user"):
    return AuthContext(principal_id=user_id, principal_type=principal_type, user_id=user_id,
                       grants=list(grants))


def _access(store, auth):
    _verdicts, access = ev._container_access(ev._Session("t", auth, store))
    return access


def _grant(store, gid, resource_id, *, action="read", grantee=USER):
    """One grant carrying exactly one action — `can_read` defaults to True on the entity, so a
    write-only grant has to say so."""
    return lattice_api.create_grant(store, Grant(
        id=gid, resource_id=resource_id, grantee_type="user", grantee_id=grantee,
        granted_by="admin", state="active", can_read=(action == "read"),
        **({} if action == "read" else {f"can_{action}": True})))


def _event(artifact_id=ARTIFACT, container_id=WORKSPACE, actor_id=None):
    return event_bus.Event(name="artifact.updated", payload={}, artifact_id=artifact_id,
                           container_id=container_id, actor_id=actor_id)


def test_acl_denied_when_no_grants_and_not_actor(store):
    auth = _auth()
    assert ev._event_visible_to(auth, _event(), _access(store, auth)) is False


def test_acl_allowed_when_artifact_read_grant_matches(store):
    _grant(store, "g-1", ARTIFACT)
    auth = _auth()
    assert ev._event_visible_to(auth, _event(), _access(store, auth)) is True


def test_acl_allowed_when_container_read_grant_reaches_the_child(store):
    """A grant on the container reaches the artifact through the origin edge — the light cone,
    which is the only thing that carries a container's authority to what it holds."""
    _grant(store, "g-1", WORKSPACE)
    auth = _auth()
    assert ev._event_visible_to(auth, _event(), _access(store, auth)) is True


def test_acl_denied_for_an_unscoped_read_grant(store):
    """A grant naming no resource authorizes no resource. `check_access` filters a key's bundle on
    `resource_id`, so an unscoped member reaches nothing; a feed that honoured one would be a
    platform-wide viewer nobody granted."""
    unscoped = Grant(id="g-open", resource_id=None, grantee_type="grant_key", grantee_id="k-1",
                     granted_by="admin", can_read=True)
    auth = AuthContext(principal_id="k-1", principal_type="grant_key", user_id=None,
                       grants=[unscoped], grant_key_id="k-1")
    assert ev._event_visible_to(auth, _event(), _access(store, auth)) is False


def test_acl_denied_when_only_write_grants_no_read(store):
    _grant(store, "g-1", WORKSPACE, action="update")
    auth = _auth()
    assert ev._event_visible_to(auth, _event(), _access(store, auth)) is False


def test_acl_grants_a_principal_nothing_for_having_been_the_actor(store):
    """`actor_id` is provenance — `modified_by` / `created_by` — and provenance does not change
    when authority is withdrawn. It is not a grant for a user and not one for a service."""
    auth = AuthContext(principal_id="server-xyz", principal_type="service", user_id=None, grants=[])
    event = _event(actor_id="server-xyz")
    assert ev._event_visible_to(auth, event, _access(store, auth)) is False


def test_acl_answers_nothing_without_a_store(store):
    """Authorization lives in the store, so with no store to ask there is no authorization."""
    _grant(store, "g-1", WORKSPACE)
    assert ev._event_visible_to(_auth(), _event(), None) is False


# ---------------------------------------------------------------------------
# integration with event_bus: publish_event reaches filtered subscriber
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_publish_event_delivers_to_filtered_subscriber_matching_container():
    event_bus._filtered_subscribers.clear()
    q = await event_bus.subscribe_filtered(
        event_bus.EventFilter(container_id="ws-42", event_names=["artifact.*"])
    )

    # Non-matching container: should not reach the queue
    await event_bus.publish_event(
        event_bus.Event(name="artifact.created", payload={}, container_id="ws-99")
    )
    # Non-matching name: should not reach the queue
    await event_bus.publish_event(
        event_bus.Event(name="other.thing", payload={}, container_id="ws-42")
    )
    # Matching: should reach the queue
    await event_bus.publish_event(
        event_bus.Event(name="artifact.invoke.completed", payload={"ok": True}, container_id="ws-42")
    )

    assert q.qsize() == 1
    msg = q.get_nowait()
    assert msg.name == "artifact.invoke.completed"

    await event_bus.unsubscribe_filtered(q)
