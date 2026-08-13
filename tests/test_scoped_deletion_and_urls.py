"""Event scoping, scoped deletion, and signed-URL integrity.

Three invariants:

1. Event visibility is decided by `check_access` and by nothing else, so it is scoped exactly as a
   grant is: a grant on one artifact reveals that artifact's events and no other tenant's stream,
   and a grant naming no resource reaches nothing at all — `check_access` matches grants on
   `resource_id`, so an unscoped grant has no resource to match.
2. `delete_workspace(..., cascade=True)` must not destroy artifacts globally via
   `delete_artifacts_by_root` when they are still linked into other containers the caller has no
   rights over; those get evicted from the deleted container instead. (The default, `cascade=False`,
   never destroys members at all — it only detaches them — so these tests pin the cascade path
   specifically, the one branch that can lose data.)
3. `generate_signed_url` must raise rather than return an unsigned URL on a signing error — the
   signature is the access control, and a caller cannot tell the two apart.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mantle.events import event_bus
from mantle.routers import events_router as ev


from mantle.db import lattice_api
from mantle.entities.artifact import Artifact
from mantle.entities.grant import Grant
from mantle.services.dependencies import AuthContext

USER = "u-1"


@pytest.fixture
def store(tmp_path):
    """Two unrelated artifacts in two unrelated containers — one tenant each."""
    db = lattice_api.LatticeDatabase(str(tmp_path / "scoped.db"), origin="node-a")
    for container, artifact in (("ws-1", "a-1"), ("ws-9", "a-9")):
        lattice_api.create_artifact(db, Artifact(
            id=container, root_id=container, collection_id="", name=container, content="",
            created_by=USER))
        lattice_api.create_artifact(db, Artifact(
            id=artifact, root_id=artifact, collection_id=container, name=artifact, content="",
            created_by=USER, modified_by=USER))
        lattice_api.add_artifact_to_collection(db, container, artifact)
    return db


def _auth(user_id=USER, grants=()):
    return AuthContext(principal_id=user_id, principal_type="user", user_id=user_id,
                       grants=list(grants))


def _access(store, auth):
    _verdicts, access = ev._container_access(ev._Session("t", auth, store))
    return access


def _grant(store, gid, resource_id, *, action="read", effect="allow"):
    """One grant carrying exactly one action — `can_read` defaults to True on the entity, so a
    write-only grant has to say so."""
    return lattice_api.create_grant(store, Grant(
        id=gid, resource_id=resource_id, grantee_type="user", grantee_id=USER,
        granted_by="admin", effect=effect, state="active", can_read=(action == "read"),
        **({} if action == "read" else {f"can_{action}": True})))


def _event(artifact_id, container_id):
    return event_bus.Event(name="artifact.updated", payload={}, artifact_id=artifact_id,
                           container_id=container_id)


# ---------------------------------------------------------------------------
# 1. Event visibility is actually scoped
# ---------------------------------------------------------------------------


def test_scoped_grant_does_not_see_unrelated_events(store):
    """A grant on one artifact must not reveal every event."""
    _grant(store, "g-1", "a-1")
    auth = _auth()

    assert ev._event_visible_to(auth, _event("a-9", "ws-9"), _access(store, auth)) is False, (
        "a read grant on a-1 exposed an event on a-9 — every user holding any "
        "grant saw every other tenant's event stream"
    )


def test_scoped_grant_still_sees_its_own_events_positive_control(store):
    """Positive control: legitimate visibility for a scoped grant on its own artifact."""
    _grant(store, "g-1", "a-1")
    auth = _auth()

    assert ev._event_visible_to(auth, _event("a-1", "ws-1"), _access(store, auth)) is True


def test_an_unscoped_grant_reaches_nothing(store):
    """A grant naming no resource matches no resource. `check_access` selects grants by
    `resource_id`, so there is no artifact an unscoped one speaks for — and the event path, which
    asks that same question, has no wider answer available to it."""
    unscoped = Grant(id="g-open", resource_id=None, grantee_type="grant_key", grantee_id="k-1",
                     granted_by="admin", can_read=True)
    auth = AuthContext(principal_id="k-1", principal_type="grant_key", user_id=None,
                       grants=[unscoped], grant_key_id="k-1")

    assert ev._event_visible_to(auth, _event("a-9", "ws-9"), _access(store, auth)) is False


def test_unscoped_deny_grant_does_not_grant_visibility(store):
    unscoped_deny = Grant(id="g-deny", resource_id=None, grantee_type="grant_key",
                          grantee_id="k-1", granted_by="admin", can_read=True, effect="deny")
    auth = AuthContext(principal_id="k-1", principal_type="grant_key", user_id=None,
                       grants=[unscoped_deny], grant_key_id="k-1")

    assert ev._event_visible_to(auth, _event("a-9", "ws-9"), _access(store, auth)) is False


def test_write_only_grant_sees_nothing(store):
    _grant(store, "g-1", "a-1", action="update")
    auth = _auth()

    assert ev._event_visible_to(auth, _event("a-1", "ws-1"), _access(store, auth)) is False


# ---------------------------------------------------------------------------
# 2. Deleting a container must not destroy shared artifacts
# ---------------------------------------------------------------------------


def test_shared_artifact_is_evicted_not_destroyed():
    """Deleting a workspace must not reach into another container's artifacts."""
    from mantle.services import workspace_service

    with (
        patch.object(workspace_service, "get_workspace"),  # ownership check is not under test
        # Blocks on a live legacy-lexical connection otherwise.
        patch("mantle.search.ingest.pipeline_unified.delete_artifact_from_index"),
        patch("mantle.db.backend.list_collection_artifacts", return_value=[{"id": "a-1", "root_id": "r-1"}]),
        patch("mantle.db.backend.count_other_containers_for_root", return_value=1),
        patch("mantle.db.backend.delete_artifacts_by_root") as destroy,
        patch("mantle.db.backend.remove_all_edges_for_root") as nuke_edges,
        patch("mantle.db.backend.remove_artifact_from_collection") as evict,
        patch("mantle.db.backend.delete_collection"),
    ):
        workspace_service.delete_workspace(MagicMock(), "user-1", "ws-1", cascade=True)

    # `assert not m.called, "..."` rather than `m.assert_not_called(), "..."`: the latter is a
    # tuple expression whose second element is dead — the check fires, the explanation never can.
    assert not destroy.called, (
        "an artifact still linked into another container was destroyed globally"
    )
    assert not nuke_edges.called, "edges in other containers were removed"
    evict.assert_called_once()
    assert evict.call_args.args[1:] == ("ws-1", "r-1"), (
        f"must evict r-1 from ws-1 only; got {evict.call_args.args[1:]}"
    )


def test_exclusively_owned_artifact_is_destroyed_positive_control():
    """Positive control: an artifact in no other container is deleted, not merely
    evicted — otherwise nothing would ever be deleted, leaking storage forever.
    """
    from mantle.services import workspace_service

    with (
        patch.object(workspace_service, "get_workspace"),  # ownership check is not under test
        # Blocks on a live legacy-lexical connection otherwise.
        patch("mantle.search.ingest.pipeline_unified.delete_artifact_from_index"),
        patch("mantle.db.backend.list_collection_artifacts", return_value=[{"id": "a-1", "root_id": "r-1"}]),
        patch("mantle.db.backend.count_other_containers_for_root", return_value=0),
        patch("mantle.db.backend.delete_artifacts_by_root") as destroy,
        patch("mantle.db.backend.remove_all_edges_for_root") as nuke_edges,
        patch("mantle.db.backend.remove_artifact_from_collection") as evict,
        patch("mantle.db.backend.delete_collection"),
    ):
        workspace_service.delete_workspace(MagicMock(), "user-1", "ws-1", cascade=True)

    destroy.assert_called_once()
    nuke_edges.assert_called_once()
    evict.assert_not_called()


def test_default_delete_detaches_members_instead_of_destroying_them():
    """`cascade` defaults to False: a member is evicted, never destroyed, and the share-count
    check — which only the cascade path needs, to decide destroy vs. evict — is never even
    consulted."""
    from mantle.services import workspace_service

    with (
        patch.object(workspace_service, "get_workspace"),
        patch("mantle.search.ingest.pipeline_unified.delete_artifact_from_index") as reindex,
        patch("mantle.db.backend.list_collection_artifacts", return_value=[{"id": "a-1", "root_id": "r-1"}]),
        patch("mantle.db.backend.count_other_containers_for_root") as shared_check,
        patch("mantle.db.backend.delete_artifacts_by_root") as destroy,
        patch("mantle.db.backend.remove_all_edges_for_root") as nuke_edges,
        patch("mantle.db.backend.remove_artifact_from_collection") as evict,
        patch("mantle.db.backend.delete_collection"),
    ):
        workspace_service.delete_workspace(MagicMock(), "user-1", "ws-1")

    assert not destroy.called, "the default delete destroyed a member instead of detaching it"
    assert not nuke_edges.called
    assert not reindex.called, "a detached member's search index has no reason to change"
    assert not shared_check.called, "share-count only matters to the cascade branch"
    evict.assert_called_once()
    assert evict.call_args.args[1:] == ("ws-1", "r-1")


def test_container_count_failure_fails_safe():
    """A failure counting containers must never destroy.

    A wrong eviction is recoverable; a wrong deletion is not. the lattice's remote
    count could fail mid-flight (and degraded to "shared" → evict); the lattice
    is in-process, so a failing count aborts the delete loudly instead. Either
    way, an UNKNOWN share-count must never reach the destroy branch.
    """
    from mantle.services import workspace_service

    with (
        patch.object(workspace_service, "get_workspace"),  # ownership check is not under test
        patch("mantle.search.ingest.pipeline_unified.delete_artifact_from_index"),
        patch("mantle.db.backend.list_collection_artifacts", return_value=[{"id": "a-1", "root_id": "r-1"}]),
        patch("mantle.db.backend.count_other_containers_for_root",
              side_effect=RuntimeError("store failure")),
        patch("mantle.db.backend.delete_artifacts_by_root") as destroy,
        patch("mantle.db.backend.remove_all_edges_for_root") as nuke_edges,
        patch("mantle.db.backend.remove_artifact_from_collection") as evict,
        patch("mantle.db.backend.delete_collection"),
    ):
        with pytest.raises(RuntimeError):
            workspace_service.delete_workspace(MagicMock(), "user-1", "ws-1", cascade=True)

    assert not destroy.called, (      # see above: `m.assert_not_called(), "..."` kills the message
        "an artifact whose share-count is unknown was destroyed globally"
    )
    nuke_edges.assert_not_called()
    evict.assert_not_called()


# ---------------------------------------------------------------------------
# 3. A signed URL is signed or it is an error
# ---------------------------------------------------------------------------


def test_signing_failure_raises_instead_of_returning_an_unsigned_url():
    """An unsigned URL is an access-control bypass, not a fallback."""
    from mantle.services import content_service

    with (
        patch.object(content_service, "ensure_edge_object_present", return_value=True),
        patch.object(
            content_service._s3_edge_public,
            "generate_presigned_url",
            side_effect=RuntimeError("signing unavailable"),
        ),
        patch.dict("os.environ", {}, clear=False),
    ):
        with pytest.raises(content_service.ContentUrlSigningError):
            content_service.generate_signed_url("some/object/key")


def test_successful_signing_returns_the_signed_url_positive_control():
    from mantle.services import content_service

    with (
        patch.object(content_service, "ensure_edge_object_present", return_value=True),
        patch.object(
            content_service._s3_edge_public,
            "generate_presigned_url",
            return_value="https://edge/key?X-Amz-Signature=abc",
        ),
    ):
        url = content_service.generate_signed_url("some/object/key")

    assert "X-Amz-Signature" in url
