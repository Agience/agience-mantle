"""§2.9 remainder: event scoping, scoped deletion, and signed-URL integrity.

Three live-fire defects:

1. **`/events` unscoped grant fallback.** `_check_grant_permission(grants, "read")`
   was called with no `resource_id`, and that helper skips the resource
   comparison entirely when it is None (`if resource_id and ...`). So the check
   did not mean "holds an unscoped grant" — it meant "holds ANY read grant", and
   every user with any grant saw every tenant's event stream.
2. **`delete_workspace` deleted artifacts globally** via `delete_artifacts_by_root`,
   destroying copies linked into other containers the caller has no rights over.
3. **`generate_signed_url` returned an UNSIGNED url** on any signing error. The
   signature IS the access control, and a caller cannot tell the two apart.

Note on (1): `test_acl_allowed_for_unscoped_read_grant` in `test_events_router.py`
encodes a real DECISION — a genuinely unscoped grant IS a platform-wide viewer.
That behaviour is preserved. What is removed is scoped grants accidentally
getting the same reach.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mantle import event_bus
from mantle.routers import events_router as ev


class _Grant:
    def __init__(self, **flags):
        for k in ("read", "update", "create", "delete", "invoke", "add", "search"):
            setattr(self, f"can_{k}", flags.get(k, False))
        self.resource_id = flags.get("resource_id")
        self.effect = flags.get("effect", "allow")


def _auth(grants=(), principal_id=None):
    return SimpleNamespace(grants=list(grants), principal_id=principal_id)


# ---------------------------------------------------------------------------
# 1. Event visibility is actually scoped
# ---------------------------------------------------------------------------


def test_scoped_grant_does_not_see_unrelated_events():
    """THE REGRESSION: a grant on one artifact must not reveal every event."""
    auth = _auth(grants=[_Grant(read=True, resource_id="a-1")])
    unrelated = event_bus.Event(
        name="x", payload={}, artifact_id="a-9", container_id="ws-9"
    )

    assert ev._event_visible_to(auth, unrelated) is False, (
        "a read grant on a-1 exposed an event on a-9 — every user holding any "
        "grant saw every other tenant's event stream"
    )


def test_scoped_grant_still_sees_its_own_events_positive_control():
    """POSITIVE CONTROL: legitimate visibility must survive the fix."""
    auth = _auth(grants=[_Grant(read=True, resource_id="a-1")])
    own = event_bus.Event(name="x", payload={}, artifact_id="a-1", container_id="ws-1")

    assert ev._event_visible_to(auth, own) is True


def test_genuinely_unscoped_grant_is_still_platform_wide():
    """The DECISION encoded by test_acl_allowed_for_unscoped_read_grant stands.

    A grant with no resource_id is a platform-wide viewer. That was deliberate;
    only the accidental widening of *scoped* grants was a defect.
    """
    auth = _auth(grants=[_Grant(read=True)])  # resource_id is None
    any_event = event_bus.Event(
        name="x", payload={}, artifact_id="a-9", container_id="ws-9"
    )

    assert ev._event_visible_to(auth, any_event) is True


def test_unscoped_deny_grant_does_not_grant_visibility():
    auth = _auth(grants=[_Grant(read=True, effect="deny")])
    any_event = event_bus.Event(
        name="x", payload={}, artifact_id="a-9", container_id="ws-9"
    )

    assert ev._event_visible_to(auth, any_event) is False


def test_write_only_grant_sees_nothing():
    auth = _auth(grants=[_Grant(update=True, resource_id="a-1")])
    own = event_bus.Event(name="x", payload={}, artifact_id="a-1", container_id="ws-1")

    assert ev._event_visible_to(auth, own) is False


# ---------------------------------------------------------------------------
# 2. Deleting a container must not destroy shared artifacts
# ---------------------------------------------------------------------------


def test_shared_artifact_is_evicted_not_destroyed():
    """THE REGRESSION: deleting your workspace must not reach into another."""
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
        workspace_service.delete_workspace(MagicMock(), "user-1", "ws-1")

    destroy.assert_not_called(), (
        "an artifact still linked into another container was destroyed globally"
    )
    nuke_edges.assert_not_called(), "edges in other containers were removed"
    evict.assert_called_once()
    assert evict.call_args.args[1:] == ("ws-1", "r-1"), (
        f"must evict r-1 from ws-1 only; got {evict.call_args.args[1:]}"
    )


def test_exclusively_owned_artifact_is_destroyed_positive_control():
    """POSITIVE CONTROL: an artifact in no other container IS deleted.

    Without this the fix could silently degrade into never deleting anything,
    leaking storage forever.
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
        workspace_service.delete_workspace(MagicMock(), "user-1", "ws-1")

    destroy.assert_called_once()
    nuke_edges.assert_called_once()
    evict.assert_not_called()


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
            workspace_service.delete_workspace(MagicMock(), "user-1", "ws-1")

    destroy.assert_not_called(), (
        "an artifact whose share-count is unknown was destroyed globally"
    )
    nuke_edges.assert_not_called()
    evict.assert_not_called()


# ---------------------------------------------------------------------------
# 3. A signed URL is signed or it is an error
# ---------------------------------------------------------------------------


def test_signing_failure_raises_instead_of_returning_an_unsigned_url():
    """THE REGRESSION: an unsigned URL is an access-control bypass, not a fallback."""
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
