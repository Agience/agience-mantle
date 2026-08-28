"""A delete reports how many members it detached and how many it destroyed.

The state this replaces. Every outcome answered `{"id": ..., "deleted": true}`. A `cascade=true`
over a 1.85M-member collection returned the same three-word body as deleting a note — and the one
distinction the caller most needs is the one it omitted: **detached is recoverable, destroyed is
not.**

The first fix was worse than the bug. It spread the service's return into the response
(`{"id": ..., "deleted": True, **(counts or {})}`). One test's mock returns the popped artifact
DOCUMENT rather than counts, and the delete answered with the whole document. **A blind spread of
an internal return into an API response is how internal fields reach a caller**: the shape is
whatever the callee happened to return that day, and nothing declares it.

The second fix defeated a different gate. Building the whole body in a helper made the handler
`return _deleted(...)`, and `test_response_envelopes_match_the_handlers` reads `return {...}`
LITERALS out of each handler, refusing any documented envelope it cannot read — *"nothing can check
what it promises"*. That rule is right. So the literal lives at the return site and only the
sanitising is shared: `_delete_counts` returns two integers and nothing else can cross.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mantle.routers.artifacts_router import DeleteArtifactResponse, _delete_counts
from mantle.services import workspace_service as ws_svc


def test_the_counts_are_read_from_the_service_result():
    assert _delete_counts({"detached": 3, "destroyed": 7, "refused": 2}) == (3, 7, 2)


@pytest.mark.parametrize("hostile", [
    {"_key": "leak", "content": "secret", "context": "{}"},   # an artifact document
    None,                                                      # a callee that returns nothing
    "not a dict",
    {"detached": "3", "destroyed": None, "refused": []},       # right keys, wrong types
])
def test_only_two_integers_can_cross_the_boundary(hostile):
    got = _delete_counts(hostile)
    assert all(isinstance(v, int) for v in got), got
    assert got == (0, 0, 0)


def test_the_declared_model_carries_the_counts():
    """A field a caller receives and cannot find in the spec is not a contract."""
    assert set(DeleteArtifactResponse.model_fields) >= {"id", "deleted", "detached",
                                                       "destroyed", "refused"}


def test_a_cascade_counts_destroyed_and_a_detach_counts_detached():
    """The counts must come from the branch that ran, not from the number of rows."""
    db = MagicMock()
    rows = [{"id": "a-1", "root_id": "r-1"}, {"id": "a-2", "root_id": "r-2"}]

    def _run(cascade, others):
        with (
            # : the container IS the members' origin parent here, so the caller's grant
            # is inherited and the destroy branch is the one under test. Without this the guard
            # refuses on a MagicMock store and this would measure the refusal path instead.
            patch("mantle.services.workspace_service.store.get_origin_parent",
                  return_value=("c-1", 0)),
            patch("mantle.services.workspace_service.store.count_other_containers_for_root",
                  return_value=others),
            patch("mantle.services.workspace_service.store.delete_artifacts_by_root"),
            patch("mantle.services.workspace_service.store.remove_all_edges_for_root"),
            patch("mantle.services.workspace_service.store.remove_artifact_from_collection"),
            patch("mantle.search.ingest.pipeline_unified.delete_artifact_from_index"),
            patch.object(ws_svc, "_emit_event"),
        ):
            return ws_svc._delete_or_detach_members(db, "c-1", "u-1", rows, cascade=cascade)

    assert _run(True, 0) == {"detached": 0, "destroyed": 2, "refused": 0}, "sole-contained members are destroyed"
    assert _run(True, 1) == {"detached": 2, "destroyed": 0, "refused": 0}, "members with another home are evicted"
    assert _run(False, 0) == {"detached": 2, "destroyed": 0, "refused": 0}, "the default destroys nothing"
