"""POST /artifacts with `source_artifact_id` must authorize the source artifact,
not just the container.

`check_access` must be applied to the source artifact, because
`POST /artifacts {container_id: <mine>, source_artifact_id: <theirs>}` does two
separate things:

  (a) returns `source.to_dict()`, the source content, decrypted. The read
      choke point picks the decryption key from the stored doc's `created_by`,
      so the caller's identity never enters the decryption path — an
      unauthorized read is served with the owner's key regardless.

  (b) writes an edge from the caller's container to the source's root.
      `check_access` walks UP via `get_origin_parent`, so a creation-lineage
      edge (`origin=True`, `propagate=None` — the defaults) would make the
      caller's container the source's origin parent and confer grants over the
      subtree, with no action masked out. The link edge must instead be
      written as non-lineage (`origin=False, propagate=[]`).

This file does NOT inherit `test_router_artifacts.py`'s autouse fixture that
stubs `check_access` to a no-op — the point here is to observe that the call
happens, so it must not be stubbed away.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def parsed():
    return SimpleNamespace(
        container_id="container-owned-by-attacker",
        source_artifact_id="artifact-owned-by-victim",
    )


@pytest.fixture
def source_entity():
    entity = MagicMock()
    entity.root_id = "victim-root"
    entity.id = "artifact-owned-by-victim"
    entity.to_dict.return_value = {"id": "artifact-owned-by-victim", "content": "victim secret"}
    return entity


def _call_link(parsed, source_entity, check_access_mock, add_edge_mock):
    from mantle.routers import artifacts_router

    with (
        patch.object(artifacts_router, "check_access", check_access_mock),
        patch("mantle.db.backend.get_artifact", return_value=source_entity),
        patch("mantle.db.backend.get_latest_committed_artifact", return_value=source_entity),
        patch("mantle.db.backend.add_artifact_to_collection", add_edge_mock),
    ):
        return artifacts_router._link_source_artifact(
            MagicMock(), parsed, SimpleNamespace(user_id="attacker")
        )


def test_source_artifact_is_authorized_for_read(parsed, source_entity):
    """The source artifact must be access-checked, not just the container."""
    check_access = MagicMock()
    add_edge = MagicMock(return_value=True)

    _call_link(parsed, source_entity, check_access, add_edge)

    checked = [c.args[1] for c in check_access.call_args_list if len(c.args) > 1]
    assert "artifact-owned-by-victim" in checked, (
        "the source artifact was never access-checked — any authenticated user "
        "can read any artifact's decrypted content by naming it as source_artifact_id"
    )

    source_calls = [
        c for c in check_access.call_args_list
        if len(c.args) > 1 and c.args[1] == "artifact-owned-by-victim"
    ]
    assert any("read" in c.args for c in source_calls), (
        f"source must be checked for 'read'; got {[c.args for c in source_calls]}"
    )


def test_denied_source_prevents_the_link_and_the_read(parsed, source_entity):
    """Positive control for the denial path: a refused check must stop everything.

    Without this, the test above would pass even if `check_access` were called
    and its result ignored.
    """
    from fastapi import HTTPException

    def deny(*args, **kwargs):
        if len(args) > 1 and args[1] == "artifact-owned-by-victim":
            raise HTTPException(status_code=404, detail="Not found")

    add_edge = MagicMock(return_value=True)

    with pytest.raises(HTTPException):
        _call_link(parsed, source_entity, MagicMock(side_effect=deny), add_edge)

    # `assert not m.called, "..."` rather than `m.assert_not_called(), "..."`: the latter is a
    # tuple expression, so both messages below were dead.
    #
    # `source_entity` is a bare MagicMock, which answers to any attribute name, so `to_dict` is
    # asserted to exist on the real entity first — otherwise a rename would leave this line
    # watching a method nothing calls and reporting success for it.
    from mantle.entities.artifact import Artifact
    assert hasattr(Artifact, "to_dict"), \
        "Artifact.to_dict was renamed — the mock assertion below now watches nothing"

    assert not add_edge.called, "no edge may be written when the source is denied"
    assert not source_entity.to_dict.called, "victim content must not be serialized"


def test_link_edge_is_not_a_grant_inheritance_path(parsed, source_entity):
    """The edge must be a link, not creation lineage.

    Defence in depth: even a caller legitimately allowed to link must not thereby
    graft the source into their own grant subtree. `get_origin_parent` filters
    `origin == true`, and `check_access` breaks when the action is absent from a
    non-None propagate mask — so either of these alone closes the path.
    """
    add_edge = MagicMock(return_value=True)

    _call_link(parsed, source_entity, MagicMock(), add_edge)

    add_edge.assert_called_once()
    kwargs = add_edge.call_args.kwargs

    assert kwargs.get("origin") is False, (
        "link edge was written with origin=True (the default) — it becomes a "
        "creation-lineage edge and check_access inherits grants through it"
    )
    assert kwargs.get("propagate") == [], (
        "propagate must be an empty list, not None. None means ALL actions "
        "propagate; the traversal tests `propagate_mask is not None`, so [] "
        "correctly blocks every action while None blocks nothing"
    )


def test_positive_control_allowed_link_still_returns_the_source(parsed, source_entity):
    """Positive control: the legitimate flow must still work end to end."""
    add_edge = MagicMock(return_value=True)

    result = _call_link(parsed, source_entity, MagicMock(), add_edge)

    assert result["id"] == "artifact-owned-by-victim"
    add_edge.assert_called_once()
