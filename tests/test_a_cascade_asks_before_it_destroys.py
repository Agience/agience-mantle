"""A cascade destroys a member only where the caller's grant actually reaches it.

This file replaces `test_cascade_destroys_without_asking_about_the_member.py`, which asserted the
ABSENCE of any check and said in its own docstring that it pinned current behaviour rather than
endorsing it. John; this asserts the new rule.

What was wrong. `DELETE /artifacts/{id}?cascade=true` authorized `delete` on the CONTAINER once,
then destroyed any member whose root sat in no other container — asking nothing about any of them.
The docstring treated sole-containment as equivalent to holding the grant. It is not, and the code
next door proves it:

  * `check_access` is *"direct grant first, then walk ORIGIN edges upward"*, and
    `get_origin_parent` returns a parent only where `is_origin` is set.
  * `_link_source_artifact` links with `origin=False` **deliberately**, because an `origin=True`
    link *"would let the linking container be returned as the source's origin parent and confer
    grants over the whole subtree."*

So a link is specifically prevented from conferring authority — and a cascade destroyed through
that same link. Linking needs only `read` on the source.

THE RULE: check the member's own delete right ONLY where this container is not its origin parent.
A member genuinely rooted here already inherits the grant, so asking again is a read that can only
say yes; the cost falls on linked members, which are the minority.

A refused destroy is EVICTED and COUNTED, not silently downgraded — `refused` in the response.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


from mantle.services import workspace_service as ws_svc

ROWS = [{"id": "a-1", "root_id": "root-1"}]
CONTAINER = "container-1"


def _run(*, origin_parent, may_delete, cascade=True, others=0):
    """Drive the real member loop with the two facts the guard turns on."""
    db = MagicMock()
    with (
        patch.object(ws_svc.store, "get_origin_parent", return_value=origin_parent),
        patch.object(ws_svc, "_may_delete_draft", return_value=may_delete) as asked,
        patch.object(ws_svc.store, "count_other_containers_for_root", return_value=others),
        patch.object(ws_svc.store, "delete_artifacts_by_root") as destroy,
        patch.object(ws_svc.store, "remove_all_edges_for_root"),
        patch.object(ws_svc.store, "remove_artifact_from_collection") as evict,
        patch("mantle.search.ingest.pipeline_unified.delete_artifact_from_index"),
        patch.object(ws_svc, "_emit_event"),
    ):
        counts = ws_svc._delete_or_detach_members(
            db, CONTAINER, "u-1", ROWS, cascade=cascade, auth=MagicMock())
    return counts, destroy, evict, asked


def test_a_member_rooted_here_is_destroyed_without_being_asked_about():
    """The container IS the origin parent, so the grant is already inherited. Asking would be a
    read that can only say yes."""
    counts, destroy, evict, asked = _run(origin_parent=(CONTAINER, 0), may_delete=False)
    destroy.assert_any_call(destroy.call_args.args[0], "root-1")
    assert counts == {"detached": 0, "destroyed": 1, "refused": 0}, counts
    assert not asked.called, "a member rooted in this container was re-checked needlessly"
    evict.assert_not_called()


def test_a_linked_member_the_caller_may_delete_is_destroyed():
    """Origin is elsewhere, but the caller holds the right in its own name."""
    counts, destroy, evict, asked = _run(origin_parent=("somewhere-else", 0), may_delete=True)
    assert counts == {"detached": 0, "destroyed": 1, "refused": 0}, counts
    assert asked.called, "a linked member was destroyed without checking the caller's own right"
    evict.assert_not_called()


def test_a_linked_member_the_caller_may_not_delete_is_EVICTED_not_destroyed():
    """The finding, closed. The artifact survives and the caller is told."""
    counts, destroy, evict, asked = _run(origin_parent=("somewhere-else", 0), may_delete=False)
    destroy.assert_not_called()
    evict.assert_any_call(evict.call_args.args[0], CONTAINER, "root-1")
    assert counts == {"detached": 0, "destroyed": 0, "refused": 1}, counts


def test_a_rootless_member_is_checked():
    """`get_origin_parent` returning None means the member is its own root — this container's grant
    does not reach it, so it must be asked about."""
    counts, destroy, evict, asked = _run(origin_parent=None, may_delete=False)
    assert asked.called
    assert counts["refused"] == 1, counts
    destroy.assert_not_called()


def test_an_unreadable_origin_edge_refuses_rather_than_assuming_yes():
    """The uncertainty this guard exists for. A failed lookup must not cost the caller an artifact."""
    db = MagicMock()
    with (
        patch.object(ws_svc.store, "get_origin_parent", side_effect=RuntimeError("edges truncated")),
        patch.object(ws_svc.store, "count_other_containers_for_root", return_value=0),
        patch.object(ws_svc.store, "delete_artifacts_by_root") as destroy,
        patch.object(ws_svc.store, "remove_all_edges_for_root"),
        patch.object(ws_svc.store, "remove_artifact_from_collection"),
        patch("mantle.search.ingest.pipeline_unified.delete_artifact_from_index"),
        patch.object(ws_svc, "_emit_event"),
    ):
        counts = ws_svc._delete_or_detach_members(
            db, CONTAINER, "u-1", ROWS, cascade=True, auth=MagicMock())
    destroy.assert_not_called()
    assert counts["refused"] == 1, counts


def test_a_member_with_another_home_is_evicted_and_never_reaches_the_guard():
    """The positive control for the rule that was already enforced: sole-containment gates
    destruction, and that check comes first."""
    counts, destroy, evict, asked = _run(
        origin_parent=("somewhere-else", 0), may_delete=False, others=1)
    destroy.assert_not_called()
    assert counts == {"detached": 1, "destroyed": 0, "refused": 0}, counts
    assert not asked.called, "an already-shared member was put through the delete-right check"


def test_the_default_branch_destroys_nothing_at_all():
    """`cascade=False` never reaches the guard, and never destroys."""
    counts, destroy, evict, asked = _run(
        origin_parent=None, may_delete=False, cascade=False)
    destroy.assert_not_called()
    assert not asked.called
    assert counts == {"detached": 1, "destroyed": 0, "refused": 0}, counts


def test_the_fail_closed_helper_still_fails_closed():
    """`_may_delete_draft` is what the guard leans on for a linked member."""
    assert ws_svc._may_delete_draft(MagicMock(), None, "a-1") is False
