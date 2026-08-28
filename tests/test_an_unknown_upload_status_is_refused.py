"""An unrecognised upload status is refused, not silently ignored.

The defect, and it was a docstring that lied. `update_upload_status` promised, in its own
docstring: *"Raises ``HTTPException(400)`` for an unrecognised ``status_value``"*. It did not. The
body tested `status_value in ("uploading", "failed")` and `status_value == "complete"` and had **no
else**, so any other string fell through every branch, changed nothing, and returned the artifact
with 200.

A client that sent `"Complete"` or `"completed"` was told it had succeeded and was left with an
upload that never finalised: the object never mirrored to durable storage, the `upload` section
never dropped, the artifact never indexed on the ordinary update path — and nothing anywhere
recording that a status had been ignored. The failure is silent on both sides, which is what makes
it worse than an error.

Nothing that worked before behaves differently: an unrecognised value already did nothing, so the
only requests newly refused are ones that were already silent no-ops.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from mantle.services import workspace_service as ws


@pytest.mark.parametrize("bad", ["banana", "Complete", "completed", "COMPLETE", "", "done"])
def test_an_unrecognised_status_raises_400(bad):
    """Including the near-misses, which are the ones a real client actually sends: a capitalised
    `Complete` and a past-tense `completed` both look right in a log."""
    with pytest.raises(HTTPException) as exc:
        ws.update_upload_status(db=None, user_id="u", workspace_id="w", upload_id="a",
                                status_value=bad)
    assert exc.value.status_code == 400
    assert "Unknown upload status" in str(exc.value.detail)
    assert "uploading" in str(exc.value.detail), (
        "the refusal must name what IS accepted, or the caller is left guessing")


def test_the_accepted_set_is_exactly_what_the_branches_act_on():
    """The constant and the code must not drift.

    `_UPLOAD_STATUSES` is now the gate, so if a branch is added for a new status and the tuple is
    not updated, that status becomes unreachable — the opposite defect, equally silent."""
    import ast
    import inspect

    src = inspect.getsource(ws.update_upload_status)
    tree = ast.parse("def _(): pass" if not src else src.lstrip())
    compared = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) \
                and node.left.id == "status_value":
            for comp in node.comparators:
                if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                    compared.add(comp.value)
                elif isinstance(comp, (ast.Tuple, ast.List)):
                    for e in comp.elts:
                        if isinstance(e, ast.Constant) and isinstance(e.value, str):
                            compared.add(e.value)
    # `_UPLOAD_STATUSES` itself appears in the guard; the branch constants are the rest.
    assert compared <= set(ws._UPLOAD_STATUSES), (
        "a branch acts on %r, which the accepted set does not allow — that status is unreachable"
        % (compared - set(ws._UPLOAD_STATUSES)))
    assert set(ws._UPLOAD_STATUSES) == compared, (
        "the accepted set allows %r, which no branch acts on — it would be a silent no-op again"
        % (set(ws._UPLOAD_STATUSES) - compared))
