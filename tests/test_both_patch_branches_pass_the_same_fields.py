"""`PATCH /artifacts/{id}` hands the same fields to both of its branches.

The bug this guards, in the router's own words. The handler splits on whether the artifact has a
container: a top-level artifact goes to `update_workspace`, a member to `update_artifact`. The two
services take different signatures, and **both branches have historically dropped fields** — the
comments at each call site record it: *"Omitting them is what made a rewrite silently return 200 and
change nothing"*, and *"their absence was the same bug one branch over: a member update returned
200"*.

That failure mode is invisible by construction. The write is accepted, the status is `200`, the
response is the artifact — and the field the caller sent is simply not there. Nothing raises and
nothing logs.

This asserts the SHAPE rather than the round trip, on purpose. A round-trip test proves the fields
that exist today survive; this proves the two branches cannot DRIFT — a new field added to the model
and wired into one branch fails here, which is exactly how the historical bug arrived. It reads the
AST rather than calling anything, so it cannot be satisfied by a mock.
"""
from __future__ import annotations

import ast
import inspect
import io

import pytest

from mantle.routers import artifacts_router as ar

#: `space_id` is not passed to either service by design: `_parse_supplied_vector(body.vector,
#: body.space_id)` folds it into `vector` before the branch. It is accounted for, not dropped —
#: and naming it here is what keeps "accounted for" different from "forgotten".
_FOLDED_INTO_VECTOR = {"space_id"}


def _branch_kwargs():
    src = io.open(inspect.getsourcefile(ar), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "update_artifact")
    out = {}
    for call in ast.walk(fn):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "offload_sync" and call.args):
            target = call.args[0]
            if isinstance(target, ast.Attribute) and target.attr in (
                    "update_workspace", "update_artifact"):
                out[target.attr] = {k.arg for k in call.keywords if k.arg}
    return out


def test_both_branches_are_present():
    """A vacuous sweep would pass forever — the handler must still have two service calls."""
    got = _branch_kwargs()
    assert set(got) == {"update_workspace", "update_artifact"}, sorted(got)


def test_the_two_branches_pass_identical_field_sets():
    got = _branch_kwargs()
    top, member = got["update_workspace"], got["update_artifact"]
    assert top == member, (
        "the PATCH branches have drifted — top-level only: %s, member only: %s. A field wired into "
        "one branch and not the other is a 200 that changes nothing on half the artifacts."
        % (sorted(top - member), sorted(member - top)))


def test_every_model_field_is_passed_or_explicitly_accounted_for():
    got = _branch_kwargs()
    declared = set(ar.UpdateArtifactRequest.model_fields)
    for branch, passed in got.items():
        unexplained = declared - passed - _FOLDED_INTO_VECTOR
        assert not unexplained, (
            "%s never receives %s, which the request model accepts. A caller can set them and get "
            "a 200 that changed nothing. If one is deliberately handled before the branch, add it "
            "to _FOLDED_INTO_VECTOR with the reason." % (branch, sorted(unexplained)))


@pytest.mark.parametrize("folded", sorted(_FOLDED_INTO_VECTOR))
def test_the_exempt_fields_are_really_used(folded):
    """An exemption list is where a dropped field goes to hide. Each entry must be read somewhere."""
    src = io.open(inspect.getsourcefile(ar), encoding="utf-8").read()
    assert "body.%s" % folded in src, (
        "%s is exempted from the branch check but never read from the body — that is a dropped "
        "field with permission" % folded)
