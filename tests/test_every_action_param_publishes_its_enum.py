"""Every `action` parameter publishes the vocabulary it is validated against.

The state this replaces. `action` was a free `string` in the spec on both routes that take one,
while the code rejected anything outside `mantle.attenuation.ACTIONS` with `400 Unknown action`.
The two descriptions hand-listed the vocabulary and both were incomplete: `/artifacts/visible` said
*"(read, create, add, update, ...)"* — five of nine — and `/grants/my-access` said
*"(read/update/invoke/…)"* — three of nine. **A caller could not learn the vocabulary from the
spec, and met a 400 it had no way to avoid.**

The enum is DERIVED from `ACTIONS` at import, never typed out. A `Literal[...]` would restate
the tuple as literals and become a second home for the vocabulary that drifts in silence — which is
exactly what the two prose lists already were. This test pins the derivation rather than the value,
so adding an action to `ACTIONS` needs no edit here, and a NEW route that forgets the enum fails.

The assertion was shown to discriminate before it was made to pass: swept against the
PRE-fix spec it read `GET /artifacts/visible -> enum=None` and
`GET /grants/my-access -> enum=None`, both `matches ACTIONS: False`.
"""
from __future__ import annotations

import pytest

from mantle.attenuation import ACTIONS
from mantle.main import app


@pytest.fixture(scope="module")
def action_params():
    spec = app.openapi()
    found = [(method.upper(), path, param)
             for path, item in spec["paths"].items()
             for method, op in item.items()
             if isinstance(op, dict)
             for param in (op.get("parameters") or [])
             if param.get("name") == "action"]
    # Both known routes that take a caller-supplied action. If this number moves, a route was
    # added or removed — decide deliberately whether it belongs, do not just re-pin.
    assert len(found) == 2, "expected 2 `action` parameters, found %d: %r" % (
        len(found), [(m, p) for m, p, _ in found])
    return found


def test_every_action_parameter_publishes_the_full_enum(action_params):
    for method, path, param in action_params:
        assert param["schema"].get("enum") == list(ACTIONS), (
            "%s %s publishes enum=%r, but the handler validates against ACTIONS=%r"
            % (method, path, param["schema"].get("enum"), list(ACTIONS)))


def test_the_description_names_every_permitted_value(action_params):
    """The prose that replaced the two truncated lists must stay complete."""
    for method, path, param in action_params:
        desc = param.get("description") or ""
        missing = [a for a in ACTIONS if a not in desc]
        assert not missing, "%s %s description omits %r" % (method, path, missing)


def test_actions_is_non_empty_so_the_assertions_above_can_fail():
    assert len(ACTIONS) >= 2
