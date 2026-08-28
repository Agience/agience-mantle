""": `candidates: true` silently replaced `size` and dropped `highlight`.

THE DEFECT, and the shape it takes generally. `candidate_budget` already said *"Ignored for
ordered recall"* — the NEW field documented what it displaced. Nothing said the reverse, so a
caller sending `size` or `highlight` alongside `candidates: true` had them replaced and discarded
with no error and nothing in the spec to warn them. **One direction of a two-way exclusion is the
half that gets written**, because whoever adds the new mode documents the new field and not the
old ones it now overrides.

This derives the overridden set from the handler rather than listing it, so a third override
added later cannot ship undocumented — a hand-written list would have to be remembered, which is
the same failure one level up.
"""
from __future__ import annotations

import ast
import io

import pytest

from mantle.main import app
from mantle.routers import artifacts_router


def _fields_overridden_by_candidates():
    """Request fields the handler replaces when `candidates` is true.

    Each is written `X=<something> if body.candidates else body.X`, so the `else` branch names the
    request field being displaced."""
    tree = ast.parse(io.open(artifacts_router.__file__, encoding="utf-8").read())
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.IfExp):
            continue
        test = node.test
        if not (isinstance(test, ast.Attribute) and test.attr == "candidates"):
            continue
        if isinstance(node.orelse, ast.Attribute):
            found.add(node.orelse.attr)
    return found


@pytest.fixture(scope="module")
def recall_properties():
    spec = app.openapi()
    body = spec["paths"]["/artifacts/recall"]["post"]["requestBody"]
    ref = body["content"]["application/json"]["schema"]["$ref"].split("/")[-1]
    return spec["components"]["schemas"][ref]["properties"]


def test_the_scan_finds_the_overrides():
    """A guard that reaches nothing reports green for ever."""
    found = _fields_overridden_by_candidates()
    assert found == {"size", "highlight"}, (
        "the candidates-mode overrides changed; this test derives them from the handler and the "
        "set is now %r — extend the contract, do not relax the test" % sorted(found))


def test_every_overridden_field_says_it_is_ignored_in_candidates_mode(recall_properties):
    silent = []
    for field in sorted(_fields_overridden_by_candidates()):
        d = (recall_properties[field].get("description") or "").lower()
        if "candidates" not in d or "ignored" not in d:
            silent.append(field)
    assert not silent, (
        "overridden without saying so: %s — a caller sends these and they are discarded with no "
        "error" % ", ".join(silent))


def test_the_other_direction_is_still_documented(recall_properties):
    """`candidate_budget` documented its own exclusion first; it must not regress while the
    reverse is being added."""
    d = (recall_properties["candidate_budget"].get("description") or "").lower()
    assert "ignored for ordered recall" in d, d
