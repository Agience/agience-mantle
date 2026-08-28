""": `POST /artifacts/recall` has two 200 shapes and declared neither.

`candidates: true` returns a different resource in everything but the URL — `{candidates,
model_id}` instead of ordered, hydrated hits. The route declared NO 200 model at all, so a
generated client had a type for neither, and `ArtifactRecallResponse` existed in the router
without ever being wired to the route it describes.

Q7 rode on the same gap: `ordering` (the OUTCOME vocabulary — `semantic` / `reach` /
`coverage` / `recency`) lives on that response model, so while the model was unwired the outcome
vocabulary reached no client at all. `sort` (the REQUEST vocabulary — `relevance` / `recency`)
was published all along, which is what made the mismatch invisible rather than merely undocumented.

Declaring two shapes is only honest if both are knowable. The ordered branch constructs
`ArtifactRecallResponse` itself. The candidates branch returns the accessor's dict verbatim — so
this file checks that every one of the accessor's returns is the literal the model promises,
rather than taking the docstring's word for it.
"""
from __future__ import annotations

import ast
import io

import pytest

from mantle.main import app


@pytest.fixture(scope="module")
def spec():
    return app.openapi()


def _recall_200(spec):
    return spec["paths"]["/artifacts/recall"]["post"]["responses"]["200"]


def test_both_shapes_are_declared(spec):
    schema = _recall_200(spec)["content"]["application/json"]["schema"]
    variants = schema.get("anyOf") or schema.get("oneOf") or []
    names = {v.get("$ref", "").split("/")[-1] for v in variants}
    assert names == {"ArtifactRecallResponse", "RecallCandidatesResponse"}, names


def test_the_outcome_vocabulary_reaches_the_client(spec):
    """Q7. `ordering` is the whole point of the response model for a caller who sent `sort`."""
    props = spec["components"]["schemas"]["ArtifactRecallResponse"]["properties"]
    assert props["ordering"].get("enum") == ["semantic", "reach", "coverage", "recency"], (
        props["ordering"])


def test_the_request_vocabulary_points_at_the_outcome_one(spec):
    """The two word-sets differ, and a client that sent `relevance` and read `coverage` needs to
    know that is the ordinary answer rather than the server ignoring it."""
    ref = (spec["paths"]["/artifacts/recall"]["post"]["requestBody"]["content"]
           ["application/json"]["schema"]["$ref"].split("/")[-1])
    sort = spec["components"]["schemas"][ref]["properties"]["sort"]
    assert sort["anyOf"][0]["enum"] == ["relevance", "recency"], sort
    assert "ordering" in (sort.get("description") or ""), (
        "`sort` does not tell the caller where to read what actually happened")


def test_the_candidates_shape_is_what_the_accessor_actually_returns():
    """The check that makes the declaration honest.

    A declared envelope nothing verifies is worse than an undeclared one. The ordered branch
    builds its own model, but the candidates branch hands back the accessor's dict verbatim — so
    the promise is only as good as that dict. Every `return` in the accessor is read here."""
    from mantle.search.mantle.sse import router_accessor

    tree = ast.parse(io.open(router_accessor.__file__, encoding="utf-8").read())
    shapes = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "candidates"):
            continue
        for r in ast.walk(node):
            if isinstance(r, ast.Return) and isinstance(r.value, ast.Dict):
                shapes.append({k.value for k in r.value.keys if isinstance(k, ast.Constant)})

    assert shapes, "no dict return found in the accessor — the scan is looking in the wrong place"
    for shape in shapes:
        assert shape == {"candidates", "model_id"}, (
            "the accessor returns %s, which `RecallCandidatesResponse` does not declare" % shape)


RETIRED = ("embedding", "aperture", "use_hybrid", "content_types")


def test_the_retired_fields_are_still_inert(spec):
    """. Four removed fields still PARSE — pydantic ignores unknowns — and do nothing.

    That is a compatibility window with no closing date, and this test does not invent one:
    closing it means an old client that works starts getting 422s, which is a call about callers
    in the field. What it does pin is that they stay INERT — a field quietly coming back as a real
    one would make the schema's "has no effect" a lie, and a client relying on being ignored would
    start being obeyed."""
    ref = (spec["paths"]["/artifacts/recall"]["post"]["requestBody"]["content"]
           ["application/json"]["schema"]["$ref"].split("/")[-1])
    schema = spec["components"]["schemas"][ref]

    declared = set(schema["properties"])
    back = sorted(f for f in RETIRED if f in declared)
    assert not back, "retired field(s) declared again as real: %s" % back

    desc = schema.get("description") or ""
    unnamed = [f for f in RETIRED if f not in desc]
    assert not unnamed, (
        "the schema does not name these retired fields, so a client sending one cannot learn it "
        "is ignored: %s" % unnamed)


def test_no_retired_field_is_read_on_the_request_path():
    """The other half: named as retired AND actually unread. A name in a docstring is a claim."""
    import ast as _ast
    import io as _io

    from mantle.routers import artifacts_router

    tree = _ast.parse(_io.open(artifacts_router.__file__, encoding="utf-8").read())
    read = set()
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Attribute) and node.attr in RETIRED
                and getattr(node.value, "id", "") == "body"):
            read.add(node.attr)
    assert not read, "retired field(s) read off the request body: %s" % sorted(read)

