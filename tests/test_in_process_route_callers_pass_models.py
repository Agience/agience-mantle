"""Anything calling an artifacts ROUTE function in-process must hand it the declared model.

The break this exists to stop, and it happened. `POST /artifacts` used to take
`body: Dict[str, Any] = Body(...)` and parse by hand. When it was changed to declare
`body: CreateArtifactRequest`, `mcp_router` — which calls the handler DIRECTLY, bypassing FastAPI
and therefore bypassing request validation — was still passing `dict(args)`. The handler reached
`body.context` and raised `'dict' object has no attribute 'context'`. Eight tests failed, and none
of them was an artifacts test: the MCP suite caught it, which is luck rather than design.

The general shape. A route function has TWO callers of different kinds — FastAPI, which validates
and coerces the body, and any in-process caller, which does neither. Changing a route's signature
is therefore a change to an internal API as well as to the wire, and nothing in the type system says
so. `mcp_router` already built the model for `update_artifact` and `recall`; `create` was the odd
one out only because the route it called had never asked for one.
"""
from __future__ import annotations

import ast
import io
import os

import pytest

_ROUTER = "routers/artifacts_router.py"
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "mantle")


def _route_functions() -> set:
    tree = ast.parse(io.open(os.path.join(_SRC, _ROUTER), encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            if isinstance(d, ast.Call):
                f = d.func
                verb = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if verb in ("get", "post", "put", "patch", "delete"):
                    names.add(node.name)
    return names


def _in_process_calls():
    """`(file, line, route, body_kind)` for every call into a route through the router module."""
    routes = _route_functions()
    out = []
    for dirpath, _dirs, files in os.walk(_SRC):
        for fn in files:
            if not fn.endswith(".py") or fn == "artifacts_router.py":
                continue
            path = os.path.join(dirpath, fn)
            text = io.open(path, encoding="utf-8").read()
            if "artifacts_router" not in text and "import artifacts" not in text:
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:                                  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr not in routes:
                    continue
                if getattr(node.func.value, "id", "") not in ("artifacts", "artifacts_router"):
                    continue
                body = next((k.value for k in node.keywords if k.arg == "body"), None)
                if body is None:
                    kind = "none"
                elif isinstance(body, ast.Call):
                    f = body.func
                    kind = "model:" + (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "?"))
                else:
                    kind = "raw:" + type(body).__name__
                out.append((os.path.relpath(path, _SRC).replace("\\", "/"),
                            node.lineno, node.func.attr, kind))
    return out


def test_the_scan_finds_the_calls_it_is_written_for():
    """A guard that reaches nothing reports green for ever. `mcp_router` is the known caller."""
    calls = _in_process_calls()
    assert len(calls) >= 5, "the scan found %d in-process route calls — it cannot have failed" % len(calls)
    assert any(c[0].endswith("mcp_router.py") for c in calls), (
        "the scan does not reach `mcp_router`, the one module known to call these handlers "
        "directly — it is scanning the wrong tree")


@pytest.mark.parametrize("call", _in_process_calls(), ids=lambda c: "%s:%d:%s" % (c[0], c[1], c[2]))
def test_a_body_passed_in_process_is_the_declared_model(call):
    path, lineno, route, kind = call
    assert not kind.startswith("raw:"), (
        "%s:%d calls `%s` with a %s body. That handler declares a pydantic model, and this call "
        "bypasses FastAPI — so nothing validates or coerces it and the handler will reach an "
        "attribute the dict does not have. Build the model, as the calls beside it do."
        % (path, lineno, route, kind[4:]))


def test_every_body_bearing_call_names_the_route_s_own_model():
    """Not merely *a* model — the right one. `update_artifact` taking a `CreateArtifactRequest`
    would satisfy the test above and still be wrong."""
    expected = {"create_artifact": "CreateArtifactRequest",
                "update_artifact": "UpdateArtifactRequest",
                "recall_artifacts": "ArtifactRecallRequest"}
    for path, lineno, route, kind in _in_process_calls():
        if not kind.startswith("model:"):
            continue
        want = expected.get(route)
        if want is None:
            continue
        assert kind == "model:" + want, (
            "%s:%d passes %s to `%s`, which declares %s" % (path, lineno, kind[6:], route, want))


# ── the shape that hid ───────────────────────────────────────────────────────────────────────

def _default_create_callers():
    """`(file, line, arg_kind)` for every call to `_default_create_artifact`, anywhere.

    SCANNED ACROSS `tests/` TOO, and that is the point. The first version of this guard scanned
    only `src/`, so it reported clean while SIX tests were passing raw dicts positionally to this
    function — a shape the scan above cannot see, because it looks for a `body=` keyword on a
    module attribute. Two different blind spots, one defect.
    """
    roots = [_SRC, os.path.dirname(os.path.abspath(__file__))]
    out = []
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                text = io.open(path, encoding="utf-8").read()
                if "_default_create_artifact" not in text:
                    continue
                try:
                    tree = ast.parse(text)
                except SyntaxError:                              # pragma: no cover
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    f = node.func
                    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                    if name != "_default_create_artifact" or not node.args:
                        continue
                    a = node.args[0]
                    kind = ("model" if isinstance(a, ast.Call) else
                            "raw-dict" if isinstance(a, ast.Dict) else
                            "name" if isinstance(a, ast.Name) else type(a).__name__)
                    out.append((os.path.basename(path), node.lineno, kind))
    return out


def test_the_default_create_scan_reaches_the_tests_that_broke():
    calls = _default_create_callers()
    assert len(calls) >= 6, "found only %d calls — the scan is not reaching them" % len(calls)
    assert any(c[0] == "test_artifact_identity.py" for c in calls), (
        "the scan does not reach `test_artifact_identity.py`, which holds six of these calls")


@pytest.mark.parametrize("call", _default_create_callers(), ids=lambda c: "%s:%d" % (c[0], c[1]))
def test_default_create_is_never_handed_a_raw_dict(call):
    path, lineno, kind = call
    assert kind != "raw-dict", (
        "%s:%d passes a raw dict to `_default_create_artifact`, which takes "
        "`CreateArtifactRequest`. Nothing coerces it — the handler reaches `body.context` and "
        "raises AttributeError." % (path, lineno))

