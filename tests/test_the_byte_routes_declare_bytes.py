""": the content routes move raw bytes and the spec described neither direction.

THE DEFECT, both ways round:

* `GET .../content` returns `Response(content=<bytes>)`, and FastAPI infers `application/json` for
  every 200 it is not told otherwise about. The spec said JSON with an empty schema, so a
  generated client built a parser for a body that is never JSON.
* `PUT .../content` reads `await request.body()`, which FastAPI cannot see, so the operation
  published **no requestBody at all** — a generated client had no method to upload with.

Both are derived here rather than listed. A route that starts moving bytes tomorrow is caught
without anyone remembering to extend a hand-written set, which is the failure one level up from
the one being fixed.
"""
from __future__ import annotations

import ast
import io

import pytest

from mantle.main import app
from mantle.routers import artifacts_router

_BINARY = "application/octet-stream"


def _handlers():
    """Route handlers in this router, with the (path, method) their decorator names."""
    tree = ast.parse(io.open(artifacts_router.__file__, encoding="utf-8").read())
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            if not (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)):
                continue
            if d.func.attr not in ("get", "post", "put", "patch", "delete") or not d.args:
                continue
            if isinstance(d.args[0], ast.Constant):
                out[node.name] = ("/artifacts" + d.args[0].value, d.func.attr)
    return out


def _op(path, method):
    return app.openapi()["paths"][path][method]


def _reads_raw_body(node):
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "body"
                and getattr(sub.func.value, "id", "") == "request"):
            return True
    return False


def _returns_raw_response(node):
    for sub in ast.walk(node):
        if not (isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "Response"):
            continue
        if any(k.arg == "content" for k in sub.keywords):
            return True
    return False


@pytest.fixture(scope="module")
def routed():
    tree = ast.parse(io.open(artifacts_router.__file__, encoding="utf-8").read())
    names = _handlers()
    return [(n, names[n.name]) for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]


def test_the_scan_finds_the_byte_routes(routed):
    """A guard that reaches nothing reports green for ever."""
    uploads = {where for node, where in routed if _reads_raw_body(node)}
    downloads = {where for node, where in routed if _returns_raw_response(node)}
    assert uploads, "no handler reads a raw body — the scan is broken, not the router"
    assert downloads, "no handler returns a raw Response — the scan is broken"


def test_every_handler_that_reads_a_raw_body_declares_one(routed):
    missing = []
    for node, (path, method) in routed:
        if not _reads_raw_body(node):
            continue
        content = (_op(path, method).get("requestBody") or {}).get("content", {})
        if _BINARY not in content:
            missing.append("%s %s (%s) -> %s" % (method.upper(), path, node.name,
                                                 sorted(content) or "no requestBody"))
    assert not missing, (
        "reads the raw body but publishes no binary requestBody, so a generated client cannot "
        "send one:" + chr(10) + "  " + (chr(10) + "  ").join(missing))


def test_every_handler_that_returns_raw_bytes_declares_a_binary_200(routed):
    wrong = []
    for node, (path, method) in routed:
        if not _returns_raw_response(node):
            continue
        r200 = (_op(path, method).get("responses", {}).get("200") or {}).get("content", {})
        # A 204-only route legitimately has no 200 body.
        if not r200:
            continue
        if _BINARY not in r200:
            wrong.append("%s %s (%s) -> %s" % (method.upper(), path, node.name, sorted(r200)))
    assert not wrong, (
        "returns raw bytes but declares a non-binary 200, so a client will try to parse it:"
        + chr(10) + "  " + (chr(10) + "  ").join(wrong))
