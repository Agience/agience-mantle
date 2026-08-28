"""Every handler MCP calls directly is called with everything that has no default.

Why this exists, and why it is a test rather than a comment. `mcp_router` does not go through
FastAPI: it imports the REST handlers and calls them with keyword arguments. That means the
handler has TWO contracts — the HTTP one FastAPI builds from its signature, and this direct one —
and **only the HTTP one is checked by anything**. Adding a parameter to a route is a normal,
type-safe change that silently breaks the MCP surface at call time.

It has now happened twice:

  * the day `POST /artifacts` declared `body: CreateArtifactRequest`, this call was still handing
    it a raw dict (recorded in `mcp_router`'s own comment);
  * 2026-08-26, when /C7 added `response: Response` so the route could answer `200` for a
    write that created nothing. Three MCP tests failed with
    `create_artifact() missing 1 required positional argument: 'response'`.

Both were caught by tests of the MCP tools rather than by anything checking the seam itself, and
the second was found only because the full suite ran. This asserts the seam directly, so the next
signature change fails with a message that names the parameter and the call.

A parameter with a `Depends(...)` default is deliberately NOT required here: it has a default, so
Python accepts the call, and `mcp_router` passes the real `auth`/`store_db` explicitly anyway.
"""
from __future__ import annotations

import ast
import inspect
import io

import pytest

from mantle.routers import artifacts_router as artifacts
from mantle.routers import mcp_router


def _direct_calls():
    """Every `artifacts.<handler>(...)` call in mcp_router, with the keywords it passes."""
    src = io.open(inspect.getsourcefile(mcp_router), encoding="utf-8").read()
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "artifacts":
            if any(k.arg is None for k in node.keywords):
                continue  # `**kwargs` — cannot be checked statically
            out.append((f.attr, {k.arg for k in node.keywords}, len(node.args), node.lineno))
    return out


def test_the_seam_is_not_empty():
    """A vacuous sweep would pass forever. mcp_router calls seven handlers directly."""
    calls = _direct_calls()
    assert len(calls) >= 6, "found only %d direct handler calls: %r" % (len(calls), calls)


@pytest.mark.parametrize("call", _direct_calls(), ids=lambda c: "%s:L%d" % (c[0], c[3]))
def test_every_direct_call_supplies_every_defaultless_parameter(call):
    name, kwargs, n_positional, lineno = call
    handler = getattr(artifacts, name, None)
    assert handler is not None, "mcp_router calls artifacts.%s, which does not exist" % name

    params = list(inspect.signature(handler).parameters.values())
    required = [p.name for p in params
                if p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
    supplied = set(kwargs) | {p.name for p in params[:n_positional]}
    missing = [r for r in required if r not in supplied]
    assert not missing, (
        "mcp_router.py:%d calls artifacts.%s without %s — these have no default, so this is a "
        "TypeError the moment the tool is used. The REST route still works; only the MCP surface "
        "breaks, and only at call time." % (lineno, name, missing))
