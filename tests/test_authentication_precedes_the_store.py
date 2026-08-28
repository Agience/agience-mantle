"""P-8: a handler must answer 401 before it touches the store.

Measured 2026-08-25 across `artifacts_router.py`: eleven handlers raise an explicit 401 and also
read the store, and **one of them did it in the wrong order**. `list_commits` ran
`check_access` → `_artifact_exists` → `401`, so an unauthenticated caller received:

    an id that does not exist  ->  400 "Commits only available for collections"
    an id that does exist      ->  401 "User identification required"

The status code answered *"does this artifact exist?"* for someone who had not identified
themselves — an existence oracle assembled out of the difference between two error codes. The
bytes were never at risk; the fact of existence was.

This reads the real router with `ast` rather than exercising each route, because the property is
about ORDER IN THE SOURCE and a route test would only catch the handler it happened to call. A new
handler that authenticates late fails here on the day it lands.
"""
from __future__ import annotations

import ast
import io
import pathlib

import pytest

_ROUTER = pathlib.Path(__file__).resolve().parents[1] / "src" / "mantle" / "routers" / "artifacts_router.py"
_SRC = io.open(_ROUTER, encoding="utf-8").read()
_TREE = ast.parse(_SRC)

#: A call whose name says it reaches the store. `offload_sync` is included because that is how this
#: router runs blocking store work off the event loop — the wrapper, not the callee, is what appears
#: at the call site.
_STORE_MARKERS = ("store_db", "get_artifact", "_artifact_exists", "offload_sync")


def _routes():
    for node in ast.walk(_TREE):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(isinstance(d, ast.Call) and getattr(d.func, "attr", "") in
               ("get", "post", "patch", "delete", "put") for d in node.decorator_list):
            yield node


def _first_401(fn):
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Raise):
            seg = ast.get_source_segment(_SRC, sub) or ""
            if "401" in seg:
                return sub.lineno
    return None


def _first_store_touch(fn):
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.Call):
            continue
        seg = ast.get_source_segment(_SRC, sub.func) or ""
        if any(m in seg for m in _STORE_MARKERS):
            return sub.lineno
    return None


def test_the_router_has_routes_to_check():
    """Guards the parametrised test below: an empty walk would make it vacuously green."""
    assert len(list(_routes())) >= 15, "expected the artifacts routes, found %d" % len(list(_routes()))


@pytest.mark.parametrize("fn", list(_routes()), ids=lambda f: f.name)
def test_a_handler_answers_401_before_it_reads_the_store(fn):
    """The property, held for every route rather than for the one that was found broken."""
    at_401 = _first_401(fn)
    if at_401 is None:
        pytest.skip("no explicit 401 in this handler")
    at_store = _first_store_touch(fn)
    if at_store is None:
        pytest.skip("this handler does not reach the store")
    assert at_401 < at_store, (
        "%s reads the store at line %d but does not raise 401 until line %d. An unauthenticated "
        "caller can then tell an existing id from a missing one by which error comes back — the "
        "existence oracle P-8 closes. Move the auth check to the top of the handler."
        % (fn.name, at_store, at_401))
