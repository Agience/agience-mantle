"""No known-synchronous store helper is called directly from an async handler.

WHY IT MATTERS, IN `offload_sync`'s OWN WORDS. *"Nothing in the store is awaitable: SQLite and
the content stores are synchronous, so an `async def` handler that calls them directly holds the
event loop for the whole call and every other in-flight request waits behind it, including ones
that would have finished in microseconds. One slow read makes the process look single-threaded
because, for the duration, it is."*

MEASURED 2026-08-26. Two calls were on the loop: `_artifact_exists` in `_default_create_artifact`
(the CREATE path — every write into a collection paid it) and in `reorder_children`. The audit item
named only the first; a sweep for the same shape found the second.

**ZERO-ASSERTION, not a baseline.** The right number here is genuinely zero: a synchronous store
read on the event loop is a defect, not debt.

The synchronous set is DISCOVERED, not maintained: any name handed to `offload_sync` as its first
argument anywhere in the module is known-synchronous by that fact. So a new helper is covered the
moment it is offloaded once, and there is no list to fall behind.

Its honest limit: a helper that is NEVER offloaded anywhere cannot be recognised, so this catches
regressions in the established set rather than proving the whole module non-blocking.
"""
from __future__ import annotations

import ast
import inspect
import io

import pytest

from mantle.routers import artifacts_router as ar


def _module_ast():
    return ast.parse(io.open(inspect.getsourcefile(ar), encoding="utf-8").read())


def _known_synchronous(tree):
    """Every name this module itself hands to `offload_sync` — that is the evidence it is sync."""
    out = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "offload_sync" and n.args):
            a = n.args[0]
            if isinstance(a, ast.Name):
                out.add(a.id)
    return out


def _direct_calls_in_async(tree, sync_names):
    hits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.AsyncFunctionDef):
            continue
        for c in ast.walk(fn):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in sync_names:
                hits.append((fn.name, c.func.id, c.lineno))
    return hits


def test_the_synchronous_set_is_not_empty():
    """A vacuous sweep would pass forever. The module offloads dozens of helpers."""
    names = _known_synchronous(_module_ast())
    assert len(names) >= 10, "only %d offloaded helpers discovered: %r" % (len(names), sorted(names))


def test_no_known_synchronous_helper_is_called_from_an_async_handler():
    tree = _module_ast()
    hits = _direct_calls_in_async(tree, _known_synchronous(tree))
    assert not hits, (
        "these run a synchronous store call on the event loop, blocking every other in-flight "
        "request for its duration:\n" + "\n".join(
            "  %s() calls %s() at artifacts_router.py:%d" % h for h in sorted(hits)))


@pytest.mark.parametrize("name", ["_artifact_exists"])
def test_the_two_repaired_calls_stay_repaired(name):
    """Named explicitly so the regression that prompted this cannot come back unnoticed even if
    the discovery above is ever narrowed."""
    tree = _module_ast()
    hits = [h for h in _direct_calls_in_async(tree, {name})]
    assert not hits, hits
