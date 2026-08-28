"""A 5xx body must not carry the text of an exception the server did not expect.

P-9 in the artifacts-API audit, and a security item rather than a tidiness one:
"an unhandled exception's text is internal detail, and it is being returned to the caller."

Measured 2026-08-26 across all 16 `5xx` sites in `mantle/routers/`: the three the audit named
by hand — `f"Recall failed: {str(e)}"` (×2) and `f"Upload initiation failed: {exc}"` — are already
curated, log with `exc_info=True`, and one says so outright: "The server log carries the underlying
error." The audit's count was stale.

One survivor it had not named: `system_router.create_issuer` caught `RuntimeError` broadly and
returned `str(exc)` as a 503. That was right for the one raise it was written for — a curated
"system principal unavailable" — and wrong for every other `RuntimeError` the call stack can produce:
a library's, sqlite's, anything. A broad handler over a narrow intent turns any internal error's
text into a response body, which is the whole of P-9 and is invisible until the day something else
raises.

The fix needed no new machinery. `SystemPrincipalUnavailable` already existed for exactly that
condition, and its docstring names the call site: "a fabricated system identity would acquire
whatever grants happened to match it, which is the failure mode `issuers.py` already fails closed
against." `issuers.py` raising a bare `RuntimeError` was the drift.

4xx is deliberately not covered. A `ValueError` raised on purpose carries a curated message naming
what the caller must change — "an external issuer must bind an 'audience'" — and returning that text
is the point of a 400. Forbidding it there would delete the useful half of the pattern to reach the
same green.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_ROUTERS = pathlib.Path(__file__).resolve().parents[1] / "src" / "mantle" / "routers"


def _files():
    return sorted(p for p in _ROUTERS.glob("*.py") if p.name != "__init__.py")


def _status_of(node: ast.Call):
    kw = {k.arg: k.value for k in node.keywords}
    code = kw.get("status_code")
    if isinstance(code, ast.Constant) and isinstance(code.value, int):
        return code.value, kw.get("detail")
    if isinstance(code, ast.Attribute) and code.attr.startswith("HTTP_"):
        digits = "".join(c for c in code.attr if c.isdigit())[:3]
        return (int(digits) if digits else None), kw.get("detail")
    return None, kw.get("detail")


#: Handlers whose exception could be anything the call stack raises. A body built from one of these
#: is returning text nobody authored for a reader.
_BROAD = {"Exception", "BaseException", "RuntimeError", "OSError", "ValueError", None}


def _caught_type(handler):
    """The single type name an `except` binds, or None for a bare/tuple/broad catch."""
    if handler is None or handler.type is None:
        return None
    if isinstance(handler.type, ast.Name):
        return handler.type.id
    if isinstance(handler.type, ast.Attribute):
        return handler.type.attr
    return None                      # a tuple of types — treat as broad


def _exception_nodes(tree: ast.AST):
    """Map each node to the nearest enclosing ExceptHandler."""
    owner = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            for sub in ast.walk(node):
                owner.setdefault(id(sub), node)
    return owner


def _exception_names(tree: ast.AST):
    """Map each node to the name bound by the nearest enclosing `except ... as NAME`."""
    owner = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.name:
            for sub in ast.walk(node):
                owner.setdefault(id(sub), node.name)
    return owner


def test_there_are_5xx_sites_to_check() -> None:
    """A derived set that quietly became empty would make the assertion below vacuous."""
    n = 0
    for f in _files():
        tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "HTTPException":
                code, _ = _status_of(node)
                if code and code >= 500:
                    n += 1
    assert n >= 10, "only %d 5xx sites found across the routers — the sweep is not working" % n


@pytest.mark.parametrize("path", _files(), ids=[f.name for f in _files()])
def test_no_5xx_body_interpolates_a_caught_exception(path: pathlib.Path) -> None:
    """The property: a 5xx says the server failed; why belongs in the log, not the response."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    owner = _exception_names(tree)
    owner_node = _exception_nodes(tree)
    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "HTTPException"):
            continue
        code, detail = _status_of(node)
        if not (code and code >= 500) or detail is None:
            continue
        exc = owner.get(id(node))
        if not exc:
            continue
        # A narrow, purpose-built exception is allowed to speak, and that is the same reasoning
        # that keeps 4xx out of this check entirely. `SystemPrincipalUnavailable` carries a message
        # authored for the caller; `RuntimeError` carries whatever happened to go wrong. The defect
        # is a broad handler over a narrow intent, not the act of returning a message.
        if _caught_type(owner_node.get(id(node))) not in _BROAD:
            continue
        names = {n.id for n in ast.walk(detail) if isinstance(n, ast.Name)}
        if exc in names:
            bad.append("%s:%d  detail=%s  (from `except ... as %s`)"
                       % (path.name, node.lineno, ast.unparse(detail)[:60], exc))
    assert not bad, (
        "a 5xx response returns the text of a caught exception — internal detail reaching the "
        "caller:\n  %s\n"
        "  Log it with `exc_info=True` and return a curated message. If the exception is one you "
        "raise deliberately with a message for the caller, catch THAT type specifically rather than "
        "a broad one." % "\n  ".join(bad))


def test_the_issuer_handler_catches_the_specific_type() -> None:
    """The regression test for the one site this file was written for.

    Asserted on the handler rather than the outcome: a broad `except RuntimeError` here would
    still pass every functional test, because the only RuntimeError raised TODAY is the curated one.
    The defect is what happens when something else raises, which no test can provoke."""
    src = (_ROUTERS / "system_router.py").read_text(encoding="utf-8", errors="replace")
    # Comments stripped first: the note explaining the narrowing itself says `except RuntimeError`,
    # so asserting against the raw source would match that comment rather than real code — the
    # same trap as prose about a removal reintroducing the forbidden string.
    code = chr(10).join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    i = code.index("async def create_issuer")
    body = code[i:i + 3000]
    assert "except SystemPrincipalUnavailable" in body, (
        "create_issuer no longer catches the specific exception; a broad handler returns any "
        "internal RuntimeError's text as a 503")
    assert "except RuntimeError" not in body, (
        "the broad `except RuntimeError` is back — it returns internal error text for every "
        "RuntimeError the call stack can raise")
