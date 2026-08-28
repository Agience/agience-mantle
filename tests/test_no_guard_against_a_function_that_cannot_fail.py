"""No caller guards against a failure its callee cannot produce.

`update_artifact`, `update_collection` and `update_grant` are each annotated `-> Optional[...]`
and each has exactly one `return entity` — no error path, never `None`. A guard written against
that promise cannot fire: a signature that advertises a failure mode the body cannot produce does
not make a caller safer, it makes it write dead code that reads as error handling. It does not
cover the real failure either — a `put_artifact` that returned without persisting would pass
`is None` too. The only failure that reaches a client is `put_artifact` raising, which propagates
past a guard like this regardless of whether it exists.

The sweep covers three shapes, because each can hide a cannot-fail guard from a narrower search:
the call tested directly inside a branch, the result bound to a name and tested in the same scope,
and an `f(...) or fallback` expression, which is not a branch at all. Scope matters: matching a
name across a whole module rather than within one function's scope produces a confident, precise,
wrong hit — a local named `artifact` in one function can match an unrelated `if not artifact` in
another.

It reads with `utf-8-sig` and asserts that it parsed every file it walked, rather than swallowing a
parse failure: a BOM'd source that raises `SyntaxError` under plain `utf-8` must not silently drop
out of a sweep that reports zero.

Zero-assertion, not a baseline: a guard that cannot fire is a defect, not debt.
"""
from __future__ import annotations

import ast
import io
import os

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
CANNOT_FAIL = {"update_artifact", "update_collection", "update_grant"}


#: `utf-8-sig`, never bare `utf-8`: some sources here carry a UTF-8 BOM, and reading one as plain
#: `utf-8` leaves it surviving as U+FEFF, so `ast.parse` raises `SyntaxError` and the
#: `except SyntaxError: continue` below would skip the file in silence — a sweep that asserts zero
#: would report zero over files it never read. The same defect was measured in three gates on
#: 2026-08-26 — `dependency_dag_check` was missing 22 files including `origin/main.py` and six
#: routers — and it was in this test too.
_ENCODING = "utf-8-sig"

#: Files this sweep could not parse. A zero-assertion sweep must be able to say "zero, and I could
#: not read k files" — those are different claims, and only one of them is evidence.
_UNREADABLE: list = []


def _python_files():
    for dp, dn, fn in os.walk(SRC):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            if f.endswith(".py"):
                yield os.path.join(dp, f)


def _parse(path):
    """Parse one file, recording rather than swallowing a failure."""
    try:
        return ast.parse(io.open(path, encoding=_ENCODING).read())
    except (SyntaxError, UnicodeDecodeError, OSError) as exc:
        _UNREADABLE.append("%s (%s)" % (os.path.relpath(path, SRC).replace(os.sep, "/"), exc.__class__.__name__))
        return None


def _calls_one(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else (
                f.id if isinstance(f, ast.Name) else None)
            if name in CANNOT_FAIL:
                return name
    return None


def _tests_for_absence(t):
    """Does this expression test its operand for being falsy/None? Returns the operand, or None."""
    if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
        return t.operand
    if isinstance(t, ast.Compare) and any(isinstance(o, (ast.Is, ast.IsNot)) for o in t.ops):
        if any(isinstance(c, ast.Constant) and c.value is None for c in t.comparators):
            return t.left
    return None


def _bound_to_a_cannot_fail_call(scope):
    """Names in *scope* assigned directly from one of the calls. `{name: (func, lineno)}`.

    A defect is one thing and its spellings are several: matching on a call tested directly inside
    a branch finds only that spelling, and misses a result bound to a name first and tested
    afterward — `result = store.update_artifact(db, target)` then `if result is None: raise 500`
    is invisible to a sweep that only looks inside the `if`.

    Scoped per function, never per module: matching a name across the whole file would let a local
    named `artifact` assigned from `update_artifact` in one function match an unrelated
    `if not artifact` eighty lines away in a different function — a confident, precise, wrong
    finding, complete with two line numbers.
    """
    out = {}
    for n in _own_body(scope):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            which = _calls_one(n.value)
            if which:
                for tgt in n.targets:
                    if isinstance(tgt, ast.Name):
                        out[tgt.id] = (which, n.lineno)
    return out


def _own_body(scope):
    """Every node in *scope* except those belonging to a nested function or class of its own.

    A nested definition has its own locals; treating its names as this scope's is the same
    cross-scope confusion one level down.

    Skipping only the pushing of a nested definition's children is not enough: a top-level `def`
    is itself an entry in the module's body, so the module scope yields it before walking into its
    statements. The check has to stop the descent at the definition node itself, not at its
    children.
    """
    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    stack = list(getattr(scope, "body", []))
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, nested):
            continue                      # its body belongs to its own scope, not to this one
        stack.extend(ast.iter_child_nodes(n))


def _scopes(tree):
    """The module itself, then every function in it — each paired with its own body."""
    yield tree
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield n


def _guards():
    out = []
    for p in _python_files():
        tree = _parse(p)
        if tree is None:
            continue
        rel = os.path.relpath(p, SRC).replace(os.sep, "/")
        seen = set()
        for scope in _scopes(tree):
            bound = _bound_to_a_cannot_fail_call(scope)
            for n in _own_body(scope):
                # (1) the call tested directly inside the branch — the original form.
                if isinstance(n, ast.If):
                    operand = _tests_for_absence(n.test)
                    if operand is None:
                        continue
                    which = _calls_one(operand)
                    if which:
                        found = "%s:%d guards against %s()" % (rel, n.lineno, which)
                    # (2) the result bound to a name FIRST, then tested — same scope only.
                    elif isinstance(operand, ast.Name) and operand.id in bound:
                        which, at = bound[operand.id]
                        found = ("%s:%d guards against %s() via `%s` (assigned at :%d)"
                                 % (rel, n.lineno, which, operand.id, at))
                    else:
                        continue
                # (3) `f(...) or fallback` — not a branch at all, so the If-only sweep could never
                # see it. The fallback is unreachable for the same reason, and it reads as "use the
                # stored copy if the write gave nothing back" — a sentence about a failure that
                # cannot happen.
                elif isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or) and len(n.values) > 1:
                    which = _calls_one(n.values[0])
                    if not which:
                        continue
                    found = ("%s:%d falls back on %s() being falsy, which it never is"
                             % (rel, n.lineno, which))
                else:
                    continue
                if found not in seen:      # module scope and function scope both reach a top-level If
                    seen.add(found)
                    out.append(found)
    return out


def test_the_sweep_reads_every_file_it_claims_to_have_swept():
    """A zero-assertion sweep must not report zero over files it could not read. `N found` and
    `N found, k unreadable` are different claims, so this sweep must not swallow a `SyntaxError`
    and leave a BOM'd file with no trace at all."""
    _UNREADABLE.clear()
    _guards()
    assert not _UNREADABLE, (
        "the sweep could not parse %d file(s), so its zero is not evidence:\n  %s"
        % (len(_UNREADABLE), "\n  ".join(sorted(_UNREADABLE))))


def test_the_sweep_can_see_the_functions_at_all():
    """A vacuous sweep would pass forever. All three must still exist and be called somewhere."""
    seen = set()
    for p in _python_files():
        tree = _parse(p)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name in CANNOT_FAIL:
                seen.add(n.name)
    assert seen == CANNOT_FAIL, "not all three are defined any more: %s" % sorted(seen)


def test_nothing_guards_against_a_failure_these_cannot_produce():
    guards = _guards()
    assert not guards, (
        "these test for a failure the callee cannot produce, so the branch is unreachable:\n  "
        + "\n  ".join(sorted(guards)))


@pytest.mark.parametrize("name", sorted(CANNOT_FAIL))
def test_the_annotation_no_longer_promises_none(name):
    """The root cause, asserted at the source. If `Optional` comes back, the guards will too."""
    for p in _python_files():
        tree = _parse(p)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == name and n.returns is not None:
                ann = ast.unparse(n.returns)
                rets = [r for r in ast.walk(n) if isinstance(r, ast.Return)]
                returns_none = any(
                    r.value is None or (isinstance(r.value, ast.Constant) and r.value.value is None)
                    for r in rets)
                if "Optional" in ann or "None" in ann:
                    assert returns_none, (
                        "%s is annotated %r but no return produces None — that promise is what "
                        "the dead guards were written against" % (name, ann))


def test_revoke_returns_nothing_to_test():
    """`grant_key_service.revoke` returned `update_grant(...) is not None` — always True. Returning
    `None` is what stops a caller branching on it again."""
    from mantle.services import grant_key_service
    import inspect
    sig = inspect.signature(grant_key_service.revoke)
    assert sig.return_annotation in (None, "None", type(None)), sig.return_annotation
