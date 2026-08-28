"""Every mantle-bound URL a sibling repo builds must reach a route this node serves.

`test_a_removed_route_has_no_in_house_caller` checks the four prefixes the removal test names. On
2026-08-26 a sweep found nine dead call sites across five repos, spanning ten planes — only
three of which that guard covered:

    /artifacts/search   astra deduplicate · sage retrieval        (swallowed -> always "unique" / no grounding)
    /search/query       sage server · prism SDK search_query      (SDK: every new server copied it)
    /collections/...    lumen ×2                                  (one swallowed, one raised)
    /api-keys           seraph rotate_api_key                     (reported a safety refusal, not a 404)
    /servers/register   crystal persona registration              (log.debug "will retry", on every boot)
    /secrets/reveal     ophan                                     (fails closed and loud — still open)

It found a tenth on the day it landed, which the narrower guard could not have: all three prism
SDKs posted `{MANTLE_URI}/hosts/register` while the receiver had been on the ember leaf since
2026-07-21 — mantle mounts 66 routes and none of them mentions `hosts`. The gap had been closed on
the wrong side of the wire, and every prism host had announced nothing on every start since.
Registration now resolves `EMBER_URI` (Prism Protocol §4 gained a sixth canonical name for it), so
this file asserts zero again rather than carrying a baseline.

So this one does not enumerate what was removed. It asks the only question that stays true as
the surface changes: does the path this call builds match a route mantle serves today? A plane
retired tomorrow is covered without anyone editing a list.

Scoped to mantle-bound calls: a sibling reaching its own service, or origin, or crystal, is not
this test's business — `ember` serves `/hosts/register` itself, and flagging it would train readers
to ignore the failure. The scope comes from the base the URL is built on, which is why a sweep
that keys on path prefixes alone reports clean while six of the nine are live.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

from mantle.main import app

_WORKSPACE = pathlib.Path(__file__).resolve().parents[2]
_SKIP = {".git", "node_modules", "_ci-work", "dist", "build", "__pycache__", ".venv"}
SIBLINGS = ("agience-chorus", "agience-crystal", "agience-ember", "agience-observe",
            "agience-prism/py")

#: Names that hold a mantle base URL at the call site. Derived by reading the real call sites
#: rather than guessed: every dead call found on 2026-08-26 was built on one of these.
#:
#: `api_uri` came off this list: prism's `Host.api_uri` resolved `MANTLE_URI` — precisely the bug
#: this test found. Now that registration resolves `EMBER_URI`, the same attribute names the leaf,
#: and leaving it here would report every correct registration call as dead. A base-name list is
#: only as true as the code it was read from; it is re-derivable, not permanent.
_MANTLE_BASES = ("MANTLE_URI", "mantle_url", "mantle_uri", "MANTLE_BASE", "_MANTLE")



def _live_matchers():
    """One regex per mounted route, with `{param}` standing for a single segment."""
    return [re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", p) + "/?$")
            for p in {getattr(r, "path", "") for r in app.routes} if p]


def _mantle_paths(tree: ast.AST):
    """Path fragments concatenated onto a mantle base inside an f-string.

    f-strings only, and that is the point: `f"{MANTLE_URI}/artifacts/recall"` keeps the base and
    the path in separate nodes, so the base says who is being called and the fragment says what.
    A bare literal carries no base and could belong to any service.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        parts = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                parts.append(("expr", ast.unparse(value.value)))
            elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(("lit", value.value))
        for i, (kind, text) in enumerate(parts):
            if kind != "expr" or not any(b in text for b in _MANTLE_BASES):
                continue
            nxt = parts[i + 1] if i + 1 < len(parts) else None
            if nxt and nxt[0] == "lit" and nxt[1].startswith("/"):
                yield nxt[1].split("?")[0], node.lineno


def _sources():
    for name in SIBLINGS:
        root = _WORKSPACE / name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if any(part in _SKIP for part in path.parts):
                continue
            if path.name.startswith("test_") or "tests" in path.parts:
                continue
            yield path


@pytest.mark.skipif(not (_WORKSPACE / "agience-chorus").is_dir(),
                    reason="siblings not checked out beside mantle")
def test_no_sibling_builds_a_mantle_url_that_reaches_nothing():
    matchers = _live_matchers()
    offenders = []
    for path in _sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for fragment, lineno in _mantle_paths(tree):
            probe = re.sub(r"\{[^}]*\}", "zzz", fragment).rstrip("/") or "/"
            if not any(m.match(probe) for m in matchers):
                # keyed WITHOUT the line number: a known-open item must not come back as new
                # because an unrelated edit moved it down the file.
                offenders.append("%s -> %s"
                                 % (path.relative_to(_WORKSPACE).as_posix(), fragment))
    unexpected = sorted(set(offenders))
    assert not unexpected, (
        "mantle-bound calls that reach no route this node serves:" + chr(10) + "  "
        + (chr(10) + "  ").join(unexpected))


@pytest.mark.skipif(not (_WORKSPACE / "agience-chorus").is_dir(),
                    reason="siblings not checked out beside mantle")
def test_the_scan_finds_real_mantle_calls():
    """The control, and it is the one that matters here. A base-name typo or an AST shape this
    misses would make the test above pass while reading nothing — the exact failure of the first
    sweep, which reported clean because its filter excluded the dead case by construction."""
    found = 0
    for path in _sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        found += sum(1 for _ in _mantle_paths(tree))
    assert found >= 10, (
        "only %d mantle-bound calls seen across the siblings — the scan is not reaching them, and "
        "a guard that reads nothing passes for ever" % found)
