"""`order_version` is a real optimistic-concurrency token, not a constant.

The state this replaces. `PATCH /artifacts/{id}/children/order` accepted an `order_version` and
read it **nowhere** — `.order_version` appeared in no expression in the tree — and answered with a
hardcoded `0`. `workspace_service.get_artifacts_order_version` existed, was called by nothing, and
also returned a literal `0`.

So two clients reordering the same container both succeeded, the second silently discarding the
first's arrangement, and a client that dutifully echoed the `0` back believed it was protected.
A field that advertises a guarantee it does not provide is worse than no field: it converts a
visible gap into an invisible one.

The version is DERIVED from the membership edges rather than stored, because those edges already
carry the ordering — a counter beside them would be a second source of truth to keep in step. It
covers the SEQUENCE of member roots, so a no-op reorder does not invalidate anyone's token, while
adding or removing a child does.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from mantle.db import open_lattice
from mantle.db.lattice_api import order_fingerprint


@pytest.fixture
def lattice():
    d = tempfile.mkdtemp()
    L = open_lattice(os.path.join(d, "l.db"), origin="test-origin", leaves=16)
    yield L
    L.artifacts.db.close()


def _member(L, container, child, order_key):
    L.graph.add_edges([(container, child, "contains", {"order_key": order_key})])


def test_the_fingerprint_is_stable_for_an_unchanged_order(lattice):
    """The property everything else rests on: reading twice without writing must agree, or every
    token would be stale the moment it was issued."""
    for i, key in enumerate(("a", "b", "c")):
        _member(lattice, "c1", "child-%d" % i, key)
    assert order_fingerprint(lattice, "c1") == order_fingerprint(lattice, "c1")


def test_reordering_changes_the_fingerprint(lattice):
    _member(lattice, "c1", "x", "a")
    _member(lattice, "c1", "y", "b")
    before = order_fingerprint(lattice, "c1")

    # swap: y now sorts before x
    lattice.graph.add_edges([("c1", "y", "contains", {"order_key": "A"})])
    after = order_fingerprint(lattice, "c1")
    assert before != after, "a reorder left the version unchanged — the token detects nothing"


def test_adding_a_child_changes_the_fingerprint(lattice):
    """A position echoed back across a membership change is stale, so this must move."""
    _member(lattice, "c1", "x", "a")
    before = order_fingerprint(lattice, "c1")
    _member(lattice, "c1", "z", "b")
    assert order_fingerprint(lattice, "c1") != before


def test_the_fingerprint_covers_the_sequence_not_the_key_spelling(lattice):
    """Two states with the same members in the same order are the SAME state, however the
    `order_key` strings were spelled. Without this, a no-op reorder would invalidate every
    outstanding token and clients would learn to ignore conflicts."""
    _member(lattice, "c1", "x", "a")
    _member(lattice, "c1", "y", "b")
    before = order_fingerprint(lattice, "c1")

    # respace the keys, preserving the order x < y
    lattice.graph.add_edges([("c1", "x", "contains", {"order_key": "m"}),
                             ("c1", "y", "contains", {"order_key": "n"})])
    assert order_fingerprint(lattice, "c1") == before, (
        "respacing the order keys changed the version even though the order did not")


def test_containers_do_not_share_a_version(lattice):
    _member(lattice, "c1", "x", "a")
    _member(lattice, "c2", "y", "a")
    assert order_fingerprint(lattice, "c1") != order_fingerprint(lattice, "c2")


def test_an_empty_container_has_a_version_and_it_is_not_an_error(lattice):
    """A container with no children is a legitimate state to hold a token on — the client may be
    about to add the first one."""
    assert isinstance(order_fingerprint(lattice, "empty"), int)


def test_the_version_is_json_safe(lattice):
    """It travels in a JSON body and is typed `int`. Above 2**53 a JavaScript client would round
    it, and two different orders could compare equal after rounding — a missed conflict."""
    for i in range(20):
        _member(lattice, "c1", "child-%d" % i, "key-%02d" % i)
    v = order_fingerprint(lattice, "c1")
    assert 0 <= v < 2 ** 53, v


def test_the_old_always_zero_helper_is_gone_or_unused():
    """`get_artifacts_order_version` returned a literal 0 and had no callers. If something starts
    calling it again, the guarantee quietly reverts to the constant this work replaced."""
    import ast
    import io
    import os as _os

    root = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
    callers = []
    for dirpath, _dirs, files in _os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = _os.path.join(dirpath, fn)
            try:
                tree = ast.parse(io.open(path, encoding="utf-8").read())
            except SyntaxError:                                  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    f = node.func
                    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                    if name == "get_artifacts_order_version":
                        callers.append("%s:%d" % (_os.path.basename(path), node.lineno))
    assert not callers, (
        "`get_artifacts_order_version` always returns 0 and is now called from %r — the ordering "
        "guarantee would revert to a constant. Use `order_fingerprint`." % callers)


# ── the route wiring ─────────────────────────────────────────────────────────────────────────

def _patched(ar, version):
    """`check_access` and the fingerprint pinned, so these tests are about the WIRING only."""
    from unittest.mock import patch
    # `_reorder_children` returns the number of edges it updated since 2026-08-26, and
    # the route refuses when that does not equal the ids asked for. A bare MagicMock returns a
    # MagicMock, which is never equal to 1 — so the count is stated here rather than defaulted.
    return (patch.object(ar, "check_access", return_value=None),
            patch.object(ar.store, "order_fingerprint", return_value=version),
            patch.object(ar, "_reorder_children", return_value=1))


@pytest.mark.asyncio
async def test_a_stale_order_version_is_refused_with_409(client):
    """The conflict a client can now actually detect."""
    from mantle.routers import artifacts_router as ar

    access, fp, svc = _patched(ar, 1234)
    with access, fp, svc as reorder:
        resp = await client.patch(
            "/artifacts/container-1/children/order",
            json={"ordered_ids": ["a-1"], "order_version": 9999},
        )
        assert resp.status_code == 409, resp.json()
        assert "changed" in resp.json()["detail"]
        reorder.assert_not_called()


@pytest.mark.asyncio
async def test_a_matching_order_version_is_accepted(client):
    """The inverted guard: a check that refused everything would satisfy the test above."""
    from mantle.routers import artifacts_router as ar

    access, fp, svc = _patched(ar, 1234)
    with access, fp, svc as reorder:
        resp = await client.patch(
            "/artifacts/container-1/children/order",
            json={"ordered_ids": ["a-1"], "order_version": 1234},
        )
        assert resp.status_code == 200, resp.json()
        assert resp.json()["order_version"] == 1234
        reorder.assert_called_once()


@pytest.mark.asyncio
async def test_omitting_the_version_stays_unconditional(client):
    """Every existing caller omits it and must keep working exactly as before. If this fails, the
    change stopped being backwards-compatible and became a required-field break."""
    from mantle.routers import artifacts_router as ar

    access, fp, svc = _patched(ar, 1234)
    with access, fp, svc as reorder:
        resp = await client.patch(
            "/artifacts/container-1/children/order", json={"ordered_ids": ["a-1"]},
        )
        assert resp.status_code == 200, resp.json()
        reorder.assert_called_once()
