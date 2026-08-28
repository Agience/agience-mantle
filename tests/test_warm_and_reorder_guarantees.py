"""Two rulings from the artifacts audit, pinned: warm's authorization (W1) and reorder's all-or-nothing (O3).

**W1 — a `read` grant used to buy an unbounded write sweep.** `warm` materializes every latent
member of a container, enqueueing an index job for each, and its gate was
`check_access(auth, artifact_id, "read", …)`. The size is the CONTAINER's, so the caller chose how
much work to cause by choosing which container to point at. A sweep that writes needs write rights.

**O3 — a reorder naming ids that do not resolve, or that are not members, dropped them
silently and answered 200.** Two drop points: `if a:` with no `else` in the resolution loop, and
`set_edge_order_key` returning False when there is no membership edge. The count needed to notice
was already computed by `reorder_collection_artifacts` and discarded one frame below the response.

The second got worse when `order_version` landed: the response would hand back a VALID TOKEN
certifying an arrangement nobody asked for. Optimistic concurrency made the silent drop harder to
notice, not easier — which is why "all or nothing" is the ruling rather than "report it".
"""
from __future__ import annotations

import ast
import io

import pytest
from unittest.mock import patch

from mantle.routers import artifacts_router as ar


# ── W1: the sweep is a write, so it needs write rights ───────────────────────────────────────

def _access_action(fn_name: str) -> str | None:
    """The action string this handler passes to `check_access`, read from the source.

    Read rather than exercised: what is under test is the ARGUMENT, and a call-through would prove
    only that some check ran.

    From the source file, not `inspect.getsource(getattr(ar, name))`. Other tests patch these
    handlers, and `getsource` on a `MagicMock` does not return the handler's code — so the live
    attribute version passed alone and failed in the full suite. A test asserting a property of the
    code reads the code."""
    tree = ast.parse(io.open(ar.__file__, encoding="utf-8").read())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn_name), None)
    assert fn is not None, "%s is gone from the router source" % fn_name
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        args = node.args
        if name == "offload_sync" and args and getattr(args[0], "id", "") == "check_access":
            args = args[1:]
        elif name != "check_access":
            continue
        if len(args) >= 3 and isinstance(args[2], ast.Constant):
            return args[2].value
    return None


def test_warm_requires_update_not_read():
    action = _access_action("warm_collection_endpoint")
    assert action == "update", (
        "warm authorizes on %r. It is a WRITE sweep over every latent member — gating it on `read` "
        "means a read grant buys writes, and an unbounded amount of them." % action)


def test_the_read_only_neighbours_still_authorize_on_read():
    """The inverted guard: if the change had been applied with too broad a brush, ordinary reads
    would now demand write rights and every reader would break."""
    for fn in ("read_artifact", "list_children"):
        assert _access_action(fn) == "read", "%s should still authorize on read" % fn


# ── O3: a reorder is one intent ──────────────────────────────────────────────────────────────

def _reorder_patches(applied: int, version: int = 1234):
    return (patch.object(ar, "check_access", return_value=None),
            patch.object(ar.store, "order_fingerprint", return_value=version),
            patch.object(ar, "_reorder_children", return_value=applied))


@pytest.mark.asyncio
async def test_a_partial_reorder_is_refused(client):
    """Three ids asked for, two applied — the third names something that does not exist or is not
    a member. Answering 200 would leave the container in an arrangement nobody chose."""
    access, fp, svc = _reorder_patches(applied=2)
    with access, fp, svc:
        resp = await client.patch(
            "/artifacts/container-1/children/order",
            json={"ordered_ids": ["a-1", "a-2", "typo"]},
        )
    assert resp.status_code == 400, resp.json()
    detail = resp.json()["detail"]
    assert "2 of 3" in detail, detail
    assert "not members" in detail or "do not exist" in detail, detail


@pytest.mark.asyncio
async def test_a_complete_reorder_still_succeeds(client):
    """The inverted guard. A refusal that fired on everything would satisfy the test above."""
    access, fp, svc = _reorder_patches(applied=3)
    with access, fp, svc:
        resp = await client.patch(
            "/artifacts/container-1/children/order",
            json={"ordered_ids": ["a-1", "a-2", "a-3"]},
        )
    assert resp.status_code == 200, resp.json()
    assert resp.json()["order_version"] == 1234


@pytest.mark.asyncio
async def test_the_refusal_does_not_hand_back_an_order_version(client):
    """The point of refusing rather than reporting. A token certifies the arrangement it names;
    issuing one for a partially-applied reorder would tell the caller their order is the state."""
    access, fp, svc = _reorder_patches(applied=1)
    with access, fp, svc:
        resp = await client.patch(
            "/artifacts/container-1/children/order",
            json={"ordered_ids": ["a-1", "nope"]},
        )
    assert resp.status_code == 400
    assert "order_version" not in resp.json(), resp.json()


def test_the_helper_returns_the_count_it_used_to_discard():
    """`_reorder_children` was typed `-> None` while the function it calls had always returned
    "edges updated". The information existed; it was thrown away one frame below the response.

    Read from the source file, not from the live attribute. The first version called
    `inspect.signature(ar._reorder_children)`, which passed alone and failed in the full suite:
    other tests patch that name, and a `MagicMock` has neither the annotation nor the body. A test
    asserting a property of the CODE must read the code — the live object is whatever the last
    fixture left there.
    """
    tree = ast.parse(io.open(ar.__file__, encoding="utf-8").read())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_reorder_children"), None)
    assert fn is not None, "`_reorder_children` is gone from the router source"
    assert isinstance(fn.returns, ast.Name) and fn.returns.id == "int", (
        "annotated %r — it must return the count of edges updated"
        % (ast.dump(fn.returns) if fn.returns else None))
    body = ast.get_source_segment(io.open(ar.__file__, encoding="utf-8").read(), fn) or ""
    assert "return store.reorder_collection_artifacts" in body, body[-300:]
