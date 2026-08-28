"""Every operation must say what it can fail with, and what it answers on success.

A baseline ratchet across every route group, not a cliff. `BASELINE` counts operations whose only
declared error is validation failure, and operations declaring no 4xx/5xx at all; `SUCCESS_BASELINE`
counts operations whose 2xx response carries no schema. Both may only shrink.

Scoped to the whole app on purpose. `tests/test_artifacts_error_contract_is_declared.py` filters on
the `Artifacts` tag by construction, so the same defect one route group over is invisible to it.

A ratchet rather than a flat assertion because declaring what an operation can raise is API
semantics rather than a lint fix, so the count going down is real work. This file makes it
impossible for the count to go up unnoticed. A tag absent from either map must measure clean, which
is what covers a new route group on the day it lands rather than on the day someone adds it here.

Any 2xx counts, not `200` alone: a created resource is exactly as undeclared as a fetched one, and a
filter narrower than the property undercounts silently. `_measure_success` reads every media type,
because `GET /artifacts/{artifact_id}/content` declares its `200` under `application/octet-stream`
and a JSON-only reader would count it as undeclared for ever.

Re-pinning a baseline upward is forbidden here. Pin a new one at what it measures on the day it
lands, never at a figure quoted from an earlier pass.
"""
from __future__ import annotations

import collections

import pytest

from mantle.main import app

#: Per tag, measured against `mantle.main.app`. Each entry is (operations advertising only 422,
#: operations advertising no 4xx/5xx at all).
BASELINE = {
    "Artifacts": (0, 0),
    #: `/my-access` raises nothing by design, so it declares nothing. Its emptiness is
    #: measured rather than pending, and it is the one operation left in this column.
    "Grants":    (1, 0),
    #: Every `/system` route is admin-gated, and most of its codes are not in the handlers at
    #: all: `get_auth` answers `401` before a handler runs and `require_platform_admin` raises
    #: `403`. A sweep that reads only handler bodies misses both, which are the commonest
    #: failures on the surface.
    "System":    (0, 0),
    "MCP":       (0, 0),
}

#: Per tag: operations whose 2xx response carries no schema at all. An operation can declare
#: every error it raises and still return `"schema": {}` for success, leaving a generated client
#: with no type for what it actually receives.
SUCCESS_BASELINE = {
    #: `ArtifactResponse` carries `extra="allow"`, so it publishes `additionalProperties: true`:
    #: the generated client is told the declared keys are always present and that more may arrive.
    #: That is what lets an open document be declared at all. Declaring it as a closed schema, or
    #: enumerating its keys into a response literal, truncates it — dropping `created_by` from
    #: every create, read and update response, plus `modified_by`, `content_ref`, and seventeen
    #: store-level fields on a task artifact. An undeclared shape is a documentation gap; a
    #: truncated one is data loss.
    #:
    #: Declared with `ok=` / `created=`, never `response_model=`, so nothing filters.
    "Artifacts": 0,
    #: The builder places each model under the route's own status code, so a `status_code=201`
    #: route cannot document a `200` it never sends.
    "Grants": 0,
    "System": 0,
    #: `POST /mcp` declares `JsonRpcResponse`; `GET /mcp` has no 2xx to declare, because `405` is
    #: its whole contract. `result` is left open in that model on purpose — every tool returns its
    #: own shape, and narrowing it would drop the payload the envelope exists to carry. That is
    #: also why it is declared with `responses=` rather than `response_model=`, which filters.
    "MCP": 0,
}


def _measure():
    spec = app.openapi()
    only_422 = collections.Counter()
    no_errors = collections.Counter()
    where = collections.defaultdict(list)
    for path, item in spec["paths"].items():
        for method, op in item.items():
            if not isinstance(op, dict) or "responses" not in op:
                continue
            tag = (op.get("tags") or ["(untagged)"])[0]
            errs = {c for c in op["responses"] if c.startswith(("4", "5"))}
            if not errs:
                no_errors[tag] += 1
                where[tag].append("%s %s — declares no 4xx/5xx at all" % (method.upper(), path))
            elif errs == {"422"}:
                only_422[tag] += 1
                where[tag].append("%s %s — 422 is its only declared error" % (method.upper(), path))
    return only_422, no_errors, where


def test_the_spec_is_readable_and_has_operations() -> None:
    """A derived measurement that quietly became empty would make every assertion below vacuous."""
    spec = app.openapi()
    ops = sum(1 for _, item in spec["paths"].items()
              for _, op in item.items() if isinstance(op, dict) and "responses" in op)
    assert ops >= 40, "only %d operations found in the spec — the sweep is not working" % ops


@pytest.mark.parametrize("tag", sorted(BASELINE), ids=sorted(BASELINE))
def test_a_route_group_never_declares_fewer_errors_than_before(tag: str) -> None:
    """The ratchet: this number may shrink and may never grow.

    A growth means a new operation shipped without saying what it can fail with, so a client
    generated from the spec has no branch for an error it will certainly receive."""
    only_422, no_errors, where = _measure()
    want_422, want_none = BASELINE[tag]
    got_422, got_none = only_422[tag], no_errors[tag]

    assert got_422 <= want_422 and got_none <= want_none, (
        "%s got WORSE: only-422 %d→%d, no-errors %d→%d.\n  %s\n"
        "  A new operation shipped without declaring its errors. Declare them; do not re-pin."
        % (tag, want_422, got_422, want_none, got_none, "\n  ".join(where[tag])))

    if (got_422, got_none) != (want_422, want_none):
        pytest.fail(
            "%s IMPROVED: only-422 %d→%d, no-errors %d→%d. Lower the baseline in this file to lock "
            "the gain in — that is the one edit this number is allowed to receive."
            % (tag, want_422, got_422, want_none, got_none))


def test_a_new_route_group_starts_clean() -> None:
    """The baseline is a map rather than a total, so a group absent from it must declare its
    errors from its first operation: a new router is covered on the day it lands rather than
    inheriting an allowance nobody chose to give it."""
    only_422, no_errors, where = _measure()
    unknown = sorted((set(only_422) | set(no_errors)) - set(BASELINE))
    assert not unknown, (
        "route group(s) %s carry undeclared errors and are not in the baseline. A new group starts "
        "clean: declare the errors rather than adding a row here.\n  %s"
        % (unknown, "\n  ".join(line for group in unknown for line in where[group])))


def test_the_artifacts_group_stays_closed() -> None:
    """The `Artifacts` group, pinned separately so it cannot regress unnoticed under a total."""
    only_422, no_errors, _ = _measure()
    assert only_422["Artifacts"] == 0 and no_errors["Artifacts"] == 0, (
        "the /artifacts group has regressed — P-2 took 20 operations to 0 and this is what keeps "
        "them there")


def _measure_success():
    """Per tag, how many operations answer 2xx with no schema, under any content type.

    Every media type counts, not `application/json` alone. `GET /artifacts/{artifact_id}/content`
    returns bytes: its `200` declares `{"type": "string", "format": "binary"}` under
    `application/octet-stream`, which is a correctly declared body and is not JSON. The property is
    "does the response declare what comes back", not "does it declare JSON" — a filter narrower
    than the property miscounts in whichever direction it is narrowed.

    Every 2xx is examined rather than the first, and the operation is counted once. Stopping at the
    first reads an operation declaring a described `200` and an undescribed `201` as clean, because
    `sorted` puts `200` first.
    """
    spec = app.openapi()
    empty = collections.Counter()
    where = collections.defaultdict(list)
    for path, item in spec["paths"].items():
        for method, op in item.items():
            if not isinstance(op, dict) or "responses" not in op:
                continue
            tag = (op.get("tags") or ["(untagged)"])[0]
            for code, r in sorted(op["responses"].items()):
                if not code.startswith("2"):
                    continue
                #: No `content` key at all means the response has no body, which is a declaration
                #: rather than an omission: `POST /mcp`’s `202` and `POST /artifacts/{id}/revert`’s
                #: `204` both say so deliberately, and counting them would demand a schema for a
                #: body that is empty by construction. An empty `schema: {}` under a content type
                #: is the defect this looks for: it promises a body and describes nothing.
                if "content" not in r:
                    continue
                content = r["content"] or {}
                #: ANY content type, not just JSON — a binary body is a declared body.
                declared = any((v or {}).get("schema") not in ({}, None)
                               for v in content.values())
                if not declared:
                    empty[tag] += 1
                    where[tag].append("%s %s — %s carries no schema" % (method.upper(), path, code))
                    break
    return empty, where


@pytest.mark.parametrize("tag", sorted(SUCCESS_BASELINE), ids=sorted(SUCCESS_BASELINE))
def test_a_route_group_never_leaves_more_success_responses_undeclared(tag: str) -> None:
    """An undeclared success is as unusable as an undeclared failure: a client generated from the
    spec has no type for what it receives, and no way to know which fields are guaranteed."""
    empty, where = _measure_success()
    want, got = SUCCESS_BASELINE[tag], empty[tag]
    assert got <= want, (
        "%s got WORSE: %d->%d operations answer 2xx with no schema: %s"
        % (tag, want, got, " | ".join(where[tag])))
    if got != want:
        pytest.fail(
            "%s IMPROVED: %d→%d. Lower SUCCESS_BASELINE to lock the gain in." % (tag, want, got))
