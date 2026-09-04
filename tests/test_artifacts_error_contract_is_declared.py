"""Every `/artifacts` operation declares the errors it can actually return.

An operation declaring only `422` while its handler raises 400, 401, 403, 404, 409, 413, 500 and
503 does not have an incomplete contract, it has a wrong one: a generated client gets a case for
validation failure and for nothing else, including the 401
it will meet on its first bad token and the 404 that is this API's way of saying "denied".

`responses=` is declaration only — it changes no runtime behaviour. These tests therefore assert on
the generated schema, which is the thing that was wrong.
"""
from __future__ import annotations

import pytest

from mantle.main import app
from mantle.routers import artifacts_router

def _executable(src: str) -> str:
    """`src` with comment lines removed.

    A check that forbids a literal, run over raw text, fires on the comment explaining why that
    literal was removed — and the rationale is then deleted to reach green.
    `agience-cloud/tests/test_status_json_and_table_agree.py` strips comments for the same reason.

    Comment lines only. A docstring is code and stays: if a forbidden literal belongs in one, say
    it without quoting it."""
    return chr(10).join(ln for ln in src.splitlines()
                        if not ln.lstrip().startswith("#"))





@pytest.fixture(scope="module")
def artifact_ops():
    spec = app.openapi()
    ops = {(path, method): op
           for path, item in spec["paths"].items()
           for method, op in item.items()
           if isinstance(op, dict) and "Artifacts" in (op.get("tags") or [])}
    # Pinned, so a new operation cannot arrive without being covered by the checks below.
    assert len(ops) == 19, (
        "expected 19 artifact operations, found %d — this suite's subject moved" % len(ops))
    return ops


def test_no_operation_declares_422_as_its_only_error(artifact_ops):
    """The headline defect, stated as the assertion that would have caught it."""
    only_422 = []
    for (path, method), op in artifact_ops.items():
        errors = {c for c in op.get("responses", {}) if c.startswith(("4", "5"))}
        if errors == {"422"}:
            only_422.append("%s %s" % (method.upper(), path))
    assert not only_422, (
        "these operations advertise validation failure as their only error:\n  "
        + "\n  ".join(sorted(only_422)))


def test_every_operation_declares_401(artifact_ops):
    """All 19 authenticate, so all 19 can refuse a credential that is present and invalid."""
    missing = ["%s %s" % (m.upper(), p) for (p, m), op in artifact_ops.items()
               if "401" not in op.get("responses", {})]
    assert not missing, missing


def test_the_404_description_states_that_denial_and_absence_are_the_same(artifact_ops):
    """The single most useful sentence in this contract, and the one a client cannot guess.

    `check_access` answers a denied read with 404, identically to a missing artifact, so that the
    endpoint cannot be used to discover what exists. A client that reads 404 as "gone" will delete
    its local copy of something it merely lost access to."""
    checked = 0
    for (_p, _m), op in artifact_ops.items():
        desc = op.get("responses", {}).get("404", {}).get("description", "")
        if not desc:
            continue
        checked += 1
        assert "not permitted" in desc, desc
        assert "indistinguishable" in desc, desc
    assert checked >= 15, "only %d operations declared a 404 — expected most of the 19" % checked


def test_declared_codes_are_ones_the_router_can_actually_raise(artifact_ops):
    """No operation may advertise a status this router never produces.

    A contract that over-declares is wrong in the other direction: it makes a client write handling
    for a response that cannot arrive, and it hides which codes are real."""
    raisable = {"400", "401", "403", "404", "409", "413", "422", "500", "503"}
    for (path, method), op in artifact_ops.items():
        declared = {c for c in op.get("responses", {}) if c.startswith(("4", "5"))}
        extra = declared - raisable
        assert not extra, "%s %s declares %s, which nothing raises" % (method.upper(), path, extra)


def test_check_access_s_unreachable_400_is_NOT_declared():
    """The one code deliberately left out, pinned so a future closure cannot put it back.

    `check_access` raises `400 Unknown action`, but every call site in this router passes a string
    literal, so no client can provoke it. If someone starts passing a caller-supplied action, this
    test should fail and the 400 should be declared on that route — the assertion is the reminder.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(artifacts_router))
    non_literal = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        args = node.args
        if name == "offload_sync" and args and getattr(args[0], "id", "") == "check_access":
            args = args[1:]
        elif name != "check_access":
            continue
        # signature: (auth, resource_id, action, db) — the action is the third
        if len(args) >= 3 and not isinstance(args[2], ast.Constant):
            non_literal.append(getattr(node, "lineno", 0))
    assert not non_literal, (
        "check_access is called with a non-literal action at line(s) %r. Its `400 Unknown action` "
        "is now reachable by a client on that route, and must be declared there." % non_literal)


def test_the_error_table_covers_every_code_the_helper_is_asked_for():
    """`_errors` would raise KeyError at import time for an undeclared code — this states why
    that table and the routes cannot drift apart."""
    for code in (400, 401, 403, 404, 409, 413, 500, 503):
        assert code in artifacts_router._ERROR_DESCRIPTIONS
        assert artifacts_router._errors(code)[code]["description"]


def test_every_path_parameter_is_described(artifact_ops):
    """Every path parameter carries a description.

    It matters more here than on a typical API. Every path template spells its first parameter
    `{artifact_id}` and never `{container_id}`, because a container is an artifact and two
    templates make a code generator emit two unrelated resources. The name therefore carries no
    hint about which role the id plays, and the roles differ: on `warm`, `children`,
    `children/order` and `commits` the id names the container and the operation acts on what it
    holds rather than on the artifact named in the path.
    """
    undescribed = []
    for (path, method), op in artifact_ops.items():
        for prm in op.get("parameters", []):
            if prm.get("in") == "path" and not prm.get("description"):
                undescribed.append("%s %s -> %s" % (method.upper(), path, prm.get("name")))
    joined = (chr(10) + "  ").join(sorted(undescribed))
    assert not undescribed, (
        "path parameters a client can only guess at:" + chr(10) + "  " + joined)


def test_the_container_routes_say_the_id_is_a_container(artifact_ops):
    """The distinction the description exists to carry. If these ever read like the plain-artifact
    text, the name and the prose agree and both are wrong.

    Every two-id operation reads container-first, so there is no exception to carve out here:
    detaching a member is `DELETE /artifacts/{artifact_id}/children/{child_id}`, where the first
    segment is the container and the second is the member."""
    want = ("/warm", "/children", "/commits")
    seen = 0
    for (path, _m), op in artifact_ops.items():
        if not path.endswith(want):
            continue
        for prm in op.get("parameters", []):
            if prm.get("in") != "path":
                continue
            seen += 1
            assert "CONTAINER" in prm.get("description", ""), (
                "%s does not say its path id names a container" % path)
    assert seen >= 3, "only %d container-route path params found — the scan missed some" % seen


def test_remove_names_both_of_its_ids_for_what_they_are(artifact_ops):
    """`DELETE /artifacts/{artifact_id}/children/{child_id}` has two path ids, and each says
    which role it plays.

    The two must not read alike. `evict` is checked against the container and the member
    survives, so a reader who has them the wrong way round expects the opposite artifact to be
    detached."""
    op = artifact_ops[("/artifacts/{artifact_id}/children/{child_id}", "delete")]
    byname = {p["name"]: (p.get("description") or "")
              for p in op.get("parameters", []) if p.get("in") == "path"}
    assert set(byname) == {"artifact_id", "child_id"}, sorted(byname)
    assert "CONTAINER" in byname["artifact_id"], byname["artifact_id"]
    assert "member" in byname["child_id"], byname["child_id"]


def test_remove_takes_no_request_body(artifact_ops):
    """The half that made the verb change safe.

    `DELETE` with a request body has no defined semantics in HTTP and intermediaries are permitted
    to drop it. Switching the verb while `container_id` stayed in the body would have put the field
    naming what to detach FROM on a carrier that may not arrive."""
    op = artifact_ops[("/artifacts/{artifact_id}/children/{child_id}", "delete")]
    assert "requestBody" not in op, (
        "the bodyless DELETE grew a request body; that is the shape M4 exists to prevent")


def test_every_content_type_filter_says_when_it_applies(artifact_ops):
    """Every `content_type` filter states when it runs and what to page by.

    Both routes carrying one filter before taking the page, so `total` is the filtered count.
    This asserts the durable half of that contract — the caller is told when the filter runs,
    and to continue from `has_more` rather than from `len(items)` — rather than one particular
    sentence, because a test that pins prose to a behaviour the code has left then defends the
    stale wording against correction."""
    missing = []
    for (path, method), op in artifact_ops.items():
        for prm in op.get("parameters", []):
            if prm.get("name") != "content_type":
                continue
            d = (prm.get("description") or "").lower()
            if "before the page" not in d or "has_more" not in d:
                missing.append("%s %s" % (method.upper(), path))
    assert not missing, (
        "a content_type filter that does not say when it applies and what to page by: "
        + ", ".join(missing))


def test_every_query_parameter_is_described(artifact_ops):
    """The companion to the path-parameter check: every query parameter carries a description.

    A generated client shows a parameter's own description and not the endpoint docstring, so a
    default explained only in the docstring reaches nobody choosing whether to send it —
    `cascade` on `DELETE /artifacts/{artifact_id}` being the expensive case."""
    undescribed = []
    for (path, method), op in artifact_ops.items():
        for prm in op.get("parameters", []):
            if prm.get("in") == "query" and not prm.get("description"):
                undescribed.append("%s %s -> %s" % (method.upper(), path, prm.get("name")))
    joined = (chr(10) + "  ").join(sorted(undescribed))
    assert not undescribed, (
        "query parameters a client can only guess at:" + chr(10) + "  " + joined)


def test_revert_declares_its_designed_no_op(artifact_ops):
    """. `204` was the one undeclared code that was definitely INTENTIONAL — described in
    the handler docstring and specified in the design doc, and absent from the spec. A generated
    client therefore had no branch for the designed no-op and would treat it as unexpected."""
    op = artifact_ops[("/artifacts/{artifact_id}/revert", "post")]
    assert "204" in op["responses"], sorted(op["responses"])
    said = op["responses"]["204"].get("description", "")
    assert "no committed version" in said.lower(), said


def test_no_handler_re_raises_http_without_translating_the_rest():
    """`except HTTPException: raise` never stands alone.

    On its own it is a no-op: it catches one type and re-raises it, which is what happens with
    no `try` at all. It is meaningful only paired with an `except Exception` that logs and turns
    the failure into a 500 a caller can act on. Without the pair, a store-level failure reaches
    FastAPI's default handler with no log line, and the half-wrapper looks like the failure was
    considered."""
    import ast
    import io as _io

    from mantle.routers import artifacts_router

    tree = ast.parse(_io.open(artifacts_router.__file__, encoding="utf-8").read())
    lonely = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for t in ast.walk(node):
            if not isinstance(t, ast.Try):
                continue
            names = [getattr(h.type, "id", None) or getattr(h.type, "attr", None) or "BARE"
                     for h in t.handlers]
            reraises_http = any(
                (getattr(h.type, "id", None) == "HTTPException")
                and len(h.body) == 1 and isinstance(h.body[0], ast.Raise)
                and h.body[0].exc is None
                for h in t.handlers)
            if reraises_http and "Exception" not in names:
                lonely.append("%s (line %d): %s" % (node.name, t.lineno, names))

    assert not lonely, (
        "a bare `except HTTPException: raise` that translates nothing else:" + chr(10) + "  "
        + (chr(10) + "  ").join(lonely))


def test_revert_declares_the_top_level_refusal(artifact_ops):
    """. A top-level artifact has no containing workspace, so there is no workspace-scoped
    draft to revert.

    Measured 2026-08-26: the handler's `or doc.get("_key")` fallback was dead — `_key` is never
    present on a lattice artifact doc (every other reader spells it `id or _key`, so `_key` aliases
    the artifact's OWN id, not its container). The caller therefore got `404 Workspace not found`,
    which names a missing workspace when the truth is that this kind of artifact has none. Had the
    clause fired it would have been worse: passing the artifact's own id as the workspace is
    rejected by `revert_artifact`'s own `target.collection_id != workspace_id` check, so the route
    would have answered `204 nothing to revert` for an artifact that had something to revert."""
    op = artifact_ops[("/artifacts/{artifact_id}/revert", "post")]
    assert "400" in op["responses"], sorted(op["responses"])


def test_the_dead_key_fallback_is_gone():
    """The specific line, so it cannot come back by copy-paste."""
    import io as _io

    from mantle.routers import artifacts_router

    src = _executable(_io.open(artifacts_router.__file__, encoding="utf-8").read())
    assert 'doc.get("collection_id") or doc.get("_key")' not in src, (
        "the dead `_key` fallback is back: `_key` aliases the artifact's own id, never its "
        "container, so it can only mislead")


def test_no_handler_in_this_router_fails_silently():
    """, as a class. `except Exception: pass` is the failure mode this whole audit keeps
    turning up: no error, no log, and a quietly worse answer.

    The one this was written for swallowed lazy-index materialization, so an artifact that
    failed to materialize stayed LATENT for ever — the read succeeded, the caller saw a normal
    `200`, and nothing on any schedule would have found it. Three more silently treated a
    malformed context as an absent one, which makes a content route answer "no content" about an
    artifact whose bytes are present.

    Swallowing is often CORRECT here — a failed announcement must not fail the read. What is
    never correct is swallowing without a trace, so this pins the count at zero rather than
    forbidding the pattern. A handler that genuinely needs to do nothing can say so with a log
    line; if one truly cannot, this test is the place to argue it."""
    import ast
    import io as _io

    from mantle.routers import artifacts_router

    tree = ast.parse(_io.open(artifacts_router.__file__, encoding="utf-8").read())
    silent = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for h in node.handlers:
            if len(h.body) == 1 and isinstance(h.body[0], ast.Pass):
                silent.append("line %d" % h.lineno)

    assert not silent, (
        "exception handlers that swallow with no trace: " + ", ".join(silent))


def test_the_recall_body_is_fully_described():
    """. The model carried ~90 lines of `#:` Sphinx comments, which pydantic does not lift
    into the JSON schema, so the published body described nothing."""
    from mantle.main import app as _app

    spec = _app.openapi()
    body = spec["paths"]["/artifacts/recall"]["post"]["requestBody"]
    ref = body["content"]["application/json"]["schema"]["$ref"].split("/")[-1]
    props = spec["components"]["schemas"][ref]["properties"]
    undescribed = sorted(k for k, v in props.items() if not v.get("description"))
    assert not undescribed, "recall body fields with no description: %s" % undescribed


def test_the_filter_grammar_reaches_the_field_it_governs():
    """. The grammar lived only in the handler docstring, which FastAPI publishes as the
    OPERATION description — and a client building an input for `query_text` reads the FIELD.

    The filterable list is DERIVED from `filterable_field_names`, the same function the
    parser's own error text uses, so the spec cannot teach a filter the parser refuses. This
    asserts that derivation rather than a fixed list: retyping the names here would recreate the
    second home the derivation exists to prevent."""
    from mantle.main import app as _app
    from mantle.search.field_filters import filterable_field_names

    spec = _app.openapi()
    body = spec["paths"]["/artifacts/recall"]["post"]["requestBody"]
    ref = body["content"]["application/json"]["schema"]["$ref"].split("/")[-1]
    desc = spec["components"]["schemas"][ref]["properties"]["query_text"]["description"]

    missing = [f for f in filterable_field_names() if ("`%s`" % f) not in desc]
    assert not missing, "filterable fields the description does not list: %s" % missing
    assert "misspelled" in desc.lower(), (
        "the description does not state the deliberate trade — that an unknown `word:value` "
        "searches as a term instead of erroring")
