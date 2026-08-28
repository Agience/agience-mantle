"""Each documented 200 envelope names exactly the keys its handler returns.

This is the price of not using `response_model=`. Routes document their 200 with
`responses={200: {"model":...}}` rather than `response_model=`, because `response_model` filters:
the day a handler adds a key and the model is not updated, FastAPI silently drops it and clients
see a field stop arriving. `responses=` cannot cause that, but it can drift, describing a shape the
handler does not return.

The drift is caught here instead, statically, by comparing each model's fields against the keys of
its route's returns. A documented envelope must be checkable: an undocumented response is a gap a
client can see, while a documented one that lies is worse, because a client will write code for it.

A route whose return shape is not knowable from the router must not carry a documented envelope.
"""
from __future__ import annotations

import ast
import io
import os

import pytest

from mantle.routers import artifacts_router as ar

_ROUTER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "src", "mantle", "routers", "artifacts_router.py")


def _documented():
    """`{handler_name: model}` for every route whose 200 carries a model."""
    src = io.open(_ROUTER, encoding="utf-8").read()
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            if not isinstance(d, ast.Call):
                continue
            for kw in d.keywords:
                if kw.arg != "responses" or not isinstance(kw.value, ast.Call):
                    continue
                for inner in kw.value.keywords:
                    if inner.arg == "ok" and isinstance(inner.value, ast.Name):
                        out[node.name] = inner.value.id
    return out


def _page_keys():
    """`_page`'s own key set, derived from its return rather than typed here.

    `_page` is the single producer of the `{items, total, has_more}` shape, so reading its literal
    means a change to that shape re-derives here instead of failing as this file being out of
    date."""
    src = io.open(_ROUTER, encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_page":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                    return {k.value for k in sub.value.keys if isinstance(k, ast.Constant)}
    return None


def _returned_keys(fn_name: str):
    """The keys of every `return` in one handler, resolving `_page(...)` to its shape.

    A `**spread` of anything other than a resolvable helper yields `None`, and an unknowable
    return must not carry a documented envelope.

    `_page(...)` and `_artifact_body(...)` both resolve, because each guarantees a key set that is
    derived from a model rather than typed — `_ARTIFACT_KEYS = tuple(ArtifactResponse.model_fields)`,
    filled by `setdefault`. What the gate needs is a return whose guaranteed keys are knowable
    statically, and a helper that guarantees them is stronger evidence than a literal, which shows
    only what one branch happens to spell.

    A returned local name resolves to its last assignment in the function, which is how
    `read_artifact` and `update_artifact` are read: both build `body = _artifact_body(...)`."""
    page = _page_keys()
    #: Both derived from their own model, so neither can drift from what it documents.
    guaranteed = {"_artifact_body": set(ar._ARTIFACT_KEYS),
                  "_artifact_detail_body": set(ar._ARTIFACT_DETAIL_KEYS)}
    src = io.open(_ROUTER, encoding="utf-8").read()
    tree = ast.parse(src)

    def _is_page_call(value):
        return (isinstance(value, ast.Call)
                and getattr(value.func, "id", "") == "_page")

    def _artifact_shape(value):
        """The key set a `_artifact_*_body(…)` call GUARANTEES, or `None` if it is not one."""
        if not isinstance(value, ast.Call):
            return None
        return guaranteed.get(getattr(value.func, "id", ""))

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != fn_name:
            continue
        shapes = []
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Return) or sub.value is None:
                continue
            value = sub.value
            #: A returned local name resolves to its last assignment BEFORE the return — both
            #: `read_artifact` and `update_artifact` bind `body = _artifact_body(…)` and return
            #: the name. `update_artifact`’s request parameter is ALSO called `body`, which the
            #: assignment rebinds; taking the last assignment above the return is what makes that
            #: read correctly rather than resolving to the request model.
            if isinstance(value, ast.Name):
                bound = [a.value for a in ast.walk(node)
                         if isinstance(a, ast.Assign) and a.lineno <= sub.lineno
                         and any(isinstance(t, ast.Name) and t.id == value.id for t in a.targets)]
                if not bound:
                    return None
                value = bound[-1]
            if _is_page_call(value):
                if page is None:
                    return None
                shapes.append(set(page))
                continue
            shape = _artifact_shape(value)
            if shape is not None:
                shapes.append(set(shape))
                continue
            if not isinstance(value, ast.Dict):
                continue
            sub = type("_S", (), {"value": value})()
            keys = set()
            for k, v in zip(value.keys, value.values):
                if k is None:                       # `**spread`
                    if _is_page_call(v) and page is not None:
                        keys |= page
                        continue
                    return None                     # spread of something unknowable
                if isinstance(k, ast.Constant):
                    keys.add(k.value)
                else:
                    return None
            shapes.append(keys)
        return shapes
    return None


def test_the_scan_finds_the_documented_routes():
    """A guard that reaches nothing reports green for ever."""
    doc = _documented()
    #: 9 → 11 → 10 on 2026-08-26, and the last step was a retreat taken on evidence.
    #: `list_visible` and `list_children` gained `PageResponse` and kept it — the `_page`
    #: resolution above already understood them.
    #:
    #: `create_artifact` lost its envelope, and so did the read and update that were about
    #: to gain one. Declaring them required enumerating a literal, and that literal truncated
    #: the artifact document: `created_by` on every response, plus `modified_by`,
    #: `content_ref`, and seventeen store-level fields on a task artifact.
    #:
    #: This gate is right that an unreadable return must not carry a documented envelope.
    #: The mistake was making the return readable by making the document closed — a
    #: hand-written `response_model`, doing the filtering this codebase refuses. An open
    #: document cannot be declared as a closed one without becoming one.
    #:
    #: 11 → 16 on 2026-08-26, and all five artifact routes came back — `create_artifact`,
    #: `read_artifact`, `update_artifact`, `upload_status` and `revert_artifact_endpoint`. The
    #: retreat above turned out to rest on a false choice: the envelope was never blocked by the
    #: document being open, it was blocked by this gate only knowing how to read a literal.
    #:
    #: Two changes made them readable without closing anything. `ArtifactResponse` gained
    #: `extra="allow"`, so the spec says `additionalProperties: true` — twelve keys guaranteed,
    #: more may arrive — and this gate learned to resolve `_artifact_body(…)` /
    #: `_artifact_detail_body(…)`, whose guaranteed key sets are derived from the models.
    #:
    #: And widening it found a real defect it was built to find: `update_artifact`’s
    #: top-level branch, `upload_status` and `revert` each returned a raw `to_dict`, which
    #: omits unset fields. Declaring `ok=` on those would have promised twelve always-present
    #: keys while shipping a shape that varied with the data. All three now go through the
    #: helper. A gate that can read more finds more.
    assert len(doc) == 16, "expected 16 documented envelopes, found %d: %r" % (len(doc), doc)


@pytest.mark.parametrize("handler,model_name", sorted(_documented().items()))
def test_the_model_names_exactly_what_the_handler_returns(handler, model_name):
    model = getattr(ar, model_name)
    declared = set(model.model_fields)
    shapes = _returned_keys(handler)

    assert shapes, (
        "%s has no statically-readable `return {...}` — it must not carry a documented envelope, "
        "because nothing can check what it promises" % handler)

    #: An open schema permits extra keys — that is what it means.
    #: `ArtifactResponse` carries `extra="allow"` and publishes `additionalProperties: true`,
    #: because an artifact document is open: a content type may add fields, and a task artifact
    #: carries seventeen store-level ones. For such a model the under-describe check would be
    #: asserting the opposite of the declaration, so only the promise check applies — every
    #: declared field must actually arrive. For a closed model both directions still hold.
    is_open = (model.model_config or {}).get("extra") == "allow"
    for returned in shapes:
        missing = returned - declared
        assert is_open or not missing, (
            "%s returns %r, which %s does not declare — the spec under-describes the response and "
            "a generated client has no field for data it is being sent"
            % (handler, sorted(missing), model_name))
        promised = declared - returned
        assert not promised, (
            "%s declares %r, which %s never returns — the spec promises a field that never "
            "arrives, and a client will write code for it"
            % (model_name, sorted(promised), handler))


def test_the_spread_return_was_replaced_by_a_declared_one():
    """This test fired because the world improved, which is what it was for.

    It once asserted the opposite: `upload_initiate` returned `{**out,...}`, its keys came from
    a service call, and "no model can honestly describe it". Its failure message was the
    instruction — "it may now be documentable, and this test is the reminder to check rather than
    assume" — and on 2026-08-26 it fired for exactly that reason.

    What the check found: `initiate_upload_and_create_artifact` has one return and it is a
    literal, `{upload_id, mode, url, method, key}`. The shape was knowable one file away, so
    the spread had been hiding a contract rather than accommodating an unknown one. The handler now
    names the keys and the route declares `UploadInitiateResponse`.

    A spread is still refused everywhere else — the parametrised test above fails any documented
    envelope whose `return {...}` it cannot read. This one asserts the fix stayed fixed."""
    keys = _returned_keys("upload_initiate")
    assert keys, "upload_initiate went back to a return no checker can read"
    assert "upload_initiate" in _documented(), (
        "its return is statically readable but the route no longer declares a model — the whole "
        "point of reading it was to be able to promise it")
    for k in ("upload_id", "mode", "url", "method", "artifact"):
        assert any(k in shape for shape in keys), (
            "%s is no longer returned; a client cannot proceed without `url`" % k)
