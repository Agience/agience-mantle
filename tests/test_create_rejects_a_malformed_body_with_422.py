"""`POST /artifacts` answers a malformed body with 422, not 500.

The defect this pins. The route takes `body: Dict[str, Any] = Body(...)` and constructs
`CreateArtifactRequest(**body)` INSIDE the handler, so a pydantic `ValidationError` is raised in
the handler rather than by FastAPI's request validation. It is not a `RequestValidationError`,
nothing converted it, and it reached `main.py`'s catch-all `@app.exception_handler(Exception)` —
which answers `500 {"detail": "Internal Server Error"}`.

A caller who sent a wrong field type was told the SERVER had broken, learned nothing about which
field, and left a 500 in the operator's log for a mistake that was theirs. Three costs, one line.

Nothing that worked before behaves differently: the same model is constructed with the same
extras policy, so exactly the same bodies are accepted. Only the answer to one already being
rejected changed.
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_a_wrongly_typed_field_is_422_and_names_the_field(client):
    """The status AND the body matter — a 422 that says nothing is barely better than the 500."""
    resp = await client.post("/artifacts", json={"content_type": 123, "content": {}})

    assert resp.status_code != 500, (
        "a malformed request body is reported as a server fault; the caller cannot tell that the "
        "mistake was theirs, and it lands in the 500 log")
    assert resp.status_code == 422, resp.status_code

    detail = resp.json().get("detail")
    assert isinstance(detail, list) and detail, resp.json()
    fields = {str(loc) for item in detail for loc in (item.get("loc") or ())}
    assert "content_type" in fields, (
        "the 422 does not name the offending field, so the caller still cannot fix the request: %r"
        % detail)


@pytest.mark.asyncio
async def test_the_error_carries_no_documentation_url(client):
    """`exc.errors(include_url=False)`. Pydantic otherwise appends a `url` to every error pointing
    at its own docs — a public error body advertising the server's validation library and version."""
    resp = await client.post("/artifacts", json={"content_type": 123, "content": {}})
    assert resp.status_code == 422
    assert all("url" not in item for item in resp.json()["detail"]), resp.json()


@pytest.mark.asyncio
async def test_a_well_formed_body_is_not_rejected_by_the_new_guard(client):
    """The vacuous-pass guard, inverted: if the guard rejected everything, the test above would
    pass for the wrong reason. This asserts only that validation is not what stops it — the request
    may still fail further in for reasons this test does not care about."""
    # Every field on `CreateArtifactRequest` is Optional[str]; `content` is a STRING, not an
    # object. My first version of this test sent `{"body": "hello"}` and was correctly refused —
    # which is the inverted guard doing its job on the test rather than on the code.
    resp = await client.post(
        "/artifacts", json={"content_type": "text/plain", "content": "hello"})
    assert resp.status_code != 422, (
        "a well-formed create body is now refused by validation: %r" % resp.json())


@pytest.mark.asyncio
async def test_an_unknown_field_is_still_ignored_rather_than_refused(client):
    """The neutrality claim that justified declaring the body model, asserted end to end.

    `POST /artifacts` used to take `body: Dict[str, Any] = Body(...)` and build the model by hand.
    Declaring `body: CreateArtifactRequest` READS like a tightening, and the audit assumed it was
    one. It is not: `CreateArtifactRequest(**body)` and `.model_validate(body)` were compared and
    produce identical results, both ignoring unknown keys — `model_config` is empty and pydantic v2
    defaults to `extra="ignore"`.

    If someone later sets `extra="forbid"` on that model, this test fails, and it should: that IS a
    breaking change to a published contract, and it must be a decision rather than a side effect."""
    resp = await client.post("/artifacts", json={
        "content_type": "text/plain",
        "content": "hello",
        "a_field_no_client_should_send": 1,
    })
    assert resp.status_code != 422, (
        "an unknown key is now REFUSED where it used to be ignored — that is a wire-contract "
        "change, not a declaration: %r" % resp.json())


@pytest.mark.asyncio
async def test_the_published_schema_names_the_fields(client):
    """The point of declaring the model. `POST /artifacts` advertised
    `{"type": "object", "additionalProperties": true}` — any object at all — so a generated client
    knew no field names, and every field doc lifted into the model reached the source and nowhere
    else."""
    from mantle.main import app

    spec = app.openapi()
    body = spec["paths"]["/artifacts"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert body.get("$ref", "").endswith("CreateArtifactRequest"), body
    props = spec["components"]["schemas"]["CreateArtifactRequest"]["properties"]
    assert "content_type" in props and "identity" in props
    assert any(v.get("description") for v in props.values()), (
        "no field carries a description — the lifted docs are not reaching the spec")

