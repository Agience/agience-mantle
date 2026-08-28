"""A create that turned out to be an update answers 200, not 201.

The defect this pins. `POST /artifacts` declares `status_code=201_CREATED` for every path, and
two paths through it create nothing: an `identity` that already names an artifact is an UPDATE,
top-level or inside a collection. `201 Created` on an overwrite is a claim the response cannot
support — and the caller most likely to be misled is the one using `identity` CORRECTLY, since
making the write an upsert is the entire reason to supply it.

The outcome could not be read from the return value: both branches hand back the artifact
document and are indistinguishable. `_default_create_artifact` and `upsert_identity_member`
therefore report into a dict, and the route sets the status from that.

Measured safe before changing a status code: `seraph/server.py:890` and `:980` already accept
`(200, 201)`, and no in-tree caller requires exactly 201.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mantle.routers import artifacts_router as ar

BODY = {"identity": "file:/repo/README.md", "content": "body", "content_type": "text/markdown"}


@pytest.mark.asyncio
async def test_an_identity_write_that_updates_answers_200(client):
    updated = MagicMock()
    updated.to_dict.return_value = {"id": "derived"}
    with (
        patch.object(ar, "_artifact_exists", return_value=True),
        patch("mantle.services.workspace_service.update_workspace", return_value=updated),
    ):
        resp = await client.post("/artifacts", json=BODY)
    assert resp.status_code == 200, (
        "an overwrite answered %s; `201 Created` says something was created and nothing was"
        % resp.status_code)


@pytest.mark.asyncio
async def test_a_first_identity_write_still_answers_201(client):
    """The narrowness guard. If this drifts to 200 the route stopped reporting creates at all."""
    created = MagicMock()
    created.to_dict.return_value = {"id": "derived"}
    with (
        patch.object(ar, "_artifact_exists", return_value=False),
        patch("mantle.services.workspace_service.create_container", return_value=created),
    ):
        resp = await client.post("/artifacts", json=BODY)
    assert resp.status_code == 201, (
        "a genuine create answered %s; the 200 branch is firing too widely" % resp.status_code)


@pytest.mark.asyncio
async def test_both_outcomes_are_declared_in_the_spec():
    """A status a caller can receive and cannot find in the spec is not a contract."""
    from mantle.main import app
    responses = app.openapi()["paths"]["/artifacts"]["post"]["responses"]
    assert "200" in responses and "201" in responses, sorted(responses)
    d = responses["200"]["description"]
    # Both causes must be named: an identity UPDATE (C6) and a source_artifact_id LINK (C7).
    assert "NOTHING WAS CREATED" in d, d
    assert "identity" in d and "source_artifact_id" in d, d


def test_the_report_dict_is_optional():
    """`report=` defaults to None so the existing three-argument call sites keep working —
    including the test that calls `_default_create_artifact` directly."""
    import inspect
    sig = inspect.signature(ar._default_create_artifact)
    assert sig.parameters["report"].default is None


@pytest.mark.asyncio
async def test_a_link_answers_200_because_no_artifact_was_created(client):
    """. `source_artifact_id` mints a membership EDGE and no artifact, and the body it
    returns is the SOURCE document — which existed before the call and is unchanged by it. `201
    Created` with that body claims the caller created the thing it linked."""
    source = MagicMock()
    source.root_id = None
    source.id = "existing"
    source.to_dict.return_value = {"id": "existing"}
    with (
        # `_link_source_artifact` imports these from `mantle.db.backend` INSIDE the function,
        # so they must be patched at their source module, not on the router.
        patch("mantle.db.backend.get_artifact", return_value=source),
        patch("mantle.db.backend.add_artifact_to_collection", return_value=None),
        patch.object(ar, "check_access", return_value=None),
        patch.object(ar, "_artifact_exists", return_value=True),
    ):
        resp = await client.post("/artifacts", json={
            "source_artifact_id": "existing", "container_id": "some-collection"})
    assert resp.status_code == 200, (
        "a link answered %s; nothing was created, so 201 misdescribes it" % resp.status_code)
