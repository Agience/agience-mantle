"""`source_artifact_id` is honoured or refused — never silently ignored.

The defect this pins. `source_artifact_id` is a declared, documented parameter — *"link an
existing artifact instead of creating one"* — and there were two ways to supply it and have it
discarded, both answered `201`:

  * **Without `container_id`**: the top-level branch handles that whole case and returns without
    ever reading it, so the caller got a new empty artifact where it asked for a link.
  * **With `identity`**: the identity branch inside the collection path returns first, so the link
    was dropped and an identity-derived member was written instead.

Both are silent data loss on a write: the request "succeeded" and did something else. A `400`
naming the conflict is the only answer that cannot be mistaken for what the caller wanted.

This converts some previous `201`s into `400`s deliberately. Unlike 's unknown-field
case — where an extra key may be harmless and the caller population is unmeasurable because
`crystal/dispatcher.py` forwards bodies verbatim — the field here is DECLARED, so a caller sending
it is already not getting what the contract promises. There is nothing to preserve.
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_source_artifact_id_without_a_container_is_refused(client):
    resp = await client.post("/artifacts", json={"source_artifact_id": "some-artifact"})
    assert resp.status_code == 400, (
        "a link request with no collection to link INTO was answered %s; it used to be a 201 that "
        "created an empty artifact instead" % resp.status_code)
    assert "container_id" in resp.json()["detail"], resp.json()


@pytest.mark.asyncio
async def test_source_artifact_id_with_identity_is_refused(client):
    resp = await client.post("/artifacts", json={
        "source_artifact_id": "some-artifact",
        "identity": "file:/repo/README.md",
        "container_id": "some-collection",
    })
    assert resp.status_code == 400, (
        "identity + source_artifact_id was answered %s; identity used to win and the link was "
        "dropped without a word" % resp.status_code)
    detail = resp.json()["detail"]
    assert "identity" in detail and "source_artifact_id" in detail, detail


@pytest.mark.asyncio
async def test_the_refusal_names_what_to_do(client):
    """A 400 that does not say which parameter to drop leaves the caller guessing."""
    resp = await client.post("/artifacts", json={"source_artifact_id": "a", "identity": "b",
                                                 "container_id": "c"})
    assert "Supply one" in resp.json()["detail"], resp.json()


@pytest.mark.asyncio
async def test_a_create_without_source_artifact_id_is_unaffected(client):
    """Guard the narrowness: the check must not fire for ordinary creates."""
    resp = await client.post("/artifacts", json={"content": "hello", "content_type": "text/plain"})
    assert resp.status_code != 400 or "source_artifact_id" not in resp.json().get("detail", ""), (
        "an ordinary create was caught by the source_artifact_id guard: %r" % resp.json())
