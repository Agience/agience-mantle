"""Collections & artifacts — the core CRUD surface.

Top-level container (no container_id) → committed collection with an owner grant;
child artifact (container_id present) → draft. Then read / update / children /
commits / batch / delete.
"""
from __future__ import annotations

import uuid


def test_create_top_level_collection_is_committed(user):
    coll = user["api"].create_collection(f"coll-{uuid.uuid4().hex[:6]}")
    assert coll["id"]
    assert coll.get("state") == "committed", coll
    # The owner can read it back.
    r = user["api"].get_artifact(coll["id"])
    assert r.status_code == 200, r.text


def test_create_child_is_draft(user):
    coll = user["api"].create_collection(f"coll-{uuid.uuid4().hex[:6]}")
    child = user["api"].create_child(coll["id"], name="note-1", content="hello world")
    assert child["id"]
    assert child.get("state") == "draft", child


def test_get_artifact_reports_children(user):
    coll = user["api"].create_collection(f"coll-{uuid.uuid4().hex[:6]}")
    user["api"].create_child(coll["id"], name="c1", content="one")
    user["api"].create_child(coll["id"], name="c2", content="two")
    r = user["api"].get_artifact(coll["id"])
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc.get("has_children") is True
    assert doc.get("child_count", 0) >= 2


def test_children_listing(user):
    coll = user["api"].create_collection(f"coll-{uuid.uuid4().hex[:6]}")
    a = user["api"].create_child(coll["id"], name="c1", content="one")
    r = user["api"].get(f"/artifacts/{coll['id']}/children", params={"workspace_id": coll["id"]})
    assert r.status_code == 200, r.text
    ids = {c.get("id") for c in r.json()["items"]}
    assert a["id"] in ids


def test_update_artifact_content(user):
    coll = user["api"].create_collection(f"coll-{uuid.uuid4().hex[:6]}")
    child = user["api"].create_child(coll["id"], name="edit-me", content="before")
    r = user["api"].patch(f"/artifacts/{child['id']}", json={"content": "after"})
    assert r.status_code == 200, r.text
    assert r.json().get("content") == "after"


def test_batch_fetch(user):
    coll = user["api"].create_collection(f"coll-{uuid.uuid4().hex[:6]}")
    a = user["api"].create_child(coll["id"], name="c1", content="one")
    b = user["api"].create_child(coll["id"], name="c2", content="two")
    r = user["api"].post("/artifacts/batch", json={"artifact_ids": [a["id"], b["id"], "does-not-exist"]})
    assert r.status_code == 200, r.text
    got = {d.get("id") for d in r.json()["items"]}
    assert {a["id"], b["id"]} <= got
    assert "does-not-exist" not in got  # inaccessible/missing silently skipped


def test_delete_artifact(user):
    coll = user["api"].create_collection(f"coll-{uuid.uuid4().hex[:6]}")
    child = user["api"].create_child(coll["id"], name="delete-me", content="x")
    r = user["api"].delete(f"/artifacts/{child['id']}")
    assert r.status_code == 200, r.text
    assert r.json().get("deleted") is True
    # Gone → 404 on read.
    assert user["api"].get_artifact(child["id"]).status_code == 404


def test_child_in_missing_container_is_404(user):
    r = user["api"].post("/artifacts", json={
        "container_id": "no-such-container", "name": "orphan", "content": "x",
        "content_type": "text/plain",
    })
    assert r.status_code == 404, r.text
