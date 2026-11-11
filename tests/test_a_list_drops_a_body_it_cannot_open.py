"""A page of artifacts answers when one body cannot be opened by this reader (BUG-051, BUG-059).

`GET /artifacts/visible` failed whole (500) for a persona's delegation, which opens no person's
content key, because one listed artifact's stored body could not be decrypted for it. A list drops
that body from its row, as the boundary documents for every list; a single read still refuses.

BUG-059 (2026-11-10): the page decrypted without refusing, and then `_normalize_artifact_doc`
decrypted the same row again, strictly. A dropped body leaves the row with its content address and
no content, which is exactly what the decrypt retries, so the page failed after all. On staging,
which holds none of production's bodies, every typed `/artifacts/visible` answered 500. The first
version of this test replaced `_normalize_artifact_doc` with a pass-through and so never saw it.
"""
from __future__ import annotations

from mantle.db import backend
from mantle.routers import artifacts_router as R


def _decrypt_like_the_boundary(seen):
    """As `doc_boundary._decrypt_body`: a row with a content address and no content is hydrated;
    one that cannot be read raises when strict, and is left with no content otherwise."""
    def decrypt(doc, *, strict=True):
        if doc.get("content_ref") and not doc.get("content"):
            seen.append(strict)
            if doc["id"] == "a":
                if strict:
                    raise RuntimeError("not in this node's local content tier")
                doc["content"] = ""
            else:
                doc["content"] = "plain"
    return decrypt


def test_a_page_drops_a_body_it_cannot_open_and_answers(monkeypatch):
    docs = {"a": {"id": "a", "root_id": "a", "state": "live", "content": "", "content_ref": "cas/a"},
            "b": {"id": "b", "root_id": "b", "state": "live", "content": "", "content_ref": "cas/b"}}
    seen = []
    monkeypatch.setattr(backend, "get_raw_artifact", lambda db, key: dict(docs[key]))
    monkeypatch.setattr(backend, "_decrypt_artifact_content", _decrypt_like_the_boundary(seen))
    out = R._hydrate_batch(object(), ["a", "b"])
    assert set(out) == {"a", "b"} and out["a"]["content"] == "" and out["b"]["content"] == "plain"
    assert seen and not any(seen), "a list asks for the body without refusing the page"


def test_a_single_read_through_normalize_still_refuses(monkeypatch):
    seen = []
    monkeypatch.setattr(backend, "_decrypt_artifact_content", _decrypt_like_the_boundary(seen))
    import pytest
    with pytest.raises(RuntimeError):
        R._normalize_artifact_doc({"id": "a", "root_id": "a", "content": "", "content_ref": "cas/a"})
