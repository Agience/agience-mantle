"""A page of artifacts answers when one body cannot be opened by this reader (BUG-051, 2026-11-10).

`GET /artifacts/visible` failed whole (500) for a persona's delegation, which opens no person's
content key, because one listed artifact's stored body could not be decrypted for it. A list drops
that body from its row, as the boundary documents for every list; a single read still refuses.
"""
from __future__ import annotations

from mantle.db import backend
from mantle.routers import artifacts_router as R


def test_a_page_drops_a_body_it_cannot_open_and_answers(monkeypatch):
    docs = {"a": {"id": "a", "root_id": "a", "state": "live", "content": "cas-ref"},
            "b": {"id": "b", "root_id": "b", "state": "live", "content": "plain"}}
    seen = []

    def decrypt(doc, *, strict=True):
        seen.append(strict)
        if doc["id"] == "a":
            if strict:
                raise RuntimeError("could not be decrypted")
            doc["content"] = ""

    monkeypatch.setattr(backend, "get_raw_artifact", lambda db, key: dict(docs[key]))
    monkeypatch.setattr(backend, "_decrypt_artifact_content", decrypt)
    monkeypatch.setattr(R, "_normalize_artifact_doc", lambda d: d)
    out = R._hydrate_batch(object(), ["a", "b"])
    assert set(out) == {"a", "b"} and out["a"]["content"] == "" and out["b"]["content"] == "plain"
    assert seen and not any(seen), "a list asks for the body without refusing the page"
