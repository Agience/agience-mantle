"""Text extraction must read the local CAS before the object store.

`workspace_service._store_content_in_s3` states the contract: "One destination. Content lives at
`cas/<sha256 of the plaintext>` ... the object store is where the ref gets promoted to rather than
an alternative place to put the bytes." So an artifact carries BOTH a `content_ref` (the local
address) and a `content_key` (the promotion address), and `extract_text_from_artifact` read only
the second.

Measured 2026-08-25 on 71/home, a node with no object-store credentials — a supported configuration
per `content_service.local_content_tier`. All 197 memory-lane captures had their blob present on
local disk; 116 of them failed to index with `ContentUrlSigningError ... NoCredentialsError`. The
body was two directories away and the indexer went to the internet for it. After the fix: 197/197
indexed, 0 failed.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from mantle.services import ingest_runner_service as irs


def _artifact(**over):
    doc = {"id": "a1", "created_by": "principal-1", "collection_id": "c1"}
    doc.update(over.pop("doc", {}))
    art = SimpleNamespace(
        id="a1",
        content="",
        context='{"content_key": "artifacts/a1.content", "content_type": "text/markdown"}',
        content_ref="cas/" + "ab" * 32,
        created_by="principal-1",
        to_dict=lambda: doc,
    )
    for k, v in over.items():
        setattr(art, k, v)
    return art


class TestTheLocalTierIsTriedFirst:

    def test_a_local_blob_is_read_without_touching_the_object_store(self):
        with (
            patch.object(irs, "extract_text_from_s3") as s3,
            patch("mantle.services.content_service.get_bytes_decrypted",
                  return_value=b"the captured body"),
        ):
            out = irs.extract_text_from_artifact(_artifact())
        assert out == "the captured body"
        s3.assert_not_called(), "the body was local; going to S3 for it is the defect"

    def test_the_object_store_is_still_the_fallback(self):
        """A node WITH credentials and no local copy must keep working exactly as before."""
        with (
            patch.object(irs, "extract_text_from_s3", return_value="from the bucket") as s3,
            patch("mantle.services.content_service.get_bytes_decrypted",
                  side_effect=RuntimeError("not local")),
        ):
            out = irs.extract_text_from_artifact(_artifact())
        assert out == "from the bucket"
        s3.assert_called_once()

    def test_an_artifact_with_no_local_ref_goes_straight_to_the_store(self):
        with patch.object(irs, "extract_text_from_s3", return_value="from the bucket") as s3:
            out = irs.extract_text_from_artifact(_artifact(content_ref=None))
        assert out == "from the bucket"
        s3.assert_called_once()

    def test_inline_content_still_wins_over_both(self):
        """Unchanged: an artifact carrying its own body needs neither tier."""
        with (
            patch.object(irs, "extract_text_from_s3") as s3,
            patch("mantle.services.content_service.get_bytes_decrypted") as local,
        ):
            out = irs.extract_text_from_artifact(_artifact(content="already here"))
        assert out == "already here"
        s3.assert_not_called()
        local.assert_not_called()
