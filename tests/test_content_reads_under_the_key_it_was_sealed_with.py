"""The read must try the principal the WRITE actually sealed under.

`doc_boundary.encrypt_artifact_content` seals under `content_key_principal` — its docstring:
"Keyed on the collection's origin root ... NOT on `created_by`". `decrypt_artifact_content` asked
for `CONTENT_KEY_PRINCIPAL or created_by`. Those agree only when `origin_root` is stamped on the
doc.

Measured 2026-08-25 on 71/home: the 310,003 bulk-ingested Wikipedia artifacts carry `collection_id`
and no `origin_root`, so every read asked for `created_by` (`54eaa8aa`, ember-source) while the
envelope had been sealed under `stage.1.grammar`. Every body failed to hydrate. The surfaced error
was `GrantDenied`, which sent the diagnosis after a missing grant — one was minted for `created_by`
and changed nothing, because that principal has no master key at all. Opening the same blob under
the collection returned 16,116 bytes on the first attempt. After the fix: 240/240 sampled bodies
hydrate.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from mantle.db import doc_boundary
from mantle.db.doc_boundary import CONTENT_KEY_PRINCIPAL


def _doc(**over):
    d = {
        "id": "wiki-simple-1",
        "content_ref": "cas/" + "ab" * 32,
        "content": "",
        "collection_id": "stage.1.grammar",
        "created_by": "54eaa8aa-ember-source",
    }
    d.update(over)
    return d


def _opens_only_for(*principals):
    """A `get_bytes_decrypted` that answers for these principals and raises for any other."""
    def fake(content_key, owner_id, *, cas_ref=None, collection_id=None):
        if owner_id in principals:
            return b"the article body"
        raise RuntimeError("no master key exists for principal %r" % owner_id)
    return fake


class TestTheReadFindsTheSealingPrincipal:

    def test_a_body_sealed_under_the_collection_is_found(self):
        """The measured case: no `origin_root`, sealed under the collection, `created_by` useless."""
        doc = _doc()
        with patch("mantle.services.content_service.get_bytes_decrypted",
                   _opens_only_for("stage.1.grammar")):
            doc_boundary.decrypt_artifact_content(doc, strict=True)
        assert doc["content"] == "the article body"

    def test_a_stamped_principal_is_still_authoritative(self):
        doc = _doc(**{CONTENT_KEY_PRINCIPAL: "the-stamped-one"})
        with patch("mantle.services.content_service.get_bytes_decrypted",
                   _opens_only_for("the-stamped-one")):
            doc_boundary.decrypt_artifact_content(doc, strict=True)
        assert doc["content"] == "the article body"

    def test_created_by_still_works_for_rows_keyed_that_way(self):
        """`content_key_principal` documents pre-migration rows as "keyed to `created_by`"."""
        doc = _doc()
        with patch("mantle.services.content_service.get_bytes_decrypted",
                   _opens_only_for("54eaa8aa-ember-source")):
            doc_boundary.decrypt_artifact_content(doc, strict=True)
        assert doc["content"] == "the article body"

    def test_when_no_candidate_opens_it_the_error_survives(self):
        """Failing closed is the point: a body that cannot be opened must not read as empty."""
        doc = _doc()
        with patch("mantle.services.content_service.get_bytes_decrypted",
                   _opens_only_for("nobody-here")):
            with pytest.raises(doc_boundary.ContentDecryptionError):
                doc_boundary.decrypt_artifact_content(doc, strict=True)

    def test_a_non_strict_read_still_drops_the_body_rather_than_raising(self):
        """List pages must not fail wholesale on one unreadable row."""
        doc = _doc()
        with patch("mantle.services.content_service.get_bytes_decrypted",
                   _opens_only_for("nobody-here")):
            doc_boundary.decrypt_artifact_content(doc, strict=False)
        assert doc["content"] == ""
