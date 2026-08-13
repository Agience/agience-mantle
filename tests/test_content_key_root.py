"""The content key roots at the collection's origin root — never at ``created_by``.

``oracle.py`` states the rule: "The principal is the collection's immutable origin
root, NOT an 'owner' / created_by. Agience has no owners — access is by grant."

Rooting crypto at ``created_by`` would be destructive: ``created_by`` is
provenance, and provenance can be corrected (for example, two identities folded
into one). ``created_by`` also doubles as the GCM AAD, so if it were the crypto
root, correcting it would leave every blob written under the old value
simultaneously underivable and unauthenticatable.

``db/content_cache.collection_key`` documents the same rule for the
target store. These tests pin it for the artifact path.
"""
from __future__ import annotations

import pytest

from mantle.db import backend as db_store
from mantle.db import doc_boundary
from mantle.entities.artifact import Artifact


class TestOriginRootIsInherited:
    """It is INHERITED at creation, never walked at read time."""

    def test_top_level_artifact_is_its_own_origin_root(self):
        a = Artifact(id="art-1", collection_id="")
        assert a.origin_root == "art-1"

    def test_child_is_not_silently_self_rooted(self):
        """An unstamped child must be visibly unstamped.

        Defaulting it to its own id would be worse than leaving it empty: the child
        would claim to be a root, get its own key tree, and be silently orphaned from
        the collection its cells live under — with nothing to distinguish it from a
        genuine top-level artifact.
        """
        a = Artifact(id="art-2", collection_id="col-1")
        assert a.origin_root is None

    def test_explicit_origin_root_is_kept(self):
        a = Artifact(id="art-3", collection_id="col-1", origin_root="root-9")
        assert a.origin_root == "root-9"

    def test_survives_the_storage_round_trip(self):
        """It must not be dropped by to_dict/from_dict: a field the entity does not
        model is silently lost on save, and a lost key root would be recomputed or
        defaulted on the next write, re-keying content whose ciphertext was written
        under the old value irrecoverably.
        """
        a = Artifact(id="art-4", collection_id="col-1", origin_root="root-9")
        assert Artifact.from_dict(a.to_dict()).origin_root == "root-9"


class TestContentKeyPrincipal:
    def test_prefers_origin_root_over_created_by(self):
        principal = doc_boundary.content_key_principal(
            {"origin_root": "root-9", "created_by": "user-alice"}
        )
        assert principal == "root-9", "created_by must not win over the origin root"

    def test_recorded_principal_is_honoured(self):
        """A blob records the root it was written under; that must be authoritative."""
        principal = doc_boundary.content_key_principal(
            {"content_key_principal": "root-7", "created_by": "user-alice"}
        )
        assert principal == "root-7"

    def test_created_by_is_the_transitional_fallback_only(self):
        """Rows without an origin_root fall back to created_by until the backfill
        in scripts/backfill_origin_root has run."""
        assert doc_boundary.content_key_principal({"created_by": "user-alice"}) == "user-alice"

    def test_no_root_at_all_fails_the_write(self):
        """An artifact with neither an origin_root nor a created_by has no content
        key principal to encrypt under, and must fail loudly rather than being
        persisted unencrypted."""
        with pytest.raises(doc_boundary.ContentEncryptionError):
            doc_boundary.content_key_principal({"id": "art-5", "collection_id": "col-1"})

    def test_encrypt_refuses_rather_than_storing_plaintext(self):
        doc = {"_key": "art-6", "collection_id": "col-1", "content": "secret"}
        with pytest.raises(doc_boundary.ContentEncryptionError):
            doc_boundary.encrypt_artifact_content(doc)
        assert doc["content"] == "secret", "content was mutated despite the failure"
        assert not doc.get("content_encrypted"), (
            "flagged as encrypted without being encrypted — the exact state that "
            "makes plaintext indistinguishable from ciphertext downstream"
        )


class TestNoLineageWalkOnTheHotPath:
    def test_resolving_the_key_principal_touches_no_database(self, monkeypatch):
        """Resolving the content key principal must not touch the database: the
        root travels with the row, and this layer is store-agnostic. Nothing here
        may look anything up.
        """
        def _boom(*a, **kw):
            raise AssertionError("the content key path hit the database")

        monkeypatch.setattr(db_store, "get_origin_root", _boom)
        monkeypatch.setattr("mantle.services.dependencies.get_store_db", _boom)

        assert doc_boundary.content_key_principal({"origin_root": "root-9"}) == "root-9"
