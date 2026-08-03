"""The content key roots at the collection's ORIGIN ROOT — never at ``created_by``.

``oracle.py`` states the rule: *"The principal is the collection's immutable origin
root, NOT an 'owner' / created_by. Agience has no owners — access is by grant."* The
search path always obeyed it; artifact content did not, so one artifact had TWO key
roots — cells under the origin root, content under a user id.

Rooting crypto at ``created_by`` is not merely inconsistent, it is destructive.
``created_by`` is PROVENANCE, and provenance gets corrected — LATTICE §3 decides
John's two identities are folded into one. But ``created_by`` was both the HKDF root
AND the GCM AAD, so folding it leaves every blob written under the dropped value
simultaneously underivable and unauthenticatable. A metadata correction would have
silently destroyed readability.

``db/lattice/content_cache.collection_key`` already documents the same rule for the
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
        """An unstamped child must be VISIBLY unstamped.

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
        """It must not be dropped by to_dict/from_dict.

        This is the ``content_encrypted`` lesson: a field the entity does not model
        is silently lost on save, and a LOST KEY ROOT is recomputed or defaulted on
        the next write — re-keying content whose ciphertext was written under the old
        value, irrecoverably.
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
        """Legacy rows keep working until the backfill runs — see
        scripts/backfill_origin_root. This assertion is expected to be DELETED once
        the fallback is removed."""
        assert doc_boundary.content_key_principal({"created_by": "user-alice"}) == "user-alice"

    def test_no_root_at_all_fails_the_write(self):
        """⛔ Previously this stored PLAINTEXT.

        The guard read ``if not content or not owner: return`` — so an artifact with
        no ``created_by`` skipped encryption entirely and was persisted unencrypted,
        returning BEFORE the try/except that refuses to persist plaintext. Silent and
        permanent: a blob with no ``MEC1`` magic reads back as legacy plaintext
        forever. LATTICE §2.1 measures 80 such rows on node 71.
        """
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
        """⚠ REGRESSION GUARD. An earlier fix resolved the root via
        ``resolve_cell_principal(next(get_store_db()), ...)``, which put a database
        round trip on EVERY artifact write — 42s of connect timeouts per test, and a
        new the lattice dependency in the layer that is supposed to be store-agnostic
        while the lattice is being retired. The root travels with the row; nothing here
        may look anything up.
        """
        def _boom(*a, **kw):
            raise AssertionError("the content key path hit the database")

        monkeypatch.setattr(db_store, "get_origin_root", _boom)
        monkeypatch.setattr("mantle.services.dependencies.get_store_db", _boom)

        assert doc_boundary.content_key_principal({"origin_root": "root-9"}) == "root-9"
