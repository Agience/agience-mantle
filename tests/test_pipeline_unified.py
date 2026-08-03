"""Tests for the unified indexing pipeline (post lexical-backend retirement).

After Step 2.6.9 part 2 the pipeline writes to MANTLE vector cells +
MANTLE-SSE posting lists. The legacy BM25 index path and its
``_prepare_base_doc`` shape are gone.

These tests target the surviving public surface:

- ``_extract_artifact_fields`` produces the long-form per-field text
  dict the SSE indexer wants.
- ``index_artifact`` calls the SSE + MANTLE hooks routed to the segment
  for the artifact's state (committed/draft/archived — each a separate
  index), and purges the other segments so a state change moves the entry.
- ``index_artifacts_batch`` aggregates across the list (all states).
- ``delete_artifact_from_index`` removes from every segment when
  ``principal_id`` / ``collection_id`` are supplied.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from mantle.services.acting_principal import acting_as
from .helpers import self_request


def _archived_artifact():
    return SimpleNamespace(
        id="art-arch",
        root_id="art-arch",
        state="archived",
        created_by="user-1",
        modified_by="user-1",
        context="{}",
        content="",
        created_time="2026-05-09T00:00:00Z",
    )


def _committed_artifact(*, content="hello world"):
    return SimpleNamespace(
        id="art-1",
        root_id="art-1",
        state="committed",
        created_by="user-1",
        modified_by="user-1",
        collection_id="ws-1",
        context='{"title": "Encryption Library", "description": "A test", "tags": ["test"]}',
        content=content,
        content_type="text/plain",
        name="",
        description="",
        created_time="2026-05-09T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Fakes for the principal-resolution round trip (write key == de-index key)
# ---------------------------------------------------------------------------

_FAKE_DB = object()


def _fake_get_store_db():
    """A fresh single-item generator per call, exactly like the real dependency."""
    return iter([_FAKE_DB])


def _fake_get_origin_root(db, collection_id):
    """Refuses anything that is not a real handle — a generator object is not one.

    The real ``db.store.get_origin_root`` runs an AQL query and would raise on a
    generator too; making that explicit here is what lets the test see the fallback
    in ``resolve_cell_principal`` instead of silently agreeing with it.
    """
    if db is not _FAKE_DB:
        raise TypeError("not a database handle: %r" % (db,))
    # ws-child is NOT its own origin root — that gap is where the defect lives.
    return {"ws-child": "origin-root-1"}.get(collection_id, collection_id)


class _FakeSseIndex:
    """Posting lists keyed by (principal, artifact) — the key the defect gets wrong."""

    def __init__(self):
        self.postings: dict[tuple[str, str], dict] = {}

    def index_artifact(self, principal_id, collection_id, artifact_id, fields, request=None):
        self.postings[(principal_id, artifact_id)] = dict(fields)
        return len(fields)

    def remove_artifact(self, principal_id, artifact_id, request=None):
        self.postings.pop((principal_id, artifact_id), None)


# ---------------------------------------------------------------------------
# _extract_artifact_fields
# ---------------------------------------------------------------------------


class TestExtractArtifactFields:
    def test_extracts_title_description_tags_content(self):
        from mantle.search.ingest import pipeline_unified

        artifact = _committed_artifact()
        with patch.object(
            pipeline_unified, "extract_text_from_artifact",
            return_value="hello world",
        ):
            fields = pipeline_unified._extract_artifact_fields(artifact)
        assert fields["title"] == "Encryption Library"
        assert fields["description"] == "A test"
        assert fields["tags"] == "test"
        assert fields["content"] == "hello world"

    def test_falls_back_to_artifact_name_for_containers(self):
        from mantle.search.ingest import pipeline_unified

        artifact = SimpleNamespace(
            id="ws-1",
            root_id="ws-1",
            state="draft",
            created_by="user-1",
            collection_id="ws-1",
            context="",
            content="",
            name="Untitled Workspace",
            description="Workspace for quick notes",
            content_type="application/vnd.agience.workspace+json",
            created_time="2026-05-09T00:00:00Z",
        )
        with patch.object(
            pipeline_unified, "extract_text_from_artifact", return_value="",
        ):
            fields = pipeline_unified._extract_artifact_fields(artifact)
        assert fields.get("title") == "Untitled Workspace"
        assert fields.get("description") == "Workspace for quick notes"

    def test_invalid_context_json_handled(self):
        from mantle.search.ingest import pipeline_unified

        artifact = _committed_artifact()
        artifact.context = "{not json"
        with patch.object(
            pipeline_unified, "extract_text_from_artifact", return_value="x",
        ):
            fields = pipeline_unified._extract_artifact_fields(artifact)
        # No title / description / tags survive a bad context, but content
        # still comes through extract_text_from_artifact.
        assert "content" in fields
        assert "title" not in fields

    def test_empty_artifact_yields_empty_dict(self):
        from mantle.search.ingest import pipeline_unified

        artifact = SimpleNamespace(
            id="art-empty",
            root_id="art-empty",
            state="draft",
            created_by="user-1",
            collection_id="ws-1",
            context="{}",
            content="",
            name="",
            description="",
            content_type="text/plain",
            created_time="2026-05-09T00:00:00Z",
        )
        with patch.object(
            pipeline_unified, "extract_text_from_artifact", return_value="",
        ):
            fields = pipeline_unified._extract_artifact_fields(artifact)
        assert fields == {}


# ---------------------------------------------------------------------------
# index_artifact / index_artifacts_batch
# ---------------------------------------------------------------------------


class TestIndexArtifact:
    def test_archived_indexes_into_archived_segment(self):
        from mantle.search.ingest import pipeline_unified

        with (
            patch.object(pipeline_unified, "_sse_index_artifact") as sse,
            patch.object(pipeline_unified, "_mantle_index_artifact") as vec_mock,
            patch.object(
                pipeline_unified, "extract_text_from_artifact",
                return_value="archived text",
            ),
        ):
            sse.return_value = pipeline_unified.ARM_WRITTEN
            vec_mock.return_value = pipeline_unified.ARM_WRITTEN
            outcome = pipeline_unified.index_artifact(_archived_artifact(), "ws-1")
        # Archived is no longer skipped — it indexes into its own segment.
        assert outcome.sse == pipeline_unified.ARM_WRITTEN
        assert outcome.vector == pipeline_unified.ARM_WRITTEN
        assert not outcome.failed
        sse.assert_called_once()
        vec_mock.assert_called_once()
        assert sse.call_args.kwargs["segment"] == "archived"
        assert vec_mock.call_args.kwargs["segment"] == "archived"

    def test_skips_artifact_with_no_fields(self):
        from mantle.search.ingest import pipeline_unified

        artifact = _committed_artifact()
        with (
            patch.object(
                pipeline_unified, "_extract_artifact_fields", return_value={},
            ),
            patch.object(pipeline_unified, "_sse_index_artifact") as sse,
            patch.object(pipeline_unified, "_mantle_index_artifact") as vec_mock,
        ):
            outcome = pipeline_unified.index_artifact(artifact, "ws-1")
        # Nothing to index is a SKIP, not a failure — the distinction is the point
        # of IndexOutcome. Both must be visible; neither may read as "written".
        assert outcome.wrote_nothing
        assert not outcome.failed
        assert outcome.sse == outcome.vector == pipeline_unified.ARM_SKIPPED
        sse.assert_not_called()
        vec_mock.assert_not_called()

    def test_calls_both_arms_for_committed_artifact(self):
        from mantle.search.ingest import pipeline_unified

        artifact = _committed_artifact()
        with (
            patch.object(pipeline_unified, "_sse_index_artifact") as sse,
            patch.object(pipeline_unified, "_mantle_index_artifact") as vec_mock,
            patch.object(
                pipeline_unified, "extract_text_from_artifact",
                return_value="hello world",
            ),
        ):
            sse.return_value = pipeline_unified.ARM_WRITTEN
            vec_mock.return_value = pipeline_unified.ARM_WRITTEN
            outcome = pipeline_unified.index_artifact(artifact, "ws-1")
        assert not outcome.failed and not outcome.wrote_nothing
        sse.assert_called_once()
        vec_mock.assert_called_once()
        # Routed to the committed segment.
        assert sse.call_args.kwargs["segment"] == "committed"
        assert vec_mock.call_args.kwargs["segment"] == "committed"
        # Both arms get the same fields dict.
        assert sse.call_args[0][2] == vec_mock.call_args[0][2]

    def test_does_not_touch_other_segments(self):
        """index_artifact writes only its state's segment — vacating another
        segment is a transition-time concern (move_artifact_segments), since a
        root can hold a committed version AND a draft at once."""
        from mantle.search.ingest import pipeline_unified

        artifact = _committed_artifact()
        with (
            patch.object(pipeline_unified, "_sse_index_artifact"),
            patch.object(pipeline_unified, "_mantle_index_artifact"),
            patch.object(pipeline_unified, "_sse_remove_artifact") as sse_rm,
            patch.object(pipeline_unified, "_mantle_remove_artifact") as vec_rm,
            patch.object(
                pipeline_unified, "extract_text_from_artifact",
                return_value="hello world",
            ),
        ):
            pipeline_unified.index_artifact(artifact, "ws-1")
        sse_rm.assert_not_called()
        vec_rm.assert_not_called()


class TestIndexArtifactsBatch:
    def test_iterates_all_states(self):
        from mantle.search.ingest import pipeline_unified

        committed = _committed_artifact()
        archived = _archived_artifact()
        with (
            patch.object(pipeline_unified, "_sse_index_artifact") as sse,
            patch.object(pipeline_unified, "_mantle_index_artifact") as vec_mock,
            patch.object(
                pipeline_unified, "extract_text_from_artifact",
                return_value="hello world",
            ),
        ):
            ok = pipeline_unified.index_artifacts_batch(
                [committed, archived], "ws-1",
            )
        assert ok is True
        # Both states indexed now, each into its own segment.
        assert sse.call_count == 2
        assert vec_mock.call_count == 2
        segs = {c.kwargs["segment"] for c in sse.call_args_list}
        assert segs == {"committed", "archived"}


class TestMoveArtifactSegments:
    def test_removes_root_from_named_segments_only(self):
        from mantle.search.ingest import pipeline_unified

        artifact = _committed_artifact()  # root_id="art-1", collection_id="ws-1"
        with (
            patch("mantle.services.dependencies.get_store_db", return_value=iter([object()])),
            patch("mantle.search.mantle.wiring.build_indexer", return_value=object()),
            patch("mantle.search.mantle.principal.resolve_cell_principal", return_value="prin-1"),
            patch.object(pipeline_unified, "_mantle_remove_artifact") as vec_rm,
            patch.object(pipeline_unified, "_sse_remove_artifact") as sse_rm,
        ):
            pipeline_unified.move_artifact_segments(artifact, "ws-1", remove_from=["draft"])
        vec_rm.assert_called_once_with("prin-1", "ws-1", "art-1", segment="draft")
        sse_rm.assert_called_once_with("prin-1", "art-1", segment="draft")

    def test_noop_when_index_backend_unavailable(self):
        from mantle.search.ingest import pipeline_unified

        artifact = _committed_artifact()
        with (
            patch("mantle.services.dependencies.get_store_db", return_value=iter([object()])),
            patch("mantle.search.mantle.wiring.build_indexer", return_value=None),
            patch.object(pipeline_unified, "_mantle_remove_artifact") as vec_rm,
            patch.object(pipeline_unified, "_sse_remove_artifact") as sse_rm,
        ):
            pipeline_unified.move_artifact_segments(
                artifact, "ws-1", remove_from=["draft", "archived"],
            )
        # Gated on the backend: no principal query, no removes.
        vec_rm.assert_not_called()
        sse_rm.assert_not_called()


class TestDeleteArtifact:
    def test_removes_from_every_segment(self):
        from mantle.search.ingest import pipeline_unified

        with (
            patch.object(pipeline_unified, "_mantle_remove_artifact") as vec_mock,
            patch.object(pipeline_unified, "_sse_remove_artifact") as sse,
        ):
            ok = pipeline_unified.delete_artifact_from_index(
                "v-1", root_id="art-1",
                principal_id="user-1", collection_id="ws-1",
            )
        assert ok is True
        # A hard delete purges all three segments.
        assert vec_mock.call_count == 3
        assert sse.call_count == 3
        assert {c.kwargs["segment"] for c in vec_mock.call_args_list} == {
            "committed", "draft", "archived",
        }
        for c in vec_mock.call_args_list:
            assert c.args == ("user-1", "ws-1", "art-1")
        for c in sse.call_args_list:
            assert c.args == ("user-1", "art-1")

    def test_deletion_actually_removes_the_postings_it_reports_removing(self):
        """The de-index principal must be the SAME one the write path used.

        This asserts the EFFECT — that the postings are gone — not the return value.
        The defect it pins (`resolve_cell_principal(get_store_db(), ...)` with no
        ``next(...)``) returned ``True`` while removing nothing, so any test asserting
        truthiness passed against the bug.

        ``ws-child``'s origin root is ``origin-root-1``, so the two differ: the write
        path keys postings under the origin root, and a de-index that falls back to
        ``collection_id`` targets a path that was never written.
        """
        from mantle.search.ingest import pipeline_unified

        sse_index = _FakeSseIndex()
        artifact = _committed_artifact()

        with (
            patch("mantle.services.dependencies.get_store_db", side_effect=_fake_get_store_db),
            patch("mantle.db.backend.get_origin_root", side_effect=_fake_get_origin_root),
            patch("mantle.search.mantle.wiring.build_sse_indexer", return_value=sse_index),
            patch("mantle.search.mantle.wiring.build_indexer", return_value=None),
            # Indexing is grant-gated, so the write needs an acting identity — in
            # production supplied by the router (or `system_acting_context` for
            # background work). `origin-root-1` is the principal being written into.
            acting_as("origin-root-1", principal_type="user"),
        ):
            # WRITE through the real path — it resolves the principal from a real handle.
            pipeline_unified._sse_index_artifact(
                artifact, "ws-child", {"content": "the secret the user deleted"},
            )
            assert list(sse_index.postings) == [("origin-root-1", "art-1")], (
                "precondition: the write path must key postings under the ORIGIN ROOT"
            )

            # DELETE through the real path, supplying only what the seven callers supply.
            ok = pipeline_unified.delete_artifact_from_index(
                "art-1", "art-1", collection_id="ws-child",
            )

        assert sse_index.postings == {}, (
            "delete_artifact_from_index left postings behind: %r. The deleted artifact's "
            "content is still retrievable through the owning principal's own search path."
            % (sse_index.postings,)
        )
        # Only meaningful once the effect above holds — reporting success is correct
        # ONLY because the removal happened.
        assert ok is True

    def test_no_owner_skips_both_arms(self):
        from mantle.search.ingest import pipeline_unified

        with (
            patch.object(pipeline_unified, "_mantle_remove_artifact") as vec_mock,
            patch.object(pipeline_unified, "_sse_remove_artifact") as sse,
        ):
            ok = pipeline_unified.delete_artifact_from_index("v-1")
        # Skipping both arms without identity is correct and still asserted below. What was WRONG
        # was calling that success: every production caller omits principal_id/collection_id, so
        # this branch is the ONLY branch they ever take — a hard delete purged the lattice and S3, this
        # returned True, and the artifact's chunks and postings (with their plaintext text) stayed
        # searchable forever. The test asserted the lie, so the assertion is corrected rather than
        # the code being softened to match it; the two behavioural assertions are unchanged.
        assert ok is False
        vec_mock.assert_not_called()
        sse.assert_not_called()


class TestGetArtifactEmbeddings:
    def test_returns_only_this_artifacts_chunks_ordered(self):
        from mantle.search.ingest import pipeline_unified

        artifact = _committed_artifact()   # root_id="art-1", collection_id="ws-1", committed
        idx = MagicMock()
        idx.collection_chunks.return_value = [
            {"artifact_id": "art-1", "chunk_id": 1, "embedding": [0.2], "model_id": "m"},
            {"artifact_id": "other", "chunk_id": 0, "embedding": [9.9]},
            {"artifact_id": "art-1", "chunk_id": 0, "embedding": [0.1], "model_id": "m"},
        ]
        with (
            patch("mantle.services.dependencies.get_store_db", return_value=iter([object()])),
            patch("mantle.search.mantle.wiring.build_indexer", return_value=idx),
            patch("mantle.search.mantle.principal.resolve_cell_principal", return_value="prin"),
        ):
            # Reading vectors back is grant-gated like any other key use, so the
            # caller needs an identity — in production this comes from the router.
            with acting_as("prin", principal_type="user"):
                chunks = pipeline_unified.get_artifact_embeddings(artifact, "ws-1")
        # only art-1's chunks, sorted by chunk_id
        assert [c["chunk_id"] for c in chunks] == [0, 1]
        assert chunks[0]["embedding"] == [0.1]

    def test_empty_when_vector_arm_unavailable(self):
        from mantle.search.ingest import pipeline_unified
        with (
            patch("mantle.services.dependencies.get_store_db", return_value=iter([object()])),
            patch("mantle.search.mantle.wiring.build_indexer", return_value=None),
        ):
            assert pipeline_unified.get_artifact_embeddings(_committed_artifact(), "ws-1") == []


# ---------------------------------------------------------------------------
# What the pipeline REPORTS about what it did
# ---------------------------------------------------------------------------
#
# THE DEFECT THESE PIN (observed on a live boot, 2026-07-31): both arms swallow
# their own exceptions by design — one arm must not lose the other, and neither
# must fail the commit — and `index_artifact` then returned a flat `True` anyway.
# A reindex whose every SSE write was refused with GrantDenied reported
# `{"indexed": 5, "failed": 0}` over an empty index.


class TestIndexOutcomeReporting:
    def test_a_failed_arm_is_not_reported_as_a_write(self):
        from mantle.search.ingest import pipeline_unified

        with (
            patch.object(pipeline_unified, "_sse_index_artifact") as sse,
            patch.object(pipeline_unified, "_mantle_index_artifact") as vec_mock,
            patch.object(
                pipeline_unified, "extract_text_from_artifact", return_value="text",
            ),
        ):
            sse.return_value = pipeline_unified.ARM_FAILED      # e.g. GrantDenied
            vec_mock.return_value = pipeline_unified.ARM_SKIPPED  # e.g. no AnchorSet
            outcome = pipeline_unified.index_artifact(_committed_artifact(), "ws-1")

        assert outcome.failed
        assert bool(outcome) is False
        assert outcome.wrote_nothing, "nothing was written; must not count as indexed"

    def test_a_skipped_arm_is_not_reported_as_a_failure(self):
        """Otherwise the failure count stops meaning anything and gets ignored."""
        from mantle.search.ingest import pipeline_unified

        with (
            patch.object(pipeline_unified, "_sse_index_artifact") as sse,
            patch.object(pipeline_unified, "_mantle_index_artifact") as vec_mock,
            patch.object(
                pipeline_unified, "extract_text_from_artifact", return_value="text",
            ),
        ):
            sse.return_value = pipeline_unified.ARM_WRITTEN
            vec_mock.return_value = pipeline_unified.ARM_SKIPPED
            outcome = pipeline_unified.index_artifact(_committed_artifact(), "ws-1")

        assert not outcome.failed
        assert not outcome.wrote_nothing
        assert bool(outcome) is True

    def test_batch_returns_false_when_an_artifact_failed(self):
        from mantle.search.ingest import pipeline_unified

        with (
            patch.object(pipeline_unified, "_sse_index_artifact") as sse,
            patch.object(pipeline_unified, "_mantle_index_artifact") as vec_mock,
            patch.object(
                pipeline_unified, "extract_text_from_artifact", return_value="text",
            ),
        ):
            sse.return_value = pipeline_unified.ARM_FAILED
            vec_mock.return_value = pipeline_unified.ARM_FAILED
            ok = pipeline_unified.index_artifacts_batch(
                [_committed_artifact()], "ws-1",
            )

        # FAILURE MODE: this returned `True` unconditionally, so a bulk index in
        # which every artifact failed produced the same value as a clean run.
        assert ok is False


class TestPlatformTrustConfigIsNotASearchTarget:
    """Issuer artifacts are platform trust config: system-owned, deliberately
    grantless, and carrying no collection (``create_issuer_artifact`` sets
    ``collection_id=""``). The bulk reindex's root-artifact convention handed each
    one its OWN id as a collection, so the encrypted-search layer asked the ledger
    whether the system principal may write into a principal that is really an
    issuer's id — a question whose answer is permanently no."""

    @staticmethod
    def _issuer_artifact():
        art = _committed_artifact()
        art.content_type = "application/vnd.agience.issuer+json"
        art.collection_id = ""
        return art

    def test_issuer_artifact_reaches_neither_arm(self):
        from mantle.search.ingest import pipeline_unified

        artifact = self._issuer_artifact()
        with (
            patch.object(pipeline_unified, "_sse_index_artifact") as sse,
            patch.object(pipeline_unified, "_mantle_index_artifact") as vec_mock,
            patch.object(
                pipeline_unified, "extract_text_from_artifact", return_value="text",
            ),
        ):
            outcome = pipeline_unified.index_artifact(artifact, artifact.id)

        sse.assert_not_called()
        vec_mock.assert_not_called()
        assert outcome.wrote_nothing and not outcome.failed

    def test_ordinary_artifacts_are_still_indexed(self):
        """NEGATIVE CONTROL: the exclusion must be narrow, or it silently empties
        the index and every assertion above still passes."""
        from mantle.search.ingest import pipeline_unified

        assert pipeline_unified.is_indexable(_committed_artifact())
        assert not pipeline_unified.is_indexable(self._issuer_artifact())
