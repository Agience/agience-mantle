"""Tests for the unified indexing pipeline (post-OpenSearch retirement).

After Step 2.6.9 part 2 the pipeline writes to MANTLE vector cells +
MANTLE-SSE posting lists. The OpenSearch BM25 path and its
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
# _extract_artifact_fields
# ---------------------------------------------------------------------------


class TestExtractArtifactFields:
    def test_extracts_title_description_tags_content(self):
        from search.ingest import pipeline_unified

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
        from search.ingest import pipeline_unified

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
        from search.ingest import pipeline_unified

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
        from search.ingest import pipeline_unified

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
        from search.ingest import pipeline_unified

        with (
            patch.object(pipeline_unified, "_sse_index_artifact") as sse,
            patch.object(pipeline_unified, "_mantle_index_artifact") as vec_mock,
            patch.object(
                pipeline_unified, "extract_text_from_artifact",
                return_value="archived text",
            ),
        ):
            ok = pipeline_unified.index_artifact(_archived_artifact(), "ws-1")
        # Archived is no longer skipped — it indexes into its own segment.
        assert ok is True
        sse.assert_called_once()
        vec_mock.assert_called_once()
        assert sse.call_args.kwargs["segment"] == "archived"
        assert vec_mock.call_args.kwargs["segment"] == "archived"

    def test_skips_artifact_with_no_fields(self):
        from search.ingest import pipeline_unified

        artifact = _committed_artifact()
        with (
            patch.object(
                pipeline_unified, "_extract_artifact_fields", return_value={},
            ),
            patch.object(pipeline_unified, "_sse_index_artifact") as sse,
            patch.object(pipeline_unified, "_mantle_index_artifact") as vec_mock,
        ):
            ok = pipeline_unified.index_artifact(artifact, "ws-1")
        assert ok is False
        sse.assert_not_called()
        vec_mock.assert_not_called()

    def test_calls_both_arms_for_committed_artifact(self):
        from search.ingest import pipeline_unified

        artifact = _committed_artifact()
        with (
            patch.object(pipeline_unified, "_sse_index_artifact") as sse,
            patch.object(pipeline_unified, "_mantle_index_artifact") as vec_mock,
            patch.object(
                pipeline_unified, "extract_text_from_artifact",
                return_value="hello world",
            ),
        ):
            ok = pipeline_unified.index_artifact(artifact, "ws-1")
        assert ok is True
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
        from search.ingest import pipeline_unified

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
        from search.ingest import pipeline_unified

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
        from search.ingest import pipeline_unified

        artifact = _committed_artifact()  # root_id="art-1", collection_id="ws-1"
        with (
            patch("services.dependencies.get_arango_db", return_value=iter([object()])),
            patch("search.mantle.wiring.build_indexer", return_value=object()),
            patch("search.mantle.principal.resolve_cell_principal", return_value="prin-1"),
            patch.object(pipeline_unified, "_mantle_remove_artifact") as vec_rm,
            patch.object(pipeline_unified, "_sse_remove_artifact") as sse_rm,
        ):
            pipeline_unified.move_artifact_segments(artifact, "ws-1", remove_from=["draft"])
        vec_rm.assert_called_once_with("prin-1", "ws-1", "art-1", segment="draft")
        sse_rm.assert_called_once_with("prin-1", "art-1", segment="draft")

    def test_noop_when_index_backend_unavailable(self):
        from search.ingest import pipeline_unified

        artifact = _committed_artifact()
        with (
            patch("services.dependencies.get_arango_db", return_value=iter([object()])),
            patch("search.mantle.wiring.build_indexer", return_value=None),
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
        from search.ingest import pipeline_unified

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

    def test_no_owner_skips_both_arms(self):
        from search.ingest import pipeline_unified

        with (
            patch.object(pipeline_unified, "_mantle_remove_artifact") as vec_mock,
            patch.object(pipeline_unified, "_sse_remove_artifact") as sse,
        ):
            ok = pipeline_unified.delete_artifact_from_index("v-1")
        assert ok is True
        vec_mock.assert_not_called()
        sse.assert_not_called()


class TestGetArtifactEmbeddings:
    def test_returns_only_this_artifacts_chunks_ordered(self):
        from search.ingest import pipeline_unified

        artifact = _committed_artifact()   # root_id="art-1", collection_id="ws-1", committed
        idx = MagicMock()
        idx.collection_chunks.return_value = [
            {"artifact_id": "art-1", "chunk_id": 1, "embedding": [0.2], "model_id": "m"},
            {"artifact_id": "other", "chunk_id": 0, "embedding": [9.9]},
            {"artifact_id": "art-1", "chunk_id": 0, "embedding": [0.1], "model_id": "m"},
        ]
        with (
            patch("services.dependencies.get_arango_db", return_value=iter([object()])),
            patch("search.mantle.wiring.build_indexer", return_value=idx),
            patch("search.mantle.principal.resolve_cell_principal", return_value="prin"),
        ):
            chunks = pipeline_unified.get_artifact_embeddings(artifact, "ws-1")
        # only art-1's chunks, sorted by chunk_id
        assert [c["chunk_id"] for c in chunks] == [0, 1]
        assert chunks[0]["embedding"] == [0.1]

    def test_empty_when_vector_arm_unavailable(self):
        from search.ingest import pipeline_unified
        with (
            patch("services.dependencies.get_arango_db", return_value=iter([object()])),
            patch("search.mantle.wiring.build_indexer", return_value=None),
        ):
            assert pipeline_unified.get_artifact_embeddings(_committed_artifact(), "ws-1") == []
