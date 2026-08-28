"""Tests for the unified indexing pipeline.

The pipeline writes to MANTLE vector cells and MANTLE-SSE posting lists only.

These tests cover the public surface:

- ``_extract_artifact_fields`` produces the long-form per-field text
  dict the SSE indexer wants.
- ``index_artifact`` calls the SSE + MANTLE hooks routed to the segment
  for the artifact's state (committed/draft/archived — each a separate
  index) and does not touch the other segments; vacating a segment on a
  state transition is ``move_artifact_segments``'s job.
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
    """Raises on anything that is not a real handle — a generator object is not one.

    The real ``db.store.get_origin_root`` runs an AQL query and would raise on a
    generator too; making that explicit here is what lets the test see the fallback
    in ``resolve_cell_principal`` instead of silently agreeing with it.
    """
    if db is not _FAKE_DB:
        raise TypeError("not a database handle: %r" % (db,))
    # ws-child is not its own origin root — that gap is where the defect lives.
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
        assert fields["content"] == "hello world"
        # A tag is group MEMBERSHIP now, not a key in the context blob (§112) — the words of the
        # groups this artifact belongs to, with the scheme dropped.
        assert fields["tags"] == "ws"

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
        # Archived indexes into its own segment, the same as any other state.
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
        # Nothing to index is a skip, not a failure — the distinction is the point
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
        """The de-index principal must be the same one the write path used.

        This asserts the effect — that the postings are gone — not the return value. A missing
        ``next(...)`` on ``resolve_cell_principal(get_store_db(), ...)`` would return ``True``
        while removing nothing, so a test asserting only truthiness would not catch that.

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
            # Write through the real path — it resolves the principal from a real handle.
            #
            # The text goes in an offer field, because that is what is indexed. The lexical arm
            # indexes the offer (`_OFFER_FIELDS`: title, description, tags) and never the raw body —
            # a body's term count is unbounded and posting lists are read-modify-write, so indexing
            # it makes write cost
            # scale with the corpus. A fixture passing only `content` now filters to nothing and
            # returns ARM_SKIPPED, so the write under test never happens and this fails on its own
            # precondition rather than on the behaviour it is checking.
            pipeline_unified._sse_index_artifact(
                artifact, "ws-child", {"title": "the secret the user deleted"},
            )
            assert list(sse_index.postings) == [("origin-root-1", "art-1")], (
                "precondition: the write path must key postings under the ORIGIN ROOT"
            )

            # Delete through the real path, supplying only what the seven callers supply.
            ok = pipeline_unified.delete_artifact_from_index(
                "art-1", "art-1", collection_id="ws-child",
            )

        assert sse_index.postings == {}, (
            "delete_artifact_from_index left postings behind: %r. The deleted artifact is "
            "still retrievable through the owning principal's own search path."
            % (sse_index.postings,)
        )
        # Only meaningful once the effect above holds — reporting success is correct
        # only because the removal happened.
        assert ok is True

    def test_no_owner_skips_both_arms(self):
        from mantle.search.ingest import pipeline_unified

        with (
            patch.object(pipeline_unified, "_mantle_remove_artifact") as vec_mock,
            patch.object(pipeline_unified, "_sse_remove_artifact") as sse,
        ):
            ok = pipeline_unified.delete_artifact_from_index("v-1")
        # Skipping both arms without identity is correct, and the return value must say so: a
        # hard delete that purges the lattice and S3 must not report success while the artifact's
        # chunks and postings (with their plaintext text) remain searchable.
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
# What the pipeline reports about what it did
# ---------------------------------------------------------------------------
#
# The property these tests pin: both arms swallow their own exceptions by design — one arm must
# not lose the other, and neither must fail the commit — so `index_artifact` must not report a
# flat success regardless. A reindex whose every SSE write raised `GrantDenied` must not report
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

        # Fails if this returns `True` unconditionally, so a bulk index in which
        # every artifact failed produces the same value as a clean run.
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
        """Negative control: the exclusion must be narrow, or it silently empties
        the index and every assertion above still passes."""
        from mantle.search.ingest import pipeline_unified

        assert pipeline_unified.is_indexable(_committed_artifact())
        assert not pipeline_unified.is_indexable(self._issuer_artifact())


class TestASecretIsNotASearchTarget:
    """A credential's value IS its content — an API key, a client secret, a refresh
    token. Indexing it tokenizes that value into the SSE posting lists and lets a
    recall hit hydrate the decrypted value into a result.

    Nothing about that leaks: the cells are encrypted at rest and every posting and
    hydration is cut by the same light cone that guards the read path. It is a wider
    surface — a secret reachable by SEARCH as well as by direct read, and a second
    representation of the plaintext in a second store. The index does not carry one.
    """

    @staticmethod
    def _credential_artifact():
        from mantle.services.bootstrap_types import CREDENTIAL_CONTENT_TYPE

        art = _committed_artifact()
        art.content_type = CREDENTIAL_CONTENT_TYPE
        return art

    def test_credential_content_type_is_non_indexable(self):
        from mantle.services.bootstrap_types import CREDENTIAL_CONTENT_TYPE
        from mantle.search.ingest import pipeline_unified

        assert CREDENTIAL_CONTENT_TYPE in pipeline_unified.NON_INDEXABLE_CONTENT_TYPES
        assert not pipeline_unified.is_indexable(self._credential_artifact())

    def test_credential_reaches_neither_arm(self):
        """A credential sits in an ordinary collection, unlike an issuer artifact —
        so it is the gate, not the missing scope, that has to stop it."""
        from mantle.search.ingest import pipeline_unified

        artifact = self._credential_artifact()
        assert artifact.collection_id, "the point of this case is a normal collection"

        with (
            patch.object(pipeline_unified, "_sse_index_artifact") as sse,
            patch.object(pipeline_unified, "_mantle_index_artifact") as vec_mock,
            patch.object(
                pipeline_unified, "extract_text_from_artifact",
                return_value="sk-live-not-in-the-index",
            ),
        ):
            outcome = pipeline_unified.index_artifact(artifact, artifact.id)

        sse.assert_not_called()
        vec_mock.assert_not_called()
        assert outcome.wrote_nothing and not outcome.failed


class TestALexiconEntrysNamesAreIndexed:
    """A synset's `lemmas` are the words that MEAN it, and for a dictionary entry that is the
    offer — the record exists to say these words mean this.

    Only the first lemma becomes the title, and the two lexicons order them differently, so the
    OEWN oxygen synset is titled `O` while its Princeton twin is titled `oxygen`. The OEWN copy is
    the one in this store's SSE index, and its gloss never says the word either. Measured before
    this rule existed, `recall("what is oxygen")` did not narrow to that artifact AT ALL and
    answered `LOX / air / artificial blood` — every hyponym carries `oxygen` inside its own title,
    so the hyponyms were findable and the concept was not.
    """

    @staticmethod
    def _entry(content_type, lemmas):
        return SimpleNamespace(
            id="wn-oewn-14672278-n",
            root_id="wn-oewn-14672278-n",
            state="committed",
            created_by="ingest",
            collection_id="stage.0.lexicon",
            context="",
            content="a nonmetallic bivalent element",
            name="O",
            description="",
            lemmas=lemmas,
            content_type=content_type,
            created_time="2026-05-09T00:00:00Z",
        )

    def test_a_synsets_lemmas_are_indexed_as_its_tags(self):
        from mantle.search.ingest import pipeline_unified

        artifact = self._entry("text/x-wordnet", ["o", "atomic number 8", "oxygen"])
        with patch.object(
            pipeline_unified, "extract_text_from_artifact", return_value="a gloss",
        ):
            fields = pipeline_unified._extract_artifact_fields(artifact)
        assert "oxygen" in fields.get("tags", ""), (
            "the word that names this concept must reach the index — %r" % (fields,)
        )

    def test_prose_lemmas_are_NOT_indexed(self):
        """`lemmas` is not one thing in this corpus.

        On a wiki artifact it is key terms extracted FROM the body by `astra/doc_index`. Indexing
        those would be indexing the content, which this pipeline is built not to do. So the rule
        reads the TYPE, never merely whether the field is present — one field, two meanings, two
        writers, which is the shape of every other defect this corpus has produced.
        """
        from mantle.search.ingest import pipeline_unified

        artifact = self._entry("text/markdown", ["gorilla", "monsoon", "wrestling"])
        with patch.object(
            pipeline_unified, "extract_text_from_artifact", return_value="a body",
        ):
            fields = pipeline_unified._extract_artifact_fields(artifact)
        assert "wrestling" not in fields.get("tags", ""), (
            "body-extracted terms must not become tags — %r" % (fields,)
        )

    def test_the_lemmas_survive_alongside_group_membership(self):
        """The lemmas are ADDED to the artifact's groups, never substituted for them.

        A tag is an edge to another artifact, so a lexicon entry's names are added to its
        membership rather than to a `tags` key in the context blob, which nothing reads. The
        property under test: a synset's other names must reach the index, because its title is only
        the first of them and its gloss often never says the word at all.
        """
        from mantle.search.ingest import pipeline_unified

        artifact = self._entry("text/x-wordnet", ["oxygen"])
        artifact.collection_id = "collection:stage.0.lexicon"
        with patch.object(
            pipeline_unified, "extract_text_from_artifact", return_value="a gloss",
        ):
            fields = pipeline_unified._extract_artifact_fields(artifact)
        tags = fields.get("tags", "")
        assert "oxygen" in tags, tags                       # the entry's own other name
        assert "lexicon" in tags, tags                      # the group it belongs to
        assert "collection" not in tags, tags               # the scheme is not a tag


class TestAMergedConceptIsIndexable:
    """A colimit of synsets names itself the way the synsets do, so it is findable like them.

    `ember.consolidate.colimit` merges the synsets that mean one thing into a single
    `application/x-concept` object carrying their union. Measured 2026-08-24 on 71/home: all 5,484
    of them carry `lemmas` and NONE of `title` / `name` / `description`, with `content` empty — so
    `_extract_artifact_fields` returned `{}` and every one was skipped as "no analyzable fields".

    That is COMPACTIFICATION §5 exactly inverted: consolidate into a heavier object, then index
    only the lighter members, and a search can never reach the thing the merge produced. It is
    also silent — the pipeline reports a skip, not a failure — so nothing anywhere said the
    consolidation had made 5,484 unfindable objects.
    """

    @staticmethod
    def _colimit(lemmas, content_type="application/x-concept",
                 colimit_of=("wn-decade.n.01", "wn-oewn-15174893-n")):
        """What the consolidator actually writes: names, no title, no body."""
        return SimpleNamespace(
            id="concept-be603d8ac3dd0185fcf653e18a8dc72d",
            root_id="concept-be603d8ac3dd0185fcf653e18a8dc72d",
            state="committed",
            created_by="op.consolidate.colimit",
            collection_id="stage.0.lexicon",
            context="",
            content="",
            name=None,
            description=None,
            lemmas=lemmas,
            colimit_of=list(colimit_of) if colimit_of else None,
            content_type=content_type,
            created_time="2026-08-24T00:00:00Z",
        )

    @staticmethod
    def _conceptnet_term(word, lemmas):
        """A ConceptNet 5.7 term: the SAME content type, and its lemmas are its title split.

        Measured 2026-08-24: 1,165,110 of these on 71/home against 5,484 colimits, all sharing
        `application/x-concept`. They carry a title and a description of their own, so nothing here
        needs their lemmas — and promoting them would put `hour` and `clock` in the tags of
        `12 hour clock`, which is the title indexed twice.
        """
        return SimpleNamespace(
            id="cn-12_hour_clock",
            root_id="cn-12_hour_clock",
            state="committed",
            created_by="ingest.conceptnet",
            collection_id="stage.2.world",
            context="",
            content="%s — ConceptNet 5.7 concept" % word,
            name=word,
            description="the concept %s" % word,
            lemmas=lemmas,
            colimit_of=None,
            content_type="application/x-concept",
            created_time="2026-08-24T00:00:00Z",
        )

    def test_a_colimit_with_only_lemmas_is_still_indexable(self):
        from mantle.search.ingest import pipeline_unified

        artifact = self._colimit(["decade", "decennary", "decennium"])
        with patch.object(
            pipeline_unified, "extract_text_from_artifact", return_value="",
        ):
            fields = pipeline_unified._extract_artifact_fields(artifact)
        assert fields, (
            "a merged concept with names must reach the index; empty fields is the skip that "
            "made 5,484 of them unfindable"
        )

    def test_the_title_is_the_first_lemma(self):
        """The synset rule applied, not restated: only the first of them becomes the title.

        A synset gets that title from its source; a colimit has no source to get one from, so it
        is derived here. `word` is not read — the consolidator writes it, it is that same first
        lemma, and a second carrier is a second thing to keep true. It does not survive
        `Artifact.from_dict` either, which is the form this function is handed.
        """
        from mantle.search.ingest import pipeline_unified

        artifact = self._colimit(["decade", "decennary", "decennium"])
        with patch.object(
            pipeline_unified, "extract_text_from_artifact", return_value="",
        ):
            fields = pipeline_unified._extract_artifact_fields(artifact)
        assert fields.get("title") == "decade", fields

    def test_every_name_the_merge_absorbed_reaches_the_index(self):
        """The point of the merge is that these words mean the same thing.

        Indexing only the title would make the colimit reachable by one of its names and leave the
        rest answered by the members it consolidated — which is the state this replaced.
        """
        from mantle.search.ingest import pipeline_unified

        artifact = self._colimit(["decade", "decennary", "decennium"])
        with patch.object(
            pipeline_unified, "extract_text_from_artifact", return_value="",
        ):
            fields = pipeline_unified._extract_artifact_fields(artifact)
        tags = fields.get("tags", "")
        assert "decennary" in tags, tags
        assert "decennium" in tags, tags

    def test_a_prose_artifacts_lemmas_are_still_not_indexed(self):
        """The rule is about the RECORD, not about the field being present.

        On a wiki artifact `lemmas` holds terms `astra/doc_index` pulled out of the body, and
        indexing those would be indexing the content.
        """
        from mantle.search.ingest import pipeline_unified

        artifact = self._colimit(["gorilla", "monsoon"], content_type="text/markdown",
                                 colimit_of=None)
        artifact.content = "a body"
        with patch.object(
            pipeline_unified, "extract_text_from_artifact", return_value="a body",
        ):
            fields = pipeline_unified._extract_artifact_fields(artifact)
        assert "monsoon" not in fields.get("tags", ""), fields
        assert fields.get("title") != "gorilla", (
            "a body term must not become the title either — %r" % (fields,)
        )

    def test_a_conceptnet_term_shares_the_type_and_is_NOT_promoted(self):
        """The content type is not the discriminator, and believing it was is the bug.

        `application/x-concept` has two writers. Keying the rule on the type promoted 1,165,110
        ConceptNet terms whose `lemmas` are their own title, split into words — putting `hour` and
        `clock` into the tags of `12 hour clock`, a record that already carries all three stems in
        its title. `colimit_of` is what separates them, because carrying it is what being a merge
        means.
        """
        from mantle.search.ingest import pipeline_unified

        artifact = self._conceptnet_term("12 hour clock", ["12", "hour", "clock"])
        with patch.object(
            pipeline_unified, "extract_text_from_artifact", return_value="a gloss",
        ):
            fields = pipeline_unified._extract_artifact_fields(artifact)
        assert fields.get("title") == "12 hour clock", fields
        assert "clock" not in fields.get("tags", ""), (
            "a ConceptNet term's lemmas are its title split up, not names it goes by — %r"
            % (fields,)
        )

    def test_a_colimit_is_recognised_by_colimit_of_and_not_by_its_type(self):
        """Same type, same field, opposite answers — decided by whether it is a merge."""
        from mantle.search.ingest import pipeline_unified

        merge = self._colimit(["decade", "decennary"])
        term = self._conceptnet_term("decade", ["decade"])
        with patch.object(
            pipeline_unified, "extract_text_from_artifact", return_value="",
        ):
            assert pipeline_unified._lemmas_are_names(merge) is True
            assert pipeline_unified._lemmas_are_names(term) is False


class TestAnArtifactAnnouncesItselfOnce:
    """One offer, not two. A description that repeats the title is not a second claim.

    Measured 2026-08-24 across 2,167,300 artifacts on 71/home: 1,175,579 carry both fields, and
    1,164,574 of those -- 99% -- have a description containing the title verbatim, in one repeated
    template (`Workspace document imported from <title>`). Nothing carries a description and no
    title. Only 11,005 descriptions in the whole corpus say something their title does not.
    """

    def test_a_template_description_is_dropped(self):
        from mantle.search.ingest import pipeline_unified

        kept = pipeline_unified._description_that_adds_something(
            "agience-build/AGENTS.md",
            "agience-build/AGENTS.md",
        )
        assert kept == "", "the same words twice is one claim, not two"

    def test_a_description_that_adds_words_is_kept(self):
        """Containment is not the test -- this one contains its title and is not a duplicate."""
        from mantle.search.ingest import pipeline_unified

        kept = pipeline_unified._description_that_adds_something(
            "oxygen", "oxygen: a nonmetallic bivalent element")
        assert "nonmetallic" in kept

    def test_a_description_with_no_title_beside_it_is_kept(self):
        from mantle.search.ingest import pipeline_unified

        assert pipeline_unified._description_that_adds_something(
            "", "a slowly moving mass of ice") == "a slowly moving mass of ice"

    def test_the_duplicate_never_reaches_the_index(self):
        """The whole point: `_fields_to_index` is what the SSE arm is handed."""
        from mantle.search.ingest import pipeline_unified

        out = pipeline_unified._fields_to_index({
            "title": "agience-build/AGENTS.md",
            "description": "agience-build/AGENTS.md",
            "tags": "workspace",
        })
        assert "description" not in out, out
        assert out.get("title") == "agience-build/AGENTS.md"
        assert out.get("tags") == "workspace"

    def test_a_real_offer_still_reaches_the_index(self):
        from mantle.search.ingest import pipeline_unified

        out = pipeline_unified._fields_to_index({
            "title": "oxygen",
            "description": "a nonmetallic bivalent element",
        })
        assert out.get("description") == "a nonmetallic bivalent element", out
