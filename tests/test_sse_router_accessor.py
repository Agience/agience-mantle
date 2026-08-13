"""Tests for `search.mantle.sse.router_accessor.MantleSseSearchAccessor`.

Covers:

- search(SearchQuery) returns a SearchResult with the router's shape.
- Empty query → empty SearchResult with parsed metadata.
- Light-cone with no authorized contexts → empty SearchResult.
- Hydration: artifact metadata (title/description/tags/state) read from
  the lattice docs; missing docs degrade to empty fields rather than dropping.
- Embedding failure: no cosine can rank, so the narrowing's own coverage orders the set
  (and the clock does when there is no coverage either).
- Trim to query.size; total = pre-trim count.
- store_db=None and narrower=None each raise (constructor contract).
"""

from __future__ import annotations

import json
from typing import Optional

import pytest

from mantle.search.mantle.engine import MantleHit
from mantle.search.mantle.sse.narrowing import Coverage
from mantle.search.mantle.sse.router_accessor import MantleSseSearchAccessor


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeArtifactStore:
    def __init__(self, docs: dict[str, dict],
                 versions: Optional[dict[str, list[dict]]] = None) -> None:
        self._docs = docs
        #: `{root_id: [version rows, oldest first]}`. Empty for most tests, which is the
        #: `id == root_id` store every artifact written without a container produces — and
        #: the case where hydration by root id and hydration by lineage are the same read.
        self._versions = versions or {}

    def get_artifact(self, artifact_id: str) -> Optional[dict]:
        return self._docs.get(artifact_id)

    def versions_of(self, root_id: str) -> list[dict]:
        return list(self._versions.get(root_id, []))


class _FakeGraph:
    def edges_of(self, node, label=None, direction="out", limit=None) -> list:
        return []


class _FakeStoreDB:
    """Lattice-shaped store handle: docs in `db.artifacts`, edges in `db.graph`
    (no edges — every artifact is its own origin root)."""

    def __init__(self, docs: Optional[dict[str, dict]] = None,
                 versions: Optional[dict[str, list[dict]]] = None) -> None:
        self.artifacts = _FakeArtifactStore(docs or {}, versions)
        self.graph = _FakeGraph()


class _FakeLightCone:
    def __init__(self, authorized: Optional[list[str]] = None) -> None:
        self._authorized = authorized or []

    def resolve(
        self,
        principal_id: str,
        *,
        action: str = "read",
        principal_type: str = "user",
    ) -> set[str]:
        return set(self._authorized)


class _FakeNarrower:
    """Stand-in for `TokenNarrower` — records what it was asked, answers with coverage.

    `matches=None` reproduces the "nothing to look up" answer (`lookup_for` returning
    `None`), which is deliberately not the same as narrowing to the empty set.

    The answer is the `{artifact_id: Coverage}` mapping the real narrower returns: its keys
    are what the resolver meets, its values are what an unranked recall orders by. `stems=1`
    for every match unless `coverage` names something else, since most cases here are about
    the vector path rather than about the counts.
    """

    def __init__(self, matches: Optional[set[str]] = None,
                 coverage: Optional[dict] = None) -> None:
        self.matches = matches
        self.coverage = coverage or {}
        self.calls: list[str] = []

    def lookup_for(self, query_text, request):
        self.calls.append(query_text)
        if self.matches is None:
            return None
        return lambda pairs: {
            a: self.coverage.get(a, Coverage(1, 0)) for a in self.matches
        }


class _FakeRanker:
    """Stand-in for `MantleQueryEngine` — records inputs, returns canned MantleHits."""

    def __init__(self, hits: Optional[list[MantleHit]] = None, *, raises=None) -> None:
        self.calls: list[dict] = []
        self._hits = hits or []
        self._raises = raises

    def search(self, query_embedding, contexts, request=None, *, top_k,
               authorized_artifacts=None):
        self.calls.append({
            "query_embedding": list(query_embedding),
            "contexts": list(contexts),
            "top_k": top_k,
            "authorized_artifacts": (
                None if authorized_artifacts is None else set(authorized_artifacts)
            ),
        })
        if self._raises is not None:
            raise self._raises
        return list(self._hits)


class _FakeEmbeddings:
    """Returns a deterministic embedding (or raises if requested)."""

    def __init__(self, *, raise_on_call: bool = False) -> None:
        self.raise_on_call = raise_on_call
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.raise_on_call:
            raise RuntimeError("embedder down")
        return [[0.1, 0.2, 0.3] for _ in texts]


def _accessor(*, lightcone=None, store_db=None, embeddings=None,
              narrower=None, ranker=None, segment="committed") -> MantleSseSearchAccessor:
    """An accessor wired the way `build_sse_search_accessor` wires one: a narrower and a
    ranker, and nothing else — `candidates()` runs on the same two."""
    return MantleSseSearchAccessor(
        lightcone if lightcone is not None else _FakeLightCone(),
        store_db=store_db if store_db is not None else _FakeStoreDB(),
        embeddings=embeddings if embeddings is not None else _FakeEmbeddings(),
        narrower=narrower if narrower is not None else _FakeNarrower(),
        ranker=ranker if ranker is not None else _FakeRanker(),
        segment=segment,
    )


def _make_query(
    query_text: str = "alpha", *,
    user_id: str = "user-1", size: int = 20, sort: str = "relevance",
):
    """Build a SearchQuery with the fields the router constructs.

    Imported lazily so the test module can be collected even if
    accessor.search_accessor's dependency tree is heavy."""
    from mantle.search.types import SearchQuery
    return SearchQuery(
        query_text=query_text,
        user_id=user_id,
        size=size,
        sort=sort,
    )


def _make_artifact_doc(artifact_id, *, title="", description="", tags=None,
                      content="", state="committed", modified="2025-01-01T00:00:00Z") -> dict:
    return {
        "id": artifact_id,
        "root_id": artifact_id,
        "context": json.dumps({
            "title": title,
            "description": description,
            "tags": tags or [],
        }),
        "content": content,
        "state": state,
        "created_by": "user-1",
        "principal_id": "user-1",
        "modified_time": modified,
    }


def _make_hit(
    artifact_id, score=0.9, *,
    collection_id="col-1", principal_id="user-1", chunk_id=0,
) -> MantleHit:
    return MantleHit(
        artifact_id=artifact_id,
        chunk_id=chunk_id,
        score=score,
        principal_id=principal_id,
        collection_id=collection_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSearchEmptyShortCircuits:
    def test_empty_query_returns_empty_result(self):
        ranker = _FakeRanker()
        acc = _accessor(ranker=ranker)
        from mantle.search.types import SearchResult
        result = acc.search(_make_query(""))
        assert isinstance(result, SearchResult)
        assert result.hits == []
        assert result.total == 0
        # Nothing ranked, and nothing was asked to.
        assert result.ordering == "recency"
        assert ranker.calls == []

    def test_no_authorized_contexts_returns_empty(self):
        ranker = _FakeRanker(hits=[_make_hit("art-1")])
        acc = _accessor(lightcone=_FakeLightCone(), ranker=ranker)
        result = acc.search(_make_query("alpha"))
        assert result.hits == []
        # Short-circuit before any key is derived or any cell is opened.
        assert ranker.calls == []

    def test_store_db_none_raises(self):
        acc = MantleSseSearchAccessor(
            _FakeLightCone(),
            store_db=None,
            embeddings=_FakeEmbeddings(),
            narrower=_FakeNarrower(),
            ranker=_FakeRanker(),
        )
        with pytest.raises(ValueError, match="store_db"):
            acc.search(_make_query("alpha"))

    def test_no_narrower_raises_rather_than_widening(self):
        """A recall with no narrower has two things it could do and both are wrong.

        Returning everything authorized widens the query into a dump of the light cone;
        returning nothing is a silent empty 200 for a query that matched. Neither is visible
        in the response, so the accessor refuses instead. `build_sse_search_accessor` cannot
        produce this state — it declines to build an accessor without a posting store.
        """
        acc = MantleSseSearchAccessor(
            _FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB({"art-1": _make_artifact_doc("art-1")}),
            embeddings=_FakeEmbeddings(),
            narrower=None,
            ranker=_FakeRanker(),
        )
        with pytest.raises(ValueError, match="TokenNarrower"):
            acc.search(_make_query("alpha"))


class TestTheNarrowingIsOnTheLivePath:
    """The blind-token lookup reaches `resolve_authorized_scope`, and its answer is a MEET."""

    def test_the_query_terms_are_what_the_narrower_is_compiled_from(self):
        narrower = _FakeNarrower(matches={"art-1"})
        docs = {"art-1": _make_artifact_doc("art-1")}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs), narrower=narrower,
            ranker=_FakeRanker(hits=[_make_hit("art-1")]),
        )
        acc.search(_make_query("budget type:application/pdf"))
        # Terms only — the filter token was lifted out by `plan_recall` and never reached it.
        assert narrower.calls == ["budget"]

    def test_a_token_miss_empties_the_recall_before_anything_ranks(self):
        ranker = _FakeRanker(hits=[_make_hit("art-1")])
        docs = {"art-1": _make_artifact_doc("art-1")}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(matches=set()),
            ranker=ranker,
        )
        result = acc.search(_make_query("alpha"))
        assert result.hits == []
        assert result.total == 0
        assert ranker.calls == [], "no cell may be decrypted for a query that narrowed to nothing"

    def test_a_token_naming_an_unauthorized_artifact_narrows_to_nothing(self):
        """The security property, observed where a caller observes it.

        The narrowing names an artifact the light cone never authorized. Meeting it against
        the authorized set leaves the empty set — the same answer a token matching nothing
        gives — so the response cannot be read as evidence that the artifact exists.
        """
        docs = {"art-mine": _make_artifact_doc("art-mine")}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-mine"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(matches={"art-someone-elses"}),
            ranker=_FakeRanker(hits=[_make_hit("art-someone-elses")]),
        )
        result = acc.search(_make_query("alpha"))
        assert result.hits == []
        assert result.total == 0

    def test_the_narrowed_set_is_what_the_ranker_is_told_to_rank_within(self):
        docs = {a: _make_artifact_doc(a) for a in ("art-1", "art-2")}
        ranker = _FakeRanker(hits=[_make_hit("art-1")])
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1", "art-2"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(matches={"art-1"}),
            ranker=ranker,
        )
        acc.search(_make_query("alpha"))
        assert ranker.calls[0]["authorized_artifacts"] == {"art-1"}


class TestHydration:
    def test_hydrates_from_artifact_doc(self):
        docs = {"art-1": _make_artifact_doc(
            "art-1",
            title="Encryption Library",
            description="A MANTLE-SSE module",
            tags=["search", "encrypted"],
            content="some content text",
            state="committed",
        )}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(matches={"art-1"}),
            ranker=_FakeRanker(hits=[_make_hit("art-1", 0.77)]),
        )
        result = acc.search(_make_query("encryption"))
        assert len(result.hits) == 1
        hit = result.hits[0]
        assert hit.doc_id == "art-1"
        assert hit.title == "Encryption Library"
        assert hit.description == "A MANTLE-SSE module"
        assert hit.tags == ["search", "encrypted"]
        assert hit.content == "some content text"
        assert hit.state == "committed"
        # The score IS the cosine — there is no fusion constant between them.
        assert hit.score == pytest.approx(0.77)
        assert hit.metadata["vector_score"] == pytest.approx(0.77)
        assert result.ordering == "semantic"

    def test_a_draft_hit_hydrates_the_draft_not_the_committed_row(self):
        """Edit-after-commit: the draft segment's hit must return the DRAFT's bytes.

        This is the one path in the API that forks a lineage. `_ensure_draft` writes the edit
        to a NEW id under the same `root_id` and leaves the committed row exactly where it is,
        while both index arms key on `root_id`. So the draft segment holds the root, and a
        hydration that reads the row sitting AT the root id returns the version the editor
        replaced — recall answering with the new version's tokens and the old version's
        content, which is indistinguishable from a correct answer at the call site.

        Nothing widens here: the light cone authorizes the root, grants are held on the root,
        and every row this can reach belongs to that same lineage. What changes is only WHICH
        version of an already-authorized lineage comes back.
        """
        committed = _make_artifact_doc(
            "root-1", title="Runbook", content="the old committed body", state="committed",
        )
        draft = _make_artifact_doc(
            "draft-1", title="Runbook", content="the edited body", state="draft",
        )
        draft["root_id"] = "root-1"
        store_db = _FakeStoreDB(
            {"root-1": committed, "draft-1": draft},
            versions={"root-1": [committed, draft]},
        )
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["root-1"]),
            store_db=store_db,
            narrower=_FakeNarrower(matches={"root-1"}),
            ranker=_FakeRanker(hits=[_make_hit("root-1", 0.9)]),
            segment="draft",
        )
        result = acc.search(_make_query("runbook"))
        assert len(result.hits) == 1
        assert result.hits[0].content == "the edited body"
        assert result.hits[0].state == "draft"
        # The lineage is what was authorized and what is reported; the version is what moved.
        assert result.hits[0].root_id == "root-1"
        assert result.hits[0].version_id == "draft-1"

    def test_a_committed_hit_still_hydrates_the_committed_row(self):
        """The same forked lineage, read from the committed segment, answers unchanged.

        The counterpart of the test above, and the reason resolution is per-STATE rather than
        to a global head: a global head would hand the draft to a committed search, which is
        the same class of error in the other direction.
        """
        committed = _make_artifact_doc(
            "root-1", title="Runbook", content="the old committed body", state="committed",
        )
        draft = _make_artifact_doc(
            "draft-1", title="Runbook", content="the edited body", state="draft",
        )
        draft["root_id"] = "root-1"
        store_db = _FakeStoreDB(
            {"root-1": committed, "draft-1": draft},
            versions={"root-1": [committed, draft]},
        )
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["root-1"]),
            store_db=store_db,
            narrower=_FakeNarrower(matches={"root-1"}),
            ranker=_FakeRanker(hits=[_make_hit("root-1", 0.9)]),
            segment="committed",
        )
        result = acc.search(_make_query("runbook"))
        assert result.hits[0].content == "the old committed body"
        assert result.hits[0].version_id == "root-1"

    def test_missing_doc_yields_empty_metadata(self):
        # A ranked hit references an artifact removed between ranking and hydration —
        # return a SearchHit with empty fields rather than dropping it.
        docs = {"art-1": _make_artifact_doc("art-1")}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1", "art-gone"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(),
            ranker=_FakeRanker(hits=[_make_hit("art-gone")]),
        )
        result = acc.search(_make_query("alpha"))
        assert len(result.hits) == 1
        assert result.hits[0].doc_id == "art-gone"
        assert result.hits[0].title == ""
        assert result.hits[0].description == ""

    def test_malformed_context_falls_back_to_empty(self):
        docs = {"art-1": {
            "id": "art-1",
            "root_id": "art-1",
            "context": "{not json",
            "content": "",
            "state": "committed",
            "created_by": "user-1",
            "principal_id": "user-1",
        }}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(),
            ranker=_FakeRanker(hits=[_make_hit("art-1")]),
        )
        result = acc.search(_make_query("alpha"))
        assert result.hits[0].title == ""
        assert result.hits[0].tags == []


class TestOrdering:
    def test_embedding_failure_falls_through_to_the_unranked_path(self):
        """No cosine can order this, so no cell is decrypted and the narrowing decides.

        The narrower here compiles to `None` — "nothing to look up" — so there is no coverage
        either, and the answer is the clock with no score on it. The coverage case is
        `test_text_only_recall_orders_by_query_coverage` below."""
        docs = {"art-1": _make_artifact_doc("art-1")}
        ranker = _FakeRanker(hits=[_make_hit("art-1")])
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs),
            embeddings=_FakeEmbeddings(raise_on_call=True),
            narrower=_FakeNarrower(),
            ranker=ranker,
        )
        result = acc.search(_make_query("alpha"))
        assert len(result.hits) == 1
        assert result.ordering == "recency"
        assert result.hits[0].score is None
        assert ranker.calls == [], "no vector means no cell is decrypted"

    def test_a_cosine_orders_when_a_vector_reaches_the_ranker(self):
        docs = {"art-1": _make_artifact_doc("art-1")}
        ranker = _FakeRanker(hits=[_make_hit("art-1", 0.5)])
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(),
            ranker=ranker,
        )
        result = acc.search(_make_query("alpha"))
        assert result.ordering == "semantic"
        assert result.hits[0].score == pytest.approx(0.5)
        assert ranker.calls[0]["query_embedding"] == [0.1, 0.2, 0.3]

    #: Three artifacts, deliberately NOT in id order and not in insertion order, so an
    #: assertion about the clock cannot pass by accident on either.
    _DATED = {
        "art-old": "2024-01-01T00:00:00Z",
        "art-mid": "2025-06-01T00:00:00Z",
        "art-new": "2026-02-01T00:00:00Z",
    }

    @classmethod
    def _dated_docs(cls):
        return {a: _make_artifact_doc(a, modified=t) for a, t in cls._DATED.items()}

    def test_text_only_recall_orders_by_query_coverage(self):
        """The whole of the text-only path: it narrows, it orders by how much of the query
        each hit matched, and it says so.

        A caller that cannot embed still asked a real question and gets a real answer — not a
        400, not an arbitrary set iteration order, and not the clock when the narrowing has
        something better to say. `art-mid` matched two of the query's stems and the other two
        matched one each, so it leads despite being neither the newest nor the first.
        """
        docs = self._dated_docs()
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=list(docs)),
            store_db=_FakeStoreDB(docs),
            embeddings=_FakeEmbeddings(raise_on_call=True),
            narrower=_FakeNarrower(matches=set(docs),
                                   coverage={"art-mid": Coverage(2, 0)}),
        )
        result = acc.search(_make_query("alpha"))
        assert result.ordering == "coverage"
        assert [h.doc_id for h in result.hits] == ["art-mid", "art-new", "art-old"]
        assert [h.score for h in result.hits] == [2, 1, 1]
        assert result.total == 3
        assert result.total_is_capped is False

    def test_equal_coverage_falls_back_to_the_clock(self):
        """The tiebreak is the existing recency order, unchanged — which is why a query whose
        stems all match everything is byte-identical to what recency ordering produced."""
        docs = self._dated_docs()
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=list(docs)),
            store_db=_FakeStoreDB(docs),
            embeddings=_FakeEmbeddings(raise_on_call=True),
            narrower=_FakeNarrower(matches=set(docs)),
        )
        result = acc.search(_make_query("alpha"))
        assert result.ordering == "coverage"
        assert [h.doc_id for h in result.hits] == ["art-new", "art-mid", "art-old"]
        assert [h.score for h in result.hits] == [1, 1, 1]

    def test_a_bigram_count_breaks_a_stem_tie_before_the_clock_does(self):
        """The second ordinal, between the first and the tiebreak. `art-old` is the OLDEST and
        wins on it, so nothing here could have come from the clock."""
        docs = self._dated_docs()
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=list(docs)),
            store_db=_FakeStoreDB(docs),
            embeddings=_FakeEmbeddings(raise_on_call=True),
            narrower=_FakeNarrower(matches=set(docs),
                                   coverage={"art-old": Coverage(1, 1)}),
        )
        result = acc.search(_make_query("alpha"))
        assert [h.doc_id for h in result.hits] == ["art-old", "art-new", "art-mid"]

    def test_the_counts_are_echoed_in_metadata(self):
        """`score` is one field and coverage is two numbers, so the second rides in
        `metadata` rather than being dropped or folded into the first."""
        docs = {"art-1": _make_artifact_doc("art-1")}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs),
            embeddings=_FakeEmbeddings(raise_on_call=True),
            narrower=_FakeNarrower(matches={"art-1"},
                                   coverage={"art-1": Coverage(3, 2)}),
        )
        result = acc.search(_make_query("alpha"))
        assert result.hits[0].metadata == {"matched_stems": 3, "matched_bigrams": 2}

    def test_a_count_belonging_to_an_unauthorized_artifact_is_never_read(self):
        """THE leak question, at the accessor.

        The narrowing's answer is a superset of the surviving set — it can name artifacts the
        light cone refused, which is why the resolver meets its KEYS rather than trusting
        them. `_by_coverage` walks the SURVIVING ids, so a refused artifact's count is never
        looked up and cannot move a score, a position or a total. Here the refused artifact
        carries the LARGEST count in the map; nothing in the body reflects it.
        """
        docs = {"art-1": _make_artifact_doc("art-1")}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs),
            embeddings=_FakeEmbeddings(raise_on_call=True),
            narrower=_FakeNarrower(
                matches={"art-1", "art-secret"},
                coverage={"art-secret": Coverage(9, 9)},
            ),
        )
        result = acc.search(_make_query("alpha"))
        assert [h.doc_id for h in result.hits] == ["art-1"]
        assert [h.score for h in result.hits] == [1]
        assert result.total == 1

    def test_an_undated_doc_sorts_last_rather_than_being_dropped(self):
        """An unread or absent timestamp is not evidence that an artifact is old — but it is
        not evidence that it is recent either, so it goes to the end and stays in the set."""
        docs = {
            "art-dated": _make_artifact_doc("art-dated", modified="2025-01-01T00:00:00Z"),
            "art-undated": {
                "id": "art-undated", "root_id": "art-undated",
                "context": "{}", "content": "", "state": "committed",
                "created_by": "user-1", "principal_id": "user-1",
            },
        }
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=list(docs)),
            store_db=_FakeStoreDB(docs),
            embeddings=_FakeEmbeddings(raise_on_call=True),
            narrower=_FakeNarrower(),
        )
        result = acc.search(_make_query("alpha"))
        assert [h.doc_id for h in result.hits] == ["art-dated", "art-undated"]

    def test_sort_recency_is_honoured_even_with_a_vector(self):
        """`sort` is read, and reading it means the ranker is not run at all."""
        docs = {
            "art-old": _make_artifact_doc("art-old", modified="2024-01-01T00:00:00Z"),
            "art-new": _make_artifact_doc("art-new", modified="2026-02-01T00:00:00Z"),
        }
        ranker = _FakeRanker(hits=[_make_hit("art-old", 0.99)])
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=list(docs)),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(),
            ranker=ranker,
        )
        result = acc.search(_make_query("alpha", sort="recency"))
        assert result.ordering == "recency"
        assert [h.doc_id for h in result.hits] == ["art-new", "art-old"]
        assert ranker.calls == [], "asking for recency must not cost a cell decryption"

    def test_relevance_with_neither_a_vector_nor_coverage_resolves_to_recency(self):
        """What `relevance` means when nothing can measure it: the best ordering available.

        With a vector that is the cosine; with query terms that is coverage; with neither —
        which is this case, a query the narrowing compiled nothing for — it is the clock. It
        is a request, so it cannot promise an outcome; `ordering` is the fact."""
        docs = {"art-1": _make_artifact_doc("art-1")}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs),
            embeddings=_FakeEmbeddings(raise_on_call=True),
            narrower=_FakeNarrower(),
        )
        result = acc.search(_make_query("alpha", sort="relevance"))
        assert result.ordering == "recency"

    def test_an_unrankable_node_still_answers_rather_than_emptying(self):
        """No AnchorSet is the install default. Recall still narrows and still answers."""
        from mantle.search.anchors.store import AnchorSetNotProvisioned

        docs = {"art-1": _make_artifact_doc("art-1")}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(),
            ranker=_FakeRanker(raises=AnchorSetNotProvisioned("no set")),
        )
        result = acc.search(_make_query("alpha"))
        assert result.ordering == "recency"
        assert [h.doc_id for h in result.hits] == ["art-1"]
        assert result.hits[0].score is None

    def test_a_ranker_that_ran_and_matched_nothing_stays_empty(self):
        """An empty ranking is an answer; recency must not re-admit what it excluded.

        The distinction is between an arm that COULD NOT RUN — which falls through to the
        clock — and one that ran over the narrowed set and kept none of it. Ordering the
        second by recency would hand back every artifact the cosine just rejected.
        """
        docs = {"art-1": _make_artifact_doc("art-1")}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(),
            ranker=_FakeRanker(hits=[]),
        )
        result = acc.search(_make_query("alpha"))
        assert result.hits == []
        assert result.total == 0
        assert result.ordering == "semantic"


class TestSizeAndTotal:
    def test_trims_to_query_size(self):
        docs = {f"art-{i}": _make_artifact_doc(f"art-{i}") for i in range(5)}
        hits = [_make_hit(f"art-{i}", score=1.0 - 0.01 * i) for i in range(5)]
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=list(docs)),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(),
            ranker=_FakeRanker(hits=hits),
        )
        result = acc.search(_make_query("alpha", size=2))
        assert len(result.hits) == 2
        # total reflects pre-trim count.
        assert result.total == 5

    def test_no_hits_from_the_ranker(self):
        docs = {"art-1": _make_artifact_doc("art-1")}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(),
            ranker=_FakeRanker(hits=[]),
        )
        result = acc.search(_make_query("alpha"))
        assert result.hits == []
        assert result.total == 0

    def test_chunks_of_one_artifact_collapse_to_one_hit(self):
        """A page is a page of artifacts. Two close chunks of one document are one result,
        scored by the closer of them."""
        docs = {"art-1": _make_artifact_doc("art-1")}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(),
            ranker=_FakeRanker(hits=[
                _make_hit("art-1", 0.4, chunk_id=0),
                _make_hit("art-1", 0.8, chunk_id=1),
            ]),
        )
        result = acc.search(_make_query("alpha"))
        assert len(result.hits) == 1
        assert result.total == 1
        assert result.hits[0].score == pytest.approx(0.8)

    def test_a_ranker_that_ignores_the_narrowed_set_still_cannot_widen(self):
        """The meet is restated on the ranker's output, so the guarantee holds at this
        boundary whichever engine is wired in."""
        docs = {"art-1": _make_artifact_doc("art-1")}
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-1"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(matches={"art-1"}),
            ranker=_FakeRanker(hits=[_make_hit("art-1"), _make_hit("art-elsewhere")]),
        )
        result = acc.search(_make_query("alpha"))
        assert [h.doc_id for h in result.hits] == ["art-1"]


class TestScopeFiltering:
    """query.scope restricts search to explicit container IDs."""

    def _make_docs_two_collections(self):
        # art-A lives in col-A; art-B lives in col-B.
        return {
            "art-A": {
                "id": "art-A", "root_id": "art-A",
                "context": '{"title": "Alpha Article"}',
                "content": "", "state": "committed",
                "created_by": "user-1", "principal_id": "user-1",
                "collection_id": "col-A",
            },
            "art-B": {
                "id": "art-B", "root_id": "art-B",
                "context": '{"title": "Beta Article"}',
                "content": "", "state": "committed",
                "created_by": "user-1", "principal_id": "user-1",
                "collection_id": "col-B",
            },
        }

    def test_scope_restricts_to_matching_collection(self):
        docs = self._make_docs_two_collections()
        ranker = _FakeRanker(hits=[_make_hit("art-A", collection_id="col-A")])
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-A", "art-B"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(),
            ranker=ranker,
        )
        from mantle.search.types import SearchQuery
        acc.search(SearchQuery(
            query_text="alpha", user_id="user-1", scope=["col-A"], size=20,
        ))
        assert ranker.calls, "the ranker should have been invoked"
        call_contexts = ranker.calls[0]["contexts"]
        assert all(col == "col-A" for _, col in call_contexts), (
            f"Expected only col-A contexts, got {call_contexts}"
        )

    def test_scope_none_searches_all_contexts(self):
        docs = self._make_docs_two_collections()
        ranker = _FakeRanker(hits=[])
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-A", "art-B"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(),
            ranker=ranker,
        )
        from mantle.search.types import SearchQuery
        acc.search(SearchQuery(
            query_text="alpha", user_id="user-1", scope=None, size=20,
        ))
        collection_ids_searched = {col for _, col in ranker.calls[0]["contexts"]}
        assert "col-A" in collection_ids_searched
        assert "col-B" in collection_ids_searched

    def test_scope_no_match_returns_empty(self):
        docs = self._make_docs_two_collections()
        ranker = _FakeRanker(hits=[_make_hit("art-A", collection_id="col-A")])
        acc = _accessor(
            lightcone=_FakeLightCone(authorized=["art-A", "art-B"]),
            store_db=_FakeStoreDB(docs),
            narrower=_FakeNarrower(),
            ranker=ranker,
        )
        from mantle.search.types import SearchQuery
        result = acc.search(SearchQuery(
            query_text="alpha", user_id="user-1", scope=["col-UNKNOWN"], size=20,
        ))
        assert result.hits == []
        assert result.total == 0
        # Short-circuit on empty contexts.
        assert ranker.calls == []
