"""Mantle's own recall reads `mantle.search.ranking`, and reading it changes nothing it may return.

## Why this exists

`mantle.search.ranking` was extracted so the base install could order candidates by what a question
is ABOUT without a persona. Extracting it put the code where the candidates are; it did not make
anything call it. Until `_order` did, every improvement measured on that module was unreachable from
`recall` — the same defect `op.retrieve` had, where the ranked path existed and the served path was
raw BM25.

So this asserts the wiring, and it asserts the two things the wiring could get wrong.

## The two hazards

**`rank` may ADD candidates.** The synset a need names is a position by construction, so the ranking
appends `wn-<name>` when the store holds that vertex — right for a corpus, and a light-cone breach
here: those vertices live in this node's lattice too and never passed the resolve. The arm meets its
output back against the pool it was handed. `test_the_ranking_cannot_add_an_artifact_to_a_recall`
puts a reachable vertex in the store, outside the authorized set, and requires it absent.

**A ranker that cannot run must not empty a recall.** `rank` reports `unavailable` /
`no-coordinate` / `unreached` rather than raising, and each of those is a statement, not an order —
so the coverage order has to survive all three, plus an outright exception. That is the base
install's behaviour and it is byte-identical to what shipped before this arm existed.
"""
from __future__ import annotations

import pytest

from mantle.search import ranking
from mantle.search.mantle.sse.router_accessor import MantleSseSearchAccessor, _Ranked
from mantle.search.types import ORDER_COVERAGE, ORDER_REACH


class _Query:
    """Only the fields `_by_reach` reads."""

    def __init__(self, text: str) -> None:
        self.query_text = text


class _Match:
    """A `match` seam with a stated geometry: `fired` maps a position to the energy it contributes.

    The same shape `sage`'s ranking fixture uses, kept here rather than shared because a mantle test
    that imported a chorus fixture would assert the layer law backwards.
    """

    def __init__(self, fired):
        self._fired = fired

    def fired_field(self, query, store):
        return dict(self._fired)

    def offer_synsets(self, text):
        return [w for w in str(text).split() if w in self._fired]

    def propagate(self, fired, targets):
        hit = [fired[t] for t in targets if t in fired]
        return (float(sum(hit)), 0.0 if hit else float("inf"))

    def expand_associative(self, store, fired):
        return fired

    def frame(self, store, names):
        return None


class _Conn:
    """The two doc reads the ranking makes, modelled as the lattice answers them.

    The batched `id IN (...)` form is the one that runs: a pool of 200 was 200 point lookups into
    a 9.7 GB file before it was batched. The single-id form is kept because a fake that only
    answers the shape currently called cannot catch the next caller getting it wrong.
    """

    def __init__(self, docs):
        self._docs = docs

    def _doc(self, cid):
        d = self._docs.get(cid)
        if d is None:
            return None
        return '{"lemmas": [%s]}' % ", ".join('"%s"' % w for w in d.split())

    def execute(self, sql, args=()):
        if " IN (" in sql:
            return _Rows([(cid, self._doc(cid)) for cid in args if self._doc(cid) is not None])
        cid = args[0] if args else None
        doc = self._doc(cid)
        return _Rows([] if doc is None else [(doc,)])


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _Db:
    def __init__(self, outer):
        self._outer = outer

    def read(self):
        return _Conn(self._outer._docs)


class _StoreDb:
    """Stands in for the `LatticeDatabase` the accessor holds — `artifacts.db.read()` and
    `artifacts.get_artifact`, which is all the ranking touches."""

    def __init__(self, docs, vertices=()):
        self._docs = docs
        self._vertices = set(vertices)
        self.artifacts = self

    @property
    def db(self):
        return _Db(self)

    def get_artifact(self, vid):
        return {"id": vid} if vid in self._vertices else None


@pytest.fixture(autouse=True)
def _no_ambient_ontology(monkeypatch):
    """No seam from anywhere unless a test installs one.

    `ranking._resolve_seam` falls back to the host seams `prism.runner` holds, so "no ontology"
    has to mean the registry too — otherwise these tests would pass or fail on whether something
    else in the session registered one.
    """
    from prism import runner
    monkeypatch.setattr(runner, "_HOST_SEAMS", {})


@pytest.fixture()
def bound(monkeypatch):
    """Bind a seam into the ranking module and restore it afterwards."""
    def _install(fired):
        m = _Match(fired)
        monkeypatch.setattr(ranking, "_MATCH_SEAM", m)
        monkeypatch.setattr(ranking, "_PROJECTION_SEAM", m)
        return m
    return _install


def _accessor(store_db):
    """An accessor with only the field this arm reads. `_by_reach` is a method on the ordering
    half of the class and touches no index, so constructing the retrieval halves would assert
    nothing this file is about."""
    a = MantleSseSearchAccessor.__new__(MantleSseSearchAccessor)
    a._store_db = store_db
    return a


def _coverage_order(*pairs):
    """`_Ranked` rows as `_by_coverage` builds them: score is the stem count, biggest first."""
    return [_Ranked(aid, float(stems), "col-1", "prin-1") for aid, stems in pairs]


def test_reach_reorders_what_coverage_ordered(bound):
    """The wiring, as a rule: the arm must be able to move a hit coverage put second.

    `doc-broad` carries both query stems and `doc-subject` carries one, so coverage ranks the
    broad one first. Reach reads what each stands on — the subject reaches hard, the two others
    barely — and the standardisation asks each candidate to beat what its own size predicts.
    """
    bound({"glacier.n.01": 8.5, "trail.n.01": 0.5, "park.n.01": 0.5})
    store = _StoreDb({
        "doc-broad": "glacier.n.01 trail.n.01 park.n.01",
        "doc-subject": "glacier.n.01",
    })
    ordered = _coverage_order(("doc-broad", 2), ("doc-subject", 1))

    out = _accessor(store)._by_reach(_Query("glacier"), ordered, top_k=50)

    assert out is not None, "an ontology is bound and the query fired; the arm should have run"
    assert out[0].artifact_id == "doc-subject", [r.artifact_id for r in out]


def test_the_ranking_cannot_add_an_artifact_to_a_recall(bound):
    """`wn-glacier.n.01` is in the store and reachable, so the ranking appends it as a position.
    It is not in the authorized pool, so it must not be in the answer — the light cone decided
    that before any of this ran, and an ordering does not get to revisit it."""
    bound({"glacier.n.01": 8.5})
    store = _StoreDb({"doc-a": "glacier.n.01"}, vertices={"wn-glacier.n.01"})
    ordered = _coverage_order(("doc-a", 1))

    out = _accessor(store)._by_reach(_Query("glacier"), ordered, top_k=50)

    assert out is not None
    assert [r.artifact_id for r in out] == ["doc-a"], (
        "a vertex the ranking supplied reached the response; the arm is unioning with the "
        "ranking's output instead of meeting the authorized pool. Got %s"
        % [r.artifact_id for r in out])


def test_the_pool_stops_at_the_horizon(bound):
    """Reach costs a propagation per position, so the arm reads `top_k` and no more. Whatever it
    returns is drawn from those; the rest keep their coverage places behind them."""
    bound({"a.n.01": 5.0})
    store = _StoreDb({("doc-%d" % i): "a.n.01" for i in range(10)})
    ordered = _coverage_order(*[("doc-%d" % i, 10 - i) for i in range(10)])

    out = _accessor(store)._by_reach(_Query("a"), ordered, top_k=3)

    assert out is not None
    assert set(r.artifact_id for r in out) <= {"doc-0", "doc-1", "doc-2"}, (
        "the arm ranked past its horizon: %s" % [r.artifact_id for r in out])


@pytest.mark.parametrize("fired, docs, why", [
    ({}, {"doc-a": "x.n.01"}, "the query fired nothing, so there is no coordinate to rank on"),
    ({"a.n.01": 1.0}, {"doc-a": "unrelated.n.01"}, "nothing in the pool is reachable"),
])
def test_a_ranking_that_cannot_run_leaves_the_coverage_order_alone(bound, fired, docs, why):
    bound(fired)
    ordered = _coverage_order(("doc-a", 1))
    assert _accessor(_StoreDb(docs))._by_reach(_Query("a"), ordered, top_k=50) is None, why


def test_no_ontology_means_no_reach_arm(monkeypatch):
    """The base install: nothing bound, so `rank` reports `unavailable` and recall is exactly the
    coverage-ordered recall it was before this arm existed."""
    monkeypatch.setattr(ranking, "_MATCH_SEAM", None)
    monkeypatch.setattr(ranking, "_PROJECTION_SEAM", None)
    ordered = _coverage_order(("doc-a", 2), ("doc-b", 1))

    assert _accessor(_StoreDb({}))._by_reach(_Query("a"), ordered, top_k=50) is None


def test_a_ranking_that_raises_does_not_fail_the_recall(bound, monkeypatch):
    """A recall that narrowed correctly has an answer. An ordering that blows up loses the better
    order and nothing else — the request does not become a 500."""
    bound({"a.n.01": 1.0})
    monkeypatch.setattr(ranking, "rank", lambda *a, **k: 1 / 0)
    ordered = _coverage_order(("doc-a", 1))

    assert _accessor(_StoreDb({"doc-a": "a.n.01"}))._by_reach(
        _Query("a"), ordered, top_k=50) is None


def test_an_empty_query_has_nothing_to_be_about(bound):
    """An embedding-only recall carries no text, and reach is measured from text. Nothing to
    measure is not an ordering."""
    bound({"a.n.01": 1.0})
    ordered = _coverage_order(("doc-a", 1))
    assert _accessor(_StoreDb({}))._by_reach(_Query("   "), ordered, top_k=50) is None


def test_the_order_names_which_arm_answered(bound):
    """`ORDER_REACH` and `ORDER_COVERAGE` are distinct values because they mean different things
    about `total`: coverage counts every narrowed match, reach counts what survived its cut. A
    client that cannot tell them apart cannot page either one correctly."""
    assert ORDER_REACH != ORDER_COVERAGE
    from mantle.routers.artifacts_router import ArtifactRecallResponse

    allowed = ArtifactRecallResponse.model_fields["ordering"].annotation.__args__
    assert ORDER_REACH in allowed, (
        "the accessor can return %r and the response model would reject it: %s"
        % (ORDER_REACH, allowed))


# ── the branch, not only the method ──────────────────────────────────────────────────────────────


def test_order_tries_reach_before_coverage_and_names_what_answered(bound):
    """`_by_reach` being correct proves nothing if `_order` never calls it — that is exactly the
    defect this whole change closes, one layer up: the ranking existed and the served path did not
    reach it. So this drives `_order` and reads back the ordering it reports."""
    from mantle.search.mantle.lightcone import AuthorizedScope
    from mantle.search.mantle.sse.narrowing import Coverage

    bound({"glacier.n.01": 8.5, "trail.n.01": 0.5, "park.n.01": 0.5})
    store = _StoreDb({
        "doc-broad": "glacier.n.01 trail.n.01 park.n.01",
        "doc-subject": "glacier.n.01",
    })
    scope = AuthorizedScope(
        contexts=[("prin-1", "col-1")],
        artifact_ids=frozenset({"doc-broad", "doc-subject"}),
        updated_at={"doc-broad": "2026-01-02", "doc-subject": "2026-01-01"},
    )
    coverage = {"doc-broad": Coverage(stems=2), "doc-subject": Coverage(stems=1)}

    ordering, ranked = _accessor(store)._order(
        _Query("glacier"), scope, [], None, coverage, top_k=50)

    assert ordering == ORDER_REACH, (
        "`_order` answered %r — the reach arm is wired in but not reached" % ordering)
    assert ranked[0].artifact_id == "doc-subject", [r.artifact_id for r in ranked]


def test_order_falls_back_to_coverage_with_no_ontology(monkeypatch):
    """The base install, through the same entry point: the ordering is `coverage` and the set is
    the whole narrowed set, uncut."""
    from mantle.search.mantle.lightcone import AuthorizedScope
    from mantle.search.mantle.sse.narrowing import Coverage

    monkeypatch.setattr(ranking, "_MATCH_SEAM", None)
    scope = AuthorizedScope(
        contexts=[("prin-1", "col-1")],
        artifact_ids=frozenset({"doc-a", "doc-b"}),
        updated_at={"doc-a": "2026-01-02", "doc-b": "2026-01-01"},
    )
    coverage = {"doc-a": Coverage(stems=2), "doc-b": Coverage(stems=1)}

    ordering, ranked = _accessor(_StoreDb({}))._order(
        _Query("glacier"), scope, [], None, coverage, top_k=50)

    assert ordering == ORDER_COVERAGE
    assert [r.artifact_id for r in ranked] == ["doc-a", "doc-b"]
