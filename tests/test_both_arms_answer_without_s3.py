"""The air-gap invariant: ``None`` for S3 is a complete configuration, on BOTH arms.

Mantle's claim is that a node is a standalone store — one SQLite file plus a filesystem
CAS, opened in-process, nothing external to provision. Recall is part of that store, so
the claim is only true if a node with no object storage can answer a recall — and answer
it on the semantic arm as well as the lexical one. Half an answer is a different product.

This is an end-to-end proof rather than a unit test of the builders: the wiring is asked
for its production objects with nothing configured, real artifacts are indexed through
both real indexers, and each arm is really asked a question. ``edge_s3_if_reachable`` is
replaced by a tripwire that fails the test if anything reaches for a bucket, so "no S3 was
used" is measured rather than assumed.

The two arms are asked SEPARATELY here, because they answer separate questions and nothing
fuses them: the postings answer membership — which artifacts carry these stems — and the
cells answer proximity. A query term and a query direction that point at DIFFERENT artifacts
is what makes "both ran" visible without a fused record to read it off.

SCOPE, because the file's shape suggests a wider claim than it makes: the variable held at
``None`` here is S3, and S3 alone. The semantic arm's OTHER prerequisite — a provisioned
AnchorSet — is supplied by the autouse fixture below, so nothing in this file says a
freshly installed node answers semantically. It does not: anchors are provisioned by an
operator and Mantle derives none, so the install default is lexical-only recall. What is
proved here is that removing object storage is not what takes the semantic arm away.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from mantle.search.mantle import wiring
from mantle.search.mantle.file_cell_store import FileCellStore
from mantle.search.mantle.sse import SqlitePostingStore
from .helpers import make_oracle, req, self_request

DIM = 16


@pytest.fixture
def air_gapped(tmp_path, monkeypatch):
    """A node with no object storage — and a tripwire on the only door to one."""
    monkeypatch.delenv("MANTLE_SSE_STORE", raising=False)
    monkeypatch.delenv("MANTLE_CELL_STORE", raising=False)
    monkeypatch.setenv("MANTLE_SSE_DIR", str(tmp_path / "sse"))
    monkeypatch.setenv("MANTLE_CELL_DIR", str(tmp_path / "cells"))

    def _tripwire(what):
        raise AssertionError(
            f"{what} reached for an S3 bucket on a node with no object storage configured"
        )

    oracle = make_oracle()
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            wiring, "edge_object_storage_is_configured", lambda: False,
        ))
        stack.enter_context(patch.object(wiring, "edge_s3_if_reachable", _tripwire))
        stack.enter_context(patch.object(wiring, "_build_oracle", lambda: oracle))
        yield


@pytest.fixture(autouse=True)
def _live_anchorset():
    """A provisioned AnchorSet — the vector arm's other prerequisite, and the one this
    file deliberately does NOT prove is present by default. See the module note in
    `pipeline_unified._mantle_index_artifact`: anchors are provisioned, never derived."""
    from mantle.search.anchors import store
    from mantle.search.anchors.anchorset import AnchorSet
    from mantle.search.anchors.repo import InMemoryAnchorRepo

    store.set_anchor_repo(InMemoryAnchorRepo())
    aset = AnchorSet("hf:test@1.0", DIM)
    aset.add_text("anchor-0", np.ones(DIM, dtype=np.float32))
    store.save_live_anchorset(aset)
    yield
    store.set_anchor_repo(None)


def _unit(*axes: tuple[int, float]) -> list[float]:
    v = np.zeros(DIM, dtype=np.float64)
    for axis, value in axes:
        v[axis] = value
    return (v / np.linalg.norm(v)).tolist()


#: A direction no query term reaches. A corpus where the two arms agree cannot show that
#: both of them ran, so the query term and the query direction point at different artifacts
#: and the fused result has to carry both.
_DIRECTION_B = _unit((10, 1.0))

_CORPUS = {
    "art-1": ("encryption keys and cards", _unit((0, 1.0), (1, 0.30))),
    "art-2": ("encryption is discussed at length here", _unit((5, 1.0))),
    "art-3": ("a library of lazy dogs", _unit((10, 1.0), (11, 0.30))),
    "art-4": ("the quick brown fox", _unit((10, 1.0), (12, 0.35))),
    "art-5": ("a map of the coastline", _unit((10, 1.0), (13, 0.40))),
    "art-6": ("recipes for bread", _unit((10, 1.0), (14, 0.45))),
}

#: Artifacts whose TEXT the lexical arm can reach with the query term below.
_LEXICAL = {"art-1", "art-2"}
_TERM = "encryption"


def _index_corpus() -> None:
    """Index the same corpus through both production indexers."""
    sse = wiring.build_sse_indexer(None)
    vector = wiring.build_indexer(None)
    assert sse is not None, "the lexical arm has nowhere to write on a standalone node"
    assert vector is not None, "the semantic arm has nowhere to write on a standalone node"

    for artifact_id, (text, embedding) in _CORPUS.items():
        sse.index_artifact(
            "owner-A", "col-1", artifact_id, {"content": text},
            self_request("owner-A", "update"),
        )
        vector.index_artifact(
            "owner-A", "col-1",
            [{"artifact_id": artifact_id, "chunk_id": 0, "embedding": embedding,
              "text": text}],
            self_request("owner-A", "update"),
        )


def _accessor():
    """The production recall stack over the local index — a narrower and a ranker."""
    acc = wiring.build_sse_search_accessor(None)
    assert acc is not None, "no accessor on a standalone node"
    return acc


def _reach(text: str) -> set:
    """What the lexical arm reaches for ``text`` over owner-A's collection."""
    lookup = _accessor()._narrower.lookup_for(text, req())
    return set() if lookup is None else set(lookup([("owner-A", "col-1")]))


class TestTheStoresAreLocal:
    def test_both_arms_resolve_to_a_local_backend(self, air_gapped):
        assert isinstance(wiring._build_sse_store(), SqlitePostingStore)
        assert isinstance(wiring._build_cell_store(), FileCellStore)

    def test_the_availability_probes_both_say_yes(self, air_gapped):
        """The gates a full reindex asks before it scans anything. If either said no, the
        corresponding arm would stay permanently empty on a standalone install."""
        assert wiring.sse_index_storage_available() is True
        assert wiring.cell_storage_available() is True

    def test_the_accessor_has_two_arms(self, air_gapped):
        accessor = _accessor()
        assert accessor._narrower is not None
        assert accessor._ranker is not None, (
            "the vector arm is None on a node with no S3 — the semantic half of recall is "
            "unreachable in the default configuration"
        )


class TestBothArmsAnswer:
    """The proof itself: index locally, query locally, get an answer out of each arm."""

    def test_the_lexical_arm_answers(self, air_gapped):
        _index_corpus()
        assert _reach(_TERM) == _LEXICAL

    def test_the_lexical_arm_counts_as_well_as_reaches(self, air_gapped):
        """The coverage counts are what orders a recall with no vector, and they come off the
        same local files. A store that returned openable blobs but lost an entry would still
        narrow — and would silently demote the artifact whose stem it dropped."""
        _index_corpus()
        lookup = _accessor()._narrower.lookup_for("encryption keys", req())
        found = lookup([("owner-A", "col-1")])
        assert found["art-1"].stems == 2, "art-1 carries both stems"
        assert found["art-2"].stems == 1

    def test_the_semantic_arm_answers(self, air_gapped):
        """Everything below comes from the encrypted cells on local disk, and every artifact
        it finds is one the query term cannot reach."""
        _index_corpus()
        hits = _accessor()._ranker.search(
            _DIRECTION_B, [("owner-A", "col-1")], req(),
        )
        found = {h.artifact_id for h in hits}
        assert found, "the semantic arm returned nothing on a node with no S3"
        assert found.isdisjoint(_LEXICAL)
        assert all(h.score is not None for h in hits)

    def test_the_two_arms_reach_different_artifacts(self, air_gapped):
        """The whole point of two arms: between them they hold artifacts neither could have
        found alone. Nothing fuses them, so this is stated as two answers rather than one."""
        _index_corpus()
        lexical = _reach(_TERM)
        semantic = {h.artifact_id for h in _accessor()._ranker.search(
            _DIRECTION_B, [("owner-A", "col-1")], req())}
        assert lexical == _LEXICAL
        assert semantic, "no hit came from the vector arm — the semantic half is dark"
        assert semantic.isdisjoint(lexical)

    def test_one_artifact_is_reachable_from_both_directions(self, air_gapped):
        """An artifact the term reaches AND the vector ranks. There is no `source` flag to
        read that off any more — with one ranker there would be nothing for it to vary over —
        so it is stated as membership in both answers."""
        _index_corpus()
        semantic = {h.artifact_id for h in _accessor()._ranker.search(
            _CORPUS["art-1"][1], [("owner-A", "col-1")], req())}
        assert "art-1" in _reach(_TERM)
        assert "art-1" in semantic

    def test_both_indexes_are_on_disk_and_answer_after_a_restart(self, air_gapped, tmp_path):
        """Durable, and durable on both arms. Without this the standalone node would search
        only what the current process happened to write, and every restart would be a silent
        reindex-or-nothing."""
        _index_corpus()
        for root in (tmp_path / "sse", tmp_path / "cells"):
            assert any(root.rglob("*")), f"{root} holds no index after indexing"

        # Fresh objects over the same directories, with every decrypted-cell cache the writing
        # objects held thrown away with them — what a restart leaves behind.
        assert _reach(_TERM) == _LEXICAL
        assert {h.artifact_id for h in _accessor()._ranker.search(
            _DIRECTION_B, [("owner-A", "col-1")], req())}
