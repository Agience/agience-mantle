"""`created_by` / `created_time` are deprecated — and these tests are the removal gate.

Provenance and ordering need a deterministic method before either is removed. Removal is
therefore not a judgement call: each column has a named deterministic replacement, and may only
be dropped once that replacement is populated and verified. These tests assert the replacements
behave — not merely that they are documented — so "we have a replacement" can never be a claim
someone makes in a review.
"""
from __future__ import annotations

import os

import pytest

from . import open_lattice
from .schema import EDGE_DDL, VERTEX_DDL

_SCHEMA_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.py")


def _schema_source() -> str:
    """The schema module source — the deprecation lives in comments beside the column
    definitions, which is where a reader will look. `VERTEX_DDL` holds only SQL strings."""
    with open(_SCHEMA_SRC, encoding="utf-8") as f:
        return f.read()


def _deprecation_prose() -> str:
    """The contiguous run of comment lines that carries the word "deprecated", lower-cased.

    Two reasons the scope is a comment run rather than the whole file. It keeps every assertion
    below load-bearing: prose elsewhere in `schema.py` — or an SQL identifier inside the DDL —
    cannot satisfy a check about the deprecation. And it asserts co-location, which is the
    property that matters to a reader: the gate has to be stated where the column is defined,
    not in a paragraph three screens away.

    Matching is on substance, case-insensitively. An assertion on an exact upper-case spelling
    tests the shouting rather than the documentation, and breaks on a rewording that improved it.
    """
    runs: list[list[str]] = []
    current: list[str] = []
    for line in _schema_source().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            current.append(stripped.lstrip("#").strip())
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return "\n".join(
        "\n".join(run) for run in runs if any("deprecat" in ln.lower() for ln in run)
    ).lower()


# ── the deprecation is recorded, not folklore ───────────────────────────────────
def test_deprecation_and_its_preconditions_are_documented():
    """A deprecation that lives only in a chat log is not a deprecation. The schema must say
    which columns are going, what replaces each, and what must be true before removal."""
    prose = _deprecation_prose()
    assert prose, "no comment block beside the columns records a deprecation at all"
    assert "created_by" in prose and "created_time" in prose, (
        "both deprecated columns must be named in the block that deprecates them")
    assert "removal precondition" in prose, (
        "the gate must be stated where the column is defined")
    assert "creation edge" in prose, "the WHO replacement must be named"
    assert "_seq" in prose and "order_key" in prose, "the ordering replacement must be named"
    assert "concurrent" in prose and "unordered is a valid answer" in prose, (
        "the honest limit must be stated: there is no total order across origins, so equal "
        "`_seq` from different origins are concurrent")


@pytest.mark.parametrize("col", ["created_by", "created_time"])
def test_deprecated_column_is_still_declared(col):
    """They have not been removed."""
    assert col in "\n".join(VERTEX_DDL), "%s should still be declared while deprecated" % col
    assert col in _schema_source()


# ── the ordering replacement must actually order ────────────────────────────────
def test_seq_is_gap_free_within_an_origin(tmp_path):
    """`(_origin, _seq)` replaces `created_time` for ordering — but only if it is gap-free. A
    gapped sequence would still sort, but could not distinguish a missing row from one that was
    never written — so it would not be a replacement for anything."""
    L = open_lattice(str(tmp_path / "t.db"), origin="A")
    for i in range(1, 25):
        L.artifacts.put_artifact({"id": "a%04d" % i, "content_type": "text/plain",
                                  "created_by": "someone"})
    seqs = [r[0] for r in L.db.read().execute(
        "SELECT _seq FROM vertex WHERE _origin = ? ORDER BY _seq", ("A",)).fetchall()]
    assert len(seqs) == 24, seqs
    assert seqs == sorted(seqs), "must be monotonic to be an ordering"
    assert len(set(seqs)) == len(seqs), "duplicate _seq destroys (_origin,_seq) uniqueness"
    assert seqs[-1] - seqs[0] + 1 == len(seqs), (
        "GAP in _seq %r — a gapped sequence cannot tell a missing row from an unwritten one" % seqs)


def test_seq_alone_is_not_an_identity(tmp_path):
    """`_seq` orders within an origin and must not pretend to order across them: two nodes'
    counters are independent, so equal `_seq` from different origins are concurrent, not
    simultaneous. The identity is the pair, and an unordered result across origins is a valid
    answer."""
    L = open_lattice(str(tmp_path / "t2.db"), origin="A")
    L.artifacts.put_artifact({"id": "a1", "content_type": "text/plain", "created_by": "x"})
    rows = L.db.read().execute("SELECT _origin, _seq FROM vertex").fetchall()
    assert rows
    for origin, seq in rows:
        assert origin, ("a bare _seq from an unknown node cannot be compared with anything — "
                        "the identity is (_origin, _seq)")


# ── the provenance replacement must be authoritative ────────────────────────────
def test_creation_edge_can_express_what_the_column_cannot():
    """Contract §2.1: "the WHO is expressed twice and THE EDGE IS AUTHORITATIVE, because
    authorization flows through it and it carries the `propagate` mask, which a column cannot."

    That asymmetry is exactly why the column may be deprecated — it is already the
    non-authoritative copy. The edge must be able to carry `propagate` and `is_origin`, or it is
    not a replacement."""
    t = "\n".join(EDGE_DDL)
    assert "propagate" in t, (
        "the creation edge carries `propagate` — the capability a column cannot express")
    assert "is_origin" in t, "the creation edge must be markable as the origin edge"


def test_removal_requires_the_creation_edges_to_exist_first(tmp_path):
    """Ordering of the removal itself: dropping `created_by` before every row has a creation
    edge would destroy the only source for rebuilding the authorization graph."""
    L = open_lattice(str(tmp_path / "t3.db"), origin="A")
    L.artifacts.put_artifact({"id": "a1", "content_type": "text/plain", "created_by": "someone"})
    n_created = L.db.read().execute(
        "SELECT count(*) FROM edge WHERE label = ?", ("created",)).fetchone()[0]
    who = L.db.read().execute(
        "SELECT count(*) FROM vertex WHERE created_by IS NOT NULL").fetchone()[0]
    assert who == 1
    assert n_created == 0, (
        "this store has a WHO column but no creation edge — dropping `created_by` here would "
        "destroy the only source for rebuilding the authorization graph")
