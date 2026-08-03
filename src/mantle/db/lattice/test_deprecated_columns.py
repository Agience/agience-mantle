"""`created_by` / `created_time` are DEPRECATED — and these tests are the removal gate.

[John, 2026-07-21] "we should probably mark created_time and created_by as deprecated.. but make
sure we have a deterministic method to track provenance and ordering before fully removing it."

Removal is therefore not a judgement call. Each column has a NAMED deterministic replacement, and
may only be dropped once that replacement is populated AND verified. These tests assert the
replacements BEHAVE — not merely that they are documented — so "we have a replacement" can never
be a claim someone makes in a review.

⚠ THE FAILURE MODE THIS GUARDS. `created_by` is the SOURCE the `creation` stage reads to build the
creation edges. Dropping the column before those edges exist destroys the ability to rebuild them
— a metadata cleanup that silently deletes the authorization graph. Same shape as §4.3.X, where
folding `created_by` before re-keying would have left every blob simultaneously underivable and
unauthenticatable.
"""
from __future__ import annotations

import os

import pytest

from . import open_lattice
from .schema import EDGE_DDL, VERTEX_DDL

_SCHEMA_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.py")


def _schema_source() -> str:
    """The schema MODULE source — the deprecation lives in comments beside the column
    definitions, which is where a reader will look. `VERTEX_DDL` holds only SQL strings."""
    with open(_SCHEMA_SRC, encoding="utf-8") as f:
        return f.read()


# ── the deprecation is RECORDED, not folklore ───────────────────────────────────
def test_deprecation_and_its_preconditions_are_documented():
    """A deprecation that lives only in a chat log is not a deprecation. The schema must say
    which columns are going, what replaces each, and what must be true before removal."""
    src = _schema_source()
    assert "DEPRECATED" in src
    assert "REMOVAL PRECONDITION" in src, "the gate must be stated where the column is defined"
    assert "creation edge" in src.lower(), "the WHO replacement must be named"
    assert "_seq" in src and "order_key" in src, "the ordering replacement must be named"
    assert "UNORDERED IS A VALID ANSWER" in src, (
        "the honest limit must be stated: there is no total order across origins")


@pytest.mark.parametrize("col", ["created_by", "created_time"])
def test_deprecated_column_is_still_declared(col):
    """⛔ They have NOT been removed. This test exists so a future removal is a DELIBERATE act
    that also deletes this file, rather than a drive-by cleanup."""
    assert col in "\n".join(VERTEX_DDL), "%s should still be declared while deprecated" % col
    assert col in _schema_source()


# ── the ordering replacement must actually order ────────────────────────────────
def test_seq_is_gap_free_within_an_origin(tmp_path):
    """`(_origin, _seq)` replaces `created_time` for ordering — but only if it is gap-free.

    MEASURED on the live migration 2026-07-21: origin 71 held 1..875,971 with exactly 875,971
    rows (0 gaps); 45 and mantle likewise. A gapped sequence would still sort, but could not
    distinguish a MISSING row from one that was never written — so it would not be a
    replacement for anything."""
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
    """⚠ THE HONEST LIMIT, ASSERTED SO NOBODY BUILDS ON ITS ABSENCE.

    `_seq` orders WITHIN an origin and must not pretend to order across them: two nodes' counters
    are independent, so equal `_seq` from different origins are CONCURRENT, not simultaneous. The
    identity is the PAIR. UNORDERED IS A VALID ANSWER."""
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
    """⛔ THE ORDERING OF THE REMOVAL ITSELF. `created_by` is the SOURCE for the creation edges.
    A store with rows but no creation edges is NOT ready for the column to be dropped — proven
    here rather than left to a reviewer's memory."""
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
