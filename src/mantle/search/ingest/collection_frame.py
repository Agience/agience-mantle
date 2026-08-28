# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
# ---------------------------------------------------------------------------

"""A collection's frame, and the proximity digest taken off it.

The frame is what a Marchenko-Pastur spectral-proximity read (the spectral-proximity instrument, or any
instrument sharing its shape) reads: **rows are the collection's artifacts, columns are terms,
cells are raw term counts.** Nothing else — no weighting, no normalisation, no vocabulary cut.
This module builds it from the tokenizer Mantle already runs at index time, and says when the
read has to be taken again. It does not take the read itself: mantle carries no proximity
instrument of its own (mantle imports no spectral library — the two-tier boundary beacon's own
modules hold), so :func:`digest_frame` takes one as an argument, the same injected-embodiment
seam `prism.instrument` uses elsewhere in this codebase.

Why the frame is counts
-----------------------
Measured over the real node-71 store — 82 collections, 2.6M artifacts, median 325 artifacts
per collection — raw counts recall at 0.804 against ppmi's 0.798 and pmi's 0.792. Counts also
carry no marginal, so the local-vs-global question a re-weighting would raise does not arise:
a cell is how many times this artifact used this stem, and that is a fact about the artifact
rather than a statement about the corpus it happens to sit in. ppmi's advantage would have had
to pay for a corpus statistic; it does not have one to pay with.

Why this module is HERE and not under ``search/mantle/sse/``
------------------------------------------------------------
Two constraints meet, and only one directory satisfies both.

* It reads the SSE tokenizer, because the frame's columns must be the same stems the lexical
  arm indexes. A frame built by a second tokenizer would be a second opinion about what a term
  is, and the digest would stop describing the collection the index describes.
* It computes a spectral read, so it imports numpy. ``search/mantle/sse/`` is numpy-free by
  install contract (``pyproject.toml``'s ``[lexical]`` extra) and that contract is enforced by
  removing numpy from the import system: ``tests/test_lexical_extra_is_numpy_free.py`` imports
  ``mantle.search.mantle.sse``, ``search.query_parser``, ``search.types`` and ``mantle.db``
  under a meta-path blocker. Putting the frame builder there would make the lexical arm
  require numpy — the exact defect that test exists to catch.

``search/ingest/`` is the directory whose job is "artifact text becomes index input", which is
precisely what a frame is, and it sits outside that closure: no module the blocker test imports
reaches ``mantle.search.ingest`` (the only mention anywhere in the SSE tree is a docstring
cross-reference in ``search/mantle/custody.py``). The import edge runs one way — this module
imports the tokenizer, the tokenizer imports nothing of this — so the lexical arm stays
numpy-free with the frame builder beside it.

What a row is
-------------
A row is a member the lexical arm would index: one that tokenizes to at least one stem. A
member with no analyzable text is not a zero row, because a zero row is not free —
``<probe>.common_prefix``'s own reasoning measures what appending zero rows costs
(they drag the median row energy, hence ``sigma^2``, hence every mode) and concludes that
padding a frame is not a no-op. The lexical arm already draws this line at
``pipeline_unified._sse_index_artifact`` (``if not fields: return ARM_SKIPPED``); this draws
the same one, so the frame's rows are the collection's indexed artifacts and not a wider set.

Row and column ORDER carry no claim. The read is invariant to row permutation (the singular
values of a row-permuted frame are its own) and to column permutation (right multiplication by
a permutation is orthogonal; the per-channel median, ``effective_width`` and the row energies
are all symmetric in the columns). Both are sorted anyway, because determinism is a promise
this module makes to the bit and sorting is how it keeps it without asserting anything about
membership order.

What is refused, and why refusal is not a threshold
---------------------------------------------------
:class:`FrameNotDigestible` is raised — never a record — in three cases, each of which is an
exact condition rather than a cut:

``NO_ROWS``
    The collection has no indexed member. There is no frame, so there is nothing to read.

``NO_RESOLVED_DIRECTION``
    The read is identically zero. This is forced for a one-artifact collection: centring a
    single row against its own median gives the zero row, whose only singular value is 0, and
    the Marchenko-Pastur prediction at ``k = N = 1`` sits at the bottom of a spectrum with no
    mass, so the read is exactly ``[0.0]``. Measured, the three single-artifact collections in
    the real population are therefore mutually at distance exactly 0 — they do not resemble
    each other, they are silent, and storing three colliding records would say the first thing
    while meaning the second. ``numpy.any`` is an exact test against zero and not a tolerance:
    a read with *any* energy anywhere is a real record, however small, and is kept.

``ENUMERATION_TRUNCATED``
    The caller could not enumerate the whole collection, so the rows it did supply are a
    prefix and not the collection. See below.

``DOES_NOT_FIT``
    The dense frame could not be allocated. 96.6% of the measured corpus's artifacts live in
    12 collections of 10,000 members or more, up to 1.84M rows; ``centre`` densifies, so such
    a frame is at least ``N^2`` cells (the measured shapes all have ``D >= N``) — 27 TB at
    1.84M rows. "Too large" is decided by the machine refusing the allocation, which is a
    measurement, not a cap: there is no number in this module a person could move, and the
    same collection digests on a host that can hold it. See :func:`build_frame`.

Refusal is recorded as a PROPERTY of the collection, which is the whole architecture: a
property that is true of a frame belongs in the lattice as something to narrow on, not in the
distance as something to compensate for.

Where the real ceiling on an oversized collection sits, measured in the code
---------------------------------------------------------------------------
It is not memory. ``db.lattice_api.list_collection_artifacts`` resolves a collection through
``_membership_edges`` (``lattice_api.py:485``), which is ``db.graph.edges_of(collection_id,
direction="out")`` — and ``edges_of`` takes ``limit: int = 1000`` and offers **no cursor**
(``db/edge.py:418``). Above a thousand members the store therefore has no complete enumeration
of a collection to give anybody, and a frame built from what it does give would be an
arbitrary 1000-row prefix presented as the collection: a fabricated measurement, which the
store's own counter design (``db/schema.c_edge_label_built``) names as worse than the slow
query it would replace.

That is why :func:`digest_collection` takes ``exhaustive`` as a REQUIRED argument rather than
inferring it. The caller is the only party that knows whether its enumeration finished, and
the store already has this exact idiom — ``db.vertex.list_by_content_type`` returns
``(docs, exhaustive)`` and says so for the same reason. A caller that cannot answer must not
get a digest, and gets ``ENUMERATION_TRUNCATED`` instead of a plausible one.

**No substructure was invented to work around this, because none is available.** The lattice
was searched for something that could name a large collection's parts:

* **Context edges** (``db/edge.py:47``) are a real relation with `set_context` / `members_of`
  behind them, and they have **zero production writers** — the module says so at
  ``db/edge.py:62``. A live store has none. They are also explicitly many-to-many
  (``db/edge.py:285``: "A node may sit in several contexts"), so they tag rather than
  partition, and ``context_edges`` reads through the same capped ``edges_of``.
* **Sub-collections** are expressible — ``add_artifact_to_collection`` type-checks nothing on
  ``dst`` — but ``db/schema.py:163`` records that no collection nests today, ``origin_root``
  is written assuming depth 1, and ``list_collection_artifacts`` never descends. Nesting is a
  shape the model allows and the store does not contain.
* **Time.** ``modified_time`` is doc-JSON with no column and no index; ``created_time`` is a
  column, unindexed, and marked deprecated; ``(_origin, _seq)`` is the real time primitive but
  is not collection-scoped and is ``UNORDERED`` across origins by construction
  (``db/constants.py:199``).
* **``_leaf``** (``db/constants.py:139``) genuinely is disjoint and covering and cheap. It is
  ``blake2b(id) mod leaves`` — a hash bucket. Splitting a frame by it would produce sub-frames
  that are uniform random samples of the collection, and a digest of a random sample is a
  digest of nothing anyone named. That is precisely the invented partition this work refuses.
* **``root_id``** partitions rows into version lineages, but ``list_collection_artifacts``
  has already collapsed each lineage to its current version, so it is not a partition of the
  frame's rows at all.

So the honest position is recorded rather than engineered around: **a collection the store
cannot enumerate does not get a digest, and "does not get a digest" is the property.** It
becomes digestible the day something names its parts — a written context edge, a real
sub-collection, or a cursor on ``edges_of`` — and nothing here has to change for that to work,
because the refusal is a fact about the enumeration and not a rule about size.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

from mantle.search.mantle.sse.tokenizer import tokenize

#: The frame convention these digests are stated against — the tokenizer, the cell rule, and
#: the row rule together. A different value means a different instrument, for the same reason
#: the digest's own ``engine_id`` field is one: a caller holding a bare array of numbers must
#: be able to tell what they were computed over. Changing the stemmer changes the columns, so
#: `sse/tokenizer.py`'s own rule applies here too — every digest must be retaken, and there is
#: no in-place migration.
FRAME_ID = "mantle.collection.counts.v1"

__all__ = [
    "FRAME_ID",
    "CollectionDigest",
    "CollectionFrame",
    "FrameNotDigestible",
    "build_frame",
    "digest_collection",
    "digest_frame",
    "digest_is_spent",
    "loads_digest",
    "dumps_digest",
]


class FrameNotDigestible(Exception):
    """A collection that has no digest, and the exact reason it has none.

    Carried as an exception rather than returned as a null record so that "this collection has
    no proximity record" cannot be confused with "this collection's record happens to be all
    zeros" — which is the collision the ``NO_RESOLVED_DIRECTION`` case exists to refuse.
    """

    NO_ROWS = "no indexed member"
    NO_RESOLVED_DIRECTION = "the read resolves no direction"
    ENUMERATION_TRUNCATED = "the collection could not be enumerated in full"
    DOES_NOT_FIT = "the dense frame does not fit in memory"

    def __init__(self, reason: str, collection_id: str = "") -> None:
        self.reason = reason
        self.collection_id = collection_id
        super().__init__(
            f"no proximity digest for {collection_id or 'this collection'}: {reason}"
        )


@dataclass(frozen=True)
class CollectionFrame:
    """One collection's term-count frame, with the labels of both axes kept beside it.

    ``matrix`` is ``(len(artifact_ids), len(terms))`` float64 raw counts. The labels are kept
    because a frame that cannot say which artifact a row is cannot be checked against the
    collection it claims to describe; they are NOT part of the digest, which keeps only the
    read.
    """

    artifact_ids: Tuple[str, ...]
    terms: Tuple[str, ...]
    matrix: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.artifact_ids)

    @property
    def columns(self) -> int:
        return len(self.terms)


@dataclass(frozen=True)
class CollectionDigest:
    """The stored record: what was read, what it was read over, and by which instrument.

    ``read`` is kept at FULL length. The Marchenko-Pastur deviation is dense — measured 64 of
    64 nonzero modes on 60 of 60 corpus frames — so the trailing-zero truncation the clipped
    excess allowed is not available, and ``<probe>.common_prefix`` compares two
    records on the shorter one's length rather than padding either.

    ``rows`` is redundant with ``len(read)`` whenever ``D >= N`` (measured on 38 of 38 real
    shapes, where ``min(N, D) == N``) and it is stored anyway, because it is the property the
    narrowing runs on and reading it off the length would be an inference rather than a fact.
    It is not a new exposure: the store already carries ``col:<id>`` in the ``counter`` table
    (``db/schema.c_collection``, read O(1) by ``db.vertex.count_in_collection``) and
    ``db.lattice_api.list_collection_artifacts`` returns the members outright, and both need
    only the store handle — strictly less than the collection key this digest is sealed under.
    So anyone who can open a digest could already count the collection, and anyone who cannot
    open it learns nothing from its length.
    """

    collection_id: str
    frame_id: str
    engine_id: str
    rows: int
    columns: int
    read: Tuple[float, ...]


# ═══════════════════════════════════════════════════════════════════════════
# The frame
# ═══════════════════════════════════════════════════════════════════════════

def _counts(fields: Mapping[str, str]) -> Counter:
    """Stem counts over every field of one artifact, through the SSE tokenizer.

    Fields are SUMMED rather than kept apart. The lexical index keeps them apart because a
    field is where a match happened and that decides a boost; a frame has no boosts, and a
    stem's count in an artifact is a fact about the artifact rather than about which of its
    fields carried it. Summing is also what makes the frame's columns the artifact's terms —
    the same set ``sse/indexer`` writes postings for, once rather than four times.
    """
    out: Counter = Counter()
    for text in fields.values():
        if text:
            out.update(tokenize(text))
    return out


def build_frame(
    members: Iterable[Tuple[str, Mapping[str, str]]],
    *,
    collection_id: str = "",
) -> CollectionFrame:
    """Build the collection's frame from ``(artifact_id, fields)`` pairs.

    ``fields`` is the per-field text dict ``pipeline_unified._extract_artifact_fields``
    produces and hands the lexical arm, so the frame is built from exactly the text that got
    indexed — not from a second extraction that could disagree with it.

    Raises :class:`FrameNotDigestible` with ``NO_ROWS`` when no member tokenizes to anything,
    and with ``DOES_NOT_FIT`` when the dense frame cannot be allocated.

    **The two-pass shape is what makes the size question answerable without a cap.** Counting
    is sparse and costs one dict per member, so a 1.84M-member collection gets that far; the
    dense array is materialised in one allocation afterwards, and it is that allocation the
    machine either serves or refuses. There is no row limit here to move, and no host-specific
    number is written down: the same collection digests wherever the frame fits and does not
    where it does not, which is the honest content of "too large to digest as one frame".
    """
    rows: List[Tuple[str, Counter]] = []
    vocabulary: set = set()
    for artifact_id, fields in members:
        counts = _counts(fields or {})
        if not counts:
            continue
        rows.append((str(artifact_id), counts))
        vocabulary.update(counts)

    if not rows:
        raise FrameNotDigestible(FrameNotDigestible.NO_ROWS, collection_id)

    rows.sort(key=lambda pair: pair[0])
    terms = tuple(sorted(vocabulary))
    column_of: Dict[str, int] = {term: i for i, term in enumerate(terms)}

    try:
        matrix = np.zeros((len(rows), len(terms)), dtype=np.float64)
    except MemoryError as exc:
        # numpy raises `_ArrayMemoryError`, a MemoryError, when the allocation cannot be
        # served. That refusal IS the answer to "is this collection too large to digest as one
        # frame" — a measurement of this host, in place of a constant nobody can derive.
        raise FrameNotDigestible(
            FrameNotDigestible.DOES_NOT_FIT, collection_id,
        ) from exc

    for i, (_artifact_id, counts) in enumerate(rows):
        for term, n in counts.items():
            matrix[i, column_of[term]] = n

    return CollectionFrame(
        artifact_ids=tuple(a for a, _ in rows), terms=terms, matrix=matrix,
    )


# ═══════════════════════════════════════════════════════════════════════════
# The read
# ═══════════════════════════════════════════════════════════════════════════

def digest_frame(frame: CollectionFrame, *, read, engine_id: str,
                 collection_id: str = "") -> CollectionDigest:
    """Take a proximity read off a frame, or refuse.

    ``read`` and ``engine_id`` name the proximity instrument to read with. This module carries
    none of its own: mantle imports no spectral library (the same two-tier boundary beacon's own
    modules hold), so the instrument is injected, the same seam ``prism.instrument`` uses
    elsewhere in this codebase. Pass ``read=<probe>.mp_deviation,
    engine_id=<probe>.ENGINE_ID_PROXIMITY`` (or any instrument sharing that
    shape — a callable frame -> array of per-mode deviations).

    Refuses with ``NO_RESOLVED_DIRECTION`` when the read is identically zero — see the module
    docstring for why a one-artifact collection is forced into that case and why returning the
    record anyway would be a collision dressed as a resemblance.
    """
    got = read(frame.matrix)
    if not bool(np.any(got)):
        raise FrameNotDigestible(
            FrameNotDigestible.NO_RESOLVED_DIRECTION, collection_id,
        )
    return CollectionDigest(
        collection_id=str(collection_id),
        frame_id=FRAME_ID,
        engine_id=str(engine_id),
        rows=frame.rows,
        columns=frame.columns,
        read=tuple(float(x) for x in got),
    )


def digest_collection(
    members: Iterable[Tuple[str, Mapping[str, str]]],
    *,
    exhaustive: bool,
    read,
    engine_id: str,
    collection_id: str = "",
) -> CollectionDigest:
    """:func:`build_frame` then :func:`digest_frame`. Deterministic, bit-identical on repeat.

    ``read`` and ``engine_id`` are :func:`digest_frame`'s — the proximity instrument, injected
    by the caller.

    ``exhaustive`` states whether ``members`` is the WHOLE collection. It has no default,
    because a default would be an assumption about somebody else's enumeration: the store's
    own member read tops out at ``edges_of``'s ``limit=1000`` with no cursor
    (``db/edge.py:418``), so above that the caller genuinely does not have the collection and
    must say so. ``False`` refuses with ``ENUMERATION_TRUNCATED`` before any work is done —
    a digest of a prefix would be a fabricated measurement of the whole.
    """
    if not exhaustive:
        raise FrameNotDigestible(
            FrameNotDigestible.ENUMERATION_TRUNCATED, collection_id,
        )
    return digest_frame(
        build_frame(members, collection_id=collection_id),
        read=read, engine_id=engine_id, collection_id=collection_id,
    )


# ═══════════════════════════════════════════════════════════════════════════
# When to take it again
# ═══════════════════════════════════════════════════════════════════════════

def digest_is_spent(rows: int, events_since: int) -> bool:
    """Has the digest's evidence run out? ``events_since >= rows``.

    **What invalidates a digest.** Every membership change and every member edit changes the
    frame: an add or a remove changes the row set, and an edit changes a row's counts and may
    add or drop columns. There is no event on the write path that leaves the frame alone, so
    there is nothing to be selective about — the question is only how many such events a
    digest may absorb before it is retaken.

    **Why the answer is ``rows``, derived twice and not chosen.**

    *From the evidence.* A digest is a statement about exactly ``rows`` rows. One invalidating
    event makes at most one of them wrong. After ``rows`` events every row the digest was taken
    over may have been replaced, so the digest no longer describes any part of the frame it
    came from: its evidence is spent, in the literal sense that the count of surviving original
    rows has reached zero. That is an exact point, not a fraction of one — a rule that fired at
    some proportion of ``rows`` would need the proportion, and the proportion is the constant.

    *From the cost.* A digest costs Theta(rows): one member read and one tokenization per row,
    then an SVD. A write costs Theta(1). Recomputing per write therefore makes indexing
    Theta(rows) per write — on a 3,000-member collection, three thousand times the cost of the
    write that triggered it, which is why it is not viable. Amortising the digest over ``k``
    writes gives Theta(rows/k) per write, and ``k = rows`` is the unique choice that brings it
    back to Theta(1) — the same order as the write itself. Smaller and indexing is superlinear
    in collection size; larger and staleness grows without bound relative to the collection.

    Both derivations land on the same rule, from opposite directions, and neither names a
    number. A never-digested collection has ``rows == 0`` and is spent at ``events_since == 0``,
    so the first write to a collection takes the first digest — the base case falls out rather
    than being special-cased.

    **What this costs, stated.** A digest may be up to one full turnover behind. Proximity only
    ever narrows (see :mod:`mantle.search.mantle.collection_proximity`), so a stale digest costs
    recall and can never admit an artifact the light cone did not; that is the same trade the
    token narrowing takes when a lookup fails, and it is why the lax end of the range is the
    safe one.
    """
    return int(events_since) >= int(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Serialization — canonical, so a re-encode is byte-identical
# ═══════════════════════════════════════════════════════════════════════════

def dumps_digest(digest: CollectionDigest) -> bytes:
    """Canonical JSON bytes. Sorted keys, no whitespace, exact float round trip.

    ``json`` writes a float through ``repr``, which is the shortest string that reads back to
    the same float64, so a digest survives the round trip to the bit — which is what lets
    ``spectral_distance`` on a stored record equal ``spectral_distance`` on the one that was
    just computed.
    """
    return json.dumps(
        {
            "collection_id": digest.collection_id,
            "frame_id": digest.frame_id,
            "engine_id": digest.engine_id,
            "rows": digest.rows,
            "columns": digest.columns,
            "read": list(digest.read),
        },
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def loads_digest(blob: bytes) -> CollectionDigest:
    """Inverse of :func:`dumps_digest`. Raises ``ValueError`` on anything else."""
    try:
        doc = json.loads(bytes(blob).decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"not a proximity digest: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError("not a proximity digest: top level is not an object")
    try:
        return CollectionDigest(
            collection_id=str(doc["collection_id"]),
            frame_id=str(doc["frame_id"]),
            engine_id=str(doc["engine_id"]),
            rows=int(doc["rows"]),
            columns=int(doc["columns"]),
            read=tuple(float(x) for x in doc["read"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"not a proximity digest: {exc}") from exc
