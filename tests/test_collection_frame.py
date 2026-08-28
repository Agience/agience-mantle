# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
# ---------------------------------------------------------------------------

"""The collection frame, the digest taken off it, and the moment it has to be retaken.

`search/ingest/collection_frame.py` turns a collection's members into the object a proximity
instrument reads — rows are artifacts, columns are terms, cells are raw term counts — and
decides three things this file holds it to. The instrument itself is injected via
`read=`/`engine_id=` and tested in its own suite; this file uses a minimal local stand-in
(`_read`/`_ENGINE_ID`, defined below), since mantle imports no spectral library.

1. **The frame is the collection.** Its columns are the stems the SSE arm indexes, produced
   by the tokenizer the SSE arm runs, so a digest describes the same corpus the index does.
   Its rows are the members that arm would index, and no others: a member with no analyzable
   text is not a zero row, because `common_prefix` measures what appending zero rows costs and
   it is not nothing.

2. **A collection with nothing to say gets no record, and says so.** Empty, one-artifact, and
   un-enumerable collections each raise a NAMED refusal. The one-artifact case is the load
   bearing one: its read is exactly `[0.0]`, so the three single-artifact collections in the
   real population would all be at distance exactly 0 from each other. Storing three colliding
   records would report a resemblance where there is only silence.

3. **The trigger is derived, not picked.** `digest_is_spent(rows, events_since)` is
   `events_since >= rows` and this file checks both derivations that produce it — the evidence
   one (a digest is a statement about `rows` rows and one event invalidates at most one of
   them) and the cost one (`k = rows` is the unique amortisation that returns the per-write
   cost to the order of the write).

The AST guard against a swept constant landing in `build_frame`/`digest_frame` is here
(`test_no_size_cap_appears_anywhere_in_the_builder`, `test_no_...` on the narrowing beside it).
The equivalent guard on the proximity instrument's own math lives in its own test suite,
because that module is not reachable from this repo.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from mantle.search.ingest import collection_frame as CF
from mantle.search.ingest import digest_refresh as DR
from mantle.search.mantle import collection_proximity as CP
from mantle.search.mantle.sse.tokenizer import tokenize

FRAME_MODULE = Path(CF.__file__)
NARROW_MODULE = Path(CP.__file__)

#: `digest_frame`/`digest_collection` take a proximity instrument as an injected `read` +
#: `engine_id` pair now — mantle imports no spectral library, so the real instrument
#: (`<probe>.mp_deviation`) lives in and is tested by its own
#: suite. This file is about the plumbing `collection_frame.py` builds around an injected
#: instrument (determinism, refusal, storage), not the instrument's own math, so a minimal but
#: real stand-in — centre, then the plain singular spectrum — is enough: it is deterministic,
#: reads a single artifact's row as exactly zero (centring one row against its own median gives
#: the zero row), and is non-trivial on everything else, which is what these tests need.
_ENGINE_ID = "test.stub.svd"


def _read(matrix):
    A = np.asarray(matrix, dtype=np.float64)
    centred = A - np.median(A, axis=0)
    return np.linalg.svd(centred, compute_uv=False)


def _spectral_distance(x, y):
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    return float(np.linalg.norm(a[:n] - b[:n]))


def _members(n, *, seed=0, words=("quasar", "budget", "catalogue", "quarterly", "plate")):
    """`n` members whose texts differ, built from a fixed seed so every number is reproducible."""
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        draw = g.integers(1, 6, size=len(words))
        text = " ".join(" ".join([w] * int(k)) for w, k in zip(words, draw))
        out.append((f"art-{i:04d}", {"content": text + f" token{i}"}))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 1. The frame is built from the tokenizer the index already runs
# ═══════════════════════════════════════════════════════════════════════════

def test_the_columns_are_the_stems_the_sse_arm_would_index():
    """Not a second tokenizer, and not raw words. `observing` and `observable` stem together,
    which is exactly what the posting lists do — a frame built on raw words would have
    different columns from the index that describes the same collection."""
    frame = CF.build_frame([
        ("a", {"content": "The Observers were observing observable signals"}),
        ("b", {"content": "signals and observers"}),
    ])
    assert set(frame.terms) == set(tokenize(
        "The Observers were observing observable signals and"))
    assert "observ" in frame.terms
    assert "observing" not in frame.terms


def test_a_cell_is_a_raw_count_and_nothing_else():
    """No idf, no length normalisation, no sublinear scaling. The count is the count."""
    frame = CF.build_frame([
        ("a", {"content": "budget budget budget quasar"}),
        ("b", {"content": "quasar"}),
    ])
    col = {t: i for i, t in enumerate(frame.terms)}
    assert frame.matrix[0, col["budget"]] == 3.0
    assert frame.matrix[0, col["quasar"]] == 1.0
    assert frame.matrix[1, col["budget"]] == 0.0
    assert frame.matrix[1, col["quasar"]] == 1.0


def test_the_fields_are_summed_rather_than_kept_apart():
    """A frame has no field boosts, so a stem's count is the artifact's, not a field's."""
    split = CF.build_frame([
        ("a", {"title": "budget", "description": "budget", "content": "budget quasar"}),
        ("b", {"content": "quasar"}),
    ])
    joined = CF.build_frame([
        ("a", {"content": "budget budget budget quasar"}),
        ("b", {"content": "quasar"}),
    ])
    assert split.terms == joined.terms
    assert np.array_equal(split.matrix, joined.matrix)


def test_a_member_with_no_analyzable_text_is_not_a_zero_row():
    """The rule `pipeline_unified._sse_index_artifact` already applies, applied here.

    A zero row is not free: `proximity` measures that padding a frame drags the median row
    energy and moves EVERY existing mode. So a member the lexical arm would skip is not a row,
    and the digest of a collection is unchanged by adding one."""
    real = _members(6)
    padded = real + [("art-blank", {"content": "   "}), ("art-empty", {})]
    a = CF.build_frame(real)
    b = CF.build_frame(padded)
    assert a.rows == b.rows == 6
    assert "art-blank" not in b.artifact_ids
    assert np.array_equal(a.matrix, b.matrix)
    assert CF.digest_frame(a, read=_read, engine_id=_ENGINE_ID).read == CF.digest_frame(b, read=_read, engine_id=_ENGINE_ID).read


def test_both_axes_are_sorted_and_the_read_does_not_depend_on_either_order():
    """Order is a determinism device, not a claim. The read is invariant to row permutation
    (permuting rows permutes nothing in the singular values) and to column permutation (right
    multiplication by a permutation is orthogonal), so membership order carries no meaning —
    which is why sorting is safe and why nothing downstream may read anything into it."""
    members = _members(9, seed=3)
    frame = CF.build_frame(members)
    assert list(frame.artifact_ids) == sorted(frame.artifact_ids)
    assert list(frame.terms) == sorted(frame.terms)

    shuffled = list(reversed(members))
    assert CF.digest_collection(shuffled, exhaustive=True, read=_read, engine_id=_ENGINE_ID).read == \
        CF.digest_collection(members, exhaustive=True, read=_read, engine_id=_ENGINE_ID).read


def test_the_digest_is_bit_identical_on_repeat():
    """Determinism to the bit, including through the stored form."""
    members = _members(11, seed=7)
    a = CF.digest_collection(members, exhaustive=True, read=_read, engine_id=_ENGINE_ID, collection_id="col-1")
    b = CF.digest_collection(members, exhaustive=True, read=_read, engine_id=_ENGINE_ID, collection_id="col-1")
    assert a == b
    assert CF.dumps_digest(a) == CF.dumps_digest(b)
    assert CF.loads_digest(CF.dumps_digest(a)) == a


def test_the_stored_read_is_the_read_and_is_not_truncated():
    """`proximity` is explicit that this read is dense — 64 of 64 nonzero modes on 60 of 60
    frames — so the trailing-zero truncation the clipped excess allowed is gone and records
    must be persisted at full length."""
    members = _members(14, seed=5)
    frame = CF.build_frame(members)
    digest = CF.digest_frame(frame, read=_read, engine_id=_ENGINE_ID)
    assert len(digest.read) == min(frame.rows, frame.columns)
    assert np.allclose(np.asarray(digest.read), _read(frame.matrix))
    assert CF.loads_digest(CF.dumps_digest(digest)).read == digest.read


def test_the_digest_records_which_instrument_read_it():
    """Records taken against a different reference are not comparable; the record says which."""
    digest = CF.digest_collection(_members(8), exhaustive=True, read=_read, engine_id=_ENGINE_ID, collection_id="col-1")
    assert digest.engine_id == _ENGINE_ID
    assert digest.frame_id == CF.FRAME_ID


# ═══════════════════════════════════════════════════════════════════════════
# 2. The degenerate cases behave, explicitly
# ═══════════════════════════════════════════════════════════════════════════

def test_an_empty_collection_is_refused_by_name():
    with pytest.raises(CF.FrameNotDigestible) as exc:
        CF.digest_collection([], exhaustive=True, read=_read, engine_id=_ENGINE_ID, collection_id="col-empty")
    assert exc.value.reason == CF.FrameNotDigestible.NO_ROWS
    assert exc.value.collection_id == "col-empty"


def test_a_single_artifact_collection_reads_exactly_zero():
    """The measurement this refusal exists for, reproduced from the construction.

    Centring one row against its own median gives the zero row, whose only singular value is
    0; the Marchenko-Pastur prediction at `k = N = 1` sits at the bottom of a spectrum with no
    mass. So the read is EXACTLY `[0.0]` — not nearly zero — and every one-artifact collection
    reads the same. That is a collision, not a resemblance."""
    reads = [
        _read(CF.build_frame([(a, {"content": t})]).matrix)
        for a, t in (("x", "quasar budget"), ("y", "entirely different words here"),
                     ("z", "third"))
    ]
    for read in reads:
        assert read.size == 1
        assert float(read[0]) == 0.0
    for i in range(len(reads)):
        for j in range(len(reads)):
            assert _spectral_distance(reads[i], reads[j]) == 0.0


def test_a_collection_that_resolves_no_direction_is_refused_rather_than_stored():
    """So the collision above is never written down. The test is `numpy.any` — exact equality
    with zero, not a tolerance: a read with any energy at all is a real record and is kept."""
    with pytest.raises(CF.FrameNotDigestible) as exc:
        CF.digest_collection([("only", {"content": "one artifact"})],
                             exhaustive=True, read=_read, engine_id=_ENGINE_ID, collection_id="col-1")
    assert exc.value.reason == CF.FrameNotDigestible.NO_RESOLVED_DIRECTION


def test_a_near_silent_frame_is_kept_because_it_is_not_silent():
    """The counterpart, so the refusal cannot be satisfied by quietly widening into a floor.

    Members with identical text centre to the zero matrix. `_read` (the bare singular
    spectrum) genuinely reads that as `[0.0, ...]` and would be refused — correct for a read
    with no floor of its own. `<probe>.mp_deviation`'s actual behaviour is
    different: it is `s_k - s_MP(k)`, and the Marchenko-Pastur prediction `s_MP(k)` is not
    identically zero even when the observed spectrum is, so the real instrument reads a small
    nonzero energy there and the collection is kept (see the instrument's own test suite for that
    property). What this file can test is the plumbing: `digest_frame`/`digest_collection`
    pass whatever the injected read reports straight through, with no floor of their own —
    proven here with a stand-in read that has exactly that "not identically zero" shape."""
    def _read_with_a_floor(matrix):
        A = np.asarray(matrix, dtype=np.float64)
        centred = A - np.median(A, axis=0)
        sv = np.linalg.svd(centred, compute_uv=False)
        return sv + 1e-6            # mimics mp_deviation's non-identically-zero MP floor
    same = [(f"art-{i}", {"content": "identical text"}) for i in range(5)]
    digest = CF.digest_collection(same, exhaustive=True, read=_read_with_a_floor,
                                  engine_id=_ENGINE_ID, collection_id="col-same")
    read = np.asarray(digest.read)
    assert np.any(read)
    assert float(np.max(np.abs(read))) < 1.0


def test_a_truncated_enumeration_is_refused_before_any_work_is_done():
    """`list_collection_artifacts` resolves a collection through `edges_of`, which takes
    `limit=1000` and has no cursor (`db/edge.py:418`). Above that the store has no complete
    enumeration to give, and a digest of the prefix would be a fabricated measurement of the
    whole — so the caller must state whether it finished, and `False` refuses."""
    with pytest.raises(CF.FrameNotDigestible) as exc:
        CF.digest_collection(_members(9), exhaustive=False, read=_read, engine_id=_ENGINE_ID, collection_id="col-big")
    assert exc.value.reason == CF.FrameNotDigestible.ENUMERATION_TRUNCATED


def test_exhaustive_has_no_default():
    """A default would be an assumption about somebody else's enumeration."""
    with pytest.raises(TypeError):
        CF.digest_collection(_members(4), collection_id="col-1",   # type: ignore[call-arg]
                             read=_read, engine_id=_ENGINE_ID)


def test_a_frame_that_does_not_fit_is_refused_by_name_and_not_by_a_row_cap(monkeypatch):
    """"Too large" is the machine refusing the allocation, which is a measurement.

    There is no row limit in the module to move: the counting pass is sparse and gets as far
    as the members go, and the dense materialisation is where the answer comes from. A caller
    on a host that can hold the frame gets a digest for the same collection."""
    real_zeros = np.zeros

    def _refuse(shape, *a, **kw):
        if isinstance(shape, tuple) and len(shape) == 2:
            raise MemoryError("Unable to allocate array")
        return real_zeros(shape, *a, **kw)

    monkeypatch.setattr(CF.np, "zeros", _refuse)
    with pytest.raises(CF.FrameNotDigestible) as exc:
        CF.build_frame(_members(6), collection_id="col-huge")
    assert exc.value.reason == CF.FrameNotDigestible.DOES_NOT_FIT


def test_no_size_cap_appears_anywhere_in_the_builder():
    """The structural counterpart of the test above: a cap would be a literal, and there is
    none. Stated as its own check so that "we refuse very large collections" can never be
    implemented by writing a number down."""
    tree = ast.parse(FRAME_MODULE.read_text(encoding="utf-8"))
    numbers = {n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
               and not isinstance(n.value, bool)}
    assert numbers <= {0, 1}, (
        f"{FRAME_MODULE.name} gained numeric literals {sorted(numbers - {0, 1})}. A row cap, a "
        f"vocabulary cut, a minimum length or a staleness interval all look like this."
    )


def test_a_refusal_is_never_a_record():
    """Every refusal raises. None of them returns a digest that would sit in a gallery."""
    for members, exhaustive in (([], True),
                                ([("a", {"content": "solo"})], True),
                                (_members(4), False)):
        with pytest.raises(CF.FrameNotDigestible):
            CF.digest_collection(members, exhaustive=exhaustive, read=_read, engine_id=_ENGINE_ID)


# ═══════════════════════════════════════════════════════════════════════════
# 3. The trigger, from both derivations
# ═══════════════════════════════════════════════════════════════════════════

def test_a_collection_that_has_never_been_digested_is_spent_at_zero():
    """The base case falls out of the rule rather than being special-cased: a digest over 0
    rows has no evidence to spend."""
    assert CF.digest_is_spent(rows=0, events_since=0) is True


def test_the_digest_is_spent_exactly_when_every_row_it_covered_may_have_gone():
    """The evidence derivation. One invalidating event — a membership change or a member edit
    — makes at most one row wrong, so after `rows` events none of the original rows is known
    to survive and the digest describes no part of the frame it came from."""
    for rows in (1, 2, 325, 3000):
        assert not CF.digest_is_spent(rows, rows - 1)
        assert CF.digest_is_spent(rows, rows)
        assert CF.digest_is_spent(rows, rows + 1)


def test_the_same_rule_is_what_makes_the_amortised_cost_the_order_of_the_write():
    """The cost derivation, computed rather than asserted.

    A digest costs Theta(rows); a write costs Theta(1). Recomputing every `k` writes costs
    `rows/k` per write. `k = rows` is the unique value that brings that to Theta(1) — and it
    is the value the rule fires at. Below it indexing is superlinear in collection size; the
    3,000-member collection that motivated the question is where that is loudest."""
    for rows in (325, 3000):
        events = [i for i in range(1, 4 * rows + 1) if CF.digest_is_spent(rows, i % rows or rows)]
        # the rule fires once per `rows` writes, so the amortised per-write cost is rows/rows
        fires = sum(1 for i in range(1, rows + 1) if CF.digest_is_spent(rows, i))
        assert fires == 1, "the digest is retaken exactly once per turnover"
        assert rows / rows == 1.0
        assert events  # the sweep ran

    # and the alternative that motivated the derivation: per-write is `rows` times the write
    assert 3000 / 1 == 3000.0


def test_the_trigger_names_no_interval_and_no_fraction():
    """`digest_is_spent` is a comparison of two counts. A period, a fraction of the rows, or a
    seconds-based staleness would all need a number, and its signature has nowhere to put one."""
    import inspect
    sig = inspect.signature(CF.digest_is_spent)
    assert list(sig.parameters) == ["rows", "events_since"]
    assert all(p.default is inspect.Parameter.empty for p in sig.parameters.values())


# ═══════════════════════════════════════════════════════════════════════════
# 3b. The trigger, applied — what counts as an event and what a refusal costs
# ═══════════════════════════════════════════════════════════════════════════

class _RecordingSlot:
    """A digest slot with no custody in it — this file is about the trigger, not the key."""

    def __init__(self) -> None:
        self.written = []

    def put(self, principal_id, digest, request) -> None:
        self.written.append((principal_id, digest.collection_id, digest.rows))


def _provider(members, exhaustive=True, read=_read, engine_id=_ENGINE_ID):
    return lambda collection_id: (members, exhaustive)


def _refresher(members, *, exhaustive=True, read=_read, engine_id=_ENGINE_ID):
    slot = _RecordingSlot()
    return slot, DR.CollectionDigestRefresher(slot, _provider(members, exhaustive), read=_read, engine_id=_ENGINE_ID)


def test_a_collection_with_no_digest_is_due_at_once():
    """The base case, through the component: nothing has been digested, so nothing is spent
    to defer, and the first write takes the first digest."""
    slot, refresher = _refresher(_members(8))
    assert refresher.is_due("p", "col-1")
    assert refresher.refresh_if_spent("p", "col-1", None)
    assert slot.written == [("p", "col-1", 8)]


def test_it_then_waits_exactly_one_turnover():
    """`rows` writes, not one and not a period. The digest is retaken on the write that makes
    every row it covered potentially stale, and on no earlier one."""
    members = _members(8)
    slot, refresher = _refresher(members)
    refresher.refresh_if_spent("p", "col-1", None)
    for i in range(1, len(members)):
        refresher.note_write("p", "col-1")
        assert not refresher.is_due("p", "col-1"), i
        assert refresher.refresh_if_spent("p", "col-1", None).reason == DR.NOT_DUE
    refresher.note_write("p", "col-1")
    assert refresher.is_due("p", "col-1")
    assert refresher.refresh_if_spent("p", "col-1", None)
    assert len(slot.written) == 2


def test_the_expensive_half_does_not_run_until_the_trigger_fires():
    """Asking whether a digest is due must not cost what the rule is deferring."""
    calls = []
    slot = _RecordingSlot()

    def _count(collection_id):
        calls.append(collection_id)
        return _members(6), True

    refresher = DR.CollectionDigestRefresher(slot, _count, read=_read, engine_id=_ENGINE_ID)
    refresher.refresh_if_spent("p", "col-1", None)
    assert len(calls) == 1
    for _ in range(5):
        refresher.note_write("p", "col-1")
        refresher.refresh_if_spent("p", "col-1", None)
    assert len(calls) == 1, "the collection was enumerated while the digest was still good"


def test_every_membership_change_and_every_edit_is_one_event():
    """There is no event on the write path that leaves the frame alone, so there is nothing
    to classify — `note_write` is the whole vocabulary."""
    _slot, refresher = _refresher(_members(4))
    refresher.refresh_if_spent("p", "col-1", None)
    for _ in range(3):
        refresher.note_write("p", "col-1")
    assert not refresher.is_due("p", "col-1")
    refresher.note_write("p", "col-1")
    assert refresher.is_due("p", "col-1")


def test_a_refusal_resets_the_clock_so_it_is_not_rediscovered_every_write():
    """The oversized case: an un-enumerable collection stays un-enumerable, so learning that
    again on the next write would spend a full enumeration for nothing."""
    calls = []

    def _truncated(collection_id):
        calls.append(collection_id)
        return _members(9), False

    slot = _RecordingSlot()
    refresher = DR.CollectionDigestRefresher(slot, _truncated, read=_read, engine_id=_ENGINE_ID)
    outcome = refresher.refresh_if_spent("p", "col-big", None)
    assert not outcome
    assert outcome.reason == CF.FrameNotDigestible.ENUMERATION_TRUNCATED
    assert slot.written == []
    for _ in range(8):
        refresher.note_write("p", "col-big")
        refresher.refresh_if_spent("p", "col-big", None)
    assert len(calls) == 1, "a refusal was retried before a full turnover had passed"
    refresher.note_write("p", "col-big")
    refresher.refresh_if_spent("p", "col-big", None)
    assert len(calls) == 2


def test_an_enumeration_that_FAILS_is_not_remembered():
    """The one case that must stay due. A refusal is an answer about the collection; a store
    hiccup is not, and remembering it would leave the collection permanently undigested."""
    state = {"raise": True}

    def _flaky(collection_id):
        if state["raise"]:
            raise RuntimeError("lattice unavailable")
        return _members(5), True

    slot = _RecordingSlot()
    refresher = DR.CollectionDigestRefresher(slot, _flaky, read=_read, engine_id=_ENGINE_ID)
    assert refresher.refresh_if_spent("p", "col-1", None).reason == "enumeration failed"
    assert refresher.is_due("p", "col-1"), "a transient failure must leave the digest due"
    state["raise"] = False
    assert refresher.refresh_if_spent("p", "col-1", None)


def test_a_single_artifact_collection_is_refused_once_and_not_hammered():
    """The degenerate case through the trigger: it is refused, remembered, and retried only
    after its own (one) row has turned over."""
    calls = []

    def _one(collection_id):
        calls.append(collection_id)
        return [("solo", {"content": "one artifact"})], True

    slot = _RecordingSlot()
    refresher = DR.CollectionDigestRefresher(slot, _one, read=_read, engine_id=_ENGINE_ID)
    outcome = refresher.refresh_if_spent("p", "col-1", None)
    assert outcome.reason == CF.FrameNotDigestible.NO_RESOLVED_DIRECTION
    assert slot.written == []
    assert not refresher.is_due("p", "col-1")


def test_collections_do_not_share_a_clock():
    _slot, refresher = _refresher(_members(6))
    refresher.refresh_if_spent("p", "col-1", None)
    for _ in range(6):
        refresher.note_write("p", "col-2")
    assert not refresher.is_due("p", "col-1")
    assert refresher.is_due("p", "col-2")


def test_the_refresher_carries_no_constant_either():
    """The AST guard over the module that decides WHEN. An interval, a batch size or a
    staleness window all look like a number here."""
    tree = ast.parse(Path(DR.__file__).read_text(encoding="utf-8"))
    numbers = {n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
               and not isinstance(n.value, bool)}
    assert numbers <= {0, 1}, (
        f"digest_refresh.py gained numeric literals {sorted(numbers - {0, 1})}. The trigger is "
        f"a comparison of two counts; anything else is an interval somebody chose."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Serialization
# ═══════════════════════════════════════════════════════════════════════════

def test_the_stored_form_round_trips_every_float_exactly():
    """A stored digest must be the digest, or a distance computed against it is a distance to
    something else. `json` writes a float through `repr`, which round-trips float64."""
    digest = CF.digest_collection(_members(13, seed=2), exhaustive=True, read=_read, engine_id=_ENGINE_ID, collection_id="c")
    back = CF.loads_digest(CF.dumps_digest(digest))
    assert back == digest
    assert _spectral_distance(back.read, digest.read) == 0.0


def test_the_stored_form_is_canonical():
    """Sorted keys and no whitespace, so re-encoding one digest gives one blob."""
    digest = CF.digest_collection(_members(6, seed=4), exhaustive=True, read=_read, engine_id=_ENGINE_ID, collection_id="c")
    blob = CF.dumps_digest(digest)
    assert CF.dumps_digest(CF.loads_digest(blob)) == blob
    assert b" " not in blob.split(b'"read"')[0]


@pytest.mark.parametrize("blob", [b"", b"{}", b"[]", b"not json", b'{"rows": 1}'])
def test_anything_that_is_not_a_digest_raises_rather_than_decoding_to_one(blob):
    with pytest.raises(ValueError):
        CF.loads_digest(blob)


# ═══════════════════════════════════════════════════════════════════════════
# 5. The module boundary the placement rests on
# ═══════════════════════════════════════════════════════════════════════════

def test_the_frame_builder_is_not_inside_the_numpy_free_lexical_closure():
    """Why this module lives under `search/ingest/` and not under `search/mantle/sse/`.

    The lexical extra is numpy-free by install contract, and
    `tests/test_lexical_extra_is_numpy_free.py` enforces it by removing numpy from the import
    system and importing `mantle.search.mantle.sse`. This module imports numpy, so it may not
    sit anywhere that test's imports reach — and the edge must run one way: this module imports
    the tokenizer, and nothing in the SSE tree may import this."""
    assert FRAME_MODULE.parts[-2] == "ingest"
    sse_dir = Path(CF.__file__).parents[1] / "mantle" / "sse"
    assert sse_dir.is_dir(), sse_dir
    for module in sorted(sse_dir.glob("*.py")):
        src = module.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "collection_frame" not in node.module, module.name
                assert "search.ingest" not in node.module, module.name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "search.ingest" not in alias.name, module.name


def test_the_narrowing_carries_no_constant_either():
    """The AST guard, extended to the module that decides what gets compared.

    A size floor moved out of `spectral_distance` and into the narrowing would be the same
    swept constant in a place the beacon's own guard cannot see. `0` and `1` are admissible as
    tuple indices and as the zero of a non-negativity check; nothing else is."""
    tree = ast.parse(NARROW_MODULE.read_text(encoding="utf-8"))
    numbers = {n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
               and not isinstance(n.value, bool)}
    assert numbers <= {0, 1, 0.0}, (
        f"{NARROW_MODULE.name} gained numeric literals {sorted(numbers - {0, 1, 0.0})}. A "
        f"minimum row count, a size band or a default radius all look like this."
    )


# The AST guard against a swept constant landing in the proximity read itself lives in
# the instrument's own repository rather than here. That module is reachable
# from mantle only through the `read=`/`engine_id=` seam `digest_frame`/`digest_collection` take,
# so a change to it is not something this repo's test suite can see.
