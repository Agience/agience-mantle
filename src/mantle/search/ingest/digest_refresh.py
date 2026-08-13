# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
# ---------------------------------------------------------------------------

"""When a collection's proximity digest is retaken, and who counts the reasons.

:func:`~mantle.search.ingest.collection_frame.digest_is_spent` says *whether* a digest has to
be retaken; this says *when the question is asked* and *what the two numbers in it are*. The
rule it applies is derived in that function's docstring and is repeated nowhere: a digest is a
statement about ``rows`` rows, one invalidating event makes at most one of them wrong, so the
digest is spent after ``rows`` events — which is also the unique amortisation that returns the
per-write cost of digesting to the order of the write itself.

What counts as an event
-----------------------
Every membership change and every member edit. An add or a remove changes the row set; an
edit changes a row's counts and can add or drop columns. There is nothing on the write path
that leaves the frame alone, so :meth:`CollectionDigestRefresher.note_write` is called for all
of them and there is no classification to get wrong.

Why the counter is in process, and what would replace it
--------------------------------------------------------
The count has to be readable in O(1) on the write path or the trigger costs more than the
digest it defers. The store's own idiom for that is the ``counter`` table — ``db/schema.py``
says outright that counters are "the only sanctioned way to answer how many", and
``c_collection`` already maintains ``col:<id>`` in the same transaction as the write that
changes it. A durable ``proximity:<id>`` counter beside it is where this belongs, and it is
deliberately NOT added here: that is a schema change with a migration, a counter-drift audit
and a merkle leaf behind it, and none of that is needed to state or to test the rule.

So the clock is a process-local dict, in the same spirit as
``sse.posting.InMemoryPostingStore`` — a real implementation of a seam, honest about being
non-durable. Losing it on restart resets every collection's count to zero, which makes the
next write re-digest: the failure mode is one extra digest per collection, never a missed one,
because a lost count reads as ``0 >= rows`` only when ``rows`` is 0 and otherwise simply
delays nothing. That direction is the safe one, and it is the reason a volatile clock is
acceptable where a volatile posting list would not be.

A refusal is an answer, and it is remembered
--------------------------------------------
:class:`~mantle.search.ingest.collection_frame.FrameNotDigestible` is not a failure to retry.
An un-enumerable collection is still un-enumerable on the next write, so a refusal resets the
clock exactly as a digest does, against the row count the attempt observed. For the oversized
collections that motivated the question — 12 collections holding 96.6% of the corpus, up to
1.84M rows — that makes the retry interval as large as the collection, so the store does not
spend its life rediscovering that it cannot enumerate them.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

from mantle.search.ingest.collection_frame import (
    CollectionDigest,
    FrameNotDigestible,
    digest_collection,
    digest_is_spent,
)

logger = logging.getLogger(__name__)

#: ``collection_id -> (members, exhaustive)``, where ``members`` is ``(artifact_id, fields)``
#: pairs in the shape ``pipeline_unified._extract_artifact_fields`` produces.
#:
#: ``exhaustive`` is part of the return value rather than assumed, because the store's member
#: read tops out at ``edges_of``'s ``limit=1000`` with no cursor (``db/edge.py:418``) and a
#: digest of a prefix would be a fabricated measurement of the whole. It is the same
#: ``(rows, exhaustive)`` shape ``db.vertex.list_by_content_type`` already returns, for the
#: same reason.
MembersProvider = Callable[
    [str], Tuple[Iterable[Tuple[str, Mapping[str, str]]], bool]
]

__all__ = ["CollectionDigestRefresher", "DigestOutcome"]


class DigestOutcome:
    """What one refresh did. Truthy iff a digest was written."""

    __slots__ = ("digest", "reason", "rows")

    def __init__(self, *, digest: Optional[CollectionDigest] = None,
                 reason: str = "", rows: int = 0) -> None:
        self.digest = digest
        self.reason = reason
        self.rows = rows

    def __bool__(self) -> bool:
        return self.digest is not None

    def __repr__(self) -> str:
        if self.digest is not None:
            return f"<DigestOutcome digested rows={self.rows}>"
        return f"<DigestOutcome refused rows={self.rows} ({self.reason})>"


#: The outcome of a refresh that was not due. Distinct from a refusal: nothing was attempted,
#: nothing was learned, and the existing digest stands.
NOT_DUE = "not due"


class CollectionDigestRefresher:
    """Counts invalidating events per collection and retakes a digest when one is spent.

    Composes a digest slot (custody + storage), a member provider (the lattice), and a
    proximity instrument (``read`` + ``engine_id``, `digest_collection`'s own). All three are
    injected for the same reason every other component in this arm injects them: the rule this
    class implements has no opinion about any of them, and holding one would be a second
    opinion about custody, about what a collection contains, or — mantle never imports
    entroptics — about which instrument takes the read. Pass
    ``read=entroptics.proximity.mp_deviation,
    engine_id=entroptics.proximity.ENGINE_ID_PROXIMITY`` (or your own).
    """

    def __init__(self, slot, members_provider: MembersProvider, *, read, engine_id: str) -> None:
        self._slot = slot
        self._members = members_provider
        self._read = read
        self._engine_id = engine_id
        self._lock = threading.RLock()
        #: events observed since the last attempt, per (principal, collection)
        self._pending: Dict[Tuple[str, str], int] = {}
        #: rows the last attempt covered, per (principal, collection). A collection that has
        #: never been attempted is absent, which reads as 0 rows and is therefore spent at
        #: once — the base case falls out of the rule rather than being special-cased.
        self._covered: Dict[Tuple[str, str], int] = {}

    # ------------------------------------------------------------------

    def note_write(self, principal_id: str, collection_id: str) -> int:
        """Record one invalidating event. Returns the running count.

        Called for every membership change and every member edit — see the module docstring
        for why there is nothing to classify. O(1) and lock-held only for the increment, so it
        can sit on the write path without being the reason a write is slow.
        """
        slot = (str(principal_id), str(collection_id))
        with self._lock:
            n = self._pending.get(slot, 0) + 1
            self._pending[slot] = n
            return n

    def is_due(self, principal_id: str, collection_id: str) -> bool:
        """Is this collection's digest spent? The trigger, read without acting on it."""
        slot = (str(principal_id), str(collection_id))
        with self._lock:
            return digest_is_spent(
                self._covered.get(slot, 0), self._pending.get(slot, 0),
            )

    def refresh_if_spent(
        self, principal_id: str, collection_id: str, request,
    ) -> DigestOutcome:
        """Retake the digest if the trigger has fired; otherwise do nothing at all.

        The enumeration and the read happen only past the trigger, which is the whole point:
        the expensive half is what the rule is deferring, so asking whether it is due must not
        cost anything the write path would notice.

        A refusal resets the clock exactly as a success does. It is an answer about this
        collection, not a transient error, so retrying it on the next write would spend a full
        enumeration to learn what the last one already established.
        """
        if not self.is_due(principal_id, collection_id):
            return DigestOutcome(reason=NOT_DUE)

        slot = (str(principal_id), str(collection_id))
        try:
            members, exhaustive = self._members(slot[1])
            members = list(members)
        except Exception:  # noqa: BLE001 — a lattice read can raise broadly
            # NOT an answer about the collection: nothing was measured, so the clock is left
            # running and the next write asks again. This is the one case that must not be
            # remembered, because remembering it would turn a store hiccup into a permanently
            # undigested collection.
            logger.warning(
                "proximity: could not enumerate %s; leaving its digest due",
                slot[1], exc_info=True,
            )
            return DigestOutcome(reason="enumeration failed")

        try:
            digest = digest_collection(
                members, exhaustive=exhaustive, collection_id=slot[1],
                read=self._read, engine_id=self._engine_id,
            )
        except FrameNotDigestible as exc:
            self._settle(slot, len(members))
            logger.info(
                "proximity: %s has no digest (%s); next attempt is one turnover away",
                slot[1], exc.reason,
            )
            return DigestOutcome(reason=exc.reason, rows=len(members))

        self._slot.put(slot[0], digest, request)
        self._settle(slot, digest.rows)
        return DigestOutcome(digest=digest, rows=digest.rows)

    # ------------------------------------------------------------------

    def _settle(self, slot: Tuple[str, str], rows: int) -> None:
        """One attempt is over: the clock restarts against the rows that attempt covered."""
        with self._lock:
            self._pending[slot] = 0
            self._covered[slot] = int(rows)
