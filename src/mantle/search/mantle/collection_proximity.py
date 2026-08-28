# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
# ---------------------------------------------------------------------------

"""Proximity over collections — narrow on what is true, then rank on what is left.

A collection's proximity digest (:mod:`mantle.search.ingest.collection_frame`) is a record a
store can hold and compare. This module is the read side: it opens the digests of the
collections a requester already holds keys for, **narrows them by properties the store already
knows**, probes what survives, and hands back artifact ids that
:func:`~mantle.search.mantle.lightcone.resolve_authorized_scope` meets into the light cone with
the same ``&=`` every other narrowing goes through.

This is not a recall path
-------------------------
``/artifacts/recall`` ranks by query coverage — how much of a query's stems each artifact
carried — which is a statement about a query and a document. This is a statement about two
collections. Nothing here is wired into recall and nothing here should be: they answer
different questions and share only the light cone.

Narrowing carries what distance should not
------------------------------------------
The design decision this module is built to is a single sentence: **properties that are true
of a frame become edges or tags in the lattice, and narrowing carries what distance should
not.** It is the same architecture recall already uses — narrow, then rank — and it is why
the injected instrument's ``spectral_distance`` (``<probe>.spectral_distance``, for
the reference instrument) is used exactly as shipped, with no size floor, no length
normalisation and no weighting added to compensate for anything.

The thing it would otherwise be tempting to compensate for is measured and real: a 4-artifact
collection beats a 1,621-artifact one at 2.05x chance, because 74.8% of a read's L2 energy
sits in its first four modes, so a short record keeps the discriminating signal on the common
prefix and loses only the tail. Every available repair — a minimum row count, dividing by the
prefix length, weighting a mode by its depth — is a number somebody picks, and a picked number
is the thing this construction exists not to have. So the comparison is not fixed. It is **not
made**: ``rows`` is a property the store carries, so a caller narrows on it and the 4-row
record and the 1,621-row record never meet in the same gallery.

That is a real obligation on the caller, stated plainly rather than defaulted away: with
``properties=None`` no size narrowing happens and the attractor is live. ``None`` means "no
narrowing" here for the same reason ``artifact_predicate=None`` does in the resolver — an
absent filter must leave the answer identical to one that never went near this module — and
:func:`same_rows` is the one size narrowing this module can offer without inventing a band.

How it composes, and why that is the whole security argument
------------------------------------------------------------
:meth:`CollectionProximityNarrower.lookup_for` compiles a query digest into a
:data:`~mantle.search.mantle.lightcone.TokenLookup` — the *same* callable shape the blind-token
narrowing compiles to, ``(pairs) -> set[artifact_id]``. It is not a new parameter and not a new
meeting place: a proximity query calls ``resolve_authorized_scope`` with this callback in the
``token_lookup`` slot, and the resolver's one line

    ids &= _token_narrowing(token_lookup, contexts)

does the rest. Everything follows from that:

* **It can only narrow.** ``ids`` is already a subset of what the light cone authorized, so
  whatever this returns — members of somebody else's collection, ids that do not exist, the
  whole store — meets a set that is already narrow. There is no vocabulary here for adding an
  id.
* **A digest naming a collection the requester cannot read is indistinguishable from one
  matching nothing.** Both leave by the ``&=``, and both leave as the empty set. The
  narrowing is therefore not an existence oracle for collections outside the light cone. This
  is the exact counterpart of
  ``tests/test_blind_token_narrowing.py::test_that_answer_is_identical_to_a_token_matching_nothing``
  and of the field filter's, and it is held by
  ``tests/test_collection_proximity_narrowing.py``.
* **Failure is closed.** ``_token_narrowing`` catches and narrows to nothing, so an
  unreadable digest store costs recall and never authority.

There is a second, independent cut in front of that one: the callback is handed the resolver's
``contexts``, so it only ever opens digests for ``(cell_principal, collection)`` pairs the
requester already holds keys for. A collection outside the light cone contributes no pair, so
its digest is never even read. The two cuts are not redundant — the meet is what makes the
property hold even if the pairs were ever wrong, which is the reading
``resolve_authorized_scope`` already gives its own ``&=``.

Where a digest lives
--------------------
In the SSE posting store, under a blind token no term can produce. A digest is a per-``(owner,
collection)`` encrypted blob and a posting list is a per-``(owner, token)`` encrypted blob;
they are the same object, so they get the same store, the same AEAD, the same slot binding and
the same key derivation rather than a second crypto opinion. :func:`digest_slot_token` blinds
``"\\x00proximity-digest:<collection_id>"`` — the tokenizer emits ``\\w+`` runs only, so no
stem and no bigram can collide with a term carrying a NUL byte and a colon.

Sealing it under the owner SSE key is what makes the digest's shape safe to store. The read's
length is ``min(N, D)``, which on real shapes is ``N`` (measured, 38 of 38), so the record's
length is the collection's member count. That is not a new exposure, and the code says why:
``db/schema.c_collection`` maintains ``col:<id>`` in the ``counter`` table and
``db.vertex.count_in_collection`` reads it in O(1), while
``db.lattice_api.list_collection_artifacts`` returns the members outright — and both need only
a store handle. Opening a digest needs the collection's SSE key, which
``oracle.LightConeGrantVerifier`` issues only inside the light cone. So the count is reachable
strictly more easily than the digest is: anyone who can read a digest could already count the
collection, and anyone who cannot read it learns nothing from a length they cannot see.
"""
from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from mantle.search.ingest.collection_frame import (
    CollectionDigest,
    dumps_digest,
    loads_digest,
)
from mantle.search.mantle.custody import MasterKeyMissing
from mantle.search.mantle.sse import posting as posting_mod
from mantle.search.mantle.sse.blind_tokens import FIELD_CONTENT, blind_token
from mantle.search.mantle.sse.keys import SseKeyProvider
from mantle.search.mantle.sse.posting import PostingStore

logger = logging.getLogger(__name__)

#: The term a digest's slot is blinded from. It opens with a NUL and carries colons, and the
#: tokenizer emits `\w+` runs (`sse/tokenizer._WORD_RE`) joined by single spaces for bigrams,
#: so no indexed term can ever hash to this slot. Domain separation by construction rather than
#: by a namespace convention somebody has to remember.
_DIGEST_TERM_PREFIX = "\x00proximity-digest:"

__all__ = [
    "CollectionProperties",
    "CollectionProximityNarrower",
    "DigestSlot",
    "PropertyNarrowing",
    "ProximityQueryError",
    "digest_slot_token",
    "same_rows",
]


class ProximityQueryError(ValueError):
    """A proximity query that cannot be answered as asked. Never swallowed into an empty set:
    an empty answer means "nothing was near", and a malformed query has not asked that."""


# ═══════════════════════════════════════════════════════════════════════════
# Storage — a digest is a posting-shaped blob, so it is stored as one
# ═══════════════════════════════════════════════════════════════════════════

def digest_slot_token(owner_sse_key: bytes, collection_id: str) -> str:
    """The blind token one collection's digest is filed under, for this owner.

    ``blind_token(key, FIELD_CONTENT, "\\x00proximity-digest:<collection_id>")``. The field is
    the content field because a digest is taken over the collection's terms; the point of the
    prefix is that the term is unreachable from any text, so the choice of field carries no
    weight beyond keeping the call in the one shape ``blind_token`` accepts.
    """
    if not collection_id:
        raise ValueError("collection_id is required")
    return blind_token(
        owner_sse_key, FIELD_CONTENT, _DIGEST_TERM_PREFIX + str(collection_id),
    )


class DigestSlot:
    """Read and write collection digests through a key provider and a posting store.

    The same two collaborators :class:`~mantle.search.mantle.sse.indexer.SseIndexer` and
    :class:`~mantle.search.mantle.sse.narrowing.TokenNarrower` compose, for the same reason:
    custody decides what opens, and the store holds opaque blobs. Nothing else is needed, and
    anything else would be a second opinion about one of those two.
    """

    def __init__(self, oracle: SseKeyProvider, posting_store: PostingStore) -> None:
        self._oracle = oracle
        self._postings = posting_store

    def put(
        self, principal_id: str, digest: CollectionDigest, request: Any,
    ) -> None:
        """Seal and store one collection's digest. Overwrites in place."""
        key = self._oracle.derive_sse_key(principal_id, request)
        token = digest_slot_token(key, digest.collection_id)
        self._postings.put_posting(
            principal_id, token,
            posting_mod.encrypt_blob(
                dumps_digest(digest),
                posting_mod.derive_posting_key(key, token),
                aad=posting_mod.posting_aad(principal_id, token),
            ),
        )

    def get(
        self, principal_id: str, collection_id: str, request: Any,
    ) -> Optional[CollectionDigest]:
        """One collection's digest, or ``None``.

        ``None`` for every reason a digest is not available — never indexed, no master key,
        an unreadable blob, a blob that is not a digest. They are one answer here on purpose:
        a caller able to tell "no digest" from "a digest that would not open" would be reading
        a signal about content it has not been shown, and the recall cost of collapsing them is
        that a corrupt digest looks like an absent one, which is the same trade
        ``TokenNarrower._entries`` takes for an unreadable posting list.
        """
        try:
            key = self._oracle.derive_sse_key(principal_id, request)
        except MasterKeyMissing:
            return None
        token = digest_slot_token(key, collection_id)
        blob = self._postings.get_posting(principal_id, token)
        if blob is None:
            return None
        try:
            return loads_digest(
                posting_mod.decrypt_blob(
                    blob, posting_mod.derive_posting_key(key, token),
                    aad=posting_mod.posting_aad(principal_id, token),
                )
            )
        except (posting_mod.PostingError, ValueError) as exc:
            logger.warning(
                "proximity: dropping unreadable digest owner=%s collection=%s reason=%s",
                principal_id, collection_id, exc,
            )
            return None


# ═══════════════════════════════════════════════════════════════════════════
# The properties — what is true, and already known
# ═══════════════════════════════════════════════════════════════════════════

class CollectionProperties(NamedTuple):
    """What the store already knows about a collection, as the narrowing sees it.

    Every field is a fact recorded before any query ran. ``rows`` and ``columns`` come off the
    digest, which is the record of the frame that was actually read — not a re-derivation that
    could disagree with it — and ``rows`` is also independently readable in O(1) through
    ``db.vertex.count_in_collection`` off the ``col:<id>`` counter.

    There is deliberately nothing derived here: a property that had to be computed at query
    time would be a score wearing a property's name, and scores rank rather than narrow.
    """

    principal_id: str
    collection_id: str
    rows: int
    columns: int


#: A narrowing over collections: a predicate on properties, exactly the shape
#: ``artifact_predicate`` has over docs. It runs before the probe is built, so what it removes
#: is never ranked, never compared, and never in a gallery — which is what "narrow, then rank"
#: means operationally rather than as an ordering of two steps.
PropertyNarrowing = Callable[[CollectionProperties], bool]


def same_rows(rows: int) -> PropertyNarrowing:
    """Narrow to collections with exactly this many rows.

    The one size narrowing this module can state without choosing anything. Equality is the
    identity relation, not a band: any tolerance — "within 10%", "the same order of magnitude",
    "at least 30 rows" — needs a number, and that number would be the swept constant the whole
    construction refuses, moved from the distance into the filter where it is harder to see.

    It is also the narrowing that makes the comparison exact rather than merely fair. Two
    records of equal length are compared on every mode either of them has, so the injected
    instrument's ``common_prefix`` truncates nothing and the short-record attractor cannot
    arise at all — it is an artefact of unequal lengths, and there are none.

    Callers wanting a band supply their own predicate. A number a caller states in a request is
    that caller's, in the same way ``type:application/pdf`` is; what must not exist is a number
    this module picked on everyone's behalf.
    """
    want = int(rows)
    return lambda props: props.rows == want


# ═══════════════════════════════════════════════════════════════════════════
# The narrowing
# ═══════════════════════════════════════════════════════════════════════════

#: ``collection_id -> the artifact ids it holds``. Injected rather than taken as a lattice
#: handle so this module keeps the same collaborator discipline as the rest of the arm.
#: ``db.lattice_api.list_collection_artifacts`` is what supplies it in the platform; note that
#: its ``edges_of`` cap (``db/edge.py:418``) can only make this return FEWER ids, which can only
#: narrow further — a truncation here costs recall and can never admit anything.
MembersOf = Callable[[str], Iterable[str]]


class CollectionProximityNarrower:
    """Compile "collections near this one" into a narrowing the light cone can meet.

    Deterministic: the digest read is deterministic, the injected probe (below) is specified to
    use ``numpy.searchsorted`` and a stable argsort, and the surviving ids are returned as a set
    the resolver intersects — so two runs over one store give one answer, to the bit.
    """

    def __init__(
        self,
        oracle: SseKeyProvider,
        posting_store: PostingStore,
        members_of: MembersOf,
        *,
        probe_factory,
    ) -> None:
        self._slot = DigestSlot(oracle, posting_store)
        self._members_of = members_of
        #: `spectra -> object with .within(query, radius)` / `.nearest(query, k)`, exact against
        #: a full scan — `<probe>.SpectrumProbe`'s contract. Injected because
        #: mantle imports no spectral library: the same reason `digest_collection`'s read is
        #: injected into `CollectionDigestRefresher` rather than imported here.
        self._probe_factory = probe_factory

    def lookup_for(
        self,
        query: CollectionDigest,
        request: Any,
        *,
        radius: Optional[float] = None,
        nearest: Optional[int] = None,
        properties: Optional[PropertyNarrowing] = None,
    ):
        """Compile a query digest into a ``lookup(pairs) -> set[artifact_id]`` callback.

        Exactly one of ``radius`` and ``nearest`` must be given. Neither has a default,
        because neither is this module's to choose: ``proximity.SpectrumProbe`` is explicit
        that the probe radius "is not chosen to trade recall against work — it *is* the answer
        radius the caller asked for", and ``k`` is the same kind of statement. A default would
        put a number here that nobody derived.

        ``properties`` narrows the gallery before the probe is built — see
        :data:`PropertyNarrowing` and :func:`same_rows`. ``None`` applies no narrowing and is
        byte-identical to a probe over every digest the pairs reach.
        """
        if (radius is None) == (nearest is None):
            raise ProximityQueryError(
                "a proximity query states exactly one of `radius` (every collection within a "
                "distance the caller names) or `nearest` (the k closest). Supplying both asks "
                "two questions and supplying neither asks none; there is no default, because "
                "the answer radius belongs to the caller and not to this module."
            )
        if radius is not None and not (radius >= 0.0):
            raise ProximityQueryError(f"radius must be non-negative; got {radius!r}")
        if nearest is not None and int(nearest) <= 0:
            raise ProximityQueryError(f"nearest must be positive; got {nearest!r}")

        def _lookup(pairs: Sequence[Tuple[str, str]]) -> Set[str]:
            return self.ids_near(
                query, pairs, request,
                radius=radius, nearest=nearest, properties=properties,
            )

        return _lookup

    # ------------------------------------------------------------------

    def candidates(
        self,
        query: CollectionDigest,
        pairs: Iterable[Tuple[str, str]],
        request: Any,
        *,
        properties: Optional[PropertyNarrowing] = None,
    ) -> List[Tuple[CollectionProperties, CollectionDigest]]:
        """The gallery: the digests that survive comparability and the property narrowing.

        Comparability is not a filter the caller chose — it is the instrument's own condition.
        ``proximity`` says it outright: records digested against different references "are not
        comparable with" each other, so a digest whose ``engine_id`` or ``frame_id`` differs
        from the query's is dropped rather than compared. Re-digest, do not mix.
        """
        out: List[Tuple[CollectionProperties, CollectionDigest]] = []
        seen: Set[Tuple[str, str]] = set()
        for principal_id, collection_id in pairs:
            if not principal_id or not collection_id:
                continue
            slot = (str(principal_id), str(collection_id))
            if slot in seen:
                continue
            seen.add(slot)
            digest = self._slot.get(slot[0], slot[1], request)
            if digest is None:
                continue
            if (digest.engine_id != query.engine_id
                    or digest.frame_id != query.frame_id):
                logger.debug(
                    "proximity: %s was digested by a different instrument (%s/%s); not "
                    "comparable with the query (%s/%s)",
                    slot[1], digest.frame_id, digest.engine_id,
                    query.frame_id, query.engine_id,
                )
                continue
            props = CollectionProperties(
                principal_id=slot[0], collection_id=slot[1],
                rows=digest.rows, columns=digest.columns,
            )
            if properties is not None and not properties(props):
                continue
            out.append((props, digest))
        out.sort(key=lambda pair: (pair[0].principal_id, pair[0].collection_id))
        return out

    def ids_near(
        self,
        query: CollectionDigest,
        pairs: Iterable[Tuple[str, str]],
        request: Any,
        *,
        radius: Optional[float] = None,
        nearest: Optional[int] = None,
        properties: Optional[PropertyNarrowing] = None,
    ) -> Set[str]:
        """The artifact ids of the collections the probe returns.

        Ids, not collection ids, because ids are what the light cone meets. The mapping from a
        surviving collection to its members is a widening in isolation — it names every member,
        authorized or not — and it is safe for exactly the reason the token narrowing's is: the
        resolver intersects, so a member the light cone did not authorize cannot survive. What
        this returns is a candidate set, never an authorization answer.
        """
        gallery = self.candidates(query, pairs, request, properties=properties)
        if not gallery:
            return set()

        probe = self._probe_factory([digest.read for _props, digest in gallery])
        if radius is not None:
            hits = probe.within(query.read, float(radius))
        else:
            hits = probe.nearest(query.read, int(nearest))

        ids: Set[str] = set()
        for hit in hits:
            props, _digest = gallery[hit.index]
            try:
                members = self._members_of(props.collection_id)
            except Exception:  # noqa: BLE001 — a lattice read can raise broadly
                logger.warning(
                    "proximity: could not enumerate %s; it contributes no candidates",
                    props.collection_id, exc_info=True,
                )
                continue
            for artifact_id in (members or ()):
                if artifact_id:
                    ids.add(str(artifact_id))
        return ids
