"""Blind-token narrowing — which artifacts contain these stems, and how many of them.

Given a query's stems and the ``(principal, collection)`` pairs a requester holds keys for,
this returns a MAPPING from artifact id to :class:`Coverage`: how much of the query that
artifact matched. Its KEYS are the membership answer — the set of artifacts the query's tokens
reach — and its VALUES are counts of the lookups that hit.

There is no score, no IDF, no term frequency, no document length and no corpus statistic
anywhere here. A :class:`Coverage` is a pair of integers produced by counting the lookups this
module was already issuing: one per ``(stem × field)`` for membership, plus one per bigram when
the query is a quoted phrase. Nothing is measured about the corpus, so nothing here is a metric
on the data.

The key set is what lets the answer compose with the light cone. An artifact id set is the same
kind of object :func:`~mantle.search.mantle.lightcone.resolve_authorized_scope` already
produces and that the ranker already applies as membership through its
``authorized_artifacts`` parameter, so narrowing is a MEET against a set rather than a filter
bolted onto a ranking. The counts ride ALONGSIDE that meet and never into it: they decide
order, never membership, so nothing an unauthorized artifact's count could say can survive the
intersection. See :meth:`~.router_accessor.MantleSseSearchAccessor._by_coverage`.

Why this is a callback and not a predicate
------------------------------------------
`resolve_authorized_scope` takes ``artifact_predicate`` as a function of one doc, because a
doc is something its authorized-doc loop is already holding. A token set is not a function of
a doc: it needs the owner SSE key, which needs the ``(principal, collection)`` contexts — and
those contexts are produced BY that loop. So the narrowing cannot be a second predicate
alongside the field filter; it is a ``token_lookup(pairs) -> set[str]`` callback the resolver
invokes once its pairs exist, met in a second phase inside the same function.
:meth:`TokenNarrower.lookup_for` compiles a query into exactly that callback.

What a match means
------------------
- **Unigram.** An artifact matches when any searched field's posting list for a query stem
  carries an entry for it in an authorized collection. Stems are met by UNION: a document
  containing one query term is a document the recall returns. ``require_all_stems=True`` asks
  for the conjunction instead. :attr:`Coverage.stems` counts how many DISTINCT query stems an
  artifact was found under, which is what the union pass already has to know.
- **Phrase.** A quoted query is gated on consecutive stem bigrams. :mod:`indexer` writes a
  posting entry per adjacent stem pair, so adjacency is available as a membership question at
  no additional index cost: an artifact clears the gate only when it carries EVERY bigram of
  the phrase, and an artifact carrying the words apart does not.
  :attr:`Coverage.bigrams` counts the phrase bigrams an artifact was found under.

Why ``bigrams`` cannot re-order anything today, and is still counted
--------------------------------------------------------------------
Bigram lookups happen for a QUOTED PHRASE only, and the phrase gate is all-or-nothing — an
artifact clears it only by carrying every bigram — so every survivor of a phrase query has the
same bigram count, and every survivor of an unquoted query has zero. The number therefore
discriminates nothing, and it is carried because it is the honest second half of "how much of
the query survived", not because it re-orders a page.

Making it discriminate would mean issuing bigram lookups for UNQUOTED multi-term queries, so
that an artifact carrying ``quarterly budget`` adjacent outranked one carrying the two words
apart. That is a real signal and it is not free: it is one store read per ``(bigram × field)``
that nothing on this path issues today — ``n-1`` more waves for an ``n``-stem query. The
counts above are a by-product of work already being done; those would be new work, so they are
not done here.

Cost is one store read per ``(term × field)`` that is not already resolved, issued serially.
Nothing here caches: a narrowing is computed once per recall, and a cache would be a second
opinion about posting-list freshness alongside the store's.

Stdlib + ``cryptography`` only, like the rest of the lexical core — see
`tests/test_sse_lexical_arm_is_separable.py`.
"""

from __future__ import annotations


import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from . import posting as posting_mod
from .blind_tokens import (
    FIELD_CONTENT,
    FIELD_DESCRIPTION,
    FIELD_TAGS,
    FIELD_TITLE,
    blind_token,
)
from .keys import SseKeyProvider
from .posting import PostingStore
from .tokenizer import bigrams as _stem_bigrams, tokenize

# The refusal a key provider raises for a principal that has never been written to, taken from
# the one module that defines it — an `except` matches on the class OBJECT, so a same-named
# local copy would match nothing, on the failure path only. `..custody` holds these two names
# and nothing else, so naming it costs this arm no dependency.
from ..custody import MasterKeyMissing

logger = logging.getLogger(__name__)


# Long-form (posting-entry / field-boost) ↔ short-form (blind-token API).
_LONG_TO_SHORT = {
    "title": FIELD_TITLE,
    "description": FIELD_DESCRIPTION,
    "tags": FIELD_TAGS,
    "content": FIELD_CONTENT,
}

#: Every indexed field. Narrowing searches all of them by default: a field boost is a ranking
#: weight, and membership has no weights to read.
DEFAULT_FIELDS: Tuple[str, ...] = ("title", "description", "tags", "content")


class Coverage(NamedTuple):
    """How much of one query an artifact matched. Two counts, no scale.

    ``stems`` — how many DISTINCT query stems this artifact was found under, across every
    searched field. Bounded above by the number of stems in the query.

    ``bigrams`` — how many of a quoted phrase's consecutive stem bigrams it was found under.
    Zero for an unquoted query, which issues no bigram lookups; constant across the survivors
    of a quoted one, which the gate admits only on the full set. See the module docstring.

    Both are ORDINALS obtained by counting lookups that hit. Neither is normalised, neither is
    weighted, and neither is comparable between two different queries — a ``stems`` of 2 means
    "two of these five" on one query and "two of these two" on another. What they are
    comparable across is the artifacts of ONE narrowing, which is the whole of what ordering
    them needs.

    ``matched`` — WHICH stems it was found under, not how many. A fact about this artifact
    alone, and the caller already holds the artifact, so naming its own matches tells a caller
    nothing it could not read off the document.

    It exists because the counts TIE: a one-stem query ties everywhere by construction (see
    above), and after salience a two-stem query ties between the artifacts matching either one.
    Measured, `what is a vaccine` narrows to 662 artifacts ALL at ``stems=1`` — those carrying
    `vaccin` and those carrying `what` are indistinguishable by the count, and the artifact
    titled `vaccine` sat at position 88 of a 50-wide reach horizon and was never ranked.

    A document frequency would break that tie, and this deliberately does NOT carry one.
    ``len(stem_ids)`` is a count over the raw index, INCLUDING artifacts the light cone will
    refuse, so a rarity computed here changes depending on whether a document the caller may not
    see exists — which is precisely the existence oracle `TestTheCountsAreNotAnExistenceOracle`
    forbids, and which it caught. The frequencies are counted by the reach arm instead, over the
    artifacts that survived the meet, where they can leak nothing.
    """

    stems: int = 0
    bigrams: int = 0
    matched: Tuple[str, ...] = ()


#: What a compiled narrowing returns: per-artifact coverage over the pairs it was handed.
#:
#: Its KEYS are the membership answer and are what
#: :func:`~mantle.search.mantle.lightcone.resolve_authorized_scope` meets — the resolver
#: intersects, so what comes back can only narrow, never widen. Its VALUES never reach the
#: resolver at all; the caller that wants an order reads them beside the meet.
CoverageLookup = Callable[[Sequence[Tuple[str, str]]], Mapping[str, Coverage]]


def phrase_stems(query_text: str) -> tuple[List[str], bool]:
    """Analyze a raw query into ``(stems, is_phrase)``.

    A query wrapped in double quotes is a phrase: the quotes are stripped before tokenizing so
    the stems reflect the phrase's terms rather than the quote characters. This is the ONE
    reading of what a phrase is on the recall path — the gate and the bigram count are both
    taken from it, so there is no second opinion for them to disagree with.
    """
    text = query_text or ""
    is_phrase = len(text) > 2 and text[0] == '"' and text[-1] == '"'
    if is_phrase:
        text = text[1:-1]
    return tokenize(text), is_phrase


class TokenNarrower:
    """Blind-token membership + query coverage over encrypted posting lists.

    Composes a :class:`~.keys.SseKeyProvider` (owner SSE keys) and a
    :class:`~.posting.PostingStore` (opaque blobs) — the same two collaborators the indexer
    writes through, and nothing else. There is no corpus-statistics store, because there is
    no corpus statistic: a coverage count is a count of this narrowing's own lookups.
    """

    def __init__(
        self,
        oracle: SseKeyProvider,
        posting_store: PostingStore,
        *,
        fields: Optional[Iterable[str]] = None,
        require_all_stems: bool = False,
    ) -> None:
        self._oracle = oracle
        self._postings = posting_store
        self._fields: List[str] = [
            f for f in (fields if fields is not None else DEFAULT_FIELDS)
            if f in _LONG_TO_SHORT
        ]
        self._require_all_stems = require_all_stems

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def lookup_for(self, query_text: str, request: Any,
                   *, salient=None) -> Optional[CoverageLookup]:
        """Compile a query into a ``lookup(pairs) -> {artifact_id: Coverage}`` callback.

        ``request`` is the key provider's policy object (``oracle.KeyRequest`` in the
        platform), carried in the closure because the resolver has no opinion about custody
        and should not have to hold one.

        Returns ``None`` when the query analyzes to no stems or no field is searchable. ``None``
        is the resolver's "absent", so a query with nothing to look up narrows nothing rather
        than narrowing to nothing — the same distinction ``artifact_predicate=None`` draws.
        """
        stems, is_phrase = phrase_stems(query_text)
        if not stems or not self._fields:
            return None

        # ── the stems that carry the question, not every stem typed ──────────────────────────
        # `salient` is the corpus's own measure, handed in by the caller because it lives behind
        # the `match` seam and this layer cannot import it. It keeps a term carrying at least the
        # query's own mean IDF, so nothing is chosen and an unmeasurable corpus keeps everything.
        #
        # Needed because the coverage ordering downstream counts stems WITHOUT weighting — that is
        # deliberate and documented in `_by_coverage`, which makes it an ordinal on the query
        # rather than a metric on the data. The consequence is that an unfiltered function word is
        # worth exactly as much as the subject: `what is a glacier` narrowed on `what`/`is`/
        # `glacier`, canon documents titled "What this is (and is not)" matched two stems, and they
        # filled the reach arm's pool before any glacier synset — which matched one — could enter.
        #
        # So the filter belongs HERE, above the count, rather than as a weighting below it. The
        # ordinal stays an ordinal; what changes is which stems it is an ordinal over.
        #
        # A PHRASE is never filtered. Its gate is a membership question over consecutive bigrams,
        # so dropping a stem would break an adjacency the query actually asked for — and a phrase
        # is an explicit statement that these words belong together, which is exactly the claim
        # salience is not entitled to overrule.
        if salient is not None and not is_phrase:
            try:
                kept = list(salient(list(stems)))
            except Exception:  # noqa: BLE001 — a measure that raises measures nothing
                kept = []
            if kept:
                stems = kept

        def _lookup(pairs: Sequence[Tuple[str, str]]) -> Mapping[str, Coverage]:
            return self.ids_for_stems(stems, pairs, request, is_phrase=is_phrase)

        return _lookup

    def ids_for_stems(
        self,
        stems: Sequence[str],
        pairs: Iterable[Tuple[str, str]],
        request: Any,
        *,
        is_phrase: bool = False,
    ) -> Dict[str, Coverage]:
        """Coverage per artifact whose postings carry ``stems``, within ``pairs``.

        ``pairs`` are ``(principal_id, collection_id)`` — the key-custody granularity. They
        decide which encrypted index is opened at all, and an entry naming a collection outside
        them is dropped even when its posting list opened, because one posting list spans every
        collection the owner indexed that term in.

        The KEYS are the membership answer. They are not an authorization answer and must not
        be read as one: they say which artifacts carry the terms, over exactly the indexes the
        caller could already open, and the caller meets them against its own authorized set.

        The VALUES are counts of the lookups that hit, per :class:`Coverage`. An artifact
        reachable under more than one owner — which needs the same id indexed by two
        principals — takes the LARGEST count either owner's index gives it, because coverage is
        a statement about the query and the best evidence for a stem is that some index this
        caller may read carries it.
        """
        stems = [s for s in stems if s]
        if not stems:
            return {}

        scope: dict[str, set[str]] = {}
        for principal_id, collection_id in pairs:
            if not principal_id or not collection_id:
                continue
            scope.setdefault(str(principal_id), set()).add(str(collection_id))
        if not scope:
            return {}

        found: Dict[str, Coverage] = {}
        for principal_id, collection_ids in scope.items():
            for artifact_id, cover in self._owner_ids(
                principal_id, collection_ids, stems, is_phrase=is_phrase,
                request=request,
            ).items():
                held = found.get(artifact_id)
                found[artifact_id] = cover if held is None else Coverage(
                    max(held.stems, cover.stems), max(held.bigrams, cover.bigrams),
                    tuple(dict.fromkeys(held.matched + cover.matched)),
                )
        return found

    # ------------------------------------------------------------------
    # Per-owner membership
    # ------------------------------------------------------------------

    def _owner_ids(
        self,
        principal_id: str,
        authorized_collections: set[str],
        stems: Sequence[str],
        *,
        is_phrase: bool,
        request: Any,
    ) -> Dict[str, Coverage]:
        try:
            owner_sse_key = self._oracle.derive_sse_key(principal_id, request)
        except MasterKeyMissing:
            # Never written to, so no key and no index. Nothing to narrow to — the same
            # answer an empty posting list gives, reached one step earlier.
            logger.debug(
                "SSE narrowing: principal %s has no master key (never indexed); skipping",
                principal_id,
            )
            return {}

        # No owner-index accelerator. It earns its risk only when every probe below is an `open()`
        # on a file store — 4,520 of them for a ten-term query over 194 owners, mostly misses — so
        # one blob holding the owner's whole token map pays for itself. `SqlitePostingStore` makes a
        # probe an indexed lookup on a primary key, so there is nothing left to collapse, and the
        # blob's own failure modes go with it: a partial index read as complete made a whole prior
        # corpus unfindable, and its read-modify-write lost concurrent writers.

        # The bigram gate. `indexer` writes a posting entry per adjacent stem pair, so a
        # phrase's adjacency is a membership question over those entries: an artifact clears
        # only if it carries EVERY consecutive pair. A single missing bigram rules the phrase
        # out of this owner's index entirely, which is why the gate runs before the unigram
        # pass rather than beside it.
        #
        # `bigram_hits` counts the same lookups the gate is already making. The gate's
        # all-or-nothing reading means every id still in `gate` at the end has been counted
        # once per bigram, so the count is uniform across survivors — see the module docstring.
        gate: Optional[Set[str]] = None
        bigram_hits: Dict[str, int] = {}
        if is_phrase and len(stems) >= 2:
            for bigram in _stem_bigrams(list(stems)):
                bigram_ids = self._ids_for_term(
                    principal_id, owner_sse_key, bigram, authorized_collections,
                )
                if not bigram_ids:
                    return {}
                for artifact_id in bigram_ids:
                    bigram_hits[artifact_id] = bigram_hits.get(artifact_id, 0) + 1
                gate = bigram_ids if gate is None else (gate & bigram_ids)
                if not gate:
                    return {}

        # Unigrams. Union across stems is the candidate set the recall returns — a document
        # carrying one query term is a document the recall returns — and `require_all_stems`
        # asks for the conjunction instead.
        #
        # `stem_hits` is the coverage count, and it is the union pass counting instead of
        # discarding: the loop already resolves one id set per stem, so "how many stems reached
        # this artifact" is a `+= 1` over sets it is holding anyway. No second lookup, no
        # corpus statistic, no weight.
        owner_ids: Optional[Set[str]] = None
        stem_hits: Dict[str, int] = {}
        # Which stems each artifact was found under. The COUNT is `stem_hits`; this is the
        # identity, and the reach arm needs it to tell two artifacts apart that the count cannot.
        matched: Dict[str, list] = {}
        for stem in stems:
            stem_ids = self._ids_for_term(
                principal_id, owner_sse_key, stem, authorized_collections,
            )
            for artifact_id in stem_ids:
                stem_hits[artifact_id] = stem_hits.get(artifact_id, 0) + 1
                matched.setdefault(artifact_id, []).append(stem)
            if self._require_all_stems:
                if not stem_ids:
                    return {}
                owner_ids = stem_ids if owner_ids is None else (owner_ids & stem_ids)
                if not owner_ids:
                    return {}
            else:
                owner_ids = stem_ids if owner_ids is None else (owner_ids | stem_ids)

        if owner_ids is None:
            return {}
        surviving = owner_ids if gate is None else (owner_ids & gate)
        return {
            artifact_id: Coverage(
                stem_hits.get(artifact_id, 0),
                bigram_hits.get(artifact_id, 0),
                tuple(matched.get(artifact_id, ())),
            )
            for artifact_id in surviving
        }

    def _ids_for_term(
        self,
        principal_id: str,
        owner_sse_key: bytes,
        term: str,
        authorized_collections: set[str],
    ) -> Set[str]:
        """Artifact ids carrying ``term`` in any searched field, within the collection set."""
        ids: Set[str] = set()
        for field_long in self._fields:
            bt = blind_token(owner_sse_key, _LONG_TO_SHORT[field_long], term)
            for entry in self._entries(principal_id, owner_sse_key, bt):
                if str(entry.get("collection_id", "")) not in authorized_collections:
                    continue
                artifact_id = entry.get("artifact_id")
                if artifact_id:
                    ids.add(str(artifact_id))
        return ids

    def _entries(
        self, principal_id: str, owner_sse_key: bytes, bt: str,
    ) -> List[dict]:
        """Decrypted posting entries for one blind token. ``[]`` for a miss and for a blob
        that fails GCM authentication — one unreadable posting list must not fail a recall,
        and a narrowing that raised would be a way to tell an unreadable token from an absent
        one.

        One probe, always — there is no accelerator to consult first. On `SqlitePostingStore` this
        is an indexed range scan over `(principal_id, blind_token)`, which is what made the
        owner-index blob unnecessary rather than merely repaired.

        Two layouts, tried in that order. A slot written since the entry layout holds one sealed
        blob per `(artifact, collection)`; one written before holds a single blob sealing them all.
        Splitting a legacy blob needs the owner's key, so there is no migration pass for it: the
        entries win where they exist, and `SseIndexer._absorb_legacy_slot` converts a slot on the
        next write that touches it. Entries-first rather than legacy-first because a slot mid-
        conversion could briefly hold both, and the entries are the newer truth.
        """
        key = posting_mod.derive_posting_key(owner_sse_key, bt)

        out: List[dict] = []
        for artifact_id, collection_id, blob in self._postings.get_entries(principal_id, bt):
            try:
                # The AAD is built from the row key, so the row key has to agree with what the
                # ciphertext says it is. The key is plaintext and mutable by anyone who can write
                # to the store; the sealed entry names the slot it was minted for. An entry moved
                # to another artifact, collection or token fails here rather than being served.
                out.append(posting_mod.unpack_entry(
                    blob, key,
                    aad=posting_mod.entry_aad(principal_id, bt, artifact_id, collection_id),
                ))
            except posting_mod.PostingError as exc:
                logger.warning(
                    "SSE narrowing: dropping unreadable entry owner=%s token=%s artifact=%s "
                    "reason=%s", principal_id, bt[:8], artifact_id, exc,
                )
        if out:
            return out

        blob = self._postings.get_posting(principal_id, bt)
        if blob is None:
            return []
        try:
            # The slot AAD the legacy writer used. `allow_unbound` stays at its default so posting
            # lists written before slot binding still open through `decrypt_blob`'s dual-read.
            return posting_mod.unpack_posting(
                blob, key, aad=posting_mod.posting_aad(principal_id, bt),
            )
        except posting_mod.PostingError as exc:
            logger.warning(
                "SSE narrowing: dropping unreadable posting list owner=%s token=%s reason=%s",
                principal_id, bt[:8], exc,
            )
            return []


__all__ = [
    "Coverage",
    "CoverageLookup",
    "DEFAULT_FIELDS",
    "TokenNarrower",
    "phrase_stems",
]
