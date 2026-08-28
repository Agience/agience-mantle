"""SseIndexer — commit-path posting list + manifest updater.

Mirrors :class:`mantle.search.mantle.indexer.MantleIndexer` for the SSE
lexical index. Composes:

- :class:`SseKeyProvider` (``.keys``) — derives the owner SSE key. The protocol, not the
  platform's ``OracleService``, which merely satisfies it.
- :mod:`tokenizer` — analysis pipeline (lowercase → possessive → stop
  → Porter stem)
- :mod:`blind_tokens` — HMAC-based exact + prefix token generation
- :mod:`posting` — encrypted posting list + manifest CRUD
- :class:`PostingStore` — abstract storage

The indexer never touches S3 directly. Production wires up an S3-backed or file-backed
``PostingStore``; tests use the in-memory variant.

What a posting entry carries
----------------------------
``{"artifact_id", "collection_id", "field"}`` and nothing else. ``tf``, ``dl`` and
``positions`` are BM25's inputs — term frequency, document length, and token offsets — and this
index runs no BM25: the recall path answers membership and counts how much of a query each artifact
matched, neither of which reads any of the three. ``positions`` was never read by anything in
``src/`` even while the scorer existed. The manifest's ``field_dls`` went the same way; it
existed to roll a document's length contribution back out of the corpus statistics on
re-index, and there are no corpus statistics.

This is a wire-format subtraction and needs no reindex. :func:`posting.deserialize_entries`
never inspected an entry's keys, and :func:`posting.deserialize_manifest` reads ``tokens`` and
ignores anything else the object carries — so an existing blob written with the old fields
still opens, still matches, and simply gets smaller the next time it is written.

API:

- :meth:`index_artifact` — analyze + write posting lists + manifest. Idempotent under
  re-index: an existing manifest's tokens are diffed against the new set, dropped tokens get
  the artifact's entry removed, kept/new tokens get an upsert.
- :meth:`remove_artifact` — read the manifest, evict the artifact's entry from every posting
  list it appears in, delete the manifest.

Field naming convention:

- The blind-token API uses single-char field codes (``t`` / ``d`` /
  ``g`` / ``c``).
- Posting entries use the long-form field names (``"title"`` / ``"description"`` / ``"tags"``
  / ``"content"``).

The indexer maps between the two via :data:`_LONG_TO_SHORT`.

Slot binding (AEAD associated data)
-----------------------------------
Every blob this module writes is bound to its slot: posting lists with
:func:`posting.posting_aad` ``(principal, blind_token)``, manifests with
:func:`posting.manifest_aad` ``(principal, artifact_id)``. The matching reads — here, and in
:mod:`narrowing` — pass the identical AAD, which is the whole requirement: a blob written
with an AAD does not open without it.

This needed no reindex. :func:`posting.decrypt_blob` dual-reads — bound AAD first, then
the legacy unbound form — so every blob written before binding still opens, and each
binds on its next write. The reads therefore keep ``allow_unbound`` at its default
``True``. **Do not flip it to False here.** Doing so makes the binding *enforcing*
rather than merely recorded, which orphans every blob not yet rewritten; it is only
safe after a completed full reindex, and that is an operational decision made once the
corpus is known migrated — not a code change to make alongside the wiring.

See ``.dev/features/mantle-sse-lexical-index.md`` § Indexing Flow.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping

from .blind_tokens import (
    FIELD_CONTENT,
    FIELD_DESCRIPTION,
    FIELD_TAGS,
    FIELD_TITLE,
    blind_token,
)
from .posting import (
    PostingError,
    PostingStore,
    derive_manifest_key,
    derive_posting_key,
    entry_aad,
    manifest_aad,
    pack_entry,
    pack_manifest,
    posting_aad,
    stamp_analyzer_generation,
    unpack_manifest,
    unpack_posting,
)
from .tokenizer import ANALYZER, bigrams as _stem_bigrams, tokenize
from .keys import SseKeyProvider

logger = logging.getLogger(__name__)


# Long-form (posting-entry) → short-form (blind-token API).
_LONG_TO_SHORT = {
    "title": FIELD_TITLE,
    "description": FIELD_DESCRIPTION,
    "tags": FIELD_TAGS,
    "content": FIELD_CONTENT,
}


# ---------------------------------------------------------------------------
# Slot-write atomicity
# ---------------------------------------------------------------------------

_FALLBACK_LOCKS: Dict[str, threading.Lock] = {}
_FALLBACK_LOCKS_GUARD = threading.Lock()


@contextmanager
def _atomic_slot_writes(postings: Any, principal_id: str) -> Iterator[None]:
    """Make one artifact's whole index update atomic, on a store that can do that.

    Indexing a token is a read-modify-write the store cannot do alone. Adding an artifact to a term
    is get → decrypt → `upsert_entry` → encrypt → put, and only this side holds the key, so no store
    can offer it as one operation. Two writers interleaving on one term each read the same posting
    list and the second `put` discards the first's entry, leaving an artifact that reports success
    and is not findable under that term.

    `PostingStore.transaction` closes that, and `SqlitePostingStore` has it: the whole sequence runs
    inside `BEGIN IMMEDIATE`, so a second writer waits rather than overwrites, and the exclusion
    holds across processes because it is the database's rather than a mutex in one interpreter. A
    file store cannot offer it — `_atomic_write` makes a single blob's publication atomic and
    nothing more.

    Reached through `getattr` like every other optional method on the protocol: a store without a
    transaction falls back to a process-wide per-principal lock, which is strictly better than
    nothing and strictly worse than a transaction. `InMemoryPostingStore` is the store that lands
    here, and it is single-process by construction.
    """
    txn = getattr(postings, "transaction", None)
    if txn is not None:
        with txn():
            yield
        return
    with _FALLBACK_LOCKS_GUARD:
        lock = _FALLBACK_LOCKS.get(principal_id)
        if lock is None:
            lock = _FALLBACK_LOCKS[principal_id] = threading.Lock()
    with lock:
        yield


# ---------------------------------------------------------------------------
# Per-field analysis
# ---------------------------------------------------------------------------


def _analyze_field(text: str) -> tuple[list[str], list[str]]:
    """Tokenize one field's text. Returns ``(tokens, distinct_terms)``.

    ``tokens`` is the analyzed token sequence (post-stem), in order, because the bigram pass
    needs adjacency. ``distinct_terms`` is the same sequence deduplicated, first-occurrence
    order kept so the written token set does not depend on set iteration.

    A term's positions are not collected. They would feed ``tf`` and a ``positions`` list on the
    posting entry, and nothing reads either — see the module docstring. A membership index needs to
    know that a term is present, and presence is what a distinct term is.

    Empty / whitespace-only text returns empty containers.
    """
    if not text:
        return [], []
    tokens = tokenize(text)
    return tokens, list(dict.fromkeys(tokens))


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


class SseIndexer:
    """Commit-path indexer for MANTLE-SSE encrypted lexical search."""

    def __init__(
        self,
        oracle: SseKeyProvider,
        posting_store: PostingStore,
    ) -> None:
        self._oracle = oracle
        self._postings = posting_store
        #: Stamped on the first write rather than here, so merely constructing an indexer against
        #: an empty store does not claim a generation it never wrote under. One flag, because the
        #: stamp is idempotent and re-writing it per artifact would be a round trip per artifact.
        self._stamped = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_artifact(
        self,
        principal_id: str,
        collection_id: str,
        artifact_id: str,
        fields: Mapping[str, str],
        request: Any,          # the provider's policy object (oracle.KeyRequest in the platform)
    ) -> int:
        """Index (or re-index) one artifact. Returns the number of distinct
        blind tokens written / updated.

        ``fields`` maps long-form field name to its raw text. Only fields
        in :data:`_LONG_TO_SHORT` are indexed; others are silently ignored
        (so callers can pass artifact context dicts without filtering).

        Re-index path: if a manifest already exists for the artifact, its prior token list is
        read and diffed against the new set. Tokens that were in the prior set but not the new
        one have the artifact's entry stripped (and the posting list deleted if it goes empty).
        """
        if not principal_id:
            raise ValueError("principal_id is required")
        if not collection_id:
            raise ValueError("collection_id is required")
        if not artifact_id:
            raise ValueError("artifact_id is required")

        # The analysis is part of the index format — a blind token is an HMAC of an ANALYSED term,
        # so a client whose pipeline has moved reads terms this store was never filed under and
        # finds nothing, silently. Recording which generation wrote it is what turns that into a
        # readable fact; `wiring` reads it back at open. Best-effort by construction: a store that
        # cannot hold a stamp still indexes.
        if not self._stamped:
            stamp_analyzer_generation(self._postings, ANALYZER)
            self._stamped = True

        owner_sse_key = self._oracle.derive_sse_key(principal_id, request)

        # ---- 1. Analyze each field -----------------------------------
        # token → posting-entry-shape data. Three keys; see the module docstring for which BM25
        # inputs this index does not carry.
        new_entries: dict[str, dict] = {}

        def _entry(field_long: str) -> dict:
            return {
                "artifact_id": artifact_id,
                "collection_id": collection_id,
                "field": field_long,
            }

        for field_long, text in fields.items():
            if field_long not in _LONG_TO_SHORT:
                continue
            tokens, terms = _analyze_field(text)
            if not tokens:
                continue
            field_short = _LONG_TO_SHORT[field_long]

            # Exact tokens: one per unique term.
            for term in terms:
                new_entries[blind_token(owner_sse_key, field_short, term)] = _entry(
                    field_long)

            # Prefix tokens (px3/px4/px5) are not written. Nothing reads them: `narrowing`
            # issues exact-term lookups only, and no query path derives a prefix token, so writing
            # them costs between one and three extra tokens per term per field against a reader
            # that does not exist.
            #
            # `blind_tokens` keeps `prefix_blind_token` and its constants, so a prefix reader turns
            # this on in one place. Prefix tokens are additional keys rather than a modification of
            # an exact-term posting, so an index written before this simply carries keys no lookup
            # names, and no reindex is required in either direction.

            # Bigram tokens: adjacent stem pairs for phrase-query support.
            # Uses blind_token with a space-joined pair as the "term" key —
            # safe because individual stems contain only alphabetic
            # characters after the Porter pipeline (no spaces in stems).
            for bigram in _stem_bigrams(tokens):
                tok = blind_token(owner_sse_key, field_short, bigram)
                if tok not in new_entries:
                    new_entries[tok] = _entry(field_long)

        new_tokens = set(new_entries.keys())

        manifest_key = derive_manifest_key(owner_sse_key, artifact_id)

        # One transaction for the whole artifact, manifest included. The manifest is the record
        # of which tokens reference this artifact, so it is what `remove_artifact` and the re-index
        # diff below both read to find the slots they must rewrite. A manifest committed without
        # its posting updates names slots that do not carry the artifact; posting updates committed
        # without their manifest are unreachable by any later deletion. Either half alone is an
        # index that lies about itself, and on the file store there was no way to ask for both.
        with _atomic_slot_writes(self._postings, principal_id):
            # ---- 2. Read existing manifest (re-index detection) ----------
            manifest_blob = self._postings.get_manifest(principal_id, artifact_id)
            old_tokens: set[str] = set()
            if manifest_blob is not None:
                old_tokens = set(unpack_manifest(
                    manifest_blob, manifest_key,
                    aad=manifest_aad(principal_id, artifact_id),
                ))

            # ---- 3. Diff: drop tokens that left, upsert tokens that stay/arrive
            for tok in old_tokens - new_tokens:
                self._strip_entry(
                    principal_id, owner_sse_key, tok, artifact_id, collection_id,
                )

            for tok, entry in new_entries.items():
                self._upsert_into_posting(principal_id, owner_sse_key, tok, entry)

            # ---- 4. Update manifest --------------------------------------
            if new_tokens:
                self._postings.put_manifest(
                    principal_id, artifact_id,
                    pack_manifest(new_tokens, manifest_key,
                                  aad=manifest_aad(principal_id, artifact_id)),
                )
            else:
                # Empty artifact (no analyzable fields) — drop any stale manifest.
                self._postings.delete_manifest(principal_id, artifact_id)

        return len(new_tokens)

    def remove_artifact(
        self, principal_id: str, artifact_id: str,
        request: Any,          # the provider's policy object (oracle.KeyRequest in the platform)
    ) -> int:
        """Strip every reference to ``artifact_id`` from the SSE index.

        Reads the artifact's manifest to find every blind token referencing
        it, evicts the entry from each posting list (deleting the list
        entirely if it goes empty), and removes the manifest itself.

        Returns the number of posting lists touched. Returns 0 if no
        manifest exists for this artifact (no-op — already removed or
        never indexed).
        """
        if not principal_id or not artifact_id:
            raise ValueError("principal_id and artifact_id are required")

        owner_sse_key = self._oracle.derive_sse_key(principal_id, request)
        manifest_key = derive_manifest_key(owner_sse_key, artifact_id)

        # One transaction, for the reason `index_artifact` states: a deletion that dropped the
        # manifest but not every posting entry leaves entries no later call can find, because the
        # manifest was the only record of where they were.
        with _atomic_slot_writes(self._postings, principal_id):
            manifest_blob = self._postings.get_manifest(principal_id, artifact_id)
            if manifest_blob is None:
                return 0

            old_tokens = unpack_manifest(
                manifest_blob, manifest_key,
                aad=manifest_aad(principal_id, artifact_id),
            )

            touched = 0
            for tok in old_tokens:
                if self._strip_artifact_from_posting(
                    principal_id, owner_sse_key, tok, artifact_id,
                ):
                    touched += 1

            # Drop the manifest itself.
            self._postings.delete_manifest(principal_id, artifact_id)

        return touched

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _upsert_into_posting(
        self,
        principal_id: str,
        owner_sse_key: bytes,
        blind_token: str,
        entry: dict,
    ) -> None:
        """Seal ONE entry and hand it to the store. No read, no scan, no re-encrypt.

        The write the entry layout exists for. A whole-slot form is `get_posting` → decrypt every
        entry in the slot → `upsert_entry`'s linear scan → re-encrypt all of them → `put_posting`,
        which costs O(artifacts already carrying that term) to add one artifact. A body contributes
        thousands of distinct stems, so that cost is what decides whether
        `pipeline_unified._OFFER_FIELDS` can include `content`.

        The entry's own AAD binds it to `(principal, token, artifact, collection)` — strictly more
        than the slot AAD could, because a slot held every collection's entries and had no single
        collection to name.
        """
        self._absorb_legacy_slot(principal_id, owner_sse_key, blind_token)
        self._postings.add_entry(
            principal_id, blind_token,
            str(entry.get("artifact_id") or ""), str(entry.get("collection_id") or ""),
            pack_entry(
                entry, derive_posting_key(owner_sse_key, blind_token),
                aad=entry_aad(principal_id, blind_token,
                              str(entry.get("artifact_id") or ""),
                              str(entry.get("collection_id") or "")),
            ),
        )

    def _absorb_legacy_slot(
        self, principal_id: str, owner_sse_key: bytes, blind_token: str,
    ) -> None:
        """Convert a pre-entry-layout blob for this slot into entries, then drop it.

        The conversion happens where the keys are, and this is the only place that is true. A
        legacy blob seals every entry for a term together, so splitting one requires the owner's SSE
        key, which a store does not have and the blob-copying migration in
        `mantle.system.manage_sse_index` cannot get. There is no migration pass: a slot converts on
        the next write that touches it, and `narrowing` reads the legacy form until then.

        Same dual-read discipline as the AAD binding, for the same reason: the alternative is a flag
        day on an index that is otherwise perfectly readable.

        A blob that will not open is DELETED rather than left. It is unreadable to the reader too
        (`narrowing._entries` drops it), so keeping it means a slot that can never converge — and the
        entries being written right now are the current truth for it.
        """
        blob = self._postings.get_posting(principal_id, blind_token)
        if blob is None:
            return
        key = derive_posting_key(owner_sse_key, blind_token)
        try:
            legacy = unpack_posting(blob, key, aad=posting_aad(principal_id, blind_token))
        except PostingError:
            logger.warning(
                "SSE indexer: legacy posting blob for %s/%s will not open; dropping it",
                principal_id, blind_token[:8], exc_info=True,
            )
            self._postings.delete_posting(principal_id, blind_token)
            return
        for old in legacy:
            artifact_id = str(old.get("artifact_id") or "")
            collection_id = str(old.get("collection_id") or "")
            if not artifact_id:
                continue
            self._postings.add_entry(
                principal_id, blind_token, artifact_id, collection_id,
                pack_entry(old, key, aad=entry_aad(principal_id, blind_token,
                                                   artifact_id, collection_id)),
            )
        self._postings.delete_posting(principal_id, blind_token)

    def _strip_entry(
        self,
        principal_id: str,
        owner_sse_key: bytes,
        blind_token: str,
        artifact_id: str,
        collection_id: str,
    ) -> bool:
        """Strip one (artifact, collection) entry from a slot. Returns whether one went.

        ONE collection, not all of them: partial revocation removes an artifact from one collection
        while it remains in others, and dropping its siblings would un-index it everywhere.
        """
        self._absorb_legacy_slot(principal_id, owner_sse_key, blind_token)
        return self._postings.delete_entry(
            principal_id, blind_token, artifact_id, collection_id or "")

    def _strip_artifact_from_posting(
        self,
        principal_id: str,
        owner_sse_key: bytes,
        blind_token: str,
        artifact_id: str,
    ) -> bool:
        """Strip every entry for ``artifact_id`` from one slot, across all its collections.

        Used by the deletion path, which knows the artifact and not which collections held it — one
        `DELETE` over a key prefix rather than a decrypt-scan-re-encrypt of the slot.
        """
        self._absorb_legacy_slot(principal_id, owner_sse_key, blind_token)
        return self._postings.delete_entries_for_artifact(
            principal_id, blind_token, artifact_id) > 0

__all__ = ["SseIndexer"]
