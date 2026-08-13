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

WHAT A POSTING ENTRY CARRIES, AND WHAT IT NO LONGER DOES
--------------------------------------------------------
``{"artifact_id", "collection_id", "field"}`` and nothing else. ``tf``, ``dl`` and
``positions`` were BM25's inputs — term frequency, document length, and token offsets — and
BM25 is gone: the recall path answers membership and counts how much of a query each artifact
matched, neither of which reads any of the three. ``positions`` was never read by anything in
``src/`` even while the scorer existed. The manifest's ``field_dls`` went the same way; it
existed to roll a document's length contribution back out of the corpus statistics on
re-index, and there are no corpus statistics.

This is a WIRE-FORMAT SUBTRACTION AND NEEDS NO REINDEX. :func:`posting.deserialize_entries`
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
from typing import Any, Mapping

from .blind_tokens import (
    FIELD_CONTENT,
    FIELD_DESCRIPTION,
    FIELD_TAGS,
    FIELD_TITLE,
    PREFIX_FIELDS,
    PREFIX_LENGTHS,
    blind_token,
    prefix_blind_token,
)
from .posting import (
    PostingStore,
    derive_manifest_key,
    derive_posting_key,
    manifest_aad,
    pack_manifest,
    pack_posting,
    posting_aad,
    remove_artifact_collection_entries,
    unpack_manifest,
    unpack_posting,
    upsert_entry,
)
from .tokenizer import bigrams as _stem_bigrams, tokenize
from .keys import SseKeyProvider

logger = logging.getLogger(__name__)


# Long-form (posting-entry) → short-form (blind-token API).
_LONG_TO_SHORT = {
    "title": FIELD_TITLE,
    "description": FIELD_DESCRIPTION,
    "tags": FIELD_TAGS,
    "content": FIELD_CONTENT,
}

# Long-form names of fields eligible for prefix tokens.
_PREFIX_LONG_FIELDS = frozenset({
    long_name for long_name, short in _LONG_TO_SHORT.items()
    if short in PREFIX_FIELDS
})


# ---------------------------------------------------------------------------
# Per-field analysis
# ---------------------------------------------------------------------------


def _analyze_field(text: str) -> tuple[list[str], list[str]]:
    """Tokenize one field's text. Returns ``(tokens, distinct_terms)``.

    ``tokens`` is the analyzed token sequence (post-stem), in order, because the bigram pass
    needs adjacency. ``distinct_terms`` is the same sequence deduplicated, first-occurrence
    order kept so the written token set does not depend on set iteration.

    A term's POSITIONS are no longer collected. They were computed to derive ``tf`` and to fill
    a ``positions`` list on the posting entry; nothing reads either — see the module docstring.
    A membership index needs to know that a term is present, and presence is what a distinct
    term is.

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

        owner_sse_key = self._oracle.derive_sse_key(principal_id, request)

        # ---- 1. Analyze each field -----------------------------------
        # token → posting-entry-shape data. Three keys; see the module docstring for what the
        # other three used to be and why their absence needs no reindex.
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

            # Prefix tokens (title + tags only): one per distinct prefix of each length. The
            # aggregation across terms sharing a prefix used to be a sum of their frequencies;
            # with nothing to sum, a prefix is present or it is not.
            if field_long in _PREFIX_LONG_FIELDS:
                for n in PREFIX_LENGTHS:
                    for prefix in {t[:n] for t in terms if len(t) >= n}:
                        new_entries[
                            prefix_blind_token(owner_sse_key, field_short, prefix, n)
                        ] = _entry(field_long)

            # Bigram tokens: adjacent stem pairs for phrase-query support.
            # Uses blind_token with a space-joined pair as the "term" key —
            # safe because individual stems contain only alphabetic
            # characters after the Porter pipeline (no spaces in stems).
            for bigram in _stem_bigrams(tokens):
                tok = blind_token(owner_sse_key, field_short, bigram)
                if tok not in new_entries:
                    new_entries[tok] = _entry(field_long)

        new_tokens = set(new_entries.keys())

        # ---- 2. Read existing manifest (re-index detection) ----------
        manifest_key = derive_manifest_key(owner_sse_key, artifact_id)
        manifest_blob = self._postings.get_manifest(principal_id, artifact_id)
        old_tokens: set[str] = set()
        if manifest_blob is not None:
            old_tokens = set(unpack_manifest(
                manifest_blob, manifest_key,
                aad=manifest_aad(principal_id, artifact_id),
            ))

        # ---- 3. Diff: drop tokens that left, upsert tokens that stay/arrive
        dropped = old_tokens - new_tokens
        for tok in dropped:
            self._strip_entry(
                principal_id, owner_sse_key, tok, artifact_id, collection_id,
            )

        for tok, entry in new_entries.items():
            self._upsert_into_posting(
                principal_id, owner_sse_key, tok, entry,
            )

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
        """Read-modify-write one posting list with a single entry upsert.

        Read and write use the SAME slot AAD, which is what makes the read-modify-write
        safe to bind in place: whatever this read accepted (bound, or legacy unbound via
        :func:`posting.decrypt_blob`'s dual-read) the write puts back bound.
        """
        key = derive_posting_key(owner_sse_key, blind_token)
        aad = posting_aad(principal_id, blind_token)
        blob = self._postings.get_posting(principal_id, blind_token)
        entries = unpack_posting(blob, key, aad=aad) if blob else []
        upsert_entry(entries, entry)
        self._postings.put_posting(
            principal_id, blind_token, pack_posting(entries, key, aad=aad),
        )

    def _strip_entry(
        self,
        principal_id: str,
        owner_sse_key: bytes,
        blind_token: str,
        artifact_id: str,
        collection_id: str,
    ) -> bool:
        """Strip one (artifact, collection) entry from a posting list.

        Drops the posting list entirely if it goes empty. Returns True if
        an entry was actually removed.
        """
        blob = self._postings.get_posting(principal_id, blind_token)
        if blob is None:
            return False
        key = derive_posting_key(owner_sse_key, blind_token)
        aad = posting_aad(principal_id, blind_token)
        entries = unpack_posting(blob, key, aad=aad)
        before = len(entries)
        entries = remove_artifact_collection_entries(
            entries, artifact_id, collection_id,
        )
        if len(entries) == before:
            return False
        if entries:
            self._postings.put_posting(
                principal_id, blind_token, pack_posting(entries, key, aad=aad),
            )
        else:
            self._postings.delete_posting(principal_id, blind_token)
        return True

    def _strip_artifact_from_posting(
        self,
        principal_id: str,
        owner_sse_key: bytes,
        blind_token: str,
        artifact_id: str,
    ) -> bool:
        """Strip every entry for ``artifact_id`` from one posting list
        (across all collections it appears in). Used by the deletion path."""
        blob = self._postings.get_posting(principal_id, blind_token)
        if blob is None:
            return False
        key = derive_posting_key(owner_sse_key, blind_token)
        aad = posting_aad(principal_id, blind_token)
        entries = unpack_posting(blob, key, aad=aad)
        before = len(entries)
        entries = [e for e in entries if e.get("artifact_id") != artifact_id]
        if len(entries) == before:
            return False
        if entries:
            self._postings.put_posting(
                principal_id, blind_token, pack_posting(entries, key, aad=aad),
            )
        else:
            self._postings.delete_posting(principal_id, blind_token)
        return True

__all__ = ["SseIndexer"]
