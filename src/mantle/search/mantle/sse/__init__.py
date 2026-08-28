"""MANTLE-SSE — encrypted lexical search.

Per `.dev/features/mantle-sse-lexical-index.md`. The canonical lexical backend:
blind-token posting lists encrypted in S3.

Design summary:

- **Blind tokens**: deterministic HMAC-SHA256 over (owner_sse_key, field+term)
  produces opaque tokens. The S3 backing store sees only hex strings — never
  plaintext terms.
- **Posting lists**: per-blind-token JSON (entries: artifact_id, collection_id,
  field), AES-256-GCM encrypted with HKDF-derived per-token keys.
- **Narrowing, not scoring**: a query's stems are looked up and MET against the light cone.
  The lookup counts how many stems each artifact matched, which is what orders a recall with
  no query vector. There is no corpus statistic, no IDF, no term frequency and no document
  length anywhere in this package.
- **Authorization**: same light-cone BFS as the MANTLE vector layer. Both
  index paths share the authorized scope.

Layered atop the existing MANTLE vector infrastructure:

- :class:`OracleService` (existing) — extended with an ``sse`` derivation
  context for owner SSE keys.
- :class:`LightConeResolver` (existing) — reused unchanged.
- New SSE modules below — tokenizer, blind tokens, posting list manager,
  indexer, narrowing.

Module map:

- :mod:`.tokenizer` — English analysis pipeline (lowercase → possessive →
  stop words → Porter stemmer).
- :mod:`.blind_tokens` — HMAC-based token generator (field prefix + prefix
  tokens for title/tags).
- :mod:`.posting` — posting list manager (S3 CRUD; HKDF + AES-256-GCM per
  blind token; manifest tracking per artifact).
- :mod:`.indexer` — commit-path indexer (parallel to ``MantleIndexer``).
- :mod:`.narrowing` — blind-token membership + query coverage: which artifacts carry these
  stems, and how many of them each one carries. Same posting lists, no corpus statistic.
- :mod:`.s3_stores` / :mod:`.sqlite_stores` — production storage backends. `.file_stores`
  holds only the filesystem path law now, shared with the vector arm's local cell store.
"""

from __future__ import annotations

from .blind_tokens import (
    FIELD_CONTENT,
    FIELD_DESCRIPTION,
    FIELD_TAGS,
    FIELD_TITLE,
    PREFIX_FIELDS,
    PREFIX_LENGTHS,
    VALID_FIELDS,
    blind_token,
    blind_tokens_for_terms,
    prefix_blind_token,
    prefix_blind_tokens,
)
from .posting import (
    InMemoryPostingStore,
    PostingError,
    PostingMalformed,
    PostingStore,
    PostingTampered,
    artifact_ids_in_entries,
    decrypt_blob,
    derive_manifest_key,
    derive_posting_key,
    deserialize_entries,
    deserialize_manifest,
    encrypt_blob,
    entry_count,
    manifest_aad,
    pack_manifest,
    pack_posting,
    posting_aad,
    remove_artifact_collection_entries,
    remove_artifact_entries,
    serialize_entries,
    serialize_manifest,
    unpack_manifest,
    unpack_posting,
    upsert_entry,
)
from .indexer import SseIndexer
from .narrowing import Coverage, CoverageLookup, TokenNarrower, phrase_stems
from .s3_stores import S3PostingStore
from .sqlite_stores import SqlitePostingStore
from .tokenizer import (
    porter_stem,
    split_words,
    strip_possessive,
    tokenize,
)

__all__ = [
    # Tokenizer
    "porter_stem",
    "split_words",
    "strip_possessive",
    "tokenize",
    # Blind tokens
    "FIELD_TITLE",
    "FIELD_DESCRIPTION",
    "FIELD_TAGS",
    "FIELD_CONTENT",
    "VALID_FIELDS",
    "PREFIX_FIELDS",
    "PREFIX_LENGTHS",
    "blind_token",
    "blind_tokens_for_terms",
    "prefix_blind_token",
    "prefix_blind_tokens",
    # Posting list manager
    "PostingError",
    "PostingMalformed",
    "PostingTampered",
    "derive_posting_key",
    "derive_manifest_key",
    # Slot binding (AEAD associated data) — the write paths in `sse/` bind every blob,
    # so the AAD builders belong on the same surface as the key derivations they pair with:
    # a caller that can reach `encrypt_blob` must be able to reach the slot it binds to.
    "posting_aad",
    "manifest_aad",
    "encrypt_blob",
    "decrypt_blob",
    "serialize_entries",
    "deserialize_entries",
    "pack_posting",
    "unpack_posting",
    "serialize_manifest",
    "deserialize_manifest",
    "pack_manifest",
    "unpack_manifest",
    "upsert_entry",
    "remove_artifact_entries",
    "remove_artifact_collection_entries",
    "entry_count",
    "artifact_ids_in_entries",
    "PostingStore",
    "InMemoryPostingStore",
    # Commit-path indexer
    "SseIndexer",
    # Blind-token narrowing — membership, and how much of the query each member matched
    "Coverage",
    "CoverageLookup",
    "TokenNarrower",
    "phrase_stems",
    # S3-backed production store
    "S3PostingStore",
    # SQLite-backed production store — the standalone index
    "SqlitePostingStore",
    # Router-shape adapter
    "MantleSseSearchAccessor",
]


# ── `router_accessor`, resolved lazily (PEP 562) ─────────────────────────────────────────
#
# It needs `embeddings` + `..lightcone` + `..oracle` + `..engine`: the vector-arm and custody
# integration, which a pure lexical/blind-token caller does not need — the vector arm is
# model-dependent, and a data store does no reasoning. Importing it at module scope would pull
# numpy and the whole custody hierarchy into every import of this package, including the
# dependency-clean modules that don't touch either.
#
# `narrowing` is NOT lazy, and the reason is worth stating because it looks like it should be. It
# catches `MasterKeyMissing` — the refusal a key provider raises for a principal that has never
# been written to — and an `except` matches on the CLASS OBJECT, so the catch must name the same
# class the raise does. When that class was defined in `..oracle`, honouring the constraint meant
# a module-scope import of the whole custodian, and an eager import here would have put it behind
# every import of this package, including `tokenizer` and `posting`. The class now lives in
# `..custody`, which defines the two refusals and nothing else, so the catch costs an import of
# two names and `narrowing` is eager with the rest of the lexical core.
#
# PEP 562 `__getattr__` keeps the public API as-is — `from ...sse import MantleSseSearchAccessor`
# still works — while the import cost lands only on the caller who asks for the name. It stays in
# `__all__` because it is still exported; resolution just happens on first access.
_LAZY = {
    "MantleSseSearchAccessor": ".router_accessor",
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    from importlib import import_module
    return getattr(import_module(target, __name__), name)


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
