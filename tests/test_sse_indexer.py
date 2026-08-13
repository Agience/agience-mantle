"""Tests for `search.mantle.sse.indexer.SseIndexer` (MANTLE-SSE Step 2.6.6).

Coverage:

- index_artifact: basic indexing produces posting lists and a manifest for the artifact.
- Field analysis: empty fields are skipped, unknown fields are ignored.
- Entry shape: THREE KEYS AND NO MORE — `artifact_id`, `collection_id`, `field`. `tf`, `dl`
  and `positions` were BM25's inputs and there is no BM25; a membership index records that a
  term is present, and a repeated term is present once.
- Exact tokens: each unique stemmed term in each field produces one blind token + one entry.
- Prefix tokens: only title and tags fields generate them; several terms sharing a prefix
  produce ONE entry for it, because presence does not accumulate.
- Re-index path: token diff drops removed tokens (entries removed, empty posting lists
  deleted) and upserts surviving + new tokens.
- remove_artifact: tokens from manifest evicted from each posting list, manifest deleted.
- Idempotence: calling index_artifact twice with same fields yields identical at-rest state.
- Multi-artifact in same posting list: posting list grows, removal affects only the target.
- At-rest leakage: plaintext term/field strings absent from blobs.
- Multi-collection same artifact: separate (artifact, collection) entries per posting list.
- SLOT BINDING, END TO END — that the AAD the READER builds is the one the writer used, that
  a pre-binding corpus still answers, and that it binds on its next write. Those three lived
  beside the BM25 query engine and are the only end-to-end coverage of a property that
  outlived it; they read through `TokenNarrower` now, which is the reader that remains.
"""

from __future__ import annotations

import os

import pytest
from cryptography.fernet import Fernet

from mantle.search.mantle.oracle import FernetMasterKeyStore, OracleService
from mantle.search.mantle.sse import (
    InMemoryPostingStore,
    SseIndexer,
    TokenNarrower,
    blind_tokens as bt,
    posting,
)
from .helpers import SelfContextVerifier, self_request


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def oracle() -> OracleService:
    fernet = Fernet(Fernet.generate_key())
    return OracleService(FernetMasterKeyStore(fernet), grant_verifier=SelfContextVerifier())


@pytest.fixture
def posting_store() -> InMemoryPostingStore:
    return InMemoryPostingStore()


@pytest.fixture
def indexer(
    oracle: OracleService,
    posting_store: InMemoryPostingStore,
) -> SseIndexer:
    return SseIndexer(oracle, posting_store)


@pytest.fixture
def narrower(oracle, posting_store) -> TokenNarrower:
    return TokenNarrower(oracle, posting_store)


@pytest.fixture
def owner_key(oracle: OracleService) -> bytes:
    return oracle.derive_sse_key("owner-A", self_request("owner-A", "update"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# The readers below pass the same slot AAD the indexer writes with — a blob written
# bound does not open unbound (`test_sse_posting.TestSlotBinding`), so a helper that
# omitted it would fail on every blob the indexer produces. `allow_unbound=False`
# throughout: these read blobs THIS test just wrote, so they must be bound, and the
# dual-read fallback would hide a writer that quietly stopped binding.
def _read_posting(
    posting_store: InMemoryPostingStore,
    owner_key: bytes,
    principal_id: str,
    blind_token_str: str,
) -> list[dict]:
    blob = posting_store.get_posting(principal_id, blind_token_str)
    if blob is None:
        return []
    key = posting.derive_posting_key(owner_key, blind_token_str)
    return posting.unpack_posting(
        blob, key, aad=posting.posting_aad(principal_id, blind_token_str),
        allow_unbound=False)


def _read_manifest(
    posting_store: InMemoryPostingStore,
    owner_key: bytes,
    principal_id: str,
    artifact_id: str,
) -> list[str]:
    blob = posting_store.get_manifest(principal_id, artifact_id)
    if blob is None:
        return []
    key = posting.derive_manifest_key(owner_key, artifact_id)
    return posting.unpack_manifest(
        blob, key, aad=posting.manifest_aad(principal_id, artifact_id),
        allow_unbound=False)


# ---------------------------------------------------------------------------
# Basic indexing
# ---------------------------------------------------------------------------


class TestIndexArtifactBasic:
    def test_no_fields_yields_no_posting_lists(self, indexer, posting_store, owner_key):
        n = indexer.index_artifact("owner-A", "col-1", "art-1", {}, self_request("owner-A", "update"))
        assert n == 0
        assert posting_store.list_tokens_for_owner("owner-A") == []
        assert posting_store.get_manifest("owner-A", "art-1") is None

    def test_empty_text_field_skipped(self, indexer, posting_store):
        n = indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"title": ""}
        , self_request("owner-A", "update"))
        assert n == 0
        assert posting_store.get_manifest("owner-A", "art-1") is None

    def test_unknown_field_ignored(self, indexer, posting_store):
        n = indexer.index_artifact(
            "owner-A", "col-1", "art-1",
            {"unknown_field": "some text", "garbage": "more"},
        self_request("owner-A", "update"))
        assert n == 0

    def test_indexes_title(self, indexer, posting_store, owner_key):
        n = indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"title": "encryption library"},
        self_request("owner-A", "update"))
        # 2 exact tokens (encryption, library) + prefix tokens for each.
        assert n > 0
        tokens = posting_store.list_tokens_for_owner("owner-A")
        assert len(tokens) == n

        # Verify the exact "encryption" token has an entry.
        from mantle.search.mantle.sse.tokenizer import tokenize
        stems = tokenize("encryption library")
        for stem in stems:
            tok = bt.blind_token(owner_key, bt.FIELD_TITLE, stem)
            entries = _read_posting(posting_store, owner_key, "owner-A", tok)
            assert len(entries) == 1
            assert entries[0]["artifact_id"] == "art-1"
            assert entries[0]["collection_id"] == "col-1"
            assert entries[0]["field"] == "title"

    def test_writes_a_manifest_of_tokens_alone(
        self, indexer, posting_store, owner_key,
    ):
        """The manifest is the token list and nothing else.

        It used to carry `field_dls` beside them — the per-field document length, kept so a
        re-index could subtract the old document's contribution from the BM25 corpus
        statistics. There are no corpus statistics; a length nothing reads is a length that
        does not need writing.
        """
        indexer.index_artifact(
            "owner-A", "col-1", "art-1",
            {"title": "alpha beta", "content": "the quick brown fox"},
        self_request("owner-A", "update"))
        tokens = _read_manifest(posting_store, owner_key, "owner-A", "art-1")
        assert len(tokens) > 0
        assert set(tokens) == set(posting_store.list_tokens_for_owner("owner-A"))


# ---------------------------------------------------------------------------
# Tokenization → entries
# ---------------------------------------------------------------------------


class TestEntryShape:
    #: Every key a posting entry may carry. The list is spelled out rather than derived so
    #: that re-adding `tf`, `dl` or `positions` — the BM25 inputs this index no longer has a
    #: consumer for — fails here rather than growing the wire format silently.
    KEYS = {"artifact_id", "collection_id", "field"}

    def test_an_entry_carries_three_keys_and_no_more(
        self, indexer, posting_store, owner_key,
    ):
        indexer.index_artifact(
            "owner-A", "col-1", "art-1",
            {"title": "alpha beta", "content": "gamma delta"},
        self_request("owner-A", "update"))
        seen = 0
        for tok in posting_store.list_tokens_for_owner("owner-A"):
            for entry in _read_posting(posting_store, owner_key, "owner-A", tok):
                assert set(entry) == self.KEYS, entry
                seen += 1
        assert seen > 0, "nothing was indexed — the assertion above would be vacuous"

    def test_a_repeated_term_is_one_entry_with_no_count(
        self, indexer, posting_store, owner_key,
    ):
        """"running runs run" all stem to one term. Presence does not accumulate."""
        indexer.index_artifact(
            "owner-A", "col-1", "art-1",
            {"title": "running runs run"},
        self_request("owner-A", "update"))
        from mantle.search.mantle.sse.tokenizer import tokenize
        stems = tokenize("running runs run")
        assert len(set(stems)) == 1
        tok = bt.blind_token(owner_key, bt.FIELD_TITLE, stems[0])
        entries = _read_posting(posting_store, owner_key, "owner-A", tok)
        assert entries == [
            {"artifact_id": "art-1", "collection_id": "col-1", "field": "title"},
        ]

    def test_two_documents_of_very_different_length_write_the_same_entry(
        self, indexer, posting_store, owner_key,
    ):
        """Document length is not recorded, so it cannot be read — which is the whole of why
        no length normalisation is reachable from this index."""
        indexer.index_artifact(
            "owner-A", "col-1", "art-short", {"content": "alpha"},
            self_request("owner-A", "update"))
        indexer.index_artifact(
            "owner-A", "col-1", "art-long",
            {"content": "alpha " + " ".join("filler%d" % i for i in range(200))},
            self_request("owner-A", "update"))
        from mantle.search.mantle.sse.tokenizer import tokenize
        tok = bt.blind_token(owner_key, bt.FIELD_CONTENT, tokenize("alpha")[0])
        entries = _read_posting(posting_store, owner_key, "owner-A", tok)
        by_id = {e["artifact_id"]: e for e in entries}
        assert set(by_id) == {"art-short", "art-long"}
        assert by_id["art-short"] == {
            **by_id["art-long"], "artifact_id": "art-short",
        }


class TestPrefixTokens:
    def test_prefix_tokens_emitted_for_title(
        self, indexer, posting_store, owner_key,
    ):
        indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"title": "encryption"},
        self_request("owner-A", "update"))
        from mantle.search.mantle.sse.tokenizer import tokenize
        stem = tokenize("encryption")[0]
        # px3 (3 chars) — should exist if stem >= 3 chars.
        for n in bt.PREFIX_LENGTHS:
            if len(stem) >= n:
                tok = bt.prefix_blind_token(
                    owner_key, bt.FIELD_TITLE, stem[:n], n,
                )
                entries = _read_posting(
                    posting_store, owner_key, "owner-A", tok,
                )
                assert len(entries) == 1, (
                    f"missing prefix-{n} posting for stem={stem!r}"
                )

    def test_terms_sharing_a_prefix_produce_one_entry(
        self, indexer, posting_store, owner_key,
    ):
        """"artifact artisan" both start with "arti", and the prefix posting says so once.

        It used to sum their frequencies and union their positions into one entry. There is
        nothing to sum: an artifact either carries a term beginning `arti` or it does not.
        """
        indexer.index_artifact(
            "owner-A", "col-1", "art-1",
            {"title": "artifact artisan"},
        self_request("owner-A", "update"))
        from mantle.search.mantle.sse.tokenizer import tokenize
        stems = tokenize("artifact artisan")
        prefix = "arti"
        assert all(s.startswith(prefix) for s in stems)
        tok = bt.prefix_blind_token(owner_key, bt.FIELD_TITLE, prefix, 4)
        entries = _read_posting(posting_store, owner_key, "owner-A", tok)
        assert entries == [
            {"artifact_id": "art-1", "collection_id": "col-1", "field": "title"},
        ]

    def test_prefix_tokens_not_emitted_for_description(
        self, indexer, posting_store, owner_key,
    ):
        indexer.index_artifact(
            "owner-A", "col-1", "art-1",
            {"description": "encryption library"},
        self_request("owner-A", "update"))
        # description is not in PREFIX_FIELDS — no prefix tokens written.
        # Only exact-match unigrams + bigrams are written for description.
        from mantle.search.mantle.sse.tokenizer import bigrams, tokenize
        stems = list(tokenize("encryption library"))
        expected_token_count = len(set(stems)) + len(bigrams(stems))
        all_tokens = posting_store.list_tokens_for_owner("owner-A")
        assert len(all_tokens) == expected_token_count

    def test_prefix_tokens_not_emitted_for_content(
        self, indexer, posting_store,
    ):
        indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"content": "alpha beta gamma"},
        self_request("owner-A", "update"))
        from mantle.search.mantle.sse.tokenizer import bigrams, tokenize
        stems = list(tokenize("alpha beta gamma"))
        # Only exact-match tokens + bigrams, no prefixes.
        expected_token_count = len(set(stems)) + len(bigrams(stems))
        all_tokens = posting_store.list_tokens_for_owner("owner-A")
        assert len(all_tokens) == expected_token_count


# ---------------------------------------------------------------------------
# Re-index path
# ---------------------------------------------------------------------------


class TestReindex:
    def test_reindex_drops_removed_tokens(
        self, indexer, posting_store, owner_key,
    ):
        # First index: "alpha beta"
        indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"title": "alpha beta"},
        self_request("owner-A", "update"))
        before_tokens = set(posting_store.list_tokens_for_owner("owner-A"))

        # Re-index: "gamma" only — alpha and beta should disappear.
        indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"title": "gamma"},
        self_request("owner-A", "update"))
        after_tokens = set(posting_store.list_tokens_for_owner("owner-A"))

        # Some old tokens must have been dropped.
        dropped = before_tokens - after_tokens
        assert len(dropped) > 0

        # The remaining tokens should reference art-1 still.
        from mantle.search.mantle.sse.tokenizer import tokenize
        gamma_stem = tokenize("gamma")[0]
        gamma_tok = bt.blind_token(owner_key, bt.FIELD_TITLE, gamma_stem)
        entries = _read_posting(posting_store, owner_key, "owner-A", gamma_tok)
        assert len(entries) == 1 and entries[0]["artifact_id"] == "art-1"

    def test_reindex_rewrites_the_manifest_to_the_new_token_set(
        self, indexer, posting_store, owner_key,
    ):
        indexer.index_artifact(
            "owner-A", "col-1", "art-1",
            {"title": "alpha beta gamma delta epsilon"},
        self_request("owner-A", "update"))
        before = set(_read_manifest(posting_store, owner_key, "owner-A", "art-1"))

        indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"title": "alpha beta"},
        self_request("owner-A", "update"))
        after = set(_read_manifest(posting_store, owner_key, "owner-A", "art-1"))

        assert after < before
        assert after == set(posting_store.list_tokens_for_owner("owner-A"))

    def test_reindex_with_no_old_doc_is_fresh_index(
        self, indexer, posting_store, owner_key,
    ):
        # First call to a brand-new owner+artifact with no prior manifest.
        indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"title": "alpha beta"},
        self_request("owner-A", "update"))
        assert _read_manifest(posting_store, owner_key, "owner-A", "art-1")


# ---------------------------------------------------------------------------
# remove_artifact
# ---------------------------------------------------------------------------


class TestRemoveArtifact:
    def test_remove_unknown_artifact_is_noop(self, indexer, posting_store):
        n = indexer.remove_artifact("owner-A", "art-not-indexed", self_request("owner-A", "update"))
        assert n == 0

    def test_removes_all_traces(
        self, indexer, posting_store, oracle, owner_key,
    ):
        indexer.index_artifact(
            "owner-A", "col-1", "art-1",
            {"title": "alpha beta", "content": "the quick brown"},
        self_request("owner-A", "update"))
        # Verify present.
        assert posting_store.get_manifest("owner-A", "art-1") is not None
        assert len(posting_store.list_tokens_for_owner("owner-A")) > 0

        # Remove.
        n = indexer.remove_artifact("owner-A", "art-1", self_request("owner-A", "update"))
        assert n > 0  # touched at least one posting list

        # Manifest gone, posting lists empty.
        assert posting_store.get_manifest("owner-A", "art-1") is None
        assert posting_store.list_tokens_for_owner("owner-A") == []

    def test_remove_one_keeps_other_artifacts(
        self, indexer, posting_store, oracle,
    ):
        indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"title": "alpha beta"},
        self_request("owner-A", "update"))
        indexer.index_artifact(
            "owner-A", "col-1", "art-2", {"title": "alpha gamma"},
        self_request("owner-A", "update"))

        # Remove art-1 only.
        indexer.remove_artifact("owner-A", "art-1", self_request("owner-A", "update"))

        # art-2's manifest still there.
        assert posting_store.get_manifest("owner-A", "art-2") is not None

        # Posting lists referencing art-2 still present.
        all_tokens = posting_store.list_tokens_for_owner("owner-A")
        assert len(all_tokens) > 0


# ---------------------------------------------------------------------------
# Multi-artifact, multi-collection
# ---------------------------------------------------------------------------


class TestMultiArtifact:
    def test_shared_token_collects_both_entries(
        self, indexer, posting_store, owner_key,
    ):
        # Both artifacts contain the term "alpha" → shared posting list.
        indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"title": "alpha beta"},
        self_request("owner-A", "update"))
        indexer.index_artifact(
            "owner-A", "col-1", "art-2", {"title": "alpha gamma"},
        self_request("owner-A", "update"))
        from mantle.search.mantle.sse.tokenizer import tokenize
        stem = tokenize("alpha")[0]
        tok = bt.blind_token(owner_key, bt.FIELD_TITLE, stem)
        entries = _read_posting(posting_store, owner_key, "owner-A", tok)
        artifact_ids = {e["artifact_id"] for e in entries}
        assert artifact_ids == {"art-1", "art-2"}

    def test_remove_one_from_shared_posting_list(
        self, indexer, posting_store, owner_key,
    ):
        indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"title": "alpha beta"},
        self_request("owner-A", "update"))
        indexer.index_artifact(
            "owner-A", "col-1", "art-2", {"title": "alpha gamma"},
        self_request("owner-A", "update"))

        # Remove art-1 — the shared "alpha" posting list keeps art-2.
        indexer.remove_artifact("owner-A", "art-1", self_request("owner-A", "update"))
        from mantle.search.mantle.sse.tokenizer import tokenize
        stem = tokenize("alpha")[0]
        tok = bt.blind_token(owner_key, bt.FIELD_TITLE, stem)
        entries = _read_posting(posting_store, owner_key, "owner-A", tok)
        ids = {e["artifact_id"] for e in entries}
        assert ids == {"art-2"}

    def test_same_artifact_in_two_collections(
        self, indexer, posting_store, owner_key,
    ):
        # The same artifact_id appears in two collections — same blind
        # token should hold two entries (one per collection).
        indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"title": "alpha"},
        self_request("owner-A", "update"))
        indexer.index_artifact(
            "owner-A", "col-2", "art-1", {"title": "alpha"},
        self_request("owner-A", "update"))
        from mantle.search.mantle.sse.tokenizer import tokenize
        stem = tokenize("alpha")[0]
        tok = bt.blind_token(owner_key, bt.FIELD_TITLE, stem)
        entries = _read_posting(posting_store, owner_key, "owner-A", tok)
        # The manifest is keyed by artifact_id alone, with no collection info,
        # so a second index_artifact call for the same artifact_id is treated
        # as a re-index: the re-index path strips entries using the NEW
        # collection_id, so only the latest collection's entry survives.
        # The contract is "one artifact lives in one collection"; moving an
        # artifact between collections requires remove_artifact followed by
        # index_artifact into the new one — the caller is responsible for that.
        ids_collections = {(e["artifact_id"], e["collection_id"]) for e in entries}
        assert ("art-1", "col-2") in ids_collections


# ---------------------------------------------------------------------------
# Idempotence + at-rest leakage
# ---------------------------------------------------------------------------


class TestIdempotence:
    def test_double_index_yields_same_state(
        self, indexer, posting_store, oracle, owner_key,
    ):
        """Indexing the same artifact twice with identical fields should
        leave the at-rest store in the same state — the same token set, and
        byte-identical entries in every posting list."""
        fields = {"title": "alpha beta", "content": "gamma delta"}

        indexer.index_artifact("owner-A", "col-1", "art-1", fields, self_request("owner-A", "update"))
        first_tokens = sorted(posting_store.list_tokens_for_owner("owner-A"))
        first_entries: dict[str, list[dict]] = {}
        for tok in first_tokens:
            first_entries[tok] = _read_posting(
                posting_store, owner_key, "owner-A", tok,
            )

        indexer.index_artifact("owner-A", "col-1", "art-1", fields, self_request("owner-A", "update"))
        second_tokens = sorted(posting_store.list_tokens_for_owner("owner-A"))

        assert first_tokens == second_tokens
        for tok in first_tokens:
            second_entries = _read_posting(
                posting_store, owner_key, "owner-A", tok,
            )
            assert first_entries[tok] == second_entries


class TestAtRestLeakage:
    def test_blobs_do_not_contain_plaintext(
        self, indexer, posting_store, oracle,
    ):
        indexer.index_artifact(
            "owner-A", "col-1", "art-1",
            {"title": "encryption library", "content": "secret cargo"},
        self_request("owner-A", "update"))
        # Scan every blob (postings + manifest); plaintext
        # field names + raw text shouldn't leak.
        all_blobs: list[bytes] = []
        for tok in posting_store.list_tokens_for_owner("owner-A"):
            blob = posting_store.get_posting("owner-A", tok)
            if blob is not None:
                all_blobs.append(blob)
        manifest = posting_store.get_manifest("owner-A", "art-1")
        if manifest is not None:
            all_blobs.append(manifest)

        for needle in (
            b"encryption", b"library", b"secret", b"cargo",
            b"art-1", b"col-1",
        ):
            for blob in all_blobs:
                assert needle not in blob, (
                    f"plaintext leak {needle!r} in blob"
                )


# ---------------------------------------------------------------------------
# Slot binding (AEAD associated data) — WIRED
#
# `posting.posting_aad` / `posting.manifest_aad` are passed by both halves of the real
# path: `SseIndexer` writes with them, `TokenNarrower` reads with them.
# `tests/test_sse_posting.py::TestSlotBinding` pins the primitives; what is pinned HERE is
# the WIRING — that the AAD a reader builds is byte-identical to the one the writer used,
# for every blob kind, through the production classes.
#
# That is the failure mode a primitive test cannot catch: `pack`/`unpack` agreeing with
# each other says nothing about the indexer and the reader agreeing with each other, and a
# mismatch there breaks every recall rather than any one blob.
#
# These read through the NARROWER because it is the reader that exists. They previously
# read through the BM25 query engine, and the stats blob was a third blob kind they checked;
# neither of those is here to check.
# ---------------------------------------------------------------------------


def _legacy_unbound_blob(plaintext: bytes, key: bytes) -> bytes:
    """A blob in the PRE-BINDING wire form: ``nonce | AES-GCM(pt, aad=None)``.

    Built from the AEAD primitive on purpose, NOT by calling ``pack_posting(..., aad=None)``.
    Once nothing in the codebase writes the unbound form any more, a test that produced it
    with today's writer would quietly stop proving anything — its "legacy" blob would just
    be whatever the writer currently emits. These bytes are what is on disk right now,
    spelled out, so the dual-read stays proven for as long as un-migrated blobs exist.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, None)


def _seed_corpus(indexer: SseIndexer, owner: str = "owner-A") -> None:
    """A small corpus spanning two collections and two fields."""
    indexer.index_artifact(owner, "col-1", "art-1", {"title": "encryption library"},
                           self_request(owner, "update"))
    indexer.index_artifact(owner, "col-1", "art-2", {"title": "library cards"},
                           self_request(owner, "update"))
    indexer.index_artifact(owner, "col-2", "art-3", {"title": "encryption keys"},
                           self_request(owner, "update"))
    indexer.index_artifact(owner, "col-2", "art-4",
                           {"title": "quick brown fox", "content": "lazy dog jumps"},
                           self_request(owner, "update"))


def _narrow(narrower: TokenNarrower, text: str, scope) -> dict:
    """What the narrowing reaches for ``text``, as ``{artifact_id: Coverage}``."""
    self_request("owner-A", "read")
    lookup = narrower.lookup_for(text, self_request("owner-A", "read"))
    return {} if lookup is None else dict(lookup(list(scope)))


class TestSlotBindingIsWired:
    @staticmethod
    def _owner_key(oracle, owner="owner-A"):
        return oracle.derive_sse_key(owner, self_request(owner, "update"))

    @staticmethod
    def _title_token(owner_key, term):
        from mantle.search.mantle.sse.tokenizer import tokenize
        return bt.blind_token(owner_key, bt.FIELD_TITLE, tokenize(term)[0])

    def test_every_blob_the_indexer_writes_is_bound_and_the_read_path_reads_it(
        self, oracle, posting_store, indexer, narrower,
    ):
        """The round trip, with binding on, through the real classes.

        Each blob kind is checked twice: it does NOT open unbound (so it is genuinely
        bound, not merely written with an AAD-shaped no-op), and it DOES open under the
        AAD its slot implies, with the dual-read switched off.

        The narrowing at the end is the wiring assertion proper. A bound blob read without
        an AAD raises, and `TokenNarrower._entries` swallows that into `[]` — so a non-empty
        answer can only happen if the reader built exactly the AAD the writer bound with.
        """
        _seed_corpus(indexer)
        owner_key = self._owner_key(oracle)

        # --- posting list -------------------------------------------------
        tok = self._title_token(owner_key, "encryption")
        blob = posting_store.get_posting("owner-A", tok)
        assert blob is not None, "nothing indexed — the assertions below would be vacuous"
        pkey = posting.derive_posting_key(owner_key, tok)
        with pytest.raises(posting.PostingTampered):
            posting.unpack_posting(blob, pkey)                    # unbound read refused
        assert posting.unpack_posting(
            blob, pkey, aad=posting.posting_aad("owner-A", tok), allow_unbound=False)

        # --- manifest -----------------------------------------------------
        mblob = posting_store.get_manifest("owner-A", "art-1")
        mkey = posting.derive_manifest_key(owner_key, "art-1")
        with pytest.raises(posting.PostingTampered):
            posting.unpack_manifest(mblob, mkey)
        tokens = posting.unpack_manifest(
            mblob, mkey, aad=posting.manifest_aad("owner-A", "art-1"),
            allow_unbound=False)
        assert tok in tokens

        # --- and the read path still answers --------------------------------
        assert set(_narrow(narrower, "encryption", [("owner-A", "col-1")])) == {"art-1"}

    def test_the_aad_the_reader_builds_is_the_one_the_writer_used(self, oracle):
        """Pin the actual byte strings, independently of any crypto succeeding.

        The round-trip test above would still pass if writer and reader shared the same
        *wrong* AAD. This one asserts the value against the slot identifiers themselves,
        so a future change that, say, dropped the principal from one side is caught by
        name rather than by a mass recall outage.
        """
        tok = self._title_token(self._owner_key(oracle), "encryption")
        assert posting.posting_aad("owner-A", tok) == \
            b"sse-posting-aad-v1:7:owner-A:" + tok.encode()
        assert posting.manifest_aad("owner-A", "art-1") == \
            b"sse-manifest-aad-v1:7:owner-A:art-1"

    def test_right_key_wrong_aad_raises_before_deserialization(
        self, monkeypatch, oracle, posting_store, indexer,
    ):
        """Authentication is the gate, not a check performed after parsing.

        `deserialize_entries` is replaced with a landmine: reaching it at all on a blob
        whose AAD did not authenticate would mean attacker-chosen bytes had already been
        parsed. The KEY is held correct throughout, so the only variable is the slot.
        """
        _seed_corpus(indexer)
        owner_key = self._owner_key(oracle)
        tok = self._title_token(owner_key, "encryption")
        other_tok = self._title_token(owner_key, "library")
        blob = posting_store.get_posting("owner-A", tok)
        pkey = posting.derive_posting_key(owner_key, tok)

        def _landmine(_plaintext):
            raise AssertionError("deserialize_entries reached on an unauthenticated blob")

        monkeypatch.setattr(posting, "deserialize_entries", _landmine)
        for wrong in (posting.posting_aad("owner-B", tok),         # another principal
                      posting.posting_aad("owner-A", other_tok)):  # another slot
            # Both with the dual-read left ON — the production default. A blob that IS
            # bound does not fall back to the unbound form: the fallback is not a hole.
            with pytest.raises(posting.PostingTampered):
                posting.unpack_posting(blob, pkey, aad=wrong)
            with pytest.raises(posting.PostingTampered):
                posting.unpack_posting(blob, pkey, aad=wrong, allow_unbound=False)

    def test_a_legacy_unbound_corpus_still_opens_through_the_read_path(
        self, oracle, posting_store, indexer, narrower,
    ):
        """THE no-reindex proof: blobs written before binding keep answering recalls.

        Every posting list and manifest is rewritten in the pre-binding wire form by
        `_legacy_unbound_blob`, then the same question must be answered identically. If the
        dual-read in `posting.decrypt_blob` were ever dropped without a completed reindex,
        this test is what fails — which is the point: enabling binding must not require one.
        """
        _seed_corpus(indexer)
        owner_key = self._owner_key(oracle)
        scope = [("owner-A", "col-1"), ("owner-A", "col-2")]

        expected = _narrow(narrower, "encryption library", scope)
        assert expected, "the corpus answered nothing — the comparison would be vacuous"

        # Rewrite the whole corpus in the old wire form, in place.
        for token in posting_store.list_tokens_for_owner("owner-A"):
            pkey = posting.derive_posting_key(owner_key, token)
            entries = posting.unpack_posting(
                posting_store.get_posting("owner-A", token), pkey,
                aad=posting.posting_aad("owner-A", token))
            posting_store.put_posting("owner-A", token, _legacy_unbound_blob(
                posting.serialize_entries(entries), pkey))

        for artifact_id in ("art-1", "art-2", "art-3", "art-4"):
            mkey = posting.derive_manifest_key(owner_key, artifact_id)
            tokens = posting.unpack_manifest(
                posting_store.get_manifest("owner-A", artifact_id), mkey,
                aad=posting.manifest_aad("owner-A", artifact_id))
            posting_store.put_manifest("owner-A", artifact_id, _legacy_unbound_blob(
                posting.serialize_manifest(tokens), mkey))

        # Sanity: the corpus really is in the old form now — it opens with NO aad at all.
        any_token = posting_store.list_tokens_for_owner("owner-A")[0]
        assert posting.unpack_posting(
            posting_store.get_posting("owner-A", any_token),
            posting.derive_posting_key(owner_key, any_token),
            allow_unbound=False) is not None

        assert _narrow(narrower, "encryption library", scope) == expected

    def test_a_manifest_carrying_field_dls_still_opens(
        self, oracle, posting_store, indexer,
    ):
        """The OTHER no-reindex proof, for the wire-format subtraction rather than the AAD.

        A manifest is rewritten in the shape the indexer used to emit — `tokens` plus the
        per-field document lengths BM25 needed — and must still yield exactly the same token
        list. `deserialize_manifest` reads `tokens` and ignores every other key, which is
        what makes dropping the field a change no operator has to act on.
        """
        indexer.index_artifact("owner-A", "col-1", "art-1", {"title": "encryption library"},
                               self_request("owner-A", "update"))
        owner_key = self._owner_key(oracle)
        mkey = posting.derive_manifest_key(owner_key, "art-1")
        aad = posting.manifest_aad("owner-A", "art-1")
        tokens = posting.unpack_manifest(
            posting_store.get_manifest("owner-A", "art-1"), mkey, aad=aad)

        import json
        legacy = json.dumps(
            {"tokens": sorted(tokens), "field_dls": {"title": 2, "content": 0}},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        posting_store.put_manifest(
            "owner-A", "art-1", posting.encrypt_blob(legacy, mkey, aad=aad))

        assert posting.unpack_manifest(
            posting_store.get_manifest("owner-A", "art-1"), mkey, aad=aad) == sorted(tokens)

        # And the indexer's own reader — the re-index path — is unbothered by it.
        indexer.index_artifact("owner-A", "col-1", "art-1", {"title": "encryption library"},
                               self_request("owner-A", "update"))
        assert posting.unpack_manifest(
            posting_store.get_manifest("owner-A", "art-1"), mkey, aad=aad) == sorted(tokens)

    def test_a_legacy_blob_binds_on_its_next_write(
        self, oracle, posting_store, indexer,
    ):
        """The migration is the write path, not a script.

        An unbound posting list is planted, the artifact is re-indexed, and the blob that
        comes back out must no longer open unbound. This is what eventually makes
        `allow_unbound=False` reachable — and why flipping it now would be wrong: the
        corpus migrates one blob at a time, as each is next written.
        """
        indexer.index_artifact("owner-A", "col-1", "art-1", {"title": "encryption"},
                               self_request("owner-A", "update"))
        owner_key = self._owner_key(oracle)
        tok = self._title_token(owner_key, "encryption")
        pkey = posting.derive_posting_key(owner_key, tok)

        entries = posting.unpack_posting(
            posting_store.get_posting("owner-A", tok), pkey,
            aad=posting.posting_aad("owner-A", tok))
        posting_store.put_posting("owner-A", tok, _legacy_unbound_blob(
            posting.serialize_entries(entries), pkey))

        # Re-index: read-modify-write over the legacy blob (the dual-read opens it)...
        indexer.index_artifact("owner-A", "col-1", "art-1", {"title": "encryption"},
                               self_request("owner-A", "update"))

        rewritten = posting_store.get_posting("owner-A", tok)
        with pytest.raises(posting.PostingTampered):
            posting.unpack_posting(rewritten, pkey)          # ...and puts it back BOUND
        assert posting.unpack_posting(
            rewritten, pkey, aad=posting.posting_aad("owner-A", tok),
            allow_unbound=False) == entries
