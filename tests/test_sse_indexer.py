"""Tests for `search.mantle.sse.indexer.SseIndexer` (MANTLE-SSE Step 2.6.6).

Coverage:

- index_artifact: basic indexing produces posting lists and a manifest for the artifact.
- Field analysis: empty fields are skipped, unknown fields are ignored.
- Entry shape: three keys and no more — `artifact_id`, `collection_id`, `field`. `tf`, `dl`
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
- Slot binding, end to end — that the AAD the READER builds is the one the writer used, that
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


# The readers below pass the same AAD the indexer writes with — a blob written bound does not open
# unbound (`test_sse_posting.TestSlotBinding`), so a helper that omitted it would fail on every blob
# the indexer produces. `allow_unbound=False` throughout: these read blobs THIS test just wrote, so
# they must be bound, and the dual-read fallback would hide a writer that quietly stopped binding.
#
# A slot is a set of per-entry blobs rather than one blob holding a list. `SseIndexer` writes through
# `add_entry`, so adding an artifact to a term costs one sealed write instead of decrypting, scanning
# and re-encrypting every entry already there. This helper hides the packing from the assertions
# below, which are about what ends up indexed rather than how it is stored — and it
# builds each entry's AAD from the identity the store returns alongside it, which is what makes the
# per-entry binding enforced rather than merely written.
def _read_posting(
    posting_store: InMemoryPostingStore,
    owner_key: bytes,
    principal_id: str,
    blind_token_str: str,
) -> list[dict]:
    key = posting.derive_posting_key(owner_key, blind_token_str)
    entries = [
        posting.unpack_entry(
            blob, key,
            aad=posting.entry_aad(principal_id, blind_token_str, artifact_id, collection_id),
            allow_unbound=False)
        for artifact_id, collection_id, blob in posting_store.get_entries(
            principal_id, blind_token_str)
    ]
    if entries:
        return entries
    # A legacy whole-slot blob, for the tests that plant one deliberately.
    blob = posting_store.get_posting(principal_id, blind_token_str)
    if blob is None:
        return []
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
    #: Every key a posting entry may carry. The list is spelled out rather than derived so that
    #: adding `tf`, `dl` or `positions` — the BM25 inputs this index has no consumer for — fails
    #: here rather than growing the wire format silently.
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
    def test_no_prefix_token_is_written_for_any_field(
        self, indexer, posting_store, owner_key,
    ):
        """px3/px4/px5 are NOT written — for title and tags either, which once had them.

        They were generated at index time and READ BY NOTHING: `narrowing` issues only exact-term
        lookups. That made them a permanent write cost — one to three extra HMACs and store writes
        per term per field — paid to an index no query names.

        This asserts the removal directly rather than by the absence of the old tests, because
        "nobody wrote a test" and "the behaviour is gone" look identical in a suite otherwise.

        `blind_tokens.prefix_blind_token` deliberately still EXISTS and is still correct — a
        future prefix reader turns this back on in one place. What must not come back on its own
        is the unconditional write.
        """
        indexer.index_artifact(
            "owner-A", "col-1", "art-1",
            {"title": "encryption", "tags": "artifact artisan"},
            self_request("owner-A", "update"))

        from mantle.search.mantle.sse.tokenizer import tokenize
        for field, text in ((bt.FIELD_TITLE, "encryption"), (bt.FIELD_TAGS, "artifact artisan")):
            for stem in tokenize(text):
                for n in bt.PREFIX_LENGTHS:
                    if len(stem) < n:
                        continue
                    tok = bt.prefix_blind_token(owner_key, field, stem[:n], n)
                    entries = _read_posting(posting_store, owner_key, "owner-A", tok)
                    assert entries == [], (
                        f"a px{n} posting was written for stem={stem!r} on field={field!r}. "
                        "Prefix tokens are write-only — nothing reads them — so writing one is "
                        "pure write amplification on the hot index path."
                    )

    def test_the_exact_term_is_still_written(
        self, indexer, posting_store, owner_key,
    ):
        """Control for the test above: prove the indexer ran and wrote something.

        Without this, a broken fixture that indexed nothing at all would satisfy
        "no prefix tokens were written" perfectly.
        """
        indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"title": "encryption"},
            self_request("owner-A", "update"))
        from mantle.search.mantle.sse.tokenizer import tokenize
        stem = tokenize("encryption")[0]
        tok = bt.blind_token(owner_key, bt.FIELD_TITLE, stem)
        assert _read_posting(posting_store, owner_key, "owner-A", tok) == [
            {"artifact_id": "art-1", "collection_id": "col-1", "field": "title"},
        ], "the exact-term posting is missing — the indexer did not run"

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
# These read through the narrower, because it is the reader that exists: there is no BM25 query
# engine and no stats blob in this index.
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

        # --- posting entries ----------------------------------------------
        # The binding is per entry, which binds more than a slot-level one can:
        # `(principal, token, artifact, collection)` rather than `(principal, token)`. A whole-slot
        # blob holds every collection's entries, so collection separation there can only be a
        # plaintext post-filter; an entry is one pair, so both go in.
        tok = self._title_token(owner_key, "encryption")
        found = posting_store.get_entries("owner-A", tok)
        assert found, "nothing indexed — the assertions below would be vacuous"
        pkey = posting.derive_posting_key(owner_key, tok)
        for artifact_id, collection_id, blob in found:
            with pytest.raises(posting.PostingTampered):
                posting.unpack_entry(blob, pkey)               # unbound read refused
            assert posting.unpack_entry(
                blob, pkey,
                aad=posting.entry_aad("owner-A", tok, artifact_id, collection_id),
                allow_unbound=False)

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
        # Every component length-prefixed, so no two different tuples can encode alike — four
        # parts here, two of them caller-supplied ids that may contain any character.
        assert posting.entry_aad("owner-A", tok, "art-1", "col-1") == \
            b"sse-entry-aad-v1:7:owner-A:" + b"%d:" % len(tok) + tok.encode() + \
            b":5:art-1:5:col-1"

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
        artifact_id, collection_id, blob = posting_store.get_entries("owner-A", tok)[0]
        pkey = posting.derive_posting_key(owner_key, tok)

        def _landmine(_plaintext):
            raise AssertionError("deserialize_entries reached on an unauthenticated blob")

        monkeypatch.setattr(posting, "deserialize_entries", _landmine)
        # Each wrong AAD moves the entry exactly one hop: to another principal, another slot,
        # another artifact, another collection. The last two are the ones a slot-level binding
        # cannot express at all, leaving an entry re-filed under a collection its owner never put
        # it in to a plaintext post-filter on the read path.
        for wrong in (posting.entry_aad("owner-B", tok, artifact_id, collection_id),
                      posting.entry_aad("owner-A", other_tok, artifact_id, collection_id),
                      posting.entry_aad("owner-A", tok, "art-999", collection_id),
                      posting.entry_aad("owner-A", tok, artifact_id, "col-999")):
            # Both with the dual-read left ON — the production default. A blob that IS
            # bound does not fall back to the unbound form: the fallback is not a hole.
            with pytest.raises(posting.PostingTampered):
                posting.unpack_entry(blob, pkey, aad=wrong)
            with pytest.raises(posting.PostingTampered):
                posting.unpack_entry(blob, pkey, aad=wrong, allow_unbound=False)

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

        # Rewrite the whole corpus in the old wire form, in place: one UNBOUND whole-slot blob
        # per token, with the per-entry rows removed. That is exactly what an index written
        # before either change looks like — pre-binding AND pre-entry-layout — so this proves
        # both dual-reads at once.
        for token in list(posting_store.list_tokens_for_owner("owner-A")):
            pkey = posting.derive_posting_key(owner_key, token)
            entries = [
                posting.unpack_entry(
                    blob, pkey,
                    aad=posting.entry_aad("owner-A", token, aid, cid))
                for aid, cid, blob in posting_store.get_entries("owner-A", token)
            ]
            for aid, cid, _blob in list(posting_store.get_entries("owner-A", token)):
                posting_store.delete_entry("owner-A", token, aid, cid)
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

    def test_a_legacy_whole_slot_blob_converts_on_its_next_write(
        self, oracle, posting_store, indexer,
    ):
        """The migration is the write path, not a script — and now it migrates TWO things.

        A legacy slot is planted: one UNBOUND whole-slot blob, no per-entry rows. Re-indexing the
        artifact must (a) absorb it into per-entry rows, (b) bind each of those to its own
        `(principal, token, artifact, collection)`, and (c) delete the blob so the slot has one
        representation again.

        There is no migration script for this, and there cannot be one. Splitting a legacy blob
        requires the owner's SSE key, which no store holds and which a blob-copying pass like
        `mantle.system.manage_sse_index` has no way to obtain. Conversion happens where the keys
        already are: here, on the next write, as AAD binding does.
        """
        indexer.index_artifact("owner-A", "col-1", "art-1", {"title": "encryption"},
                               self_request("owner-A", "update"))
        owner_key = self._owner_key(oracle)
        tok = self._title_token(owner_key, "encryption")
        pkey = posting.derive_posting_key(owner_key, tok)

        entries = [
            posting.unpack_entry(blob, pkey,
                                 aad=posting.entry_aad("owner-A", tok, aid, cid))
            for aid, cid, blob in posting_store.get_entries("owner-A", tok)
        ]
        for aid, cid, _blob in list(posting_store.get_entries("owner-A", tok)):
            posting_store.delete_entry("owner-A", tok, aid, cid)
        posting_store.put_posting("owner-A", tok, _legacy_unbound_blob(
            posting.serialize_entries(entries), pkey))
        assert posting_store.get_entries("owner-A", tok) == [], "the slot must start legacy-only"

        # Re-index: absorb the legacy blob (the dual-read opens it), write entries, drop it.
        indexer.index_artifact("owner-A", "col-1", "art-1", {"title": "encryption"},
                               self_request("owner-A", "update"))

        assert posting_store.get_posting("owner-A", tok) is None, (
            "the legacy blob survived its own conversion — the slot now has two representations "
            "and the reader prefers the entries, so the blob is unreachable garbage"
        )
        converted = posting_store.get_entries("owner-A", tok)
        assert converted, "the conversion produced no entries"
        for aid, cid, blob in converted:
            with pytest.raises(posting.PostingTampered):
                posting.unpack_entry(blob, pkey)             # ...bound, not merely rewritten
            assert posting.unpack_entry(
                blob, pkey, aad=posting.entry_aad("owner-A", tok, aid, cid),
                allow_unbound=False)
        assert sorted(
            posting.unpack_entry(
                blob, pkey, aad=posting.entry_aad("owner-A", tok, aid, cid),
                allow_unbound=False)["artifact_id"]
            for aid, cid, blob in converted
        ) == sorted(e["artifact_id"] for e in entries), (
            "the conversion changed which artifacts the slot reaches"
        )
