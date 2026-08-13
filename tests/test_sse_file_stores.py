"""Tests for the file-backed SSE PostingStore — the standalone index.

These are the proof that a mantle with no object storage can actually search, and that giving it
a local disk did not turn the encrypted index into a readable one. Every docstring below states
what would go wrong if the property under test did not hold, because a storage test that only
asserts "put then get returns the same bytes" cannot fail in any of the ways this backend can
actually go wrong.

Grouped by the question each group answers:

- ``TestPathSafety`` — can a caller-supplied id escape the index root, or collide with another?
- ``TestPostingRoundTrip`` — is it a faithful PostingStore?
- ``TestCiphertextOnly`` — is anything readable on disk?
- ``TestRealRecall`` — does a narrowing over a file-backed index reach the right artifacts?
- ``TestPersistence`` — does the index survive the store object being thrown away and reopened?
- ``TestUnauthorizedPrincipalGetsNothing`` — the negative control.
"""

from __future__ import annotations

import os

import pytest
from cryptography.fernet import Fernet

from mantle.search.mantle.oracle import (
    FernetMasterKeyStore,
    GrantDenied,
    OracleService,
)
from mantle.search.mantle.sse import (
    FilePostingStore,
    SseIndexer,
    TokenNarrower,
    blind_tokens as bt,
    posting,
    tokenize,
)
from mantle.search.mantle.sse.file_stores import decode_component, encode_component
from .helpers import SelfContextVerifier, grant_request, make_oracle, req, self_request


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path) -> str:
    return str(tmp_path / "sse-index")


@pytest.fixture
def posting_store(root: str) -> FilePostingStore:
    return FilePostingStore(root, prefix="mantle-sse")


@pytest.fixture
def oracle() -> OracleService:
    return make_oracle(FernetMasterKeyStore(Fernet(Fernet.generate_key())))


@pytest.fixture
def indexer(oracle, posting_store) -> SseIndexer:
    return SseIndexer(oracle, posting_store)


@pytest.fixture
def narrower(oracle, posting_store) -> TokenNarrower:
    return TokenNarrower(oracle, posting_store)


def _reach(narrower: TokenNarrower, text: str, contexts, request=None) -> set:
    """The artifact ids the narrowing reaches for ``text`` over ``contexts``."""
    lookup = narrower.lookup_for(text, request if request is not None else req())
    return set() if lookup is None else set(lookup(list(contexts)))


@pytest.fixture
def owner_key(oracle: OracleService) -> bytes:
    return oracle.derive_sse_key("owner-A", self_request("owner-A", "update"))


def _seed_corpus(indexer: SseIndexer, owner: str = "owner-A") -> None:
    """art-1/art-2 in col-1, art-3/art-4 in col-2 — mirrors the in-memory query suite."""
    upd = self_request(owner, "update")
    indexer.index_artifact(owner, "col-1", "art-1", {"title": "encryption library"}, upd)
    indexer.index_artifact(owner, "col-1", "art-2", {"title": "library cards"}, upd)
    indexer.index_artifact(owner, "col-2", "art-3", {"title": "encryption keys"}, upd)
    indexer.index_artifact(
        owner, "col-2", "art-4",
        {"title": "quick brown fox", "content": "lazy dog jumps"}, upd,
    )


def _indexed_token(owner_key: bytes, term: str) -> str:
    """The blind token the indexer actually wrote for ``term`` in the title field.

    The index stores Porter stems, not raw words, so `blind_token(key, field, "encryption")`
    addresses a file that was never written. A negative control built on the raw word would pass
    for the wrong reason — it would be asserting that a typo finds nothing.
    """
    stems = tokenize(term)
    assert stems, f"{term!r} tokenizes to nothing — pick a term the analyzer keeps"
    return bt.blind_token(owner_key, bt.FIELD_TITLE, stems[0])


def _all_files(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        out.extend(os.path.join(dirpath, f) for f in filenames)
    return out


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


class TestPathSafety:
    @pytest.mark.parametrize(
        "raw",
        [
            "", "a", "owner-A", "OWNER-A", "a" * 64, "../../etc/passwd", "a/b/c",
            "..", ".", "nul", "con", "lpt9", "~", "~41", "naïve", "sp ace", "d\\rive",
        ],
    )
    def test_escape_is_reversible(self, raw):
        """The escape must be lossless: two distinct ids that encoded to one filename would share
        one blob — one owner's postings would silently overwrite another's — and
        `list_tokens_for_owner` would return tokens that were never written."""
        encoded = encode_component(raw)
        assert decode_component(encoded) == raw

    @pytest.mark.parametrize("raw", ["", "..", "a/b", "..\\..\\x", "nul", "con", "~"])
    def test_escaped_name_is_a_single_safe_segment(self, raw):
        """An id containing a separator or `..` must not interpolate into a path that writes
        outside the index root — that would be an arbitrary-file-write primitive reachable from
        an artifact id."""
        encoded = encode_component(raw)
        assert encoded, "an escaped component must never be empty"
        assert "/" not in encoded and "\\" not in encoded and os.sep not in encoded
        assert "." not in encoded          # so `..` is unconstructible
        assert encoded.upper() not in {
            "CON", "PRN", "AUX", "NUL", "COM1", "COM9", "LPT1", "LPT9",
        }

    def test_traversing_id_stays_inside_the_root(self, posting_store, root):
        """Proves the path actually written is under the root, not merely that the encoder would
        have been safe if used."""
        posting_store.put_posting("../../escapee", "../../also-escapee", b"x")
        written = _all_files(root)
        assert written, "nothing was written — the assertion below would pass vacuously"
        for path in written:
            assert os.path.commonpath([os.path.abspath(root), path]) == os.path.abspath(root)

    def test_ids_differing_only_in_case_do_not_share_a_blob(self, posting_store):
        """On a case-insensitive filesystem (Windows, default macOS), `Owner-A` and `owner-A`
        must not resolve to one file — that would let one principal read and overwrite the
        other's encrypted postings, a cross-principal leak created by the filesystem rather than
        by any code path that looks wrong."""
        # The token is held identical and only the owner's case varies. Varying the token too
        # would make this pass for the wrong reason: the fan-out shard is sha256 of the escaped
        # name, so two case-different tokens land in different shard directories and cannot
        # collide even when the escaping is case-blind. The owner directory is the real hazard.
        token = "a" * 64
        posting_store.put_posting("owner-A", token, b"upper")
        posting_store.put_posting("owner-a", token, b"lower")
        assert posting_store.get_posting("owner-A", token) == b"upper"
        assert posting_store.get_posting("owner-a", token) == b"lower"


# ---------------------------------------------------------------------------
# PostingStore Protocol
# ---------------------------------------------------------------------------


class TestPostingRoundTrip:
    def test_missing_returns_none(self, posting_store):
        """An absent posting returns None rather than raising: the query engine reads `None` as
        "this term is not in the corpus", and a raise here would fail every query containing an
        unseen term."""
        assert posting_store.get_posting("owner-A", "a" * 64) is None
        assert posting_store.get_manifest("owner-A", "art-1") is None

    def test_put_get_overwrite_delete(self, posting_store):
        """Re-indexing an artifact overwrites its posting blob in place — the write is
        idempotent. If `put` appended, or `delete` left the bytes readable, the index would
        diverge from the corpus with no error anywhere."""
        posting_store.put_posting("owner-A", "a" * 64, b"v1")
        assert posting_store.get_posting("owner-A", "a" * 64) == b"v1"
        posting_store.put_posting("owner-A", "a" * 64, b"v2")
        assert posting_store.get_posting("owner-A", "a" * 64) == b"v2"
        posting_store.delete_posting("owner-A", "a" * 64)
        assert posting_store.get_posting("owner-A", "a" * 64) is None

    def test_delete_missing_is_a_noop(self, posting_store):
        """Deleting an absent key is a no-op rather than a raise. The re-index path deletes
        posting lists that went empty; a raise there would abort the whole re-index over one
        removed token."""
        posting_store.delete_posting("owner-A", "a" * 64)
        posting_store.delete_manifest("owner-A", "art-1")

    def test_owner_isolation(self, posting_store):
        """The owner is part of the address: two principals holding the same blind token (which
        happens only if their SSE keys collide, but the store must not assume that) still get
        separate blobs."""
        posting_store.put_posting("owner-A", "a" * 64, b"A")
        posting_store.put_posting("owner-B", "a" * 64, b"B")
        assert posting_store.get_posting("owner-A", "a" * 64) == b"A"
        assert posting_store.get_posting("owner-B", "a" * 64) == b"B"

    def test_postings_and_manifests_are_independent(self, posting_store):
        """Postings and manifests live in separate namespaces. A manifest and a posting list
        sharing an id would otherwise overwrite each other, and the deletion path (manifest →
        every posting list) would read a posting blob as a manifest."""
        posting_store.put_posting("owner-A", "a" * 64, b"posting")
        posting_store.put_manifest("owner-A", "a" * 64, b"manifest")
        assert posting_store.get_posting("owner-A", "a" * 64) == b"posting"
        assert posting_store.get_manifest("owner-A", "a" * 64) == b"manifest"

    def test_list_tokens_for_owner(self, posting_store):
        """The listing includes every sharded token for the owner and none of another owner's.
        Bulk re-key and migration walk this list; a token it omits is never re-keyed and becomes
        permanently unreadable after rotation."""
        tokens = [f"{i:064x}" for i in range(25)]
        for tok in tokens:
            posting_store.put_posting("owner-A", tok, b"x")
        posting_store.put_posting("owner-B", "f" * 64, b"y")
        assert sorted(posting_store.list_tokens_for_owner("owner-A")) == sorted(tokens)
        assert posting_store.list_tokens_for_owner("owner-unknown") == []

    def test_list_tokens_ignores_non_index_files(self, posting_store, root):
        """An interrupted write's temp file must not be read back as a token. `mkstemp`
        leftovers carry no `.enc` suffix; counting one as a blind token would feed a garbage key
        into the re-key path."""
        posting_store.put_posting("owner-A", "a" * 64, b"x")
        leaf = os.path.dirname(posting_store._posting_path("owner-A", "a" * 64))
        with open(os.path.join(leaf, "tmp1234"), "wb") as fh:
            fh.write(b"leftover")
        assert posting_store.list_tokens_for_owner("owner-A") == ["a" * 64]

    def test_rejects_non_bytes(self, posting_store):
        """Only bytes are accepted, not str. The blob is `nonce ‖ ciphertext ‖ tag`; a str that
        round-tripped through a text encoding would be silently mangled and fail GCM on read,
        indistinguishable from tampering."""
        with pytest.raises(TypeError):
            posting_store.put_posting("owner-A", "a" * 64, "not-bytes")  # type: ignore[arg-type]

    def test_requires_a_root(self, root):
        """An empty root is rejected rather than silently rooting the encrypted index at the
        process's working directory — a location nobody chose and nothing backs up."""
        with pytest.raises(ValueError, match="root"):
            FilePostingStore("", prefix="mantle-sse")

    def test_segments_do_not_share_a_tree(self, root):
        """The per-state index segments (committed / draft / archived) stay in separate trees, so
        a draft's postings never answer a committed-only query."""
        committed = FilePostingStore(root, prefix="mantle-sse")
        draft = FilePostingStore(root, prefix="mantle-sse-draft")
        committed.put_posting("owner-A", "a" * 64, b"committed")
        draft.put_posting("owner-A", "a" * 64, b"draft")
        assert committed.get_posting("owner-A", "a" * 64) == b"committed"
        assert draft.get_posting("owner-A", "a" * 64) == b"draft"


# ---------------------------------------------------------------------------
# Encrypted round-trip + what lands on disk
# ---------------------------------------------------------------------------


class TestCiphertextOnly:
    def test_encrypted_blobs_round_trip_byte_for_byte(self, posting_store, owner_key):
        """The store must not re-encode what it stores. Any transformation of the blob — a
        newline translation, a text-mode open, a trailing byte — breaks GCM authentication on
        read, and the index would report tampering it caused itself."""
        token = bt.blind_token(owner_key, bt.FIELD_TITLE, "encryption")
        pkey = posting.derive_posting_key(owner_key, token)
        entries = [{
            "artifact_id": "art-1", "collection_id": "col-1", "field": "title",
        }]
        blob = posting.pack_posting(entries, pkey)
        posting_store.put_posting("owner-A", token, blob)
        assert posting_store.get_posting("owner-A", token) == blob
        assert posting.unpack_posting(posting_store.get_posting("owner-A", token), pkey) == entries

        mkey = posting.derive_manifest_key(owner_key, "art-1")
        mblob = posting.pack_manifest([token], mkey)
        posting_store.put_manifest("owner-A", "art-1", mblob)
        assert posting.unpack_manifest(
            posting_store.get_manifest("owner-A", "art-1"), mkey) == [token]

    def test_nothing_readable_reaches_the_disk(self, indexer, root):
        """The invariant that makes a file-backed index acceptable at all: if any indexed term,
        field name, artifact id, or collection id appears in the bytes on disk, the local store
        has traded the encrypted index for a readable one, and every other guarantee here holds
        only if this one does."""
        _seed_corpus(indexer)
        files = _all_files(root)
        assert files, "nothing was indexed — the scan below would pass vacuously"
        corpus = b"".join(open(p, "rb").read() for p in files)
        for secret in (
            b"encryption", b"library", b"cards", b"keys", b"quick", b"brown", b"fox",
            b"lazy", b"dog", b"jumps", b"art-1", b"art-4", b"col-1", b"col-2",
            b"title", b"content", b"owner-A",
        ):
            assert secret not in corpus, f"{secret!r} is READABLE in the local index"

    def test_no_readable_side_car_is_written(self, indexer, root):
        """No convenience index — a manifest listing owners, a `.json` of tokens — is written
        beside the ciphertext. Such a file would be exactly the plaintext the blind tokens exist
        to remove, and it would look like harmless bookkeeping."""
        _seed_corpus(indexer)
        files = _all_files(root)
        assert files, "nothing was indexed — the scan below would pass vacuously"
        for path in files:
            assert path.endswith(".enc"), f"non-ciphertext file in the index: {path}"


# ---------------------------------------------------------------------------
# A real query over a file-backed index
# ---------------------------------------------------------------------------


class TestRealRecall:
    def test_a_narrowing_reaches_the_artifacts_carrying_the_term(self, indexer, narrower):
        """A faithful dictionary store is not enough on its own: the reader must get back blobs
        it can open. Otherwise `POST /artifacts/recall` answers 200 with an empty list forever,
        indistinguishable from an empty corpus."""
        _seed_corpus(indexer)
        assert _reach(narrower, "encryption",
                      [("owner-A", "col-1"), ("owner-A", "col-2")]) == {"art-1", "art-3"}

    def test_the_coverage_counts_come_back_off_the_disk_too(self, indexer, narrower):
        """The order a text-only recall comes back in is read off these files. A store that
        returned openable blobs but lost an entry would still narrow — and would silently
        demote the artifact whose stem it dropped."""
        _seed_corpus(indexer)
        lookup = narrower.lookup_for("encryption library", req())
        found = lookup([("owner-A", "col-1"), ("owner-A", "col-2")])
        assert found["art-1"].stems == 2, "art-1 is titled with both stems"
        assert found["art-2"].stems == 1 and found["art-3"].stems == 1

    def test_a_narrowing_respects_the_authorized_collection_set(self, indexer, narrower):
        """The reader intersects the posting list the file store returns with the authorized
        collections. Without that intersection, an artifact from an unauthorized collection
        would enter a result set the caller may not see."""
        _seed_corpus(indexer)
        assert _reach(narrower, "encryption", [("owner-A", "col-1")]) == {"art-1"}

    def test_reindex_then_recall_has_no_stale_hits(self, indexer, narrower):
        """Deletes actually delete. If `delete_posting` left the file behind, a term removed
        from an artifact would keep matching it — the index would only ever grow and would
        answer for text that no longer exists."""
        _seed_corpus(indexer)
        indexer.index_artifact(
            "owner-A", "col-1", "art-1", {"title": "unrelated"},
            self_request("owner-A", "update"),
        )
        assert _reach(narrower, "encryption", [("owner-A", "col-1")]) == set()
        assert _reach(narrower, "unrelated", [("owner-A", "col-1")]) == {"art-1"}


# ---------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_index_survives_a_reopen(self, oracle, root):
        """Pins the durability claim: this is the one thing the in-memory stores cannot do, and
        the reason this backend exists. The store objects are discarded and rebuilt over the
        same directory, exactly as a mantle restart does."""
        writer_postings = FilePostingStore(root, prefix="mantle-sse")
        _seed_corpus(SseIndexer(oracle, writer_postings))
        del writer_postings

        reopened = TokenNarrower(oracle, FilePostingStore(root, prefix="mantle-sse"))
        assert _reach(reopened, "encryption",
                      [("owner-A", "col-1"), ("owner-A", "col-2")]) == {"art-1", "art-3"}

    def test_a_second_store_sees_the_first_store_s_writes(self, root):
        """State lives on disk, not cached in the store object. A store that answered from its
        own memory would pass every single-instance test above and lose the index on restart
        anyway."""
        FilePostingStore(root).put_posting("owner-A", "a" * 64, b"durable")
        assert FilePostingStore(root).get_posting("owner-A", "a" * 64) == b"durable"


# ---------------------------------------------------------------------------
# Negative control — an unauthorized principal gets nothing
# ---------------------------------------------------------------------------


class TestUnauthorizedPrincipalGetsNothing:
    """The four ways a durable local index could leak to a principal with no grant.

    "Nothing" here means nothing derivable, not "filtered results": no key is issued, so no
    posting key exists; the blind token an unauthorized principal can compute addresses a file
    that is not there; and the ciphertext that is there does not open. Each is asserted
    separately, because any one holding alone would be an accident rather than a design.
    """

    @pytest.fixture
    def self_only_oracle(self) -> OracleService:
        # Authorizes a principal for its own contexts and no one else's — the production shape
        # (every principal holds an explicit owner grant on its own collections).
        return OracleService(
            FernetMasterKeyStore(Fernet(Fernet.generate_key())),
            grant_verifier=SelfContextVerifier(),
        )

    @pytest.fixture
    def seeded(self, self_only_oracle, posting_store):
        _seed_corpus(SseIndexer(self_only_oracle, posting_store))
        return posting_store

    def test_sensitivity_control_the_same_corpus_is_readable_when_authorized(self, tmp_path):
        """Not a feature test — the control that stops every assertion below from passing for the
        wrong reason.

        A negative control that proves nothing is worse than none. "Mallory gets nothing" is
        satisfied just as well by a broken fixture, an unseeded corpus, a store that lost the
        writes, or a query engine that returns [] for everyone. This runs the same store and the
        same corpus through a verifier that authorizes the requester, and requires real hits — so
        the emptiness below is attributable to the authorization decision and to nothing else.
        """
        # Same store class and same corpus, its own directory: a fresh oracle derives different
        # keys, and re-seeding on top of another oracle's blobs would fail as tampering rather
        # than as a permission decision — a third reason for emptiness, not a control against it.
        postings = FilePostingStore(str(tmp_path / "authorized"), prefix="mantle-sse")
        authorized = make_oracle(FernetMasterKeyStore(Fernet(Fernet.generate_key())))
        _seed_corpus(SseIndexer(authorized, postings))
        assert _reach(TokenNarrower(authorized, postings), "encryption",
                      [("owner-A", "col-1"), ("owner-A", "col-2")]) == {"art-1", "art-3"}

    def test_no_key_is_issued(self, self_only_oracle, seeded):
        """Key issuance is coupled to the grant check. Everything below depends on this raising —
        if an unauthorized principal could obtain owner-A's SSE key, the index would be readable
        no matter what the storage layer does."""
        with pytest.raises(GrantDenied):
            self_only_oracle.derive_sse_key("owner-A", grant_request("mallory"))

    def test_the_narrowing_yields_nothing_even_with_forged_contexts(
        self, self_only_oracle, seeded,
    ):
        """Authorization is not a post-hoc filter over results. The contexts passed here are
        forged — they name owner-A's real collections — which is precisely the state a
        light-cone bypass would produce. The oracle refuses regardless, so the reader has
        nothing to open; filtered-but-nonempty results here would make the light cone a display
        convention rather than the access decision.

        The refusal is NOT caught here, and `resolve_authorized_scope` is where it lands: its
        catch-all narrows to the empty set, so a refused key costs recall and stays
        indistinguishable from a token matching nothing."""
        narrower = TokenNarrower(self_only_oracle, seeded)
        lookup = narrower.lookup_for("encryption", grant_request("mallory"))
        with pytest.raises(GrantDenied):
            lookup([("owner-A", "col-1"), ("owner-A", "col-2")])

    def test_an_unauthorized_principal_s_own_tokens_address_nothing(
        self, self_only_oracle, seeded, posting_store,
    ):
        """Blind tokens are bound to the owner key. Mallory can index her own corpus and so can
        compute tokens for any term she likes; if the token for "encryption" were
        owner-independent, she could read owner-A's posting file straight off the disk without
        any key at all — a directory listing would become a search."""
        owner_key = self_only_oracle.derive_sse_key(
            "owner-A", self_request("owner-A", "read"),
        )
        mallory_key = self_only_oracle.derive_sse_key(
            "mallory", self_request("mallory", "update"),
        )
        for term in ("encryption", "library", "keys"):
            # Positive control first: the identical construction under owner-A's key does address
            # a stored blob. Without it, the assertions below would hold for a store that had
            # written nothing at all.
            assert posting_store.get_posting("owner-A", _indexed_token(owner_key, term)) is not None
            mallory_token = _indexed_token(mallory_key, term)
            assert posting_store.get_posting("owner-A", mallory_token) is None
            assert posting_store.get_posting("mallory", mallory_token) is None

    def test_ciphertext_does_not_open_under_another_principal_s_key(
        self, self_only_oracle, seeded, posting_store,
    ):
        """The blob is bound to its identity. Reading the file is trivial — it is a file — so the
        bytes must be useless without the owner's key: this hands the raw blob to the unpacker
        under a key derived from mallory's own SSE key and requires it to fail rather than
        return entries."""
        owner_key = self_only_oracle.derive_sse_key(
            "owner-A", self_request("owner-A", "read"),
        )
        mallory_key = self_only_oracle.derive_sse_key(
            "mallory", self_request("mallory", "update"),
        )
        token = _indexed_token(owner_key, "encryption")
        blob = posting_store.get_posting("owner-A", token)
        assert blob is not None, "nothing was indexed — the assertion below would pass vacuously"
        # Positive control: the blob does open under the owner's own posting key, so a failure
        # below is the wrong key being refused, not an unreadable blob. The `aad=` mirrors what
        # `SseIndexer` binds the blob to — the negatives below fail on the KEY and so need none.
        assert posting.unpack_posting(
            blob, posting.derive_posting_key(owner_key, token),
            aad=posting.posting_aad("owner-A", token),
        )

        with pytest.raises(posting.PostingError):
            posting.unpack_posting(blob, posting.derive_posting_key(mallory_key, token))
        # ...nor under the right owner's key applied to the wrong token slot.
        other = _indexed_token(owner_key, "library")
        with pytest.raises(posting.PostingError):
            posting.unpack_posting(blob, posting.derive_posting_key(owner_key, other))
