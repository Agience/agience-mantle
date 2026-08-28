"""`SqlitePostingStore` — the properties that made it worth replacing the file store.

The file-backed store put every posting list and every manifest in its own file. That one decision
produced four measured problems, and this file is organised around them because "we moved to SQLite"
is not a property and cannot fail:

**Write cost linear in corpus size.** 3.59 MB rewritten per single-artifact write at 400 docs;
1,200 docs did not finish in two minutes. Here a slot is a row.

**An object explosion.** One file per (owner × term × field) plus one per manifest. Here it is
three files, whatever the corpus.

**An accelerator that had to exist.** Every probe was an `open` — 4,520 for a ten-term query
over 194 owners, mostly misses — so the owner-index blob existed to collapse them, and brought a
partial-index-read-as-complete failure that made a whole prior corpus unfindable. A probe here is an
indexed lookup on a primary key, so the blob is gone rather than repaired.

**No atomicity across a posting's read-modify-write.** `_atomic_write` made one blob's
publication atomic and nothing else. Indexing a term is get → decrypt → upsert → encrypt → put, and
only the caller holds the key, so the store can never offer it as one operation — but it can offer a
transaction to run it in, and that one holds across processes.

The protocol conformance half comes first, because a faster store that answers differently is not a
faster store.
"""
from __future__ import annotations

import threading

import pytest

from mantle.search.mantle.sse.indexer import SseIndexer
from mantle.search.mantle.sse.narrowing import TokenNarrower
from mantle.search.mantle.sse.posting import InMemoryPostingStore
from mantle.search.mantle.sse.sqlite_stores import SqlitePostingStore


TOKEN_A = "a" * 64
TOKEN_B = "b" * 64


@pytest.fixture
def store(tmp_path):
    s = SqlitePostingStore(str(tmp_path / "sse" / "committed.db"))
    yield s
    s.close()


class _Oracle:
    """A deterministic per-principal key — the key being PER OWNER is what forces the fan-out."""

    def derive_sse_key(self, principal_id: str, request):        # noqa: ANN001
        return (principal_id.encode("utf-8") * 32)[:32]


# ── conformance: it is a PostingStore ────────────────────────────────────────────────────────


def test_a_posting_round_trips(store):
    assert store.get_posting("owner-A", TOKEN_A) is None
    store.put_posting("owner-A", TOKEN_A, b"sealed-bytes")
    assert store.get_posting("owner-A", TOKEN_A) == b"sealed-bytes"


def test_a_put_overwrites_rather_than_duplicating(store):
    store.put_posting("owner-A", TOKEN_A, b"first")
    store.put_posting("owner-A", TOKEN_A, b"second")
    assert store.get_posting("owner-A", TOKEN_A) == b"second"
    assert store.list_tokens_for_owner("owner-A") == [TOKEN_A], (
        "the primary key must make a re-put an update in place, not a second row"
    )


def test_delete_is_a_no_op_when_absent(store):
    store.delete_posting("owner-A", TOKEN_A)          # must not raise — the Protocol says no-op
    store.put_posting("owner-A", TOKEN_A, b"x")
    store.delete_posting("owner-A", TOKEN_A)
    assert store.get_posting("owner-A", TOKEN_A) is None


def test_owners_are_separated(store):
    """One owner's slot must never answer another's. The blind token is derived from a per-owner
    key so a collision should be impossible upstream — this is the store keeping its half."""
    store.put_posting("owner-A", TOKEN_A, b"A")
    store.put_posting("owner-B", TOKEN_A, b"B")
    assert store.get_posting("owner-A", TOKEN_A) == b"A"
    assert store.get_posting("owner-B", TOKEN_A) == b"B"


def test_manifests_are_a_separate_namespace(store):
    """A manifest is keyed by artifact id and a posting by blind token. The two must not collide
    even when the strings coincide, which they can: nothing stops an artifact id from being 64 hex
    characters."""
    store.put_posting("owner-A", TOKEN_A, b"posting")
    store.put_manifest("owner-A", TOKEN_A, b"manifest")
    assert store.get_posting("owner-A", TOKEN_A) == b"posting"
    assert store.get_manifest("owner-A", TOKEN_A) == b"manifest"


def test_it_refuses_anything_that_is_not_bytes(store):
    """The store moves ciphertext. A `str` here means a caller skipped the sealing step, and
    SQLite would happily store it as TEXT."""
    with pytest.raises(TypeError):
        store.put_posting("owner-A", TOKEN_A, "not bytes")      # type: ignore[arg-type]
    with pytest.raises(TypeError):
        store.put_manifest("owner-A", "art-1", "not bytes")      # type: ignore[arg-type]


def test_an_empty_path_is_refused(tmp_path):
    """Same refusal the file store made about its root: an empty path puts the encrypted index in
    whatever the working directory happens to be, which is not a location anybody chose."""
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="database path is required"):
            SqlitePostingStore(bad)


def test_it_creates_its_parent_directory(tmp_path):
    """A standalone install should not have to `mkdir` before it can search."""
    deep = tmp_path / "a" / "b" / "c" / "sse.db"
    s = SqlitePostingStore(str(deep))
    try:
        s.put_posting("owner-A", TOKEN_A, b"x")
        assert deep.exists()
    finally:
        s.close()


def test_the_index_survives_a_reopen(tmp_path):
    """A store is not a process-lifetime cache."""
    path = str(tmp_path / "sse.db")
    first = SqlitePostingStore(path)
    first.put_posting("owner-A", TOKEN_A, b"durable")
    first.close()
    second = SqlitePostingStore(path)
    try:
        assert second.get_posting("owner-A", TOKEN_A) == b"durable"
    finally:
        second.close()


def test_segments_do_not_see_each_other(tmp_path):
    """`committed` / `draft` / `archived` stay physically separate — one file each, which is what
    the file store's `prefix` did with one directory tree each."""
    committed = SqlitePostingStore(str(tmp_path / "mantle-sse.db"))
    draft = SqlitePostingStore(str(tmp_path / "mantle-sse-draft.db"))
    try:
        committed.put_posting("owner-A", TOKEN_A, b"committed")
        assert draft.get_posting("owner-A", TOKEN_A) is None
    finally:
        committed.close()
        draft.close()


def test_list_owners_covers_manifest_only_owners(store):
    """An owner whose postings were all evicted can still hold manifests, and a rebuild that could
    not see it would leave those manifests referencing nothing forever."""
    store.put_posting("owner-A", TOKEN_A, b"x")
    store.put_manifest("owner-B", "art-1", b"y")
    assert store.list_owners() == ["owner-A", "owner-B"]


# ── the reason it replaced the file store ────────────────────────────────────────────────────


def test_the_whole_index_is_three_files_whatever_the_corpus(tmp_path):
    """The object explosion, gone. The file store wrote one file per (owner × term × field) plus
    one per manifest; a modest corpus was tens of thousands of files, on a layout that needed two
    levels of hash fan-out to stop directories becoming pathological.

    Three files here — the database, its WAL and its shared-memory index — and the count does not
    move with the corpus.
    """
    path = tmp_path / "sse" / "committed.db"
    s = SqlitePostingStore(str(path))
    try:
        indexer = SseIndexer(_Oracle(), s)
        for i in range(40):
            indexer.index_artifact(
                "owner-1", "coll-1", "art-%d" % i,
                {"title": "authorization in the encrypted lattice %d" % i,
                 "description": "grants attenuation light cone merkle proper time"},
                None,
            )
        files = sorted(p.name for p in path.parent.iterdir())
        assert len(files) <= 3, f"expected the database and its sidecars, got {files}"
        assert any(f.endswith(".db") for f in files)
    finally:
        s.close()


def test_indexing_an_artifact_costs_the_same_however_crowded_the_terms_are(tmp_path):
    """The write cost, which is the whole reason for the entry layout.

    The old shape sealed a term's entries together in one blob, so adding an artifact to a term was
    `get_posting` → decrypt every entry → linear scan → re-encrypt every entry → `put_posting`:
    **O(artifacts already carrying that term)**, for a write about one artifact. Measured on 71/home
    before this changed — 14.6s for a POST carrying one name, 16.4s for 4 KB of prose, 3.5s for 4 KB
    of ``'x '`` — "cost is terms, not bytes". A body contributes thousands of distinct stems, which is
    why `pipeline_unified._OFFER_FIELDS` could not afford to include `content`.

    So the property is not "one write per slot", it is **cost independent of crowding**: indexing an
    artifact into terms fifty other artifacts already share must cost what the first one did.

    Measured as bytes moved rather than as calls or seconds. A call count can be satisfied
    vacuously — counting `put_posting`, which the entry layout never calls, compares zero to zero
    and passes over any regression. Bytes cannot: a per-slot rewrite shows up here as a number that
    grows.
    """
    s = SqlitePostingStore(str(tmp_path / "sse.db"))
    try:
        moved = {"bytes": 0, "adds": 0}
        real_add, real_get = s.add_entry, s.get_entries

        def counting_add(principal_id, blind_token, artifact_id, collection_id, blob):
            moved["bytes"] += len(blob)
            moved["adds"] += 1
            return real_add(principal_id, blind_token, artifact_id, collection_id, blob)

        def counting_get(principal_id, blind_token):
            rows = real_get(principal_id, blind_token)
            moved["bytes"] += sum(len(b) for _a, _c, b in rows)
            return rows

        s.add_entry = counting_add                              # type: ignore[method-assign]
        s.get_entries = counting_get                            # type: ignore[method-assign]
        indexer = SseIndexer(_Oracle(), s)

        fields = {"title": "alpha beta gamma", "description": "delta epsilon"}
        # Equal-length ids on purpose: an entry seals its own `artifact_id`, so ids of different
        # lengths differ by a byte per token and would blur the comparison. What must not vary is the
        # crowding; the id length is not the variable under test.
        indexer.index_artifact("owner-1", "coll-1", "art-aaaa", fields, None)
        first = dict(moved)

        # Fifty more artifacts carrying exactly the same terms, so every slot is crowded.
        for i in range(50):
            indexer.index_artifact("owner-1", "coll-1", "crowd-%d" % i, fields, None)

        moved["bytes"] = moved["adds"] = 0
        indexer.index_artifact("owner-1", "coll-1", "art-zzzz", fields, None)
        last = dict(moved)

        assert last["adds"] == first["adds"], (
            f"the crowded write issued {last['adds']} entry writes against the first's "
            f"{first['adds']}"
        )
        assert last["bytes"] == first["bytes"], (
            f"indexing into terms 51 artifacts already share moved {last['bytes']} bytes against "
            f"{first['bytes']} for the first artifact — the write cost is a function of the corpus "
            f"again, which is exactly what stopped `content` from being indexable"
        )
    finally:
        s.close()


def test_a_probe_is_a_lookup_so_no_accelerator_is_needed(tmp_path):
    """THE ACCELERATOR IS GONE, not repaired.

    The owner-index blob existed because a probe was an `open()`. It answered a miss from its own
    token set, which is what made a partial index read as complete lose a whole prior corpus. The
    store must now answer a miss itself, correctly, with nothing consulted first — including for a
    term written by a DIFFERENT artifact than the one just indexed, which is the case the partial
    index got wrong.
    """
    s = SqlitePostingStore(str(tmp_path / "sse.db"))
    try:
        oracle = _Oracle()
        indexer, narrower = SseIndexer(oracle, s), TokenNarrower(oracle, s)
        indexer.index_artifact("owner-1", "coll-1", "art-old", {"title": "pangolin"}, None)
        indexer.index_artifact("owner-1", "coll-1", "art-new", {"title": "quetzal"}, None)

        assert dict(narrower.lookup_for("pangolin", None)([("owner-1", "coll-1")]))
        assert dict(narrower.lookup_for("quetzal", None)([("owner-1", "coll-1")]))
        assert dict(narrower.lookup_for("nothingmatchesthis", None)([("owner-1", "coll-1")])) == {}
        assert not hasattr(s, "get_owner_index"), (
            "the store grew an owner index back; `narrowing` would start preferring it"
        )
    finally:
        s.close()


def test_concurrent_indexers_do_not_lose_each_others_entries(tmp_path):
    """The transaction, which is the capability a file store cannot offer.

    Four threads index four artifacts whose titles share a term, so all four contend on the same
    posting list. Each does get → decrypt → upsert → encrypt → put; unserialized, the last writer's
    list is missing the other three's entries and those artifacts are silently unfindable under the
    shared term.

    `SqlitePostingStore.transaction` is `BEGIN IMMEDIATE`, so the losers wait instead of clobbering
    — and because the exclusion is the database's rather than a mutex in one interpreter, it holds
    for a second process too.
    """
    s = SqlitePostingStore(str(tmp_path / "sse.db"))
    try:
        oracle = _Oracle()
        threads = [
            threading.Thread(
                target=SseIndexer(oracle, s).index_artifact,
                args=("owner-1", "coll-1", "art-%d" % i,
                      {"title": "shared %s" % word}, None),
            )
            for i, word in enumerate(["pangolin", "quetzal", "narwhal", "axolotl"])
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        found = dict(TokenNarrower(oracle, s).lookup_for("shared", None)([("owner-1", "coll-1")]))
        assert len(found) == 4, (
            f"the shared term reached {sorted(found)} — concurrent writers on one posting list "
            f"lost each other's entries"
        )
    finally:
        s.close()


def test_a_failed_write_leaves_no_half_update(tmp_path):
    """The transaction must roll back, not partially commit.

    An artifact's index update writes several posting rows AND its manifest; the manifest is the
    only record of which slots reference the artifact, so a manifest committed without its postings
    names slots that do not carry it, and postings committed without their manifest are unreachable
    by any later deletion. Either half alone is an index that lies about itself.
    """
    s = SqlitePostingStore(str(tmp_path / "sse.db"))
    try:
        with pytest.raises(RuntimeError):
            with s.transaction() as cur:
                cur.execute(
                    "INSERT INTO posting(principal_id, blind_token, blob) VALUES(?,?,?)",
                    ("owner-1", TOKEN_A, b"x"))
                raise RuntimeError("caller exploded mid-update")
        assert s.get_posting("owner-1", TOKEN_A) is None, "the failed transaction committed anyway"
    finally:
        s.close()


def test_a_nested_transaction_joins_the_outer_one(tmp_path):
    """Reentrant, for the reason `LatticeConn.write` is: a store method may call another store
    method without deadlocking itself or — far worse — committing half of a caller's atomic unit.
    `put_posting` opens a transaction of its own, and the indexer calls it from inside one."""
    s = SqlitePostingStore(str(tmp_path / "sse.db"))
    try:
        with pytest.raises(RuntimeError):
            with s.transaction():
                s.put_posting("owner-1", TOKEN_A, b"inner")     # nested — must join, not commit
                raise RuntimeError("outer exploded after the inner write")
        assert s.get_posting("owner-1", TOKEN_A) is None, (
            "the nested put committed independently of the transaction that contained it"
        )
    finally:
        s.close()


# ── it answers identically to the store the tests were written against ───────────────────────


@pytest.mark.parametrize("query", [
    "authorization",
    "authorization lattice",
    "nothingmatchesthis",
    "authorization nothingmatchesthis",
    '"authorization lattice"',            # a phrase — exercises the bigram gate
])
def test_it_answers_exactly_as_the_in_memory_store_does(tmp_path, query):
    """A store is a place bytes go, never what they mean — so two backends over the same corpus
    must return the same artifacts for the same query. This is the assertion that would catch a
    key-derivation or slot-binding difference introduced by the move."""
    oracle = _Oracle()
    corpus = [
        ("owner-1", "coll-1", "art-1",
         {"title": "authorization in the lattice", "description": "grants and encryption"}),
        ("owner-2", "coll-1", "art-2", {"title": "the encryption grant", "tags": "lattice"}),
    ]
    pairs = [("owner-1", "coll-1"), ("owner-2", "coll-1")]

    mem = InMemoryPostingStore()
    sql = SqlitePostingStore(str(tmp_path / "sse.db"))
    try:
        for store_ in (mem, sql):
            ix = SseIndexer(oracle, store_)
            for owner, col, aid, fields in corpus:
                ix.index_artifact(owner, col, aid, fields, None)

        from_mem = dict(TokenNarrower(oracle, mem).lookup_for(query, None)(pairs))
        from_sql = dict(TokenNarrower(oracle, sql).lookup_for(query, None)(pairs))
        assert from_mem == from_sql, (
            f"the two backends disagree on {query!r}: {from_mem} vs {from_sql}"
        )
    finally:
        sql.close()


# ── the migration off the retired file tree ──────────────────────────────────────────────────


def _old_tree_slot(root_prefix, principal_id, kind, name, blob):
    """Write one blob exactly where the retired `FilePostingStore` would have put it."""
    import hashlib
    import os as _os

    from mantle.search.mantle.sse.file_stores import encode_component

    enc = encode_component(name)
    digest = hashlib.sha256(enc.encode("ascii")).hexdigest()
    path = _os.path.join(root_prefix, encode_component(principal_id), "sse", kind,
                         digest[:2], digest[2:4], enc + ".enc")
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)
    return path


def test_the_migration_carries_sealed_blobs_across_unchanged(tmp_path):
    """It copies ciphertext and derives nothing — no oracle, no grant, no acting principal.

    That is what separates it from an owner-index rebuild, which derives each owner's SSE key to
    invert posting lists and so cannot run without an identity that already read the corpus. A
    slot's identity is recoverable from its path and its bytes are
    already sealed, so this pass needs neither.
    """
    from mantle.system.manage_sse_index import migrate

    root_prefix = str(tmp_path / "mantle-sse")
    _old_tree_slot(root_prefix, "owner-A", "posting", TOKEN_A, b"sealed-posting")
    _old_tree_slot(root_prefix, "owner-A", "manifests", "art-1", b"sealed-manifest")
    _old_tree_slot(root_prefix, "owner-B", "posting", TOKEN_B, b"sealed-B")

    store = SqlitePostingStore(str(tmp_path / "mantle-sse.db"))
    try:
        counts = migrate(root_prefix, store)
        assert counts == {"owners": 2, "postings": 2, "manifests": 1, "unreadable": 0}
        assert store.get_posting("owner-A", TOKEN_A) == b"sealed-posting"
        assert store.get_manifest("owner-A", "art-1") == b"sealed-manifest"
        assert store.get_posting("owner-B", TOKEN_B) == b"sealed-B"
    finally:
        store.close()


def test_the_migration_is_idempotent(tmp_path):
    """Re-runnable, because copying a sealed blob to the row it belongs in cannot conflict with
    having already done so — `put_posting` is an upsert."""
    from mantle.system.manage_sse_index import migrate

    root_prefix = str(tmp_path / "mantle-sse")
    _old_tree_slot(root_prefix, "owner-A", "posting", TOKEN_A, b"sealed")
    store = SqlitePostingStore(str(tmp_path / "mantle-sse.db"))
    try:
        migrate(root_prefix, store)
        migrate(root_prefix, store)
        assert store.list_tokens_for_owner("owner-A") == [TOKEN_A]
    finally:
        store.close()


def test_the_migration_skips_what_is_not_a_sealed_blob(tmp_path):
    """`stats.enc` (retired BM25 aggregates) and `index.enc` (the retired owner-index accelerator)
    sit in the same tree and are not slots. An empty file is not one either — copying it would put a
    row in the database that fails GCM authentication on read, which is indistinguishable from
    tampering, where leaving it out just means the slot is absent."""
    import os as _os

    from mantle.system.manage_sse_index import migrate
    from mantle.search.mantle.sse.file_stores import encode_component

    root_prefix = str(tmp_path / "mantle-sse")
    _old_tree_slot(root_prefix, "owner-A", "posting", TOKEN_A, b"sealed")
    _old_tree_slot(root_prefix, "owner-A", "posting", TOKEN_B, b"")          # empty
    owner_sse = _os.path.join(root_prefix, encode_component("owner-A"), "sse")
    for stray in ("stats.enc", "index.enc"):
        with open(_os.path.join(owner_sse, stray), "wb") as fh:
            fh.write(b"retired")

    store = SqlitePostingStore(str(tmp_path / "mantle-sse.db"))
    try:
        counts = migrate(root_prefix, store)
        assert counts["postings"] == 1
        assert counts["unreadable"] == 1, "the empty file should be counted, not copied"
        assert store.list_tokens_for_owner("owner-A") == [TOKEN_A], (
            "a retired side-car was carried across as if it were a posting list"
        )
    finally:
        store.close()


def test_a_migrated_index_actually_answers(tmp_path):
    """The end-to-end claim: index through the OLD layout's blobs, migrate, and search.

    Built by indexing into a SQLite store, exporting its rows into the old tree shape, then
    migrating back — which exercises the path decode against real blind tokens rather than against
    hand-made names, and proves the blobs survive the trip byte-for-byte.

    The tree holds whole-slot blobs, which is what a real migration finds. So this builds them one
    `pack_posting` blob per token, sealing all of a slot's entries, and the recall at the end goes
    through `narrowing`'s legacy fallback, because a blob-copying migration holds no key and cannot
    split one into entries.
    `SseIndexer._absorb_legacy_slot` does that on the next write, and
    `test_a_legacy_whole_slot_blob_converts_on_its_next_write` in `test_sse_indexer.py` is where that
    is pinned.
    """
    from mantle.search.mantle.sse import posting as posting_mod
    from mantle.system.manage_sse_index import migrate

    oracle = _Oracle()
    owner_key = oracle.derive_sse_key("owner-1", None)
    source = SqlitePostingStore(str(tmp_path / "source.db"))
    try:
        SseIndexer(oracle, source).index_artifact(
            "owner-1", "coll-1", "art-1",
            {"title": "authorization in the encrypted lattice"}, None)
        root_prefix = str(tmp_path / "mantle-sse")
        for tok in source.list_tokens_for_owner("owner-1"):
            pkey = posting_mod.derive_posting_key(owner_key, tok)
            entries = [
                posting_mod.unpack_entry(
                    blob, pkey,
                    aad=posting_mod.entry_aad("owner-1", tok, aid, cid))
                for aid, cid, blob in source.get_entries("owner-1", tok)
            ]
            _old_tree_slot(root_prefix, "owner-1", "posting", tok, posting_mod.pack_posting(
                entries, pkey, aad=posting_mod.posting_aad("owner-1", tok)))
        _old_tree_slot(root_prefix, "owner-1", "manifests", "art-1",
                       source.get_manifest("owner-1", "art-1"))
    finally:
        source.close()

    migrated = SqlitePostingStore(str(tmp_path / "migrated.db"))
    try:
        migrate(root_prefix, migrated)
        found = dict(TokenNarrower(oracle, migrated).lookup_for("authorization", None)(
            [("owner-1", "coll-1")]))
        assert found, "the migrated index answers nothing — the blobs did not survive the trip"
    finally:
        migrated.close()


# ── the encrypted-index claim, measured on this backend directly ──────────────────────────────


def test_nothing_readable_reaches_the_database(tmp_path):
    """The corpus text must not be recoverable from the file. Without this, "encrypted index" is a
    label rather than a property.

    `test_the_apache_store_is_a_product.py` asserts the same thing end to end; this is here as well
    because this is the file someone reads when they change the schema, and a store that started
    writing a readable column should fail in its own test rather than in a scenario three files away.

    Identifiers are a separate question, and they are present. `principal_id` and `artifact_id`
    arrive as cleartext from the caller and are the row keys, so they are at rest here — as they are
    in S3's object keys. Blinding them needs a key no store holds by design, so it belongs at the
    caller across every backend. Asserted in the positive, so this notices the day it changes.
    """
    path = tmp_path / "sse.db"
    s = SqlitePostingStore(str(path))
    try:
        SseIndexer(_Oracle(), s).index_artifact(
            "owner-1", "coll-1", "art-1",
            {"title": "authorization in the encrypted lattice",
             "description": "merkle attenuation aperture"}, None)
    finally:
        s.close()

    blob = b"".join(p.read_bytes() for p in sorted(path.parent.iterdir()))
    for term in (b"authorization", b"encrypted", b"lattice", b"merkle", b"attenuation",
                 b"aperture", b"title", b"description"):
        assert term not in blob, f"corpus text {term!r} is readable in the index at rest"
    assert b"owner-1" in blob and b"art-1" in blob, (
        "the identifiers are expected at rest — see the docstring; if they are now blinded that is "
        "an improvement and this assertion is what noticed"
    )


def test_a_second_store_over_the_same_file_sees_the_first_s_writes(tmp_path):
    """Two store objects over one database are one index — the in-process proxy for two processes.

    The file store passed this by accident: every read went to the filesystem, so there was nothing
    to be stale. Here there is a connection per thread and a WAL, so it is a real question — a store
    that cached rows, or opened its connection before another process's commit and never saw it,
    would answer from a snapshot.
    """
    path = str(tmp_path / "sse.db")
    writer = SqlitePostingStore(path)
    reader = SqlitePostingStore(path)
    try:
        assert reader.get_posting("owner-A", TOKEN_A) is None
        writer.put_posting("owner-A", TOKEN_A, b"written-by-the-other-handle")
        assert reader.get_posting("owner-A", TOKEN_A) == b"written-by-the-other-handle"
        writer.delete_posting("owner-A", TOKEN_A)
        assert reader.get_posting("owner-A", TOKEN_A) is None, "a delete must be visible too"
    finally:
        writer.close()
        reader.close()
