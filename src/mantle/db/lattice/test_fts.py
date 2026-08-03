"""Tests for the FTS5 lexical arm — Phase 2.1.

Synthetic corpus only. The real corpus is needed for Phase 0.A's recall@k A/B,
not for the mechanics — and the mechanics are where the silent failures live.

The tests that matter most are the ones asserting things that fail SILENTLY:
  * `test_snippet_on_contentless_returns_null` — pins the measured FTS5
    behaviour this module is designed around.
  * `test_snippets_land_in_content_not_highlight` — §A.1 trap 1.
  * `test_rowid_*` — the contentless rowid mapping.
  * `test_bm25_order_is_ascending` — negative scores; a descending sort returns
    the worst matches first and looks fine.
"""
from __future__ import annotations

import sqlite3

import pytest

from mantle.db.lattice.fts import (
    FIELDS,
    PER_DOC_CHARS,
    TOTAL_CHARS,
    FtsDocument,
    FtsIndex,
    build_match_query,
    extract_span,
    make_vertex_resolver,
    to_search_hits,
    tokenize_spans,
)


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


@pytest.fixture()
def idx(conn):
    i = FtsIndex(conn)
    i.ensure_schema()
    return i


DOCS = [
    FtsDocument("v-mesh", "Mesh plane", "S3 segment logs",
                "mesh s3", "The mesh plane replicates encrypted per-node segment logs."),
    FtsDocument("v-ember", "Ember node", "The local leaf node",
                "ember node", "Ember connects chorus and runs the autonomous worker loop."),
    FtsDocument("v-fts", "Lexical retrieval", "FTS5 BM25 over the lattice",
                "fts bm25", "FTS5 ships Okapi BM25 in the SQLite core, replacing blind tokens."),
]


# ═══════════════════════════════════════════════════════════════════════════
# The measured trap this module exists to route around
# ═══════════════════════════════════════════════════════════════════════════

def test_snippet_on_contentless_returns_null(conn):
    """snippet()/highlight() return NULL — NOT an error — on a contentless table.

    This is the whole reason span extraction is done in Python. If a future
    SQLite makes these raise (or work), this test fails and the module can be
    simplified. Until then, reaching for snippet() here produces empty `content`,
    which Lumen drops silently.
    """
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(a, b, content='', "
                 "tokenize='porter unicode61')")
    conn.execute("INSERT INTO t(rowid, a, b) VALUES(1, 'title', 'quick brown fox')")

    assert conn.execute(
        "SELECT snippet(t,1,'[',']','...',8) FROM t WHERE t MATCH 'quick'"
    ).fetchone()[0] is None
    assert conn.execute(
        "SELECT highlight(t,1,'[',']') FROM t WHERE t MATCH 'quick'"
    ).fetchone()[0] is None
    assert conn.execute(
        "SELECT b FROM t WHERE t MATCH 'quick'").fetchone()[0] is None
    # ...while the posting itself is perfectly intact:
    assert conn.execute(
        "SELECT rowid FROM t WHERE t MATCH 'quick'").fetchone()[0] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Rowid mapping — the part that silently breaks
# ═══════════════════════════════════════════════════════════════════════════

def test_rowid_mapping_roundtrips(idx):
    idx.index_many(DOCS)
    for d in DOCS:
        rid = idx.rowid_for(d.vertex_id)
        assert rid is not None
        assert idx.vertex_for(rid) == d.vertex_id


def test_rowid_is_stable_across_reindex(idx):
    rid1 = idx.index(FtsDocument("v-a", "One", "", "", "alpha"))
    rid2 = idx.index(FtsDocument("v-a", "Two", "", "", "beta"))
    assert rid1 == rid2, "re-indexing must reuse the vertex's rowid"


def test_reindex_replaces_rather_than_duplicates(idx, conn):
    """Failure mode 2: two posting sets for one rowid inflate term frequency."""
    idx.index(FtsDocument("v-a", "", "", "", "alpha alpha alpha"))
    idx.index(FtsDocument("v-a", "", "", "", "beta"))
    assert conn.execute(
        "SELECT count(*) FROM fts WHERE fts MATCH 'alpha'").fetchone()[0] == 0
    assert conn.execute(
        "SELECT count(*) FROM fts WHERE fts MATCH 'beta'").fetchone()[0] == 1


def test_rowid_never_reused_after_delete(idx):
    """Failure mode 1: a reissued rowid resolves stale postings to the WRONG
    vertex — wrong answers, not missing ones. AUTOINCREMENT prevents it."""
    rid_a = idx.index(FtsDocument("v-a", "", "", "", "alpha"))
    idx.delete("v-a")
    rid_b = idx.index(FtsDocument("v-b", "", "", "", "beta"))
    assert rid_b != rid_a
    assert idx.vertex_for(rid_b) == "v-b"


def test_delete_removes_postings_and_mapping(idx):
    idx.index_many(DOCS)
    assert idx.delete("v-mesh") is True
    assert idx.rowid_for("v-mesh") is None
    assert [h.vertex_id for h in idx.search("mesh")] == []
    assert idx.delete("v-mesh") is False


def test_search_skips_orphan_postings(idx, conn):
    """A posting whose map row vanished must not crash and must not be named."""
    idx.index(FtsDocument("v-a", "", "", "", "alpha"))
    conn.execute("DELETE FROM fts_map WHERE vertex_id = 'v-a'")
    assert idx.search("alpha") == []


# ═══════════════════════════════════════════════════════════════════════════
# BM25 ordering and column weights
# ═══════════════════════════════════════════════════════════════════════════

def test_bm25_order_is_ascending(idx, conn):
    """FTS5 bm25() is NEGATIVE, most-negative = best. Guards against a
    descending sort, which returns the worst matches while looking plausible."""
    idx.index_many(DOCS)
    rows = conn.execute(
        "SELECT rowid, bm25(fts,1.0,1.0,1.0,1.0) AS s FROM fts "
        "WHERE fts MATCH 'bm25' ORDER BY s ASC").fetchall()
    assert rows and all(r[1] <= 0 for r in rows)

    hits = idx.search("mesh plane segment")
    assert [h.rank for h in hits] == list(range(len(hits)))
    assert hits[0].vertex_id == "v-mesh"
    scores = [h.score for h in hits]
    assert scores == sorted(scores), "hits must be in ascending bm25 order"


def test_no_field_weighting(idx):
    """FAILURE MODE: a per-field boost creeps back in and silently re-ranks.
    bm25() is called with no weight arguments, so a term hit in `title` and the
    same hit in `description` must score identically."""
    idx.index(FtsDocument("v-t", "zebra", "filler words here", "x", "body text"))
    idx.index(FtsDocument("v-d", "filler", "zebra appears here", "x", "body text"))

    hits = idx.search("zebra")
    assert {h.vertex_id for h in hits} == {"v-t", "v-d"}
    assert hits[0].score == pytest.approx(hits[1].score), (
        "title and description must contribute equally — no field weighting"
    )
    assert not hasattr(idx, "weights")


# ═══════════════════════════════════════════════════════════════════════════
# Stemming — DECISION 1, the rebuild
# ═══════════════════════════════════════════════════════════════════════════

def test_porter_stemming_is_active(idx):
    idx.index(FtsDocument("v-r", "", "", "", "the foxes were running quickly"))
    assert [h.vertex_id for h in idx.search("run")] == ["v-r"]
    assert [h.vertex_id for h in idx.search("fox")] == ["v-r"]


def test_fts5_porter_differs_from_prefix_stemming(idx):
    """DECISION 1 evidence: FTS5's porter is not a suffix-truncator, so stems
    are not always prefixes of the source word. Any reimplementation drifts."""
    from mantle.db.lattice.fts import _STEMMER
    stems = _STEMMER.stem_many(["quickly", "happy", "happiness", "universities"])
    assert stems["quickly"] == "quickli"
    assert stems["happy"] == "happi"
    assert stems["happiness"] == "happi"
    assert stems["universities"] == "univers"


def _fts5_tokens(conn, text):
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(c, content='', "
                 "tokenize='porter unicode61')")
    conn.execute("INSERT INTO t(rowid, c) VALUES(1, ?)", (text,))
    conn.execute("CREATE VIRTUAL TABLE tv USING fts5vocab(t,'instance')")
    return conn.execute("SELECT offset, term FROM tv ORDER BY offset").fetchall()


@pytest.mark.parametrize("name,text", [
    ("ascii",      "The quick brown foxes, running: they jumped over 3 lazy dogs!"),
    ("underscore", "open_store snake_case CONST_NAME __dunder__"),
    ("apostrophe", "don't it's John's O'Brien"),
    ("hyphen",     "well-known state-of-the-art co-operate"),
    ("accents",    "café naïve résumé Zürich"),
    ("cjk",        "mesh 网络 plane データ node"),
    ("numbers",    "v1.2.3 3.14159 2026-07-20 0xFF"),
    ("urls",       "https://example.com/a/b?q=1&r=2 user@host.io"),
    ("code",       "def foo(bar): return bar.baz['k']  # comment"),
    ("whitespace", "a\tb\nc\r\nd   e"),
])
def test_tokenizer_aligns_with_fts5_offsets(conn, name, text):
    """Our tokenizer supplies char offsets; FTS5 supplies the indexed terms.
    Exact agreement across every realistic text class in this corpus."""
    from mantle.db.lattice.fts import _STEMMER
    spans = tokenize_spans(text)
    stems = _STEMMER.stem_many([tok for _, _, tok in spans])
    fts = _fts5_tokens(conn, text)
    assert len(fts) == len(spans), f"{name}: {len(fts)} FTS5 vs {len(spans)} ours"
    for offset, term in fts:
        _, _, tok = spans[offset]
        assert stems[tok] == term, f"{name} offset {offset}: {tok!r} != {term!r}"


def test_emoji_divergence_is_real_and_pinned(conn):
    """unicode61 carries a built-in separator table, not a category rule: emoji
    and ½ are TOKEN characters, © is a separator. Pinned so a future SQLite
    change is visible rather than silent."""
    assert _fts5_tokens(conn, "x\U0001F642y") == [(0, "x\U0001F642y")]
    assert len(tokenize_spans("x\U0001F642y")) == 2      # we split; FTS5 does not


def test_emoji_divergence_degrades_gracefully():
    """The divergence CANNOT mis-locate a span, because token offsets never
    cross the boundary — extract_span tokenizes both sides itself. Worst case
    is fallback to head-of-text: never empty, never wrong, never raising."""
    text = "intro text \U0001F642 " + ("body filler " * 300) + " tail marker"
    span = extract_span(text, ["x\U0001F642y"])
    assert span.strip() != ""
    assert len(span) <= PER_DOC_CHARS
    assert text.lstrip().startswith(span.split(" …")[0][:20])


# ═══════════════════════════════════════════════════════════════════════════
# Query sanitization
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("raw", [
    'what is a "grant" (edge)?',
    "NEAR AND OR NOT",
    "col:value ^anchor term*",
    "-minus -- comment",
    "unbalanced (paren",
    'stray " quote',
])
def test_hostile_queries_do_not_raise(idx, raw):
    """Raw user text reaching MATCH is a syntax error, which raises, which
    degrades to zero grounding in a fail-soft caller with no diagnostic."""
    idx.index_many(DOCS)
    idx.search(raw)  # must not raise


def test_match_query_quotes_every_term():
    q = build_match_query('mesh "plane"')
    assert q == '"mesh" OR "plane"'
    assert build_match_query("a b", conjunctive=True) == '"a" AND "b"'
    assert build_match_query("   ") == ""
    assert build_match_query("!!!") == ""


def test_empty_query_returns_empty(idx):
    idx.index_many(DOCS)
    assert idx.search("") == []
    assert idx.search("???") == []


def test_or_semantics_beat_and_for_natural_language(idx):
    """Under AND a single absent word yields zero hits, hence zero grounding."""
    idx.index_many(DOCS)
    assert idx.search("mesh plane nonexistentword") != []
    assert idx.search("mesh plane nonexistentword", conjunctive=True) == []


# ═══════════════════════════════════════════════════════════════════════════
# Span extraction and the Lumen budget
# ═══════════════════════════════════════════════════════════════════════════

def test_span_respects_budget():
    text = ("filler " * 2000) + "the target phrase lives here " + ("tail " * 2000)
    span = extract_span(text, ["target", "phrase"])
    assert len(span) <= PER_DOC_CHARS
    assert "target phrase" in span


def test_span_finds_densest_cluster():
    """Prefers a window covering DISTINCT terms over one repeating a single
    term — three terms once is better evidence than one term nine times."""
    noise = "padding " * 400
    text = ("alpha " * 9) + noise + "alpha beta gamma together here" + noise
    span = extract_span(text, ["alpha", "beta", "gamma"])
    assert "alpha beta gamma" in span
    assert len(span) <= PER_DOC_CHARS


def test_span_never_empty_for_nonempty_text():
    """An empty `content` is dropped by Lumen — so a doc with no query term in
    its body must still return its head, never ''."""
    text = "wholly unrelated prose " * 200
    span = extract_span(text, ["zzzznomatch"])
    assert span.strip() != ""
    assert len(span) <= PER_DOC_CHARS


def test_span_short_text_is_returned_whole():
    assert extract_span("short and sweet", ["short"]) == "short and sweet"
    assert extract_span("", ["x"]) == ""


def test_span_never_exceeds_budget_with_ellipses():
    """Boundary snapping and ellipsis decoration must not push back over cap."""
    for budget in (40, 120, 400, PER_DOC_CHARS):
        text = "lorem ipsum dolor sit amet " * 500
        assert len(extract_span(text, ["dolor"], budget=budget)) <= budget


def test_total_budget_is_enforced(conn):
    idx = FtsIndex(conn)
    idx.ensure_schema()
    big = "mesh segment log replication " * 300  # ~8400 chars each
    docs = [FtsDocument(f"v-{i}", f"Doc {i}", "d", "t", big) for i in range(12)]
    idx.index_many(docs)
    idx.resolver = lambda ids: {d.vertex_id: d for d in docs if d.vertex_id in ids}

    hits = idx.search("mesh segment", limit=12)
    total = sum(len(h.content) for h in hits)
    assert total <= TOTAL_CHARS, f"grounding budget blown: {total}"
    assert all(len(h.content) <= PER_DOC_CHARS for h in hits)


def test_top_k_default_is_six(conn):
    idx = FtsIndex(conn)
    idx.ensure_schema()
    docs = [FtsDocument(f"v-{i}", "mesh", "", "", "mesh plane") for i in range(20)]
    idx.index_many(docs)
    idx.resolver = lambda ids: {d.vertex_id: d for d in docs if d.vertex_id in ids}
    assert len(idx.search("mesh")) <= 6


# ═══════════════════════════════════════════════════════════════════════════
# §A.1 trap 1 — THE test
# ═══════════════════════════════════════════════════════════════════════════

def test_snippets_land_in_content_not_highlight(conn):
    """§A.1 trap 1. Lumen sends "highlight": true and never reads a `highlight`
    field — it reads `content`. Snippets under `highlight` with an empty
    `content` are dropped by `if not content and not title: continue`, and
    grounding goes to ZERO with no error and no log line.
    """
    idx = FtsIndex(conn)
    idx.ensure_schema()
    docs = [FtsDocument("v-1", "Mesh plane", "S3 segment logs", "mesh",
                        "The mesh plane replicates encrypted segment logs " * 40)]
    idx.index_many(docs)
    idx.resolver = lambda ids: {d.vertex_id: d for d in docs if d.vertex_id in ids}

    wire = to_search_hits(idx.search("mesh segment"))
    assert wire, "no hits — the rest of this test would vacuously pass"
    for h in wire:
        assert h["content"].strip(), "empty content => Lumen drops the hit"
        assert "highlight" not in h, "snippets must not hide under `highlight`"
        assert "highlights" not in h


def test_lumen_would_keep_every_hit(conn):
    """Replays Lumen's exact filter (retrieval.py:72-77) over our wire shape."""
    idx = FtsIndex(conn)
    idx.ensure_schema()
    idx.index_many(DOCS)
    idx.resolver = lambda ids: {d.vertex_id: d for d in DOCS if d.vertex_id in ids}

    wire = to_search_hits(idx.search("mesh ember bm25"))
    kept = []
    for h in wire:                                    # verbatim Lumen logic
        content = (h.get("content") or "").strip()
        title = (h.get("title") or h.get("description") or "").strip()
        if not content and not title:
            continue
        kept.append(h)
    assert len(kept) == len(wire) > 0, "Lumen would silently drop hits"


def test_wire_shape_is_flat(conn):
    """§A.1: flat, no nesting; exactly the keys Lumen reads."""
    idx = FtsIndex(conn)
    idx.ensure_schema()
    idx.index_many(DOCS)
    idx.resolver = lambda ids: {d.vertex_id: d for d in DOCS if d.vertex_id in ids}

    for h in to_search_hits(idx.search("mesh")):
        assert set(h) == {"id", "version_id", "title", "description",
                          "content", "score"}
        assert all(not isinstance(v, (dict, list)) for v in h.values())


def test_wire_order_preserves_rank(conn):
    """`score` is parsed but never used — LIST ORDER is the contract."""
    idx = FtsIndex(conn)
    idx.ensure_schema()
    idx.index_many(DOCS)
    idx.resolver = lambda ids: {d.vertex_id: d for d in DOCS if d.vertex_id in ids}

    hits = idx.search("mesh ember bm25")
    wire = to_search_hits(hits)
    assert [w["id"] for w in wire] == [h.vertex_id for h in hits]
    assert [w["score"] for w in wire] == sorted(w["score"] for w in wire)


def test_deprecated_to_lumen_hits_alias_still_resolves():
    """The lumen service is retired; the alias stays until external callers migrate."""
    from mantle.db.lattice.fts import to_lumen_hits
    assert to_lumen_hits is to_search_hits


# ═══════════════════════════════════════════════════════════════════════════
# Resolver
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def vconn(conn):
    conn.execute("CREATE TABLE vertex (id TEXT PRIMARY KEY, content_ref TEXT, doc TEXT)")
    return conn


def test_vertex_resolver_reads_context(vconn):
    import json
    vconn.execute("INSERT INTO vertex VALUES(?,?,?)", (
        "v-1", None, json.dumps({"content": "C",
                                 "context": {"name": "T", "description": "D",
                                             "sources": ["a", "b"]}})))
    got = make_vertex_resolver(vconn)(["v-1"])
    assert got["v-1"].title == "T"
    assert got["v-1"].description == "D"
    assert got["v-1"].tags == "a b"
    assert make_vertex_resolver(vconn)([]) == {}


def test_vertex_resolver_handles_encoded_context(vconn):
    """§A.1 trap 2: context arrives nested, JSON-encoded, or DOUBLY encoded."""
    import json
    ctx = {"name": "T", "description": "D"}
    vconn.execute("INSERT INTO vertex VALUES('v-1',NULL,?)",
                  (json.dumps({"context": ctx, "content": "c"}),))
    vconn.execute("INSERT INTO vertex VALUES('v-2',NULL,?)",
                  (json.dumps({"context": json.dumps(ctx), "content": "c"}),))
    vconn.execute("INSERT INTO vertex VALUES('v-3',NULL,?)",
                  (json.dumps({"context": json.dumps(json.dumps(ctx)), "content": "c"}),))
    got = make_vertex_resolver(vconn)(["v-1", "v-2", "v-3"])
    assert [got[f"v-{i}"].title for i in (1, 2, 3)] == ["T", "T", "T"]


def test_vertex_resolver_titles_markdown_first_line(vconn):
    """Measured: wiki rows carry ZERO context keys; the title is line one."""
    import json
    vconn.execute("INSERT INTO vertex VALUES('v-1',NULL,?)", (
        json.dumps({"content": "Hercule Poirot\n\nA fictional Belgian detective."}),))
    assert make_vertex_resolver(vconn)(["v-1"])["v-1"].title == "Hercule Poirot"


def test_preview_only_is_loud_not_silent(vconn):
    """⚠ THE ONE THAT MATTERS. `doc.content` is a ~300-char CAS preview cut
    mid-word (measured: 98% of 3,000 wiki rows are exactly 300 chars). Indexing
    it truncates the corpus with no error, no metric, no log — so this resolver
    refuses to do it quietly."""
    import json
    vconn.execute("INSERT INTO vertex VALUES('v-1','cas/deadbeef',?)",
                  (json.dumps({"content": "x" * 300}),))

    strict = make_vertex_resolver(vconn, strict=True)
    with pytest.raises(Exception) as e:
        strict(["v-1"])
    assert "content_loader" in str(e.value)

    lenient = make_vertex_resolver(vconn)
    lenient(["v-1"])
    assert lenient.preview_only == 1, "truncation must be counted, not swallowed"

    loaded = make_vertex_resolver(vconn, content_loader=lambda ref: "FULL " * 500)
    assert loaded(["v-1"])["v-1"].content.startswith("FULL")
    assert loaded.preview_only == 0


def test_vertex_resolver_survives_bad_json(vconn):
    vconn.execute("INSERT INTO vertex VALUES('v-1', NULL, 'not json')")
    vconn.execute("INSERT INTO vertex VALUES('v-2', NULL, NULL)")
    got = make_vertex_resolver(vconn)(["v-1", "v-2"])
    assert got["v-1"].content == "" and got["v-2"].content == ""


def test_search_without_resolver_yields_no_content(conn):
    """Documents the degraded mode: no resolver => empty content => Lumen drops
    the hit unless `title` survives. Callers on the Lumen path MUST set one."""
    idx = FtsIndex(conn)
    idx.ensure_schema()
    idx.index_many(DOCS)
    hits = idx.search("mesh")
    assert hits and all(h.content == "" for h in hits)


# ═══════════════════════════════════════════════════════════════════════════
# No prefix tokens
# ═══════════════════════════════════════════════════════════════════════════

def test_no_prefix_tokens_written(conn):
    """px3/px4/px5 were written on every SSE index and never read. Their
    absence is the point: the term vocabulary holds stems only."""
    idx = FtsIndex(conn)
    idx.ensure_schema()
    idx.index(FtsDocument("v-1", "replication", "", "", "segment"))
    conn.execute("CREATE VIRTUAL TABLE v USING fts5vocab(fts,'row')")
    terms = {r[0] for r in conn.execute("SELECT term FROM v")}
    assert terms == {"replic", "segment"}
    assert not any(t.startswith("px") for t in terms)
