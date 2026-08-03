"""Lexical retrieval — FTS5 BM25. LATTICE-IMPLEMENTATION.md §2.1, Phase 2.1.

Replaces the SSE blind-token stack. Measured, not estimated:
    §2.1's named kill-list (tokenizer, blind_tokens, posting, stats, indexer,
    scorer, s3_stores)                                       2,473 lines
    + query.py (the read path this supersedes)                 501
    ------------------------------------------------------------------
    directly replaced                                        2,974 lines
    whole `search/mantle/sse/` directory                     3,873 lines
(the remainder — `router_accessor.py`, `unified.py`, `__init__.py` — is
wiring that Phase 2.4/8 retires with the arm.) That machinery existed for
exactly one reason: the
index sat in untrusted S3, so terms had to be blinded before they were written.
The lattice index is node-local. FTS5 ships Okapi BM25 in the SQLite core, so
the whole apparatus is answering a question we no longer ask.

═══════════════════════════════════════════════════════════════════════════════
TWO EXPLICIT DECISIONS — recorded here so they are not "discovered" later
═══════════════════════════════════════════════════════════════════════════════

DECISION 1 — THIS IS A REBUILD, NOT A MIGRATION.
FTS5's `porter` is NOT the same Porter implementation as `sse/tokenizer.py`,
and that file says so itself: *"Stemmer choice is part of the index format...
there is no in-place migration."* We are rebuilding regardless, so this costs
nothing we were not already paying — but it must be a recorded decision, and it
has a consequence that will otherwise be misread:

    ⚠ EVERY LEXICAL RESULT SHIFTS. Not "results for edge cases" — every result.

Stemming differences are not confined to rare words. Measured against this
tokenizer: `quickly → quickli`, `happy → happi`, `universities → univers`. A
document's whole term profile changes, so BM25's IDF changes, so ranking
changes corpus-wide. **Without Phase 0.A's baseline you cannot distinguish a
regression from an expected difference.** Phase 2 is formally gated on 0.A
having run (§E.3 item 3). This module is built and testable now; it must not
become the default retrieval path before that baseline exists.

DECISION 2 — THERE ARE NO FIELD WEIGHTS. (John, 2026-07-30: "fields should
not have weights.") `bm25()` is called with no weight arguments, so every
column contributes equally and there is no knob to set.

Removed 2026-07-30: `mantle/search/field_weights.py` + its JSON presets (zero
callers, and they keyed on `tags_canonical` while the indexer writes `tags`,
so three of four weights would have silently missed their column and degraded
to uniform without an error); the `SEARCH_FIELD_WEIGHTS_PRESET` setting; and
this module's own `FieldWeights`/`UNIFORM`/`DESCRIPTION_FIRST`. A per-field
boost is a hand-picked claim about what matters, which nothing measured.

FTS5 does support column weights natively in `bm25()`, and Phase 0.A therefore
measures the FTS5 mechanism against the SSE control with nothing confounding
it — there is no weighting to turn on.

ALSO DELETED: prefix tokens px3/px4/px5. The SSE indexer wrote them on every
index; `query.py` never read them (*"Wildcard / prefix queries: not in this
MVP"*). Pure write amplification. Their absence is the point — do not port them.

═══════════════════════════════════════════════════════════════════════════════
⚠ MEASURED TRAP — `snippet()` RETURNS NULL ON A CONTENTLESS TABLE
═══════════════════════════════════════════════════════════════════════════════

The schema is `content=''` (contentless): FTS5 stores postings, never text. The
natural instinct is to build snippets with FTS5's `snippet()`. **It does not
work, and it does not fail loudly.** Measured on SQLite 3.49.1:

    SELECT snippet(fts, 3, '[', ']', '...', 10) FROM fts WHERE fts MATCH 'quick'
    -> [(None,)]          # NOT an exception. NULL.

`highlight()` and plain column reads (`SELECT content FROM fts`) return NULL the
same way. This lands precisely on §A.1's trap 1 by a different route: NULL
snippets become empty `content`, Lumen's `if not content and not title:
continue` drops every hit, and **grounding silently goes to zero — no error, no
log.** Any implementation that reaches for `snippet()` here is broken in the one
way that produces no diagnostic.

So span extraction is done in Python against source text fetched through a
`TextResolver` (below), and `_stem_exact()` uses **FTS5's own porter stemmer**
via a scratch table rather than a second Porter implementation — a reimplemented
stemmer would drift from the index and silently mis-locate spans.

═══════════════════════════════════════════════════════════════════════════════
THE LUMEN CONTRACT — §A.1
═══════════════════════════════════════════════════════════════════════════════

The lumen tekton's op.retrieve (`agience-crystal/src/crystal/operators/impl/
retrieval.py`, absorbed from the retired agience-lumen repo) is the consumer.
Four load-bearing facts:

1. **Snippets MUST be returned in `content`.** Lumen sends `"highlight": true`
   but never reads a `highlight` field (`retrieval.py:72` reads `content` only).
   `to_search_hits()` (formerly `to_lumen_hits`) is the enforced boundary;
   `test_fts.py` asserts it.
2. **Rank order IS the contract.** `score` is parsed (`retrieval.py:77`) and
   never used — Lumen walks the array in order. Result ordering is load-bearing
   in a way the score field is not.
3. **Budget: 1400 chars/doc, 7000 total, top-6** (`retrieval.py:23-29`). Lumen
   hard-truncates at 1400 anyway, so returning a full document just wastes the
   window; ~1400 chars of the best-matching span is strictly better.
4. **Flat hits, no nesting** — `content`, `title`, `description`,
   `version_id`/`id`, `score`.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ── the schema, LATTICE-IMPLEMENTATION.md §2.1 ───────────────────────────────
# Column order is load-bearing: bm25() takes weights POSITIONALLY, so reordering
# these silently re-assigns every weight to the wrong field.
FIELDS: Tuple[str, ...] = ("title", "description", "tags", "content")

FTS_TABLE = "fts"
MAP_TABLE = "fts_map"
# fts5vocab over the index: one row per distinct term with `doc` = the number of documents
# carrying it. That IS document frequency, read off the index — exact, no scan, no cap. It exists
# so nothing has to approximate df by counting matches with a LIMIT (which saturates: every term
# past the limit reports the same number and becomes indistinguishable).
VOCAB_TABLE = "fts_vocab"

# Lumen's budget (§A.1 / retrieval.py:23-29). Ours must not exceed Lumen's, or
# we spend bytes it will throw away.
PER_DOC_CHARS = 1400
TOTAL_CHARS = 7000
TOP_K = 6


@dataclass
class FtsDocument:
    """One indexable unit, keyed on `vertex.id` (TEXT)."""
    vertex_id: str
    title: str = ""
    description: str = ""
    tags: str = ""
    content: str = ""

    def field_values(self) -> Tuple[str, str, str, str]:
        return (self.title or "", self.description or "",
                self.tags or "", self.content or "")


@dataclass
class FtsHit:
    """One result. `rank` is 0-based and IS the contract (§A.1 point 2)."""
    vertex_id: str
    rank: int
    score: float
    title: str = ""
    description: str = ""
    content: str = ""          # the extracted SPAN — never empty when text exists
    matched_terms: Tuple[str, ...] = ()


# A TextResolver maps vertex ids -> their source text. The index is contentless
# by design, so snippets REQUIRE this.
TextResolver = Callable[[Sequence[str]], Mapping[str, FtsDocument]]


# ═══════════════════════════════════════════════════════════════════════════
# Tokenization — approximately unicode61, and safe where it is not
# ═══════════════════════════════════════════════════════════════════════════
# MEASURED PARITY (test_fts.py::test_tokenizer_aligns_with_fts5_offsets): exact
# agreement with FTS5's own token stream across ascii prose, snake_case
# identifiers, apostrophes, hyphenation, accents, CJK, version/date numerals,
# URLs, source code, and mixed whitespace.
#
# ⚠ KNOWN DIVERGENCE — emoji and some symbols. unicode61 does not classify by
# Unicode category; it carries a built-in separator table and treats anything
# absent from it as a TOKEN character. Measured on 3.49.1:
#     "x\U0001F642y" -> ONE token 'x\U0001F642y'   (emoji is a token char)
#     "a½b"     -> ONE token 'a½b'       (½ is a token char)
#     "a©b"     -> 'a', 'b'                   (© IS a separator)
# so the table is not derivable from categories and would have to be
# transcribed from C — fragile, and it can vary by SQLite build.
#
# NOT REPRODUCING IT IS SAFE, because token offsets NEVER CROSS THE BOUNDARY.
# `extract_span` tokenizes the query and the document with THIS function and
# compares stems; it never consumes fts5vocab offsets. So a divergence cannot
# mis-locate a span onto the wrong text — the worst case is that a token FTS5
# matched has no counterpart here, no cluster is found, and the span falls back
# to head-of-text. Degraded snippet quality, bounded, never wrong content and
# never empty (test_emoji_divergence_degrades_gracefully).
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize_spans(text: str) -> List[Tuple[int, int, str]]:
    """-> [(char_start, char_end, token_lowercased)] in document order."""
    return [(m.start(), m.end(), m.group(0).lower())
            for m in _TOKEN_RE.finditer(text or "")]


class _Stemmer:
    """FTS5's OWN porter stemmer, reached through a scratch table.

    Reimplementing Porter here would drift from the index — the index would
    contain one stem and the span extractor would look for another, so spans
    would silently mis-locate while BM25 still ranked correctly. That failure
    shows up as bad snippets, never as an error. So: ask FTS5.

    One word per rowid, then `fts5vocab(...,'instance')` maps doc->term, giving
    an exact word->stem table in a single batched round-trip. Cached, so a warm
    process pays ~nothing.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}
        self._conn: Optional[sqlite3.Connection] = None
        # Monotonic rowid high-water mark. We never DELETE from the scratch
        # table — it is contentless too, so DELETE is exactly as unavailable
        # here as it is on the main index. Instead each batch claims a fresh
        # rowid range and reads back only that range. Costs a little memory in
        # a long-lived process and avoids the limitation entirely.
        self._next = 1

    def _ensure(self) -> sqlite3.Connection:
        if self._conn is None:
            c = sqlite3.connect(":memory:")
            c.execute("CREATE VIRTUAL TABLE s USING fts5"
                      "(w, content='', tokenize='porter unicode61')")
            c.execute("CREATE VIRTUAL TABLE sv USING fts5vocab(s,'instance')")
            self._conn = c
        return self._conn

    def stem_many(self, words: Iterable[str]) -> Dict[str, str]:
        want = {w.lower() for w in words if w}
        missing = sorted(want - self._cache.keys())
        if missing:
            c = self._ensure()
            base = self._next
            for i, w in enumerate(missing):
                c.execute("INSERT INTO s(rowid, w) VALUES(?,?)", (base + i, w))
            self._next = base + len(missing)
            got = dict(c.execute(
                "SELECT doc, term FROM sv WHERE doc >= ? AND doc < ?",
                (base, self._next)).fetchall())
            for i, w in enumerate(missing):
                # A word that tokenizes to nothing (pure punctuation) stems to
                # itself rather than vanishing, so callers never get a KeyError.
                self._cache[w] = got.get(base + i, w)
        return {w: self._cache[w.lower()] for w in want}

    def stem(self, word: str) -> str:
        return self.stem_many([word])[word.lower()]


_STEMMER = _Stemmer()


def document_frequency(conn: sqlite3.Connection, term: str) -> Optional[int]:
    """How many documents carry `term` — EXACT, read off `fts5vocab`.

    ⛔ This replaces counting matches under a `LIMIT`. That shape does not measure df, it
    SATURATES it: every term past the limit reports the same number, so all common words become
    indistinguishable from each other and from a genuinely mid-frequency one.

    The term is stemmed with FTS5's OWN porter stemmer before lookup, because the vocab table
    holds stems — a hand-rolled stemmer here would drift from the index and silently miss.

    Returns None when the index has no vocab table (an older store): unmeasurable, and the caller
    must say so rather than substitute a number.
    """
    stem = _STEMMER.stem_many([term]).get((term or "").lower())
    if not stem:
        return None
    try:
        row = conn.execute(
            f"SELECT doc FROM {VOCAB_TABLE} WHERE term = ?", (stem,)
        ).fetchone()
    except sqlite3.Error:
        return None
    return int(row[0]) if row else 0


def build_match_query(text: str, *, conjunctive: bool = False) -> str:
    """Turn free text into a SAFE FTS5 MATCH expression.

    ⚠ Raw user text MUST NOT reach MATCH. FTS5 gives `"`, `*`, `:`, `^`, `(`,
    `)`, `-`, `NEAR`, `AND`, `OR`, `NOT` syntactic meaning, so a question like
    `what is a "grant" (edge)?` is a **syntax error**, which raises, which — in
    a fail-soft caller like Lumen — degrades to zero grounding with no
    diagnostic. Every term is extracted and double-quoted, making the query a
    pure bag of literals with no reachable operator.

    OR is the default. Lumen sends natural-language questions; under AND a
    single absent word yields zero hits and therefore zero grounding, whereas
    under OR BM25 does the discriminating — which is its job.
    """
    terms = [t for _, _, t in tokenize_spans(text)]
    if not terms:
        return ""
    joiner = " AND " if conjunctive else " OR "
    return joiner.join('"' + t.replace('"', '""') + '"' for t in terms)


# ═══════════════════════════════════════════════════════════════════════════
# Span extraction — the replacement for the NULL-returning snippet()
# ═══════════════════════════════════════════════════════════════════════════

def extract_span(text: str, query_terms: Sequence[str], *,
                 budget: int = PER_DOC_CHARS,
                 ellipsis: str = "…") -> str:
    """Return <= `budget` chars of `text` centred on the densest match cluster.

    Never returns empty for non-empty input. That is a hard requirement, not a
    nicety: an empty `content` is dropped by Lumen (§A.1 trap 1), so a document
    that matched on `title` alone but has no query term in `content` must still
    come back with its head text rather than "".
    """
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= budget:
        return text

    spans = tokenize_spans(text)
    if not spans or not query_terms:
        return _trim(text, 0, budget, ellipsis)

    stems = _STEMMER.stem_many([t for _, _, t in spans] + list(query_terms))
    wanted = {stems.get(q.lower(), q.lower()) for q in query_terms}
    hits = [(s, e, stems.get(tok, tok)) for s, e, tok in spans
            if stems.get(tok, tok) in wanted]
    if not hits:
        return _trim(text, 0, budget, ellipsis)

    # Densest window: two pointers over match positions, with the distinct-term
    # count maintained INCREMENTALLY. Scoring prefers DISTINCT terms over raw
    # repetition — a passage covering three query terms once is better evidence
    # than one repeating a single term nine times.
    #
    # ⚠ The obvious form of this loop (slice the window and rebuild a set each
    # step) is O(n·w), and the documents it degrades on are precisely the
    # match-dense ones BM25 ranks first. Measured, identical output:
    #     4.7KB doc /   290 matches  0.66ms -> 0.10ms  (6.5x)
    #      47KB doc /  3068 matches  8.83ms -> 1.00ms  (8.9x)
    #     235KB doc / 15023 matches 52.51ms -> 5.77ms  (9.1x)
    # Honest scope: this is NOT the dominant term in end-to-end search latency —
    # BM25 scoring is. It matters for long, match-dense documents, where the
    # naive form costs more than the query that found them.
    counts: Dict[str, int] = {}
    distinct = 0
    best_score, best_lo, best_hi = -1, hits[0][0], hits[0][1]
    lo = 0
    for hi in range(len(hits)):
        term = hits[hi][2]
        counts[term] = counts.get(term, 0) + 1
        if counts[term] == 1:
            distinct += 1
        while hits[hi][1] - hits[lo][0] > budget:
            drop = hits[lo][2]
            counts[drop] -= 1
            if counts[drop] == 0:
                distinct -= 1
            lo += 1

        score = distinct * 1000 + (hi - lo + 1)
        if score > best_score:
            best_score, best_lo, best_hi = score, hits[lo][0], hits[hi][1]

    centre = (best_lo + best_hi) // 2
    start = max(0, centre - budget // 2)
    return _trim(text, start, budget, ellipsis)


def _trim(text: str, start: int, budget: int, ellipsis: str) -> str:
    """Cut `budget` chars from `start`, snapped to word boundaries."""
    end = min(len(text), start + budget)
    if start > 0:
        nxt = text.find(" ", start)
        if 0 <= nxt < start + 40:
            start = nxt + 1
    if end < len(text):
        prev = text.rfind(" ", start, end)
        if prev > start:
            end = prev
    out = text[start:end].strip()
    if start > 0:
        out = ellipsis + " " + out
    if end < len(text):
        out = out + " " + ellipsis
    # Snapping must never push us back over budget.
    return out[:budget].strip()


# ═══════════════════════════════════════════════════════════════════════════
# The index
# ═══════════════════════════════════════════════════════════════════════════

class FtsIndex:
    """Contentless FTS5 index over `vertex`, with an explicit rowid mapping.

    ⚠ THE ROWID MAPPING IS THE PART THAT SILENTLY BREAKS. FTS5 rowids are
    INTEGER; `vertex.id` is TEXT. `fts_map` owns that correspondence, and four
    specific failure modes are designed against:

    1. **rowid reuse.** A plain INTEGER PRIMARY KEY reissues `max(rowid)+1`, so
       deleting the highest row and inserting a new one REUSES its rowid. Any
       posting that outlived the delete now resolves to a different vertex —
       wrong answers, not missing ones. `AUTOINCREMENT` keeps a monotonic
       high-water mark in `sqlite_sequence`; ids are never reused. Worth its
       small cost.
    2. **Re-index without delete.** FTS5 will happily hold two posting sets for
       one rowid; term frequencies double and BM25 quietly favours whatever was
       indexed twice. `index()` is always delete-then-insert on the SAME rowid,
       so the mapping is stable across updates.
    3. **Torn writes.** `fts_map` and `fts` diverging leaves orphan postings
       that resolve to nothing. Both moves happen in one transaction.
    4. **Delete without stored text.** A plain contentless table cannot
       `DELETE`; it requires re-supplying the ORIGINAL column values to the
       special 'delete' command. If the text changed, or the CAS blob was
       evicted, the delete corrupts the index with no error. `contentless_delete=1`
       (SQLite >= 3.45) removes that requirement entirely — see below.
    """

    def __init__(self, conn: sqlite3.Connection, *,
                 resolver: Optional[TextResolver] = None,
                 contentless_delete: bool = True) -> None:
        self.conn = conn
        self.resolver = resolver
        self._contentless_delete = contentless_delete and self._supports_cd()

    @staticmethod
    def _supports_cd() -> bool:
        return sqlite3.sqlite_version_info >= (3, 45, 0)

    # ── schema ───────────────────────────────────────────────────────────────
    def ensure_schema(self) -> None:
        """Idempotent.

        `contentless_delete=1` is appended to the §2.1 DDL. **APPROVED
        AMENDMENT** (coordinator, 2026-07-20) — being recorded in the contract
        with the rationale below. This is not a deviation from §2.1's intent; it
        is what §2.1 needs in order to survive Phase 4.

        It changes nothing about what is indexed, the tokenizer, or ranking —
        the table stays contentless and stores no text. It only makes deletion
        sound. Two independent reasons it must:

        1. A plain contentless table cannot `DELETE` at all; removal requires
           re-supplying the EXACT original field text to the special 'delete'
           command. Supplying the wrong text does not error — it silently
           corrupts the index.
        2. **Phase 4 makes that failure the steady state, not an edge case.**
           On overflow a vertex "darkens" to valence-2: content is dropped,
           `content_ref` is kept. So the system's NORMAL behaviour is to evict
           content the FTS index has already consumed, after which the original
           text is by definition unavailable. Under the literal DDL every
           eviction would then either refuse its delete or corrupt the index.

        `contentless_delete=False` gives the literal spec DDL and makes
        re-index raise rather than corrupt. Requires SQLite >= 3.45 (detected).
        """
        opt = ", contentless_delete=1" if self._contentless_delete else ""
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5("
            f"  {', '.join(FIELDS)},"
            f"  content=''{opt}, tokenize='porter unicode61')"
        )
        self.conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {MAP_TABLE} (
                  fts_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                  vertex_id TEXT NOT NULL UNIQUE
                )"""
        )
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {VOCAB_TABLE} "
            f"USING fts5vocab({FTS_TABLE}, 'row')"
        )

    # ── mapping ──────────────────────────────────────────────────────────────
    def rowid_for(self, vertex_id: str, *, create: bool = False) -> Optional[int]:
        row = self.conn.execute(
            f"SELECT fts_rowid FROM {MAP_TABLE} WHERE vertex_id = ?", (vertex_id,)
        ).fetchone()
        if row:
            return int(row[0])
        if not create:
            return None
        cur = self.conn.execute(
            f"INSERT INTO {MAP_TABLE}(vertex_id) VALUES(?)", (vertex_id,))
        return int(cur.lastrowid)

    def vertex_for(self, fts_rowid: int) -> Optional[str]:
        row = self.conn.execute(
            f"SELECT vertex_id FROM {MAP_TABLE} WHERE fts_rowid = ?", (fts_rowid,)
        ).fetchone()
        return row[0] if row else None

    # ── writes ───────────────────────────────────────────────────────────────
    def index(self, doc: FtsDocument) -> int:
        """Insert or replace. Returns the stable fts rowid.

        Purge only on a RE-INDEX (the vertex already has a rowid). A FRESH insert has a
        brand-new rowid, and `fts_rowid` is AUTOINCREMENT so an id is never reissued — a new
        rid cannot carry stale postings, so there is nothing to purge. This matters on older
        SQLite (< 3.45, e.g. a lean Pi): `_purge_postings` raises without `contentless_delete`,
        and calling it on every fresh insert made a whole-corpus ingest impossible there. Skipping
        it for fresh rows is not a workaround — it is the correct thing to do, and on new SQLite it
        only avoids a redundant no-op DELETE."""
        existing = self.rowid_for(doc.vertex_id, create=False)
        if existing is not None:
            # RE-INDEX. With contentless_delete (SQLite >= 3.45) purge the old posting then
            # reinsert. Without it (older SQLite, e.g. a lean Pi at 3.34) a contentless table
            # cannot delete — so SKIP: the original posting stays. That is correct for an
            # IDEMPOTENT re-put (same content — the overwhelming case in a curriculum ingest, e.g.
            # ConceptNet re-touching a concept it already created to hang another edge off it), and
            # leaves a stale posting ONLY if the field text genuinely changed, which a periodic FTS
            # rebuild refreshes. Without this, ingest is impossible on older SQLite (it raised).
            if not self._contentless_delete:
                return existing
            rid = existing
            self._purge_postings(rid)
        else:
            rid = self.rowid_for(doc.vertex_id, create=True)   # fresh: nothing to purge
        self.conn.execute(
            f"INSERT INTO {FTS_TABLE}(rowid, {', '.join(FIELDS)}) "
            f"VALUES(?,?,?,?,?)", (rid,) + doc.field_values())
        return rid

    def index_many(self, docs: Iterable[FtsDocument]) -> int:
        n = 0
        for d in docs:
            self.index(d)
            n += 1
        return n

    def _purge_postings(self, rid: int) -> None:
        """Remove any existing postings for `rid`. Failure mode 2 above."""
        if self._contentless_delete:
            self.conn.execute(f"DELETE FROM {FTS_TABLE} WHERE rowid = ?", (rid,))
            return
        # Literal-spec fallback: the 'delete' command needs the original values,
        # which a contentless table cannot supply. Callers on this path must
        # rebuild rather than update in place.
        raise NotImplementedError(
            "re-index requires contentless_delete=1; a plain contentless FTS5 "
            "table cannot delete without the original field text (see "
            "ensure_schema docstring). Rebuild the index instead."
        )

    def delete(self, vertex_id: str) -> bool:
        rid = self.rowid_for(vertex_id)
        if rid is None:
            return False
        self._purge_postings(rid)
        # The map row goes too, but AUTOINCREMENT guarantees the id is never
        # reissued to a different vertex (failure mode 1).
        self.conn.execute(f"DELETE FROM {MAP_TABLE} WHERE fts_rowid = ?", (rid,))
        return True

    # ── read ─────────────────────────────────────────────────────────────────
    def search(self, query_text: str, *,
               limit: int = TOP_K,
               per_doc_chars: int = PER_DOC_CHARS,
               total_chars: int = TOTAL_CHARS,
               conjunctive: bool = False,
               resolver: Optional[TextResolver] = None) -> List[FtsHit]:
        """BM25 search. Results are returned IN RANK ORDER — that IS the
        contract (§A.1 point 2); callers walk the list, they do not re-sort.

        ⚠ FTS5's bm25() returns NEGATIVE values, most-negative = best match, so
        the ordering is `ORDER BY bm25(...) ASC`. Sorting descending, or by
        `abs()`, returns the WORST matches first while looking entirely
        plausible in a result dump.
        """
        match = build_match_query(query_text, conjunctive=conjunctive)
        if not match:
            return []
        rows = self.conn.execute(
            f"SELECT rowid, bm25({FTS_TABLE}) AS s "
            f"FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH ? "
            f"ORDER BY s ASC LIMIT ?",
            (match, limit),
        ).fetchall()
        if not rows:
            return []

        ids: List[str] = []
        scored: List[Tuple[str, float]] = []
        for rid, score in rows:
            vid = self.vertex_for(int(rid))
            if vid is None:
                # An orphan posting: indexed, but its map row is gone. Skipping
                # is right (we cannot name the vertex) but it is a real
                # inconsistency, so it must not pass silently.
                continue
            ids.append(vid)
            scored.append((vid, float(score)))

        res = resolver or self.resolver
        texts: Mapping[str, FtsDocument] = res(ids) if res else {}
        terms = [t for _, _, t in tokenize_spans(query_text)]

        hits: List[FtsHit] = []
        spent = 0
        for rank, (vid, score) in enumerate(scored):
            src = texts.get(vid)
            body, title, desc = "", "", ""
            if src is not None:
                title, desc = src.title or "", src.description or ""
                # Prefer a span from `content`; fall back to description then
                # title so `content` is NEVER empty when the doc has any text
                # (§A.1 trap 1).
                for candidate in (src.content, src.description, src.title):
                    if (candidate or "").strip():
                        body = extract_span(candidate, terms, budget=per_doc_chars)
                        break
            if spent and spent + len(body) > total_chars:
                break
            spent += len(body)
            hits.append(FtsHit(vertex_id=vid, rank=rank, score=score,
                               title=title, description=desc, content=body,
                               matched_terms=tuple(terms)))
        return hits


# ═══════════════════════════════════════════════════════════════════════════
# The search-hit boundary — §A.1 (consumer: the lumen tekton's op.retrieve,
# crystal/operators/impl/retrieval.py)
# ═══════════════════════════════════════════════════════════════════════════

def to_search_hits(hits: Sequence[FtsHit]) -> List[dict]:
    """Serialize to the retrieval consumer's flat hit shape.

    ⚠ THE SNIPPET GOES IN `content`. Lumen sends `"highlight": true` but never
    reads a `highlight` field — `retrieval.py:72` reads `content`, full stop. A
    hit whose text landed under `highlight` with an empty `content` is dropped
    by `if not content and not title: continue`, and **grounding goes to zero
    with no error and no log entry**. This function is deliberately the only
    place that builds the wire shape, and `test_fts.py::test_snippets_land_in_
    content_not_highlight` asserts the invariant, so the trap cannot be
    reintroduced by a caller assembling dicts by hand.

    Flat, no nesting (§A.1). `score` is emitted for compatibility but Lumen
    never uses it — LIST ORDER is what carries rank.
    """
    return [{"id": h.vertex_id,
             "version_id": h.vertex_id,
             "title": h.title,
             "description": h.description,
             "content": h.content,
             "score": h.score}
            for h in hits]


# DEPRECATED alias (2026-07-23): the lumen SERVICE is retired (lumen is now the
# wisdom/inference tekton in chorus); the boundary is named for what it does, not
# for a dead service. Kept so external callers don't break — new code uses
# `to_search_hits`.
to_lumen_hits = to_search_hits


# ═══════════════════════════════════════════════════════════════════════════
# Default resolver
# ═══════════════════════════════════════════════════════════════════════════
# ⚠ NARROW LOCAL INTERFACE — pending reconciliation with Unit L (vertex.py).
# The contentless index cannot produce snippets without source text, and Unit L
# owns vertex access. Rather than block, this reads the `vertex` table directly
# using contract §2's fixed column set. Swap it for Unit L's accessor when
# `vertex.py` lands; the `TextResolver` signature is the seam.

ContentLoader = Callable[[str], Optional[str]]


class PreviewOnlyIndex(RuntimeError):
    """Raised when full text was required but only a CAS preview was available."""


def _coerce_context(raw) -> dict:
    """`context` arrives EITHER as a nested object OR as a JSON-encoded string.

    §A.1 trap 2: over the Lumen POST path `content` and `context` are posted as
    JSON-encoded strings, and `context` is DOUBLY encoded. Unit X's extract, by
    contrast, stores it already decoded. Both shapes are real, so both are
    handled — and the loop below tolerates double encoding rather than assuming
    a fixed depth.
    """
    import json
    for _ in range(3):
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return {}
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return {}
    return raw if isinstance(raw, dict) else {}


def make_vertex_resolver(conn: sqlite3.Connection,
                         *, doc_column: str = "doc",
                         content_loader: Optional[ContentLoader] = None,
                         strict: bool = False) -> TextResolver:
    """Resolve source text for a set of `vertex.id`s.

    ═══════════════════════════════════════════════════════════════════════════
    ⚠ MEASURED ON THE REAL CORPUS — `doc.content` IS A TRUNCATED PREVIEW
    ═══════════════════════════════════════════════════════════════════════════
    Sampled from Unit X's node-71 extract (`lattice-export/extract.sqlite`),
    3,000 consecutive `wiki-en-*` rows:

        with content_ref : 3000 / 3000   (100%)
        len(doc.content) == 300 exactly : 2933 (98%)
        tail of a 300-char row: '...published between 1920 and 1975.\\n\\nPoir'

    It cuts mid-word. `doc.content` is a ~300-char PREVIEW; the full plaintext
    lives in CAS under `content_ref` (contract §2: *"cas/<sha256(plaintext)>;
    NULL = inline in doc"*).

    **Indexing `doc.content` therefore builds an index over previews.** That
    failure is entirely silent: FTS5 indexes the 300 chars happily, BM25 scores
    look sane, snippets render, and lexical recall for anything past the first
    ~50 words of every article is simply gone. Nothing errors and no metric
    moves. It is the same shape as §A.1 trap 1 — a total failure with no signal.

    So `content_loader` (`content_ref -> plaintext`) is REQUIRED for a real
    index. Without it this resolver runs in preview mode and says so:
      * `strict=True`  -> raises `PreviewOnlyIndex` on the first truncated row.
      * `strict=False` -> logs a warning once and counts the rows
        (`resolve.preview_only`), so a caller can assert on it.

    ⚠ DO NOT SOFTEN THIS TO A SILENT FALLBACK. (Explicit coordinator
    instruction, 2026-07-20.) Making preview mode quiet — dropping the raise,
    the warning, or the counter — restores exactly the silent-truncation
    failure this exists to expose, and it would look like a cleanup. The
    raise-or-count-and-warn behaviour is load-bearing.

    ⚠ IN PROGRESS: Unit X's extract carries `content_ref` values but no CAS
    blobs, and there is no local CAS directory. Unit X has been tasked with
    adding CAS blob retrieval to the exporter; `content_loader` is the seam it
    plugs into. Until then, Phase 0.A's lexical A/B cannot measure anything
    meaningful — it would be measuring recall over previews, i.e. a different
    system than the one being evaluated.

    ⚠ FIELD MAPPING IS PROVISIONAL. The real corpus does not carry
    `title`/`description`/`tags` — measured, 400 `wiki-en-*` rows have ZERO
    context keys (title is the first markdown line), while `concept`/`operator`
    rows carry `name`, `kind`, `role`, `description`, `sources`. The SSE
    pipeline derived these via `extract_text_from_context()` /
    `parse_tags_from_context()`. The minimal mapping below is a data-shape
    adapter, not a semantic rule table — but which context key feeds which FTS
    column is a Phase 5.0 migration decision, not this unit's to settle alone.
    """
    import json
    import logging

    log = logging.getLogger("mantle.lattice.fts")

    def resolve(vertex_ids: Sequence[str]) -> Mapping[str, FtsDocument]:
        if not vertex_ids:
            return {}
        qs = ",".join("?" * len(vertex_ids))
        out: Dict[str, FtsDocument] = {}
        for vid, ref, raw in conn.execute(
            f"SELECT id, content_ref, {doc_column} FROM vertex WHERE id IN ({qs})",
            tuple(vertex_ids),
        ):
            try:
                d = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                d = {}
            ctx = _coerce_context(d.get("context"))

            body = str(d.get("content") or "")
            if ref:
                full = content_loader(ref) if content_loader else None
                if full is not None:
                    body = full
                else:
                    resolve.preview_only += 1        # type: ignore[attr-defined]
                    if strict:
                        raise PreviewOnlyIndex(
                            f"{vid}: content_ref={ref} but no content_loader; "
                            "indexing the preview would silently truncate the "
                            "corpus (see make_vertex_resolver docstring)")
                    if not resolve.warned:           # type: ignore[attr-defined]
                        resolve.warned = True        # type: ignore[attr-defined]
                        log.warning(
                            "fts: indexing CAS previews, not full text (%s has "
                            "content_ref but no content_loader). Lexical recall "
                            "will be silently truncated.", vid)

            tags = ctx.get("tags") or ctx.get("sources") or ctx.get("kind") or ""
            if isinstance(tags, (list, tuple)):
                tags = " ".join(str(t) for t in tags)
            title = ctx.get("name") or ctx.get("title") or ""
            if not title and body:
                title = body.splitlines()[0][:200]   # markdown: first line
            out[vid] = FtsDocument(
                vertex_id=vid,
                title=str(title),
                description=str(ctx.get("description") or ctx.get("role") or ""),
                tags=str(tags),
                content=body)
        return out

    resolve.preview_only = 0     # type: ignore[attr-defined]
    resolve.warned = False       # type: ignore[attr-defined]
    return resolve


# ═══════════════════════════════════════════════════════════════════════════
# THE ONE INDEX — artifact projection, maintenance, and coverage
# ═══════════════════════════════════════════════════════════════════════════
# [John, 2026-07-23: *"There should be only one lexical index. Whatever is more
# universe-al."*] Two existed: this one, fuller but with zero runtime callers,
# and a legacy-era `fts_vertex` built solely by a MIGRATION script — which the
# runtime read, and which therefore did not exist at all on a store built the
# ordinary way. This section is what the runtime now uses, so there is one.
#
# THE BASELINE THIS MODULE GATED ADOPTION ON HAS NOW BEEN RUN (2026-07-23, live
# OEWN corpus, 120,684 rows, like-for-like: same salient terms, same
# content-type filter, same OR-of-quoted-literals, so the ONLY variable is the
# index). Top-10 agreed on 1 of 14 queries, and every meaningful divergence went
# THIS way — because `porter` stems the query onto the lemma:
#
#     dogs                  -> dog breeding, dog do, dog sled   (was: bait, Nyctereutes)
#     cats                  -> alley cat, cat food, Egyptian cat (was: clowder, caterwaul)
#     limits and derivatives-> derivable, derivative, differentiate (was: purine, pareve)
#     flying machines       -> plaster machine, brick-making machine (was: amusement arcade)
#     a bank that holds money-> bank holding company, central bank (was: bank, bank, bank)
#     studies of the mind   -> psychopharmacology, study         (was: mentally, evoke)
#
# So the shift is not merely "expected difference": the mechanism is better on
# the exact failure it was suspected of causing. Recorded because the gate asked
# for a measurement, not a preference.
#
# ⚠ WHY `description` AND `tags` ARE THE RIGHT FIELD NAMES, not corpus-specific
# ones. An artifact ADVERTISES AN OFFER (offers-and-needs retrieval): the offer
# IS its description. Its keyed terms ARE its tags. So the §2.1 field set is the
# universal shape and `gloss`/`context`/`lemmas` are one corpus's spelling of
# it — which is why the projection lives here, with the index, rather than in
# whichever leaf happens to be writing rows.

def project_artifact(doc: Mapping) -> FtsDocument:
    """An artifact document -> the indexable four fields.

    `description` prefers a described `context` (the artifact's own offer, which
    is what a need is matched against) and falls back to `gloss` for a row that
    has not been described yet — a dark artifact must still be findable, or it
    can never be described. `tags` are the keyed lemmas.

    ⚠ This is the DEFAULT projection. A type that declares its own `field_source`
    should eventually drive it (describers evolve), which is the open seam here;
    until then every type projects the same way and that is stated, not hidden.
    """
    ctx = doc.get("context")
    desc = ctx if isinstance(ctx, str) else ""
    if not desc:
        desc = str(doc.get("gloss") or "")
    tags = doc.get("lemmas") or doc.get("tags") or ()
    if isinstance(tags, str):
        tags = [tags]
    return FtsDocument(
        vertex_id=str(doc.get("id") or ""),
        title=str(doc.get("title") or ""),
        description=desc,
        tags=" ".join(str(t) for t in tags),
        content=str(doc.get("content") or ""))


# The index carries its OWN maintained count, for the same reason the store does: `count(*)`
# dereferences every record (EXPLAIN: 6M rows loaded to produce one integer) and this package bans
# it outright — `test_no_count_star_reaches_sqlite` caught the first version of `coverage()` below
# doing exactly that. The counter is bumped in the same transaction as the posting, so it cannot
# drift from what it counts.
C_FTS_TOTAL = "fts:total"


def _bump(conn: sqlite3.Connection, delta: int) -> None:
    conn.execute("INSERT INTO counter(name, n) VALUES(?, ?) "
                 "ON CONFLICT(name) DO UPDATE SET n = n + excluded.n", (C_FTS_TOTAL, delta))


def retract_artifact(conn: sqlite3.Connection, vertex_id: str) -> bool:
    """Remove ONE artifact from the lexical index — postings, map row, AND the counted total.

    🔴 THE DEFECT THIS EXISTS FOR (measured 2026-07-30). `FtsIndex.delete` had **zero callers**
    anywhere in this package outside `index_artifacts`'s archived branch, so
    `vertex.delete_artifact()` — which retracts `listkey` postings, the demand row, the merkle leaf
    and every counter — left the `fts`/`fts_map` rows standing. A deleted artifact therefore kept
    answering lexical search, and `search()` handed back a `vertex_id` whose `get_artifact()` is
    `None`. **That is a WRONG answer, not an empty one**: the caller cannot tell a hit it may not
    resolve from a hit that simply has no text, and Lumen's `if not content and not title: continue`
    silently drops it — grounding degrades with no error and no log.

    STATE THE FAILURE MODE, so this is not a check that cannot fail: without this call
    `test_delete_artifact_retracts_the_lexical_index` finds the deleted id still ranked first for
    its own words, and `fts:total` one higher than the rows the index holds.

    ⚠ BOTH MOVES OR NEITHER. Purging the postings without decrementing `fts:total` trades a ghost
    row for a drifted counter — `coverage()` would then report `missing` as a negative-going number
    that no rebuild explains, and `verify_counters` compares that total against a real walk of
    `fts_map`. The bump happens on the caller's cursor, i.e. inside the SAME transaction as the
    vertex delete, so the two can never be observed disagreeing.

    Returns True when something was actually retracted. A store whose index was never built
    (`is_built` False — no `fts_map` table) has nothing to retract and says so rather than raising:
    a delete must not fail because a derived index was absent.
    """
    if not is_built(conn):
        return False
    conn.execute("CREATE TABLE IF NOT EXISTS counter (name TEXT PRIMARY KEY, n INTEGER NOT NULL)")
    if not FtsIndex(conn).delete(str(vertex_id)):
        return False
    _bump(conn, -1)
    return True


def index_artifacts(conn: sqlite3.Connection, docs: Iterable[Mapping]) -> int:
    """Index (or re-index) artifact documents. Returns rows indexed.

    An index is DERIVED data, so the path that writes the rows derives it —
    nothing else knows the rows exist. Archived rows are REMOVED rather than
    indexed: a content-decides snapshot must not surface alongside its own head.
    """
    idx = FtsIndex(conn)
    idx.ensure_schema()
    conn.execute("CREATE TABLE IF NOT EXISTS counter (name TEXT PRIMARY KEY, n INTEGER NOT NULL)")
    n, delta = 0, 0
    for d in docs:
        vid = str(d.get("id") or "")
        if not vid:
            continue
        existed = idx.rowid_for(vid) is not None
        if str(d.get("state") or "") == "archived":
            # ONE retraction path (`retract_artifact`), shared with `vertex.delete_artifact` — it
            # bumps the counter itself, which is why no `delta` is applied here.
            retract_artifact(conn, vid)
            continue
        idx.index(project_artifact(d))
        if not existed:
            delta += 1                 # a RE-index is not a new row; the count must not double it
        n += 1
    if delta:
        _bump(conn, delta)
    return n


def rebuild_from_vertices(conn: sqlite3.Connection, *, doc_column: str = "doc") -> int:
    """Rebuild the whole index from the vertex table — for a store whose rows
    landed before the writer indexed them."""
    import json
    conn.execute(f"DROP TABLE IF EXISTS {FTS_TABLE}")
    conn.execute(f"DROP TABLE IF EXISTS {MAP_TABLE}")
    conn.execute("CREATE TABLE IF NOT EXISTS counter (name TEXT PRIMARY KEY, n INTEGER NOT NULL)")
    conn.execute("DELETE FROM counter WHERE name = ?", (C_FTS_TOTAL,))   # the rows went with it
    FtsIndex(conn).ensure_schema()
    docs = []
    for row in conn.execute(f"SELECT {doc_column} FROM vertex").fetchall():
        try:
            docs.append(json.loads(row[0]))
        except Exception:
            continue
    return index_artifacts(conn, docs)


def is_built(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (MAP_TABLE,)).fetchone() is not None


def coverage(conn: sqlite3.Connection) -> Dict[str, int]:
    """Indexed rows vs vertices — the number that would have shown a fresh store
    was unsearchable. PUBLISHED so health monitoring reads it, rather than
    anyone probing for it by hand."""
    def _counter(name: str) -> int:
        row = conn.execute("SELECT n FROM counter WHERE name = ?", (name,)).fetchone()
        return int(row[0]) if row else 0
    try:
        n_idx = _counter(C_FTS_TOTAL)
        # `c_vertex_total()` is a FUNCTION so a typo is an AttributeError at import rather than a
        # silently wrong number — a missing counter row reads as 0, which is the wrong-answer class.
        from . import schema as _schema
        n_v = _counter(_schema.c_vertex_total())
    except Exception:
        n_idx = n_v = 0
    return {"indexed": n_idx, "vertices": n_v,
            "missing": n_v - n_idx, "built": is_built(conn)}


# ── the store-level entry points ─────────────────────────────────────────────
# ⚠ THESE LIVED IN EMBER, as a `lexical_index` adapter module. That was a second NAME for one thing,
# and a second name is a second path in every way that matters — it drifts, it has to be kept in
# step, and a reader has to check which one a call site meant. [John, 2026-07-23: "leave one path.
# the only path. No constants, no fitting. no forcing."] The impedance is real (a leaf holds a
# lattice, this module speaks sqlite3 connections) but it belongs HERE, with the index, not in every
# leaf that writes rows.

def index_for(db, docs: Iterable[Mapping]) -> int:
    """Index documents through the lattice `db` handle (the thing with `read()`/`write()`).

    Rarely needed: the STORE maintains this index in its own write path, so a caller that writes
    through `put_artifact`/`put_many` never touches it. This exists for a rebuild and for tests."""
    with db.write() as cur:
        return index_artifacts(cur, docs)


def rebuild_for(db) -> int:
    """Rebuild the whole index from the vertex table — for a store whose rows landed before the
    writer maintained it."""
    with db.write() as cur:
        return rebuild_from_vertices(cur)


def coverage_for(db) -> Dict[str, Any]:
    return coverage(db.read())


def is_built_for(db) -> bool:
    try:
        return is_built(db.read())
    except Exception:
        return False
