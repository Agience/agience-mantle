"""Rank a candidate set by measured reach, and derive how many of them are the answer.

## Why this is in mantle

The store's own search — `recall_artifacts` and the SSE narrowing behind it — produces candidates
and has no way to order them by what a question is ABOUT. That ordering is this module, and mantle
carries it so the base install has it out of the box: no chorus, no persona, no ontology required.

Every part of it degrades to a measured statement rather than a guess:

    no `match` seam anywhere  -> the candidate order passes through, `{"reach": "unavailable"}`
    a query that fires nothing -> the same, `{"reach": "no-coordinate"}`
    nothing reachable          -> the same, `{"reach": "unreached"}` with the count
    two ingests, one concept   -> the rows fold to one and the account names what was folded
    no synset in the ranking   -> the cut has no frame to read and falls back to `_knee`
    no instrument registered   -> the cut falls back to `_knee`, a proportional-drop rule

So the base install ranks lexically and cuts by `_knee`; attaching the ontology (the `match` seam)
turns on reach, and attaching an instrument turns the cut into the aperture's own `k_signal`. That
is the same two-tier shape `search/beacon` has for the spectral read — a thin arm that is not less
correct, and a sharper one when it is present.

Attaching it takes no call: a host that registers its seams with `prism.runner` — which is what
`ember/runtime/seams.py` does at import of `ember` — equips this module by having done that. An
explicit `bind()` still wins where a process wants to state something narrower. What matters is
that neither is chorus: mantle plus a runner is the whole requirement.

A STANDALONE mantle node has no runner in its process, so it orders by coverage — the correct
answer for a node with no ontology in it. `MANTLE_ONTOLOGY_HOST=ember` is the operator's way to
put one there, and importing what it names is the entire mechanism.

## What it does NOT do

It does not retrieve. Candidates arrive from whatever produced them — mantle's own recall, an FTS
index, a caller's list — as `(id, content_type, score)` with score in BM25's sign convention
(negative, most-negative best). Keeping retrieval out is what lets one implementation serve both
the encrypted store's search and a corpus one, which are different indexes over different content.

It also holds no answer composition. Turning a ranked set into prose is a persona's job and stays
in `sage`.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["rank", "cut_for", "relevance_cut", "knee"]


#: How a candidate's position is read. `None` until a host binds one — the base install has no
#: ontology, so a candidate has no position and `rank` passes the order through.
_MATCH_SEAM: Optional[Any] = None
_PROJECTION_SEAM: Optional[Any] = None


def bind(*, match=None, projection=None) -> None:
    """Attach the ontology. `match` supplies `fired_field` / `propagate` / `offer_synsets`;
    `projection` supplies `frame(store, names)` for the cut's screen. Either may be left unbound,
    and each absence is reported rather than worked around."""
    global _MATCH_SEAM, _PROJECTION_SEAM
    if match is not None:
        _MATCH_SEAM = match
    if projection is not None:
        _PROJECTION_SEAM = projection


#: The module whose IMPORT registers this node's host seams, named by the operator.
#:
#: `_registered_seam` reads what a host registered with `prism.runner`, and on a node where the
#: runner shares mantle's process that map is already populated. A STANDALONE mantle node — the
#: store running on its own port, which is how node 71 runs it — imports no runner, so the map is
#: empty and every recall answers `ordering: "coverage"`. That is the honest answer for a node with
#: no ontology in it, and it is also a knob an operator may want to turn: an editor talking to that
#: node over MCP gets the reach-ranked `recall` the moment the ontology is in the same process.
#:
#:     MANTLE_ONTOLOGY_HOST=ember
#:
#: names a module to import, and importing it is the whole mechanism — `ember/runtime/seams.py`
#: registers `match` and `projection` at import of `ember`, so there is nothing here that knows
#: what an ontology is. Unset (the default) imports nothing and changes nothing.
_ONTOLOGY_HOST_ENV = "MANTLE_ONTOLOGY_HOST"

#: Import-once state for the above: the module name that was tried, so a node whose host module is
#: missing logs once rather than on every recall.
_HOST_TRIED: Optional[str] = None


def _load_ontology_host() -> None:
    """Import the operator-named host module, at most once per process.

    Failure is logged and swallowed: a node configured to load an ontology it does not have should
    serve coverage-ordered recalls, not 500 on every query. The log names the module so the
    misconfiguration is visible rather than inferred from the ordering.
    """
    global _HOST_TRIED
    import os
    target = (os.getenv(_ONTOLOGY_HOST_ENV) or "").strip()
    if not target or _HOST_TRIED == target:
        return
    _HOST_TRIED = target
    import importlib
    import logging
    try:
        importlib.import_module(target)
        logging.getLogger(__name__).info(
            "%s=%s imported; host seams registered: %s",
            _ONTOLOGY_HOST_ENV, target, sorted(_registered_names()))
    except Exception as exc:  # noqa: BLE001 — a bad module name must not break recall
        logging.getLogger(__name__).warning(
            "%s=%s could not be imported (%s: %s); recalls order by coverage",
            _ONTOLOGY_HOST_ENV, target, type(exc).__name__, exc)


def _registered_names():
    try:
        from prism.runner import registered_seams
    except Exception:
        return ()
    return registered_seams().keys()


def _registered_seam(name: str):
    """The seam a HOST registered with `prism.runner`, imported. `None` when none did.

    `bind()` above requires someone to call it, and for a while the only caller was
    `sage.content_search` — so the ranking was mantle's but reaching it still meant installing
    chorus, which is the dependency the extraction existed to remove. A runner already answers
    exactly this question: `prism.runner.register_seam(name, dotted_module)` is how a host says
    which module fills a name, and `ember/runtime/seams.py` registers `match` and `projection` at
    import of `ember`. Reading that map here is what makes mantle + a runner enough.

    It is a named indirection and not a dependency: the value is a string, this module imports it
    only if a host put one there, and a process that registered nothing imports nothing. mantle's
    static import graph is unchanged and still names no ember or crystal module.

    A standalone node has no runner in its process and so no registrations; `MANTLE_ONTOLOGY_HOST`
    is the operator's way to put one there — see `_ONTOLOGY_HOST_ENV`.
    """
    _load_ontology_host()
    try:
        from prism.runner import registered_seams
    except Exception:
        return None
    target = registered_seams().get(name)
    if not target:
        return None
    import importlib
    return importlib.import_module(target)


def _resolve_seam(name: str):
    """The seam, PROVEN reachable — not merely present.

    Two sources, in this order:

    1. what a host passed to `bind()`, which wins because passing an object is a statement about
       THIS process that a registry lookup cannot contradict;
    2. what a host registered with `prism.runner` — see `_registered_seam`.

    Then the attribute is touched. A host may bind a lazy proxy that resolves on first attribute
    access, so a seam that is bound can still be unreachable; touching it here means a host with
    nothing behind the proxy fails on this line, inside the caller's `try`, rather than one line
    later where the failure would escape as a traceback instead of the honest "unavailable" this
    module promises.
    """
    seam = _MATCH_SEAM if name == "match" else _PROJECTION_SEAM
    if seam is None:
        seam = _registered_seam(name)
    if seam is None:
        raise ImportError("no %r seam is bound or registered; the base install ranks without one"
                          % name)
    getattr(seam, "fired_field" if name == "match" else "frame")
    return seam


def _knee(scores: Sequence[float], *, frame=None) -> int:
    """How many leading results form the relevant cluster — derived from the corpus's own scores.
    `scores` are BM25-signed (negative; most-negative = best) in best-first order.

    `prism.resolution.partition` finds which split best separates the series, by maximum
    between-class variance: non-parametric, scale-invariant, no tunable constant, global rather than
    local. It reports how much of the variance the split explains, so a series with nothing to
    separate can say so instead of being cut anyway."""
    from prism.resolution import signal_end
    return signal_end([-float(s) for s in scores], frame=frame)


def _relevance_cut(scores: Sequence[float], *, query: str | None = None, frame=None) -> int:
    """Where the relevance signal ends — the derived span count. A seam.

    Default: `_knee` (a proportional-drop rule, unit-tested). Optionally `adaptive_cut` — the aperture
    gates whether there is coherent signal above its derived noise floor (`K_signal`) and the
    scale-invariant relative gap says where: model-free, no embedding, no tunable constant. Gated by
    `EMBER_ADAPTIVE_MODE` (off|on|shadow), default off, so the serve path is byte-for-byte `_knee`.
    `shadow` serves `_knee` unchanged and records the adaptive pick alongside it
    (`EMBER_ADAPTIVE_SHADOW_LOG`) for a label-free A/B."""
    baseline = _knee(scores, frame=frame)
    from prism import adaptive_cut
    m = adaptive_cut.mode()
    if m == "off":
        return baseline
    picked = adaptive_cut.cut(scores, frame=frame)   # None → unreadable → defer to baseline
    if picked is None:
        return baseline
    if m == "shadow":
        adaptive_cut.record_shadow(query, scores, baseline, picked)
        return baseline                        # shadow is observational: behavior unchanged
    return picked


_TITLE_WORDS = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _salient_title(doc: dict, store, matcher) -> str:
    """The part of a document's title that states its subject. Not called.

    Kept because the reasoning is sound and the measurement that retired it is worth more than the
    code: it prices at +2 on `bench_canon --by-title` while every OEWN distance
    was 0 (§123), and at -3 once `refresh_ic` restored the metric. Trimming a title's generic
    anchors helps only when the anchors cannot be told apart by distance; when they can, the
    generic ones are already far and dropping them removes real reach.


    §110 anchors a prose artifact's positions on its TITLE and keeps body positions within one
    correlation length of one of them. Every anchor is treated alike, and a title is not made of
    alike words:

        `README — Read in this order`      3 anchors, including `inch.n.01` from the word `in`
        `prism-protocol — prism-protocol`  2 anchors, both the subject

    A generic anchor is near a great deal of the taxonomy — correctly, because `jc_tree` already
    encodes information content and a low-IC node IS close to everything under it. So the radius is
    not the problem and widening or narrowing it cannot be the fix: the problem is that a word
    which states nothing became an anchor at all.

    Measured, that is what decided `prism protocol`: README admitted 124 body positions to
    prism-protocol's 12, per-position reach was equal to within 3%, and the standardisation is
    size-aware by design (§118), so README won on size alone.

    The corpus already answers which words carry a piece of text, and this path already uses that
    answer for the QUERY — `salient_terms`, a term kept when it carries at least the text's own mean
    information. A title is a piece of text. Applying the same measure to it is the same rule on the
    same seam, not a second one: `README`/`Read` survive, `in`/`this`/`order` do not, and a title
    that is all subject — `prism-protocol` — is untouched.

    Falls back to the whole title whenever the measure cannot run, which is `salient_terms`' own
    convention: an unmeasurable corpus must not silently narrow anything.

    `matcher` is passed rather than looked up. `_match` is a local inside `_reach_rank`, resolved
    from the seam per call, so referring to it by name here raises `NameError` into this function's
    own `except` and returns every title unchanged in every process. by-title does not move by a
    single question under that failure, which is the only reason it was
    caught.
    """
    title = str(doc.get("title") or "")
    if not title.strip():
        return title
    try:
        from mantle.search.mantle.sse.tokenizer import tokenize
        stems = list(dict.fromkeys(tokenize(title) or []))
        if len(stems) < 3:
            return title              # too short to separate; the whole title is the subject
        keep = set(matcher.salient_terms(stems, store) or stems)
        all_words = _TITLE_WORDS.findall(title)
        words = [w for w in all_words if any(t in keep for t in (tokenize(w) or []))]
        if not words or len(words) == len(all_words):
            return title             # nothing to drop: hand back the title, not a rebuilt copy
        return " ".join(words)
    except Exception:  # noqa: BLE001 — an unmeasurable title anchors on all of itself
        return title


def _reach_rank(cand: List[Tuple[str, str, float]], query: str, store,
                *, own_position: bool = True,
                ) -> Tuple[List[Tuple[str, str, float]], Dict[str, Any], set]:
    """Re-rank BM25 candidates by measured reach. Returns (ranked, gap_account, reached_ids).

    Ranked scores follow BM25's sign convention (more negative = better) so the cut is unchanged:
    when reach is measured the score is `-energy`; when it degrades, the BM25 scores pass through
    untouched. `reached_ids` are the candidates the geometry actually placed — aboutness the caller
    can use instead of guessing at it lexically."""
    try:
        # the screened propagator, through the declared `match` seam. `HostSeamUnfilled` is an
        # `ImportError`, so a host that bound nothing lands in this same `except` on the same input
        # and the answer is still the honest `{"reach": "unavailable"}` it always was.
        _match = _resolve_seam("match")
    except Exception:
        return cand, {"reach": "unavailable"}, set()
    # Weighted by the corpus's own informativeness (IDF), not uniform — a filler word must not
    # outrank the subject just because it happens to name a synset (see match.fired_field).
    fired = _match.fired_field(query, store)
    if not fired:
        return cand, {"reach": "no-coordinate"}, set()
    # `expand_associative` carries the fired field one hop through the 24 relation types §13.25
    # recovered, under the same screened propagator, so the steps compose (weights multiply =
    # distances add, in nats) with no blend coefficient anywhere.
    #
    # Off by default because the measured effect is mixed, not because it is unfinished. Live, with
    # it on: "the bank of a river" loses the financial sense (right), "photosynthesis" stops
    # answering with "flora" once the cost is two-sided (right) — but "mass physics" answers
    # `bulk / Mass(Eucharist) / batch / Requiem`, and "I" returns to the head of "what is a star".
    # Some queries improve, some regress, with no query set yet large enough to say which dominates.
    #
    # Gated by `EMBER_ASSOC_HOP` (off | on), default off, following the same discipline as
    # `adaptive_cut`.
    import os as _os
    if (_os.getenv("EMBER_ASSOC_HOP") or "off").strip().lower() == "on":
        fired = _match.expand_associative(store, fired)

    # A modifier's projection, memoised for this query. `projected_positions_for` walks the store,
    # and a pool repeats positions heavily, so this is read once per distinct synset.
    _projected: Dict[str, List[str]] = {}

    def _is_modifier(name: str) -> bool:
        """Adjective, satellite adjective or adverb, read off the synset name.

        Both id shapes in this corpus carry the part of speech in the name — `oewn-00001740-a` and
        `able.a.01` — so this needs no store read. A name in neither shape is not a modifier as far
        as this can tell, and falls through to being its own position, which is what it was before.
        """
        tail = name.rsplit("-", 1)[-1] if name.startswith("oewn-") else (
            name.rsplit(".", 2)[-2] if name.count(".") >= 2 else "")
        return tail in ("a", "s", "r")

    def _position(cid: str) -> List[str]:
        """The candidate's own position in meaning-space.

        ## A modifier has none, and is projected — the same way the QUERY already is

        `fired_field` projects a modifier to the nouns it is about, because an adjective carries an
        information content but no hypernym parent: `jc_tree` has nothing to measure and a synset
        admitted directly would score a distance that came from nowhere. This side returned the
        synset's own name regardless, so a modifier ANSWER was placed where no question could reach
        it. Measured:

            glacier  (noun)   energy 4.1131   distance 0.0    <- the query's own position
            able     (adj)    energy 0.1674   distance 1.53
            abaxial  (adj)    energy 0.0000   distance inf

        One side projecting and the other not is the asymmetry, not the projection. So a modifier
        candidate is read at the nouns it is about, through the same seam function and the same
        relations the query side uses. 66% of modifiers project; the rest place nowhere and keep
        the position they had, which is the honest outcome — the source holds no link for them.
        """
        if not cid.startswith("wn-"):
            return []                            # prose: resolved below, from its own text
        name = cid[3:]
        if _is_modifier(name):
            if name not in _projected:
                try:
                    _projected[name] = list(
                        _match.projected_positions_for(name, store) or [])
                except Exception:  # noqa: BLE001 — no projection seam: behave as before
                    _projected[name] = []
            if _projected[name]:
                return _projected[name]
        return [name]                            # a synset vertex is a position

    conn = None
    try:
        conn = store.artifacts.db.read()
    except Exception:
        pass

    # The need's own position is always a candidate — no pool depth required. BM25 is a lexical
    # ranker, so whether it surfaces the very synset the need names is an accident of term density:
    # the head sense of "dog" (`Canis familiaris`, whose gloss never says "dog") sits below rank 200
    # while `dog do` and `dog collar` rank above it, so the correct answer would be absent from the
    # pool and no re-rank could recover it. Raising `_POOL` would only be a larger guess. The need
    # already names where it points, so that node is a candidate by construction; its reach is then
    # measured like any other — it wins on merit (energy 9.33 at distance 0.0 vs 4.19 for a hyponym)
    # or it does not.
    # ── `own_position=False`: do NOT add it, for a caller that will only throw it away ──────────
    # A recall filters the returned order back to the ids that went in, because an id that entered
    # from the ontology never passed the light cone. Adding one anyway is not merely wasted work —
    # it LOSES the answer. The added row and the narrowed row are the same concept from two
    # ingests, so `_fold_sources` collapses them, and it keeps the best-scoring row:
    #
    #     fired("what is a glacier")   glacier.n.01 8.226   oewn-09312237-n 4.113
    #     folded                       {wn-glacier.n.01: [wn-oewn-09312237-n]}
    #
    # `wn-glacier.n.01` is Princeton and is not in the SSE index; `wn-oewn-09312237-n` is, and is
    # what the narrowing actually found. The ontology-injected row outscores it, absorbs it, and is
    # then dropped by the caller's filter — taking the real answer with it. Measured on the served
    # path, `what is a glacier` answered with `Piedmont glacier / continental glacier / polar
    # glacier`: every hyponym survived because nothing folded it, and the head sense did not.
    #
    # With 77% of Princeton surfaces also present in OEWN (§67), that is most of a corpus.
    #
    # The default stays `True`. For a CORPUS search the injection is right and load-bearing — the
    # need already names where it points, and BM25 will not surface a head sense whose gloss never
    # says the word. Only a caller that filters by provenance afterwards must decline it.
    known = {c[0] for c in cand}
    if own_position:
        for _name in fired:
            _vid = "wn-" + _name
            if _vid in known:
                continue
            try:
                if store.artifacts.get_artifact(_vid):
                    cand = list(cand) + [(_vid, "text/x-wordnet", 0.0)]
                    known.add(_vid)
            except Exception:
                pass

    # ── one propagation per POSITION, and a candidate is the sum of its own ──────────────────────
    # The screened propagator is additive over targets and its pair gate is per-pair
    # (`prism.propagation.screened_accumulate`: no cap, no cross-target coupling, `nearest` a min
    # over pairs), so for any candidate
    #
    #     propagate(fired, targets) == (sum_t energy_t, min_t distance_t)
    #
    # where each `(energy_t, distance_t)` is `propagate(fired, [t])`. That identity is why this
    # cache is not merely an optimisation: the standardisation below ALREADY needs every position
    # on its own, so the per-position reads happen either way, and the batch call over the same
    # targets was recomputing exactly the distances they had just computed. Positions repeat
    # heavily across candidates — every article about glaciers stands on `glacier.n.01` — so the
    # per-position read is cached across the whole pool while the batch call could not be.
    #
    # A position whose synset does not resolve reads `(0.0, inf)`, which is what the batch call
    # scored it as by skipping it, so the identity holds there too.
    _per_position: Dict[str, Tuple[float, float]] = {}

    def _reach_of(position: str) -> Tuple[float, float]:
        """`(energy, nearest_distance)` for one position, measured once per query."""
        if position not in _per_position:
            try:
                _e, _d = _match.propagate(fired, [position])
                _per_position[position] = (float(_e), float(_d))
            except Exception:
                _per_position[position] = (0.0, float("inf"))
        return _per_position[position]

    # ── one read for every prose candidate, not one read each ────────────────────────────────────
    # The lattice is a 9.7 GB file and a candidate's doc row is a point lookup in it, so a pool of
    # 200 was 200 round trips before anything was measured — and the pass is I/O bound, not compute
    # bound: sampled during a live run it spent 46 seconds of CPU in 300 seconds of wall clock.
    # `id` is the primary key, so `IN (...)` is the same index walk done once. Chunked because
    # SQLite's parameter limit is finite and a pool has no fixed size.
    #
    # A missing id simply does not come back, which is the same absence `fetchone() -> None` was.
    #
    # EVERY candidate is read, not only prose. A concept candidate takes its position from its own
    # id and needs no doc for that — but the fold below needs its lemmas and its source, and the
    # read is one statement either way.
    _docs: Dict[str, str] = {}
    if conn is not None:
        _ids = [c[0] for c in cand]
        for _i in range(0, len(_ids), 400):
            _chunk = _ids[_i:_i + 400]
            try:
                _sql = ("SELECT id, doc FROM vertex WHERE id IN (%s)"
                        % ",".join("?" * len(_chunk)))
                for _row in conn.execute(_sql, _chunk):
                    _docs[_row[0]] = _row[1]
            except Exception:
                break                            # fall through: no positions from prose this query

    scored, unreached, sample = [], 0, []
    for cid, ct, bm in cand:
        targets = _position(cid)
        if not targets and cid in _docs:         # prose: the terms the document itself is keyed on
            try:
                import json as _json
                row = (_docs[cid],)
                if row:
                    d = _json.loads(row[0])
                    # ── the document's OWN key terms, not a re-derivation from its title ─────────
                    # `astra/doc_index.py` already extracted these at ingest: the content words
                    # that carry at least the document's own mean information as the corpus
                    # measures it. That bar is the document's own mean, so it scales with the
                    # document and with the corpus and nothing about it is chosen — the same rule
                    # `corpus_stats._salient` applies to a query.
                    #
                    # What this replaces was `title + gloss + content` truncated at 400 characters.
                    # Measured on the live corpus, that truncation was inert — a wiki artifact's
                    # doc row carries 6 to 50 characters of text (mean 12), because `content` lives
                    # in CAS behind decryption and never reaches this read. So the window cut
                    # nothing and the position came from the TITLE alone.
                    #
                    #     Gorilla Monsoon   title  -> gorilla, monsoon
                    #                       lemmas -> gorilla, monsoon, wrestling, hall, fame, wwf
                    #
                    # The lemmas are what place that article away from a weather query.
                    # ── and its TITLE, joined rather than superseded ─────────────────────────
                    # `lemmas` beats a title for wiki prose, which is why it took over here. But
                    # it took over COMPLETELY, and a sectioned document is the case where that
                    # loses the subject: `astra/doc_index` extracts a section's key terms from
                    # the section BODY, and a body does not repeat the name of the document it is
                    # part of. Measured on canon:
                    #
                    #     canon:AGENT-HOST-DESIGN#0
                    #        title  -> AGENT-HOST-DESIGN — 0. The finding under all the findings
                    #        lemmas -> finding, findings, six, audits, workspace, pattern, ...
                    #                  (109 terms, and NONE of them agent, host, or design)
                    #
                    # So the query "agent host design" positioned that section away from its own
                    # subject, and `UNIVERSAL-ECONOMICS#intro` — which never says it in its title
                    # but does in its body — outranked it. The title is the only place a section
                    # carries what document it belongs to.
                    #
                    # Joining costs nothing when the title is redundant: `offer_synsets` dedupes
                    # by synset name, so a title word already among the lemmas adds no position,
                    # and `n` — the denominator of the standardisation below — does not move. A
                    # title term that IS new adds one position and is then measured like every
                    # other: it lifts the candidate only if it actually reaches the fired field,
                    # and costs it `mu` if it does not. Nothing here is weighted or preferred.
                    # ── a candidate is placed as a subject, and a subject can be a verb ─────
                    # `offer_synsets` is noun-only by design, because it places an offer, where a
                    # verb is the frame rather than the subject: "describes markdown documents"
                    # contributes `markdown`/`document` and nothing from "describes".
                    #
                    # An artifact is the other case. `cn-frighten` is ABOUT frightening, and placing
                    # it by its nouns alone places it nowhere. Measured 2026-08-25 over 500
                    # ConceptNet terms drawn uniformly by rowid: noun-only 46.8%, and a further
                    # **2.8% ± 0.7pp** — ~33,000 artifacts — placed only once verbs are asked for
                    # (`frighten.v.01` · `embrittle.v.01` · `evolve.v.01` · `mislead.v.01`).
                    #
                    # The verb taxonomy has a hypernym tree, an information content, and a diameter
                    # (1.109) inside the horizon (1.691), so `jc_tree` measures it like any other.
                    # A noun-verb pair has no subsumer and the mass gap already refuses it.
                    # Adjectives and adverbs stay excluded: no hypernym parent, nothing to measure.
                    #
                    # `getattr` because an older seam has only the offer half; its absence is the
                    # noun-only answer this branch had before.
                    lem = d.get("lemmas") or []
                    txt = " ".join([str(d.get("title") or "")]
                                   + [str(x) for x in lem]).strip()
                    _subject = getattr(_match, "subject_synsets", None)
                    targets = ((_subject(txt) if _subject is not None
                                else _match.offer_synsets(txt)) or [])
                    # ── keep the positions that agree with each other (§110) ─────────────
                    # One position per key term means a document is placed at twenty-six
                    # points spanning 88% of the corpus diameter — smeared across the space
                    # rather than located in it, so its distance to a query barely depends
                    # on the query. `coherent_core` keeps the densest neighbourhood within
                    # one correlation length; `xi` is the propagator's own scale, derived in
                    # `_derive_geometry`, so the radius is the corpus's and not a tuning.
                    #
                    # Reached through `getattr` because an older seam has no such function
                    # and must keep working — an absent core is "nothing to group", which is
                    # the full position set, exactly as before this existed.
                    _core = getattr(_match, "coherent_core", None)
                    if _core is not None and targets:
                        try:
                            _anchor = _match.offer_synsets(str(d.get("title") or ""))
                            targets = list(_core(targets, store,
                                                 anchor=list(_anchor or []))) or targets
                        except Exception:  # noqa: BLE001 — grouping must never fail a rank
                            pass
            except Exception:
                targets = []
        energy, dist = 0.0, float("inf")
        for _t in targets:
            _e, _d = _reach_of(_t)
            energy += _e
            if _d < dist:
                dist = _d
            sample.append(_e)
        if energy <= 0.0:
            unreached += 1
        scored.append([energy, len(targets), dist, (cid, ct, bm)])

    # ── reach standardised against this query's own null ─────────────────────────────────────────
    # `propagate` sums over the positions it is handed, so raw energy is EXTENSIVE: a candidate that
    # names more things scores higher for naming more things. A synset names exactly one — itself —
    # and a document names every term the corpus keyed it on, so the two are not comparable at all.
    # Measured on the live corpus, "what is a glacier" under the raw sum:
    #
    #     Highline Trail (Glacier National Park)   10.31   rank 1
    #     glacier.n.01                              8.56   rank 31
    #
    # Every fixed correction for that is a guess at how reach should scale with size, and each one
    # buys precision by discarding documents. Measured over 18 questions — target at rank 1, and how
    # many documents survive into the top ten:
    #
    #     raw sum            1/18     156 documents
    #     energy / n        17/18      24
    #     max per position   7/18     140
    #     energy / sqrt(n)  15/18     117
    #     standardised      17/18      52
    #
    # `energy / n` is what an earlier pass shipped, and it is why the answers became lexicon-only: a
    # concept always has exactly one position, the smallest denominator available, so dividing by
    # count hands it a structural win no measurement gave it.
    #
    # The standardisation asks the question the corpus can answer instead. Under a null where a
    # candidate's positions say nothing about the need, its total reach is the sum of `n` draws from
    # the reach this pool actually exhibits — mean `mu`, spread `sigma`, both measured HERE, from
    # the positions this query reached. A sum of `n` such draws has expectation `n*mu` and spread
    # `sqrt(n)*sigma`, so
    #
    #     z = (energy - n*mu) / (sqrt(n)*sigma)
    #
    # is how far this candidate stands above what a candidate of its size would reach by nothing.
    # Size leaves the comparison because it is in both terms, not because it was divided out; a big
    # document is not penalised for being big, it is asked to beat what its own size predicts. No
    # exponent is chosen and no constant appears — `mu` and `sigma` are this query's own.
    #
    # `sigma == 0` means every reached position scored alike and there is no spread to divide by;
    # the numerator alone still orders them, and that is the whole of what was measured.
    if sample:
        _mu = sum(sample) / float(len(sample))
        _var = sum((x - _mu) ** 2 for x in sample) / float(len(sample))
        _sigma = _var ** 0.5
    else:
        _mu = _sigma = 0.0
    for row in scored:
        _energy, _n = row[0], row[1]
        if not _n:
            row[0] = float("-inf")
        else:
            _dev = _energy - _n * _mu
            row[0] = _dev / ((_n ** 0.5) * _sigma) if _sigma > 0 else _dev

    # `unreached` counts candidates that reached NOTHING, and that is the test — not the sign of the
    # standardised score. A `z` below zero means "reached less than its size predicts", which is a
    # real reading of a real candidate; treating it as unreached would report a measured result as
    # an absence.
    if unreached >= len(scored):
        # The need has a position but this corpus offers no path to it. Keep the teleport order and
        # report it: an unclosable gap is a finding.
        return cand, {"reach": "unreached", "unreached": unreached}, set()
    scored.sort(key=lambda t: (-t[0], t[2]))     # most reach first; ties broken by true distance
    best_z, _n, best_d, _c = scored[0]
    # The fold runs AFTER the measurement and after the sort. `mu` and `sigma` are what this pool
    # exhibited, and dropping rows before measuring it would make the null a function of which
    # ingests happen to be loaded. What the fold changes is the ANSWER — one concept, one row —
    # which is also why it runs before the cut: the cut counts answers.
    ranked, folded = _fold_sources(
        [(cid, ct, -z) for z, _n, _d, (cid, ct, _bm) in scored], _docs)
    account = {"reach": "measured", "reach_energy": round(best_z, 6),
               "nearest_distance": (None if best_d == float("inf") else round(best_d, 4)),
               "unreached": unreached}
    if folded:
        # Named, not counted away: these are where the other source's gloss lives, and a caller
        # showing "also described by" needs the ids rather than a tally.
        account["folded"] = folded
    return ranked, account, {c[0] for z, _n, _d, c in scored if _n}


#: A well-formed Interlingual Index id. Measured 2026-08-25 on 71/home: 555,344 values match,
#: and the ONLY value that does not is the literal `"in"`, on exactly 3,216 rows — one bad parse
#: (a language code reaching the field), not scattered corruption. Unvalidated, those 3,216 fold
#: into a single 3,216-member "concept", which is the one way this key can be worse than no key.
_ILI = re.compile(r"^i\d+$")


def _fold_key(doc: Dict[str, Any]):
    """What a row IS, if the lexicons say so; otherwise what it NAMES.

    ## The ILI key — an identity, not a heuristic

    Measured 2026-08-25: the Interlingual Index is present on 558,560 of 676,225 synset rows —
    every OEWN row (120,630) and all but 6 OMW rows (437,930) — giving 117,127 groups with more
    than one member, **440,902 collapsible rows**, and 116,944 groups spanning more than one
    lexicon. It is the cross-lexicon identity these sources are published with.

    It also reaches what the lemma key cannot. An OMW row is a TRANSLATION: `wn-omw-hr-08925093-n`
    names its concept in Croatian, so `(pos, lemma set)` never matches its English twin and 437,936
    translation rows fold onto nothing. Under ILI they fold onto the row they translate.

    **The Princeton side carried none until 2026-08-25, and now carries one on 90.6% of its
    rows.** Re-measured 2026-08-26 over the `wn-` id range on 71/home: pwn 106,640 of 117,659
    (90.6%), oewn 120,630 of 120,630, omw 437,930 of 437,936. The backfill derived the mapping from
    THIS corpus — OMW ids encode the PWN 3.0 offset and OMW rows carry the ILI, so
    `Princeton synset → offset → wn-omw-*-<offset>-<pos> → ili` is a local derivation. Nothing was
    downloaded: an external map would be a claim about this corpus from outside it, needing its own
    provenance, where this is the corpus agreeing with itself. Across 117,187 OMW (offset, pos)
    keys the languages disagreed on the ILI **0 times**.

    ILI is still a first key rather than a replacement, and the reason is now the 11,019 rows
    (9.4%) that carry none: 10,866 have no OMW twin at their offset, and 153 are refusals where
    OEWN publishes a different ILI — counted, not guessed. Those keep the lemma key below, and it
    keeps pairing them with OEWN exactly as measured.

    ## `(pos, the folded lemma set)` — what a row NAMES, independent of which source named it.

    Diacritics are folded and case dropped, the same alphabet `crystal.ontology.lookup` uses, so
    two lexicons spelling `Canis familiaris` and `canis familiaris` are one name. `None` when the
    doc carries no lemmas: a row that names nothing cannot be folded onto another.

    The satellite adjective pos `s` is NOT normalised to `a`. Measured on the live corpus,
    normalising it merged `able.a.01` with `able.s.02` — a head adjective and its satellite, two
    real senses — and raised same-lexicon key collisions from 7,958 to 8,331.
    """
    import unicodedata

    # The lexicons' own identity wins where they publish one. Shape-checked: see `_ILI`.
    ili = doc.get("ili")
    if ili and _ILI.match(str(ili)):
        return ("ili", str(ili))

    lemmas = doc.get("lemmas") or []
    if not lemmas:
        return None
    out = set()
    for lemma in lemmas:
        text = unicodedata.normalize("NFD", str(lemma))
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = text.lower().replace("_", " ").strip()
        if text:
            out.add(text)
    if not out:
        return None
    return (str(doc.get("pos") or ""), tuple(sorted(out)))


def _fold_sources(ranked: List[Tuple[str, str, float]], docs: Dict[str, str]
                  ) -> Tuple[List[Tuple[str, str, float]], Dict[str, List[str]]]:
    """Collapse rows that are ONE answer arriving from more than one source. Best-scoring row wins.

    ## The defect

    This corpus was seeded by three ingests — `op.source.wordnet` (Princeton names),
    `op.source.oewn` and `op.source.omw` — and the first two both describe English. The same
    concept is therefore a vertex under each, ranked twice, and both copies land inside a cut the
    instrument derived. Measured over the 25 generated examples, **24 of 149 shown answer rows
    (16.1%) repeated a title already in that answer**: `volcano` and `photosynthesis` each answer
    twice, and each repeat costs a slot.

    ## Why this key, and why the source test

    The obvious key — the id — cannot work: OEWN renumbered its synsets, so a Princeton name does
    NOT derive its OEWN counterpart. Measured: of 117,659 Princeton synset vertices, deriving
    `wn-oewn-<offset>-<pos>` from the nltk offset hits an existing vertex 651 times (0.6%), and the
    hits are wrong — `a_posteriori.r.01` derives the id of `aboard / alongside`. The ILI those
    lexicons carry IS the key, and since 2026-08-25 the Princeton side carries one too — see
    `_fold_key`. It is a first key and not a replacement, because 9.4% of Princeton rows still
    have none and fall through to the lemma key described here.

    What the rows do carry is what they NAME. On the live corpus, `(pos, lemma set)` gives 103,896
    keys pairing a Princeton row with an OEWN one, and on a 4,000-pair sample **97.5% have
    byte-identical glosses**; every pair in the 0.3% tail with little gloss overlap is still the
    same concept said differently (`rappel` / `abseil`, `genus Spirillum`, `Arctocephalus`). Zero
    false pairs in the sample.

    The same key ALSO collides 7,958 times WITHIN one lexicon, where the two rows are genuinely
    different senses — so the key alone is not identity and is not used as identity. A fold happens
    only between rows whose docs name DIFFERENT `operator`s. Two senses from one ingest keep their
    own rows; a concept described by two ingests becomes one. That test is what makes this a
    statement about duplicate SOURCES rather than a claim that a lemma set identifies a concept.

    Returns the folded order and `{kept_id: [ids folded into it]}` — the dropped rows are named,
    not silently discarded, because they are where a caller finds the other source's gloss.
    """
    import json as _json

    parsed: Dict[str, Dict[str, Any]] = {}
    for cid, blob in docs.items():
        try:
            d = _json.loads(blob)
            if isinstance(d, dict):
                parsed[cid] = d
        except Exception:
            continue

    held: Dict[Any, Tuple[str, str]] = {}          # fold key -> (kept id, that row's operator)
    folded: Dict[str, List[str]] = {}
    out = []
    for row in ranked:                             # best-first, so the first row seen is the keeper
        cid = row[0]
        doc = parsed.get(cid)
        key = _fold_key(doc) if doc else None
        if key is None:
            out.append(row)
            continue
        source = str(doc.get("operator") or doc.get("via") or "")
        prior = held.get(key)
        # ── the source test guards the LEMMA key, and only it ────────────────────────────────
        # `(pos, lemma set)` collides 7,958 times WITHIN one lexicon on genuinely different senses,
        # so a fold there is only safe between rows from different ingests. An ILI collision is not
        # a collision: the index exists to say these rows are one concept, and two OMW rows sharing
        # it are two translations of it, published by the same ingest. Applying the source test to
        # an identity would leave 15 translations of one synset sitting in one answer.
        by_identity = isinstance(key, tuple) and key and key[0] == "ili"
        if prior is not None and (by_identity or (source and prior[1] and source != prior[1])):
            folded.setdefault(prior[0], []).append(cid)
            continue
        if prior is None:
            held[key] = (cid, source)
        out.append(row)
    return out, folded


def _cut_for(ranked: Sequence[Tuple[str, str, float]], *, query: str | None = None,
             store=None, frame=None) -> int:
    """How many of a ranked list are the answer — `relevance_cut` with its frame.

    `relevance_cut` takes a `frame` and gives a sharper answer when it has one: without it the
    adaptive instrument has no coordinates to read and declines, so the cut silently falls back to
    `_knee` and every caller who forgot the argument gets the thin tier while believing they asked
    for the sharp one. That is exactly what happened the first time mantle's recall called the cut
    — measured on the live corpus, four of six queries cut to 50 of 50, which is no cut at all.
    The frame is derivable from the ranked list and the store, so deriving it here is what stops
    the argument from being forgettable.

    The frame is spanned by the SYNSET candidates in the ranking — `wn-` ids, whose names are
    coordinates the projection can place. A ranked list with none of them yields no frame, and the
    cut is `_knee` over the scores, which is the honest thin-tier answer rather than a failure.

    `frame=` still wins when a caller already holds one: building it twice would read the same
    coordinates twice for the same answer.
    """
    scores = [s for _i, _c, s in ranked]
    if not scores:
        return 0
    if frame is None and store is not None:
        names = [str(i)[3:] for i, _c, _s in ranked if str(i).startswith("wn-")]
        if names:
            try:
                frame = _resolve_seam("projection").frame(store, names)
            except Exception:
                frame = None
    try:
        return int(_relevance_cut(scores, query=query, frame=frame))
    except Exception:
        # A cut that cannot be read is not a reason to answer with nothing: the ranking above it
        # ran and is the order. Keeping every candidate says "this could not be cut" in the only
        # way the return type has, and no caller loses a result it earned.
        return len(scores)


# ── the published names ──────────────────────────────────────────────────────────────────────────
# The underscored originals stay as the implementation and these are what a caller reaches for, so
# `sage.content_search` can keep its own private names pointing here without either file pretending
# the other's spelling is its own.
knee = _knee
relevance_cut = _relevance_cut
rank = _reach_rank
cut_for = _cut_for
