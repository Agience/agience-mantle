"""Surface forms -> concepts: the store-side LOOKUP half of matching.

Given a word, which concepts could it name? Given a concept, what is near it? These read the corpus
through the ontology driver and the lattice graph, and they compute nothing that needs the signal.

⚠ SPLIT OUT OF `ember/ontology/match.py` — 2026-08-02, the chorus->ember DAG work. `match` was one
module doing two jobs, and the seam is sharp once measured: these seven functions reach the STORE and
the DRIVER and never touch `beam`, while the half left behind (`propagate`, `fired_field`,
`signal_offers`, `tekton_basis`, the geometry derivations) needs beam's resolution/propagation and so
cannot live below it.

That asymmetry is the whole reason for the split. mantle is beam's SIBLING in the DAG — it may not
import beam — so the beam-touching half had to stay in ember, which is the only layer permitted to
reach both. But the LOOKUP half has no such constraint, and it was the half the personas actually
wanted: `wn_synsets_for` alone accounted for six `chorus -> ember` import sites, every one of them
asking the corpus a question the store can answer by itself.

⛔ THE CUT IS BY DEPENDENCY, NOT BY THEME, and it was computed rather than judged: the transitive call
closure of each function was checked against every `beam.` reference in the module. `related` brought
`hop_cost` with it; `_offers` brought `offer_synsets`; `wn_synsets_for` and `invalidate` came alone.
Nothing here calls anything that stayed. Splitting on what a reader thinks the functions are ABOUT
would have cut straight through `fired_field`, which is lookup and physics in one body.

⚠ `_WORD` LIVES HERE AND EMBER IMPORTS IT BACK. It is a two-line regex, so copying it would have been
the easy thing; it is single-homed because the tokenisation that decides which words become offers is
a measurement, and two copies of a measurement drift silently. `ember.ontology.match` imports this
module for it, and for the five functions its remaining half still calls.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from prism import law as _law

from mantle.ontology import driver as wn

# ⛔ A CACHE MOVES WITH THE FUNCTION THAT OWNS IT. `related` fills `_ASSOC_CACHE`, so it lives here.
# `_OFFER_CACHE` did NOT come: `_offers` stayed in ember (it reaches `stats.list_by_content_type`,
# which is still up there), and a cache split from its writer is two dicts — one filled, the other
# cleared — so stale offers would survive every invalidation, silently.
_ASSOC_CACHE: Dict[int, Dict[Tuple[str, str, str], float]] = {}

#: Tokeniser for offer text. ONE home — `ember.ontology.match.fired_field` imports it from here.
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def offer_synsets(text: str) -> List[str]:
    """The synsets an offer names — its POSITION in meaning-space, not a pooled direction.

    Kept as a SET OF NODES rather than one vector precisely so real geodesic distance stays
    measurable per node. Most-frequent-sense only (`[:1]`): an offer is a description, and taking
    every sense of every noun smears an offer across the ontology, which is what let
    "health status disk memory" sit near "python function".

    ⚠ NOUNS ONLY, inherited from the coordinate — verbs and adjectives carry no hypernym tree, so
    "describes markdown documents" contributes `markdown`/`document` and nothing from "describes"."""
    from mantle.ontology import driver as wn
    if not text or not str(text).strip():
        return []
    out, seen = [], set()
    for tok in _WORD.findall(str(text).lower()):
        senses = wn.synsets(tok, pos=wn.NOUN)
        if not senses:
            continue
        name = senses[0].name()
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _wn_prefix(surface: str) -> bool:
    """Does the lexicon hold any entry starting with this underscore surface? (`wn_store`'s keyed
    range probe — the stopping rule for longest-match, in place of a maximum compound length.)"""
    from mantle.ontology import driver as wn
    try:
        return wn.entry_prefix_exists(surface)
    except Exception:
        return False


def wn_synsets_for(word: str) -> List[str]:
    """The synset names a word resolves to, most-frequent-sense first (the same walk
    `offer_synsets` uses, exposed so weighting can pair node -> word).

    ⛔ DROP SINGLE-CHARACTER-LEMMA MATCHES. `wn.synsets` runs morphy, so the function word "is"
    stems to the letter "i" and matches iodine's chemical symbol "I" (also one.n.01, i.n.03), and
    "a" matches angstrom/ampere. A natural-language query word never MEANS a bare symbol, so a match
    that connects ONLY through a length-1 lemma is spurious — it is exactly what put "iodine" atop
    "what is a glacier". Keep a synset only when the word matches a MULTI-CHAR lemma (exact, or via
    the noun morphy form). Not a stop-list: real content words keep every legitimate sense; only the
    symbol/letter mismatch is removed."""
    from mantle.ontology import driver as wn
    w = (word or "").strip().lower()
    if len(w) <= 1:
        return []                                        # a bare letter is never a query subject
    res: List[str] = []
    try:
        m = wn.morphy(w, wn.NOUN)
    except Exception:
        m = None
    for x in wn.synsets(w, pos=wn.NOUN):
        lemset = {l.name().lower() for l in x.lemmas()}
        if w in lemset or (m and len(m) > 1 and m in lemset):
            res.append(x.name())
    return res


def hop_cost(store, src: str, dst: str, label: str) -> Optional[float]:
    """The ambiguity, in nats, of following `label` from `src` to `dst` — measured BOTH WAYS.

        d = log(out-degree(src, label)) + log(in-degree(dst, label))

    ⛔ THE OUT-DEGREE ALONE WAS NOT ENOUGH, and the failure was immediate and measurable. A link
    that is the ONLY one of its kind leaving a node scored 0.000 nats, which is no attenuation at
    all — so the neighbour fired exactly as strongly as the node itself. MEASURED live: `star
    --domain_topic--> astronomy` cost 0, and `photosynthesis` then answered with **flora** ahead of
    photosynthesis, while `mass physics` pulled the Eucharist back in. A hop that costs nothing is
    not a hop; it is an identity.

    The missing half is the TARGET's side. `astronomy` is the domain topic of hundreds of terms, so
    arriving there tells you very little — many things point at it. An edge is informative only when
    it is rare in BOTH directions: few ways out of the source AND few ways in to the target. Both
    are counted from the graph, in nats, and nothing is chosen."""
    # ⛔ RETURNED 0.0 ON ANY DB ERROR, AND CACHED IT. `except: out = inn = 1` makes the cost
    # `log(1) + log(1)` = 0.0 — a FREE hop, i.e. maximally informative, the strongest possible
    # claim about an edge whose degrees could not be counted. It was then memoised in
    # `_ASSOC_CACHE` for the process lifetime, so ONE transient fault (a locked DB, a closed
    # connection, the count(*) shape that zombies node 71) permanently teleported the reach
    # through that edge. An edge with no measured cost has no place in a measured propagation:
    # return None and let `related` drop it. Failures are NOT cached, so a transient fault is
    # retried rather than frozen in.
    conn = store.artifacts.db.read()
    cache = _ASSOC_CACHE.setdefault(id(store), {})
    ck = (src, dst, label)
    if ck not in cache:
        try:
            out = conn.execute("SELECT count(dst) FROM edge WHERE src = ? AND label = ?",
                               (src, label)).fetchone()[0]
            inn = conn.execute("SELECT count(src) FROM edge WHERE dst = ? AND label = ?",
                               (dst, label)).fetchone()[0]
        except Exception:
            return None
        if not out or not inn:
            return None          # an edge the graph does not attest in both directions
        cache[ck] = math.log(max(1.0, float(out))) + math.log(max(1.0, float(inn)))
    return cache[ck]


def related(store, name: str, *, exclude: Tuple[str, ...] = ("hypernym", "instance_of",
                                                             "instance_hypernym")) -> List[Tuple[str, float]]:
    """One associative hop from `name`: `[(target, distance_in_nats)]`.

    IS-A is excluded because `jc_tree` already travels it — including it here would count the tree
    twice. Everything else the source named is available."""
    vid = name if name.startswith("wn-") else "wn-" + name
    try:
        conn = store.artifacts.db.read()
        rows = conn.execute("SELECT dst, label FROM edge WHERE src = ?", (vid,)).fetchall()
    except Exception:
        return []
    out: List[Tuple[str, float]] = []
    for dst, label in rows:
        if label in exclude:
            continue
        d = hop_cost(store, vid, str(dst), label)
        if d is None:
            continue        # cost unmeasurable -> the edge is not traversed, never traversed free
        out.append((str(dst)[3:] if str(dst).startswith("wn-") else str(dst), d))
    return out



