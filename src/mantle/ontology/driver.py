"""The ONTOLOGY DRIVER — the reasoning layer's structure, read from OUR corpus, not from a package.

⭐ THIS FILE WAS `ember/ontology/wn_store.py` UNTIL 2026-08-02, AND THE RENAME IS THE POINT.
[John: *"There shouldn't be a separate module for wordnet I don't think."*] The behaviour below already
agreed: the id prefix, content type, POS alphabet and is-a edge labels are read from the stored
`language:*` transducer spec, with the WordNet values only as DEFAULTS (`_ONTOLOGY_DEFAULTS`), and this
header has long claimed that "a second ontology drops in as DATA". A module named for one lexicon
contradicted its own contract, and a filename is the first thing a reader believes. What is served here
is an ONTOLOGY; WordNet is the dataset this corpus happens to hold.

⚠ STILL ONLY HALF-TRUE — said plainly so the rename is not read as the job being done: the WRITER
(`seed_lattice.build`, phase 5) does not emit those vocabulary keys yet, so every corpus today takes
the WordNet defaults. Making them data-driven end to end is a FEATURE, not a move; tracked in
`_scratch/ONTOLOGY-TO-MANTLE.md` §5.

⚠ MOVED FROM EMBER because it reads the STORE (9 sites below). ember is the RUNNER; the personas needed
this to ask the corpus what it knows, and reaching it through the runner was the single largest source
of `chorus → ember`. mantle sits below both, so both reach it downward. It did NOT go to beam: beam is
signal-only and is mantle's SIBLING, so a store-backed driver there would create `beam → mantle` and
destroy the "beam is signal only" target the dependency audit records as achieved. See
`mantle/ontology/__init__.py` for the driver-vs-computation seam this establishes.

Concept structure lives in the corpus as artifacts. scripts/enrich_wordnet.py materialized that
STRUCTURE onto them — `hypernyms` (the IS-A graph), `instance_hypernyms`, `ic` (Resnik information
content), `lemma_counts` (SemCor sense-frequency) — alongside the gloss/lemmas/pos they already carried.
So everything the geometry / activation / templates / forgetting layers need is in the store, and this
module serves it, replacing `from nltk.corpus import wordnet`.

Drop-in: `from mantle.ontology import driver as wn` — call sites still spell it `wn`, so `wn.synset(…)`
and `wn.NOUN` read exactly as they always did. It reproduces the small slice of the nltk WordNet API this
codebase actually uses:  NOUN/VERB/… constants · synset(name) · synsets(word, pos) · morphy(word, pos) ·
Synset.name()/pos()/ic()/hypernyms()/instance_hypernyms()/lemmas()/common_hypernyms() · Lemma.name()/count().

The index is loaded ONCE per process from the store (WordNet is static, so it never needs reloading)
and held as a module singleton — first access pays the load, everything after is in-memory.

⛔ THE STORE IS A PARAMETER, NOT AN AMBIENT (fixed 2026-07-30)
--------------------------------------------------------------
`_arts()` used to be `local_store.open_store().artifacts` — unconditionally, with no way to say
otherwise. So EVERY caller read ember's process-default store no matter which store it was handed:
a persona given store B, asking B's ontology a question, got an answer measured on store A and no
error anywhere. This is a measurement, and a measurement whose instrument is chosen by an
environment variable is not parameterised at all.

The store now resolves in one order — **explicit argument → bound store (`bind`) → the process
default, LOUDLY**. The fallback is kept because a script that legitimately has no store (the
enrichment drains, `serve`'s warm-up) is a real caller, but it now says so through `logging` once
per process instead of being indistinguishable from a correct call.

⚠ THE INDEX IS A PROCESS SINGLETON, SO THE MODULE IS COHERENT WITH ONE STORE AT A TIME. Threading
a `store=` argument through while keeping one global cache would have replaced a silent
wrong-store read with a silent first-store-wins read — the same defect wearing a parameter. So the
resolved store is REMEMBERED (`_SOURCE`), and handing in a different one INVALIDATES every cache
and says so. Hand it store B and it reads B.

⭐ THE ONTOLOGY'S OWN VOCABULARY IS DATA (2026-07-30). The id prefix, content type, POS alphabet
and IS-A edge labels are read from the `language:*` transducer spec the same way `morphy`'s
exceptions already are (`_lang_spec`), so a second ontology drops in as DATA. Where the spec is
silent the WordNet values below apply unchanged — see `_ONTOLOGY_DEFAULTS`.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# POS constants — same letters WordNet/nltk use (satellite adjectives are "s"). These stay module
# constants because they are the drop-in nltk API surface callers spell as `wn.NOUN`; what varies
# per-ontology is the ALPHABET AND ITS ORDER, which `_pos_order()` reads from the spec.
NOUN, VERB, ADJ, ADV, ADJ_SAT = "n", "v", "a", "r", "s"
_ALL_POS = (NOUN, VERB, ADJ, ADV, ADJ_SAT)

# ── the ontology's vocabulary, when the spec does not carry it ────────────────────────────────
# ⚠ These are DEFAULTS, not constants: they describe WordNet, which is the one ontology in the
# corpus today and the only shape the stored `language:en` spec was written for. They are the
# values this module hardcoded before, kept byte-identical so an existing corpus is unaffected.
# A spec that names any of them wins — that is the seam a second ontology enters through, and it
# needs no code change on this side. The WRITER (`seed_lattice.build`, phase 5) does not emit
# these keys yet; until it does every corpus takes the defaults. See `_scratch/ONTOLOGY-TO-MANTLE.md`.
_ONTOLOGY_DEFAULTS: Dict[str, object] = {
    "id_prefix": "wn-",                 # `wn-dog.n.01` — the artifact id for synset `dog.n.01`
    "content_type": "text/x-wordnet",   # the discriminator `list_artifacts` filters the build on
    "pos_order": list(_ALL_POS),        # nltk's `synsets()` order: noun before verb before adj…
    "isa_labels": ["hypernym"],                     # the edges that MEAN is-a
    "instance_labels": ["instance_of", "instance_hypernym"],   # …and its instance form
    # ⭐ `entry_label` IS ALREADY WRITTEN by `seed_lattice.build` (phase 5, `"entry_label": label`),
    # so this one is live data today, not a default waiting on a writer.
    "entry_label": "lex:en",            # `lemma:<surface> --lex:en--> wn-<synset>` — the entry edge
    "lemma_prefix": "lemma:",           # the entry edge's source-node prefix
}

_LOCK = threading.Lock()
_INDEX: Optional[Tuple[Dict[str, "Synset"], Dict[Tuple[str, str], List[str]]]] = None
_IC_STATS: Optional[Dict[str, int]] = None
# (surface form, pos) -> base lemma. Filled from the source's own `forms` at index load.
_EXC: Dict[Tuple[str, str], str] = {}

# ── WHAT THE STORED `ic` WAS MEASURED FROM ────────────────────────────────────────────────────
# The corpus's own record of its IC measurement, kept as an ARTIFACT (the shape
# `projection.BASIS_ID` already uses for the corpus basis) rather than as a field repeated on
# 676,225 synsets: it describes ONE measurement taken over the whole corpus, so it is one row.
#
# ⛔ IT EXISTS BECAUSE A NUMBER THAT CANNOT SAY WHAT IT WAS MEASURED FROM IS NOT A MEASUREMENT.
# Intrinsic IC is `1 - log(desc+1)/log(N+1)`, so it is only meaningful beside the `N` it was taken
# against; without that, a corpus that has GROWN carries stale IC that still reads as current, and
# a source added later gets no IC at all while nothing reports a hole. See the derivation gate in
# `_load_index` for the measured failure this cost.
IC_BASIS_ID = "geom.ic-basis"
IC_BASIS_CT = "application/vnd.agience.ic-basis+json"
INTRINSIC_IC_SOURCE = "intrinsic"
INTRINSIC_IC_FORMULA = "1 - log(desc(s)+1)/log(N+1)"     # Seco et al., over the is-a tree


def ic_basis(store=None) -> Dict[str, object]:
    """The corpus's record of what its stored `ic` was measured from — `{}` when it carries none.

    `{"source": "intrinsic", "n": <corpus size>, "formula": ...}` for the tree-derived measurement;
    a corpus enriched from an external frequency corpus records its own `source` and is then left
    alone by the derivation (see `_load_index`)."""
    try:
        doc = _observe(store).get_artifact(IC_BASIS_ID) or {}
    except Exception:
        return {}
    b = doc.get("basis")
    return dict(b) if isinstance(b, dict) else {}


def derive_intrinsic_ic(idx: Dict[str, "Synset"]) -> Dict[str, float]:
    """Intrinsic IC (Seco et al.) for every synset in `idx`, read off the is-a tree `idx` carries:

        IC(s) = 1 - log(|descendants(s)| + 1) / log(N + 1)

    A synset with many descendants is general (low IC); a leaf is maximally specific (IC -> 1).
    Nothing is imported and nothing is chosen — the structure IS the measurement.

    ⚠ `N` IS `len(idx)`, SO THIS IS A FUNCTION OF THE WHOLE CORPUS, NOT OF ONE SYNSET. Two synsets
    whose IC was derived against different `N` are on different scales and must not be compared;
    that is exactly what `ic_basis` records and what makes the re-derivation trigger honest.
    Factored out of `_load_index` so the READER and the WRITER (`seed_lattice`) share one
    derivation rather than two implementations that can drift apart."""
    import math as _math
    desc: Dict[str, int] = {}
    for _n_ in idx:
        seen_up, stack = set(), list(idx[_n_]._hyper) + list(idx[_n_]._inst)
        while stack:                       # descendants, not children: credit every ancestor once
            q = stack.pop()
            if q in seen_up or q not in idx:
                continue
            seen_up.add(q)
            desc[q] = desc.get(q, 0) + 1
            stack.extend(list(idx[q]._hyper) + list(idx[q]._inst))
    _logN = _math.log(float(len(idx)) + 1.0)
    return {n: 1.0 - (_math.log(desc.get(n, 0) + 1.0) / _logN) for n in idx}


def ic_coverage(*, store=None) -> Dict[str, object]:
    """How much of the loaded WordNet actually carries information content.

    The geometry, the category layer and the forgetting curve all compute over IC and all treat a
    missing value as 0.0. Without this, a corpus that was never enriched — or one caught mid-run,
    since the enrichment drain is keyed on `ic IS NULL` and its `--force` path REMOVEs `ic`
    corpus-wide before rewriting — is indistinguishable from a corpus of genuinely zero-IC
    synsets, and every metric derived from it reads as a measurement. Report this next to any
    IC-derived number rather than assuming the enrichment ran."""
    _index(store)                               # ensure loaded
    st = dict(_IC_STATS or {"synsets": 0, "with_ic": 0, "without_ic": 0,
                            "with_ic_se": 0, "without_ic_se": 0})
    n = int(st.get("synsets") or 0)
    st["coverage"] = round(int(st.get("with_ic") or 0) / n, 4) if n else None
    # Reported SEPARATELY, not folded into `coverage`: `ic` and `ic_se` are two channels and a
    # corpus routinely has full coverage of the first and none of the second. One number averaging
    # them would report a half-enriched corpus as half-covered in BOTH, which is false in both
    # directions.
    st["se_coverage"] = round(int(st.get("with_ic_se") or 0) / n, 4) if n else None
    return st


class WordNetError(Exception):
    """Raised for an unknown synset name (mirrors nltk.corpus.reader.wordnet.WordNetError)."""


class RankUnavailable(WordNetError):
    """Raised when a word's senses carry no stored `rank`, so they cannot be ordered by sense
    frequency. Sorting them anyway falls through to the synset NAME and returns an ALPHABETICAL
    list that still presents as sense-rank order — the measured "dog -> the fireplace andiron"
    failure. An ordering the source did not supply is not an ordering."""


class _Lemma:
    """A word-sense pairing. Only .name() and .count() (SemCor frequency) are used by the reasoning."""
    __slots__ = ("_name", "_count")

    def __init__(self, name: str, count: int):
        self._name = name
        self._count = count

    def name(self) -> str:
        return self._name

    def count(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return f"Lemma('{self._name}')"


class Synset:
    """A WordNet sense, backed by its enriched `wn-*` artifact. Immutable; identified by name.

    ⭐ IT CARRIES THE STORE IT WAS READ FROM, and that is the whole point of `_store`. The
    parameterisation fix (2026-07-30, `test_wn_store_parameterised.py`) threaded `store=` through
    `synsets()` / `synset()` / `_index()` and stopped there — at the module's ENTRY points. But a
    synset is not a leaf: the moment anything walks the taxonomy (`hypernyms()`, `_closure()`,
    `common_hypernyms()`, and therefore every `geometry.tree_path` / `jc_tree` / `canonical_parent`
    above them) the object had no store to resolve through and fell back to the ambient bind. So a
    read that was correctly handed store B measured B's first hop and A's entire ancestry — the
    exact defect that file's docstring describes, surviving one level down from where it was fixed.

    ⛔ AND THE FALLBACK IS NOT MERELY WRONG, IT BLOCKS. `_resolve(name)` with no store reaches
    `_arts(None)` -> `open_store()` -> `open_lattice()` -> `ensure_schema`, which takes a
    `BEGIN IMMEDIATE` write lock on whatever the process default is. Under a unit test that is the
    LIVE 5.7 GB lattice, and while the services are up it does not fail — it hangs, and the suite
    reports a timeout with no failing assertion. Measured 2026-08-01: the ember suite stalled inside
    `geometry.tree_path` -> `canonical_parent` -> `hypernyms()` for exactly this reason.

    `_store` is None for a synset built without one, which is the pre-existing ambient behaviour
    unchanged — the fix removes a silent substitution, it does not add a requirement."""
    __slots__ = ("_name", "_pos", "_hyper", "_inst", "_ic", "_ic_se", "_counts", "_store")

    def __init__(self, name: str, pos: str, hyper: List[str], inst: List[str],
                 ic: float, counts: Dict[str, int], ic_se: Optional[float] = None, store=None):
        self._name = name
        self._pos = pos
        self._hyper = hyper
        self._inst = inst
        self._ic = ic
        self._ic_se = ic_se
        self._counts = counts
        self._store = store

    def name(self) -> str:
        return self._name

    def pos(self) -> str:
        return self._pos

    def ic(self) -> float:
        """Resnik information content (stored — derived once from the Brown corpus during enrichment).

        Returns 0.0 when the field is ABSENT, because every downstream consumer does arithmetic
        with this. Use `has_ic()` to tell the two apart — see the class note in `_load_index`."""
        return self._ic if self._ic is not None else 0.0

    def ic_se(self) -> Optional[float]:
        """The SECOND CHANNEL: standard error of this synset's IC, in nats, or **None**.

        Written by the Laplace-smoothing enrichment path (`geometry.smooth_ic`) as
        `sqrt((1 - p) / cum')`. Unlike `ic()` this returns **None**, not 0.0, when absent — see
        `geometry.ic_se_of`. 0.0 would assert "this IC is exact", which is a claim, whereas None
        says "unknown", which is the truth for any corpus enriched before `ic_se` existed. Callers
        do NOT do unconditional arithmetic with an uncertainty the way they do with a value, so the
        `ic()`-style 0.0 convention buys nothing here and costs the distinction.

        ⛔ NEVER FOLD THIS INTO `ic()`. See `geometry.jc_tree_se` for the measurement (7.1e-15
        carrying it beside the value, 5.99 perturbing the value with it)."""
        return self._ic_se

    def has_ic_se(self) -> bool:
        """Was a standard error actually STORED? (`ic_se() is not None`, spelled for symmetry
        with `has_ic()`.)"""
        return self._ic_se is not None

    def has_ic(self) -> bool:
        """Was information content actually STORED for this synset?

        ⛔ WITHOUT THIS, "never enriched" AND "genuinely zero" ARE THE SAME NUMBER, and that one
        ambiguity propagates into three subsystems that each then report maximum confidence from
        an absence of data:
          * `geometry.faithfulness_check` — all-zero IC makes every `jc_tree` 0, so the series is
            constant and (before the round-9 fix) Spearman came out exactly 1.0 alongside
            `prehash_max_abs_err: 0.0`, i.e. "the exactness claim" for a measurement that never
            happened.
          * `geometry.canonical_parent` — picks `max(parents, key=(ic, name))`, so when every
            parent reads 0.0 the spanning-tree parent is chosen ALPHABETICALLY and every JC
            distance is computed over an arbitrary tree and returned as a measurement.
          * `forgetting._sim` — `exp(-jc_tree(...))` becomes 1.0 for EVERY concept pair.
        The value stays 0.0 for arithmetic; what changes is that the absence is now knowable and
        countable (`ic_coverage()`)."""
        return self._ic is not None

    def hypernyms(self) -> List["Synset"]:
        # The store THIS synset came from — never the ambient default. See the class note.
        return [s for s in (_resolve(n, self._store) for n in self._hyper) if s is not None]

    def instance_hypernyms(self) -> List["Synset"]:
        return [s for s in (_resolve(n, self._store) for n in self._inst) if s is not None]

    def lemmas(self) -> List[_Lemma]:
        return [_Lemma(n, c) for n, c in self._counts.items()]

    def _closure(self) -> Dict[str, "Synset"]:
        """This synset plus ALL its ancestors (transitive hypernyms + instance-hypernyms), by name.
        Matches nltk's _all_hypernyms (which includes the synset itself)."""
        seen: Dict[str, "Synset"] = {}
        stack = [self]
        while stack:
            s = stack.pop()
            if s._name in seen:
                continue
            seen[s._name] = s
            stack.extend(s.hypernyms())
            stack.extend(s.instance_hypernyms())
        return seen

    def common_hypernyms(self, other: "Synset") -> List["Synset"]:
        a = self._closure()
        b = other._closure()
        return [a[n] for n in a if n in b]

    def __repr__(self) -> str:
        return f"Synset('{self._name}')"

    def __eq__(self, other) -> bool:
        return isinstance(other, Synset) and other._name == self._name

    def __hash__(self) -> int:
        return hash(self._name)


# ══ KEYED LAZY PATH ═══════════════════════════════════════════════════════════════════════════════
# The whole-corpus `_load_index` below is the BUILD path — it reads all 555k synsets and both the
# 460 MB doc scan and the (unindexed-label) 42 s hypernym-edge scan to materialize one in-memory index.
# The CHAT path needs none of that: a synset is one keyed `get_artifact`, its hypernyms one keyed
# `edges WHERE src=?` (ix_e_src), its IC a stored value on the doc, and word→senses a keyed reverse
# `lex:en` edge lookup — the language transducer's entry. This section serves all of that lazily, per name,
# and caches. It activates only once the substrate exists (`op.transducer.language.en` written by the
# tekton, after the `lex:en` edges and stored IC); until then every accessor uses the legacy full load
# unchanged, so nothing regresses before the build runs.
import json as _json

# name -> (generation it was verified at, its freshness stamp, the Synset). The stamp is what
# makes a repair to the corpus reach a RUNNING process; see `_gate` and `ontology/freshness.py`.
_CACHE: Dict[str, Tuple[int, object, "Synset"]] = {}
_KEYED_READY: Optional[bool] = None
_LANG_SPEC: Optional[Dict[str, object]] = None
from prism.grounding import TRANSDUCER_OP      # op-id prefix — `op.transducer.` since the 2026-07-30 migration
_TRANSDUCER_ID = TRANSDUCER_OP + "language.en"


_SOURCE = None            # the artifacts face every cache below was built from — see `_arts`
_AMBIENT_WARNED = False

# ── FRESHNESS: the store's own write bookkeeping, so a data change reaches this process ───────
# ⛔ WITHOUT THIS, A REPAIR DOES NOT LAND. MEASURED 2026-08-01 on 71: 141,102 stored `ic` values
# were rewritten and the running lumen/ember went on serving the pre-repair coordinates, because
# every cache in this module is a process singleton that was built once and never asked again. The
# remedy on offer was "restart it" — the operator procedure PLAN §0.0 exists to forbid. The full
# argument, the three shapes this deliberately is NOT (a TTL, a refresh endpoint, a "call
# invalidate() after you write"), and the measured costs are in `ontology/freshness.py`.
from mantle.ontology import freshness as _freshness

_MARK: object = None      # the store WRITE MARK every cache here was last verified against
_GEN: int = 0             # bumped on every MOVE of that mark — see `generation()`
_INDEX_GEN: int = -1      # the generation `_INDEX` / `_IC_STATS` / `_EXC` were built at
_INDEX_FOREIGN = False    # …unless the index was INSTALLED, not read — see `install_index`
_SPEC_GEN: int = -1       # the generation `_KEYED_READY` / `_LANG_SPEC` were verified at
_SPEC_STAMP: object = None    # …and the transducer artifact's stamp they were read from


def generation() -> int:
    """How many times this module has SEEN the store change under it.

    ⭐ THE HANDLE A DERIVED CACHE HANGS ON, and the reason it is a counter rather than a callback:
    a cache built on `wn_store`'s output (`geometry._DENSE_CACHE`) cannot ask the store anything —
    `dense_vec(synset, ic)` is handed an object, not a store — but it CAN key itself on the
    generation the object was served under, and then a stale entry is not detected, it is
    unreachable. That is the same move `_DENSE_CACHE`'s own comment already argued for with
    `id(ic)` ("a genuinely new table is a new object, so the cache misses BY CONSTRUCTION rather
    than by a version number someone has to remember to bump") — pointed at the identity that
    actually varies now that IC rides on the synset and `load_ic()` returns a constant `None`.

    It advances ONLY when the store's write mark moved, i.e. only when something was really
    written. It is not a clock and not a counter of reads: on an idle store it never moves, which
    is exactly what lets the caches below stay caches. MEASURED on 71 with all four services live:
    0 moves in 80 samples over 20 s."""
    return _GEN


def _gate(a) -> None:
    """OBSERVE THE STORE: poll its write mark, and if it MOVED, mark every cache here as needing
    re-verification.

    ⛔ WHERE THIS IS CALLED FROM IS PART OF THE DESIGN, AND THE OBVIOUS PLACE WAS WRONG. It was
    first hung on `_arts()` — the one funnel every accessor already passes through, which looked
    like the answer precisely because nobody could forget it. MEASURED: `_arts` runs THREE times
    per warm `synset()` and **71 times** per warm `tree_path()` over a 9-node path, so a ~6 µs
    poll turned a 0.57 µs lookup into 56 µs and a 20 µs path walk into ~430 µs. That is not a
    tuning complaint. Re-deriving a coordinate from scratch costs 37.6 µs, so paying 430 µs to
    check whether the cached one is still good is a cache that costs more than no cache — the
    "fixed it by disabling caching" outcome, arrived at while every test still passed.

    ⭐ SO THE GATE SITS ON THE MODULE'S PUBLIC BOUNDARY, ONCE PER CALL IN. `synset`, `synsets`,
    `morphy`, `all_synsets`, `ic_coverage` — every way into this module from outside observes the
    store on the way in. Internal recursion does NOT: `Synset.hypernyms()` resolves ancestors
    through the private `_resolve`, which is why a 9-node `tree_path` costs ZERO extra polls while
    a caller asking a fresh question always pays exactly one (14.1 µs against a 37.6 µs coordinate
    rebuild and a 370 µs synset rebuild; MEASURED warm `tree_path` 12.4 µs, unchanged). The paths that were going to read the store anyway —
    `_conn()`, the `_get_synset` miss, the transducer read, `_load_index` — observe it too, which
    costs nothing beside what they already spend.

    ⛔ **AN EARLIER PLACEMENT WAS TOO SPARSE AND THE MEASUREMENT CAUGHT IT — NOT THE SUITE.** The
    gate was first put only on the store-touching paths, on the argument that "every turn resolves
    its seeds through `_entry_names`, which always hits `_conn()`". That is true of `synsets(word)`
    and FALSE of `synset(name)`, which `projection.frame` calls directly for every row of a screen:
    a warm `synset(name)` touches nothing at all, so a repair written by another process was still
    being served stale. The whole ember suite passed in that state. What found it was writing the
    repair from a SEPARATE PROCESS and asserting the new value came back — which is why that is
    the measurement of record and not a unit test with a mocked store.

    ⚠ THE FAILURE MODE, STATED ([[verification-that-cannot-fail]]): a process that never calls into
    this module again never learns the store changed. That is not a hole, it is the definition —
    such a process is not answering anything. The bound is exactly "one call into `wn_store`", and
    nothing can be served from these caches without one.

    ⭐ THE CHEAP HALF IS THE NEGATIVE READING. An unchanged mark proves that nothing was written to
    this store, hence that every derivation built from it is still the derivation of what is in it
    — no per-entry checks, no artifact reads, nothing. MEASURED on 71's 5.7 GB lattice: **14.1 µs**,
    against 370 µs to rebuild one Synset and 25.4 s to rebuild `_INDEX`. This is the whole reason
    the caches survive the hook.

    A mark that HAS moved says only that SOMETHING changed. Attribution is then per-cache and by
    its own dependency:

      · `_CACHE`, `_KEYED_READY`, `_LANG_SPEC` each derive from ONE named artifact, so each is
        re-verified against `freshness.stamp()` — LAZILY, on next touch, so an entry nobody asks
        for costs nothing and a write that changed something else drops nothing.
      · `_INDEX` / `_IC_STATS` / `_EXC` derive from the WHOLE `text/x-wordnet` population. There is
        no cheap exact stamp for that, so they are DROPPED. ⚠ That is conservative, not exact: an
        unrelated write discards them. It is the honest reading of a derivation whose input is most
        of the store, and it is affordable here because the keyed substrate is live on 71
        (`_keyed_ready()` is True), so nothing on the serving path loads the index — `all_synsets`
        is reached by the consolidation jobs (`projection.build_basis`, `match._derive_geometry`),
        never by a chat turn. If it ever returns to a hot path, the fix is a per-content-type write
        mark in mantle's writer beside the `ct:` counters it already maintains — NOT a longer
        window here.

    ⚠ A STORE THAT CANNOT REPORT A MARK IS RE-VERIFIED ON EVERY READ, deliberately: `write_mark`
    returns None, the generation advances every poll, and every cached entry is therefore
    unreachable. Caching is a claim that a value is still current; a store that cannot support that
    claim does not get to have it made on its behalf ([[absence-is-not-an-affirmative-claim]])."""
    global _MARK, _GEN, _INDEX, _IC_STATS
    m = _freshness.write_mark(a)
    if m is not None and m == _MARK:
        return
    _MARK = m
    _GEN += 1
    if _INDEX_FOREIGN:
        return                    # an INSTALLED index is not this store's derivation to drop
    if _INDEX is not None or _IC_STATS is not None:
        _INDEX = None
        _IC_STATS = None
        _EXC.clear()


def bind(store) -> None:
    """Bind the store this module reads. The explicit parameterisation the personas should use.

    `store` may be a `LocalStore` (anything with `.artifacts`) or an artifacts face directly.
    `bind(None)` releases the binding and returns the module to the ambient fallback."""
    _arts(store) if store is not None else _release()


def _release() -> None:
    global _SOURCE
    with _LOCK:
        _SOURCE = None
    _invalidate()


def invalidate() -> None:
    """PUBLIC: drop every cache, because the substrate underneath them changed.

    ⚠ THIS EXISTS SO NOTHING HAS TO REACH INTO THIS MODULE'S PRIVATES. `seed_lattice.build` — a
    PRODUCTION path — used to write `_wn._INDEX = None`, `_wn._CACHE.clear()`, `_wn._KEYED_READY`
    and `_wn._LANG_SPEC` directly, four separate assignments, once before the build and once after.
    That is a back door in the strict sense: a second way to change this module's state that its
    own invariants do not see. It also could not stay correct — every cache added here had to be
    remembered at both of those call sites, and the next one added would not be.

    Callers that legitimately change the substrate (an ingest, a transducer build) call this. There
    is nothing else to know."""
    _invalidate()


def install_index(idx: Dict[str, "Synset"], word_index: Dict[Tuple[str, str], List[str]],
                  *, ic_stats: Optional[Dict[str, int]] = None) -> None:
    """PUBLIC: install an index this module did NOT read from a store.

    ⚠ THE MISSING HALF OF `invalidate()`, AND IT EXISTS FOR THE SAME REASON. `invalidate` was
    written because `seed_lattice.build` assigned four privates by hand; the offline-WordNet test
    fixture assigns two more (`_INDEX`, `_IC_STATS`) from nltk. That back door was harmless while
    nothing else owned those names — and it stopped being harmless the moment the freshness gate
    did, because an index built from nltk has no relationship to the ambient store's write mark and
    the gate correctly dropped it, sending every geometric test into a 25.4 s reload of the live
    corpus. MEASURED: seven `test_match` assertions failed with `inf < inf` (an unbuilt index gives
    every pair infinite distance) — the exact failure the fixture's own comment already warns about.

    ⛔ AND `_INDEX_FOREIGN` IS NOT AN EXEMPTION FROM FRESHNESS, IT IS A STATEMENT OF FACT. This
    index is not a derivation of the store, so the store's write mark says nothing about whether it
    is current, and dropping it on a store write would be a non-sequitur — it was never fresh with
    respect to that store in the first place. Whoever installs an index owns its correctness.
    `bind()` / `bind(None)` still clear it, because those change WHICH substrate is being read.

    Nothing in `src/` calls this; the callers are `agience-ember/tests/_fakes.py::
    _install_offline_wordnet` and `agience-chorus/src/sage/tests/test_match.py`, which had
    independent copies of the same back door."""
    global _INDEX, _IC_STATS, _INDEX_FOREIGN
    _INDEX = (dict(idx), dict(word_index))
    _IC_STATS = dict(ic_stats) if ic_stats else None
    _INDEX_FOREIGN = True


def _invalidate(*, keep_installed: bool = False) -> None:
    """Drop every cache. Called when the store the caches were built from changes underneath them.

    ⚠ STILL HERE, AND STILL NARROWER THAN IT LOOKS. `_gate` makes a DATA change reach this module
    on its own; this remains the answer to a different event — the store itself being swapped or
    released (`bind`/`_release`), where the caches are not stale but simply belong to something
    else. It is also the explicit escape hatch for a caller that changed the substrate and does not
    want to wait for the next gate poll. The generation advances with it, so anything keyed on
    `generation()` misses too — dropping the caches here without that would have left
    `geometry._DENSE_CACHE` serving vectors for a store this module no longer reads.

    ⭐ `keep_installed` SEPARATES TWO EVENTS THAT LOOKED THE SAME (2026-08-01). Every derived cache
    here belongs to a store, so a store swap invalidates all of them — but an INSTALLED index
    belongs to NO store, which is `install_index`'s own stated reason for exempting it from
    freshness. Dropping it because some other store appeared in a call is the same non-sequitur one
    step further on: it was never a derivation of the store being left, so leaving that store says
    nothing about it.

    The distinction is between a caller SAYING which substrate to read (`bind` / `_release` —
    those still clear it, exactly as `install_index` documents) and a store merely APPEARING as an
    argument. Only the second is exempt, and the index it preserves is still only ever consulted for
    a store that cannot be asked for an ontology at all (`_index`). A store that CAN be asked always
    answers for itself, even when its honest answer is an empty ontology."""
    global _INDEX, _IC_STATS, _KEYED_READY, _LANG_SPEC, _MARK, _GEN, _SPEC_STAMP, _INDEX_FOREIGN
    if not (keep_installed and _INDEX_FOREIGN and _INDEX is not None):
        _INDEX_FOREIGN = False
        _INDEX = None
    _IC_STATS = None
    _KEYED_READY = None
    _LANG_SPEC = None
    _SPEC_STAMP = None
    _MARK = None
    _GEN += 1
    _CACHE.clear()
    _EXC.clear()


def _arts(store=None):
    """The artifacts face to read from: **explicit → bound → the process default, LOUDLY.**

    ⛔ THE LOUD FALLBACK IS THE POINT. Reaching `local_store.open_store()` silently is what made 18
    persona call sites read ember's process-default store instead of the store they were handed.
    A caller that genuinely has no store still works; it is now visible in the log that the
    instrument was chosen by the environment rather than by the caller.

    Adopting an explicitly-named store as `_SOURCE` is deliberate: the index below is a process
    singleton, so this module can only be coherent with one store at a time, and the store the
    caller just named is the one it must be coherent with. A DIFFERENT store INVALIDATES the caches
    rather than silently answering from the previous one's index — threading a `store=` argument
    through while keeping one cache keyed on nothing would just have swapped a wrong-store read for
    a first-store-wins read.

    ⚠ ONLY AN EXPLICIT STORE IS TRACKED; the ambient handle deliberately is not. `open_store()`
    CONSTRUCTS A FRESH `LocalStore` ON EVERY CALL, so comparing ambient handles by identity would
    find a "different store" on every single read and drop the 117k-synset index each time. An
    ambient-only process therefore behaves exactly as it did before this change."""
    global _SOURCE, _AMBIENT_WARNED
    a = getattr(store, "artifacts", None) or store
    if a is None:
        if _SOURCE is not None:
            return _SOURCE
        from mantle.shard.local_store import open_store
        if not _AMBIENT_WARNED:
            _AMBIENT_WARNED = True
            _log.warning(
                "wn_store: no store passed and none bound — reading the PROCESS-DEFAULT store. "
                "The ontology read is a MEASUREMENT; pass `store=` or call wn_store.bind(store) "
                "so it is measured on the store the caller actually holds.")
        return open_store().artifacts
    if a is not _SOURCE:
        # ⛔ THIS WARNED ON EVERY FIRST BIND OF EVERY PROCESS AND NOTHING WAS EVER LOST.
        # `a is not _SOURCE` is trivially true when `_SOURCE is None`, which is the state at the
        # first `bind()` — so the host logged "dropping the loaded index and caches" at every boot
        # while `_invalidate()` cleared six things that were all already empty. MEASURED 2026-07-31,
        # snapshot taken immediately before the call during a full 7-persona load:
        #
        #     _SOURCE=None _INDEX=None _CACHE=0 _EXC=0 _KEYED_READY=None _LANG_SPEC=None
        #
        # and `_arts` was called exactly ONCE in the entire boot — by `bind()` itself. There was no
        # earlier reader to be inconsistent with. A warning that cannot be true is the same defect as
        # a check that cannot fail: it costs nothing to emit, it reads as a real fault in the log,
        # and it trains the reader to ignore the line that will one day matter.
        #
        # The guard tests STORE IDENTITY; what it protects is CACHE VALIDITY. Those coincide only
        # when a cache exists. Nothing stale can be dropped when nothing has been built, so the
        # invalidation and the warning both belong behind that condition — and the genuine
        # cross-store thrash still warns, loudly, exactly as before.
        if _INDEX is not None or _CACHE or _EXC or _KEYED_READY is not None or _LANG_SPEC is not None:
            _log.warning("wn_store: reading a different store than the caches were built from — "
                         "dropping the loaded index and caches (was %s)",
                         "the process default" if _SOURCE is None else type(_SOURCE).__name__)
            # A store APPEARING in a call is not a caller SAYING which substrate to read, so an
            # installed index survives it — see `_invalidate`. Every store-derived cache still goes.
            _invalidate(keep_installed=True)
        _SOURCE = a
    return a


def _observe(store=None):
    """`_arts()`, plus the gate — the resolution to use when a STORE READ is about to happen.

    Two names for two different things, deliberately. `_arts` answers *which store*; this answers
    *which store, as it is right now*. Everything in this module that goes to disk goes through
    here, and everything that answers from a cache does not — which is what keeps the poll off the
    hot path without anyone having to remember where the hot path is."""
    a = _arts(store)
    _gate(a)
    return a


def _conn(store=None):
    """The raw read connection. Observes the store, because it is about to read it anyway."""
    return _observe(store).db.read()


def _spec_str(key: str, store=None) -> str:
    v = _lang_spec(store).get(key)
    return str(v) if isinstance(v, str) and v else str(_ONTOLOGY_DEFAULTS[key])


def _spec_list(key: str, store=None) -> List[str]:
    v = _lang_spec(store).get(key)
    if isinstance(v, (list, tuple)) and v:
        return [str(x) for x in v if x]
    return list(_ONTOLOGY_DEFAULTS[key])            # type: ignore[arg-type]


def _prefix(store=None) -> str:
    """The ontology's artifact-id prefix (`wn-`), from the spec."""
    return _spec_str("id_prefix", store)


def _ct(store=None) -> str:
    """The ontology's content type (`text/x-wordnet`), from the spec."""
    return _spec_str("content_type", store)


def _pos_order(store=None) -> List[str]:
    """The POS alphabet IN ORDER — noun before verb before … — from the spec.

    This is ontology knowledge, not ours: it is what makes `synsets()` return senses in nltk's
    order. A source with a different tag set names it and needs no code change here."""
    return _spec_list("pos_order", store)


def _isa_labels(store=None) -> Tuple[List[str], List[str]]:
    """`(is_a, instance_of)` edge labels, from the spec. §13.1: relations ARE edges, and the label
    that MEANS is-a is the SOURCE's word for it — `instance_hypernym` is OEWN's, `instance_of` the
    legacy corpus's. A reader that knew only one silently lost every instance edge on the other."""
    return _spec_list("isa_labels", store), _spec_list("instance_labels", store)


def _n(x: str, store=None) -> str:
    x = str(x or "")
    p = _prefix(store)
    return x[len(p):] if p and x.startswith(p) else x


def _spec_fresh(arts) -> None:
    """Re-verify `_KEYED_READY` / `_LANG_SPEC` against the transducer artifact, once per generation.

    Both derive from exactly ONE artifact, so the exact question is askable and cheap: has
    `op.transducer.language.en` itself been written since we read it? MEASURED 5.1 µs, and only
    after the gate has already reported a write. A changed stamp drops both; an unchanged one is a
    positive verification, not an assumption. `stamp() is None` means the store cannot answer, and
    that drops them too — see `freshness` on why unverifiable must not read as fresh.

    ⚠ THE SPEC IS WHAT SAYS *WHICH* ARTIFACTS THE ONTOLOGY EVEN IS (`id_prefix`, `content_type`,
    the is-a labels), so serving a stale one is not a stale field — it is reading the corpus with
    the wrong alphabet."""
    global _KEYED_READY, _LANG_SPEC, _SPEC_GEN, _SPEC_STAMP
    if _SPEC_GEN == _GEN:
        return
    st = _freshness.stamp(arts, _TRANSDUCER_ID)
    if st is None or st != _SPEC_STAMP:
        _KEYED_READY = None
        _LANG_SPEC = None
        _SPEC_STAMP = st
    _SPEC_GEN = _GEN


def _installed_index_answers(store) -> bool:
    """An INSTALLED index answers on its own, because there is no store to ask.

    ⭐ THIS IS A STATEMENT OF FACT, NOT AN EXEMPTION — the same argument `install_index` already
    makes about freshness, one question further on. An installed index is not a derivation of any
    store, so "is the keyed substrate ready?" and "observe the store before reading it" are not
    questions that apply to it. Asking them anyway reaches for a store NOBODY NAMED: no `store=`
    argument, no `bind()`, and a foreign index sitting right there holding the answer.

    ⛔ AND THE REACH IS NOT A WRONG ANSWER, IT IS A HANG. `_arts(None)` with nothing bound falls
    through to `open_store()` -> `open_lattice()` -> `ensure_schema`, which takes a `BEGIN IMMEDIATE`
    write lock on the process default. Under the offline-WordNet fixture that default is the live
    5.7 GB lattice, and while the node's services hold the lock it does not fail — it BLOCKS.
    MEASURED 2026-08-01: the ember suite stalled in `test_delegate_isolation.py::
    test_foreign_traces_are_never_read` and chorus in `match._derive_geometry`, both reported as
    bare timeouts with no failing assertion, which is what made the cause so hard to see.

    Deliberately narrow: it requires that NOTHING was passed AND nothing is bound. A caller that
    names a store gets that store, always — an installed index never overrides an explicit one."""
    return store is None and _SOURCE is None and _INDEX_FOREIGN and _INDEX is not None


def _keyed_ready(store=None) -> bool:
    """Is the keyed substrate present? The tekton writes the transducer artifact LAST — after the `lex:en`
    entry edges and the stored IC — so its presence is the single, keyed readiness flag."""
    global _KEYED_READY
    if _installed_index_answers(store):
        return False                         # an installed index is not the keyed substrate
    arts = _arts(store)                      # resolve FIRST: it may invalidate _KEYED_READY
    _spec_fresh(arts)                        # …and the transducer artifact may have been rewritten
    if _KEYED_READY is None:
        arts = _observe(store)               # going to the store: observe it while we are here
        with _LOCK:
            if _KEYED_READY is None:
                try:
                    _KEYED_READY = arts.get_artifact(_TRANSDUCER_ID) is not None
                except Exception:
                    _KEYED_READY = False
    return bool(_KEYED_READY)


def _lang_spec(store=None) -> Dict[str, object]:
    """The `language:<lang>` transducer artifact's spec — carried as DATA in the artifact (read once,
    keyed, cached). It carries morphy's exceptions, and (when the writer emits them) the ontology's
    own id prefix, content type, POS order and edge labels."""
    global _LANG_SPEC
    arts = _arts(store)                      # resolve FIRST: it may invalidate _LANG_SPEC
    _spec_fresh(arts)                        # …and the transducer artifact may have been rewritten
    if _LANG_SPEC is None:
        arts = _observe(store)               # going to the store: observe it while we are here
        try:
            doc = arts.get_artifact(_TRANSDUCER_ID) or {}
        except Exception:
            doc = {}
        _LANG_SPEC = dict((doc.get("spec") or {}))
    return _LANG_SPEC


def _synset_from_doc(name: str, doc: Dict[str, object], store=None) -> "Synset":
    """Build one Synset from its keyed artifact. Hypernyms come from keyed `edges WHERE src=?`
    (falling back to the doc field for corpora that still carry it); IC is the stored value."""
    pos = doc.get("pos") or (name.rsplit(".", 2)[-2] if name.count(".") >= 2 else NOUN)
    _c = doc.get("lemma_counts")
    counts = {str(k): int(v) for k, v in _c.items()} if isinstance(_c, dict) else {}
    if not counts:
        counts = {str(x): 0 for x in (doc.get("lemmas") or []) if x}
    _ic_raw = doc.get("ic")
    _ic = None if _ic_raw is None else float(_ic_raw)
    _se_raw = doc.get("ic_se")
    _se = None if _se_raw is None else float(_se_raw)
    hyper: List[str] = [_n(h, store) for h in (doc.get("hypernyms") or [])]
    inst: List[str] = [_n(h, store) for h in (doc.get("instance_hypernyms") or [])]
    if not hyper and not inst:
        isa, inst_labels = _isa_labels(store)
        labels = list(isa) + list(inst_labels)
        try:
            rows = _conn(store).execute(
                "SELECT dst, label FROM edge WHERE src=? AND label IN (%s)"
                % ",".join("?" * len(labels)),
                tuple([_prefix(store) + name] + labels)).fetchall()
        except Exception:
            rows = []
        for dst, label in rows:
            (inst if label in inst_labels else hyper).append(_n(dst, store))
    return Synset(str(name), str(pos), hyper, inst, _ic, counts, _se, store=store)


def _get_synset(name: str, store=None) -> Optional["Synset"]:
    """Keyed, lazy, cached read of one synset by name — VERIFIED against the artifact it came from.

    ⭐ THE VERIFICATION IS LAZY AND PER-ENTRY, WHICH IS WHAT MAKES IT AFFORDABLE. While the gate has
    not moved (`gen == _GEN`) this is the old dict hit and nothing more: MEASURED 0.57 µs. Only
    after a write does the stamp get re-read, and only for entries someone actually asks for —
    MEASURED 18.6 µs for both halves of the stamp against **370 µs** to rebuild the Synset, so
    re-verifying is 20x cheaper than the rebuild it usually avoids. A store that wrote something
    unrelated therefore drops nothing at all; a store whose `ic` was rewritten drops exactly the
    synsets that changed, on the next read of each.

    ⚠ BOTH HALVES OF THE STAMP ARE LOAD-BEARING. MEASURED on 71, `wn-dog.n.01` carries no
    `hypernyms` field — `_synset_from_doc` below reads the taxonomy out of `edge WHERE src=?` — and
    an edge write does not move the vertex's `_seq`. Verifying only the vertex would keep serving a
    synset whose parents had moved, which is the same silent-substitution failure one level down.

    A rebuild REPLACES the object rather than mutating it. That is what lets a cache built on this
    module's output (`geometry._DENSE_CACHE`, keyed on `generation()`) miss by construction instead
    of holding a vector derived from values the Synset no longer carries."""
    ent = _CACHE.get(name)
    if ent is not None and ent[0] == _GEN:
        return ent[2]                        # verified at the current observation: no store touch
    arts = _observe(store)                   # we are going to the store: observe it while we do
    aid = _prefix(store) + name
    gen = _GEN                               # …after the observation, which may have moved it
    # ⚠ THE STAMP IS TAKEN BEFORE THE DOC IS READ, and the order is not incidental. Stamped after,
    # a write landing between the two reads would file the OLD doc under the NEW stamp and it would
    # verify clean for ever. Stamped first, the same race files a NEW doc under an OLD stamp, which
    # merely costs one extra rebuild at the next gate move. Conservative in the direction of the
    # rebuild, never in the direction of the stale answer.
    st = _freshness.stamp(arts, aid, edges=True)
    if ent is not None and st is not None and st == ent[1]:
        _CACHE[name] = (gen, st, ent[2])             # re-verified: same artifact, same edges
        return ent[2]
    doc = arts.get_artifact(aid)
    if not doc:
        _CACHE.pop(name, None)                       # gone from the store is a change, not a hit
        return None
    s = _synset_from_doc(name, doc, store)
    if st is not None:
        _CACHE[name] = (gen, st, s)                  # unverifiable ⇒ uncached, never assumed fresh
    return s


def _resolve(name: str, store=None) -> Optional["Synset"]:
    """One synset by name — keyed when the substrate is ready, else via the legacy index."""
    if _keyed_ready(store):
        return _get_synset(name, store)
    return _index(store).get(name)


def _entry_names(word: str, pos: Optional[str] = None, store=None) -> List[str]:
    """The language:en transducer's ENTRY, keyed: `lemma:<word> --lex:en--> wn-<synset>`, ordered as nltk's
    `synsets()` orders — POS first (noun before verb before …), then the source's own (proper-noun,
    sense-rank). POS is carried on the edge, so filtering and ordering need no doc read."""
    order = _pos_order(store)
    rank = {p: i for i, p in enumerate(order)}
    # ⛔ WAS `except Exception: rows = []`. A locked DB, a closed connection or any transient store
    # fault then made the word indistinguishable from a genuine OOV token — and the honest-refusal
    # gate tests exactly "fires no synset", so an infrastructure fault became a CONFIDENT REFUSAL
    # with measured-looking justification. The refusal path must be able to tell "the corpus does
    # not attest this" from "I could not ask the corpus"; a check that fails open into the refusal
    # branch proves nothing. Let it raise.
    rows = _conn(store).execute(
        "SELECT dst, props FROM edge WHERE src=? AND label=?",
        (_spec_str("lemma_prefix", store) + word, _spec_str("entry_label", store))).fetchall()
    out: List[Tuple[int, int, float, str]] = []
    for dst, props in rows:
        p = _json.loads(props) if props else {}
        _pos = p.get("pos")
        if pos is not None and _pos != pos:
            continue
        out.append((rank.get(_pos, len(order)),         # POS group first — noun before verb, as nltk
                    int(p.get("proper", 0)),                # common before proper
                    float(p.get("rank")) if isinstance(p.get("rank"), (int, float)) else float("inf"),
                    _n(dst, store)))                        # then sense number, then name
    # ⛔ SENSE ORDER IS ONLY AN ORDER IF THE SOURCE SUPPLIED IT. When `rank` was never materialized
    # (a partial import, a re-enrichment) every candidate took `inf`, the sort fell through to the
    # FOURTH element — the synset NAME — and the senses came back ALPHABETICAL while still
    # presenting as sense-rank order. That is the mechanism behind the measured "dog -> the
    # fireplace andiron" failure on node 45, and it was silent: `synsets()` still returned a
    # plausible non-empty list, and every consumer that trusts position (the 1/(1+i) priors in
    # match.py and activation.py) was reading noise. Say so instead.
    if out and all(_rk == float("inf") for _po, _pr, _rk, _nm in out):
        raise RankUnavailable(
            f"no sense rank stored for {word!r} ({len(out)} senses); ordering would be alphabetical, "
            "not sense-frequency — re-run the enrichment that materializes `rank`"
        )
    out.sort()
    return [nm for _po, _pr, _rk, nm in out]


def entry_prefix_exists(surface: str, store=None) -> bool:
    """Does the lexicon hold ANY entry whose surface STARTS WITH `surface`?

    ⭐ THIS IS WHAT REPLACES EVERY MAXIMUM-COMPOUND-LENGTH CONSTANT. Longest-match over a lexicon was
    written twice in this package (`match.fired_field`, `activation.seeds_from_text`) as
    `for j in range(min(n, i + 4), i + 1, -1): if j - i < 2: break` — i.e. **a multi-word entry may
    be at most three tokens long**, typed in, in two places, in a system whose lexicon holds
    entries like `united_states_of_america` (4) and `american_standard_code_for_information_
    interchange` (6). Those concepts were unreachable by name and nothing said so.

    The fix is not to measure the maximum and cap at it — that is the same constant with better
    provenance. It is to ASK THE LEXICON WHERE TO STOP. A window is worth extending exactly while
    some entry still begins with it; when none does, no longer window can match either, so the walk
    ends. That is a trie's own stopping rule, it needs no bound, and it is LINEAR in the token count
    rather than quadratic — which is what makes removing the cap affordable on a document as well as
    on a query.

    Keyed range read on the indexed `edge.src` (`ix_e_src`), `LIMIT 1`: MEASURED on 71's 5.7 GB
    lattice, 11.8 µs for a miss and 30–40 µs for a hit. The upper bound is the prefix with the
    highest code point appended, so the range is exactly the prefix's subtree of the index.

    A store that cannot answer returns **False** — and that is the honest direction: an unanswerable
    lexicon does not extend the window, so the single tokens stand, exactly as if the compound were
    not held. It never invents a compound."""
    p = _spec_str("lemma_prefix", store) + (surface or "")
    try:
        row = _conn(store).execute(
            "SELECT 1 FROM edge WHERE src >= ? AND src < ? LIMIT 1",
            (p, p + "\U0010FFFF")).fetchone()
    except Exception:
        return False
    return row is not None


def _load_index(store=None) -> None:
    """Read every ontology artifact once and build (name -> Synset) + ((lemma,pos) -> [names]) indexes.

    ⛔ THE STORE IS THE ONE PASSED IN. This opened `local_store.open_store()` directly — a SECOND
    ambient bind beside `_arts`'s, so even a caller that had bound a store got the process default
    here. It now goes through `_arts`, which is the single resolution point."""
    global _INDEX, _IC_STATS
    arts = _observe(store)                   # a full-corpus read: observe the store before taking it
    prefix = _prefix(store)
    idx: Dict[str, Synset] = {}
    word: Dict[Tuple[str, str], List[str]] = {}
    n_missing_ic = 0
    n_missing_se = 0

    for a in arts.list_artifacts(content_type=_ct(store)):
        aid = a.get("id") or ""
        name = aid[len(prefix):] if prefix and aid.startswith(prefix) else aid
        if not name:
            continue
        pos = a.get("pos") or (name.rsplit(".", 2)[-2] if name.count(".") >= 2 else NOUN)
        counts = a.get("lemma_counts") or {}
        if not isinstance(counts, dict):
            counts = {}
        counts = {str(k): int(v) for k, v in counts.items()}
        # ⛔ `Synset.lemmas()` WAS EMPTY ON EVERY FRESH CORPUS, silently. It reads `_counts`, which
        # comes from `lemma_counts` — a SemCor sense-frequency map written by the retired nltk
        # enrichment, which a self-contained ingest does not produce. The word index survived
        # (it falls back to `lemmas` two lines below), so nothing looked broken; but any caller
        # asking a SYNSET what word it names got back nothing. MEASURED live 2026-07-23: that is
        # why an answer rendered as "Oewn-01044274-n — ... It is a kind of oewn-01030024-n" instead
        # of "Mass — the celebration of the Eucharist".
        #
        # The lemmas the row DOES carry are the answer, in source order (primary first). Their
        # count is 0 — honestly "no measured SemCor frequency", not a fabricated one — and every
        # caller of `count()` already defaults to 0 for an unknown word.
        if not counts:
            counts = {str(x): 0 for x in (a.get("lemmas") or []) if x}
        # `float(a.get("ic") or 0.0)` folded THREE states into one number: field absent, field
        # present-and-zero, and (because `or` treats 0 as falsy) a genuinely stored 0. Absence is
        # now preserved as None so `has_ic()` can distinguish it; the arithmetic value is
        # unchanged. NOTE the producer still conflates one pair — `enrich_wordnet.py` writes a
        # literal `ic=0.0` for synsets absent from the nltk build, which is indistinguishable
        # from a real root-level zero AT THE SOURCE and cannot be recovered here.
        _ic_raw = a.get("ic")
        _ic_val = None if _ic_raw is None else float(_ic_raw)
        # The SECOND CHANNEL, counted separately. A corpus can carry `ic` and no `ic_se` (every
        # corpus enriched before smoothing existed does), so one coverage number cannot stand for
        # both -- see `ic_coverage()`.
        _se_raw = a.get("ic_se")
        _se_val = None if _se_raw is None else float(_se_raw)
        node = Synset(name, pos, list(a.get("hypernyms") or []),
                      list(a.get("instance_hypernyms") or []), _ic_val, counts, _se_val,
                      store=store)
        if _ic_val is None:
            n_missing_ic += 1
        if _se_val is None:
            n_missing_se += 1
        idx[name] = node
        # word -> senses: from the synset's lemma names (fall back to the plain `lemmas` field)
        lemma_names = list(counts.keys()) or list(a.get("lemmas") or [])
        # `lemmas` is LOWERCASED (it is the keyed-lookup field and lookup is case-insensitive);
        # `sense_ranks` keeps the WRITTEN form, because that is where the source's case lives.
        # Match them case-insensitively and read the case off the rank key.
        # ⭐ THE SOURCE'S OWN IRREGULAR FORMS become morphy's exception list. `aardwolves`,
        # `abaci`, `feet` follow no detachment rule — the rules would strip them to nothing real,
        # and the lemma-index validator would then correctly refuse them, leaving the word
        # unresolvable. LMF lists them per entry (`forms`, captured at ingest), so the exceptions
        # are DATA the source supplied rather than a table anyone maintains.
        for _lm, _fs in (a.get("forms") or {}).items():
            for _f in (_fs or []):
                _EXC.setdefault((str(_f).lower().replace(" ", "_"), pos), str(_lm).lower().replace(" ", "_"))
        _ranks = a.get("sense_ranks") or {}
        _by_lower = {str(k).lower(): (str(k), v) for k, v in _ranks.items()}
        for lm in lemma_names:
            _written, _r = _by_lower.get(lm.lower(), (lm, None))
            # ⭐ CAPITALIZATION IS THE SOURCE'S PROPER-NOUN MARKER, and it is the only thing that can
            # order two entries that share a lowercase form. `mass` (the physical quantity) and
            # `Mass` (the Eucharist) are SEPARATE LexicalEntries, so a sense number is per-entry and
            # BOTH are sense 0. With the case thrown away the tie fell through to synset-offset
            # order, and MEASURED live: "mass physics" answered with the celebration of the
            # Eucharist. A lowercase query means the common noun; the proper noun sorts after it.
            # ⚠ VERIFY THIS THE HARD WAY. An earlier version read the case off `lm`, which is
            # ALWAYS lowercase — so `proper` was always 0 and did nothing. It still measured
            # correctly on "mass physics", but only because the case MISMATCH left the Eucharist
            # row's rank unresolvable (inf) and it sorted last by accident. A fix that works for a
            # reason you cannot state is not a fix; two lemmas colliding in lowercase with both
            # ranks resolvable would have gone straight back to offset order.
            proper = 1 if _written != _written.lower() else 0
            # ⛔ MULTI-WORD LEMMAS WERE UNREACHABLE. The key was `lm.lower()` — a SPACE form
            # ("baseball bat") — but every lookup normalizes to underscores (`synsets()` does
            # `word.lower().replace(" ", "_")`, and the `_EXC` table above already stores underscore
            # keys). So `baseball_bat.n.01` existed, carried the lemma, and could NEVER be found by
            # name: `wn_synsets_for("baseball bat")` -> []. MEASURED: compound concepts (baseball bat,
            # fruit bat, power plant) returned ungrounded. The index must key in the same normal form
            # the lookup uses — underscores — so a queried compound resolves to its synset.
            word.setdefault((lm.lower().replace(" ", "_"), pos), []).append(
                (proper, float(_r) if isinstance(_r, (int, float)) else float("inf"), name))

    # ── THE TREE COMES FROM EDGES (§13.1: relations ARE edges) ───────────────────────────────────
    # A synset's hypernyms were historically duplicated into a doc field; the OEWN ingest writes them
    # where they belong — as `hypernym` edges in the lattice. Reading the doc field alone left every
    # synset with an EMPTY parent list, so `jc_tree` had no path to walk and every pair measured
    # distance 0.0: the ontology had nodes and no tree, and no propagator could discriminate
    # `dog(animal)` from `dog(blackguard)`. Read the edges; fall back to the doc field for corpora
    # that still carry it (both shapes coexist, neither is forced).
    # The IS-A tree is read from the labels that MEAN is-a, whatever the source called them —
    # `_isa_labels`, which reads them from the spec. `instance_hypernym` is OEWN's own name;
    # `instance_of` is the alias the legacy corpus carries. Both are the same relation, and a
    # reader that knew only one silently lost every instance edge on the other corpus.
    _isa, _inst_labels = _isa_labels(store)
    _labels = list(_isa) + list(_inst_labels)
    try:
        conn = arts.db.read()
        rows = conn.execute(
            "SELECT src, dst, label FROM edge WHERE label IN (%s)" % ",".join("?" * len(_labels)),
            tuple(_labels)).fetchall()
    except Exception:
        rows = []
    if rows:
        def _strip(x: str) -> str:
            x = str(x or "")
            return x[len(prefix):] if prefix and x.startswith(prefix) else x
        for src, dst, label in rows:
            node = idx.get(_strip(src))
            if node is None:
                continue
            target = _strip(dst)
            bucket = node._inst if label in _inst_labels else node._hyper
            if target not in bucket:
                bucket.append(target)

    # ── INTRINSIC IC — the metric derived from the tree, not imported ────────────────────────────
    # Jiang-Conrath needs information content: `IC(s1)+IC(s2)-2·IC(lcs)`. Historically IC came from
    # nltk's Brown/SemCor via `scripts/enrich_wordnet.py` — an EXTERNAL corpus, and the very
    # dependency [[self-contained-wordnet]] dropped from the runtime. A corpus ingested without that
    # bootstrap carries no `ic`, every term of JC is zero, EVERY PAIR MEASURES DISTANCE 0.0, and no
    # propagator can tell `dog(animal)` from `dog(blackguard)`.
    #
    # Intrinsic IC (Seco et al.) reads the same quantity off the tree we already have:
    #     IC(s) = 1 - log(|hyponyms(s)| + 1) / log(N)
    # A synset with many descendants is general (low IC); a leaf is maximally specific (IC -> 1).
    # Nothing is imported and nothing is chosen — the structure IS the measurement, which is why
    # this is the self-contained form rather than a substitute for the external one.
    #
    # ⛔ THE GATE WAS `n_missing_ic == len(idx)` — "only when EVERY synset lacks it" — AND THAT IS
    # WHAT LEFT 120,630 CONCEPTS WITHOUT A COORDINATE. Intrinsic IC is a measurement OF THE CORPUS:
    # BOTH terms of `1 - log(desc+1)/log(N+1)` move when a source is added. An all-or-nothing gate
    # encodes the opposite — "IC arrives once and never changes" — so adding a source to a corpus
    # that already carries IC leaves the new source silently unmeasured.
    #
    # MEASURED 2026-08-01 on 71's live lattice, the exact sequence:
    #   * every stored `pwn30` IC solves EXACTLY to N+1 = 555,596 (`wn-cat.n.01` ic 0.7230406747,
    #     desc 38; `wn-entity.n.01` ic 0.1445381081, desc 82,114) — i.e. the pre-OEWN corpus,
    #     117,659 pwn30 + 437,936 omw.
    #   * OEWN landed 2026-07-31 (+120,630) -> n_missing_ic = 120,630, len(idx) = 676,225, gate
    #     FALSE, block skipped, OEWN `_ic` left None.
    #   * `seed_lattice.build()` then wrote `Synset.ic()` — the documented ABSENCE SENTINEL 0.0 —
    #     onto all 120,630 of them as if it were a measurement.
    #   * `_path_edges` weights each edge `max(IC(c) - IC(p), 0)`; on an all-zero path every weight
    #     is 0, `sparse_vec` keeps only `d > 0`, so it returns {} and `dense_vec` is ALL-ZERO. The
    #     hypernym edges were there the whole time (304,764 out of `wn-oewn-*`, 93,446 of them
    #     `hypernym`); it was the WEIGHTS that were missing, not the path.
    #
    # SO THE TRIGGER IS PROVENANCE, NOT ABSENCE. `ic_basis(store)` is the corpus's own record of
    # what its IC was measured from (`geom.ic-basis`, written by `seed_lattice`). The measurement is
    # re-taken when that record says it was taken against a DIFFERENT corpus, or when there is no
    # record at all — a stored number that cannot state what it was measured from is not a
    # measurement we may keep ([[absence-is-not-an-affirmative-claim]]). A corpus carrying an
    # EXTERNAL measurement (Resnik over Brown, `scripts/enrich_wordnet.py`) records a basis whose
    # `source` is not `intrinsic`, and is left untouched — the two shapes still coexist, but now by
    # each saying what it is rather than by a count of holes.
    _basis = ic_basis(store)
    _src = _basis.get("source")
    if _src and _src != INTRINSIC_IC_SOURCE:
        _fresh = True                 # an external measurement — `N` does not describe it
    else:
        _fresh = (_src == INTRINSIC_IC_SOURCE
                  and _basis.get("n") == len(idx) and not n_missing_ic)
    if idx and not _fresh:
        for _n_, _v_ in derive_intrinsic_ic(idx).items():
            idx[_n_]._ic = _v_
        n_missing_ic = 0

    # ── SENSE ORDER (§13.14) ─────────────────────────────────────────────────────────────────────
    # ⛔ This was `names.sort()`, justified as "sense-number order (dog.n.01 < dog.n.02) — matches
    # nltk's synsets() order". That justification held ONLY for nltk-style names, where the sense
    # number is the sort key by construction. OEWN names are offsets (`oewn-09467004-n`), so the same
    # sort silently became OFFSET order — arbitrary with respect to meaning. MEASURED: `star`
    # resolved to the network-topology sense, and every downstream reader that trusted "first sense"
    # inherited that. The convention changed; the sort's reason did not follow it.
    #
    # The order is now the SOURCE's: the sense number LMF records for each word (`sense_ranks`).
    # A corpus that carries no ranks sorts by name exactly as before — the old shape still works, it
    # is simply no longer mistaken for a statement about meaning.
    for key, pairs in list(word.items()):
        pairs.sort(key=lambda t: (t[0], t[1], t[2]))   # common before proper, then sense number
        word[key] = [n for _p, _r, n in pairs]
    _IC_STATS = {"synsets": len(idx), "with_ic": len(idx) - n_missing_ic, "without_ic": n_missing_ic,
                 "with_ic_se": len(idx) - n_missing_se, "without_ic_se": n_missing_se}
    _INDEX = (idx, word)


def _cannot_serve_an_ontology(store) -> bool:
    """Can this store be ASKED for an ontology at all? Not: does it have one.

    ⛔ "CANNOT BE ASKED" IS NOT "HAS NONE", and collapsing them fabricates an empty corpus. A store
    handle that cannot be enumerated or queried has told us nothing about its ontology — it has told
    us it is not an ontology substrate. Building an empty index from that and serving it is the same
    defect as `.get("grounded", True)`: an affirmative claim manufactured out of an absence.

    MEASURED 2026-08-01: this is what threading the DELEGATE's store into the ontology read exposed.
    A delegate's store is its COGNITION store; in production it is the lattice and carries both, but
    a test holding a minimal fake carries neither — and `wolf.n.01`, which the installed offline
    index has, came back as "no synset in store". Two different substrates were being named by one
    handle, and only the failure said so.

    ⚠ BOTH CAPABILITIES ARE REQUIRED, AND ASKING FOR ONLY ONE WAS A BUG I WROTE. This first checked
    `list_artifacts` alone; the fakes have neither that nor `db`, so the enumeration deferred to the
    installed index and the very next call went to `_conn(store)` and died on `.db` instead. A
    predicate that answers "can this be asked?" with half the question just moves the failure one
    frame. An ontology read needs BOTH — enumeration to build the index, SQL to walk the entry edges
    — so a handle missing either cannot serve one."""
    a = getattr(store, "artifacts", store)
    return not (callable(getattr(a, "list_artifacts", None)) and getattr(a, "db", None) is not None)


def _index(store=None) -> Dict[str, "Synset"]:
    global _INDEX, _INDEX_GEN, _INDEX_FOREIGN
    if _installed_index_answers(store):
        return _INDEX[0]                     # installed, not read — there is no store to observe
    # An INSTALLED index answers for a store that cannot be asked. This is not the ambient fallback:
    # nothing is opened, nothing is guessed, and a store that CAN be asked always answers for itself
    # even when its answer is an empty ontology.
    if store is not None and _INDEX_FOREIGN and _INDEX is not None and _cannot_serve_an_ontology(store):
        return _INDEX[0]
    _observe(store)                          # resolve + observe FIRST: either may invalidate _INDEX
    _stale = (not _INDEX_FOREIGN) and _INDEX_GEN != _GEN
    if _INDEX is None or _stale:
        with _LOCK:
            if _INDEX is None or ((not _INDEX_FOREIGN) and _INDEX_GEN != _GEN):
                # ⚠ THE GENERATION IS CAPTURED BEFORE THE LOAD, NOT AFTER. The load takes 25.4 s on
                # 71's corpus and calls `_arts` throughout, so the gate can fire DURING it. Stamping
                # the finished index with the generation it STARTED at means a write that landed
                # mid-load leaves it visibly out of date and the next call rebuilds; stamping it
                # afterwards would certify an index as current against a store it never fully saw.
                g = _GEN
                _load_index(store)
                _INDEX_GEN = g
                _INDEX_FOREIGN = False       # this one IS the store's derivation
    return _INDEX[0]


def _word_index(store=None) -> Dict[Tuple[str, str], List[str]]:
    _index(store)
    return _INDEX[1]


# ── module-level API (the nltk.corpus.wordnet slice this codebase uses) ───────────────────────────
#
# ⭐ EVERY PUBLIC ENTRY OBSERVES THE STORE ON THE WAY IN — `_observe(store)`, one 14.1 µs read of the
# write mark. This is the boundary the freshness hook is hung on, and it is hung HERE rather than on
# `_arts` (71 polls per `tree_path`, measured — a cache costing more than no cache) or on the
# store-touching paths alone (a warm `synset(name)` touches nothing, so a repair never landed —
# measured, and the whole suite passed while it was broken). See `_gate`.
def synset(name: str, *, store=None) -> Synset:
    _observe(store)
    n = _resolve(name, store)
    if n is None:
        raise WordNetError(f"no synset {name!r} in store")
    return n


def synsets(word: str, pos: Optional[str] = None, *, store=None) -> List[Synset]:
    """All senses of `word` (optionally restricted to a POS), in sense-number order — as nltk returns.

    ⛔ AN INFLECTED WORD USED TO RESOLVE TO NOTHING, silently. MEASURED on the live corpus:
    `dogs`, `cats`, `limits` and `derivatives` all returned ZERO senses, so a need containing them
    contributed NO POSITION AT ALL — `cats and dogs` fired nothing whatsoever and the whole question
    grounded on nothing. `morphy` (the WordNet detachment rules + the lemma index as the validator)
    was already implemented in this file and simply was never called on the lookup path. The rules
    are part of the WordNet standard, not our invention — the same category as sense order: source
    knowledge that was present and unused.

    The lemma index is what makes this safe: a candidate is only accepted if it IS a lemma we hold,
    so `bus` -> `bu` (strip "s") is refused because `bu` is not a word. No guessing survives."""
    _observe(store)                                      # public entry: observe the store — see `_gate`
    w = word.lower().replace(" ", "_")
    if _keyed_ready(store):                              # keyed language-transducer entry — no full load
        names = _entry_names(w, pos, store)
        if not names:
            base = morphy(w, pos, store=store)
            if base and base != w:
                names = _entry_names(base, pos, store)
        return [s for s in (_get_synset(n, store) for n in names) if s is not None]
    wi = _word_index(store)
    _poss = _pos_order(store)
    if pos is not None:
        names = wi.get((w, pos), [])
    else:
        names = [n for p in _poss for n in wi.get((w, p), [])]
    if not names:
        base = morphy(w, pos, store=store)
        if base and base != w:
            if pos is not None:
                names = wi.get((base, pos), [])
            else:
                names = [n for p in _poss for n in wi.get((base, p), [])]
    idx = _index(store)
    return [idx[n] for n in names if n in idx]


# nltk's morphy strips inflection via exception lists + suffix rules; we approximate with the rules and
# validate the candidate against our own lemma index (we know every real lemma). Used only to normalize
# taught-relation verbs for matching, so an approximation is sufficient.
_RULES = {
    NOUN: [("s", ""), ("ses", "s"), ("xes", "x"), ("zes", "z"), ("ches", "ch"), ("shes", "sh"),
           ("men", "man"), ("ies", "y")],
    VERB: [("s", ""), ("ies", "y"), ("es", "e"), ("es", ""), ("ed", "e"), ("ed", ""),
           ("ing", "e"), ("ing", "")],
    ADJ: [("er", ""), ("est", ""), ("er", "e"), ("est", "e")],
    ADV: [],
}


def morphy(word: str, pos: Optional[str] = None, *, store=None) -> Optional[str]:
    """The base form of an inflected word, or None. Exceptions first, then the detachment rules,
    every candidate validated against the lemma index we actually hold."""
    _observe(store)                                 # public entry: observe the store — see `_gate`
    w = word.lower().replace(" ", "_")
    # The rule table is keyed by POS, so the alphabet to try is the spec's minus the satellite
    # (which carries no detachment rules of its own) — i.e. exactly the tags `_RULES` names.
    poss = [pos] if pos else [p for p in _pos_order(store) if p in _RULES]
    if _keyed_ready(store):                              # keyed: existence = a `lex:en` entry edge
        exc = _lang_spec(store).get("exceptions") or {}  # source irregulars, carried in the transducer artifact
        def _known(cand: str, p: str) -> bool:
            return bool(_entry_names(cand, p, store))
        for p in poss:                                   # already a known lemma of that POS
            if _known(w, p):
                return w
        for p in poss:                                   # the SOURCE's irregulars, before any rule
            base = exc.get(w + "|" + p) or (exc.get(w) if isinstance(exc.get(w), str) else None)
            if base and _known(base, p):
                return base
        for p in poss:                                   # else the detachment rules, validated keyed
            for suf, repl in _RULES.get(p, []):
                if suf and w.endswith(suf):
                    cand = w[: -len(suf)] + repl
                    if _known(cand, p):
                        return cand
        return None
    wi = _word_index(store)
    for p in poss:                                  # already a known lemma of that POS
        if (w, p) in wi:
            return w
    for p in poss:                                  # the SOURCE's irregulars, before any rule
        base = _EXC.get((w, p))
        if base and (base, p) in wi:
            return base
    for p in poss:                                  # else try inflection rules, validate against index
        for suf, repl in _RULES.get(p, []):
            if suf and w.endswith(suf):
                cand = w[: -len(suf)] + repl
                if (cand, p) in wi:
                    return cand
    return None


def all_synsets(pos: Optional[str] = None, *, store=None) -> List[Synset]:
    idx = _index(store)
    return [s for s in idx.values() if pos is None or s.pos() == pos]


__all__ = [
    "NOUN", "VERB", "ADJ", "ADV", "ADJ_SAT", "WordNetError", "Synset",
    "bind", "synset", "synsets", "morphy", "all_synsets", "ic_coverage",
    "entry_prefix_exists",
]
