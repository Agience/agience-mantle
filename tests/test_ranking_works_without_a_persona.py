"""The ranking is mantle's, and the base install gets it with no ontology and no chorus.

## Why this module exists where it does

`search -> rank -> cut` lived in `sage/content_search.py`, which put it ABOVE the store that
produces the candidates. mantle's own `recall` had no way to reach it, so the base install ranked
lexically and nothing the ranking learned was available without a persona installed.

The candidates come from mantle, so the ordering belongs to mantle. Retrieval deliberately did NOT
move with it: candidates arrive as `(id, content_type, score)` from whatever produced them —
mantle's recall, an FTS index, a caller's list — which is what lets one implementation serve both
the encrypted store's search and a corpus one, over different indexes and different content.

## The two tiers

    base install      no `match` seam -> the candidate order passes through, and the account says
                      `unavailable`; the cut falls back to `_knee`
    ontology attached `bind(match=…, projection=…)` -> reach ranking, and the cut becomes the
                      aperture's own `k_signal` where an instrument is registered

The same shape `search/beacon` has for the spectral read: a thin arm that is not less correct, and
a sharper one when it is present.
"""
from __future__ import annotations

import pytest

from mantle.search import ranking


@pytest.fixture(autouse=True)
def unbound(monkeypatch):
    """Every test here starts as a base install: nothing bound AND nothing registered.

    The registry is cleared as well as the bind slots, because "base install" has to mean the same
    thing however this process was assembled. `_resolve_seam` falls back to the host seams
    `prism.runner` holds, so a test that only cleared `_MATCH_SEAM` would quietly pass or fail on
    whether something else in the session had registered one.
    """
    from prism import runner
    monkeypatch.setattr(ranking, "_MATCH_SEAM", None)
    monkeypatch.setattr(ranking, "_PROJECTION_SEAM", None)
    monkeypatch.setattr(runner, "_HOST_SEAMS", {})
    monkeypatch.delenv(ranking._ONTOLOGY_HOST_ENV, raising=False)
    monkeypatch.setattr(ranking, "_HOST_TRIED", None)


def test_it_imports_with_no_ontology_present():
    """The base install has no crystal and no ember. A module that could not be imported without
    them would make the whole point moot."""
    assert ranking.__all__ == ["rank", "cut_for", "relevance_cut", "knee"]


def test_ranking_without_a_seam_passes_the_order_through_and_says_so():
    """Not an error and not a guess: the candidates keep the order they arrived in, and the account
    names why. A caller can tell "no ontology here" from "the ontology found nothing"."""
    cand = [("a", "t", -9.0), ("b", "t", -1.0)]
    ranked, account, reached = ranking.rank(cand, "anything", store=None)

    assert ranked == cand
    assert account["reach"] == "unavailable"
    assert reached == set()


def test_the_cut_still_derives_without_an_instrument():
    """`_knee` is the fallback, so the base install still answers "how many" from the scores rather
    than from a number someone picked."""
    assert ranking.knee([-9.0, -8.8, -8.6, -1.0, -0.9]) == 3
    assert ranking.relevance_cut([-9.0, -8.8, -8.6, -1.0, -0.9]) == 3


def test_a_bound_but_unreachable_seam_is_reported_not_raised():
    """A host may bind a lazy proxy with nothing behind it. Proving the seam is reachable INSIDE the
    guard is what keeps that an honest `unavailable` instead of a traceback from one line later."""
    class _Hollow:
        def __getattr__(self, name):
            raise ImportError("nothing bound behind this proxy")

    ranking.bind(match=_Hollow())
    cand = [("a", "t", -9.0), ("b", "t", -1.0)]
    ranked, account, reached = ranking.rank(cand, "q", store=None)

    assert ranked == cand and account["reach"] == "unavailable" and reached == set()


def test_binding_an_ontology_turns_reach_on():
    """The other tier, with a stated geometry: `b` reaches, `a` does not, so the order changes."""
    class _Match:
        def fired_field(self, query, store):
            return {"need": 1.0}

        def offer_synsets(self, text):
            return []

        def propagate(self, fired, targets):
            return (9.0, 0.0) if "need" in targets else (0.0, float("inf"))

    class _Store:
        artifacts = None

    ranking.bind(match=_Match())
    cand = [("wn-x", "t", -9.0), ("wn-need", "t", -1.0)]
    ranked, account, reached = ranking.rank(cand, "need", _Store())

    assert account["reach"] == "measured", account
    assert ranked[0][0] == "wn-need", [r[0] for r in ranked]
    assert "wn-need" in reached


# ── the cut, and the argument that was forgettable ───────────────────────────────────────────────


class _Projection:
    """A projection seam that records what it was asked to place, and on which store."""

    def __init__(self):
        self.asked = None
        self.store = None

    def frame(self, store, names):
        self.asked, self.store = list(names), store
        return "FRAME-%d" % len(names)


#: Stands in for the store handle every real caller holds. The frame is a MEASUREMENT over that
#: store, so `cut_for` does not build one without it — a caller with no store gets `_knee`, which
#: is the same thin tier a caller with no projection gets.
STORE = object()


def test_the_cut_reads_the_frame_the_ranking_already_names(monkeypatch):
    """The defect this closes, as a rule.

    `relevance_cut` gives the sharp answer only when it has a frame, and the frame is spanned by the
    synset candidates already sitting in the ranked list. Left to each caller to assemble, mantle's
    recall assembles none and gets `_knee` while believing it asked for the instrument: on the live
    corpus four of six queries then cut 50 of 50, which is no cut at all, where deriving the frame
    here takes the same six to 3/8/5/3/6/5.
    """
    proj = _Projection()
    ranking.bind(projection=proj)
    seen = {}
    monkeypatch.setattr(ranking, "_relevance_cut",
                        lambda scores, **kw: seen.update(kw) or 2)

    ranked = [("wn-glacier.n.01", "t", -3.0), ("doc-a", "t", -2.0), ("wn-ice.n.01", "t", -1.0)]
    assert ranking.cut_for(ranked, query="glacier", store=STORE) == 2

    assert seen.get("frame") == "FRAME-2", (
        "the cut ran without the frame the ranking could have handed it (%r)" % seen)
    assert proj.store is STORE, "the frame was read off a store the caller did not hand over"
    assert proj.asked == ["glacier.n.01", "ice.n.01"], (
        "the frame must be spanned by the SYNSET candidates, whose names are coordinates the "
        "projection can place; got %r" % (proj.asked,))


def test_a_ranking_with_no_synsets_cuts_on_the_scores_alone(monkeypatch):
    """Prose-only candidates name no coordinates, so there is no frame to build. That is the thin
    tier answering, not a failure — and the projection must not be asked to place nothing."""
    proj = _Projection()
    ranking.bind(projection=proj)
    seen = {}
    monkeypatch.setattr(ranking, "_relevance_cut", lambda scores, **kw: seen.update(kw) or 1)

    assert ranking.cut_for([("doc-a", "t", -2.0)], query="q", store=STORE) == 1
    assert seen.get("frame") is None and proj.asked is None


def test_a_caller_holding_a_frame_is_not_made_to_build_it_twice(monkeypatch):
    proj = _Projection()
    ranking.bind(projection=proj)
    monkeypatch.setattr(ranking, "_relevance_cut", lambda scores, **kw: 1 if kw.get("frame") == "MINE" else 0)

    assert ranking.cut_for([("wn-a.n.01", "t", -1.0)], query="q", store=STORE,
                           frame="MINE") == 1
    assert proj.asked is None, "a frame was rebuilt over one the caller already held"


def test_nothing_ranked_is_a_cut_of_nothing():
    assert ranking.cut_for([], query="q") == 0


def test_a_cut_that_cannot_be_read_keeps_every_candidate(monkeypatch):
    """The ranking above it ran and IS the order. Answering with nothing because the cut failed
    would discard a measured result to report an unmeasured one."""
    monkeypatch.setattr(ranking, "_relevance_cut", lambda scores, **kw: 1 / 0)
    ranked = [("a", "t", -3.0), ("b", "t", -2.0)]
    assert ranking.cut_for(ranked, query="q", store=STORE) == 2


# ── the seam a host registered, which is what removes the chorus requirement ─────────────────────


def test_a_registered_host_seam_equips_the_ranking_with_no_bind_call(monkeypatch):
    """The requirement this closes.

    `bind()` needs a caller, and for a while the only one was `sage.content_search` — so the
    ranking lived in mantle and reaching it still meant installing chorus. A runner already answers
    "which module fills this name": `prism.runner.register_seam`, which `ember/runtime/seams.py`
    calls at import of `ember`. Reading that map is what makes mantle plus a runner enough.

    Verified against the live corpus as well as here: in a process that imports `ember` and nothing
    else, `_resolve_seam("match")` returns `ember.ontology.match` and a two-candidate ranking on
    `what is a glacier` reports `reach: measured` with `bind()` never called.
    """
    import sys
    import types
    from prism import runner

    mod = types.ModuleType("fake_ontology_seam")
    mod.fired_field = lambda query, store: {"need": 1.0}
    mod.offer_synsets = lambda text: []
    mod.propagate = lambda fired, targets: ((9.0, 0.0) if "need" in targets
                                            else (0.0, float("inf")))
    monkeypatch.setitem(sys.modules, "fake_ontology_seam", mod)
    runner.register_seam("match", "fake_ontology_seam")

    class _Store:
        artifacts = None

    ranked, account, reached = ranking.rank(
        [("wn-x", "t", -9.0), ("wn-need", "t", -1.0)], "need", _Store())

    assert account["reach"] == "measured", (
        "a host registered its ontology and the ranking did not find it: %s" % account)
    assert ranked[0][0] == "wn-need" and "wn-need" in reached


def test_an_explicit_bind_wins_over_the_registry(monkeypatch):
    """Passing an object is a statement about THIS process; a registry lookup cannot contradict it.
    A host that binds a narrower seam than it registered gets the one it bound."""
    import sys
    import types
    from prism import runner

    wrong = types.ModuleType("wrong_seam")
    wrong.fired_field = lambda query, store: {}
    monkeypatch.setitem(sys.modules, "wrong_seam", wrong)
    runner.register_seam("match", "wrong_seam")

    class _Bound:
        def fired_field(self, query, store):
            return {"need": 1.0}

        def offer_synsets(self, text):
            return []

        def propagate(self, fired, targets):
            return (9.0, 0.0) if "need" in targets else (0.0, float("inf"))

    ranking.bind(match=_Bound())

    class _Store:
        artifacts = None

    _ranked, account, _reached = ranking.rank([("wn-need", "t", -1.0)], "need", _Store())
    assert account["reach"] == "measured", (
        "the registry overrode an explicitly bound seam: %s" % account)


def test_a_registered_seam_that_does_not_import_is_reported_not_raised(monkeypatch):
    """A host may register a module that is not installed. That is the same absence as no seam at
    all and must read as one — the alternative is a recall failing on an import error."""
    from prism import runner
    runner.register_seam("match", "no.such.module.anywhere")

    ranked, account, reached = ranking.rank([("a", "t", -1.0)], "q", store=None)
    assert ranked == [("a", "t", -1.0)] and account["reach"] == "unavailable" and reached == set()


# ── the standalone node, and the one knob that equips it ─────────────────────────────────────────


def test_a_standalone_node_orders_by_what_it_has_and_says_so(monkeypatch):
    """No runner in the process, no env var: the registry is empty and the ranking says
    `unavailable`. This is the SHIPPED behaviour of a mantle-only node and is not a defect —
    a node with no ontology in it has no reach to measure."""
    ranked, account, _reached = ranking.rank([("a", "t", -1.0)], "q", store=None)
    assert ranked == [("a", "t", -1.0)] and account["reach"] == "unavailable"


def test_the_operator_names_a_host_module_and_importing_it_is_the_mechanism(monkeypatch):
    """`MANTLE_ONTOLOGY_HOST` names a module to import. Importing it is the WHOLE mechanism: the
    module registers its own seams with `prism.runner` on the way in, and nothing here knows what
    an ontology is. Verified live: a process importing only mantle, with
    `MANTLE_ONTOLOGY_HOST=ember`, resolves `match` to `ember.ontology.match` and ranks
    `what is a glacier` with `reach: measured`."""
    import sys
    import types
    from prism import runner

    seam = types.ModuleType("fake_seam_module")
    seam.fired_field = lambda query, store: {"need": 1.0}
    seam.offer_synsets = lambda text: []
    seam.propagate = lambda fired, targets: ((9.0, 0.0) if "need" in targets
                                             else (0.0, float("inf")))
    monkeypatch.setitem(sys.modules, "fake_seam_module", seam)

    host = types.ModuleType("fake_host_package")
    runner.register_seam("match", "fake_seam_module")     # what importing it would have done
    monkeypatch.setitem(sys.modules, "fake_host_package", host)
    monkeypatch.setenv(ranking._ONTOLOGY_HOST_ENV, "fake_host_package")

    class _Store:
        artifacts = None

    _r, account, _reached = ranking.rank([("wn-need", "t", -1.0)], "need", _Store())
    assert account["reach"] == "measured", account


def test_a_host_module_that_is_not_installed_is_a_warning_not_a_failure(monkeypatch, caplog):
    """A node configured for an ontology it does not have must serve coverage-ordered recalls, not
    500 on every query. The log names the module, so the misconfiguration is visible rather than
    inferred from the ordering."""
    monkeypatch.setenv(ranking._ONTOLOGY_HOST_ENV, "no.such.host.package")

    with caplog.at_level("WARNING"):
        ranked, account, _reached = ranking.rank([("a", "t", -1.0)], "q", store=None)

    assert ranked == [("a", "t", -1.0)] and account["reach"] == "unavailable"
    assert any("no.such.host.package" in r.getMessage() for r in caplog.records), (
        "the misconfigured module name was not named in any log record")


def test_the_host_module_is_imported_once_per_process(monkeypatch):
    """`_registered_seam` runs on every ranking, and importing on each one would put an import lock
    in the recall path. The name tried is remembered, so a second call is free."""
    import importlib
    import sys
    import types

    calls = []
    monkeypatch.setenv(ranking._ONTOLOGY_HOST_ENV, "fake_counted_host")
    monkeypatch.setitem(sys.modules, "fake_counted_host", types.ModuleType("fake_counted_host"))
    orig = importlib.import_module

    def _counting(name, *a, **kw):
        if name == "fake_counted_host":
            calls.append(name)
        return orig(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", _counting)

    ranking.rank([("a", "t", -1.0)], "q", store=None)
    ranking.rank([("a", "t", -1.0)], "q", store=None)
    ranking.rank([("a", "t", -1.0)], "q", store=None)

    assert len(calls) == 1, "the host module was imported %d times" % len(calls)


# ── one concept, one row, however many ingests described it ──────────────────────────────────────


def _doc(lemmas, pos="n", operator="op.source.wordnet", **extra):
    import json
    d = {"lemmas": list(lemmas), "pos": pos, "operator": operator}
    d.update(extra)
    return json.dumps(d)


def test_two_ingests_describing_one_concept_become_one_row():
    """The measured defect. This corpus was seeded by `op.source.wordnet` and `op.source.oewn`,
    both English, so the same concept is a vertex under each and both used to land inside a cut the
    instrument derived — 24 of 149 shown answer rows repeated a title already in that answer.

    The best-scoring row survives (the list arrives best-first) and the other is NAMED, not
    discarded: it is where the other source's gloss lives.
    """
    ranked = [("wn-volcano.n.02", "t", -4.6), ("wn-oewn-09495727-n", "t", -4.5)]
    docs = {"wn-volcano.n.02": _doc(["volcano"]),
            "wn-oewn-09495727-n": _doc(["volcano"], operator="op.source.oewn")}

    out, folded = ranking._fold_sources(ranked, docs)

    assert [r[0] for r in out] == ["wn-volcano.n.02"], [r[0] for r in out]
    assert folded == {"wn-volcano.n.02": ["wn-oewn-09495727-n"]}


def test_two_senses_from_ONE_ingest_are_two_answers():
    """The control, and the reason the fold tests the SOURCE rather than trusting the key.

    Measured on the live corpus, `(pos, lemma set)` collides 7,958 times within a single lexicon,
    and those rows are genuinely different senses — `how does a telescope work` returns two OEWN
    verbs both lemma'd `telescope`. Folding on the key alone would delete one of them.
    """
    ranked = [("wn-oewn-00245809-v", "t", -4.9), ("wn-oewn-01597703-v", "t", -4.1)]
    docs = {"wn-oewn-00245809-v": _doc(["telescope"], pos="v", operator="op.source.oewn"),
            "wn-oewn-01597703-v": _doc(["telescope"], pos="v", operator="op.source.oewn")}

    out, folded = ranking._fold_sources(ranked, docs)

    assert len(out) == 2 and folded == {}, (out, folded)


def test_the_name_is_read_on_the_lexicon_alphabet():
    """Diacritics folded, case dropped, underscores spaced — the alphabet
    `crystal.ontology.lookup` uses, so two sources spelling one concept differently are one name."""
    assert (ranking._fold_key({"lemmas": ["Canis_familiaris", "dog"], "pos": "n"})
            == ranking._fold_key({"lemmas": ["canis familiaris", "DOG"], "pos": "n"}))
    assert (ranking._fold_key({"lemmas": ["café"], "pos": "n"})
            == ranking._fold_key({"lemmas": ["cafe"], "pos": "n"}))


def test_a_satellite_adjective_is_not_its_head():
    """`s` is not normalised to `a`. Measured: normalising it merged `able.a.01` with `able.s.02`,
    a head adjective and its satellite, and raised same-lexicon key collisions 7,958 -> 8,331."""
    assert (ranking._fold_key({"lemmas": ["able"], "pos": "a"})
            != ranking._fold_key({"lemmas": ["able"], "pos": "s"}))


def test_a_row_that_names_nothing_is_never_folded():
    """A prose artifact carries no lemma set here, so there is no name to match. Two of them must
    not collapse into each other on the strength of both having nothing."""
    ranked = [("doc-a", "t", -3.0), ("doc-b", "t", -2.0)]
    docs = {"doc-a": _doc([], operator="op.ingest.wiki"),
            "doc-b": _doc([], operator="op.ingest.other")}

    out, folded = ranking._fold_sources(ranked, docs)
    assert len(out) == 2 and folded == {}
    assert ranking._fold_key({"lemmas": [], "pos": "n"}) is None


def test_a_row_with_no_doc_passes_through():
    """The store did not return a row for it — a version id, a candidate from another index. An
    unknown row is kept, because dropping one would be a claim about a document never read."""
    ranked = [("known", "t", -3.0), ("unknown", "t", -2.0)]
    docs = {"known": _doc(["thing"])}
    out, folded = ranking._fold_sources(ranked, docs)
    assert [r[0] for r in out] == ["known", "unknown"] and folded == {}


def test_the_order_is_otherwise_untouched():
    """The fold removes rows; it never reorders the ones it keeps."""
    ranked = [("a", "t", -5.0), ("dup", "t", -4.0), ("b", "t", -3.0), ("c", "t", -2.0)]
    docs = {"a": _doc(["alpha"]), "dup": _doc(["alpha"], operator="op.source.oewn"),
            "b": _doc(["beta"]), "c": _doc(["gamma"])}
    out, folded = ranking._fold_sources(ranked, docs)
    assert [r[0] for r in out] == ["a", "b", "c"]
    assert [r[2] for r in out] == [-5.0, -3.0, -2.0]
    assert folded == {"a": ["dup"]}


def test_a_third_source_folds_onto_the_same_keeper():
    ranked = [("pwn", "t", -5.0), ("oewn", "t", -4.0), ("omw", "t", -3.0)]
    docs = {"pwn": _doc(["thing"]),
            "oewn": _doc(["thing"], operator="op.source.oewn"),
            "omw": _doc(["thing"], operator="op.source.omw")}
    out, folded = ranking._fold_sources(ranked, docs)
    assert [r[0] for r in out] == ["pwn"]
    assert folded == {"pwn": ["oewn", "omw"]}


# ── the need's own position: right for a corpus, ruinous for a recall ─────────────────────────────


class _NeedIsAVertex:
    """An ontology where the need names a synset the store happens to hold."""

    def fired_field(self, query, store):
        return {"need": 1.0}

    def offer_synsets(self, text):
        return []

    def propagate(self, fired, targets):
        return (9.0, 0.0) if "need" in targets else (1.0, 0.5)


class _StoreHoldingTheNeed:
    class artifacts:
        @staticmethod
        def get_artifact(artifact_id):
            return {"id": artifact_id} if artifact_id == "wn-need" else None


def test_the_need_s_own_position_is_added_by_default():
    """A corpus search wants it: BM25 will not surface a head sense whose gloss never says the
    word, so the node the need names is a candidate by construction."""
    ranking.bind(match=_NeedIsAVertex())
    ranked, account, _reached = ranking.rank(
        [("wn-other", "t", -1.0)], "need", _StoreHoldingTheNeed(),
    )
    assert "wn-need" in [r[0] for r in ranked], (
        "the default must still add the node the need names — %s" % ([r[0] for r in ranked],)
    )


def test_own_position_false_adds_nothing():
    """A recall filters the order back to the ids that went in, so an added id can never be
    returned — and adding it is not neutral. `_fold_sources` collapses one concept arriving from
    two ingests and keeps the best-scoring row, so the injected row absorbs the narrowed row and
    is then discarded WITH it. Measured on the live path, that is why `what is a glacier` answered
    `Piedmont glacier / continental glacier / polar glacier` and never `glacier`: the hyponyms
    survived because nothing folded them.
    """
    ranking.bind(match=_NeedIsAVertex())
    ranked, account, _reached = ranking.rank(
        [("wn-other", "t", -1.0)], "need", _StoreHoldingTheNeed(),
        own_position=False,
    )
    assert [r[0] for r in ranked] == ["wn-other"], (
        "no candidate may be added when the caller declined it — %s"
        % ([r[0] for r in ranked],)
    )
    assert account["reach"] == "measured", account
