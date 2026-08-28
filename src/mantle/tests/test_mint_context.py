"""A mint records what the store observed, and never invents what it did not.

The failure this guards is not a crash. It is a context that LOOKS complete — every key present,
every value plausible — and is partly made up. That artifact is then indistinguishable from a
correctly-minted one forever, because context is not derivable from the row after the fact.

So most of these tests assert ABSENCE.
"""
from __future__ import annotations

import hashlib

import pytest

from mantle.services import mint_context as mc


def test_a_mint_with_nothing_observable_records_only_what_it_saw():
    """No collection, no content, no principal — the mint still happens and claims nothing extra."""
    ctx = mc.mint(artifact_id="a1")
    assert ctx[mc.ARM_KEY] == mc.ARM_THIN
    assert ctx["minted"]["at"]                    # the one thing always knowable
    assert "placement" not in ctx, "placement was invented from no collection"
    assert "addressing" not in ctx, "addressing was invented from no content"
    assert "screen" not in ctx, "a screen was invented where nothing was co-present"


def test_the_hash_is_of_the_plaintext_so_context_and_cas_agree():
    body = "the lattice is the memory"
    ctx = mc.mint(artifact_id="a1", content=body, content_type="text/markdown")
    assert ctx["addressing"]["sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert ctx["addressing"]["bytes"] == len(body.encode("utf-8"))
    assert ctx["addressing"]["content_type"] == "text/markdown"


def test_empty_content_is_addressed_and_absent_content_is_not():
    """`""` is a body that hashes; `None` is a body nobody supplied. They must not read alike."""
    assert "sha256" in mc.mint(artifact_id="a", content="")["addressing"]
    assert "addressing" not in mc.mint(artifact_id="a")


def test_the_screen_records_what_was_co_written():
    ctx = mc.mint(artifact_id="a2", co_written=["a1", "a3"])
    assert ctx["screen"]["co_written"] == ["a1", "a3"]


def test_an_artifact_is_never_its_own_screen():
    """A row co-present with itself would inflate every screen by one and mean nothing."""
    ctx = mc.mint(artifact_id="a2", co_written=["a1", "a2", "a3"])
    assert "a2" not in ctx["screen"]["co_written"]


def test_a_capped_sibling_list_says_that_it_was_capped(monkeypatch):
    """A truncated screen that does not announce the truncation is a claim of completeness.

    Same rule as the retrieval canon's §"No silent truncation": a capped selection reports
    `capped`. Here the report is what tells a later reader the screen is a floor, not the set.
    """
    monkeypatch.setattr(mc, "_siblings", lambda db, cid, *, limit: [f"s{i}" for i in range(limit + 5)])
    ctx = mc.mint(artifact_id="a", collection_id="c1", sibling_limit=4)
    assert ctx["screen"]["siblings_capped"] == 4
    assert len(ctx["screen"]["siblings"]) <= 4


def test_an_uncapped_sibling_list_does_not_claim_a_cap(monkeypatch):
    monkeypatch.setattr(mc, "_siblings", lambda db, cid, *, limit: ["s1", "s2"])
    ctx = mc.mint(artifact_id="a", collection_id="c1", sibling_limit=64)
    assert "siblings_capped" not in ctx["screen"]


def test_caller_assertions_are_never_merged_into_observed_facets():
    """What a writer ASSERTS and what the store OBSERVED must stay tellable apart.

    A caller can claim any `sha256` it likes; the store's own addressing must not absorb it, or the
    hash stops being evidence about the bytes the store actually holds.
    """
    ctx = mc.mint(artifact_id="a", content="real", caller={"sha256": "deadbeef", "source": "x.md"})
    assert ctx["caller"]["sha256"] == "deadbeef"
    assert ctx["addressing"]["sha256"] != "deadbeef"


def test_no_seam_means_no_enriched_key_at_all(monkeypatch):
    """A store with no host must be distinguishable from one whose seam ran and found nothing."""
    monkeypatch.setattr(mc, "_enrich", lambda base, **kw: None)
    assert mc.ENRICHED_KEY not in mc.mint(artifact_id="a")


def test_a_seam_that_raises_leaves_the_thin_mint_standing(monkeypatch):
    """An enrichment service being down may not fail a write whose db-level context is complete."""
    import sys
    import types
    mod = types.ModuleType("fake_ctx_seam")
    mod.enrich = lambda base, **kw: (_ for _ in ()).throw(RuntimeError("seam down"))
    sys.modules["fake_ctx_seam"] = mod
    monkeypatch.setattr("prism.runner.registered_seams", lambda: {"context": "fake_ctx_seam"},
                        raising=False)
    ctx = mc.mint(artifact_id="a", content="x")
    assert ctx["addressing"]["sha256"]
    assert mc.ENRICHED_KEY not in ctx


def test_enrichment_lands_beside_the_observed_facets_never_inside_them(monkeypatch):
    import sys
    import types
    mod = types.ModuleType("fake_ctx_seam2")
    mod.enrich = lambda base, **kw: {"lemmas": ["lattice"], "licence": "CC-BY-4.0"}
    sys.modules["fake_ctx_seam2"] = mod
    monkeypatch.setattr("prism.runner.registered_seams", lambda: {"context": "fake_ctx_seam2"},
                        raising=False)
    ctx = mc.mint(artifact_id="a", content="x", content_type="text/markdown")
    assert ctx[mc.ENRICHED_KEY]["licence"] == "CC-BY-4.0"
    assert "licence" not in ctx["addressing"]
    assert ctx[mc.ARM_KEY] == mc.ARM_THIN, "enrichment must not claim to be the minting arm"


def test_facets_reports_what_is_there_and_not_what_ought_to_be():
    thin = mc.mint(artifact_id="a")
    assert mc.facets(thin) == {"minted": True, "placement": False, "addressing": False,
                               "screen": False, "enriched": False}
    full = mc.mint(artifact_id="a", content="x", collection_id=None, co_written=["b"])
    f = mc.facets(full)
    assert f["addressing"] and f["screen"]


def test_facets_of_nothing_is_all_false_not_an_error():
    """The gate runs over rows minted before this existed; `None` must read as 'no facets'."""
    assert mc.facets(None) == {"minted": False, "placement": False, "addressing": False,
                               "screen": False, "enriched": False}


def test_mint_never_raises_without_an_acting_principal():
    """Key custody fails closed on a missing principal. A DESCRIPTION must not.

    Recording who minted is the least dangerous field in the record; refusing the whole context for
    want of it would lose the four facets that were observable.
    """
    ctx = mc.mint(artifact_id="a", content="x")
    assert ctx["minted"]["at"]


def test_the_module_imports_no_tekton():
    """Mantle is Apache and must stay standalone-useful; chorus is AGPL.

    `beacon/__init__` states the property this protects. The seam is a NAME resolved at call time,
    so the static import graph here may not mention a tekton.
    """
    import pathlib
    src = pathlib.Path(mc.__file__).read_text(encoding="utf-8")
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", "*", ">")))
    for forbidden in ("import astra", "from astra", "import chorus", "from chorus",
                      "import ember", "from ember"):
        assert forbidden not in body, f"mint_context reaches a tekton: {forbidden!r}"
