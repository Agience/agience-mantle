"""`LocalStore` must answer the `resolve_text` seam chorus asks for.

THE DEFECT, measured 2026-08-27. `sage/describe._split` reads a body through chorus's own
`sage/content.resolve_text`, which is a duck-typed seam:

    r = getattr(bundle, "resolve_text", None)
    if callable(r):
        return r(artifact)
    return artifact.get("content") or ""

`LocalStore` does not answer it, so every artifact whose body lives in the CAS falls through to
inline `content` — which for those artifacts is EMPTY. The describer then extracts no terms from an
empty string and takes its always-terminating fallback, writing `lemmas=['document']` (874 rows) or
`['module']` (473). `describe_dark` skipped them from then on.

**319,307 artifacts have a `content_ref` and no inline content**: 317,594 markdown (the Wikipedia
corpus and the capture lane), 1,616 python, 97 plain. Every one reads as an empty document to the
describer. This is not a custody problem and no grant would fix it — the describer never reaches
the content at all.

An earlier diagnosis in this thread said the describer was handed CIPHERTEXT and needed a principal.
That was mantle's `resolve_text`, which this caller does not use: chorus imports `from . import
content as C`, its own module, exactly as it mirrors `corpus_fts.py`. Both readings describe real
code; only this one is on the path the describer takes.

THE SEAM ALREADY HAS A PRECEDENT IN THIS FILE. `address_inline` was added for the same reason — the
chorus operator registrars ask for it by name (`getattr(store, "address_inline", None)`) rather than
importing mantle, because chorus does not depend on mantle and must not gain that edge. `resolve_text`
is the read-side twin of the same arrangement.
"""
from __future__ import annotations

from mantle.shard.local_store import LocalStore


def test_the_store_answers_the_seam_chorus_asks_for():
    """`getattr(bundle, "resolve_text", None)` must find something callable."""
    r = getattr(LocalStore, "resolve_text", None)
    assert callable(r), (
        "chorus's sage/content.resolve_text asks the bundle for this by name and falls back to "
        "inline `content` when it is absent — which reads 319,307 CAS-backed artifacts as empty")


def test_it_takes_one_argument_the_artifact():
    """The seam calls `r(artifact)`, so the bound method takes exactly the artifact."""
    import inspect

    sig = inspect.signature(LocalStore.resolve_text)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["artifact"], (
        "the seam calls `bundle.resolve_text(artifact)`; a different arity silently misses, "
        "because `getattr` only checks that the name is callable: %s" % params)


def test_inline_content_still_wins_when_there_is_no_ref():
    """An artifact carrying its body inline must not need the CAS at all."""
    store = LocalStore(artifacts=None, graph=None, content=None)
    got = store.resolve_text({"id": "a1", "content": "# hello\n\nbody"})
    assert got == "# hello\n\nbody"


def test_no_ref_and_no_content_is_empty_not_an_error():
    store = LocalStore(artifacts=None, graph=None, content=None)
    assert store.resolve_text({"id": "a1"}) == ""


def test_the_seam_hydrates_through_the_doc_boundary_not_the_raw_tier():
    """It must open the MEC1 envelope, not hand back what the tier stored.

    A first version of this method delegated to `mantle.shard.content.resolve_text`, which reads
    the content TIER — a plaintext surface for the layer it owns, but the bytes stored there are
    themselves the doc-boundary envelope, so it returns `MEC1…` and the caller sees ciphertext.
    `decrypt_artifact_content` is the function that opens that envelope: it passes `content_key`
    AND `cas_ref` together and tries ordered principal candidates, which is why
    `capture_offer.py` — which calls it — read real bodies and segmented sessions into 1,008 turns
    while the describer saw nothing.

    Asserted by NAME rather than by behaviour because the alternative needs a live store, keys and
    a principal; what this guards is that the seam is wired to the hydrating function at all."""
    import inspect

    from mantle.shard.local_store import LocalStore

    src = inspect.getsource(LocalStore.resolve_text)
    assert "decrypt_artifact_content" in src, (
        "the seam reads the raw tier, so every CAS-backed body comes back as an unopened MEC1 "
        "envelope — which is what the describer saw")
