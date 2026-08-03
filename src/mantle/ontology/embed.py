"""The local embedder, and the alignment that makes it swappable.

THE PROBLEM
-----------
Region ids are ``{principal}/{collection}/{anchor_id}``, and ``anchor_id`` is a UUID5 of
``sha256(label, model_id, embedding)``. So a different embedder mints *different anchor ids*
and therefore *different region ids* — a leaf that embedded with its own model would compute
cell names nobody else uses, and its shards would be unshareable with the mesh. Silently. The
routing would still "work"; it would just be routing into a private universe.

THE FIX — project, don't re-anchor
----------------------------------
Ember does **not** build its own AnchorSet. It caches the **canonical** one (anchors are
artifacts, so they cache like anything else) and projects its local query vectors into that
space with a cross-walk (``mesh.anchors.crosswalk``): same dim → orthogonal
Procrustes; **cross dim → rectangular least-squares**. That is what makes the ontology
embedding-dimension agnostic in practice — a 384-dim leaf can route against a 1024-dim
canonical space and land on **the same region ids**.

And it can get aligned with the network unplugged: an ``Anchor`` carries BOTH its ``label``
and its canonical ``embedding``, so Ember embeds the cached anchors' *labels* with its own
small model, pairs them against the canonical vectors already present, and fits the cross-walk
locally. No cloud call to become alignable — only to learn about *new* anchors.

UPGRADING
---------
"Something small while disconnected, upgrade during sync" is then just: refresh the cached
AnchorSet artifact and refit. The cross-walk's ``error_bound`` says how much fidelity the small
embedder is costing, so the upgrade is measurable rather than a matter of faith.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Protocol, Sequence

import numpy as np

from mantle.search.anchors.anchorset import AnchorSet
from mantle.search.anchors.crosswalk import fit_crosswalk


class Embedder(Protocol):
    """Turn text into vectors. Deliberately tiny surface — the cross-walk is what lets the
    implementation be swapped without changing a single region id."""

    model_id: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashEmbedder:
    """A deterministic, dependency-free embedder. **Not semantic — for tests and demos.**

    It exists so the whole read path (routing, alignment, hit/miss, refusal) can be exercised
    with numpy alone, and so the scaffold has no model download in its critical path. It is
    stable across processes and machines, which is all the plumbing needs.

    Do not ship this as a real embedder: hashed tokens have no meaning, so "nearest" is
    arbitrary. There is no longer a "real" embedder to swap in — see the deletion note below;
    semantic retrieval is the computed ontology coordinate, not a learned vector space.
    """

    model_id = "hash-embed-v1"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in (t or "").lower().split():
                h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                out[i, idx] += sign
        from prism import vector as _vec
        return _vec.unit(out, axis=1)


# ⛔ `Model2VecEmbedder` WAS HERE AND IS DELETED. [John, 2026-07-20: "no models, period"]
#
# It loaded `minishlab/potion-*` via `StaticModel.from_pretrained` — a static *distilled* table,
# but distilled FROM a trained sentence-transformer, so every number in it is a trained weight.
# "Static" and "deterministic" described its inference cost, not its provenance; that distinction
# is exactly what let trained weights sit in this tree while the docs called the path model-free.
#
# There is no replacement embedder and there must not be one — not a smaller model, not an
# Apache-2.0 model, not a fallback. The semantic arm is `ember/geometry.py`'s exact Jiang-Conrath
# ontology coordinate (LATTICE-IMPLEMENTATION §2.2), which is COMPUTED from our own WordNet IC
# rather than trained. `HashEmbedder` above remains only as the plumbing exerciser it always was.


class Aligner:
    """Projects this leaf's local vectors into the canonical anchor space.

    Holds the cross-walk. If the local embedder already *is* the canonical model, the
    cross-walk is skipped entirely (identity) rather than fitted — a leaf running the real
    model should pay nothing for the abstraction.
    """

    def __init__(self, embedder: Embedder, anchors: AnchorSet) -> None:
        self.embedder = embedder
        self.anchors = anchors
        self._crosswalk = None
        # ⛔ `_native` COMPARED ONLY `model_id`, AND A MODEL ID DOES NOT IDENTIFY A SPACE.
        # `HashEmbedder.model_id` is the class constant "hash-embed-v1" while `dim` is a
        # CONSTRUCTOR parameter, so `HashEmbedder(dim=64)` and `HashEmbedder(dim=128)` are two
        # incompatible vector spaces wearing one identity. With a 128-dim anchorset and a 64-dim
        # local embedder this returned native=True, `fit()` returned immediately without fitting
        # anything, and `encode_query` handed back the raw 64-dim vector AS IF CANONICAL — no
        # crosswalk, and therefore no `dim_in` check anywhere to catch it. `error_bound` then
        # returned None, which reads as "native, nothing projected, no loss" rather than
        # "unvalidated". That is routing into a private universe, silently — precisely what this
        # module's docstring — and the now-deleted `Model2VecEmbedder`'s own comment ("two
        # different potion models are two different spaces") — exist to prevent.
        #
        # Comparing dim as well is not merely a guard, it is the CORRECT behaviour: a differing
        # dim simply means not-native, so the normal path fits a rectangular least-squares
        # crosswalk — which this module already supports and advertises ("cross dim ->
        # rectangular least-squares"). The mismatch stops being silent AND starts being handled.
        self._native = (embedder.model_id == anchors.model_id
                        and int(getattr(embedder, "dim", -1)) == int(getattr(anchors, "dim", -2)))
        self._adopted = False   # did we reuse a shared walk instead of refitting?

    @property
    def native(self) -> bool:
        return self._native

    @property
    def error_bound(self) -> Optional[float]:
        """Mean residual of the projection — how much the small embedder is costing.
        ``None`` when native (nothing is being projected)."""
        return None if self._native else getattr(self._crosswalk, "error_bound", None)

    @property
    def adopted(self) -> bool:
        """True when we reused a shared cross-walk artifact rather than refitting."""
        return self._adopted

    def fit(self, cached: Optional[dict] = None) -> "Aligner":
        """Align this leaf. Offline by construction.

        ``cached`` is an optional cross-walk **artifact** (see `crosswalk_artifact`). The fit is
        deterministic given (embedder, AnchorSet), so every leaf on the same embedder computes
        the identical matrix — refitting per device is pure waste. If a valid one is supplied we
        adopt it; otherwise we fit from the anchors themselves, which is always possible with no
        network because an Anchor carries BOTH its label and its canonical embedding.

        A cached walk that does not verify is treated as a **cache miss, not an error**: we
        refit silently. That is deliberate — a stale or foreign walk still projects and still
        routes, just into the wrong cells, so the only safe response is to ignore it.
        """
        if self._native:
            return self
        if cached is not None:
            from mantle.search.anchors.crosswalk_artifact import from_artifact, verify_crosswalk_artifact
            ok, _why = verify_crosswalk_artifact(cached, self.anchors, self.embedder.model_id)
            if ok:
                self._crosswalk = from_artifact(cached)
                self._adopted = True
                return self
        anchors = self.anchors.anchors
        if not anchors:
            raise ValueError("cannot fit a cross-walk against an empty AnchorSet")
        labels = [a.label for a in anchors]
        source = self.embedder.encode(labels)                       # our space
        target = np.vstack([a.embedding for a in anchors])          # canonical space
        self._crosswalk = fit_crosswalk(
            source, target,
            source_model_id=self.embedder.model_id,
            target_model_id=self.anchors.model_id,
            method="auto",   # same dim -> procrustes; cross dim -> rectangular linear
        )
        return self

    def as_artifact(self, fitted_by: Optional[str] = None) -> dict:
        """This leaf's cross-walk, as a shareable artifact — so the next leaf need not refit."""
        if self._native:
            raise ValueError("a native embedder has no cross-walk to publish")
        if self._crosswalk is None:
            raise RuntimeError("Aligner.fit() must run before as_artifact()")
        from mantle.search.anchors.crosswalk_artifact import to_artifact
        return to_artifact(self._crosswalk, self.anchors, fitted_by=fitted_by)

    def encode_query(self, text: str) -> np.ndarray:
        """Text -> a vector in the CANONICAL space, ready to route."""
        v = self.embedder.encode([text])[0]
        if self._native:
            return v
        if self._crosswalk is None:
            raise RuntimeError("Aligner.fit() must run before encode_query() (no cross-walk)")
        return self._crosswalk.apply(v)


__all__ = ["Embedder", "HashEmbedder", "Aligner"]
