"""AnchorSet — the shared coordinate system, routing centroids, and grounding.

See `.dev/features/mantle-canonical-architecture.md` §3 (the AnchorSet is one
object: commons verified anchor ontology == MANTLE routing centroids ==
Beacon-LLM grounding) and §4 (the native language derived from it).

The AnchorSet is the FACET-commons verified anchor ontology realized in MANTLE:
a set of fully-disclosed reference points a client seeds. Routing is by *nearest
anchor* — the anchors ARE the centroids; there is no separate k-means partition,
and nothing here fits, grows or reconciles the set it was handed.

INVARIANT (§1): geometry only. Operates on plaintext vectors; never touches
cell keys / light-cone / oracle / ledger.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Fixed namespace so an anchor's artifact id is a *deterministic* UUID of its
# content — same anchor content → same id everywhere (idempotent creation,
# stable cell-routing id across rebuilds). The sha256 content-hash is the
# address; the UUID5 of it is the artifact ``_key`` (a valid UUID, as the
# artifact model expects).
_ANCHOR_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "anchors.agience.ai")

#: Domain separator for the anchor content hash. Part of the address, so it is spelled once.
_ANCHOR_HASH_DOMAIN = b"agience/anchor/v1"

# The TIER vocabulary an anchor record may state. Descriptive, carried through save/load
# untouched: the client that authored the set decides what each anchor is, and nothing here reads
# the value or changes it. Routing is by cosine and ignores the tier entirely.
CANDIDATE, WORKING, CANONICAL = "candidate", "working", "canonical"

#: How many offending anchors a refusal names before summarising the rest. The count is always
#: exact; this only bounds how much scrolls past whoever ran the load.
_REPORTED_OFFENDERS = 5


def l2norm(v: np.ndarray) -> np.ndarray:
    """Unit-normalize along the last axis (zero-safe), in anchor precision (float32).

    Zero-safe, in anchor precision (float32). mantle is dependency-free (storage owns its anchor
    geometry), so this is inline rather than through prism.vector."""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


def _preserve_unit(v: np.ndarray) -> np.ndarray:
    """Return ``v`` as float32, normalizing only if it is not already unit.

    Reading an anchor back must not move it. `l2norm` is NOT bitwise idempotent in float32 —
    dividing an already-unit vector by its recomputed norm shifts about a third of them by one
    ulp — and an anchor's id is the hash of its exact bytes, so a reader that re-normalizes
    hands back a vector whose hash no longer produces the id it arrived with. Every anchor that
    exists came through :meth:`Anchor.make`, which normalizes once, so the stored vector is
    already unit and the right thing to do with it is nothing. A vector that is genuinely off
    the sphere (norm away from 1 by more than float32 can explain) is still normalized: it did
    not come from `make`, and a non-unit anchor would silently distort every cosine it appears in.
    """
    v = np.ascontiguousarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v if abs(n - 1.0) <= 1e-6 else l2norm(v)


def anchor_content_hash(label: str, model_id: str, embedding: Sequence[float] | np.ndarray) -> str:
    """Content-address ``(label, model_id, embedding)`` — hashing the embedding **as given**.

    No normalization happens here, and that is the point. :meth:`Anchor.make` normalizes once
    and hands the result in; a verifier re-deriving the hash of a *stored* anchor must hash the
    stored vector, not a re-normalized copy of it, for the float32 reason in
    :func:`_preserve_unit`. Re-normalizing on the way in would fail roughly a third of a
    perfectly good canonical set against its own ids.
    """
    emb = np.ascontiguousarray(embedding, dtype=np.float32).ravel()
    h = hashlib.sha256()
    h.update(_ANCHOR_HASH_DOMAIN)
    h.update(label.encode("utf-8"))
    h.update(model_id.encode("utf-8"))
    h.update(emb.tobytes())
    return h.hexdigest()


def anchor_id_for(content_hash: str) -> str:
    """The artifact id an anchor with this content hash must carry."""
    return str(uuid.uuid5(_ANCHOR_NS, content_hash))


def verify_anchor_id(
    anchor_id: str, label: str, model_id: str,
    embedding: Sequence[float] | np.ndarray,
) -> Optional[str]:
    """``None`` when ``anchor_id`` is the id this content produces; else the reason it is not.

    The id IS the cluster id, and it propagates into the cell storage path, the HKDF key `info`,
    the AEAD associated data and the mesh region name. An anchor carrying an id its content does
    not produce therefore routes into a region no other holder of the same canonical set
    computes — and nothing downstream can notice, because every one of those uses accepts an
    arbitrary string. This is the only place the claim can be checked, so it is checked here.
    """
    expected = anchor_id_for(anchor_content_hash(label, model_id, embedding))
    if expected == str(anchor_id):
        return None
    return (
        f"anchor {label!r} states id {anchor_id} but its content produces {expected} "
        "(label, model_id and embedding are what the id is derived from)"
    )


def anchorset_fingerprint(anchors: "AnchorSet") -> str:
    """A content hash of a whole AnchorSet — its identity as a coordinate system.

    Over the *anchor ids*, sorted — ids are already content-addressed over
    (label, model_id, embedding), so this transitively covers every anchor's vector without
    re-hashing float arrays (whose bytes are platform-sensitive). Sorted because an AnchorSet
    is a set: insertion order must not change its identity.

    Lives here, beside the set it names. Everything that has to answer "is this the same
    coordinate system?" — the geometry a store's cells were written under, two operators
    comparing nodes, a client checking the file it seeded is the one that loaded — asks the
    same question of the same bytes.
    """
    h = hashlib.sha256()
    h.update(b"agience/anchorset/v1")
    h.update(anchors.model_id.encode("utf-8"))
    h.update(str(anchors.dim).encode("utf-8"))
    for aid in sorted(a.anchor_id for a in anchors.anchors):
        h.update(aid.encode("utf-8"))
    return h.hexdigest()


class AnchorSetCorrupt(ValueError):
    """A canonical AnchorSet that cannot be admitted as one set.

    Either a stated anchor id disagrees with the anchor's own content, or the anchors do not
    share one model and width. Both are refusals, not warnings: the first mints region ids no
    peer computes, and the second is two vocabularies in one file, of which any partial load
    silently picks whichever the reader happened to see first.
    """


@dataclass
class Anchor:
    """A fully-disclosed reference point — **an artifact** (canonical model:
    everything is an artifact). ``anchor_id`` is the artifact id (a deterministic
    UUID of ``content_hash``); the embedding and the rest live in the artifact's
    ``context`` (see :meth:`to_context` / :meth:`from_context`).

    Field set is superset-compatible with the facet-commons anchor record and
    Beacon-LLM's ``Anchor``. ``embedding`` is dense, unit-norm, and lives in the
    ``model_id`` space — which is the space every vector routed against this set
    must also be in, since Mantle bridges no two spaces.
    """

    anchor_id: str
    label: str
    embedding: np.ndarray
    model_id: str
    type_id: str = "text/plain"
    tier: str = WORKING
    placed_frame: int = 0
    status: str = "active"
    content_hash: str = ""

    @staticmethod
    def make(
        label: str,
        embedding: Sequence[float] | np.ndarray,
        model_id: str,
        *,
        type_id: str = "text/plain",
        tier: str = WORKING,
        placed_frame: int = 0,
    ) -> "Anchor":
        """Content-address the anchor over (label, model_id, embedding); the
        artifact id is a deterministic UUID5 of that hash."""
        emb = l2norm(np.asarray(embedding, dtype=np.float32).ravel())
        content_hash = anchor_content_hash(label, model_id, emb)
        return Anchor(
            anchor_id=anchor_id_for(content_hash),
            label=label,
            embedding=emb,
            model_id=model_id,
            type_id=type_id,
            tier=tier,
            placed_frame=placed_frame,
            content_hash=content_hash,
        )

    # ------------------------------------------------------------------ artifact form
    def to_context(self) -> dict:
        """The anchor as an artifact ``context`` dict (embedding as a float list)."""
        return {
            "label": self.label,
            "embedding": self.embedding.astype(float).tolist(),
            "model_id": self.model_id,
            "content_hash": self.content_hash,
            "type_id": self.type_id,
            "tier": self.tier,
            "status": self.status,
            "placed_frame": int(self.placed_frame),
        }

    @classmethod
    def from_context(cls, anchor_id: str, ctx: dict) -> "Anchor":
        """Rebuild an Anchor from an artifact id + its ``context`` dict.

        The embedding is preserved rather than re-normalized (:func:`_preserve_unit`), so the
        anchor that comes back out of the store hashes to the id it went in with and
        :func:`verify_anchor_id` is answerable about a stored anchor at all.
        """
        return cls(
            anchor_id=anchor_id,
            label=ctx.get("label", ""),
            embedding=_preserve_unit(np.asarray(ctx.get("embedding", []), dtype=np.float32)),
            model_id=ctx.get("model_id", ""),
            type_id=ctx.get("type_id", "text/plain"),
            tier=ctx.get("tier", WORKING),
            status=ctx.get("status", "active"),
            placed_frame=int(ctx.get("placed_frame", 0)),
            content_hash=ctx.get("content_hash", ""),
        )


class AnchorSet:
    """A set of anchors in one model space.

    Routing uses a cached ``(K, dim)`` unit-norm matrix, small enough to keep in memory. K is
    whatever the seeded file states — there is no fixed count and nothing here chooses one.
    """

    def __init__(self, model_id: str, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.model_id = model_id
        self.dim = int(dim)
        self._anchors: List[Anchor] = []
        self._ids: set[str] = set()
        self._matrix: Optional[np.ndarray] = None  # (K, dim), unit-norm

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self._anchors)

    @property
    def anchors(self) -> List[Anchor]:
        return list(self._anchors)

    @property
    def matrix(self) -> Optional[np.ndarray]:
        return self._matrix

    def _rebuild(self) -> None:
        self._matrix = (
            np.vstack([a.embedding for a in self._anchors]).astype(np.float32)
            if self._anchors
            else None
        )

    # --------------------------------------------------------------- mutation
    def add(self, anchor: Anchor) -> Anchor:
        if anchor.embedding.shape[-1] != self.dim:
            raise ValueError(
                f"anchor dim {anchor.embedding.shape[-1]} != AnchorSet dim {self.dim}"
            )
        if anchor.model_id != self.model_id:
            raise ValueError(
                f"anchor model {anchor.model_id!r} != AnchorSet model {self.model_id!r} — "
                "one set is one space, and Mantle projects between none"
            )
        if anchor.anchor_id in self._ids:  # content-addressed idempotency
            return anchor
        self._anchors.append(anchor)
        self._ids.add(anchor.anchor_id)
        self._rebuild()
        return anchor

    def add_text(
        self,
        label: str,
        embedding: Sequence[float] | np.ndarray,
        *,
        tier: str = WORKING,
        type_id: str = "text/plain",
        placed_frame: int = 0,
    ) -> Anchor:
        return self.add(
            Anchor.make(
                label, embedding, self.model_id,
                type_id=type_id, tier=tier, placed_frame=placed_frame,
            )
        )

    # --------------------------------------------------------------- queries
    def nearest(self, vec: Sequence[float] | np.ndarray, k: int = 8) -> List[Tuple[Anchor, float]]:
        """Top-``k`` anchors by cosine. Routing = nearest anchor(s)."""
        if self._matrix is None or k < 1:
            return []
        q = l2norm(np.asarray(vec, dtype=np.float32).ravel())
        if q.shape[-1] != self.dim:
            # Dimension mismatch (a different embedder / wrong model): the vector cannot be
            # placed against the anchors. Return empty — routing surfaces this as an error,
            # since there is no flat fallback to place it in instead.
            return []
        sims = self._matrix @ q
        k = min(k, len(self._anchors))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self._anchors[int(i)], float(sims[int(i)])) for i in idx]

    def position(self, anchor_id: str) -> int:
        for i, a in enumerate(self._anchors):
            if a.anchor_id == anchor_id:
                return i
        raise KeyError(anchor_id)

    # --------------------------------------------------- where a set comes from
    #
    # From the client, whole, as a file. An anchor id is content-addressed over
    # `(label, model_id, embedding)`, so cluster centers computed from a node's own local corpus
    # would mint region ids no other node computes: two deployments would each route confidently,
    # into disjoint cells, with no overlap between them.
    #
    # So there is no constructor that fits a set, none that extends one, and none that maps one
    # onto another. A leaf must never author its own set; `AnchorSet.load` and the anchor repo are
    # the ways in.

    # ------------------------------------------------------------- persistence
    #
    # ONE serialisation. This single-file JSON form is what a canonical AnchorSet travels as
    # (`agience-ember`'s `ember ingest --anchors PATH` reads exactly this), what
    # `mantle.system.manage_anchors --action load` admits into the store, and what
    # `shard.store.save_anchors` caches. There is no second shape and no lenient reader.
    def save(self, path: str | Path) -> None:
        """JSON dump (commons-aligned-ish). Embeddings as float lists.

        ``content_hash`` rides along so the file states its own address, but it is a convenience
        only: :meth:`load` re-derives the hash from the record and never believes this field.
        """
        payload = {
            "model_id": self.model_id,
            "dim": self.dim,
            "anchors": [
                {
                    "anchor_id": a.anchor_id,
                    "label": a.label,
                    "model_id": a.model_id,
                    "type_id": a.type_id,
                    "tier": a.tier,
                    "placed_frame": a.placed_frame,
                    "status": a.status,
                    "content_hash": a.content_hash,
                    "embedding": a.embedding.astype(float).tolist(),
                }
                for a in self._anchors
            ],
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AnchorSet":
        """Read a canonical AnchorSet, **verifying every anchor id against its content**.

        Ids are preserved exactly — the id is the cluster id, so a reader that re-mints them
        would put this node's cells in a universe of their own. Preserving an id is a claim, and
        the claim is checkable: `uuid5(_ANCHOR_NS, sha256(label ‖ model_id ‖ embedding))` is
        recomputed for each record and must equal the id the file states. A set whose ids do not
        match its contents is corrupt, and corrupt is refused whole
        (:class:`AnchorSetCorrupt`) — a partial load is the silent failure this check exists to
        remove.

        Embeddings are stored unit-norm by :meth:`Anchor.make`, and are taken as they are
        (:func:`_preserve_unit`): the verification hashes the bytes on disk, so the bytes in
        memory have to be the same ones. That also makes save→load→save a fixed point.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            model_id, dim = data["model_id"], int(data["dim"])
        except (KeyError, TypeError, ValueError) as e:
            raise AnchorSetCorrupt(f"{path}: no model_id/dim header ({e})") from e

        s = cls(model_id=model_id, dim=dim)
        bad: List[str] = []
        for i, a in enumerate(data.get("anchors", [])):
            try:
                anchor_id, label = str(a["anchor_id"]), a["label"]
                emb = _preserve_unit(np.asarray(a["embedding"], dtype=np.float32).ravel())
                a_model = a["model_id"]
            except (KeyError, TypeError, ValueError) as e:
                bad.append(f"anchor #{i}: unreadable record ({e})")
                continue
            reason = verify_anchor_id(anchor_id, label, a_model, emb)
            if reason is not None:
                bad.append(f"anchor #{i}: {reason}")
                continue
            if emb.shape[-1] != dim or a_model != model_id:
                # Reported here rather than left to `add`, which raises on the first offender
                # and so would name one anchor of however many disagree.
                bad.append(
                    f"anchor #{i} ({label!r}): dim {emb.shape[-1]} / model {a_model!r} "
                    f"against a set of dim {dim} / model {model_id!r}"
                )
                continue
            s.add(Anchor(
                anchor_id=anchor_id,
                label=label,
                embedding=emb,
                model_id=a_model,
                type_id=a.get("type_id", "text/plain"),
                tier=a.get("tier", WORKING),
                placed_frame=int(a.get("placed_frame", 0)),
                status=a.get("status", "active"),
                content_hash=anchor_content_hash(label, a_model, emb),
            ))
        if bad:
            raise AnchorSetCorrupt(
                f"{path}: {len(bad)} of {len(data.get('anchors', []))} anchors do not hold up; "
                "the set is refused whole because a partial one routes into regions no peer "
                "computes. " + " | ".join(bad[:_REPORTED_OFFENDERS])
                + (f" | ... and {len(bad) - _REPORTED_OFFENDERS} more"
                   if len(bad) > _REPORTED_OFFENDERS else "")
                + ". TO FIX: re-export the file from whatever authored it, with `AnchorSet.save`, "
                  "which writes each id from the record it belongs to. Editing the ids by hand "
                  "cannot fix this — the id IS the content hash."
            )
        return s
