"""Ember's local store handle — the SQLite lattice + FS content, reached through Mantle.

**Mantle is the one data-access path.** `open_store()` returns the lattice store
(`mantle.db` + `FileContentCache`) via `sqlite_store.open_sqlite_store()`. There is no
other backend: SQLite + `FsContentStore` is the only one.

The store is the durable truth (artifacts, offers, edges). Ember's anchor/IVF index is a derived,
rebuildable cache on top of it — delete the index, rebuild from the store.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:                       # names for annotations only — never imported at runtime
    from mantle.db.store import ArtifactStore, GraphStore, ContentStore
else:
    ArtifactStore = GraphStore = ContentStore = Any


@dataclass
class LocalStore:
    """The three store faces of the local lattice shard, bound and schema-ready."""
    artifacts: ArtifactStore
    graph: GraphStore
    content: ContentStore
    conn: Optional[Any] = None           # always None on the lattice backend — see `ready()`
    keys_dir: Optional[Path] = None      # for the content-encryption key (content.py)
    # The READ-seam tier (mantle.db.content_tier.TieredContentStore): FileContentCache in
    # front of the S3/CDN mirror, plaintext surface. Deliberately a SEPARATE handle from `content`:
    # the legacy write path (content.put_content) hands `content` CIPHERTEXT, and the tier's
    # plaintext-verifying surface would loudly refuse it. Readers (resolve_text) prefer this when
    # present; writers keep the surface they have. None ⇒ local-only (air-gapped is first-class).
    content_tier: Optional[Any] = None

    def address_inline(self, docs):
        """Move each doc's inline ``content`` into the CAS, leaving the ADDRESS behind. Returns
        ``docs``.

        WHY THIS IS A METHOD ON THE STORE [ruling 2, John 2026-08-26]. The chorus operator
        registrars write inline ``content`` and are the last writers in the system that do —
        all 25 rows `artifacts_holding_inline_plaintext` reports are theirs. They receive a store
        and nothing else, and they must not grow an import of mantle: chorus does not depend on
        mantle today, and adding one would be a new cross-repo edge for four lines of body-moving.
        A method on the object they are already handed needs no import at all — they ask
        ``getattr(store, "address_inline", None)`` and use it if it is there.

        Addressing, not sealing, and the difference is why this works. ``encrypt_artifact_content``
        goes through the key oracle, which needs an acting principal a bulk writer does not have —
        ``corpus/stage0_sources._sealed`` recorded that in writing, and two attempts to put sealing
        in ``put_artifact``/``put_many`` broke 113 tests before that note was found. ``put_content``
        keys off the KEYS DIRECTORY instead, so it works with no principal in scope. The row then
        carries only ``content_ref``, which every reader already prefers and which
        ``encrypt_artifact_content`` treats as already handled.

        Degrades rather than refuses, deliberately and for the same reason `_sealed` does: a store
        with no content tier keeps the body inline, which is what the row already does. A registrar
        that dropped its operators because the tier was unmounted would trade a confidentiality
        property for a functional one. Which also means a caller handed only an ArtifactStore —
        every chorus test does this — simply gets its docs back unchanged, so no caller breaks.
        """
        content, keys_dir = self.content, self.keys_dir
        if content is None or keys_dir is None:
            return docs
        try:
            from mantle.shard import content as C
        except Exception:                        # no CAS reachable — see the note above
            return docs
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            body = doc.get("content")
            if not body or doc.get("content_ref"):
                continue
            try:
                ref, size = C.put_content(content, keys_dir, str(body).encode("utf-8"),
                                          collection=doc.get("collection_id"))
            except Exception:                    # one bad body must not stop a registration
                continue
            doc["content_ref"] = ref
            doc["size"] = size
            doc.pop("content", None)
        return docs

    def resolve_text(self, artifact) -> str:
        """The artifact's full text, from the CAS when its body lives there.

        WHY THIS IS A METHOD ON THE STORE, and it is the read-side twin of `address_inline` above.
        `sage/content.resolve_text` is a duck-typed seam:

            r = getattr(bundle, "resolve_text", None)
            if callable(r):
                return r(artifact)
            return artifact.get("content") or ""

        chorus asks the bundle for it BY NAME rather than importing mantle — the same arrangement
        `address_inline` uses, and for the same reason: chorus does not depend on mantle and must
        not gain that edge. Unanswered, the seam falls through to inline `content`.

        MEASURED 2026-08-27: **319,307 artifacts have a `content_ref` and NO inline content** —
        317,594 markdown (the Wikipedia corpus and the capture lane), 1,616 python, 97 plain. Every
        one of them read as an EMPTY DOCUMENT to `sage/describe`, which then extracted no terms and
        wrote its always-terminating fallback: `lemmas=['document']` on 874 capture rows and
        `['module']` on 473. `describe_dark` skipped each of them from then on, so one empty read
        became permanent.

        IT HYDRATES THROUGH THE DOC BOUNDARY, NOT THE RAW TIER, and the difference is the whole
        point. `mantle.shard.content.resolve_text` reads the content TIER, which is a plaintext
        surface for the layer IT owns — but the bytes stored there are themselves the doc-boundary
        envelope, so that route returns `MEC1…` and the caller sees ciphertext.
        `decrypt_artifact_content` is what opens it: it passes `content_key` AND `cas_ref`
        together and tries ordered principal candidates. Measured — `capture_offer.py` calls it and
        read real bodies, segmenting sessions into 1,008 turns, while the describer on the tier
        route saw nothing at all.

        A first version of this method delegated to the tier route and was wrong for exactly that
        reason.

        An earlier reading of this thread had the describer receiving ciphertext and needing a
        principal. Custody IS consulted — the candidate loop tries the collection first and the
        grant holder second, and the second is the one that opens these bodies — but no NEW grant
        was ever missing. `doc_boundary` records the same wrong turn being taken in August on the
        310,003 Wikipedia rows: "a grant on `created_by` was minted to chase it and changed
        nothing."
        """
        if artifact.get("content"):
            return artifact.get("content") or ""
        if not artifact.get("content_ref"):
            return ""
        from mantle.db.doc_boundary import decrypt_artifact_content
        raw = dict(artifact)
        decrypt_artifact_content(raw, strict=True)
        return raw.get("content") or ""

    def ready(self) -> bool:
        """Is the store actually usable? The lattice is a local file opened in-process — there is no
        server to ping — so ask the store: a keyed read that can raise. A store whose file is gone,
        locked or corrupt says so rather than returning a blind `True` (see contract §5 on
        unmeasured-rendered-as-healthy)."""
        try:
            self.artifacts.get_artifact("__ready_probe__")   # keyed lookup: no scan, no count(*)
            return True
        except Exception:
            return False


def open_store(keys_dir: Optional[Path] = None, *, ensure_schema: bool = True) -> LocalStore:
    """Open the local store — the SQLite lattice + FS content, via Mantle (the one data path).

    SQLite + `FsContentStore` is the sole backend, so this is a thin alias for
    `sqlite_store.open_sqlite_store()`. The store resolves its own keys dir from the environment
    and ensures its own schema.

    Same defect class as `wn_store._arts()` binding an ambient store: a parameter that is accepted
    and discarded is worse than no parameter, because the call site reads as if it were honoured.
    So `open_store()` refuses a `keys_dir` argument outright: no caller passes it — every one
    passes only `ensure_schema=` — so refusing costs nothing and closes the trap. To point at a
    different store, set `EMBER_SQLITE_DIR` / `EMBER_SQLITE_DB`, which is what actually decides.

    `ensure_schema` is likewise not consulted (the lattice always ensures its schema), but it is
    still accepted because real callers pass it and its stated meaning is what happens anyway."""
    from mantle.shard.sqlite_store import open_sqlite_store
    if keys_dir is not None:
        raise TypeError(
            "open_store() does not take a keys_dir/path — accepting and silently ignoring it "
            "would let a caller believe they have an isolated store when they are reading the "
            "process-default one. Set EMBER_SQLITE_DIR / EMBER_SQLITE_DB (and "
            "EMBER_STORE_KEYS_DIR) instead; got %r"
            % (keys_dir,))
    return open_sqlite_store()
