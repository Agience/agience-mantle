"""Mantle's data layer over the LATTICE — the store, not a surface over one.

**MANTLE IS THE STANDALONE DB**: one SQLite file + FS-CAS content, opened in-process. Zero external
DB processes.

⚠ THIS MODULE'S HEADER USED TO DESCRIBE IT AS A COMPATIBILITY SURFACE — "lets the routers flip
`import db.lattice as lattice` -> `import db.lattice_api as lattice` with call sites unchanged: same
function names, same signatures". That framing was true during the flip and is now actively
misleading: there is nothing to be compatible WITH. The function names and `db`-first signatures are
simply this layer's shape, and where they still read as somebody else's they should be changed on
their own merits, not preserved for a caller that no longer exists.
[John, 2026-07-23: "leave one path. the only path. No constants, no fitting. no forcing."]

Two behaviours remain deliberately deferred, and they are gaps, not compatibility:
`_emit_artifact_change` (the change-event wire the stream router reads) and per-principal inline
content encryption — the lattice keys content on the collection origin-root (`FileContentCache`,
P9.3), and reconciling the two is §6d gap 4.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Type, TypeVar

try:                                    # both path styles, like the rest of the lattice package
    from mantle.db.lattice import open_lattice
except ImportError:                     # mantle dir itself on the path
    from mantle.db.lattice import open_lattice

try:
    from mantle.entities.artifact import Artifact as ArtifactEntity
except ImportError:
    from mantle.entities.artifact import Artifact as ArtifactEntity

try:                                    # the SHARED persistence boundary (crypto + change events)
    from mantle.db import doc_boundary as _boundary
except ImportError:
    from mantle.db import doc_boundary as _boundary

T = TypeVar("T")

# ─────────────────────────────────────────────────────────────────────────────
# Ordering keys — ported VERBATIM from db.store (pure math; no store in them).
# ─────────────────────────────────────────────────────────────────────────────
_ALPH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def after_key(a: Optional[str]) -> str:
    """Return a key strictly greater than *a* (or 'U' if None)."""
    if not a:
        return "U"
    last = _ALPH.find(a[-1])
    if last == -1 or last == len(_ALPH) - 1:
        return a + "U"
    return a[:-1] + _ALPH[last + 1]


def mid_key(a: Optional[str], b: Optional[str]) -> str:
    """Return a key strictly between (a, b)."""
    pad = "U"
    if a is None and b is None:
        return pad
    if a is None:
        a = ""
    if b is None:
        return a + pad
    i = 0
    while True:
        ca = _ALPH.find(a[i]) if i < len(a) else _ALPH.find(pad)
        cb = _ALPH.find(b[i]) if i < len(b) else len(_ALPH) - 1
        if ca + 1 < cb:
            return (a[:i] if i < len(a) else a) + _ALPH[(ca + cb) // 2]
        i += 1
        if i > max(len(a), len(b)) + 4:
            return a + _ALPH[1]


# ─────────────────────────────────────────────────────────────────────────────
# The standalone db handle
# ─────────────────────────────────────────────────────────────────────────────
class LatticeDatabase:
    """The `db` the routers pass around — Mantle's OWN store, opened in-process.

    One SQLite file (artifacts + graph, single `LatticeConn`, one proper-time sequence) + FS-CAS
    content beside it. No server, no external process — this handle IS the standalone database."""

    def __init__(self, path: str, *, origin: str, leaves: int = None):
        kwargs = {"origin": origin}
        if leaves is not None:
            kwargs["leaves"] = leaves
        L = open_lattice(path, **kwargs)
        self.artifacts = L.artifacts
        self.graph = L.graph
        self.conn = L.db
        self.origin = L.origin


def open_database(path: Optional[str] = None, *, origin: Optional[str] = None) -> LatticeDatabase:
    """Open (or create) standalone Mantle's database. Env-configurable, zero external processes.

    ⛔ SAY WHICH STORE WAS OPENED, ALWAYS. The default is the RELATIVE path `mantle-lattice.db`, so
    an unset `MANTLE_LATTICE_PATH` silently opens — or CREATES — a store in whatever the working
    directory happens to be. Measured 2026-07-31: mantle ran from the runtime dir, minted a 4 KB
    `agience-home/mantle-lattice.db`, and reported "Found 5 artifacts to reindex" while the real
    5.8 GB / 2.15 M-vertex store sat untouched on `D:`. Nothing errored, because a store with 5
    artifacts in it is a perfectly valid store.

    The startup log named the encryption key, the nonce secret, the issuer and all three URIs — and
    never the one path that decides what this node actually serves. That absence is what made the
    fault invisible; the log now carries it, absolute and resolved.
    """
    raw = path or os.getenv("MANTLE_LATTICE_PATH", "mantle-lattice.db")
    resolved = os.path.abspath(os.path.expanduser(str(raw)))
    origin = origin or os.getenv("MANTLE_ORIGIN") or os.getenv("EMBER_NODE_ID") or "mantle"

    exists = os.path.exists(resolved)
    size = os.path.getsize(resolved) if exists else 0
    logging.getLogger("mantle.db").info(
        "lattice store: %s (%s, %.1f MB) origin=%s  [MANTLE_LATTICE_PATH=%s]",
        resolved, "existing" if exists else "NEW — will be created",
        size / 1048576.0, origin, os.getenv("MANTLE_LATTICE_PATH") or "<unset, using default>")
    if not os.getenv("MANTLE_LATTICE_PATH"):
        logging.getLogger("mantle.db").warning(
            "MANTLE_LATTICE_PATH is unset — falling back to the relative default %r, which resolves "
            "against the CURRENT WORKING DIRECTORY (%s). If that is not the store you meant, this "
            "node will come up healthy and serve an empty universe.", raw, os.getcwd())
    return LatticeDatabase(resolved, origin=origin)


# ─────────────────────────────────────────────────────────────────────────────
# doc ↔ entity mapping (the lattice keeps `id` directly — no `_key` indirection)
# ─────────────────────────────────────────────────────────────────────────────
def _strip_nones(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_nones(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nones(v) for v in obj if v is not None]
    return obj


def to_lattice_doc(entity: Any) -> Dict[str, Any]:
    """Entity → lattice doc. Mirrors `to_store_doc` minus the `_key` rename. Artifact content is
    envelope-encrypted at this boundary via the SHARED `db.doc_boundary` implementation — same
    MEC1 wire format and origin-root key principal as the store path, so the flip changes the
    store, never the security posture. (Externalizing content to the encrypted FS-CAS
    (`FileContentCache`) is the follow-on; it changes this boundary's internals only.)"""
    ent_type = getattr(entity, "PREFIX", None)
    if hasattr(entity, "to_dict") and callable(entity.to_dict):
        doc = entity.to_dict()
    elif isinstance(entity, dict):
        doc = dict(entity)
    else:
        doc = {k: v for k, v in entity.__dict__.items() if not k.startswith("_")}
    if not ent_type:
        raise ValueError("to_lattice_doc: unable to resolve entity type (PREFIX)")
    doc = _strip_nones(doc)
    doc["_type"] = ent_type
    if ent_type == "Artifact":
        _boundary.encrypt_artifact_content(doc)
    return doc


# Lattice-internal columns that must never leak into an entity.
_LATTICE_INTERNAL = ("_origin", "_seq", "_rev", "_fp", "_type")


def from_lattice_doc(raw: Optional[Dict[str, Any]], cls: Type[T]) -> Optional[T]:
    if not raw:
        return None
    doc = {k: v for k, v in raw.items() if k not in _LATTICE_INTERNAL}
    # Single read chokepoint — every artifact read funnels through here, so all callers see
    # plaintext (or a loud failure; never ciphertext dressed as content).
    if getattr(cls, "PREFIX", None) == "Artifact":
        _boundary.decrypt_artifact_content(doc)
    return cls.from_dict(doc)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact CRUD — same names + `db`-first signatures as db.store
# ─────────────────────────────────────────────────────────────────────────────
def _stamp_origin_root(db: LatticeDatabase, entity: ArtifactEntity) -> None:
    """Stamp `origin_root` by INHERITING it from the parent collection (store-identical).

    ⛔ AN EARLIER DRAFT SELF-ROOTED CHILDREN (`root_id or id`). That silently gave every child its
    OWN key principal — content encrypted under a fresh principal no grant reaches, so the very
    first router-path artifact write was GrantDenied. The principal must be the COLLECTION's
    immutable origin root (single point-read; a legacy unstamped parent falls back to the one-time
    lineage walk). A top-level artifact IS its own root."""
    if getattr(entity, "origin_root", None):
        return
    collection_id = getattr(entity, "collection_id", None)
    if not collection_id:
        entity.origin_root = getattr(entity, "id", None)     # top-level artifact IS the root
        return
    try:
        parent = db.artifacts.get_artifact(collection_id)
        inherited = (parent or {}).get("origin_root")
        if inherited:
            entity.origin_root = inherited
            return
        entity.origin_root = get_origin_root(db, collection_id)   # legacy parent: walk once
    except Exception:
        pass                                   # unstamped is VISIBLE; never guess a principal


def create_artifact(db: LatticeDatabase, entity: ArtifactEntity) -> ArtifactEntity:
    _stamp_origin_root(db, entity)
    db.artifacts.put_artifact(to_lattice_doc(entity))
    _boundary.emit_artifact_change(entity, "artifact.created")
    return entity


def get_artifact(db: LatticeDatabase, artifact_id: str) -> Optional[ArtifactEntity]:
    """Fetch a single artifact version by its id (version id)."""
    return from_lattice_doc(db.artifacts.get_artifact(artifact_id), ArtifactEntity)


def update_artifact(db: LatticeDatabase, entity: ArtifactEntity) -> Optional[ArtifactEntity]:
    db.artifacts.put_artifact(to_lattice_doc(entity))
    _boundary.emit_artifact_change(entity, "artifact.updated")
    return entity


def delete_artifact(db: LatticeDatabase, artifact_id: str) -> bool:
    """Hard-delete a single artifact version document."""
    try:
        db.artifacts.delete_artifact(artifact_id)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# BRICK 2 — drafts / version history / children (same names + signatures as db.store)
#
# Ordering note: db.store sorted versions by `created_time DESC` — a CLAIM stamped by the writer.
# The lattice orders a lineage by `(_origin, _seq)` (proper time, gap-free, globally unique), which
# is what `versions_of` returns oldest-first. "Newest" here = last in proper time — the honest
# ordering, same intent as the store sort without trusting the writer's clock.
#
# Containment note: store modeled containment as an edge with `relationship == null`. On the
# lattice every edge carries a label; containment IS the label `_CONTAINS`. The collections brick
# writes them; children reads count them.
# ─────────────────────────────────────────────────────────────────────────────
_CONTAINS = "contains"


def _versions(db: LatticeDatabase, root_id: str):
    """A lineage's docs, oldest → newest in proper time."""
    return db.artifacts.versions_of(root_id) or []


def get_draft_artifact(db: LatticeDatabase, root_id: str,
                       collection_id: str) -> Optional[ArtifactEntity]:
    """Return the single draft record for (root_id, collection_id) if any."""
    for doc in _versions(db, root_id):
        if doc.get("state") == "draft" and doc.get("collection_id") == collection_id:
            return from_lattice_doc(doc, ArtifactEntity)
    return None


def get_latest_committed_artifact(db: LatticeDatabase, root_id: str,
                                  collection_id: Optional[str] = None) -> Optional[ArtifactEntity]:
    """Newest committed version for *root_id* (optionally restricted to a collection)."""
    for doc in reversed(_versions(db, root_id)):                 # newest first in proper time
        if doc.get("state") != "committed":
            continue
        if collection_id and doc.get("collection_id") != collection_id:
            continue
        return from_lattice_doc(doc, ArtifactEntity)
    return None


def get_current_in_collection(db: LatticeDatabase, collection_id: str,
                              root_id: str) -> Optional[ArtifactEntity]:
    """The *current* artifact for (collection, root): the draft if one exists, else the latest
    committed version in this collection."""
    return (get_draft_artifact(db, root_id, collection_id)
            or get_latest_committed_artifact(db, root_id, collection_id))


def list_version_history(db: LatticeDatabase, root_id: str) -> list:
    """All committed versions for a root_id, newest first (proper-time order)."""
    return [from_lattice_doc(doc, ArtifactEntity)
            for doc in reversed(_versions(db, root_id)) if doc.get("state") == "committed"]


def list_draft_artifacts(db: LatticeDatabase, collection_id: str) -> list:
    """Every draft artifact in a collection (used by commit)."""
    rows = db.artifacts.list_artifacts(collection_id=collection_id, state="draft")
    return [from_lattice_doc(dict(doc), ArtifactEntity) for doc in rows if doc]


def count_children(db: LatticeDatabase, root_id: str) -> int:
    """Count outbound containment edges from an artifact (as a container).

    ⛔ Same defect as `has_children` above: `except Exception: return 0` served a fault as a
    measurement. Removed 2026-07-31.
    """
    return len(db.graph.edges_of(root_id, label=_CONTAINS, direction="out"))


def has_children(db: LatticeDatabase, root_id: str) -> bool:
    """Whether an artifact has any containment children.

    ⛔ THIS CAUGHT EVERY EXCEPTION AND RETURNED `False` — served straight to clients as
    `doc["has_children"]` by `artifacts_router`. A transient DB fault therefore made a POPULATED
    container look empty to every caller, and the router's next line then forced `child_count` to 0
    without even attempting the count. Nothing distinguished that from a genuinely childless
    artifact. [John, 2026-07-31: fail loudly — let it 5xx.]

    "I could not ask" is not "no". A read that cannot run is an error, not a negative answer.
    """
    return bool(db.graph.edges_of(root_id, label=_CONTAINS, direction="out", limit=1))


# ─────────────────────────────────────────────────────────────────────────────
# BRICK 3 — collections (Group 2): container CRUD, membership edges, order keys
#
# "A container IS an artifact" (`entities/collection.py`: `Collection = Artifact`) — container docs
# live in the same store, discriminated by content_type, so the collection CRUD IS the artifact CRUD.
#
# Membership = an edge collection → root. Relation mapping (lattice-native): **the LABEL is the
# relation kind** — containment is the label `contains` (what store expressed as
# `relationship == null`), and a typed relationship (e.g. "operator") is its own label. The lattice
# edge carries `order_key` / `is_origin` / `propagate` first-class; membership resolution walks the
# root's lineage in proper time exactly like brick 2.
# ─────────────────────────────────────────────────────────────────────────────
CollectionEntity = ArtifactEntity          # container-as-artifact — one entity, one store


def create_collection(db: LatticeDatabase, entity: Any) -> Any:
    """A container create is an artifact create — same chokepoint, same change event
    ("everything is an artifact, and deserves an event")."""
    db.artifacts.put_artifact(to_lattice_doc(entity))
    _boundary.emit_artifact_change(entity, "artifact.created")
    return entity


def get_collection_by_id(db: LatticeDatabase, id: str) -> Optional[Any]:
    return from_lattice_doc(db.artifacts.get_artifact(id), CollectionEntity)


def update_collection(db: LatticeDatabase, entity: Any) -> Optional[Any]:
    db.artifacts.put_artifact(to_lattice_doc(entity))
    return entity


def delete_collection(db: LatticeDatabase, id: str) -> bool:
    try:
        db.artifacts.delete_artifact(id)
        return True
    except Exception:
        return False


def _eprop(edge: Dict[str, Any], key: str) -> Any:
    """Edge attribute, column-first then props — robust to which side of the row it landed on."""
    v = edge.get(key)
    if v is None:
        v = (edge.get("props") or {}).get(key)
    return v


# The lattice `propagate` column is TEXT; the API convention is a list of action names (or None =
# unrestricted, [] = nothing propagates). Serialize at the boundary. A non-JSON string (e.g. the
# substrate's compact `"r"` on ember creation edges) passes through untouched.
def _ser_propagate(mask: Any) -> Any:
    return json.dumps(mask) if isinstance(mask, (list, tuple)) else mask


def _prop_mask(edge: Dict[str, Any]) -> Any:
    v = _eprop(edge, "propagate")
    if isinstance(v, str) and v.startswith("["):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _membership_edges(db: LatticeDatabase, collection_id: str) -> list:
    """Every outbound edge from a container is a membership (any label — the label is the relation)."""
    return db.graph.edges_of(collection_id, direction="out") or []


def get_last_order_key(db: LatticeDatabase, collection_id: str) -> Optional[str]:
    """The maximum `order_key` currently used in this collection."""
    keys = [k for k in (_eprop(e, "order_key") for e in _membership_edges(db, collection_id)) if k]
    return max(keys) if keys else None


def add_artifact_to_collection(db: LatticeDatabase, collection_id: str, root_id: str,
                               order_key: Optional[str] = None, *,
                               origin: bool = True,
                               propagate: Optional[list] = None,
                               relationship: Optional[str] = None) -> bool:
    """Upsert the membership edge collection → root. No `order_key` → placed at the end.
    `origin` marks the creation edge (grants propagate through it); `relationship=None` is
    containment (`contains`), else the relationship IS the edge label."""
    if order_key is None:
        order_key = after_key(get_last_order_key(db, collection_id))
    try:
        db.graph.add_edge(collection_id, root_id, (relationship or _CONTAINS),
                          {"order_key": order_key, "is_origin": bool(origin),
                           "propagate": _ser_propagate(propagate), "root_id": root_id})
        return True
    except Exception:
        return False


def _find_membership(db: LatticeDatabase, collection_id: str, root_id: str) -> Optional[Dict[str, Any]]:
    for e in _membership_edges(db, collection_id):
        if e.get("dst") == root_id:
            return e
    return None


def remove_artifact_from_collection(db: LatticeDatabase, collection_id: str, root_id: str) -> bool:
    """Delete the membership edge for (collection, root). Idempotent."""
    e = _find_membership(db, collection_id, root_id)
    if e is None:
        return True
    try:
        db.graph.delete_edge(collection_id, root_id, e.get("label"))
        return True
    except Exception:
        return False


def set_edge_order_key(db: LatticeDatabase, collection_id: str, root_id: str,
                       new_order_key: str) -> bool:
    """Update the order_key on a single membership edge (upsert by (src,dst,label) preserves the
    rest of the edge's attributes)."""
    e = _find_membership(db, collection_id, root_id)
    if e is None:
        return False
    props = dict(e.get("props") or {})
    props.update({"order_key": new_order_key,
                  "is_origin": _eprop(e, "is_origin"),
                  "propagate": _eprop(e, "propagate")})
    try:
        db.graph.add_edge(collection_id, root_id, e.get("label"), props)
        return True
    except Exception:
        return False


def reorder_collection_artifacts(db: LatticeDatabase, collection_id: str,
                                 ordered_root_ids: list) -> int:
    """Assign monotonically increasing order_keys along the given sequence. Returns edges updated."""
    prev: Optional[str] = None
    updated = 0
    for rid in ordered_root_ids or []:
        key = after_key(prev)
        if set_edge_order_key(db, collection_id, rid, key):
            updated += 1
        prev = key
    return updated


def list_collection_artifacts(db: LatticeDatabase, collection_id: str, *,
                              include_archived: bool = False,
                              draft_workspace_id: Optional[str] = None) -> list:
    """Resolve a collection's contents via the membership edges: for each edge, the CURRENT version
    of its root — draft-preferred in this collection, else newest in-collection, else the latest
    committed version anywhere (published cross-collection link), else (gated by the caller) a draft
    homed in `draft_workspace_id`. Each dict carries `order_key` / `has_committed_version` /
    `origin` / `propagate` / `relationship` from the edge; sorted by order_key."""
    out = []
    for e in _membership_edges(db, collection_id):
        root = e.get("dst")
        versions = _versions(db, root)
        in_coll = [v for v in versions if v.get("collection_id") == collection_id]
        current = (next((v for v in reversed(in_coll) if v.get("state") == "draft"), None)
                   or (in_coll[-1] if in_coll else None))
        committed = [v for v in versions if v.get("state") == "committed"]
        chosen = current or (committed[-1] if committed else None)
        if chosen is None and draft_workspace_id:
            chosen = next((v for v in reversed(versions)
                           if v.get("state") == "draft"
                           and v.get("collection_id") == draft_workspace_id), None)
        if chosen is None:
            continue
        if not include_archived and chosen.get("state") == "archived":
            continue
        d = {k: v for k, v in chosen.items() if k not in _LATTICE_INTERNAL}
        # Non-strict on the list path: one undecryptable row must not fail the page —
        # its content drops (visibly incomplete), never ciphertext dressed as content.
        _boundary.decrypt_artifact_content(d, strict=False)
        d["order_key"] = _eprop(e, "order_key")
        d["has_committed_version"] = bool(committed)
        d["origin"] = bool(_eprop(e, "is_origin"))
        d["propagate"] = _prop_mask(e)
        d["relationship"] = None if e.get("label") == _CONTAINS else e.get("label")
        out.append(d)
    out.sort(key=lambda d: (d.get("order_key") is None, d.get("order_key") or ""))
    return out


def count_other_containers_for_root(db: LatticeDatabase, root_id: str,
                                    excluding_collection_id: str) -> int:
    """How many OTHER containers still hold this root (shared-not-owned check before a destroy).

    ⛔ NO error-swallow here: this count GATES DESTRUCTION. Returning 0 on a store error would
    read as "not shared" and let a delete proceed on an unknown share count — the fail-OPEN
    polarity. A store error propagates; the caller aborts loudly (pinned by
    `test_scoped_deletion_and_urls.test_container_count_failure_fails_safe`)."""
    edges = db.graph.edges_of(root_id, direction="in") or []
    return sum(1 for e in edges if e.get("src") != excluding_collection_id)


# ─────────────────────────────────────────────────────────────────────────────
# BRICK 5 — origin lineage & the propagation light-cone (the rest of Group 3)
#
# The origin chain is the creation lineage: `is_origin` + containment edges, immutable after
# creation. `get_origin_root` is the encrypted-search PRINCIPAL (the master-key root — stable under
# grant churn); `list_origin_descendants` is the grant light-cone: BFS down origin containment,
# pruned by each edge's `propagate` action mask. Plus the artifact lifecycle utilities that ride
# these edges (archive / batch-commit / delete-by-root / remove-edges).
# ─────────────────────────────────────────────────────────────────────────────
try:
    from mantle.entities.relation import derive_relation
except ImportError:
    from mantle.entities.relation import derive_relation


def get_edge(db: LatticeDatabase, collection_id: str, root_id: str) -> Optional[Dict[str, Any]]:
    """The raw membership edge (store-shaped keys), or None."""
    e = _find_membership(db, collection_id, root_id)
    if e is None:
        return None
    rel = None if e.get("label") == _CONTAINS else e.get("label")
    return {
        "root_id": root_id,
        "order_key": _eprop(e, "order_key"),
        "origin": bool(_eprop(e, "is_origin")),
        "propagate": _prop_mask(e),
        "relationship": rel,
        "relation": derive_relation(origin=bool(_eprop(e, "is_origin")), relationship=rel),
    }


def add_artifacts_to_collection_batch(db: LatticeDatabase, collection_id: str,
                                      root_id_order_pairs: list, *,
                                      origin: bool = True,
                                      propagate: Optional[list] = None) -> bool:
    """Batch upsert of membership edges; pairs are (root_id, order_key)."""
    ok_all = True
    for rid, order_key in (root_id_order_pairs or []):
        if not add_artifact_to_collection(db, collection_id, rid, order_key,
                                          origin=origin, propagate=propagate):
            ok_all = False
    return ok_all


def remove_all_edges_for_root(db: LatticeDatabase, root_id: str) -> int:
    """Delete every edge pointing AT this root. Returns count."""
    removed = 0
    try:
        for e in (db.graph.edges_of(root_id, direction="in") or []):
            if db.graph.delete_edge(e.get("src"), root_id, e.get("label")):
                removed += 1
    except Exception:
        return removed
    return removed


def get_origin_parent(db: LatticeDatabase, root_id: str):
    """``(parent_id, propagate_mask)`` via the creation edge, or None (already a root)."""
    try:
        for e in (db.graph.edges_of(root_id, label=_CONTAINS, direction="in") or []):
            if _eprop(e, "is_origin"):
                return (e.get("src"), _prop_mask(e))
    except Exception:
        pass
    return None


def get_origin_root(db: LatticeDatabase, artifact_id: str) -> str:
    """Walk the immutable origin chain to the top-most ancestor — the stable principal id.

    ⛔ `max_depth=32` REMOVED 2026-07-30. `visited` is the real termination guard and the graph
    is finite, so the walk always terminates without it. The cap was a bare claim that origin
    chains are never deeper than 32 — and on a longer chain it returned the DEEPEST ID REACHED
    as if it were the root. That id is the stable principal used for cell-key derivation, so a
    truncated walk meant a WRONG ENCRYPTION KEY, silently. Depth was described as 4, 10, 25, 32
    and 64 across five files for this one lattice; nothing measured any of them."""
    current = artifact_id
    visited = {current}
    while True:
        parent = get_origin_parent(db, current)
        if not parent:
            return current
        parent_id = parent[0]
        if not parent_id or parent_id in visited:
            return current
        visited.add(parent_id)
        current = parent_id


def list_origin_descendants(db: LatticeDatabase, root_ids: list, action: str) -> set:
    """The propagation light-cone: every id reachable from *root_ids* via origin containment
    edges whose `propagate` mask allows *action* (None mask = unrestricted). BFS,
    globally-unique vertices, seeds excluded.

    ⛔ `max_depth=4` REMOVED 2026-07-30. `seen` is the real termination guard — every vertex is
    admitted once and the graph is finite, so the BFS always drains. The cap silently truncated
    the light-cone, so a grant more than 4 levels up produced a FALSE DENY indistinguishable
    from "no such artifact"."""
    if not root_ids:
        return set()
    seen = set(root_ids)
    frontier = list(root_ids)
    out: set = set()
    while True:
        nxt = []
        for node in frontier:
            try:
                edges = db.graph.edges_of(node, label=_CONTAINS, direction="out") or []
            except Exception:
                continue
            for e in edges:
                if not _eprop(e, "is_origin"):
                    continue
                mask = _prop_mask(e)
                if mask is not None and action not in mask:
                    continue                       # prune: the whole subtree behind this edge
                dst = e.get("dst")
                if not dst or dst in seen:
                    continue
                seen.add(dst)
                out.add(dst)
                nxt.append(dst)
        if not nxt:
            break
        frontier = nxt
    return out


def get_relationship_target(db: LatticeDatabase, from_root_id: str,
                            relationship: str) -> Optional[str]:
    """The root_id behind the first outbound edge with this typed relationship label."""
    try:
        for e in (db.graph.edges_of(from_root_id, label=relationship, direction="out") or []):
            return e.get("dst")
    except Exception:
        pass
    return None


def batch_commit_drafts(db: LatticeDatabase, collection_id: str, artifact_ids: list,
                        committed_by: str, committed_time: str) -> int:
    """Flip a batch of in-collection drafts to `committed`. Returns docs updated."""
    updated = 0
    for aid in (artifact_ids or []):
        raw = db.artifacts.get_artifact(aid)
        if (raw is None or raw.get("collection_id") != collection_id
                or raw.get("state") != "draft"):
            continue
        raw["state"] = "committed"
        raw["modified_by"] = committed_by
        raw["modified_time"] = committed_time
        db.artifacts.put_artifact(raw)
        updated += 1
    return updated


def delete_artifacts_by_root(db: LatticeDatabase, root_id: str) -> list:
    """Hard-delete every version (draft + committed + archived) with this root. Returns ids."""
    deleted = []
    for v in _versions(db, root_id):
        vid = v.get("id")
        if not vid:
            continue
        try:
            db.artifacts.delete_artifact(vid)
            deleted.append(vid)
        except Exception:
            continue
    return deleted


def archive_artifact(db: LatticeDatabase, user_id: str, artifact_id: str) -> bool:
    """Soft delete: mark archived, gated on a `can_delete` grant on the parent collection."""
    raw = db.artifacts.get_artifact(artifact_id)
    if raw is None:
        return False
    parent_id = raw.get("collection_id")
    if not parent_id:
        return False
    grants = get_active_grants_for_principal_resource(
        db, grantee_id=user_id, resource_id=parent_id)
    if not any(getattr(g, "can_delete", False) for g in grants):
        return False
    raw["state"] = "archived"
    db.artifacts.put_artifact(raw)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# BRICK 4 — grants (Group 4): the CRUDEASIO authorization plane
#
# Everything is an artifact — a grant included. The lattice scoped grants by putting them in their own
# collection (COLLECTION_GRANTS); the lattice instead discriminates by a stamped
# `content_type = _GRANT_CT` on the doc (`Grant.from_dict` ignores unknown keys, so it never leaks
# into the entity), and the grant's lifecycle `state` (active / revoked / pending_accept) rides the
# vertex `state` filter — so "active grants" is a SQL-level cut, not a scan-and-sift.
#
# Expiry: store compared `expires_at > DATE_ISO8601(DATE_NOW())` as strings. `_unexpired` parses
# both sides when it can (tolerating 'Z' vs '+00:00' — the string compare misorders those) and
# falls back to the string compare only when parsing fails.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from mantle.entities.grant import Grant as GrantEntity
except ImportError:
    from mantle.entities.grant import Grant as GrantEntity

_GRANT_CT = "application/vnd.agience.grant+json"


def _to_grant_doc(entity: Any) -> Dict[str, Any]:
    d = to_lattice_doc(entity)
    d["content_type"] = _GRANT_CT          # the collection-scoping discriminator, doc-side only
    return d


def _grant_docs(db: LatticeDatabase, *, state: Optional[str] = None):
    return db.artifacts.list_artifacts(content_type=_GRANT_CT, state=state)


def _unexpired(doc: Dict[str, Any]) -> bool:
    exp = doc.get("expires_at")
    if not exp:
        return True
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    try:
        e = datetime.fromisoformat(exp.replace("Z", "+00:00"))
        if e.tzinfo is None:
            e = e.replace(tzinfo=timezone.utc)
        return e > now
    except (ValueError, AttributeError):
        return str(exp) > now.isoformat()


def create_grant(db: LatticeDatabase, entity: Any) -> Any:
    db.artifacts.put_artifact(_to_grant_doc(entity))
    return entity


def get_grant_by_id(db: LatticeDatabase, grant_id: str) -> Optional[Any]:
    raw = db.artifacts.get_artifact(grant_id)
    if raw is None or raw.get("content_type") != _GRANT_CT:   # preserve store's collection scoping
        return None
    return from_lattice_doc(raw, GrantEntity)


def update_grant(db: LatticeDatabase, entity: Any) -> Optional[Any]:
    db.artifacts.put_artifact(_to_grant_doc(entity))
    return entity


def get_active_grants_for_principal_resource(db: LatticeDatabase, grantee_id: str,
                                             resource_id: str) -> list:
    return [from_lattice_doc(d, GrantEntity) for d in _grant_docs(db, state="active")
            if d.get("grantee_id") == grantee_id
            and d.get("resource_id") == resource_id
            and _unexpired(d)]


def get_active_grants_for_grantee(db: LatticeDatabase, grantee_id: str,
                                  grantee_type: str = "api_key") -> list:
    return [from_lattice_doc(d, GrantEntity) for d in _grant_docs(db, state="active")
            if d.get("grantee_id") == grantee_id
            and d.get("grantee_type") == grantee_type
            and _unexpired(d)]


def get_active_collection_ids_for_user(db: LatticeDatabase, user_id: str) -> list:
    """The user's read light-cone: every resource an active, unexpired, readable user-grant reaches."""
    return [d["resource_id"] for d in _grant_docs(db, state="active")
            if d.get("grantee_id") == user_id
            and d.get("grantee_type") == "user"
            and d.get("can_read")
            and d.get("resource_id")
            and _unexpired(d)]


def get_grants_for_collection(db: LatticeDatabase, collection_id: str) -> list:
    """ALL grants on a resource, any state (the admin/share management view)."""
    return [from_lattice_doc(d, GrantEntity) for d in _grant_docs(db)
            if d.get("resource_id") == collection_id]


def upsert_user_collection_grant(db: LatticeDatabase, *, user_id: str, collection_id: str,
                                 granted_by: str,
                                 can_create: bool = False, can_read: bool = True,
                                 can_update: bool = False, can_delete: bool = False,
                                 can_evict: bool = False, can_invoke: bool = False,
                                 can_add: bool = False, can_share: bool = False,
                                 can_admin: bool = False,
                                 name: Optional[str] = None):
    """Upsert a user→artifact grant. Returns ``(grant, changed)`` — logic ported verbatim from
    `db.store` (find the active user grant, update flags in place, else mint one)."""
    import uuid as _uuid

    existing = get_active_grants_for_principal_resource(
        db, grantee_id=user_id, resource_id=collection_id)
    user_grant = next((g for g in existing if g.grantee_type == "user"), None)
    new_flags = dict(
        can_create=can_create, can_read=can_read, can_update=can_update,
        can_delete=can_delete, can_evict=can_evict, can_invoke=can_invoke,
        can_add=can_add, can_share=can_share, can_admin=can_admin,
    )
    if user_grant is not None:
        changed = any(getattr(user_grant, k) != v for k, v in new_flags.items())
        if name is not None:
            changed = changed or (user_grant.name != name)
        if not changed:
            return user_grant, False
        for k, v in new_flags.items():
            setattr(user_grant, k, v)
        if name is not None:
            user_grant.name = name
        return (update_grant(db, user_grant) or user_grant), True
    grant = GrantEntity(
        id=str(_uuid.uuid4()),
        resource_id=collection_id,
        grantee_type="user",
        grantee_id=user_id,
        granted_by=granted_by,
        name=name,
        **new_flags,
    )
    create_grant(db, grant)
    return grant, True


def get_active_grants_by_key(db: LatticeDatabase, token: str) -> list:
    """Grant-key auth: hash the presented token, match `grantee_type == "grant_key"`."""
    try:
        from mantle.services import auth_service
    except ImportError:
        from mantle.services import auth_service
    return get_active_grants_for_grantee(
        db, grantee_id=auth_service.hash_api_key(token), grantee_type="grant_key")


def get_active_grant_key_grants_for_collection(db: LatticeDatabase, collection_id: str) -> list:
    return [from_lattice_doc(d, GrantEntity) for d in _grant_docs(db, state="active")
            if d.get("resource_id") == collection_id
            and d.get("grantee_type") == "grant_key"
            and _unexpired(d)]


def get_collections_by_owner_and_type(db: LatticeDatabase, owner_id: str,
                                      content_type: str) -> list:
    return get_containers_for_user(db, owner_id, content_type)


def get_containers_for_user(db: LatticeDatabase, user_id: str,
                            content_type: Optional[str] = None) -> list:
    """Containers reachable through the user's read light-cone, optionally one content_type.
    (the legacy store's untyped branch required `content_type != null` — preserved: a container IS typed.)"""
    out = []
    for cid in get_active_collection_ids_for_user(db, user_id):
        raw = db.artifacts.get_artifact(cid)
        if raw is None:
            continue
        ct = raw.get("content_type")
        if ct is None or ct == _GRANT_CT:
            continue
        if content_type is not None and ct != content_type:
            continue
        out.append(from_lattice_doc(raw, CollectionEntity))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BRICK 6 — commits (Group 5) + API keys / server credentials / server JWKs
#
# Same move as grants: each store side-collection becomes a stamped `content_type` plane in the
# one store. Commit provenance (`get_commit_by_id` / `get_commits_for_collection`) keeps store's
# RAW-dict return shape — the routers render those directly.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from mantle.entities.api_key import APIKey as APIKeyEntity
    from mantle.entities.server_credential import ServerCredential as ServerCredentialEntity
except ImportError:
    from mantle.entities.api_key import APIKey as APIKeyEntity
    from mantle.entities.server_credential import ServerCredential as ServerCredentialEntity

_COMMIT_CT = "application/vnd.agience.commit+json"
_COMMIT_ITEM_CT = "application/vnd.agience.commit-item+json"
_API_KEY_CT = "application/vnd.agience.api-key+json"
_SERVER_CRED_CT = "application/vnd.agience.server-credential+json"
_SERVER_JWK_CT = "application/vnd.agience.server-jwk+json"


def _put_typed(db: LatticeDatabase, entity: Any, ct: str) -> Any:
    d = to_lattice_doc(entity)
    d["content_type"] = ct
    db.artifacts.put_artifact(d)
    return entity


def _typed_docs(db: LatticeDatabase, ct: str):
    return db.artifacts.list_artifacts(content_type=ct)


def _get_typed(db: LatticeDatabase, id: str, ct: str, cls: Type[T]) -> Optional[T]:
    raw = db.artifacts.get_artifact(id)
    if raw is None or raw.get("content_type") != ct:
        return None
    return from_lattice_doc(raw, cls)


# ── commits ──────────────────────────────────────────────────────────────────
def create_commit(db: LatticeDatabase, commit: Any) -> Any:
    return _put_typed(db, commit, _COMMIT_CT)


def create_commit_items(db: LatticeDatabase, items: list) -> list:
    return [_put_typed(db, item, _COMMIT_ITEM_CT).id for item in (items or [])]


def get_commit_by_id(db: LatticeDatabase, commit_id: str) -> Optional[Dict[str, Any]]:
    """Raw commit dict (store returned the raw doc with `id` set — preserved)."""
    raw = db.artifacts.get_artifact(commit_id)
    if raw is None or raw.get("content_type") != _COMMIT_CT:
        return None
    return {k: v for k, v in raw.items() if k not in _LATTICE_INTERNAL}


def get_commits_for_collection(db: LatticeDatabase, collection_id: str) -> list:
    """Commits whose item set touches this collection, newest first (raw dicts)."""
    item_ids = {d.get("id") for d in _typed_docs(db, _COMMIT_ITEM_CT)
                if d.get("collection_id") == collection_id}
    out = []
    for c in _typed_docs(db, _COMMIT_CT):
        if set(c.get("item_ids") or []) & item_ids:
            out.append({k: v for k, v in c.items() if k not in _LATTICE_INTERNAL})
    out.sort(key=lambda c: c.get("timestamp") or "", reverse=True)
    return out


# ── API keys ─────────────────────────────────────────────────────────────────
def create_api_key(db: LatticeDatabase, entity: Any) -> Any:
    return _put_typed(db, entity, _API_KEY_CT)


def get_api_key_by_hash(db: LatticeDatabase, key_hash: str) -> Optional[Any]:
    """Active, unexpired key matching this hash (the auth hot path)."""
    for d in _typed_docs(db, _API_KEY_CT):
        if d.get("key_hash") == key_hash and d.get("is_active"):
            if not _unexpired(d):
                return None                        # store: an expired match ends the lookup
            return from_lattice_doc(d, APIKeyEntity)
    return None


def get_api_key_by_id(db: LatticeDatabase, id: str) -> Optional[Any]:
    return _get_typed(db, id, _API_KEY_CT, APIKeyEntity)


def get_api_keys_by_user(db: LatticeDatabase, user_id: str) -> list:
    return [from_lattice_doc(d, APIKeyEntity) for d in _typed_docs(db, _API_KEY_CT)
            if d.get("user_id") == user_id]


def update_api_key(db: LatticeDatabase, entity: Any) -> Optional[Any]:
    return _put_typed(db, entity, _API_KEY_CT)


def update_api_key_last_used(db: LatticeDatabase, key_id: str, timestamp: str) -> bool:
    raw = db.artifacts.get_artifact(key_id.split("/")[-1])
    if raw is None or raw.get("content_type") != _API_KEY_CT:
        return False
    raw["last_used_at"] = timestamp
    db.artifacts.put_artifact(raw)
    return True


def delete_api_key(db: LatticeDatabase, id: str) -> bool:
    try:
        db.artifacts.delete_artifact(id)
        return True
    except Exception:
        return False


# ── server credentials ───────────────────────────────────────────────────────
def create_server_credential(db: LatticeDatabase, entity: Any) -> Any:
    return _put_typed(db, entity, _SERVER_CRED_CT)


def get_server_credential_by_client_id(db: LatticeDatabase, client_id: str) -> Optional[Any]:
    for d in _typed_docs(db, _SERVER_CRED_CT):
        if d.get("client_id") == client_id and d.get("is_active"):
            return from_lattice_doc(d, ServerCredentialEntity)
    return None


def get_server_credential_by_id(db: LatticeDatabase, id: str) -> Optional[Any]:
    return _get_typed(db, id, _SERVER_CRED_CT, ServerCredentialEntity)


def get_server_credentials_by_user(db: LatticeDatabase, user_id: str) -> list:
    return [from_lattice_doc(d, ServerCredentialEntity) for d in _typed_docs(db, _SERVER_CRED_CT)
            if d.get("user_id") == user_id]


def get_all_server_credentials(db: LatticeDatabase) -> list:
    return [from_lattice_doc(d, ServerCredentialEntity) for d in _typed_docs(db, _SERVER_CRED_CT)]


def update_server_credential(db: LatticeDatabase, entity: Any) -> Optional[Any]:
    return _put_typed(db, entity, _SERVER_CRED_CT)


def update_server_credential_last_used(db: LatticeDatabase, cred_id: str, timestamp: str) -> bool:
    raw = db.artifacts.get_artifact(cred_id.split("/")[-1])
    if raw is None or raw.get("content_type") != _SERVER_CRED_CT:
        return False
    raw["last_used_at"] = timestamp
    db.artifacts.put_artifact(raw)
    return True


def delete_server_credential(db: LatticeDatabase, id: str) -> bool:
    try:
        db.artifacts.delete_artifact(id)
        return True
    except Exception:
        return False


# ── WHERE-index materialization bookkeeping ──────────────────────────────────
# A membership set: "this artifact version has been sent for WHERE indexing". Namespaced ids —
# the marker for artifact X must not collide with X itself in the shared id space.
_MATERIALIZED_CT = "application/vnd.agience.materialized-marker+json"


def is_materialized(db: LatticeDatabase, artifact_id: str) -> bool:
    raw = db.artifacts.get_artifact("materialized:" + artifact_id)
    return raw is not None and raw.get("content_type") == _MATERIALIZED_CT


def mark_materialized(db: LatticeDatabase, artifact_id: str) -> None:
    """Idempotent (upsert) — called wherever indexing is enqueued."""
    from datetime import datetime, timezone
    try:
        db.artifacts.put_artifact({
            "id": "materialized:" + artifact_id,
            "content_type": _MATERIALIZED_CT,
            "at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass


# ── from-import compat: the handful of names routers/services import directly ─
_WORKSPACE_CT = "application/vnd.agience.workspace+json"

#: store collection-name compat — `query_documents` routes each to its typed plane.
COLLECTION_GRANTS = "grants"
COLLECTION_ARTIFACTS = "artifacts"
_COLLECTION_TO_CT = {COLLECTION_GRANTS: _GRANT_CT}

#: The typed side-planes that share the vertex table. An "artifacts" query must NOT see them —
#: in store they were separate collections, and that scoping is part of the contract.
_SIDE_PLANE_CTS = frozenset({
    _GRANT_CT, _COMMIT_CT, _COMMIT_ITEM_CT, _API_KEY_CT,
    _SERVER_CRED_CT, _SERVER_JWK_CT, _MATERIALIZED_CT,
})

# The stream router imports the decrypt hook under this name; ONE boundary either way.
_decrypt_artifact_content = _boundary.decrypt_artifact_content


def query_documents(db: LatticeDatabase, cls: Type[T], collection_name: str,
                    filters: dict) -> list:
    """Equality-filtered scan of one typed plane (store's generic AQL helper).
    `artifacts` = every doc NOT in a typed side-plane (store's collection scoping preserved)."""
    if collection_name == COLLECTION_ARTIFACTS:
        docs = (d for d in db.artifacts.list_artifacts()
                if d.get("content_type") not in _SIDE_PLANE_CTS)
    else:
        ct = _COLLECTION_TO_CT.get(collection_name)
        if ct is None:
            raise ValueError("query_documents: unmapped collection %r on the lattice"
                             % (collection_name,))
        docs = _typed_docs(db, ct)
    out = []
    for d in docs:
        if all(d.get(k) == v for k, v in (filters or {}).items()):
            out.append(from_lattice_doc(d, cls))
    return out


def get_collections_by_owner_id(db: LatticeDatabase, owner_id: str) -> list:
    return get_containers_for_user(db, owner_id)


def get_artifacts_by_creator_id(db: LatticeDatabase, creator_id: str) -> list:
    """Non-archived artifacts by `created_by` (the person-card / owner-memory lookups)."""
    return [from_lattice_doc(d, ArtifactEntity)
            for d in db.artifacts.list_artifacts(created_by=creator_id)
            if d.get("content_type") not in _SIDE_PLANE_CTS]


def get_collection_ids_for_root(db: LatticeDatabase, root_id: str) -> list:
    """Every collection (excluding workspaces) holding an edge to this root."""
    out = []
    try:
        for e in (db.graph.edges_of(root_id, direction="in") or []):
            cid = e.get("src")
            if not cid or cid in out:
                continue
            col = db.artifacts.get_artifact(cid)
            if col is not None and col.get("content_type") == _WORKSPACE_CT:
                continue
            out.append(cid)
    except Exception:
        return out
    return out


def batch_get_collection_ids_for_roots(db: LatticeDatabase, root_ids: list) -> Dict[str, list]:
    return {r: get_collection_ids_for_root(db, r) for r in (root_ids or [])}


def list_committed_artifacts_by_context_content_type(db: LatticeDatabase, content_type: str, *,
                                                     created_by: Optional[str] = None) -> list:
    """COMMITTED artifacts whose ``context.content_type`` matches (trust-config loads).
    `created_by` narrows to one principal — the trust boundary: a label alone never confers
    trust, only provenance does.

    ⛔ THIS FULL-SCANNED THE LATTICE AND HUNG MANTLE'S BOOT. It passed neither `content_type` nor
    a limit, so `list_artifacts` emitted `SELECT doc FROM vertex WHERE json_extract(doc,'$.state')
    = ? AND created_by = ? ORDER BY id` — `state` is a JSON function (unindexed) and `created_by`
    has no index either, so SQLite read and sorted all 2.15M rows and this loop ran `json.loads`
    on every one of them. To find FIVE artifacts.

    MEASURED 2026-08-01 on 71 (5.8 GB lattice). `seed_platform_issuer_artifacts` runs inside the
    FastAPI lifespan, so the port never opened: py-spy showed the boot parked in exactly this call,
    and crystal logged `upstream 127.0.0.1:8082 unreachable` on a loop the whole time — a service
    that looks crashed while it is quietly scanning a table.

        indexed `ct` lookup          21.6 ms   -> the 5 issuer rows
        the scan this used to do     minutes   (~7.8 s of raw reads + 2.15M json.loads + a sort)

    `ct` IS indexed (`ix_v_ct`), and both callers already write the value they search for into the
    artifact's own `content_type` — issuers at `services/issuers.py:144`, secrets at
    `routers/secrets_router.py:230` (which even guards on `art.content_type != SECRET_CT`). So the
    narrowing is EXACT here, not a heuristic. Verified on the live store: all 5 issuer rows carry
    `ct`, `context.content_type` and `state='committed'` in agreement.

    The `context.content_type` test below STAYS and remains authoritative — `ct` narrows, context
    decides. A row whose two disagree is a writer bug, and it will now be excluded rather than
    silently changing what this returns. [[limit-bounds-output-not-work]]"""
    out = []
    for d in db.artifacts.list_artifacts(state="committed", created_by=created_by,
                                         content_type=content_type):
        ctx = d.get("context")
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except ValueError:
                ctx = {}
        if isinstance(ctx, dict) and ctx.get("content_type") == content_type:
            out.append(from_lattice_doc(d, ArtifactEntity))
    return out


# ── server JWK registry ──────────────────────────────────────────────────────
def upsert_server_jwk(db: LatticeDatabase, server_client_id: str, public_jwk: dict) -> None:
    db.artifacts.put_artifact({"id": server_client_id, "public_jwk": public_jwk,
                               "content_type": _SERVER_JWK_CT})


def get_server_jwk(db: LatticeDatabase, server_client_id: str) -> Optional[dict]:
    raw = db.artifacts.get_artifact(server_client_id)
    if raw is None or raw.get("content_type") != _SERVER_JWK_CT:
        return None
    return raw.get("public_jwk")


__all__ = [
    "LatticeDatabase", "open_database",
    "after_key", "mid_key",
    "to_lattice_doc", "from_lattice_doc",
    "create_artifact", "get_artifact", "update_artifact", "delete_artifact",
    "get_draft_artifact", "get_latest_committed_artifact", "get_current_in_collection",
    "list_version_history", "list_draft_artifacts",
    "count_children", "has_children",
    "CollectionEntity",
    "create_collection", "get_collection_by_id", "update_collection", "delete_collection",
    "get_last_order_key", "add_artifact_to_collection", "remove_artifact_from_collection",
    "set_edge_order_key", "reorder_collection_artifacts", "list_collection_artifacts",
    "count_other_containers_for_root",
    "GrantEntity",
    "create_grant", "get_grant_by_id", "update_grant",
    "get_active_grants_for_principal_resource", "get_active_grants_for_grantee",
    "get_active_collection_ids_for_user", "get_grants_for_collection",
    "upsert_user_collection_grant", "get_active_grants_by_key",
    "get_active_grant_key_grants_for_collection", "get_containers_for_user",
    "get_edge", "add_artifacts_to_collection_batch", "remove_all_edges_for_root",
    "get_origin_parent", "get_origin_root", "list_origin_descendants",
    "get_relationship_target",
    "batch_commit_drafts", "delete_artifacts_by_root", "archive_artifact",
    "create_commit", "create_commit_items", "get_commit_by_id", "get_commits_for_collection",
    "APIKeyEntity", "create_api_key", "get_api_key_by_hash", "get_api_key_by_id",
    "get_api_keys_by_user", "update_api_key", "update_api_key_last_used", "delete_api_key",
    "ServerCredentialEntity", "create_server_credential", "get_server_credential_by_client_id",
    "get_server_credential_by_id", "get_server_credentials_by_user",
    "get_all_server_credentials", "update_server_credential",
    "update_server_credential_last_used", "delete_server_credential",
    "upsert_server_jwk", "get_server_jwk",
    "get_collections_by_owner_and_type", "is_materialized", "mark_materialized",
]


def typed_method(artifacts, name):
    """The typed lattice-store method `name`, or None on a store that does not have it.

    ⚠ MOVED FROM `ember/surface/stats.py::_typed` ON 2026-07-31. It probes THIS module's own typed
    API, so it lived one repo away from the thing it was asking about — and `store/content_tier.py`
    imported ember's SERVE SURFACE to get it, which is why a content tier looked coupled to a stats
    page.

    Absence means "this store predates the typed rewrite" — it never means "the answer is empty".
    Every caller must therefore fall back to a path that can FAIL, never to a default."""
    fn = getattr(artifacts, name, None)
    return fn if callable(fn) else None
