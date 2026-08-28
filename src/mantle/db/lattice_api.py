"""Mantle's data layer over the lattice — the store itself, not a surface over one.

Mantle is the standalone database: one SQLite file plus FS-CAS content, opened in-process, with
no external DB process.

Two behaviours are deliberately deferred: `_emit_artifact_change` (the change-event wire the
stream router reads) and per-principal inline content encryption — the lattice keys content on
the collection origin-root (`FileContentCache`, P9.3), and reconciling the two is §6d gap 4.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional, Type, TypeVar

#: Module logger. `query_documents(unreadable="skip")` is the only user: a skipped document is a
#: fact the store noticed and the caller may not repeat, so it is said once here as well.
_log = logging.getLogger(__name__)

#: The one attenuation operator's column form. Safe to import at module scope from the data
#: layer: `mantle.attenuation` is stdlib-only and imports nothing else in `mantle` — that
#: dependency floor exists precisely so the enforcement points below the service layer can
#: reach it. Aliased with a leading underscore so it does not join this module's public
#: surface (`__all__`).
from mantle.attenuation import propagates as _propagates

#: Where the store lives when nobody says — `<BASE_DIR>/.data/mantle-lattice.db`, absolute.
#: `mantle.config` is stdlib-only and imports nothing else in `mantle`, the same floor
#: `mantle.attenuation` sits on, so the embeddable surface can reach it. Re-exported (not
#: re-spelled) because `db/backend.py` needs the same answer: one constant, two readers.
from mantle.config import DEFAULT_LATTICE_PATH  # noqa: F401  (re-export for db.backend)

try:                                    # both path styles, like the rest of the lattice package
    from mantle.db import open_lattice
except ImportError:                     # mantle dir itself on the path
    from mantle.db import open_lattice

#: The one reading of a doc that carries no `state` — see `db/constants.STATE_WHEN_ABSENT`. The
#: lineage predicates below match on it rather than on `doc.get("state")` so a stateless doc is
#: resolved the same way here, in the entity layer, and in the index segment map.
from mantle.db.constants import state_of

#: Raised by `graph.edges_of` when more edges exist than it was allowed to return. Imported by
#: NAME rather than caught as `Exception`, because several readers below deliberately absorb a
#: failed edge read (an unreadable row withholds reach — fail-closed) and truncation is the one
#: failure where absorbing it produces a WRONG answer rather than a conservative one. Every
#: `except Exception` around an `edges_of` call in this module re-raises this first.
from mantle.db.edge import EdgesTruncated
# Stdlib-only module (it says so), so importing it here adds no cycle. Needed as an
# `except` clause, which cannot be satisfied by a lazy in-function import.
from mantle.services.acting_principal import KeyCustodyDenied as _KeyCustodyDenied

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
# Ordering keys — ported verbatim from db.store (pure math; no store in them).
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

    The startup log always names the resolved, absolute database path alongside the encryption
    key, nonce secret, issuer, and URIs it already logs — the path that decides what this node
    actually serves is as visible at boot as everything else that matters.
    """
    raw = path or os.getenv("MANTLE_LATTICE_PATH", str(DEFAULT_LATTICE_PATH))
    resolved = os.path.abspath(os.path.expanduser(str(raw)))
    origin = origin or os.getenv("MANTLE_ORIGIN") or os.getenv("EMBER_NODE_ID") or "mantle"

    if raw == str(DEFAULT_LATTICE_PATH):
        # The default names a directory nobody has necessarily created yet — `.data/` is runtime
        # state, not a checked-in directory. SQLite will not create it, and the failure is an
        # opaque "unable to open database file". A caller who supplied a path gets no such
        # courtesy: a typo there should surface as a missing directory, not become one.
        os.makedirs(os.path.dirname(resolved), exist_ok=True)

    exists = os.path.exists(resolved)
    size = os.path.getsize(resolved) if exists else 0
    logging.getLogger("mantle.db").info(
        "lattice store: %s (%s, %.1f MB) origin=%s  [MANTLE_LATTICE_PATH=%s]",
        resolved, "existing" if exists else "NEW — will be created",
        size / 1048576.0, origin, os.getenv("MANTLE_LATTICE_PATH") or "<unset, using default>")
    # Only when the DEFAULT is what opened — a caller that passed `path` said which store it
    # meant, and an env var it never consulted is not news.
    if not path and not os.getenv("MANTLE_LATTICE_PATH"):
        logging.getLogger("mantle.db").warning(
            "MANTLE_LATTICE_PATH is unset — falling back to the install-root default %s. It does "
            "not follow the working directory, so it is the same store from wherever this node is "
            "started; if it is not the store you meant, this node will come up healthy and serve "
            "an empty universe.", resolved)
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
    envelope-encrypted at this boundary via the shared `db.doc_boundary` implementation — same
    MEC1 wire format and origin-root key principal as the store path, so the security posture is
    independent of which store is in use."""
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
    """Stamp `origin_root` by inheriting it from the parent collection (store-identical), falling
    back to a one-time lineage walk for a legacy parent. A top-level artifact is its own root."""
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


#: NOT `Optional`. The annotation said this could
#: return `None` and the body never could: one `return entity`, no error path. **Eight guards
#: across three files were written against that promise and not one of them could fire.**
#:
#: The guards were individually reasonable; the defect was one level down. A signature that
#: advertises a failure mode the body cannot produce does not make callers safer — it makes them
#: write dead code that reads as error handling.
#:
#: And it did not even cover the real failure: if `put_artifact` ever returned without
#: persisting, `is None` would not have caught it either. The only failure that reaches a client
#: is `put_artifact` RAISING, which propagates past every one of those guards.
def update_artifact(db: LatticeDatabase, entity: ArtifactEntity) -> ArtifactEntity:
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
        if state_of(doc) == "draft" and doc.get("collection_id") == collection_id:
            return from_lattice_doc(doc, ArtifactEntity)
    return None


def get_latest_committed_artifact(db: LatticeDatabase, root_id: str,
                                  collection_id: Optional[str] = None) -> Optional[ArtifactEntity]:
    """Newest committed version for *root_id* (optionally restricted to a collection)."""
    for doc in reversed(_versions(db, root_id)):                 # newest first in proper time
        if state_of(doc) != "committed":
            continue
        if collection_id and doc.get("collection_id") != collection_id:
            continue
        return from_lattice_doc(doc, ArtifactEntity)
    return None


def _current_in(docs: list, collection_id: str) -> Optional[Dict[str, Any]]:
    """The current doc for one collection out of an ALREADY-FETCHED lineage: the draft if one
    exists, else the newest committed version in that collection.

    Split out from `get_current_in_collection` so a caller asking the same question of several
    collections pays for the lineage once. Same predicates, same proper-time ordering."""
    for doc in docs:
        if state_of(doc) == "draft" and doc.get("collection_id") == collection_id:
            return doc
    for doc in reversed(docs):                                   # newest first in proper time
        if state_of(doc) == "committed" and doc.get("collection_id") == collection_id:
            return doc
    return None


def get_current_in_collection(db: LatticeDatabase, collection_id: str,
                              root_id: str) -> Optional[ArtifactEntity]:
    """The *current* artifact for (collection, root): the draft if one exists, else the latest
    committed version in this collection.

    One lineage fetch, not two: `get_draft_artifact` and `get_latest_committed_artifact` each
    walk `versions_of(root_id)`, so calling them in sequence reads the same lineage twice."""
    return from_lattice_doc(_current_in(_versions(db, root_id), collection_id), ArtifactEntity)


def get_current_in_any_collection(db: LatticeDatabase, root_id: str,
                                  collection_ids) -> Optional[ArtifactEntity]:
    """The current artifact for `root_id` in the FIRST of `collection_ids` that holds one.

    Collapses the collection axis: asking collection by collection re-reads the whole lineage once
    per collection — O(collections) round trips for an answer that depends on one lineage. The
    lineage is fetched once here and matched against the candidate collections in memory, in the
    order given, so the answer is the one the collection-at-a-time loop produced.

    The root axis is `get_current_in_any_collection_many`'s to collapse; this is the one-root
    form and stays the right call for a caller that holds one root."""
    ids = list(collection_ids or [])
    if not ids:
        return None
    docs = _versions(db, root_id)
    if not docs:
        return None
    for collection_id in ids:
        doc = _current_in(docs, collection_id)
        if doc is not None:
            return from_lattice_doc(doc, ArtifactEntity)
    return None


def _versions_many(db: LatticeDatabase, root_ids) -> Dict[str, list]:
    """Each distinct root's lineage docs, oldest → newest in proper time, in ONE store read.

    The plural of `_versions`. Roots with no versions are absent from the mapping, so a caller
    reads a miss the same way `_versions` reads `[]`."""
    return db.artifacts.versions_of_many(root_ids) or {}


def get_current_in_collection_many(db: LatticeDatabase, collection_id: str,
                                   root_ids) -> Dict[str, ArtifactEntity]:
    """`{root_id: current artifact}` for each named root as seen from ONE collection — the draft
    if that collection holds one, else the newest version committed there.

    The plural of `get_current_in_collection`, and the same two predicates in the same
    proper-time order. Every lineage arrives in one chunked read
    (`ArtifactStore.versions_of_many`), so a caller holding a page of roots pays for the read
    once instead of once per root.

    Roots are deduplicated. A root with no version in this collection is absent from the
    mapping — the same answer the singular form gives as `None`."""
    lineages = _versions_many(db, root_ids)
    out: Dict[str, ArtifactEntity] = {}
    for root_id, docs in lineages.items():
        doc = _current_in(docs, collection_id)
        if doc is not None:
            out[root_id] = from_lattice_doc(doc, ArtifactEntity)
    return out


def get_current_in_any_collection_many(db: LatticeDatabase, root_ids,
                                       collection_ids) -> Dict[str, ArtifactEntity]:
    """`{root_id: current artifact}` for each named root in the FIRST of `collection_ids` that
    holds one — "where can this caller see each of these roots", answered for the whole set.

    The batch primitive behind `collection_service.get_collection_artifacts_batch_global`, which
    keeps its `*_batch` name precisely because this exists: the siblings dropped that name while
    the store published no multi-root lineage read and a per-root round trip was the floor, and
    `ArtifactStore.versions_of_many` is what lifted it.

    Both axes collapse here. The lineages are one chunked read for every root, and the candidate
    collections are matched in memory per lineage, so a page of R roots over C accessible
    collections costs `ceil(R / IN_CHUNK)` statements rather than R x C. Order within
    `collection_ids` still decides which collection wins, so a root visible through more than one
    resolves exactly as the root-at-a-time loop resolved it.

    Roots are deduplicated, and a root visible through none of the candidates is absent."""
    ids = list(collection_ids or [])
    if not ids:
        return {}
    out: Dict[str, ArtifactEntity] = {}
    for root_id, docs in _versions_many(db, root_ids).items():
        for collection_id in ids:
            doc = _current_in(docs, collection_id)
            if doc is not None:
                out[root_id] = from_lattice_doc(doc, ArtifactEntity)
                break
    return out


def list_version_history(db: LatticeDatabase, root_id: str) -> list:
    """All committed versions for a root_id, newest first (proper-time order)."""
    return [from_lattice_doc(doc, ArtifactEntity)
            for doc in reversed(_versions(db, root_id)) if state_of(doc) == "committed"]


def list_draft_artifacts(db: LatticeDatabase, collection_id: str) -> list:
    """Every draft artifact in a collection (used by commit)."""
    rows = db.artifacts.list_artifacts(collection_id=collection_id, state="draft")
    return [from_lattice_doc(dict(doc), ArtifactEntity) for doc in rows if doc]


def count_children(db: LatticeDatabase, root_id: str) -> int:
    """Count outbound containment edges from an artifact (as a container).

    Exact or it raises: `edges_of` propagates `EdgesTruncated` rather than silently capping at
    1000 and returning that count for every container larger than it — a count that would be
    wrong in a way no caller could detect, the kind of "measurement I cannot justify" this store's
    discipline refuses.
    """
    return len(db.graph.edges_of(root_id, label=_CONTAINS, direction="out"))


def has_children(db: LatticeDatabase, root_id: str) -> bool:
    """Whether an artifact has any containment children.

    "I could not ask" is not "no". A read that cannot run is an error, not a negative answer.

    `partial_ok=True` because one row answers the question: this is the deliberate bounded peek
    `edges_of` keeps the flag for, not an accidental truncation.
    """
    return bool(db.graph.edges_of(root_id, label=_CONTAINS, direction="out", limit=1,
                                  partial_ok=True))


# ─────────────────────────────────────────────────────────────────────────────
# BRICK 3 — collections (Group 2): container CRUD, membership edges, order keys
#
# "A container IS an artifact" (`entities/collection.py`: `Collection = Artifact`) — container docs
# live in the same store, discriminated by content_type, so the collection CRUD is the artifact CRUD.
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


#: NOT `Optional`, same shape as `update_artifact` above: one
#: `return entity`, no error path, and callers testing it wrote dead guards.
def update_collection(db: LatticeDatabase, entity: Any) -> Any:
    """A container update is an artifact update — same chokepoint as `create_collection`.

    A rename or a description edit is the change a live tree is most likely to be showing
    stale, so it announces itself exactly like any other artifact write."""
    db.artifacts.put_artifact(to_lattice_doc(entity))
    _boundary.emit_artifact_change(entity, _boundary.UPDATED)
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


def order_fingerprint(db: LatticeDatabase, collection_id: str) -> int:
    """A version for this container's current child order — derived, never stored.

    Derived, not a counter: ordering lives in the edges' `order_key`, so the edges are the
    version — a counter beside them would be a second source of truth to keep in step. This reads
    the one that already exists.

    A hardcoded literal here would repeat what cost this surface a real guarantee: an
    `order_version` that is accepted but read nowhere, answered with a constant, lets two clients
    reordering the same container both succeed — the second silently discarding the first's
    arrangement — while a client echoing the constant back believes it is protected. A field that
    advertises a guarantee it does not keep is worse than not offering one.

    The fingerprint covers the sequence of member roots, not their `order_key` values: two states
    with the same members in the same order are the same state, however the keys are spelled, so a
    no-op reorder does not invalidate anyone's token. Adding or removing a child does change it —
    correctly, because a position echoed back across a membership change is stale.
    """
    pairs = sorted(
        ((_eprop(e, "order_key") or "", str(e.get("dst") or ""))
         for e in _membership_edges(db, collection_id)))
    payload = "\n".join(dst for _key, dst in pairs)
    # Six bytes: ~2.8e14 values, far inside JSON's safe integer range, and wide enough that a
    # collision — which would mean a MISSED conflict — is not a practical concern.
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:6], "big")


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
        current = (next((v for v in reversed(in_coll) if state_of(v) == "draft"), None)
                   or (in_coll[-1] if in_coll else None))
        committed = [v for v in versions if state_of(v) == "committed"]
        chosen = current or (committed[-1] if committed else None)
        if chosen is None and draft_workspace_id:
            chosen = next((v for v in reversed(versions)
                           if state_of(v) == "draft"
                           and v.get("collection_id") == draft_workspace_id), None)
        if chosen is None:
            continue
        if not include_archived and state_of(chosen) == "archived":
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
    """How many other containers still hold this root — the shared-not-owned check run before a
    destroy."""
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
    except EdgesTruncated:
        # "Delete EVERY edge pointing at this root" cannot be answered from a clipped list, and
        # reporting the count it managed would read as completion. `graph.delete_edges_touching`
        # is the uncapped form if a caller needs one.
        raise
    except Exception:
        return removed
    return removed


def get_origin_parent(db: LatticeDatabase, root_id: str):
    """``(parent_id, propagate_mask)`` via the creation edge, or None (already a root).

    `None` means "this is a root", which is an authority statement — a root is its own encrypted
    -search principal. A truncated read falling through to `None` would promote an artifact out of
    its parent's key scope, so truncation is raised rather than absorbed with the unreadable-row
    case below.
    """
    try:
        for e in (db.graph.edges_of(root_id, label=_CONTAINS, direction="in") or []):
            if _eprop(e, "is_origin"):
                return (e.get("src"), _prop_mask(e))
    except EdgesTruncated:
        raise
    except Exception:
        pass
    return None


def get_origin_root(db: LatticeDatabase, artifact_id: str) -> str:
    """Walk the immutable origin chain to the top-most ancestor — the stable principal id."""
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


#: How far an origin chain may run before the lattice is declared malformed. A containment chain is
#: a few levels deep in practice; a number this size is reached only by a cycle the edge writer
#: should have made impossible.
ORIGIN_WALK_CEILING = 10_000


def origin_chain(db: LatticeDatabase, artifact_id: str, action: str,
                 *, root_id: Optional[str] = None, ceiling: int = ORIGIN_WALK_CEILING):
    """The resources a grant could sit on to reach `artifact_id` under `action`, nearest first.

    Yields the artifact, then its root, then each origin ancestor — and STOPS at the first edge
    whose propagate mask does not carry `action`.

    That mask is the attenuation, and it is why this is a walk rather than a lookup. A grant does
    not reach an artifact because it names an ancestor; it reaches it because every edge on the
    path between them carries the action. `list_origin_descendants` prunes the subtree behind such
    an edge on the way DOWN; this stops at the same edge on the way UP, through the same
    `attenuation.propagates` operator, so the two directions cannot disagree about which edges
    conduct.

    It answers WHERE to look, not whether the answer is yes: a caller supplies its own notion of
    "does this principal hold a grant on this resource", because that differs between a user (a
    ledger lookup) and a grant key (a bundle already resolved and masked at authentication). One
    walk, two grant sources.

        for resource in origin_chain(db, artifact_id, "read"):
            g = my_grants_on(resource)
            if g is not None:
                return g

    A chain that does not terminate raises rather than returning what it managed to collect: a
    truncated authorization answer is not a smaller one, it is a different one.
    """
    seen = set()
    first = str(artifact_id)
    yield first
    seen.add(first)

    # `root_id` is a parameter because every caller already holds the document — re-reading it here
    # would put a second store read on an authorization path that runs per artifact. The fallback
    # exists for a caller that has only an id.
    cursor = str(root_id) if root_id else None
    if cursor is None:
        try:
            doc = db.artifacts.get_artifact(first)
        except Exception:  # noqa: BLE001 — store reads raise broadly; an unreadable doc has no chain
            doc = None
        cursor = str((doc or {}).get("root_id") or first)
    if cursor not in seen:
        yield cursor
        seen.add(cursor)

    while True:
        if len(seen) > ceiling:
            raise OriginChainUnterminated(
                "origin chain from %r did not terminate within %d hops; the lattice is malformed"
                % (artifact_id, ceiling))
        parent = get_origin_parent(db, cursor)
        if parent is None:
            return
        parent_id, mask = parent
        if not parent_id or str(parent_id) in seen:
            return
        if not _propagates(mask, action):
            return                      # attenuated: nothing above this edge reaches through it
        parent_id = str(parent_id)
        yield parent_id
        seen.add(parent_id)
        cursor = parent_id


class OriginChainUnterminated(RuntimeError):
    """An origin chain ran past its ceiling. Raised rather than truncated, because a partial chain
    would answer a narrower authorization question than the one that was asked."""


def list_origin_descendants(db: LatticeDatabase, root_ids: list, action: str) -> set:
    """The propagation light-cone: every id reachable from *root_ids* via origin containment
    edges whose `propagate` mask allows *action* (None mask = unrestricted). BFS,
    globally-unique vertices, seeds excluded.

    The per-edge prune is `attenuation.propagates` rather than a local membership test. Spelled
    inline as `mask is not None and action not in mask` it would be one rule written twice, here and
    in `services.dependencies.check_access`, which is the shape the attenuation module exists to
    stop. `propagates` decodes the column through `Mask` and asks `allows`, and the decoder is
    proved bit-for-bit equal to that expression across every
    known column shape × every action in `tests/test_attenuation_algebra.py`
    (`test_the_decoder_reproduces_the_lattice_column_on_every_known_shape`), so no live edge
    changes meaning: NULL still propagates everything, `'[]'` and the compact `"r"` form still
    propagate nothing.

    One deliberate difference, in the fail-closed direction: an *unknown* action name propagates
    nothing, because `Mask.allows` answers False for any verb outside CRUDEASIO. A NULL mask
    would instead pass everything, since `action not in None` never runs — the `is not None`
    check short-circuits first. Both in-tree callers already reject unknown actions before
    reaching here (`lightcone.resolve` returns the empty set, `check_access` raises 400), so this
    only closes a hole for a future caller that does not."""
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
            except EdgesTruncated:
                # Raised rather than swallowed. This walk is the light cone: `lightcone.resolve`
                # seeds from it and hands the result to `OracleService` to derive content keys.
                # `except Exception: continue` suits an unreadable edge row, where dropping a node
                # under-reaches and is fail-closed; a truncated read means "there are more members
                # and I did not look at them", and continuing past it
                # produces an authorization answer that is quietly different from the one the graph
                # supports. Nothing above this can distinguish that from a genuinely small
                # container, so it has to travel.
                raise
            except Exception:
                continue
            for e in edges:
                if not _eprop(e, "is_origin"):
                    continue
                if not _propagates(_prop_mask(e), action):
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
    """The root_id behind the first outbound edge with this typed relationship label.

    "First" is now a defined thing: `edges_of` orders by `edge_key`, which is
    `blake2b(src ‖ dst ‖ label)` and therefore identical on every node. Before that ordering this
    function returned whichever row SQLite visited first, so a node could disagree with its peer
    about a typed relationship's target while both held the same edges.
    """
    try:
        for e in (db.graph.edges_of(from_root_id, label=relationship, direction="out") or []):
            return e.get("dst")
    except EdgesTruncated:
        raise
    except Exception:
        pass
    return None


def _feed_doc(raw: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """A stored doc reshaped into the descriptor the change feed carries.

    The body never rides the feed, whichever side the doc came from. A doc read back out of the
    store carries `content` as ciphertext (`to_lattice_doc` envelope-encrypts on the way in), so
    forwarding it verbatim hands every subscriber a blob it holds no key for; an entity from a
    write path carries the plaintext instead, because `to_lattice_doc` encrypts a fresh dict and
    leaves the entity's own `content` untouched. Neither belongs on a wire that fans out to every
    selected subscriber and, where a durable log is installed, into an un-ACL'd `event_log` row.

    `event_bus.redacted_artifact` holds the rule and `event_bus.publish_event` enforces it at the
    seam, so this is a local spelling of one decision rather than a second copy of it. The
    identity, container and state are what a subscriber acts on; a reader with custody fetches
    the content through the read chokepoint that decrypts it.
    """
    from mantle.events.event_bus import redacted_artifact
    d = redacted_artifact(raw)
    d.update(overrides)
    return d


def batch_commit_drafts(db: LatticeDatabase, collection_id: str, artifact_ids: list,
                        committed_by: str, committed_time: str) -> int:
    """Flip a batch of in-collection drafts to `committed`. Returns docs updated.

    Writes the doc directly rather than through `update_artifact`, so the announcement is made
    here — one event per document, because a subscriber filters per artifact and a single
    batch-shaped event would reach nobody watching one of the artifacts in it. Emitted after
    each write returns, matching how the boundary orders itself around a put."""
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
        _boundary.emit_artifact_change(_feed_doc(raw), _boundary.UPDATED)
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
    """Soft delete: mark archived, gated on a `can_delete` grant on the parent collection.

    Announced as `artifact.deleted`, not `artifact.updated`: every container read filters
    archived versions out (`_current_in`, `list_collection_artifacts`), so from a subscriber's
    side the artifact has left the container, and a delete is the verb it must act on to stop
    showing it. The container is readable here — it is on the doc being archived — so unlike a
    hard delete this one can be announced at the write.

    The descriptor states `modified_by` as the archiving user so the event carries the right
    actor. It describes the act, not the stored doc: the doc's own provenance is untouched."""
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
    _boundary.emit_artifact_change(_feed_doc(raw, modified_by=user_id), _boundary.DELETED)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# BRICK 4 — grants (Group 4): the CRUDEASIO authorization plane
#
# Everything is an artifact — a grant included. Where the store scoped grants by putting them in
# their own collection (COLLECTION_GRANTS), the lattice instead discriminates by a stamped
# `content_type = _GRANT_CT` on the doc (`Grant.from_dict` ignores unknown keys, so it never leaks
# into the entity), and the grant's lifecycle `state` (active / revoked / pending_accept) lives in
# `doc` beside it.
#
# The narrowing predicate is the grantee or the resource, never the content type: every grant
# shares one `ct`, so `ct` alone selects the whole authorization plane. `_grant_docs_by` is the
# read shape for a question about one principal or one resource, and `state` / expiry are then
# applied to that small set. `_grant_docs` — the whole plane — is for the two questions that
# genuinely range over all of it.
#
# Expiry: store compared `expires_at > DATE_ISO8601(DATE_NOW)` as strings. `_unexpired` parses
# both sides when it can (tolerating 'Z' vs '+00:00' — the string compare misorders those) and
# falls back to the string compare only when parsing fails.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from mantle.entities.grant import Grant as GrantEntity, mask_of as _mask_of
except ImportError:
    from mantle.entities.grant import Grant as GrantEntity, mask_of as _mask_of

_GRANT_CT = "application/vnd.agience.grant+json"


def _doc_mask(d: Dict[str, Any]):
    """A raw grant DOCUMENT's authority as the one attenuation `Mask` — bits AND effect.

    The lattice reads grants as dicts and never as entities on the hot paths, so the natural
    spelling here was `d.get("can_read")` — half the question. A `deny`-effect grant carries
    the bits naming what it denies, so the bare `.get` answers True for it and the grant reads
    as authorizing. That is, in the encoding the entity-side detectors cannot see.

    `SimpleNamespace` rather than handing the mapping straight to `mask_of`: `grant_is_allow`
    is `getattr`-duck-typed, and a dict would present as having no `effect` at all — closed,
    but wrong in a way that would deny every grant in the store. The namespace makes the doc
    look exactly like the entity to the predicates, so no part of the effect vocabulary or the
    flag set is restated here. `Grant.from_dict` is deliberately not used: its `can_read=True`
    default would WIDEN a doc that is missing the column, which is the wrong direction for a
    decoder feeding an authorization decision.
    """
    from types import SimpleNamespace
    return _mask_of(SimpleNamespace(**d))


def _to_grant_doc(entity: Any) -> Dict[str, Any]:
    d = to_lattice_doc(entity)
    d["content_type"] = _GRANT_CT          # the collection-scoping discriminator, doc-side only
    return d


def _grant_docs(db: LatticeDatabase, *, state: Optional[str] = None):
    """EVERY grant in the store, optionally narrowed to one lifecycle state.

    The whole-plane read. `ix_v_ct` narrows it to the grant bucket and no further — all grants
    share one content type — so its cost is the number of grants in the store, not the number
    the caller is actually asking about. That is the right shape for the two questions that
    genuinely range over the whole authorization plane (`access.gated_collections`,
    `access.gated_owner_map`) and the wrong shape for "what does this principal hold", which is
    `_grant_docs_by`."""
    return db.artifacts.list_artifacts(content_type=_GRANT_CT, state=state)


# A `LIMIT` that silently truncates an authorization answer is a security defect, not a slow
# path: the caller cannot tell a principal with no grant on a resource from a principal whose
# grant fell off the end of the page. So the seek takes a ceiling far above any real grant
# fan-out (a principal holds tens of grants; a resource, hundreds), and saturating it falls back
# to the exhaustive scan rather than returning a short list. The answer is exact at any size; the
# seek is what makes the ordinary size cheap.
_GRANT_SEEK_CEILING = 20_000


def _grant_docs_by(db: LatticeDatabase, field: str, value: Any) -> list:
    """Grant docs whose `doc.<field>` equals `value` — an index seek, not a scan.

    `field` is `grantee_id` or `resource_id`, each backed by a partial expression index
    (`schema.ix_v_grantee` / `ix_v_resource`). This is what makes an authorization cost
    O(this principal's grants) instead of O(every active grant on the platform), which matters
    because the light-cone resolver runs it on nearly every authenticated request.

    Returns docs in ANY lifecycle state, filtered by nothing but the field — `list_by_doc_field`
    applies no `state` predicate, so each caller states its own (`== "active"`, or
    `!= "archived"` for the admin view) and the answer stays identical to the whole-plane read
    it replaces. Expiry likewise stays with the caller: `expires_at` is a comparison against now,
    not an equality, so no index can serve it."""
    if not value:
        return []
    try:
        rows = db.artifacts.list_by_doc_field(
            content_type=_GRANT_CT, field=field, value=value, limit=_GRANT_SEEK_CEILING)
    except (AttributeError, ValueError):
        rows = None                       # a store without the seek — fall through to the scan
    if rows is not None and len(rows) < _GRANT_SEEK_CEILING:
        return rows
    return [d for d in db.artifacts.list_artifacts(content_type=_GRANT_CT,
                                                   include_archived=True)
            if d.get(field) == value]


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


#: NOT `Optional`, same shape as `update_artifact` above: one
#: `return entity`, no error path, and callers testing it wrote dead guards.
def update_grant(db: LatticeDatabase, entity: Any) -> Any:
    db.artifacts.put_artifact(_to_grant_doc(entity))
    return entity


def get_active_grants_for_principal_resource(db: LatticeDatabase, grantee_id: str,
                                             resource_id: str) -> list:
    """The direct grants one principal holds on one resource — the light-cone hot path.

    Seeks by `grantee_id` rather than by `resource_id` because a principal's grant count is
    bounded by what that person was given, while a resource's is bounded by how widely it was
    shared; the former is the smaller and steadier of the two."""
    return [from_lattice_doc(d, GrantEntity) for d in _grant_docs_by(db, "grantee_id", grantee_id)
            if d.get("state") == "active"
            and d.get("resource_id") == resource_id
            and _unexpired(d)]


def get_active_grants_for_grantee(db: LatticeDatabase, grantee_id: str,
                                  grantee_type: str = "user") -> list:
    return [from_lattice_doc(d, GrantEntity) for d in _grant_docs_by(db, "grantee_id", grantee_id)
            if d.get("state") == "active"
            and d.get("grantee_type") == grantee_type
            and _unexpired(d)]


def get_active_collection_ids_for_user(db: LatticeDatabase, user_id: str) -> list:
    """The user's read light-cone: every resource an active, unexpired user-grant AUTHORIZES a
    read on. This is what `db.access.reachable_collections` walks outward from, so it
    is the seed of the whole read decision on the embeddable surface.

    `_doc_mask(d).allows("read")` and not `d.get("can_read")`: the bare column read is True for
    a `deny`-effect grant too — a deny grant's bits name the actions it denies — so a grant
    written to say "this user must not read this" seeded the reader's light cone with it. Same
    defect as `access.invokable_resources`, one layer down, and it failed OPEN for reads.

    Deny is then subtracted, matching the deny-first precedence `services.dependencies.
    check_access` enforces, so holding both an allow and a deny on the same resource denies
    whichever order the store returns them in. The subtraction is DIRECT-only: containment
    expansion happens in the caller, so a deny on a child does not prune that child out of the
    cone its allowed parent projects. That is a known partial — narrower than before, not as
    narrow as `check_access`, which re-tests deny at every level of its walk — and it is the
    fail-closed direction at every step, never the open one.

    Order is preserved (a list, deduplicated by the store's own seek order) because callers
    treat it as a sequence.
    """
    docs = [d for d in _grant_docs_by(db, "grantee_id", user_id)
            if d.get("state") == "active"
            and d.get("grantee_type") == "user"
            and d.get("resource_id")
            and _unexpired(d)]
    masked = [(d["resource_id"], _doc_mask(d)) for d in docs]
    denied = {rid for rid, m in masked if m.is_deny and m.carries("read")}
    return [rid for rid, m in masked if m.allows("read") and rid not in denied]


def get_grants_for_collection(db: LatticeDatabase, collection_id: str) -> list:
    """ALL grants on a resource, any state (the admin/share management view).

    Head-only: `!= "archived"` is what the whole-plane read applied through
    `list_artifacts`, restated here because the seek carries no state predicate of its own."""
    return [from_lattice_doc(d, GrantEntity)
            for d in _grant_docs_by(db, "resource_id", collection_id)
            if state_of(d) != "archived"]


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
        #: NO `or user_grant` FALLBACK. `update_grant` has one
        #: `return entity` and `GrantEntity` defines neither `__bool__` nor `__len__`, so the
        #: right-hand side was never evaluated. It read as "use the stored copy if the write
        #: gave nothing back", which is a sentence about a failure mode this function does not
        #: have — and the two sides were the same object anyway.
        update_grant(db, user_grant)
        return user_grant, True
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
    """Grant-key auth: hash the presented token, match `grantee_type == "grant_key"`.

    This finds only the ROOT grant. A bundle root carries no `resource_id` of its own,
    so a caller that stops here sees an empty-handed key; expanding the members and
    applying the root's ceiling is `grant_key_service.resolve`, which is what the auth
    path actually uses. This stays as the storage-layer primitive underneath it.
    """
    from mantle.services.grant_key_service import hash_token
    return get_active_grants_for_grantee(
        db, grantee_id=hash_token(token), grantee_type="grant_key")


def get_active_grant_key_grants_for_collection(db: LatticeDatabase, collection_id: str) -> list:
    return [from_lattice_doc(d, GrantEntity)
            for d in _grant_docs_by(db, "resource_id", collection_id)
            if d.get("state") == "active"
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
# BRICK 6 — commits (Group 5)
#
# Same move as grants: each store side-collection becomes a stamped `content_type` plane in the
# one store. Commit provenance (`get_commit_by_id` / `get_commits_for_collection`) keeps store's
# RAW-dict return shape — the routers render those directly.
# ─────────────────────────────────────────────────────────────────────────────

_COMMIT_CT = "application/vnd.agience.commit+json"
_COMMIT_ITEM_CT = "application/vnd.agience.commit-item+json"


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
    """Commits whose item set touches this collection, newest first (raw dicts).

    Each dict carries its resolved `adds` / `removes`, derived from `CommitItem` — the only
    place the changed version-ids live. Without this, the commits API would publish `adds: []`
    and `removes: []` for every commit: a changeset history in which nothing ever changed, and
    silent because the route would default the missing attributes to `[]`.

    It costs nothing extra to resolve here: this function already scans every `CommitItem` doc to
    decide which commits touch the collection, so resolving from that same scan avoids the
    per-commit lookup a router-level fix would add.

    Scoped to `collection_id`: a commit spanning two collections reports only the ids that moved
    in this one, which is what a caller asking for this container's history means."""
    items_here = {}
    for d in _typed_docs(db, _COMMIT_ITEM_CT):
        if d.get("collection_id") == collection_id and d.get("id"):
            items_here[d["id"]] = d
    out = []
    for c in _typed_docs(db, _COMMIT_CT):
        mine = [items_here[i] for i in (c.get("item_ids") or []) if i in items_here]
        if not mine:
            continue
        row = {k: v for k, v in c.items() if k not in _LATTICE_INTERNAL}
        adds, removes = [], []
        for it in mine:
            target = removes if it.get("item_type") == "remove" else adds
            target.extend(it.get("artifact_version_ids") or [])
        row["adds"] = adds
        row["removes"] = removes
        out.append(row)
    out.sort(key=lambda c: c.get("timestamp") or "", reverse=True)
    return out


# ── WHERE-index materialization bookkeeping ──────────────────────────────────
# A membership set: "this artifact version has been sent for WHERE indexing". Namespaced ids —
# the marker for artifact X must not collide with X itself in the shared id space.
_MATERIALIZED_CT = "application/vnd.agience.materialized-marker+json"


def is_materialized(db: LatticeDatabase, artifact_id: str) -> bool:
    raw = db.artifacts.get_artifact("materialized:" + artifact_id)
    return raw is not None and raw.get("content_type") == _MATERIALIZED_CT


def mark_materialized(db: LatticeDatabase, artifact_id: str) -> None:
    """Idempotent (upsert) — called by the index JOB, after the work.

    Called after the work, not at enqueue time: a marker written at enqueue time would say work
    was queued while its reader treats it as work done. Called from
    `pipeline_unified._mark_indexed`.

    Best-effort by design: the marker is an optimisation, and failing to write one costs a
    re-index, not correctness. It is not best-effort silently: the reader
    (`services/workspace_service.py:1164`) treats a missing marker as "not yet indexed", which is
    the safe direction, and a present one as "skip" — so a silent write failure is survivable, but
    a silent write is not diagnosable, which is why a write failure is logged at debug with the
    artifact id and the exception type. The swallow itself is kept deliberately, because raising
    here would fail an ingest over a cache entry.
    """
    from datetime import datetime, timezone
    try:
        db.artifacts.put_artifact({
            "id": "materialized:" + artifact_id,
            "content_type": _MATERIALIZED_CT,
            "at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:  # noqa: BLE001 — see the docstring: survivable, but not silent
        _log.debug("mark_materialized(%s) did not write: %s: %s",
                   artifact_id, type(exc).__name__, exc)


# ── from-import compat: the handful of names routers/services import directly ─
_WORKSPACE_CT = "application/vnd.agience.workspace+json"

#: store collection-name compat — `query_documents` routes each to its typed plane.
COLLECTION_GRANTS = "grants"
COLLECTION_ARTIFACTS = "artifacts"
_COLLECTION_TO_CT = {COLLECTION_GRANTS: _GRANT_CT}

#: The typed side-planes that share the vertex table. An "artifacts" query must NOT see them —
#: in store they were separate collections, and that scoping is part of the contract.
_SIDE_PLANE_CTS = frozenset({
    _GRANT_CT, _COMMIT_CT, _COMMIT_ITEM_CT, _MATERIALIZED_CT,
    # Retired planes. Nothing writes these, but a lattice provisioned by an older version may
    # still hold their rows, and scoping is about what a query may SEE — dropping the name here
    # would surface those rows in `artifacts` results.
    "application/vnd.agience.api-key+json",
    "application/vnd.agience.server-credential+json",
    "application/vnd.agience.server-jwk+json",
})

# The stream router imports the decrypt hook under this name; ONE boundary either way.
_decrypt_artifact_content = _boundary.decrypt_artifact_content


def _is_missing_acting_principal(exc: BaseException) -> bool:
    """Is this decryption failure really "nobody is acting", rather than "the bytes will not open"?

    Walks `__cause__`/`__context__` because the boundary wraps the original. Matched on the
    exception TYPE, not on message text: a message is prose and changes, and a scan that silently
    skipped every document because a rename broke a substring match is exactly the failure this
    guard exists to prevent.
    """
    try:
        from mantle.services.acting_principal import NoActingPrincipal
    except Exception:
        return False
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, NoActingPrincipal):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _is_key_custody_denial(exc: BaseException) -> bool:
    """Is this a REFUSAL to issue key material, rather than bytes that will not open?

    `KeyCustodyDenied` is the base every refusal subclasses, chosen over listing `GrantDenied` by
    hand for the reason its own docstring gives: "any new refusal type is caught by the first
    clause automatically, without a hand-listed tuple that a new subclass could silently fall
    through into". Matched on TYPE and walked through `__cause__`/`__context__`, like
    `_is_missing_acting_principal`, because the boundary wraps the original.

    A denial is NOT damage. `propagate='[]'` on a `contains` edge is `attenuation`'s absorbing
    deny — a deliberate statement that no authority crosses that edge — and an artifact behind one
    is not broken, it is simply not this caller's to read.
    """
    try:
        from mantle.services.acting_principal import KeyCustodyDenied
    except Exception:
        return False
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, KeyCustodyDenied):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def iter_documents(db: LatticeDatabase, cls: Type[T], collection_name: str,
                   filters: dict, *, unreadable: str = "raise",
                   skipped_out: Optional[list] = None,
                   denied: str = "raise",
                   denied_out: Optional[list] = None):
    """Equality-filtered scan of one typed plane (store's generic AQL helper).
    `artifacts` = every doc NOT in a typed side-plane (store's collection scoping preserved).

    `unreadable` decides what a document that cannot be hydrated does to the scan:

      * ``"raise"`` (default) — it ends the scan. A read that returns a short list because one
        document was unreadable is worse than no read: the caller cannot tell a filtered result
        from a truncated one, so every ordinary caller keeps this.
      * ``"skip"`` — it is left out and the scan continues. This is for maintenance passes, whose
        job is to do what can be done to the rest, and only where the caller reports what it
        skipped. `search.init_search` is the case it exists for: without it, a single artifact whose
        content was written under a key the node does not hold raises out of this loop and blocks
        the rebuild of every other document.

    `skipped_out`, when given, receives one `(id, reason)` per skipped document. An explicit
    collector rather than an attribute on the returned list: a list cannot carry one, and a caller
    that must report what it could not read should have to ask for that, not discover it.

    Skipping is opt-in and per call rather than a property of the store, because "carry on without
    it" is a statement about what the CALLER is doing, and a store cannot know that.

    `denied` is the same shape for a DIFFERENT thing, and the two are kept apart on purpose:

      * ``unreadable`` — the bytes will not open. Damage.
      * ``denied`` — key custody was REFUSED. The caller may not read this document, which is a
        real and correct answer rather than a fault. `propagate='[]'` on a `contains` edge is
        `attenuation`'s absorbing deny; an artifact behind one is working exactly as configured.

    Folding a denial into ``unreadable`` would file a deliberate authorization decision as
    corruption, and would report a store that is behaving correctly as a damaged one.

      * ``"raise"`` (default) — a denial ends the scan. `KeyCustodyDenied`'s own docstring states
        the rule this keeps: **"not authorized" must not become "no results"**. Every ordinary
        caller keeps this, so no read can quietly narrow to what the caller happens to be allowed.
      * ``"omit"`` — the document is left out and the scan continues, for a maintenance pass whose
        job is to act on what it MAY read. Indexing a document you may not read is not a thing
        that should happen, so for those callers omission is the correct behaviour rather than a
        concession. `denied_out` receives one `(id, reason)` per omitted document.

    A pass that is denied EVERYTHING still raises, even under ``"omit"``: that is a
    misprovisioned run, not a store with a few deny edges in it, and returning an empty list
    would report "nothing to do" for "I was allowed to see nothing".
    """
    if unreadable not in ("raise", "skip"):
        raise ValueError("query_documents: unreadable must be 'raise' or 'skip', got %r"
                         % (unreadable,))
    if denied not in ("raise", "omit"):
        raise ValueError("query_documents: denied must be 'raise' or 'omit', got %r" % (denied,))
    if collection_name == COLLECTION_ARTIFACTS:
        docs = (d for d in db.artifacts.list_artifacts()
                if d.get("content_type") not in _SIDE_PLANE_CTS)
    else:
        ct = _COLLECTION_TO_CT.get(collection_name)
        if ct is None:
            raise ValueError("query_documents: unmapped collection %r on the lattice"
                             % (collection_name,))
        docs = _typed_docs(db, ct)
    skipped: list = []
    refused: list = []
    considered = 0
    for d in docs:
        if not all(d.get(k) == v for k, v in (filters or {}).items()):
            continue
        considered += 1
        if unreadable == "raise" and denied == "raise":
            yield from_lattice_doc(d, cls)
            continue
        _id = str(d.get("id") or d.get("_key") or "?")   # off the RAW doc: hydration is what failed
        try:
            hydrated = from_lattice_doc(d, cls)
        # A refusal needs its OWN clause: `GrantDenied` is `KeyCustodyDenied` -> `PermissionError`
        # and is NOT a `ContentDecryptionError`, so it reaches here unwrapped and the clause below
        # never sees it. Measured on 71/home — one `GrantDenied` escaping this loop ended a
        # 2,165,867-artifact rebuild. `NoActingPrincipal` subclasses the same base, so the
        # fails-for-every-document case is re-raised here exactly as it is below.
        except _KeyCustodyDenied as exc:
            if denied == "raise" or _is_missing_acting_principal(exc):
                raise
            refused.append((_id, "%s: %s" % (type(exc).__name__, exc)))
        except _boundary.ContentDecryptionError as exc:
            # The boundary also WRAPS a refusal in some paths, so the same question is asked again
            # on the chain. A deny is not damage: it is `attenuation`'s absorbing deny doing its
            # job, and filing it under `unreadable` would report a correctly-configured store as a
            # corrupted one.
            if _is_key_custody_denial(exc) and not _is_missing_acting_principal(exc):
                if denied == "raise":
                    raise
                refused.append((_id, "%s: %s" % (type(exc).__name__, exc)))
                continue
            # Narrow by design. `ContentDecryptionError` covers two different things, and only one
            # of them is a document worth skipping:
            #
            #   * the content does not decrypt with the keys this node holds — data, and the case
            #     this mode exists for;
            #   * the CALLER has no acting principal, so no key material could be issued at all
            #     (`acting_principal.NoActingPrincipal`, surfaced through the same boundary).
            #
            # The second is a misconfigured run, not a damaged store, and it fails for EVERY
            # document. Skipping it would rebuild an index from whatever happened to be readable
            # and report that as complete — measured here, 142 healthy artifacts read as unreadable
            # from a script that simply had no principal in scope. So that case is re-raised: a run
            # that cannot read anything must say so, not quietly produce a smaller index.
            if unreadable == "raise" or _is_missing_acting_principal(exc):
                raise
            skipped.append((_id, "%s: %s" % (type(exc).__name__, exc)))
        else:
            yield hydrated
    if skipped:
        if skipped_out is not None:
            skipped_out.extend(skipped)
        _log.warning("query_documents(%s): %d document(s) could not be hydrated and were skipped; "
                     "first is %s (%s)", collection_name, len(skipped), skipped[0][0], skipped[0][1])
    if refused:
        # Denied EVERYTHING is a misprovisioned run, not a store with a few deny edges in it.
        # Returning [] there would report "nothing to do" for "I was allowed to see nothing" —
        # the same failure the `NoActingPrincipal` guard above exists to prevent, one layer out.
        if considered and len(refused) == considered:
            raise PermissionError(
                "query_documents(%s): key custody was refused for ALL %d document(s), so this "
                "caller may read none of them. Returning an empty result would report 'nothing to "
                "do' for 'I was allowed to see nothing'. First: %s (%s)"
                % (collection_name, considered, refused[0][0], refused[0][1]))
        if denied_out is not None:
            denied_out.extend(refused)
        _log.warning("query_documents(%s): %d of %d document(s) refused key custody and were "
                     "omitted — these are deny decisions, not damage; first is %s (%s)",
                     collection_name, len(refused), considered, refused[0][0], refused[0][1])



def query_documents(db: LatticeDatabase, cls: Type[T], collection_name: str,
                    filters: dict, *, unreadable: str = "raise",
                    skipped_out: Optional[list] = None,
                    denied: str = "raise",
                    denied_out: Optional[list] = None) -> list:
    """:func:`iter_documents`, materialised. The list form every ordinary caller wants.

    A scan of the artifacts plane hydrates every document, which means DECRYPTING every
    document, so the list is far larger than the rows it came from. Measured on 71/home: the
    materialised form of 2,165,743 artifacts drove a reindex past 16 GB of working set and into
    the pagefile before it had written a single index entry. A caller walking the whole plane
    should iterate:func:`iter_documents` in bounded chunks and let each chunk fall out of scope;
    this wrapper is for the bounded reads — one collection, one filter — where the whole result
    is the point and it is small.
    """
    return list(iter_documents(db, cls, collection_name, filters,
                               unreadable=unreadable, skipped_out=skipped_out,
                               denied=denied, denied_out=denied_out))


def get_collections_by_owner_id(db: LatticeDatabase, owner_id: str) -> list:
    return get_containers_for_user(db, owner_id)


def get_artifacts_by_creator_id(db: LatticeDatabase, creator_id: str) -> list:
    """Non-archived artifacts by `created_by` (the person-card / owner-memory lookups)."""
    return [from_lattice_doc(d, ArtifactEntity)
            for d in db.artifacts.list_artifacts(created_by=creator_id)
            if d.get("content_type") not in _SIDE_PLANE_CTS]


def _collection_ids_for_root(db: LatticeDatabase, root_id: str,
                             is_workspace: Dict[str, bool]) -> list:
    """`get_collection_ids_for_root` with the workspace verdict carried in from outside.

    Deciding whether a container is a workspace costs one artifact read, and the same handful of
    containers recur across every root in a batch — so the verdict is memoized by the caller and
    the read happens once per container rather than once per (root, container) pair."""
    out: list = []
    try:
        for e in (db.graph.edges_of(root_id, direction="in") or []):
            cid = e.get("src")
            if not cid or cid in out:
                continue
            verdict = is_workspace.get(cid)
            if verdict is None:
                col = db.artifacts.get_artifact(cid)
                verdict = col is not None and col.get("content_type") == _WORKSPACE_CT
                is_workspace[cid] = verdict
            if verdict:
                continue
            out.append(cid)
    except EdgesTruncated:
        # "Every collection holding an edge to this root" is what the sharing checks and the
        # collection breadcrumbs read; a clipped list here quietly turns a shared artifact into an
        # unshared one.
        raise
    except Exception:
        return out
    return out


def get_collection_ids_for_root(db: LatticeDatabase, root_id: str) -> list:
    """Every collection (excluding workspaces) holding an edge to this root."""
    return _collection_ids_for_root(db, root_id, {})


def batch_get_collection_ids_for_roots(db: LatticeDatabase, root_ids: list) -> Dict[str, list]:
    """`{root_id -> collection ids}` for many roots, sharing one workspace-verdict cache.

    Still one edge read per root — containment is indexed per root and there is no cross-root
    edge query — but the container reads behind the workspace filter collapse to one per
    distinct container across the whole batch."""
    is_workspace: Dict[str, bool] = {}
    return {r: _collection_ids_for_root(db, r, is_workspace) for r in (root_ids or [])}


def list_committed_artifacts_by_context_content_type(db: LatticeDatabase, content_type: str, *,
                                                     created_by: Optional[str] = None) -> list:
    """Committed artifacts whose ``context.content_type`` matches (trust-config loads).
    `created_by` narrows to one principal — the trust boundary: a label alone never confers
    trust, only provenance does.

    The query first narrows by the indexed `content_type` column (`ix_v_ct`), then confirms the
    match against `context.content_type`, which stays authoritative: `content_type` narrows,
    context decides. The caller already writes the value it searches for into the artifact's own
    `content_type` (issuers, at `services/issuers.py`), so the
    narrowing is exact rather than a heuristic. A row whose `content_type` and
    `context.content_type` disagree is a writer bug, and is excluded rather than silently
    included."""
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


__all__ = [
    "LatticeDatabase", "open_database",
    "after_key", "mid_key",
    "to_lattice_doc", "from_lattice_doc",
    "create_artifact", "get_artifact", "update_artifact", "delete_artifact",
    "get_draft_artifact", "get_latest_committed_artifact", "get_current_in_collection",
    "get_current_in_any_collection",
    "get_current_in_collection_many", "get_current_in_any_collection_many",
    "list_version_history", "list_draft_artifacts",
    "count_children", "has_children",
    "CollectionEntity",
    "create_collection", "get_collection_by_id", "update_collection", "delete_collection",
    "get_last_order_key", "add_artifact_to_collection", "remove_artifact_from_collection",
    "origin_chain", "OriginChainUnterminated", "ORIGIN_WALK_CEILING",
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
    "get_collections_by_owner_and_type", "is_materialized", "mark_materialized",
]


def typed_method(artifacts, name):
    """The typed lattice-store method `name`, or None on a store that does not have it.

    Absence means "this store predates the typed rewrite" — it never means "the answer is empty".
    Every caller must therefore fall back to a path that can FAIL, never to a default."""
    fn = getattr(artifacts, name, None)
    return fn if callable(fn) else None
