import json
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Set, TYPE_CHECKING, Any, Dict, Mapping, Sequence

import mantle.events.event_bus as event_bus

try:
    from fastapi import HTTPException, status  # type: ignore
except Exception:
    # Minimal fallbacks for environments without FastAPI (static analysis, tests, etc.)
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str = ""):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _Status:
        HTTP_400_BAD_REQUEST = 400
        HTTP_404_NOT_FOUND = 404
        HTTP_500_INTERNAL_SERVER_ERROR = 500

    status = _Status()
if TYPE_CHECKING:
    from mantle.db.store import Database
else:
    Database = Any


from mantle.entities.artifact import Artifact as ArtifactEntity
from mantle.entities.collection import Collection as CollectionEntity
from mantle.entities.grant import Grant as GrantEntity

from mantle.db.backend import (
    # Collections
    create_collection as db_create_collection,
    get_collection_by_id as db_get_collection_by_id,
    get_collections_by_owner_id as db_get_collections_by_owner_id,
    update_collection as db_update_collection,
    delete_collection as db_delete_collection,
    get_collection_ids_for_root as db_get_collection_ids_for_root,
    batch_get_collection_ids_for_roots as db_batch_get_collection_ids_for_roots,

    # Artifacts (unified store)
    get_artifact as db_get_artifact,
    get_artifacts_by_creator_id as db_get_artifacts_by_creator_id,
    get_latest_live_artifact as db_get_latest_live_artifact,
    get_current_in_collection as db_get_current_in_collection,
    get_current_in_collection_many as db_get_current_in_collection_many,
    get_current_in_any_collection_many as db_get_current_in_any_collection_many,
    list_collection_artifacts as db_list_collection_artifacts,
    archive_artifact as db_archive_artifact,
    remove_artifact_from_collection as _db_remove_edge,


    # Grants - unified authorization records
    get_active_collection_ids_for_user as db_get_active_collection_ids_for_user,
)


# Local thin adapters that re-shape unified-store calls to the names this
# module's body uses.

def db_get_artifact_by_version_id(db, version_id):
    """One artifact version by its version id, or ``None``.

    An archived version reads as absent here, so callers that treat ``None`` as "not found"
    do not have to re-check ``state`` themselves.
    """
    a = db_get_artifact(db, version_id)
    if a and a.state == "archived":
        return None
    return a


def db_get_latest_artifact_version_by_root_id(db, root_id):
    """The newest non-archived version of a lineage in any collection, or ``None`` if the root
    has none."""
    return db_get_latest_live_artifact(db, root_id)


def db_get_artifact_by_collection_id_and_root_id(db, collection_id, root_id):
    """The current version of ``root_id`` as seen from ``collection_id`` — the newest
    non-archived version there. ``None`` when the root has no version in this collection."""
    return db_get_current_in_collection(db, collection_id, root_id)


def db_get_artifacts_by_collection_id(db, collection_id):
    """The collection's current members as ``Artifact`` entities, one per membership edge,
    ordered by ``order_key``. Archived members are omitted."""
    from mantle.entities.artifact import Artifact
    rows = db_list_collection_artifacts(db, collection_id)
    return [Artifact.from_dict(r) for r in rows]


def db_delete_collection_artifact_by_collection_and_root(db, collection_id, root_id):
    """Drop the membership edge for (collection, root). Idempotent: ``True`` means the edge is
    gone, whether or not it was there to begin with."""
    return _db_remove_edge(db, collection_id, root_id)

from mantle.search.ingest.pipeline_unified import (  # noqa: E402
    index_artifact,
    delete_artifact_from_index,
)
from mantle.services.content_service import generate_signed_url  # noqa: E402

logger = logging.getLogger(__name__)


def _emit(collection_id: str, name: str, payload: dict, *, actor_id: Optional[str] = None) -> None:
    # artifact.created / artifact.updated are emitted at the db chokepoint
    # (db.store) so every write is covered exactly once; only forward the rest.
    if name in ("artifact.created", "artifact.updated"):
        return
    try:
        event_bus.emit_artifact_event_sync(collection_id, name, payload, actor_id=actor_id)
    except Exception:
        logger.debug("event bus emit failed", exc_info=True)


def ensure_collection_descriptor(db: Database, collection: CollectionEntity) -> CollectionEntity:
    """Ensure container artifacts are searchable.

    Container-as-artifact means we index the container artifact itself
    (workspace/collection), not a separate descriptor artifact.
    """
    try:
        index_artifact(collection, collection.id, is_head=True)
    except Exception as e:
        logger.warning("Failed to index container artifact %s: %s", collection.id, e)
    return collection

def _attach_committed_collection_ids(db: Database, artifacts: Sequence[ArtifactEntity]) -> None:
    """Populate committed_collection_ids on each artifact entity."""
    if not artifacts:
        return

    root_ids: List[str] = []
    for artifact in artifacts:
        root_value = getattr(artifact, "root_id", None)
        if not root_value:
            continue
        root_ids.append(str(root_value))
    if not root_ids:
        for artifact in artifacts:
            setattr(artifact, "committed_collection_ids", [])
        return

    # Deduplicate roots to keep queries lean
    unique_root_ids = list(dict.fromkeys(root_ids))

    memberships: Dict[str, List[str]] = {}
    try:
        memberships = db_batch_get_collection_ids_for_roots(db, unique_root_ids)
    except Exception:
        logger.exception("Failed batch fetching memberships; falling back to per-root queries")
        memberships = {}

    # Fill any missing roots via individual lookups
    for root_id in unique_root_ids:
        if root_id in memberships:
            continue
        try:
            memberships[root_id] = db_get_collection_ids_for_root(db, root_id) or []
        except Exception:
            memberships[root_id] = []

    # Attach sorted unique memberships per artifact
    for artifact in artifacts:
        root_id = getattr(artifact, "root_id", None)
        committed = memberships.get(root_id or "", []) if root_id else []
        setattr(artifact, "committed_collection_ids", sorted(dict.fromkeys(committed)))


def attach_committed_collection_ids(db: Database, artifacts: Sequence[ArtifactEntity]) -> None:
    """Public API: populate committed_collection_ids on a list of artifacts."""
    _attach_committed_collection_ids(db, artifacts)


def attach_memberships(db: Database, docs: list) -> None:
    """Give each artifact document `committed_collection_ids` (its collections, workspaces left
    out, as the sharing checks read them) and `container_ids` (every container holding it,
    workspaces included: a workspace is a collection is an artifact)."""
    from mantle.db.backend import batch_get_container_ids_for_roots
    from mantle.entities.artifact import Artifact as ArtifactEntity
    entities = [ArtifactEntity.from_dict(d) for d in docs]
    _attach_committed_collection_ids(db, entities)
    roots = [str(d.get("root_id") or d.get("id") or "") for d in docs]
    containers = batch_get_container_ids_for_roots(db, list(dict.fromkeys(r for r in roots if r)))
    for doc, entity, root in zip(docs, entities, roots):
        doc["committed_collection_ids"] = getattr(entity, "committed_collection_ids", [])
        doc["container_ids"] = sorted(containers.get(root, []))


# === Collection CRUD ===

def create_new_collection(
    db: Database,
    owner_id: str,
    name: str,
    description: str,
    is_personal: bool = False,
    container_id: Optional[str] = None,
) -> CollectionEntity:
    """Create a collection. When ``container_id`` is given, the collection is
    homed in that container and linked with an origin edge — exactly how any
    artifact created in a workspace becomes a member of it. A collection is an
    artifact, so a collection created in a workspace shows up in that workspace
    like anything else. Without ``container_id`` it is created top-level."""
    from mantle.entities.collection import COLLECTION_CONTENT_TYPE
    now = datetime.now(timezone.utc).isoformat()
    entity = CollectionEntity(
        id=owner_id if is_personal else str(uuid.uuid4()),
        name=name,
        description=description,
        created_by=owner_id,
        content_type=COLLECTION_CONTENT_TYPE,
        state=CollectionEntity.STATE_LIVE,
        collection_id=container_id or "",
        created_time=now,
        modified_time=now,
    )
    created = db_create_collection(db, entity)
    ensure_collection_descriptor(db, created)

    # Issue explicit full-CRUDEASIO grant to the creator.
    from mantle.db.backend import upsert_user_collection_grant as db_upsert_creator_grant
    db_upsert_creator_grant(
        db,
        user_id=owner_id,
        collection_id=created.id,
        granted_by=owner_id,
        can_create=True,
        can_read=True,
        can_update=True,
        can_delete=True,
        can_invoke=True,
        can_add=True,
        can_share=True,
        can_admin=True,
    )

    # Make it a member of its container (origin edge) so it appears there, just
    # like any other artifact created in that workspace/collection.
    if container_id:
        from mantle.db.backend import add_artifact_to_collection as db_add_to_container
        db_add_to_container(db, container_id, created.root_id or created.id, origin=True)

    return created


def get_collections_for_user(
    db: Database,
    user_id: Optional[str],
    grant_key: Optional[GrantEntity] = None,
) -> List[CollectionEntity]:
    """
    Get collections accessible to user/grant key.
    Includes: owned collections + grant-accessible collections + grant_key collection.
    """
    from mantle.db.backend import get_containers_for_user as db_get_containers_for_user
    owned = db_get_containers_for_user(db, user_id) if user_id else []
    seen_ids: Set[str] = {c.id for c in owned}

    # Grant-accessible collections (e.g. Agience platform collections shared read-only)
    if user_id:
        try:
            grant_col_ids = db_get_active_collection_ids_for_user(db, user_id)
            for col_id in grant_col_ids:
                if col_id not in seen_ids:
                    col = db_get_collection_by_id(db, col_id)
                    if col:
                        owned.append(col)
                        seen_ids.add(col_id)
        except Exception:
            logger.exception("get_collections_for_user: failed resolving user grants")

    # Grant key grant
    if grant_key:
        if (
            grant_key.grantee_type == GrantEntity.GRANTEE_GRANT_KEY
            and grant_key.can_read
        ):
            read_requires_identity = (
                grant_key.read_requires_identity
                if grant_key.read_requires_identity is not None
                else grant_key.requires_identity
            )
            if not (read_requires_identity and not user_id):
                shared = db_get_collection_by_id(db, grant_key.resource_id)
                if shared and shared.id not in seen_ids:
                    owned.append(shared)
    
    return owned


def get_collection_for_user(
    db: Database,
    user_id: Optional[str],
    collection_id: str,
) -> CollectionEntity:
    """Load a collection by id. Raises ``HTTPException(404)`` if it does not exist.

    ``user_id`` is accepted for call-site symmetry with the rest of this module but is not
    consulted: this performs no access check, so the caller must authorize the read itself.
    """
    collection = db_get_collection_by_id(db, collection_id)
    if not collection:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return collection


def update_user_collection(
    db: Database,
    user_id: Optional[str],
    collection_id: str,
    name: Optional[str],
    description: Optional[str],
) -> CollectionEntity:
    """Rename and/or re-describe a collection, returning the updated entity.

    ``None`` and empty values leave the corresponding field unchanged, so this cannot be used to
    blank a name or description. Requires an active grant carrying ``can_update`` on the
    collection; a caller without one — including an anonymous ``user_id=None`` — gets
    ``HTTPException(404)``, the same answer as a collection that does not exist, so refusal does
    not disclose existence. Raises ``HTTPException(500)`` if the store rejects the write.
    """
    collection = db_get_collection_by_id(db, collection_id)
    if not collection:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")

    from mantle.db.backend import get_active_grants_for_principal_resource
    grants = get_active_grants_for_principal_resource(
        db, grantee_id=user_id, resource_id=collection_id,
    ) if user_id else []
    if not any(getattr(g, "can_update", False) for g in grants):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")

    if name:
        collection.name = name
    if description:
        collection.description = description
    collection.modified_time = datetime.now(timezone.utc).isoformat()
    updated = db_update_collection(db, collection)
    if not updated:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to update collection")
    return updated


def delete_user_collection(db: Database, user_id: Optional[str], collection_id: str) -> None:
    """
    Delete a collection and clean up references:
    - Remove links from all artifacts to this collection
    - If an artifact (root) has no other collection references after unlink, delete all its versions
    - Remove corresponding search index documents
    """
    collection = db_get_collection_by_id(db, collection_id)
    if not collection:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")

    from mantle.db.backend import get_active_grants_for_principal_resource
    grants = get_active_grants_for_principal_resource(
        db, grantee_id=user_id, resource_id=collection_id,
    ) if user_id else []
    if not any(getattr(g, "can_delete", False) for g in grants):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")

    # 1) Fetch all artifacts currently linked to this collection
    linked_cards = db_get_artifacts_by_collection_id(db, collection_id)

    # 2) For each linked root: unlink and optionally delete root (if last reference)
    for artifact in linked_cards:
        root_id = getattr(artifact, "root_id", None)
        version_id = getattr(artifact, "id", None)
        if not root_id:
            continue

        # Unlink from this collection
        try:
            db_delete_collection_artifact_by_collection_and_root(db, collection_id, root_id)
        except Exception:
            logger.exception("Failed unlinking root %s from collection %s", root_id, collection_id)

        # Remove unified index doc for this (collection, root, version)
        if version_id:
            try:
                delete_artifact_from_index(version_id, root_id, collection_id=collection_id)
            except Exception:
                logger.exception("Failed to delete index for version %s (root %s)", version_id, root_id)

        # Check if this root remains in any collection; if not, delete all its versions
        try:
            memberships = db_get_collection_ids_for_root(db, root_id)  # type: ignore[attr-defined]
        except Exception:
            memberships = []
        if not memberships:
            # Delete all versions for this root and remove their index docs. A root leaving its
            # last collection must not leak every version and its index docs, so
            # `delete_artifacts_by_root` (db/lattice_api.py) does the hard delete this docstring
            # promises; failure is logged, never swallowed into a fake empty result.
            try:
                from mantle.db.backend import delete_artifacts_by_root as db_delete_artifacts_by_root
                deleted_versions = db_delete_artifacts_by_root(db, root_id)
            except Exception:
                logger.exception("Failed deleting versions for root %s (last membership gone)",
                                 root_id)
                deleted_versions = []
            for vid in deleted_versions:
                try:
                    delete_artifact_from_index(vid, root_id, collection_id=collection_id)
                except Exception:
                    logger.exception("Failed to delete index for version %s (root %s)", vid, root_id)

    # 3) Delete the collection itself
    if not db_delete_collection(db, collection_id):
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Deletion failed")


# === Artifact Operations ===

def get_unattached_artifacts(db: Database, user_id: str) -> List[ArtifactEntity]:
    """
    Return artifacts created by `user_id` that have no CollectionArtifact link.
    We check link existence per root. If a version lacks a visible root_id field,
    we conservatively skip it.
    """
    all_cards = db_get_artifacts_by_creator_id(db, user_id)

    # group one representative version per root
    roots_seen: Set[str] = set()
    reps: List[ArtifactEntity] = []
    for c in all_cards:
        rid = getattr(c, "root_id", None)
        if not rid:
            continue
        if rid not in roots_seen:
            roots_seen.add(rid)
            reps.append(c)

    unattached: List[ArtifactEntity] = []
    for c in reps:
        rid = getattr(c, "root_id", None)
        if not rid:
            continue
        # Use the store helper to get collection ids linked to this root.
        linked_ids = db_get_collection_ids_for_root(db, rid) or []
        if not linked_ids:
            unattached.append(c)

    _attach_committed_collection_ids(db, unattached)
    return unattached


def get_artifacts_for_user(db: Database, user_id: str) -> List[ArtifactEntity]:
    """Every non-archived artifact version ``user_id`` created, with
    ``committed_collection_ids`` hydrated.

    Authorship rather than access: the listing is keyed on ``created_by``, so it includes versions
    in collections the user cannot read and excludes ones they can read but did not write.
    """
    artifacts = db_get_artifacts_by_creator_id(db, user_id)
    _attach_committed_collection_ids(db, artifacts)
    return artifacts


def get_collection_artifact(
    db: Database,
    user_id: Optional[str],
    collection_id: str,
    artifact_root_id: str,
) -> ArtifactEntity:
    """The current version of ``artifact_root_id`` in ``collection_id``, with
    ``committed_collection_ids`` hydrated.

    Addressed by ROOT id, not version id. Raises ``HTTPException(404)`` when the root has no
    version in this collection. ``user_id`` is not consulted — the caller authorizes the read.
    """
    artifact = db_get_artifact_by_collection_id_and_root_id(db, collection_id, artifact_root_id)
    if not artifact:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    _attach_committed_collection_ids(db, [artifact])
    return artifact


def get_collection_artifact_content_url(
    db: Database,
    user_id: Optional[str],
    collection_id: str,
    artifact_root_id: str,
    *,
    expires_in: int,
) -> Dict[str, Any]:
    """Mint a download URL for a collection artifact's stored content.

    Returns ``{"url", "expires_in", "filename"}``. An artifact whose context declares
    ``access: "public"`` and carries a ``uri`` returns that uri verbatim with
    ``expires_in=None``; everything else returns a signed URL over the artifact's own object
    key, valid for ``expires_in`` seconds.

    The key is derived from the artifact id rather than taken from context — a context-supplied
    key naming a different artifact is refused, since authoring an artifact would otherwise mint
    a download URL for another tenant's object.

    Raises ``HTTPException(404)`` if the collection or the artifact is absent, and
    ``HTTPException(400)`` if the content is not agience-hosted (nothing to sign).
    """
    collection = db_get_collection_by_id(db, collection_id)
    if not collection:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    artifact = db_get_artifact_by_collection_id_and_root_id(db, collection_id, artifact_root_id)
    if not artifact:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")

    raw_context = getattr(artifact, "context", None)
    if isinstance(raw_context, str):
        try:
            ctx = json.loads(raw_context or "{}")
        except Exception:
            ctx = {}
    elif isinstance(raw_context, Mapping):
        ctx = dict(raw_context)
    else:
        ctx = {}

    access = ctx.get("access", "private")
    if access == "public" and "uri" in ctx:
        return {
            "url": ctx["uri"],
            "expires_in": None,
            "filename": ctx.get("filename", "download"),
        }

    if ctx.get("content_source") != "agience-content":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Not an agience-hosted file")

    # Validated, not trusted — see `workspace_service._safe_content_key`. Unchecked, a caller who
    # can author an artifact's `context` mints a signed download URL for another tenant's object.
    from mantle.services.workspace_service import _safe_content_key
    key = _safe_content_key(ctx, artifact_root_id, default_prefix=str(collection.created_by))
    filename = ctx.get("filename", "download")
    content_type = ctx.get("content_type", "application/octet-stream")
    url = generate_signed_url(
        key=key,
        filename=filename,
        content_type=content_type,
        expires_in=expires_in,
    )
    return {
        "url": url,
        "expires_in": expires_in,
        "filename": filename,
    }


def get_collection_artifacts_by_root_ids(
    db: Database,
    user_id: Optional[str],
    collection_id: str,
    artifact_root_ids: List[str],
) -> List[ArtifactEntity]:
    """The current version of each named root in ``collection_id``, archived ones omitted, with
    ``committed_collection_ids`` hydrated. Roots absent from the collection are skipped, so the
    result may be shorter than the input, and results follow the order the roots were named in.

    One chunked lineage read for the whole set rather than one per root: the store publishes a
    multi-root lineage read (``ArtifactStore.versions_of_many``, reached through
    :func:`lattice_api.get_current_in_collection_many`), so a per-root round trip is not the floor.
    Repeats in ``artifact_root_ids`` are read once and yielded once. The membership
    hydration afterwards is collapsed across the set the same way — see
    :func:`_attach_committed_collection_ids`.

    ``user_id`` is not consulted; the caller authorizes the read.
    """
    roots = list(dict.fromkeys(artifact_root_ids or []))
    current = db_get_current_in_collection_many(db, collection_id, roots)
    artifacts = [a for a in (current.get(root_id) for root_id in roots)
                 if a is not None and not getattr(a, "is_archived", False)]
    _attach_committed_collection_ids(db, artifacts)
    return artifacts


def get_collection_artifacts_batch_global(
    db: Database,
    user_id: Optional[str],
    artifact_root_ids: List[str],
) -> List[ArtifactEntity]:
    """Batch fetch multiple artifacts across all accessible collections by root IDs.

    A genuine batch on both axes: one chunked store read covers every distinct root
    (``ArtifactStore.versions_of_many``), and which collection a root is visible through is
    decided in memory against the caller's accessible set. The lineage is the same lineage
    whichever collection is being asked about, so re-reading it per candidate collection buys
    nothing, and re-reading it per root made the cost O(roots x collections) round trips for an
    answer that depends on one read.

    Candidate collections are ordered owned-first, then grant-reachable, so a root visible
    through more than one resolves the same way on every call. A `set` alone leaves that tie to
    hash order, which is per-process. Results follow the order the roots were named in."""
    owned_collections = db_get_collections_by_owner_id(db, user_id) if user_id else []
    collection_ids: List[str] = list(dict.fromkeys(c.id for c in owned_collections))
    if user_id:
        try:
            granted = db_get_active_collection_ids_for_user(db, user_id)
        except Exception:
            logger.exception("get_collection_artifacts_batch_global: failed resolving user grants")
            granted = []
        seen_collections = set(collection_ids)
        for cid in granted:
            if cid not in seen_collections:
                seen_collections.add(cid)
                collection_ids.append(cid)

    roots = list(dict.fromkeys(artifact_root_ids or []))
    current = db_get_current_in_any_collection_many(db, roots, collection_ids)
    artifacts = [a for a in (current.get(root_id) for root_id in roots)
                 if a is not None and not getattr(a, "is_archived", False)]

    _attach_committed_collection_ids(db, artifacts)
    return artifacts


def get_collection_artifacts(
    db: Database,
    user_id: Optional[str],
    collection_id: str,
) -> List[ArtifactEntity]:
    """Every current, non-archived member of a collection, ordered by ``order_key``, with
    ``committed_collection_ids`` hydrated.

    Returns ``[]`` for a collection that is empty and for one that does not exist — the two are
    indistinguishable here. ``user_id`` is not consulted; the caller authorizes the read.
    """
    all_artifacts = db_get_artifacts_by_collection_id(db, collection_id)
    active_artifacts = [artifact for artifact in all_artifacts if not getattr(artifact, "is_archived", False)]
    _attach_committed_collection_ids(db, active_artifacts)
    return active_artifacts


def _note_membership_change(user_id: str, collection_id: str) -> None:
    """Tell the proximity digest that this collection changed, if a host installed one.

    A collection's digest is a read over its members, so a membership change is what makes the
    stored one out of date. The refresher counts the change and retakes the digest once the
    collection has turned over; both halves live in `search.ingest.digest_refresh`.

    Imported here rather than at module scope so a store with no proximity instrument never loads
    it, and wrapped so a digest fault cannot fail the membership change that prompted it.
    """
    try:
        from mantle.search.ingest.digest_refresh import note_membership_change
    except Exception:  # noqa: BLE001 - the semantic extra is not installed
        return
    note_membership_change(user_id, collection_id)


def archive_artifact_by_version_id(db: Database, user_id: str, version_id: str) -> bool:
    """
    Archives an artifact by its version ID.
    """
    artifact = db_get_artifact_by_version_id(db, version_id)
    if not artifact:
        logger.warning("Artifact %s not found for archiving", version_id)
        return False
    return db_archive_artifact(db, user_id, version_id)


# === Commit ===

# === Provenance helper ===

# ---------------------------------------------------------------------------
# Native dispatch handler — called by operation_dispatcher for type.json
# ``dispatch: { kind: "native", target: "collection_service.<fn>" }``
# ---------------------------------------------------------------------------

async def dispatch_create_collection(artifact: dict, body: dict, ctx: Any) -> dict:
    """Create a collection via the ``create`` operation on collection type.

    When the request carries a ``container_id`` (the active workspace/collection),
    the new collection is homed there and shows up like any other artifact — but
    only after verifying the caller may create in that container, so a crafted
    request can't drop a collection into someone else's workspace.
    """
    name = (body or {}).get("name", "New Collection")
    description = (body or {}).get("description", "")
    container_id = (body or {}).get("container_id") or None

    if container_id and ctx.user_id:
        try:
            from mantle.services.dependencies import AuthContext, check_access
            check_access(
                AuthContext(principal_id=ctx.user_id, principal_type="user", user_id=ctx.user_id),
                container_id, "create", ctx.store_db,
            )
        except Exception:
            container_id = None  # not allowed → create top-level instead

    coll = create_new_collection(
        ctx.store_db, ctx.user_id, name, description, container_id=container_id
    )
    return coll.to_dict()

