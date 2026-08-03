# entities/artifact.py
# Unified Artifact entity — one table, one entity, three states.
# See .dev/features/universal-artifact-model.md for design rationale
# (it supersedes the archived unified-artifact-store + container-as-artifact docs).

from typing import Optional, Dict, Any
from .base import BaseEntity

# Re-export container content-type constants so importers can get them here.
WORKSPACE_CONTENT_TYPE = "application/vnd.agience.workspace+json"
COLLECTION_CONTENT_TYPE = "application/vnd.agience.collection+json"


class Artifact(BaseEntity):
    """
    Unified artifact entity. Every artifact lives in exactly one collection
    and carries a state in {draft, committed, archived}.

    Container artifacts (workspaces, collections) are distinguished by
    `content_type`. A container IS an artifact — same table, same entity.

    Ordering is NOT stored on the artifact — it lives on the
    `collection_artifacts` edge (see db/lattice.py).
    """

    PREFIX = "Artifact"
    STATE_DRAFT = "draft"
    STATE_COMMITTED = "committed"
    STATE_ARCHIVED = "archived"
    VALID_STATES = {STATE_DRAFT, STATE_COMMITTED, STATE_ARCHIVED}

    def __init__(
        self,
        id: Optional[str] = None,
        root_id: Optional[str] = None,
        collection_id: str = "",
        context: str = "",
        content: str = "",
        state: str = STATE_DRAFT,
        created_by: Optional[str] = None,
        created_time: Optional[str] = None,
        modified_by: Optional[str] = None,
        modified_time: Optional[str] = None,
        # Container fields (optional — only set on container artifacts)
        name: Optional[str] = None,
        description: Optional[str] = None,
        content_type: Optional[str] = None,
        # True while `content` still holds ciphertext — i.e. the read path could
        # not decrypt it. Modeled here so the flag SURVIVES the storage round
        # trip: without it, a failed decrypt handed ciphertext to the entity,
        # the flag was dropped, and saving re-encrypted the ciphertext and
        # destroyed the original irrecoverably.
        content_encrypted: bool = False,
        # The collection's IMMUTABLE ORIGIN ROOT — the key root for this artifact's
        # content, and the same principal MANTLE cells are keyed under.
        #
        # Modeled here for the same reason as `content_encrypted` above: it must
        # SURVIVE the storage round trip. A key root that a `from_dict` silently
        # dropped would be recomputed — or defaulted — on the next save, re-keying
        # content whose ciphertext was written under the old value.
        #
        # It is INHERITED, never walked: a root artifact is its own origin root, a
        # child takes its parent's. That keeps it O(1) at creation and means no read
        # path ever needs a lineage traversal — the field travels with the row, like
        # the lattice store's `(_origin, _seq)`.
        origin_root: Optional[str] = None,
    ):
        super().__init__(id=id, created_time=created_time, modified_time=modified_time)

        if state not in self.VALID_STATES:
            raise ValueError(f"Invalid state '{state}'")

        # First version of an artifact: id == root_id. The root doc persists forever
        # and is the stable target of `collection_artifacts` edges.
        self.root_id = root_id or self.id
        self.collection_id = collection_id
        # A top-level artifact (no parent collection) IS its own origin root. A child
        # is stamped from its parent at create time (`db.lattice.create_artifact`);
        # leaving it None here rather than guessing `self.id` means an unstamped
        # child is VISIBLE as unstamped instead of silently claiming to be a root —
        # which would give it its own key tree and orphan it from its collection.
        self.origin_root = origin_root or (self.id if not collection_id else None)
        self.context = context
        self.content = content
        self.content_encrypted = content_encrypted
        self.state = state
        self.created_by = created_by
        self.modified_by = modified_by
        self.name = name
        self.description = description
        self.content_type = content_type

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "root_id": self.root_id,
            "collection_id": self.collection_id,
            "context": self.context,
            "content": self.content,
            "state": self.state,
            "created_by": self.created_by,
            "created_time": self.created_time,
            "modified_by": self.modified_by,
            "modified_time": self.modified_time,
            "origin_root": self.origin_root,
        }
        if self.name is not None:
            d["name"] = self.name
        if self.description is not None:
            d["description"] = self.description
        if self.content_type is not None:
            d["content_type"] = self.content_type
        # Emitted only when True. The trailing filter drops None but keeps False,
        # and a stray `content_encrypted: false` on every artifact would be noise;
        # what matters is that True is never lost.
        if self.content_encrypted:
            d["content_encrypted"] = True
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        return cls(
            id=data.get("id") or data.get("_key"),
            root_id=data.get("root_id"),
            collection_id=data.get("collection_id", ""),
            context=data.get("context", ""),
            content=data.get("content", ""),
            state=data.get("state", cls.STATE_DRAFT),
            created_by=data.get("created_by"),
            created_time=data.get("created_time"),
            modified_by=data.get("modified_by"),
            modified_time=data.get("modified_time"),
            origin_root=data.get("origin_root"),
            name=data.get("name"),
            description=data.get("description"),
            content_type=data.get("content_type"),
            content_encrypted=bool(data.get("content_encrypted", False)),
        )
