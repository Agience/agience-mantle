# entities/artifact.py
# Unified Artifact entity — one table, one entity, three states.
# See .dev/features/universal-artifact-model.md for design rationale.

from typing import Optional, Dict, Any

from mantle.db.constants import STATE_WHEN_ABSENT as _STATE_WHEN_ABSENT, state_of

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
    `collection_artifacts` edge (see db.py).
    """

    PREFIX = "Artifact"
    STATE_DRAFT = "draft"
    STATE_COMMITTED = "committed"
    STATE_ARCHIVED = "archived"
    VALID_STATES = {STATE_DRAFT, STATE_COMMITTED, STATE_ARCHIVED}

    #: What a STORED doc that carries no `state` is in. Re-exported from the store layer, which
    #: is where the one definition lives (`db/constants.STATE_WHEN_ABSENT`) — the search index is
    #: partitioned by state into separately-keyed encrypted trees, so a second answer here would
    #: relocate an artifact between trees on a read-modify-write that changed nothing.
    #:
    #: Distinct from the `state=` default on `__init__` below, which is `draft` and stays `draft`:
    #: constructing a NEW artifact asserts that an unpublished edit exists, which is exactly the
    #: claim absence cannot make.
    STATE_WHEN_ABSENT = _STATE_WHEN_ABSENT

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
        # trip: without it, a failed decrypt would hand ciphertext to the entity,
        # the flag would be dropped, and saving would re-encrypt the ciphertext
        # and destroy the original irrecoverably.
        content_encrypted: bool = False,
        # Where the bytes live: `cas/<sha256 of the plaintext>`, or None on a row written before
        # the content tier was wired in.
        #
        # Content does not live in the lattice. The vertex has carried a `content_ref` column
        # since `db/schema.py`, `db/vertex.py` reads `doc["content_ref"]` to decide whether a
        # write is a re-describe or a new version, and `db/test_lattice.py` has always covered
        # both — but nothing in the service layer ever SET it, so every artifact's bytes went
        # into the document instead. Measured on 71/dev before this field existed: 0 of 709
        # vertices carried a ref, and 7.2 MB of 7.9 MB of doc bytes — 92% of the lattice — was
        # artifact content that belonged in the CAS.
        #
        # Modeled on the entity for the same reason as `content_encrypted` and `origin_root`:
        # it must SURVIVE the storage round trip. A ref that `from_dict` dropped would be
        # recomputed as None on the next save, and the row would silently go back to carrying
        # its own bytes.
        content_ref: Optional[str] = None,
        # The collection's immutable origin root — the key root for this artifact's
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
        # The words a LEXICON entry goes by — `oxygen`, `O`, `atomic number 8` are one concept.
        #
        # Modeled here for the same reason as `content_ref` and `origin_root` above: it must
        # SURVIVE the storage round trip. It did not, and the cost was measurable. Only the first
        # lemma becomes the title, and the two lexicons order them differently, so the OEWN oxygen
        # synset is titled `O` while its gloss ("a nonmetallic bivalent element ...") never says
        # the word. The names that would have found it were dropped by `from_dict`, so
        # `_extract_artifact_fields` never saw them, so `recall("what is oxygen")` did not narrow
        # to that artifact at all and answered with its hyponyms — every one of which carries
        # `oxygen` inside its own title.
        #
        # `None` rather than `[]`: absent and empty are different claims, and only a lexicon
        # entry has names in this sense. On prose, the doc's `lemmas` are key terms extracted from
        # the BODY, which is a different fact under the same word — see
        # `pipeline_unified._extract_artifact_fields` for why the indexing rule reads the type.
        lemmas: Optional[list] = None,
        # The synsets a COLIMIT merged, and the only thing that says this record IS a merge.
        #
        # Modelled here for the same reason `lemmas` is, and the same bug: `pipeline_unified`
        # cannot read what `from_dict` drops. It matters more than it looks. The first version of
        # the indexing rule keyed on `content_type == "application/x-concept"` — and measured
        # 2026-08-24, that type has two writers: 5,484 colimits whose `lemmas` are the names of
        # everything they absorbed, and 1,165,110 ConceptNet terms whose `lemmas` are their own
        # title split into words. The type cannot tell them apart. `colimit_of` can, because
        # carrying it is what being a merge means.
        colimit_of: Optional[list] = None,
    ):
        super().__init__(id=id, created_time=created_time, modified_time=modified_time)

        if state not in self.VALID_STATES:
            raise ValueError(f"Invalid state '{state}'")

        # First version of an artifact: id == root_id. The root doc persists forever
        # and is the stable target of `collection_artifacts` edges.
        self.root_id = root_id or self.id
        self.collection_id = collection_id
        # A top-level artifact (no parent collection) IS its own origin root. A child
        # is stamped from its parent at create time (`db.create_artifact`);
        # leaving it None here rather than guessing `self.id` means an unstamped
        # child is VISIBLE as unstamped instead of silently claiming to be a root —
        # which would give it its own key tree and orphan it from its collection.
        self.origin_root = origin_root or (self.id if not collection_id else None)
        self.lemmas = lemmas
        self.colimit_of = colimit_of
        self.context = context
        self.content = content
        self.content_encrypted = content_encrypted
        self.content_ref = content_ref
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
            "content_ref": self.content_ref,
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
        # Emitted whenever present, because the round trip is the whole reason this field is on
        # the entity: `from_dict` reading a value that `to_dict` drops is worse than never
        # modelling it, since the first save of a lexicon entry would then delete the words that
        # name it. The trailing filter drops None, so an artifact that has no lemmas — which is
        # every artifact that is not a lexicon entry — emits nothing, exactly as before.
        if self.lemmas is not None:
            d["lemmas"] = list(self.lemmas)
        # Same round-trip contract as `lemmas` above: a save that dropped this would turn a merge
        # into an ordinary concept, and the indexing rule keyed on it would stop firing with
        # nothing to show why.
        if self.colimit_of is not None:
            d["colimit_of"] = list(self.colimit_of)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        return cls(
            id=data.get("id") or data.get("_key"),
            root_id=data.get("root_id"),
            collection_id=data.get("collection_id", ""),
            context=data.get("context", ""),
            content=data.get("content", ""),
            state=state_of(data),
            created_by=data.get("created_by"),
            created_time=data.get("created_time"),
            modified_by=data.get("modified_by"),
            modified_time=data.get("modified_time"),
            origin_root=data.get("origin_root"),
            # `name` is this entity's title slot — `_extract_artifact_fields` falls back to it for
            # the indexed title, and `field_filters._title` reads it too. A document may spell that
            # `name` or `title` depending on which path wrote it: the API files a title inside
            # `context`, a bulk ingest writes one top-level. Reading only `name` dropped it for
            # every ingested artifact, so `glacier.n.01` was indexed on its gloss alone — "a slowly
            # moving mass of ice", which never says "glacier" — and no lexical narrowing could find
            # a concept by its own name.
            name=data.get("name") or data.get("title"),
            description=data.get("description"),
            content_type=data.get("content_type"),
            content_encrypted=bool(data.get("content_encrypted", False)),
            content_ref=data.get("content_ref"),
            lemmas=data.get("lemmas"),
            colimit_of=data.get("colimit_of"),
        )
