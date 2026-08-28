"""Unified artifact indexing pipeline.

Every artifact — regardless of content type — goes through the same path:

    artifact (any type, any collection)
      → SSE: tokenize title/description/tags/content → encrypted posting lists
      → MANTLE: chunk content text → embed → encrypted IVF cells

Both arms are unconditional once the wiring prerequisites (Oracle, S3,
the lattice) are met. No feature flags. The router converts missing
prerequisites to 503 (no plaintext fallback by design).

The vector arm has one prerequisite the wiring cannot supply: a provisioned **AnchorSet**, the
coordinate system every chunk is routed against. Nothing in Mantle derives one, so a node
nobody has provisioned has none, and on that node :func:`_mantle_index_artifact` returns
``ARM_SKIPPED`` for every write while the lexical arm indexes normally. That is the state of a
fresh install, not an error: the write succeeds, the artifact is still narrowed to by its
terms, a WARNING names the gap, and recall answers ordered by how much of the query each hit
matched (``ordering: "coverage"``, an integer stem count as each score) until an operator
provisions anchors. See
:mod:`mantle.search.anchors.store`.

Vector ingress
--------------
The vector arm's embed step has no model behind it — Mantle never embeds. A write may
carry a vector the writer already computed (``vector=`` here, validated at the router
by ``api/vectors.py``), and it enters through the ordinary provider seam as a
:class:`embeddings.WriterSuppliedEmbeddings`. A vector-bearing write indexes as ONE
chunk: the supplied vector describes the artifact, and distributing it over several
chunks would attribute to each of them a claim the writer made about the whole.

Without a vector nothing changes: the provider resolves against the long-term cache
and returns empty for anything it has never seen, and the arm skips.

See `.dev/features/mantle-mvp.md` and
`.dev/features/mantle-sse-lexical-index.md`.
"""

from __future__ import annotations

import re

import logging
import time
from typing import Optional

from mantle.search.embeddings import (
    Embeddings,
    WriterSuppliedEmbeddings,
    model_id as emb_model_id,
)
from mantle.entities.artifact import Artifact

from mantle.search.ingest.chunking import (
    chunk_text,
    extract_text_from_context,
    should_chunk_content,
)
from mantle.search.ingest.tags import (
    normalize_tags,
)
from mantle.search.mantle.oracle import MasterKeyMissing
from mantle.services.acting_principal import KeyCustodyDenied
from mantle.services.bootstrap_types import CREDENTIAL_CONTENT_TYPE
from mantle.services.ingest_runner_service import extract_text_from_artifact
from mantle.services.issuers import ISSUER_CONTENT_TYPE

logger = logging.getLogger(__name__)

_embeddings = Embeddings()


# Each artifact state indexes into its own physically separate index segment
# (separate S3 prefixes — see search.mantle.wiring._segment_prefixes). The
# segment name equals the state name. States are mutually exclusive, so an
# artifact's entry lives in exactly one segment; a transition (draft→committed,
# →archived, unarchive) moves it (index into the new segment, purge the others).
_SEGMENTS = ("committed", "draft", "archived")


def _segment_for_state(state: str) -> str:
    """Map an artifact state to its index segment (1:1).

    An absent or unrecognised state resolves to `Artifact.STATE_WHEN_ABSENT` — the single
    definition of what a doc with no `state` is in (`db/constants`), which every other reader
    derives from too. A second answer here would file a stateless doc into one tree and the entity
    layer would move it to another on the next write."""
    return state if state in _SEGMENTS else Artifact.STATE_WHEN_ABSENT


#: Content types that are never indexed. Two criteria admit an entry, and an artifact
#: qualifies under one of them or it is indexed like everything else.
#:
#: **1 — platform trust configuration, not user content.** All three must hold:
#:
#:   a. The record is owned by the system principal and read back by a
#:      ``created_by`` + content-type query, not through the grant ledger.
#:   b. It is deliberately grantless. There is no principal who should be able to
#:      find it by searching, because a grant that made it indexable would also make
#:      the platform's own trust config a search result for whoever held that grant.
#:   c. It carries no collection. ``create_issuer_artifact`` sets ``collection_id=""``.
#:
#: **2 — the artifact's content IS a secret.** A credential's value is its content: an
#: API key, an OAuth client secret, a refresh token. Indexing it tokenizes that value
#: into the SSE posting lists, and a recall hit hydrates the decrypted value into
#: ``SearchHit.content`` — so the secret becomes reachable by SEARCH as well as by
#: direct read.
#:
#: That is not a leak. Cells stay encrypted at rest, and every posting and every
#: hydration is cut by the same light cone that guards the read path, so no principal
#: learns anything it could not already fetch by id. It is a wider surface, chosen
#: rather than inherited: recall answers a question nobody needs answered about a
#: secret ("which of these contains this string?"), and a full-text index is a second
#: representation of the plaintext, in a second store, with a second decrypt path and
#: a second set of ways to be wrong. A secret is not something you full-text search,
#: so the index does not carry one. Reaching a credential stays a matter of naming it
#: — through the artifact it hangs off, which is indexed and searchable as usual.
#:
#: This is a different exclusion from the storage-plane one (``lattice_api._SIDE_PLANE_CTS``):
#: everything named here is still an artifact — governable, audited, versioned — and stays
#: fully visible to the artifact API, content included. It is excluded only from search.
NON_INDEXABLE_CONTENT_TYPES = frozenset({
    ISSUER_CONTENT_TYPE,                       # criterion 1
    CREDENTIAL_CONTENT_TYPE,                   # criterion 2
})


def is_indexable(artifact: Artifact) -> bool:
    """False for trust config and for secret material — see
    :data:`NON_INDEXABLE_CONTENT_TYPES`."""
    return getattr(artifact, "content_type", None) not in NON_INDEXABLE_CONTENT_TYPES


# ---- Per-arm outcome -------------------------------------------------------
#
#
# SKIPPED is a distinct third answer: "this arm had nothing to do here" (no content,
# prerequisites absent, no AnchorSet provisioned) is not a failure, and counting it
# as one drains the failure count of meaning.
ARM_WRITTEN = "written"
ARM_SKIPPED = "skipped"
ARM_FAILED = "failed"


class IndexOutcome:
    """What each arm did for one artifact.

    Truthy iff no arm failed, so ``if index_artifact(...)`` callers keep their
    meaning; the per-arm detail is available to callers that report counts.
    """

    __slots__ = ("sse", "vector", "reason")

    def __init__(self, *, sse: str, vector: str, reason: str = "") -> None:
        self.sse = sse
        self.vector = vector
        self.reason = reason

    @property
    def failed(self) -> bool:
        return ARM_FAILED in (self.sse, self.vector)

    @property
    def wrote_nothing(self) -> bool:
        return ARM_WRITTEN not in (self.sse, self.vector)

    def __bool__(self) -> bool:
        return not self.failed

    def __repr__(self) -> str:
        detail = f" ({self.reason})" if self.reason else ""
        return f"<IndexOutcome sse={self.sse} vector={self.vector}{detail}>"


# Optional async queue
try:
    from mantle.search.ingest import index_queue
except Exception as exc:  # pragma: no cover — queue optional during static analysis
    logger.error("Failed to import index_queue: %s", exc, exc_info=True)
    index_queue = None  # type: ignore[assignment]


# ============================================================
#  Field extraction (shared by SSE + MANTLE)
# ============================================================


#: The offer as the artifact states it — what it announces about itself in its own words. Every one
#: of these is bounded by the artifact: a title, a one-line description, a tag list.
#:
#: `_extract_artifact_fields` still returns `content` beside these. The body is not indexed as
#: itself; what is indexed is what the body announces — see :func:`_body_offer`.
#: A lexicon entry, whose `lemmas` are the names it goes by rather than terms taken from a body.
#: See `_extract_fields` for why the distinction has to be made by type and not by the field.
LEXICON_CONTENT_TYPE = "text/x-wordnet"

#: Not a discriminator, and the reason this comment is long. `application/x-concept` has two
#: writers on this store, measured 2026-08-24:
#:
#:     cn-*        1,165,110   ConceptNet 5.7 terms.  lemmas are the title, split:
#:                             `12 hour clock` -> ['12', 'hour', 'clock']
#:     concept-*       5,484   `ember.consolidate.colimit` merges.  lemmas are the union of the
#:                             NAMES of the synsets merged: ['apparel','clothes','dress', ...]
#:
#: So a rule keyed on this type promotes 1.16M sets of word fragments into the tags of records that
#: already carry those same words in their title. That is the one-field-two-meanings-two-writers
#: shape the block in `_extract_artifact_fields` is about, and it caught this file out AGAIN — the
#: first version of this constant read the type, exactly as that block advises, and the advice does
#: not hold here because the type does not separate the writers.
CONCEPT_CONTENT_TYPE = "application/x-concept"


def _lemmas_are_names(artifact) -> bool:
    """Does this record's ``lemmas`` hold the names it goes BY, rather than words taken from it?

    Three writers put three different things in one field, so this asks about the record and not
    about the field being present:

    * a **synset** (`text/x-wordnet`) — its lemmas are the words that mean it, and that IS its
      offer; the record exists to say these words mean this;
    * a **colimit** — a merge of synsets, carrying the union of their names. Recognised by
      ``colimit_of``, which names the members and is what makes it a merge. Not by content type:
      ConceptNet shares that type and splits its title into the same field;
    * everything else — on a wiki artifact `lemmas` is key terms `astra/doc_index` pulled out of
      the body, and indexing those would be indexing the content, which this pipeline is built not
      to do.
    """
    if getattr(artifact, "content_type", None) == LEXICON_CONTENT_TYPE:
        return True
    return bool(getattr(artifact, "colimit_of", None))

_STATED_OFFER_FIELDS = ("title", "description", "tags")


def _description_that_adds_something(title: str, description: str) -> str:
    """The description, or ``""`` when it carries no stem the title does not already carry.

    An artifact should not announce itself twice. Measured 2026-08-24 across 2,167,300 artifacts
    on 71/home:

        carry both a title and a description        1,175,579   (54%)
          description CONTAINS the title verbatim   1,164,574   (99% of those)
        carry a description and no title                    0
        description saying anything the title does not 11,005   (0.5% of the corpus)

    and the shape of the 1.16M is one template, repeated:

        title        agience-build/AGENTS.md
        description  Workspace document imported from agience-build/AGENTS.md

    That is PROVENANCE wearing the offer's clothes, and this file already carries the argument
    against it one function up -- "provenance lives in `citation` / `source_path` / `via` and is
    never reachable by the matcher" -- with the measured consequence that promoting a provenance
    string put all 6,480 canon documents on the same two nodes.

    The test is per-artifact and needs no judgement about what a description is FOR: subtract the
    title's stems, and what remains is what the second field contributed. Nothing remaining means
    the artifact said one thing in two places, and the second copy is dropped. This subtracts
    STEMS rather than testing containment, because `oxygen` / `oxygen: a nonmetallic bivalent
    element` contains its title and is not a duplicate -- that is the 11,005.

    NOT a ranking fix: `Coverage.stems` counts DISTINCT query stems, so a stem indexed under two
    fields never counted twice. What this saves is one posting per (duplicated stem x field) over
    1.16M records, and it stops the corpus teaching the index that a template is an offer.
    """
    if not description:
        return ""
    if not title:
        return description
    try:
        from mantle.search.mantle.sse.tokenizer import tokenize
    except Exception:                              # noqa: BLE001 -- no tokenizer, no opinion
        return description
    try:
        title_stems = set(tokenize(title))
        if not title_stems:
            return description
        if set(tokenize(description)) - title_stems:
            return description
    except Exception:                              # noqa: BLE001
        return description
    return ""


def _fields_to_index(fields: dict[str, str]) -> dict[str, str]:
    """Exactly what the lexical arm is given: the stated offer, plus what the body announces.

    A free function so the decision is testable on its own. Inline in `_sse_index_artifact` it
    needs a store, an oracle and an acting principal to reach, which leaves a test exercising the
    indexer rather than the choice of what to hand it.

    The raw body never appears in the result. `_body_offer` returns a projection, and `""` when
    there is nothing to project, in which case the stated offer alone is indexed.

    Both the offer and the projected body belong here. The offer is how a thing is found; the
    content is what the finding is worth. Measured on 150 retrievals:

        the retrieved entry NAMES the answer   (89 items)   +0.1420
        it does not                            (61 items)   +0.0073

    Twenty times the value, and it is carried by the material an offer does not contain. The
    alignment that makes `description` the single offer field invites the next step — "context is
    what we index, so index context" — and that step would delete the thing the value was measured
    in. §84 already leaves ~70% of a canon body outside this index by design, and that 70% is also
    70% of the material that could have held the answer. Both channels, doing different jobs.
    """
    if not fields:
        return {}
    body = _body_offer(fields.get("content") or "")
    fields = dict(fields)
    # One offer, not two. See `_description_that_adds_something`.
    kept = _description_that_adds_something(fields.get("title") or "",
                                            fields.get("description") or "")
    if kept:
        fields["description"] = kept
    else:
        fields.pop("description", None)
    out = {k: v for k, v in fields.items() if k in _STATED_OFFER_FIELDS and v}
    if body:
        out["content"] = body
    return out


def _body_offer(content: str) -> str:
    """What the body announces about itself: the spans that stand out from its own others.

    Still the offer rather than the body: a title is what an artifact says about itself in a line,
    and this is what a body says about itself when nobody supplied a line. What goes into the index
    is a projection, never the text.

    `beacon.density.dense_excerpt` is that projection, and it fits for the same reason it makes a
    poor preview. Entropy is query-independent by construction, so one artifact yields the same span
    to every question asked of it — measured on 71/dev, three unrelated questions each got the same
    368 characters of this project's README. A preview should answer the question that was asked; at
    index time there is no question to answer, and an offer must be the same offer whoever reads it.

    It is bounded without a cap. `top_break` keeps whichever windows stand out, at whatever count
    that turns out to be, so the extent is read off the document rather than chosen for it. That is
    what keeps "a body contributes thousands of distinct stems" from making write cost a function of
    the corpus; the entry layout in `sse/posting.py` covers the part of that cost which is O(entries
    already in the slot).

    Returns `""` for empty content, and for a body so short it is a single window the whole of it is
    its own densest span (`dense_windows`' no-break convention) — which is correct: a two-line body
    announces itself entirely.
    """
    if not content or not content.strip():
        return ""
    try:
        from mantle.search.beacon.density import dense_excerpt
    except Exception:                              # noqa: BLE001 — beacon is the permissive half
        # Without beacon there is no cut, and no cut means no projection. Indexing the raw body
        # instead would reinstate the write cost this whole design avoids, so the honest answer is
        # to index the stated offer alone — which is exactly what happened before this existed.
        logger.debug("beacon.density unavailable; the body announces nothing", exc_info=True)
        return ""
    return dense_excerpt(content)


#: Separators inside a group id. `collection:agience-pharos/features` and `stage.0.lexicon` are
#: both paths; the words in them are what a person would type.
_GROUP_SPLIT = re.compile(r"[:/._\-]+")


def _group_terms(artifact: Artifact) -> list[str]:
    """The groups this artifact belongs to, as searchable words (§112).

    A tag, a collection, a group and an attribute are the same thing: an edge to another artifact.
    So there is no `tags` field to read — membership is the tag set, and `collection_id` /
    `collections` are the field mirror of the `contains` edges that record it
    (`mantle.shard.curate._memberships` reads both names, which is why both are read here).

    A `tags` key in the context blob would be a second, parallel answer to a question the graph
    already answers, and the two could
    disagree with nothing to notice: an artifact moved between collections kept whatever tag
    strings its context happened to carry.

    The id is split into words rather than emitted whole, because a group id is a path and the
    words in it are what a person types — `collection:agience-pharos/features` yields
    `agience`, `pharos`, `features`. `normalize_tags` then dedupes and canonicalises, so a term
    repeated across an artifact's groups costs nothing.
    """
    ids: list[str] = []
    for value in (getattr(artifact, "collections", None) or []):
        if value:
            ids.append(str(value))
    own = str(getattr(artifact, "collection_id", "") or "").strip()
    if own:
        ids.append(own)

    terms: list[str] = []
    seen: set[str] = set()
    for gid in ids:
        # ── the SCHEME is a type marker, not a name ──────────────────────────────────────────
        # `collection:agience-pharos/features` splits to `collection, agience, pharos, features`,
        # and `collection` is then a tag on every artifact that is in a collection — which is all
        # of them. A term carried by every member of a corpus cannot distinguish between them:
        # the same defect as `canon knowledge` in the offer (§111) and the cardinal numbers in
        # the position set (§110), a third time and from a third direction. The part before the
        # first `:` is a scheme by construction, so dropping it is structural rather than a
        # judgement about which words are boring.
        body = gid.split(":", 1)[1] if ":" in gid else gid
        for word in _GROUP_SPLIT.split(body):
            word = word.strip().lower()
            if word and not word.isdigit() and word not in seen:
                seen.add(word)
                terms.append(word)
    return terms


def _extract_artifact_fields(artifact: Artifact) -> dict[str, str]:
    """Build the long-form per-field text dict the SSE indexer wants.

    Returns ``{"title": ..., "description": ..., "tags": ..., "content": ...}``,
    omitting empty fields so the indexer skips them. ``content`` here is
    the full analyzable text (artifact.content + extracted text fields)
    — the same corpus the MANTLE chunker walks for embedding.
    """
    # ── the offer is TOP-LEVEL; `context` is a compatibility read ────────────────────────────
    # Five names, four jobs. `title` is the name; `description` is the offer, the one thing a need
    # is matched against; provenance lives in `citation` / `source_path` / `via` and is never
    # reachable by the matcher; and a tag / collection / group / attribute is an edge to another
    # artifact rather than a field.
    #
    # `context` is read after the explicit fields, so a non-JSON context string cannot be promoted
    # whole to `description`. Two ingests write exactly such a string — "canon knowledge: <doc>
    # §<section>" and "the concept <w>: a ConceptNet 5.7 English term node" — and promoting it makes
    # provenance the stated offer: all 6,480 canon documents then position on `canon.n.01` and
    # `cognition.n.01`, the same two nodes.
    #
    # Top-level first, context second and only when structured, so rows written before the
    # alignment keep working and rows written after are not overridden by a stale carrier.
    text_fields = extract_text_from_context(artifact.context)

    # Decided before the title, because the title falls back to the first of these — see the
    # block below for why this is a decision about the TYPE and not about the field.
    lemma_names: list[str] = []
    if _lemmas_are_names(artifact):
        lemma_names = [str(lemma) for lemma in (getattr(artifact, "lemmas", None) or []) if lemma]

    title = (
        (getattr(artifact, "title", "") or "").strip()
        or text_fields.get("title", "").strip()
        or (getattr(artifact, "name", "") or "").strip()
        # ── a record whose names ARE its lemmas titles itself with the first of them ─────────
        # This is the synset rule below, applied rather than restated: "only the first of them
        # becomes the title". A synset gets that title from its source; measured 2026-08-24,
        # all 5,484 `application/x-concept` colimits carry `lemmas` and NONE of `title` / `name`
        # / `description`, with `content` empty — so without this they extract to `{}`, "no
        # analyzable fields", and the merge is unindexable while every member it consolidates
        # stays findable. That is COMPACTIFICATION §5 inverted, and it is silent: the pipeline
        # reports a skip, not a failure.
        #
        # Derived from `lemmas` rather than read from the `word` field the consolidator also
        # writes, because `word` is that field's first lemma and a second carrier is a second
        # thing to keep true. (It does not survive `Artifact.from_dict` either, which is the
        # form this sees.)
        or (lemma_names[0].strip() if lemma_names else "")
    )
    description = (
        (getattr(artifact, "description", "") or "").strip()
        or text_fields.get("description", "").strip()
    )

    raw_tags = _group_terms(artifact)
    # ── a lexicon entry's OTHER NAMES ────────────────────────────────────────────────────────
    # A synset's `lemmas` are the words that mean it, and for a dictionary entry that IS the
    # offer: the whole point of the record is to say that these words mean this. Only the first
    # of them becomes the title, and the two lexicons order them differently, so:
    #
    #     wn-oewn-14672278-n   title 'O'        lemmas: o, atomic number 8, oxygen
    #     wn-oxygen.n.01       title 'oxygen'   lemmas: oxygen, o, atomic number 8
    #
    # and the OEWN copy is the one in the SSE index. Its gloss ("a nonmetallic bivalent element
    # that is normally a colorless odorless tasteless nonflammable diatomic gas") never says the
    # word either. Measured, `recall("what is oxygen")` did not narrow to that artifact at all and
    # answered `LOX / air / artificial blood` — every hyponym carries `oxygen` inside its own
    # title, so the hyponyms were findable and the concept itself was not.
    #
    # Restricted to the lexicon content type deliberately. `lemmas` does not mean one thing across
    # this corpus: on a wiki artifact it is key terms extracted FROM the body by `astra/doc_index`,
    # and indexing those would be indexing the content, which this pipeline is built not to do
    # ("we should be indexing (needs/offers) on context, not content"). On a synset they are names.
    # One field, two meanings, two writers — the same shape as every other defect in this corpus,
    # which is why this reads the type rather than the field's presence.
    if lemma_names:
        raw_tags = list(raw_tags or []) + lemma_names
    tags_canonical = normalize_tags(raw_tags)
    tags_text = " ".join(t for t in tags_canonical if t)

    content_text = (extract_text_from_artifact(artifact) or "").strip()
    if not content_text:
        content_text = (artifact.content or "").strip()

    fields: dict[str, str] = {}
    if title:
        fields["title"] = title
    if description:
        fields["description"] = description
    if content_text:
        fields["content"] = content_text
    # ── membership DESCRIBES an artifact; it is not one ──────────────────────────────────────
    # Every artifact belongs to a group, so tags alone are never empty, and emitting them for a
    # row with no title, no offer and no content would make "an artifact with nothing to find is
    # not indexed" false for every artifact in the store. It would also index that empty row under
    # its group's words, where it would be returned for every query those words answer.
    #
    # So tags ride along with something to describe, and do not stand alone.
    if tags_text and fields:
        fields["tags"] = tags_text
    return fields


def _build_chunk_id(root_id: str, chunk_id: int) -> str:
    return f"{root_id}:chunk:{chunk_id}"


def _content_chunks(content: str) -> list[dict]:
    """Chunk content for the MANTLE arm.

    Single source of truth so the bulk-reindex prewarm and the per-artifact
    index produce identical chunk texts — identical cache keys, so the warm
    pass actually hits the cache the prewarm populated.
    """
    if not content:
        return []
    if should_chunk_content(content):
        return list(chunk_text(content))
    return [{"chunk_id": 0, "text": content}]


# ============================================================
#  MANTLE vector hook — encrypted IVF, chunks + embeddings
# ============================================================


def _mantle_index_artifact(
    artifact: Artifact,
    collection_id: str,
    fields: dict[str, str],
    *,
    segment: str = "committed",
    vector=None,
) -> str:
    """Chunk + embed the artifact's content, write to MANTLE cells.

    ``vector`` is a writer-supplied :class:`api.vectors.SuppliedVector` when the write
    carried one. It selects both the provider (the vector itself, through
    :class:`embeddings.WriterSuppliedEmbeddings`) and the chunking (one chunk, since the
    vector describes the artifact rather than any part of it).

    Returns :data:`ARM_WRITTEN` / :data:`ARM_SKIPPED` / :data:`ARM_FAILED`; see
    :func:`_sse_index_artifact` for why the status is a return value and not a log line.

    On a node with no provisioned AnchorSet this returns :data:`ARM_SKIPPED` for every
    artifact — the write still succeeds and still indexes lexically. That is the fresh-install
    state and it does not clear itself; see the AnchorSet check below and
    :mod:`mantle.search.anchors.store`.
    """
    content = fields.get("content", "")
    if not content:
        return ARM_SKIPPED

    if not collection_id:
        return ARM_SKIPPED

    artifact_root = artifact.root_id or artifact.id

    if vector is not None:
        # One vector, one chunk. The writer embedded this artifact, not a window of it,
        # so the record's text is the artifact's text and there is nothing to split.
        embedded_chunks = [{"chunk_id": 0, "text": content}]
        embedder = Embeddings(WriterSuppliedEmbeddings(vector.values, vector.space_id))
        supplied_space_id = vector.space_id
    else:
        # Chunk + embed (shared chunker — see _content_chunks). On the bulk-reindex
        # path these exact texts were already embedded in batch, so this call is a
        # cache hit (no per-artifact round-trip).
        embedded_chunks = [c for c in _content_chunks(content) if c.get("text")]
        embedder = _embeddings
        supplied_space_id = None

    texts = [c["text"] for c in embedded_chunks]
    if not texts:
        return ARM_SKIPPED
    try:
        embeddings = embedder(texts)
    except Exception:
        logger.warning(
            "MANTLE: embedding failed for artifact %s",
            artifact.id, exc_info=True,
        )
        return ARM_FAILED
    if not any(embeddings):
        return ARM_SKIPPED

    # The AnchorSet is the one coordinate system (canonical plan §3). It is seeded by a client,
    # never derived here (deriving it locally mints region ids no peer computes), so its absence
    # is a deployment state, not an error: skip the vector arm and let the commit succeed.
    try:
        from mantle.search.anchors.store import (
            AnchorSetNotProvisioned,
            require_live_anchorset,
        )
        anchorset = require_live_anchorset()
    except AnchorSetNotProvisioned:
        # One line, no traceback: the exception's message is self-contained, and this
        # state is a provisioning gap rather than a fault in this call path.
        logger.warning(
            "MANTLE: no AnchorSet provisioned; skipping vector index for %s",
            artifact.id,
        )
        return ARM_SKIPPED
    except Exception:
        logger.warning(
            "MANTLE: AnchorSet unavailable; skipping vector index for %s",
            artifact.id, exc_info=True,
        )
        return ARM_SKIPPED

    # Provenance is the space the vector actually came from. A writer-supplied vector
    # lives in the space the writer named, so its own name is recorded rather than the
    # AnchorSet's — labelling it with the local space is what would make two
    # incomparable vectors look comparable later.
    chunk_model_id = supplied_space_id or anchorset.model_id or emb_model_id()

    mantle_chunks = []
    for i, emb in enumerate(embeddings):
        if not emb:
            continue
        record = {
            "artifact_id": artifact_root,
            "chunk_id": int(embedded_chunks[i].get("chunk_id", i)),
            "embedding": emb,
            "text": embedded_chunks[i].get("text", ""),
            "model_id": chunk_model_id,
        }
        if supplied_space_id:
            # The space a writer named for their own vector, kept as they gave it. It is the
            # record of which coordinate system these numbers are statements in, which is the
            # one thing that cannot be recovered from the numbers.
            record["space_id"] = supplied_space_id
        mantle_chunks.append(record)
    if not mantle_chunks:
        return ARM_SKIPPED

    try:
        from mantle.services.dependencies import get_store_db
        from mantle.search.mantle.principal import resolve_cell_principal
        from mantle.search.mantle.wiring import build_indexer
    except Exception:
        logger.debug("MANTLE wiring unavailable; skipping vector index", exc_info=True)
        return ARM_SKIPPED

    try:
        store_db = next(get_store_db())
    except Exception:
        logger.debug("MANTLE: lattice handle unavailable; skipping", exc_info=True)
        return ARM_SKIPPED

    indexer = build_indexer(store_db, segment=segment)
    if indexer is None:
        logger.debug("MANTLE indexer prerequisites missing; skipping")
        return ARM_SKIPPED

    # The cell-key principal is the collection's immutable origin root (not
    # created_by / ownership) — index and query resolve it identically, so the
    # same key is derived at both ends. See search.mantle.principal.
    principal_id = resolve_cell_principal(store_db, collection_id)
    if not principal_id:
        return ARM_SKIPPED

    try:
        touched = indexer.index_artifact(
            principal_id, collection_id, mantle_chunks,
            _ingest_key_request(principal_id),
        )
        logger.info(
            "MANTLE indexed artifact %s (principal=%s collection=%s, %d cells)",
            artifact.id, principal_id, collection_id, touched,
        )
        return ARM_WRITTEN
    except Exception:
        logger.warning(
            "MANTLE indexing failed for artifact %s",
            artifact.id, exc_info=True,
        )
        return ARM_FAILED



def _ingest_key_request(principal_id: str):
    """The key request for a write into ``principal_id``'s cells, as the acting caller.

    The acting principal reaches this layer (:mod:`services.acting_principal`), so the
    question the light cone answers is the real one: *may this caller write into this
    context?* — ``GRANT`` with ``action="update"``. The grant ledger checks it, in the
    same place and by the same code path as the query arm.

    System-initiated indexing — ``collection_service`` auto-index on create, the
    ``init_search`` bulk reindex, the seed path — has no request context and must
    therefore declare its identity explicitly with
    :func:`services.acting_principal.system_acting_context`. It runs as the platform
    system principal and is checked like any other principal; it is not exempt.
    Anything that forgets raises ``NoActingPrincipal`` rather than quietly indexing
    under an unchecked identity.
    """
    from mantle.search.mantle.oracle import KeyPurpose, KeyRequest

    from mantle.services.acting_principal import require_acting_principal

    actor = require_acting_principal()
    return KeyRequest(requester_id=actor.principal_id, purpose=KeyPurpose.GRANT,
                      requester_type=actor.principal_type, action="update")


def _read_key_request(principal_id: str):
    """The key request for reading a principal's cells back (not writing them).

    Same identity rule as :func:`_ingest_key_request`, but ``action="read"`` — so a
    caller who may read but not write is not refused, and a read never mints a
    master key that would decrypt nothing.
    """
    from mantle.search.mantle.oracle import KeyPurpose, KeyRequest

    from mantle.services.acting_principal import require_acting_principal

    actor = require_acting_principal()
    return KeyRequest(requester_id=actor.principal_id, purpose=KeyPurpose.GRANT,
                      requester_type=actor.principal_type, action="read")


def _mantle_remove_artifact(
    principal_id: str, collection_id: str, artifact_id: str,
    *, segment: str = "committed",
) -> None:
    """Strip an artifact's chunks from MANTLE cells in one index segment."""
    try:
        from mantle.services.dependencies import get_store_db
        from mantle.search.mantle.wiring import build_indexer
        store_db = next(get_store_db())
        indexer = build_indexer(store_db, segment=segment)
        if indexer is None:
            return
        indexer.remove_artifact(
            principal_id, collection_id, artifact_id,
            _ingest_key_request(principal_id),
        )
    except Exception:
        logger.warning(
            "MANTLE remove failed for artifact %s (owner=%s, collection=%s)",
            artifact_id, principal_id, collection_id, exc_info=True,
        )


# ============================================================
#  MANTLE-SSE hook — encrypted lexical, posting lists
# ============================================================


def _sse_index_artifact(
    artifact: Artifact,
    collection_id: str,
    fields: dict[str, str],
    *,
    segment: str = "committed",
    indexer=None,
) -> str:
    """Write per-field text into the SSE blind-token posting lists.

    Returns one of :data:`ARM_WRITTEN` / :data:`ARM_SKIPPED` / :data:`ARM_FAILED`,
    which the caller reports. Exceptions stay swallowed here so the other arm and
    the commit survive; the return value is what carries the result out.

    The lexical arm indexes the offer rather than the body. `db/vertex.py` states it —
    "`doc['context']` is conceptually the offer" — and the `offer` column with `ix_v_offer` carries
    it. What narrows a search is what an artifact announces itself as; the body is what the semantic
    arm answers with, and it goes there (`_mantle_index_artifact` reads `fields["content"]`
    untouched).

    Indexing the raw body would make write cost a function of the corpus, which is the one thing
    this system cannot have: the aperture is finite and the substrate is not. `sse/posting.py`
    maintains
    each blind token's posting list by READ-MODIFY-WRITE — `get_posting` → decrypt every entry →
    `upsert_entry`'s linear scan → re-encrypt every entry → `put_posting` — so ONE term costs
    O(artifacts already carrying that term). A body contributes thousands of distinct stems, and
    `indexer.index_artifact` multiplies each by its prefixes and bigrams. Measured on 71/home
    (2.9M vertices, 12.1M edges) before this changed:

        raw SQLite insert into the 9.7 GB store   0.008s   <- the substrate was never the problem
        POST, no content and no name              0.2s     <- nothing to index
        POST, no content, ONE name               14.6s
        POST, 4KB of real prose                  16.4s
        POST, 4KB of `'x '` (one distinct term)   3.5s     <- cost is terms, not bytes

    The same signature explains an earlier puzzle: an 800 KB transcript of `'x '` stored fine while
    a 120 KB one of real prose blew every timeout in the chain. Every cap written to work around
    that — `_MAX_CHARS`, `_MIN_GROWTH_CHARS`, the raised HTTP and hook budgets — was scar tissue
    over this line, and none of them addressed it.

    An artifact's offer is bounded by the artifact. Its body is bounded by nothing.

    The body reaches the index as a projection, under the `content` field, via :func:`_body_offer`.
    The rule above holds: the raw text never reaches a posting list. Two properties make that
    affordable.

    *The O(slot) term is gone.* `sse/posting.py` holds one sealed blob per `(artifact, collection)`
    rather than one per term, so adding an artifact to a term does not decrypt, scan and re-encrypt
    every entry already there. That is the term that made the numbers above grow with the corpus
    rather than with the write.

    *The remaining term is bounded by the cut rather than by a cap.* "A body contributes thousands
    of distinct stems" is true of a body and would be unaffordable; it is not true of the
    projection, because `beacon.density` keeps whichever windows stand out from the document's own
    others, so the indexed extent is read off the document rather than decreed for it.

    The corpus is heterogeneous in this. An artifact carries its full text under `content`, none of
    it, or the projection, according to what the write path did at the time; the read path probes
    `content` on every stem, so all three answer, with different amounts of the body behind them. A
    reindex is what makes it uniform.
    """
    if not fields:
        return ARM_SKIPPED

    fields = _fields_to_index(fields)
    if not fields:
        return ARM_SKIPPED

    if not collection_id:
        return ARM_SKIPPED
    artifact_id = artifact.root_id or artifact.id

    try:
        from mantle.services.dependencies import get_store_db
        from mantle.search.mantle.principal import resolve_cell_principal
        from mantle.search.mantle.wiring import build_sse_indexer
    except Exception:
        logger.debug("SSE wiring unavailable; skipping lexical index", exc_info=True)
        return ARM_SKIPPED

    try:
        store_db = next(get_store_db())
    except Exception:
        logger.debug("SSE: lattice handle unavailable; skipping", exc_info=True)
        return ARM_SKIPPED

    # A caller may supply the indexer. Building one per artifact is not merely wasteful — each
    # `SqlitePostingStore` carries its own connection and its own thread-local transaction depth,
    # so a fresh instance per artifact CANNOT join a transaction the caller has already opened.
    # Sharing one is what lets a bulk pass commit a whole chunk once instead of once per artifact,
    # and SQLite commit cost is fsync-dominated. `None` keeps the per-artifact path exactly as it
    # was, so every existing caller is unchanged.
    if indexer is None:
        indexer = build_sse_indexer(store_db, segment=segment)
    if indexer is None:
        logger.debug("SSE indexer prerequisites missing; skipping")
        return ARM_SKIPPED

    # Same principal as the vector arm: the collection's origin root.
    principal_id = resolve_cell_principal(store_db, collection_id)
    if not principal_id:
        return ARM_SKIPPED

    try:
        n = indexer.index_artifact(
            principal_id, collection_id, artifact_id, fields,
            _ingest_key_request(principal_id),
        )
        logger.info(
            "SSE indexed artifact %s (principal=%s collection=%s, %d tokens)",
            artifact.id, principal_id, collection_id, n,
        )
        return ARM_WRITTEN
    except Exception:
        logger.warning(
            "SSE indexing failed for artifact %s",
            artifact.id, exc_info=True,
        )
        return ARM_FAILED


def _sse_remove_artifact(
    principal_id: str, artifact_id: str, *, segment: str = "committed",
) -> None:
    """Strip an artifact's references from the SSE index in one segment."""
    try:
        from mantle.services.dependencies import get_store_db
        from mantle.search.mantle.wiring import build_sse_indexer
        store_db = next(get_store_db())
        indexer = build_sse_indexer(store_db, segment=segment)
        if indexer is None:
            return
        indexer.remove_artifact(
            principal_id, artifact_id, _ingest_key_request(principal_id),
        )
    except Exception:
        logger.warning(
            "SSE remove failed for artifact %s (owner=%s)",
            artifact_id, principal_id, exc_info=True,
        )


# ============================================================
#  Public API: index / batch / delete
# ============================================================


def move_artifact_segments(
    artifact: Artifact, collection_id: str, *, remove_from: list[str],
) -> None:
    """Remove the artifact's root from the given index segments.

    Called at a state transition to vacate the segment(s) the artifact is leaving
    (the new segment is (re)indexed separately). The index is root-keyed, and a
    root can legitimately occupy two segments at once — a committed version and a
    WIP draft of the same root coexist — so we never blanket-purge "all others";
    the caller names exactly the segment(s) being left:

        commit    (draft→committed):  remove_from=["draft"]
        archive   (→archived):        remove_from=["committed", "draft"]
        unarchive (archived→draft):   remove_from=["archived"]

    Best-effort and gated on the index backend being available, so it's a fast
    no-op where search isn't wired (no principal query is run).
    """
    if not collection_id or not remove_from:
        return
    try:
        from mantle.search.mantle.wiring import build_indexer
        # Availability gate first — build_indexer doesn't use the db handle, and
        # resolving the handle is the expensive part. This makes the whole thing
        # a fast no-op where search isn't wired (e.g. tests with no S3/oracle),
        # before any DB connection is attempted.
        if build_indexer(None) is None:
            return
        from mantle.services.dependencies import get_store_db
        from mantle.search.mantle.principal import resolve_cell_principal
        store_db = next(get_store_db())
        principal_id = resolve_cell_principal(store_db, collection_id)
    except Exception:
        logger.warning(
            "segment move: cell principal unresolved for collection %s — the "
            "artifact was NOT vacated from segments %s and will keep matching "
            "searches of a state it has left",
            collection_id, remove_from, exc_info=True,
        )
        return
    if not principal_id:
        return
    root = artifact.root_id or artifact.id
    for seg in remove_from:
        if seg not in _SEGMENTS:
            continue
        _mantle_remove_artifact(principal_id, collection_id, root, segment=seg)
        _sse_remove_artifact(principal_id, root, segment=seg)


def get_artifact_embeddings(artifact: Artifact, collection_id: str) -> list[dict]:
    """Return the artifact's stored MANTLE vector chunk records for its current state's
    segment: each ``{chunk_id, embedding, model_id, ...}``, ordered by chunk_id.

    Reads the vectors back out of the encrypted cells — the inverse of indexing.
    Empty if the vector arm isn't wired (no cell store) or the artifact has
    nothing stored (e.g. a container, or a lexical-only deploy with no embeddings
    provider configured; see search/embeddings.py)."""
    if not collection_id:
        return []
    segment = _segment_for_state(artifact.state)
    try:
        from mantle.services.dependencies import get_store_db
        from mantle.search.mantle.principal import resolve_cell_principal
        from mantle.search.mantle.wiring import build_indexer
        store_db = next(get_store_db())
        indexer = build_indexer(store_db, segment=segment)
        if indexer is None:
            return []
        principal_id = resolve_cell_principal(store_db, collection_id)
        if not principal_id:
            return []
        root = artifact.root_id or artifact.id
        chunks = [
            c for c in indexer.collection_chunks(
                # read, not update: this reads vectors back out.
                principal_id, collection_id, _read_key_request(principal_id),
            )
            if c.get("artifact_id") == root
        ]
        chunks.sort(key=lambda c: c.get("chunk_id", 0))
        return chunks
    except MasterKeyMissing:
        # Never indexed, so there is nothing to read back. Genuinely empty.
        return []
    except KeyCustodyDenied:
        raise
    except Exception:
        logger.debug(
            "get_artifact_embeddings failed for %s", getattr(artifact, "id", "?"),
            exc_info=True,
        )
        return []


def index_artifact(
    artifact: Artifact,
    collection_id: str,
    *,
    is_head: bool = True,
    fields: Optional[dict[str, str]] = None,
    vector=None,
    sse_indexer=None,
) -> IndexOutcome:
    """Index one artifact into the index segment for its current state.

    ``draft`` / ``committed`` / ``archived`` each have a separate physical index
    (separate S3 prefixes per arm), so the artifact is written into the segment
    matching its state. This does not touch the other segments — a root may hold
    a committed version *and* a WIP draft simultaneously; vacating a segment on a
    state transition is the caller's job via :func:`move_artifact_segments`.

    ``is_head`` is preserved for caller compatibility and does not drive index
    branching (versioning is artifact-level). ``fields`` may be supplied by the
    caller — the bulk reindex extracts them once and prewarms the embeddings
    cache, then passes them here so this path neither re-extracts nor makes a
    per-artifact embed round-trip. When ``None`` they are extracted here.

    ``vector`` is a writer-supplied :class:`api.vectors.SuppliedVector` when the write
    carried one; it feeds the vector arm and is ignored by the lexical arm, which reads
    text. Nothing else in the pipeline changes shape when it is ``None``.

    Returns an :class:`IndexOutcome` naming what each arm did. It is truthy iff no
    arm failed, so ``if index_artifact(...)`` keeps its meaning, and "both arms were
    refused" is distinguishable from "both arms wrote".
    """
    segment = _segment_for_state(artifact.state)
    try:
        if not is_indexable(artifact):
            logger.debug(
                "Artifact %s is platform trust config (%s); not a search target",
                artifact.id, getattr(artifact, "content_type", None),
            )
            return IndexOutcome(
                sse=ARM_SKIPPED, vector=ARM_SKIPPED, reason="non-indexable content type",
            )
        if fields is None:
            fields = _extract_artifact_fields(artifact)
        if not fields:
            logger.debug(
                "Artifact %s has no analyzable fields; skipping", artifact.id,
            )
            return IndexOutcome(
                sse=ARM_SKIPPED, vector=ARM_SKIPPED, reason="no analyzable fields",
            )
        outcome = IndexOutcome(
            sse=_sse_index_artifact(artifact, collection_id, fields, segment=segment,
                                    indexer=sse_indexer),
            vector=_mantle_index_artifact(
                artifact, collection_id, fields, segment=segment, vector=vector,
            ),
        )
        # Name the per-arm result rather than logging a single unconditional success line,
        # so an artifact whose every arm was refused doesn't read as indexed.
        logger.log(
            logging.WARNING if outcome.failed else logging.INFO,
            "Indexed artifact %s in collection %s (segment=%s): sse=%s vector=%s",
            artifact.id, collection_id, segment, outcome.sse, outcome.vector,
        )
        return outcome
    except Exception:
        logger.error(
            "Indexing failed for artifact %s",
            artifact.id, exc_info=True,
        )
        return IndexOutcome(sse=ARM_FAILED, vector=ARM_FAILED, reason="pipeline error")


# ============================================================
#  Bulk reindex prep: extract fields once + batch-warm embeddings
# ============================================================

# Texts per Embeddings() call when prewarming the bulk reindex. A batch resolves from
# the long-term cache only — texts already embedded resolve, uncached texts come back
# empty — so this bounds per-call list size.
EMBED_BATCH_SIZE = 64


def prepare_reindex_items(
    items: list[tuple[str, Artifact]],
    *,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[tuple[str, Artifact, dict[str, str]]]:
    """Prepare a bulk-reindex work list with the embeddings cache prewarmed.

    For each ``(collection_id, artifact)``: extract its analyzable fields once
    (skipping archived / field-less artifacts) and collect every unique MANTLE
    chunk text across all artifacts. Then embed those texts in batched HTTP
    calls, which populates the long-term embeddings cache.

    The returned ``(collection_id, artifact, fields)`` tuples feed
    :func:`index_artifact` (pass ``fields=...``); its per-artifact embed then
    hits the warm cache instead of making one round-trip per artifact.

    Built for a cold cache — the batched prewarm is the fast path, not an
    optimization that assumes prior warmth. Identical texts (boilerplate shared
    across artifacts) are embedded once.
    """
    prepared: list[tuple[str, Artifact, dict[str, str]]] = []
    all_texts: list[str] = []
    seen: set[str] = set()

    for collection_id, artifact in items:
        # All states are indexed (into their own segment); the prewarm cache
        # is keyed by chunk text, so it's segment-agnostic.
        try:
            fields = _extract_artifact_fields(artifact)
        except Exception:
            logger.warning(
                "reindex prep: field extraction failed for %s",
                artifact.id, exc_info=True,
            )
            continue
        if not fields:
            continue
        prepared.append((collection_id, artifact, fields))
        for chunk in _content_chunks(fields.get("content", "")):
            text = chunk.get("text")
            if text and text not in seen:
                seen.add(text)
                all_texts.append(text)

    if all_texts:
        batches = (len(all_texts) + batch_size - 1) // batch_size
        logger.info(
            "Reindex prewarm: embedding %d unique chunk texts in %d batch(es) of <=%d",
            len(all_texts), batches, batch_size,
        )
        start = time.time()
        embedded = 0
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            try:
                _embeddings(batch)  # populates the long-term cache
                embedded += len(batch)
            except Exception:
                logger.warning(
                    "Reindex prewarm: batch embed failed at offset %d", i, exc_info=True,
                )
        logger.info(
            "Reindex prewarm complete: %d/%d texts embedded in %.2fs",
            embedded, len(all_texts), time.time() - start,
        )

    return prepared


def index_artifacts_batch(
    artifacts: list[Artifact],
    collection_id: str,
    *,
    is_head: bool = True,
) -> bool:
    """Bulk-index a list of artifacts.

    Each artifact runs its own SSE + MANTLE flow. Embedding batching
    happens inside :func:`_mantle_index_artifact` — the MANTLE indexer
    handles per-artifact embedding without cross-artifact batching.
    For very large bulk reindex jobs, the admin command runs many of
    these in parallel via the index queue.
    """
    if not artifacts:
        return True

    start_time = time.time()
    logger.info("Starting bulk index of %d artifacts", len(artifacts))
    indexed = 0
    skipped = 0
    failed = 0

    for artifact in artifacts:
        # index_artifact routes each artifact to its state's segment (no skips).
        outcome = index_artifact(artifact, collection_id, is_head=is_head)
        if outcome.failed:
            failed += 1
        elif outcome.wrote_nothing:
            skipped += 1
        else:
            indexed += 1

    total_time = time.time() - start_time
    logger.log(
        logging.WARNING if failed else logging.INFO,
        "Bulk index: %d written, %d skipped, %d FAILED of %d in %.3fs",
        indexed, skipped, failed, len(artifacts), total_time,
    )
    return failed == 0


def delete_artifact_from_index(
    version_id: str,
    root_id: Optional[str] = None,
    *,
    principal_id: Optional[str] = None,
    collection_id: Optional[str] = None,
) -> bool:
    """Remove an artifact from MANTLE vector + MANTLE-SSE lexical indexes.

    ``principal_id`` is required for both arms. ``collection_id`` is required
    for the MANTLE vector arm (cells are scoped per collection); the SSE
    arm scans the artifact's manifest and removes from every posting
    list it appears in regardless of collection.

    Callers without ``principal_id`` get a no-op — there's nothing to
    remove without identity.
    """
    try:
        root = root_id or version_id
        # Resolve identity here rather than at seven call sites — the same derivation the write
        # path uses (`resolve_cell_principal`, above), so index and de-index agree on the key by
        # construction instead of by convention. Callers need only pass `collection_id`.
        # The artifact row is already deleted by the time this runs (workspace_service purges
        # the lattice first), so the principal cannot be recovered from the artifact — it has to come
        # from the container, which still exists. That is why `collection_id` is the argument
        # callers supply.
        if collection_id and not principal_id:
            try:
                from mantle.services.dependencies import get_store_db
                from mantle.search.mantle.principal import resolve_cell_principal
                # `next(...)` is not optional: `get_store_db` is a generator function, so a bare
                # call yields a generator object, not a Database — `get_origin_root` would then
                # raise, and the caller must not swallow that into a wrong-but-plausible fallback.
                # Matches every other use of `get_store_db()` in this file.
                principal_id = resolve_cell_principal(next(get_store_db()), collection_id)
            except Exception:
                logger.warning(
                    "delete_artifact_from_index(%s): could not resolve the cell principal for "
                    "collection %s — nothing will be removed from the index",
                    version_id, collection_id, exc_info=True,
                )
        # Hard delete: the artifact is gone for good, so purge it from every
        # segment (we don't track which state it was last indexed under).
        for seg in _SEGMENTS:
            if principal_id and collection_id:
                _mantle_remove_artifact(principal_id, collection_id, root, segment=seg)
            if principal_id:
                _sse_remove_artifact(principal_id, root, segment=seg)
        if not (principal_id and collection_id):
            logger.warning(
                "delete_artifact_from_index(%s): no principal_id/collection_id supplied, so NOTHING "
                "was removed from the search index — the artifact's chunks and postings (including "
                "their plaintext text) remain searchable. Caller must pass identity.", version_id,
            )
            return False
        logger.info("Deleted artifact %s from search", version_id)
        return True
    except Exception:
        logger.error(
            "Failed to delete artifact %s from search",
            version_id, exc_info=True,
        )
        return False


# ============================================================
#  Enqueue helpers
# ============================================================


def _mark_indexed(artifact_id: str) -> None:
    """Stamp the materialization marker after the indexing work, from inside the job.

    Stamping at enqueue time instead would say "this was queued" while the only reader
    (`workspace_service`, as a skip condition) reads it as "this is indexed" — with a queue
    configured to return immediately, the job runs later on a worker thread, so a job that never
    ran would leave an artifact stamped done, carrying no postings, and never re-enqueued: stored
    and unfindable, with nothing reporting it.

    Not hypothetical: `mantle` was stopped and restarted several times on `71/home` on
    2026-08-25 — twice in an unplanned outage, once for the WAL maintenance window, once when the
    supervisor moved under its scheduled task. Every index job queued at those moments was dropped.

    Stamping here makes a dropped job self-healing: no marker, so the next access re-enqueues it.
    The cost, which is real and worth stating: between the write and the job finishing,
    `is_materialized` reads False, so a second write in that window enqueues a duplicate job.
    Indexing overwrites, so that spends work rather than correctness — the opposite trade to
    stamping at enqueue time, which spends correctness silently.
    """
    try:
        from mantle.db.backend import mark_materialized, store_handle

        mark_materialized(store_handle(), artifact_id)
    except Exception:
        # `mark_materialized` is best-effort by contract — a missing marker costs a re-index, not
        # correctness. Logged rather than swallowed, because "the marker never wrote" and "the
        # marker was never needed" must not look the same from outside.
        logger.warning("could not mark %s materialized after indexing", artifact_id, exc_info=True)


def enqueue_index_artifact(
    artifact: Artifact,
    collection_id: str,
    *,
    is_head: bool = True,
    tenant_id: Optional[str] = None,
    vacate: Optional[list[str]] = None,
    vector=None,
) -> None:
    """Enqueue an artifact for async indexing; falls back to sync.

    ``vacate`` names index segment(s) the artifact is leaving on a state
    transition — they're removed in the same job, right after (re)indexing into
    the new state's segment. Folding the move into this one job keeps the
    transition atomic from the caller's view and gives a single mock point.

    ``vector`` rides the job. It is captured by the closure rather than re-read later,
    because the write that carried it has already returned by the time the job runs.
    """
    def _act() -> bool:
        outcome = index_artifact(artifact, collection_id, is_head=is_head, vector=vector)
        if vacate:
            move_artifact_segments(artifact, collection_id, remove_from=vacate)
        # bool(outcome) is False iff an arm failed — a skip is not a job failure.
        ok = bool(outcome)
        if ok:
            # Only on success: a failed arm must leave no marker, so the work is retried.
            _mark_indexed(artifact.id)
        return ok

    desc = f"index artifact {artifact.id} -> {collection_id}"
    if index_queue:
        try:
            index_queue.enqueue(_act, description=desc, tenant_id=tenant_id)
            return
        except RuntimeError:
            pass
    logger.debug("Index queue unavailable, indexing synchronously: %s", desc)
    _act()


def enqueue_index_artifacts_batch(
    artifacts: list[Artifact],
    collection_id: str,
    *,
    is_head: bool = True,
    tenant_id: Optional[str] = None,
) -> None:
    """Enqueue a batch for async bulk indexing; falls back to sync."""
    def _act() -> bool:
        ok = bool(index_artifacts_batch(artifacts, collection_id, is_head=is_head))
        if ok:
            for _a in artifacts:
                _mark_indexed(_a.id)
        return ok

    desc = f"batch index {len(artifacts)} artifacts -> {collection_id}"
    if index_queue:
        try:
            index_queue.enqueue(_act, description=desc, tenant_id=tenant_id)
            return
        except RuntimeError:
            pass
    logger.debug("Index queue unavailable, indexing batch synchronously: %s", desc)
    _act()


__all__ = [
    "ARM_FAILED",
    "ARM_SKIPPED",
    "ARM_WRITTEN",
    "IndexOutcome",
    "NON_INDEXABLE_CONTENT_TYPES",
    "is_indexable",
    "delete_artifact_from_index",
    "enqueue_index_artifact",
    "enqueue_index_artifacts_batch",
    "get_artifact_embeddings",
    "index_artifact",
    "index_artifacts_batch",
    "move_artifact_segments",
    "prepare_reindex_items",
]
