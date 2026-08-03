"""MantleSseSearchAccessor — SearchResult-shaped adapter (Step 2.6.9).

Bridges the MANTLE-SSE unified accessor (which returns ``UnifiedHit``)
into the ``search(SearchQuery) -> SearchResult`` contract the artifacts
router expects. After the lexical-backend retirement (Step 2.6.9 part 2), this
is the canonical search backend — there's no legacy fallback.

Adapter responsibilities:

1. Parse the query (reuse :func:`parse_query`) so empty / corrected
   queries get the same metadata the prior accessor produced.
2. Resolve light-cone authorized contexts via
   :func:`resolve_authorized_contexts`.
3. Embed the query for the vector arm if an :class:`Embeddings` is
   wired. Embedding errors don't fail the search — the SSE arm survives.
4. Run the unified accessor's RRF fusion.
5. Hydrate each :class:`UnifiedHit` into a :class:`SearchHit` by reading
   the artifact's metadata from the lattice (since neither index stores
   plaintext text).

See ``.dev/features/mantle-sse-lexical-index.md`` § Query Flow.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

from mantle.embeddings import Embeddings

from ..lightcone import LightConeResolver
from ..oracle import KeyPurpose, KeyRequest
from .unified import MantleUnifiedAccessor, UnifiedHit

logger = logging.getLogger(__name__)


def _raw_artifact(store_db, artifact_id: str):
    """Raw artifact doc by id — mode-selected via `db.backend.get_raw_artifact`
    (one implementation; never shape-sniff the handle — mocks have every attribute)."""
    from mantle.db.backend import get_raw_artifact
    return get_raw_artifact(store_db, artifact_id)


def _key_request(query, action: str = "read") -> KeyRequest:
    """The REQUESTER identity for every key this search will need.

    Built from ``query.user_id`` — the principal actually making the request —
    NOT from the objects being read. That inversion is the whole point of FIX 3:
    previously the oracle derived a key from whatever principal the *object*
    named, so identity flowed from the data instead of from the caller, and any
    caller could obtain any principal's key.

    Raises ``ValueError`` on a missing ``user_id`` rather than defaulting to an
    anonymous request: an unauthenticated search must fail closed, and a search
    with no requester is exactly that.
    """
    user_id = getattr(query, "user_id", None)
    if not user_id:
        raise ValueError(
            "search requires a requesting principal (query.user_id); refusing to "
            "issue keys for an anonymous request"
        )
    return KeyRequest(
        requester_id=str(user_id), purpose=KeyPurpose.GRANT,
        requester_type="user", action=action,
    )


def resolve_authorized_contexts(
    store_db,
    principal_id: str,
    *,
    lightcone: LightConeResolver,
    action: str = "read",
    principal_type: str = "user",
) -> List[Tuple[str, str]]:
    """Map the resolver's authorized artifact set into ``(cell_principal, collection)`` contexts.

    The light-cone resolver returns a flat set of artifact ids the
    requesting principal can ``read``. Each authorized artifact's
    ``collection_id`` is the MANTLE / SSE search scope; its **cell-key
    principal** is the collection's immutable origin root (NOT the
    artifact's ``created_by``) — the exact same value the index path used,
    so the derived keys match. We dedupe ``(cell_principal, collection)``.

    ``principal_type`` is the requester's acting-context entity kind. It defaults to
    ``"user"`` because the two query call sites below are user searches; the SYSTEM
    principal reaches this through the oracle's grant verifier with ``"service"``, and
    the resolver maps both onto the ledger's grantee vocabulary (see
    ``lightcone.ledger_grantee_type``). Before 2026-07-31 this parameter did not exist
    here and the resolver always ran at its ``"user"`` default.

    Returns an empty list when the principal has no authorized artifacts
    or when the lattice lookups fail. Empty result is safe — both engines
    return no hits for empty contexts.

    Originally lived in `mantle.search.mantle.accessor` (the legacy
    legacy-lexical+MANTLE fusion accessor); moved here when that module went
    away in Step 2.6.9 part 2.
    """
    from ..principal import resolve_cell_principal

    authorized = lightcone.resolve(
        principal_id, action=action, principal_type=principal_type
    )
    if not authorized:
        return []

    pairs: set[Tuple[str, str]] = set()
    principal_by_collection: dict[str, str] = {}
    for artifact_id in authorized:
        try:
            doc = _raw_artifact(store_db, artifact_id)
        except Exception:  # noqa: BLE001 — store reads can raise broadly
            continue
        if not doc:
            continue
        # A root artifact (no parent collection) self-references its own id.
        # `_key` is the legacy doc-id shape, `id` the lattice shape — read both.
        collection_id = doc.get("collection_id") or doc.get("_key") or doc.get("id")
        if not collection_id:
            continue
        collection_id = str(collection_id)
        cell_principal = principal_by_collection.get(collection_id)
        if cell_principal is None:
            cell_principal = resolve_cell_principal(store_db, collection_id)
            principal_by_collection[collection_id] = cell_principal
        if not cell_principal:
            continue
        pairs.add((cell_principal, collection_id))

    return sorted(pairs)


class MantleSseSearchAccessor:
    """Canonical router-shape search accessor (post lexical-backend retirement).

    Returns the same :class:`SearchResult` shape as the legacy accessor
    so the router's response-mapping code is unaffected.
    """

    def __init__(
        self,
        unified: MantleUnifiedAccessor,
        lightcone: LightConeResolver,
        *,
        store_db,
        embeddings: Optional[Embeddings] = None,
    ) -> None:
        self._unified = unified
        self._lightcone = lightcone
        self._store_db = store_db
        self._embeddings = embeddings or Embeddings()

    def search(self, query) -> "object":
        """Run SSE + MANTLE fused search, return a :class:`SearchResult`."""
        from mantle.search.query_parser import parse_query
        from mantle.search.types import SearchResult

        if self._store_db is None:
            raise ValueError(
                "MantleSseSearchAccessor needs an store_db for hydration"
            )

        parsed = parse_query(query.query_text)
        provided_embedding = getattr(query, "query_embedding", None)
        # Empty text is only a no-op when there's also no raw query vector;
        # an embedding-only query ("embedding activation") proceeds to kNN.
        if parsed.is_empty() and not provided_embedding:
            return SearchResult(
                hits=[],
                total=0,
                parsed_query=parsed,
                corrections=parsed.corrections,
                used_hybrid=False,
            )

        # Light-cone authorization — single ACL path post-2.6.9.
        contexts = resolve_authorized_contexts(
            self._store_db,
            principal_id=query.user_id,
            lightcone=self._lightcone,
        )

        # Respect explicit scope from the router (body.scope → query.scope).
        # When scope is set, restrict to only the requested containers.
        # This is distinct from collection_ids (which carries the full authorized
        # set); scope is set only for user-explicit or principal-restricted searches.
        scope = getattr(query, "scope", None)
        if scope:
            allowed = set(scope)
            contexts = [(principal, col) for principal, col in contexts if col in allowed]

        logger.info(
            "MantleSseSearchAccessor: '%s' authorized to %d contexts",
            query.query_text, len(contexts),
        )

        if not contexts:
            return SearchResult(
                hits=[],
                total=0,
                parsed_query=parsed,
                corrections=parsed.corrections,
                used_hybrid=False,
            )

        # Vector arm: use a caller-provided query vector directly ("embedding
        # activation"), else embed the query text. Embedding errors degrade to
        # SSE-only — the lexical arm carries the search.
        embedding = (
            list(provided_embedding)
            if provided_embedding
            else self._embed_or_none(query.query_text, parsed)
        )

        # Wider top_k from each arm so RRF has rank diversity.
        # ⛔ top_k MUST COVER THE OFFSET, not just the page size. It was `size * 3`, which cannot
        # reach page 2 at all — see the slice below.
        _from = max(0, int(getattr(query, "from_", 0) or 0))
        _want = _from + query.size
        unified_hits = self._unified.search(
            query.query_text,
            contexts,
            _key_request(query),
            query_embedding=embedding,
            top_k=max(_want * 3, 50),
        )

        if not unified_hits:
            return SearchResult(
                hits=[],
                total=0,
                parsed_query=parsed,
                corrections=parsed.corrections,
                used_hybrid=embedding is not None,
            )

        # ⛔ THE PAGINATION OFFSET WAS SILENTLY IGNORED — EVERY PAGE RETURNED PAGE 1.
        # This was `unified_hits[: query.size]`, and `SearchQuery.from_` was read NOWHERE in this
        # accessor. `POST /artifacts/search` with `from=20, size=20` returned the identical 20
        # hits as `from=0`, and the router echoed `"from": 20` back — so results 21+ were
        # unreachable through the API while the response claimed to be page 2.
        hits = self._hydrate(unified_hits[_from:_want])
        return SearchResult(
            hits=hits,
            # ⚠ `total` IS A CANDIDATE COUNT, NOT A MATCH COUNT — and it is capped.
            # `unified_hits` has already been truncated to `top_k` by `unified.search`, so this
            # can never exceed that bound however many artifacts actually match: with size=20 it
            # used to max out at 60 no matter whether 60 or 60,000 documents matched, and a UI
            # rendering "N results" or a page count from it is wrong by orders of magnitude.
            # Left as the candidate count rather than silently changed: producing a true match
            # count needs an uncapped count from the unified layer, which does not exist yet.
            # `total_is_capped` says so explicitly rather than letting the number lie.
            total=len(unified_hits),
            total_is_capped=len(unified_hits) >= max(_want * 3, 50),
            parsed_query=parsed,
            corrections=parsed.corrections,
            used_hybrid=embedding is not None,
        )

    def candidates(
        self,
        query,
        *,
        candidate_budget: int = 200,
        include_vectors: bool = False,
    ) -> dict:
        """Raw retrieval primitive — the single authorization chokepoint.

        Resolves the light-cone for ``query.user_id`` and returns ONLY the
        authorized candidate set (pre-hydration) with per-arm scores. Search
        *flavors* (the open standard one, or an external premium one like
        Beacon) rank within this set — they can never widen it, so MANTLE §1
        holds by construction. See ``.dev/features/search-as-artifact.md``.

        ``include_vectors`` is accepted for the premium re-rank path; candidate
        embeddings are not yet surfaced by the fusion layer, so vectors are
        omitted for now (TODO: thread candidate vectors through MantleQueryEngine
        → UnifiedHit).
        """
        from mantle.search.query_parser import parse_query

        if self._store_db is None:
            raise ValueError("MantleSseSearchAccessor needs an store_db")

        parsed = parse_query(query.query_text)
        provided_embedding = getattr(query, "query_embedding", None)
        if parsed.is_empty() and not provided_embedding:
            # ⛔ WAS `return {"candidates": [], "model_id": None}` — a SILENT EMPTY.
            # Flipped to a raise as the pre-cutover guard (Unit Z §2.2): this shape is
            # indistinguishable from "the retrieval stack is gone and nobody noticed", and it must
            # be loud BEFORE anything below it is removed, not after. `search_router.py:89` wraps
            # this into an HTTP 500 with the detail attached, so the caller is told why.
            #
            # Raising is correct on its own merits here, independent of the cutover: a query that
            # parses to nothing AND carries no embedding is a malformed request, not a search that
            # legitimately matched zero documents. Returning `[]` conflated those two, and only one
            # of them is the caller's bug.
            raise ValueError(
                "empty query: %r parsed to no terms and no embedding was supplied; "
                "there is nothing to retrieve on" % (query.query_text,)
            )

        contexts = resolve_authorized_contexts(
            self._store_db,
            principal_id=query.user_id,
            lightcone=self._lightcone,
        )
        scope = getattr(query, "scope", None)
        if scope:
            allowed = set(scope)
            contexts = [(p, c) for p, c in contexts if c in allowed]

        logger.info(
            "raw query: '%s' authorized to %d contexts (budget=%d)",
            query.query_text, len(contexts), candidate_budget,
        )
        if not contexts:
            # ⚠ DELIBERATELY **NOT** FLIPPED TO A RAISE — and this is a considered deviation from
            # the staged-cutover instruction, recorded here rather than silently taken.
            #
            # The other site above is a malformed request. This one is not: it is the light-cone
            # answering truthfully that this principal is authorized to ZERO contexts. That is the
            # correct answer for every brand-new user with no collections yet. Raising here routes
            # through `search_router.py:89` into an HTTP 500 "Search failed", i.e. it converts a
            # correct authorization result into a server error for the most common first-run state.
            #
            # The guard the cutover actually needs is "did the machinery vanish", and that question
            # is asked ABOVE this line (empty parse) and BELOW it (the unified search call), not
            # here. An authorization-empty is not a machinery-empty, and making them share a
            # failure mode would destroy the distinction the flip exists to create.
            #
            # `model_id` is None because nothing ran — that is accurate, not a placeholder.
            return {"candidates": [], "model_id": None}

        embedding = (
            list(provided_embedding)
            if provided_embedding
            else self._embed_or_none(query.query_text, parsed)
        )
        budget = max(int(candidate_budget), 1)
        unified_hits = self._unified.search(
            query.query_text,
            contexts,
            _key_request(query),
            query_embedding=embedding,
            top_k=budget,
        )

        out = []
        for h in unified_hits:
            rec = {
                "artifact_id": h.artifact_id,
                "collection_id": h.collection_id,
                "principal_id": h.principal_id,
                "sse_score": h.sse_score,
                "vector_score": h.vector_score,
                "rrf_score": h.rrf_score,
                "source": h.source,
            }
            if include_vectors:
                rec["vector"] = None  # TODO: surface candidate embeddings from the engine
            out.append(rec)
        return {"candidates": out, "model_id": getattr(self._unified, "model_id", None)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_or_none(self, query_text: str, parsed) -> Optional[list[float]]:
        """Embed the query (or its semantic-flagged terms) for the vector arm.

        Returns ``None`` if embedding fails — the SSE-only path is still
        valid. Selects ~-flagged semantic terms when present (matching
        the prior accessor's behavior); falls back to the full query.
        """
        try:
            from mantle.search.query_parser import TermModifier
        except Exception:
            TermModifier = None  # type: ignore[assignment]

        text_for_embedding = query_text
        if TermModifier is not None:
            semantic_terms = [
                t.text for t in parsed.terms
                if getattr(t, "modifier", None) == TermModifier.SEMANTIC
            ]
            if semantic_terms:
                text_for_embedding = " ".join(semantic_terms)

        if not text_for_embedding.strip():
            return None
        try:
            results = self._embeddings([text_for_embedding])
            if not results:
                return None
            return results[0]
        except Exception:
            logger.exception("MantleSseSearchAccessor: embedding failed")
            return None

    def _hydrate(self, unified_hits: list[UnifiedHit]) -> list:
        """Read each artifact's metadata from the lattice and produce SearchHits.

        Neither SSE nor MANTLE stores plaintext title / description /
        tags / content — those live in the ``artifacts`` collection.
        Hydration is one the lattice ``get`` per fused hit. Failed lookups
        produce a SearchHit with empty metadata fields rather than
        dropping the hit entirely (the doc may have been removed
        between fusion and hydration).
        """
        from mantle.search.types import SearchHit

        out: list = []
        for hit in unified_hits:
            doc = self._safe_get(self._store_db, hit.artifact_id)
            if doc:
                # Decrypt inline content before it enters a search hit (raw-doc path).
                from mantle.db.doc_boundary import decrypt_artifact_content as _decrypt_artifact_content
                # Non-strict: a bad row degrades to no content, never ciphertext,
                # and never fails the whole result set.
                _decrypt_artifact_content(doc, strict=False)
            ctx = self._parse_context(doc)
            tags_raw = ctx.get("tags") or ctx.get("tags_canonical") or []
            tags = (
                [str(t) for t in tags_raw if str(t).strip()]
                if isinstance(tags_raw, list)
                else []
            )
            out.append(
                SearchHit(
                    doc_id=hit.artifact_id,
                    score=hit.rrf_score,
                    root_id=(
                        (doc or {}).get("root_id")
                        or hit.artifact_id
                    ),
                    version_id=(
                        (doc or {}).get("_key")       # lattice doc-id shape
                        or (doc or {}).get("id")      # lattice doc-id shape
                        or hit.artifact_id
                    ),
                    title=str(ctx.get("title") or ctx.get("name") or ""),
                    description=str(ctx.get("description") or ""),
                    content=str((doc or {}).get("content") or ""),
                    tags=tags,
                    metadata={
                        "sse_score": hit.sse_score,
                        "vector_score": hit.vector_score,
                        "source": hit.source,
                    },
                    collection_id=hit.collection_id,
                    principal_id=hit.principal_id,
                    state=(doc or {}).get("state"),
                    is_head=None,
                    highlights=None,
                )
            )
        return out

    @staticmethod
    def _safe_get(store_db, artifact_id: str):
        try:
            return _raw_artifact(store_db, artifact_id)
        except Exception:  # noqa: BLE001 — store reads raise broadly
            return None

    @staticmethod
    def _parse_context(doc) -> dict:
        if not doc:
            return {}
        ctx = doc.get("context")
        if not ctx:
            return {}
        if isinstance(ctx, dict):
            return ctx
        if isinstance(ctx, str):
            try:
                parsed = json.loads(ctx)
            except (TypeError, ValueError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}


__all__ = ["MantleSseSearchAccessor", "resolve_authorized_contexts"]
