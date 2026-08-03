"""Raw search query primitive — the authorization chokepoint for flavored search.

`POST /search/query` resolves the light-cone for the calling user and returns the
**authorized candidate set** (per-arm scores, optionally vectors). It does no
flavored ranking. Search *flavors* — the open standard one, or an external
premium one like Beacon — call this and rank within the returned set; they can
never widen access (MANTLE §1 holds by construction).

This is additive: the legacy `POST /artifacts/search` is unchanged. See
`.dev/features/search-as-artifact.md`.
"""

from typing import List, Optional

from mantle.db.store import Database
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from mantle.services.dependencies import AuthContext, get_store_db, get_auth

search_router = APIRouter(prefix="/search", tags=["Search"])


class RawQueryRequest(BaseModel):
    # No `embedding` field. A caller-supplied query vector is trained-model output —
    # BYOK by another name — and accepting one lit the vector arm with no provider
    # configured. Removed 2026-07-30 under the no-models rule (universal, incl. BYOK).
    query_text: Optional[str] = None
    scope: Optional[List[str]] = None            # restrict to these container IDs
    state: str = "committed"                     # index segment: committed (default) | draft | archived
    candidate_budget: int = 200                  # how many candidates to retrieve for ranking
    include_vectors: bool = False                # reserved for vector-level premium re-rank


@search_router.post("/query")
async def raw_query(
    body: RawQueryRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Return the authorized candidate set for a query. Auth = the calling user;
    candidates are filtered to that user's light-cone inside the accessor."""
    user_id = auth.user_id
    if not user_id and not auth.bearer_grant:
        raise HTTPException(status_code=401, detail="Missing authorization")

    if not (body.query_text and body.query_text.strip()):
        raise HTTPException(status_code=400, detail="query_text is required")

    from mantle.search.types import SearchQuery
    from mantle.search.mantle.wiring import VALID_SEGMENTS, build_sse_search_accessor

    segment = (body.state or "committed").lower()
    if segment not in VALID_SEGMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"state must be one of {', '.join(VALID_SEGMENTS)}",
        )

    query = SearchQuery(
        query_text=body.query_text or "",
        query_embedding=None,
        user_id=user_id or "",
        scope=body.scope,
        use_hybrid=None,
        from_=0,
        size=body.candidate_budget,
        sort="relevance",
        highlight=False,
    )

    accessor = build_sse_search_accessor(store_db, segment=segment)
    if accessor is None:
        raise HTTPException(
            status_code=503,
            detail="Encrypted search is not available (Oracle / S3 / the lattice prerequisite missing)",
        )

    try:
        return accessor.candidates(
            query,
            candidate_budget=body.candidate_budget,
            include_vectors=body.include_vectors,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")
