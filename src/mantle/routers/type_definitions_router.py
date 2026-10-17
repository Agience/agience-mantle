"""`PUT /type-definitions/{server}` — Crystal publishes a server's type definitions.

Crystal holds the type registry; Mantle reads one part of it. A type's `offer` says which fields
give its artifacts their title, description and tags, and the indexer reads it from here
(`services/type_offers.py`). Not under `/system`: that namespace is the platform admin's, and
this route is a service's.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from mantle.api.errors import ERROR_DESCRIPTIONS
from mantle.api.unknown_fields import CountsUnknownFields
from mantle.db.store import Database
from mantle.services import type_offers
from mantle.services.acting_principal import SystemPrincipalUnavailable
from mantle.services.dependencies import AuthContext, get_auth, get_store_db, offload_sync

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Types"])

#: The services that may publish type definitions. Crystal holds the type registry; nothing else
#: decides how every artifact of a type is indexed.
TYPE_PUBLISHERS = {"crystal"}


class TypesPublishRequest(CountsUnknownFields):
    """Every type definition one server owns, as its `type.json` files hold them."""
    types: List[Dict[str, Any]] = Field(default_factory=list)


class TypesPublishResponse(BaseModel):
    """What the publish changed. `offer_changed` names the types whose artifacts are reindexed."""
    server: str
    stored: int
    unchanged: int
    archived: int
    refused: Dict[str, List[str]] = Field(default_factory=dict)
    offer_changed: List[str] = Field(default_factory=list)


def _publish(store_db: Database, server: str, definitions: List[dict]) -> dict:
    from mantle.services.system_identity import system_acting_context
    with system_acting_context(scope="platform.types"):
        return type_offers.publish(store_db, server, definitions)


def _reindex(content_types: List[str]) -> None:
    from mantle.search.init_search import reindex_content_types
    try:
        reindex_content_types(content_types)
    except Exception:
        logger.warning("reindex after a type offer change failed for %s", content_types,
                       exc_info=True)


@router.put(
    "/type-definitions/{server}",
    responses={c: {"description": ERROR_DESCRIPTIONS[c]} for c in (401, 403, 500, 503)},
    response_model=TypesPublishResponse,
    summary="Publish a server's type definitions",
    description=(
        "Store every type definition `server` owns as a system-owned type artifact, replacing "
        "what it published before; a type it no longer sends is archived. A type's `offer` "
        "says which fields give its artifacts their title, description and tags, and the "
        "artifacts of a type whose offer changed are indexed again. Crystal only."
    ),
)
async def publish_type_definitions(
    server: str,
    body: TypesPublishRequest,
    background: BackgroundTasks,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> TypesPublishResponse:
    if auth.principal_type != "service" or auth.authority not in TYPE_PUBLISHERS:
        raise HTTPException(status_code=403, detail="Only the type registry publishes types")
    try:
        result = await offload_sync(_publish, store_db, server, body.types)
    except SystemPrincipalUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.error("type publish failed for %s", server, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to publish the types")
    if result["offer_changed"]:
        background.add_task(_reindex, result["offer_changed"])
    return TypesPublishResponse(server=server, **result)
