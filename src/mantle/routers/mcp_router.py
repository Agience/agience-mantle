"""MCP over Streamable HTTP — the lattice, reachable from anywhere that speaks MCP.

`mantle.agience.ai/mcp` exposes the store as Model Context Protocol tools, so any MCP client
(Claude, an IDE, an agent runtime) can use the lattice without a bespoke integration.

The surface is round: a client that can only read what something else put there has no use for
a store, so the tools cover the whole loop: `create_artifact`, `update_artifact` and
`delete_artifact` write, `recall` finds. Neither is a second implementation — each dispatches
into the same REST handler `POST`/`PATCH`/`DELETE /artifacts` use, so the self-grant, the
indexing, the field filters, the coverage ordering and every 400 are the API's own. A
collection is an artifact like any other (`content_type` `application/vnd.agience.collection+json`),
so these five tools are also the whole CRUD+recall surface for collections — there is no
second set of collection-shaped tools to keep in step with these.

TRANSPORT. Streamable HTTP (the transport that replaced HTTP+SSE): a single endpoint where the client
POSTs JSON-RPC 2.0 and MAY open a GET stream for server-initiated messages. This server is
request/response only — it originates nothing — so `GET /mcp` answers 405 with an `Allow` header,
which the spec provides for precisely this case. Answering 200 with a stream that never emits is
worse than refusing: the client waits on a channel that will never carry anything.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from mantle.db.store import Database
from mantle.services.dependencies import AuthContext, get_auth, get_store_db
from mantle.routers import artifacts_router as artifacts

logger = logging.getLogger(__name__)

mcp_router = APIRouter(prefix="/mcp", tags=["MCP"])

#: The spec revision this server implements. Echoed on `initialize`; a client asking for a version
#: we do not speak is answered with ours, and it decides whether to proceed.
PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "agience-mantle", "title": "Agience Mantle", "version": "1"}

# ── JSON-RPC 2.0 ──────────────────────────────────────────────────────────────────────────────
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _result(req_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


class JsonRpcResponse(BaseModel):
    """The body of a `200` from this endpoint's scope, applied 2026-08-26.

    This operation published `"schema": {}`: a generated client received a JSON-RPC envelope
    and had no type for it, on the one transport where the envelope carries the ERROR as well as
    the result.

    `result` and `error` are mutually exclusive and exactly one is present — that is JSON-RPC,
    not a convention of this server. **A caller must branch on which key it got, not on the HTTP
    status**: a tool that fails answers `200` with `error` set, because the transport succeeded
    even though the call did not.

    Declared via `responses=`, never `response_model=`, for the reason the `/artifacts` audit
    recorded: `response_model` FILTERS, and `result` is deliberately open — every tool returns its
    own shape, so a model that narrowed it would silently drop the payload it exists to carry.
    """
    jsonrpc: str = Field("2.0", description="Always `2.0`.")
    id: Optional[Any] = Field(
        None, description="Echoes the request's `id`. `null` when the request could not be parsed "
                          "far enough to have one.")
    result: Optional[Any] = Field(
        None, description="Present on success. Its shape is the TOOL's — deliberately open.")
    error: Optional[Dict[str, Any]] = Field(
        None, description="Present on failure: `{code, message}`. ⚠ Its presence, not the HTTP "
                          "status, is how a caller learns the call failed.")


# ── the tool surface ──────────────────────────────────────────────────────────────────────────
#: Every entry names the REST handler it dispatches into. Adding a tool here must never mean adding
#: a query — if a tool needs data no endpoint exposes, add the endpoint first and call it.
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "list_artifacts",
        "title": "List artifacts",
        "description": (
            "List artifacts the caller may act on, in id order — NOT newest first, and not by "
            "relevance to anything. This tool matches on nothing; use `recall` to find something. "
            "Returns only what the caller's grants reach, so an artifact absent from this list "
            "may exist and simply not be visible."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content_type": {"type": "string",
                                 "description": "Exact MIME content_type filter. Omit for all."},
                "action": {"type": "string", "default": "read",
                           "description": "CRUDEASIO action to filter by (read, create, update, …)."},
                "limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 1000,
                          "description": "Page size."},
                "offset": {"type": "integer", "default": 0, "minimum": 0,
                           "description": "How many authorized artifacts to skip."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_artifact",
        "title": "Get an artifact",
        "description": "Read one artifact by id, including its context and provenance.",
        "inputSchema": {
            "type": "object",
            "properties": {"artifact_id": {"type": "string"}},
            "required": ["artifact_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_children",
        "title": "List collection members",
        "description": "List the artifacts contained by a collection artifact.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "content_type": {"type": "string", "description": "Exact MIME filter."},
                "limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 1000,
                          "description": "Page size."},
                "offset": {"type": "integer", "default": 0, "minimum": 0,
                           "description": "How many children to skip."},
            },
            "required": ["artifact_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_artifact",
        "title": "Create an artifact",
        "description": (
            "Store something in the lattice — a conversation, a note, a document — and get the "
            "stored artifact back, including the `id` to read it by. The creator receives an "
            "owner grant on it, so what you write here you can read and recall afterwards.\n"
            "Omit `container_id` for a top-level artifact: it is COMMITTED and immediately "
            "recallable. Supply `container_id` to file the artifact inside that collection "
            "instead — the caller needs create permission on it, and a member of a collection "
            "starts as a DRAFT, which is indexed in a separate segment and is found by `recall` "
            "only with `state: \"draft\"` until something commits it — UNLESS you also pass "
            "`identity`, which writes the member COMMITTED and overwrites it in place, because "
            "an identity names something that lives outside the store rather than a draft you "
            "are working on here.\n"
            "PASS `identity` WHEN THIS THING MAY BE STORED AGAIN — a file you will re-capture, a "
            "session you will extend, a note you will revise. It makes the write idempotent, so "
            "the second call updates the first artifact rather than leaving two copies with "
            "nothing marking which is current."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "The body to store. A whole conversation transcript is an ordinary "
                        "value here — it is encrypted at rest and its words are what the "
                        "`query_text` terms of `recall` match against."
                    ),
                },
                "name": {"type": "string", "description": "Short human-readable label."},
                "description": {
                    "type": "string",
                    "description": "One-line summary. Recallable through the `description:` filter.",
                },
                "content_type": {
                    "type": "string",
                    "description": (
                        "MIME label for the content, e.g. `text/markdown`. An opaque label — "
                        "Mantle resolves it to no schema — and the value the `content_type:` "
                        "(alias `type:`) recall filter selects on.\n"
                        "SUPPLY IT FOR ANYTHING THAT IS NOT A CONTAINER: omitted, this defaults to "
                        "`application/vnd.agience.collection+json`, so a stored conversation or "
                        "note comes back labelled as a COLLECTION — a thing other artifacts are "
                        "filed inside — and every `type:` filter that would have found it misses."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Metadata beside the content, as a JSON OBJECT ENCODED IN A STRING — a "
                        "string, not an object. Its `title` and `tags` entries are what the "
                        "`title:` and `tags:` recall filters read."
                    ),
                },
                "identity": {
                    "type": "string",
                    "description": (
                        "A stable name for the THING this artifact is of — `file:/repo/README.md`, "
                        "`session:7c7bcb7b`, `decision:retry-policy`. Supplying it makes this call "
                        "IDEMPOTENT: the id is derived from the name, so storing the same thing "
                        "again updates that one artifact instead of creating a second copy, and "
                        "you never have to remember an id to update it later.\n"
                        "SUPPLY IT WHENEVER THE SAME THING WILL BE STORED MORE THAN ONCE. Without "
                        "it every write mints a new artifact, and a store holding several copies "
                        "of one document answers `recall` with whichever copy scored best — which "
                        "may be any of them, including a stale one.\n"
                        "Scoped to you: the same name from a different principal is a different "
                        "artifact.\n"
                        "VALID WITH `container_id`, AND WORTH COMBINING. Filing your artifacts in "
                        "one collection makes them share an origin root, which is the unit the "
                        "encrypted index is keyed on — so recall reads one owner instead of one "
                        "per artifact. Top-level artifacts are each their own root, which is "
                        "correct for a collection itself and expensive for a hundred notes."
                    ),
                },
                "container_id": {
                    "type": "string",
                    "description": (
                        "Artifact id of a collection to file this inside. Omit for a top-level "
                        "artifact; see the draft note above before supplying one."
                    ),
                },
                "vector": {
                    "type": "array", "items": {"type": "number"},
                    "description": (
                        "The semantic arm's ingress. Mantle NEVER EMBEDS, so the only way a "
                        "vector reaches the vector arm is a writer handing one over on the "
                        "write that produced the content it describes. Shape is validated "
                        "(finite, bounded dimension, non-zero norm); quality never is — that "
                        "would be a claim about someone else's model.\n"
                        "Requires `space_id`. Omit both and the artifact is lexical-only, "
                        "which is what every write did before."
                    ),
                },
                "space_id": {
                    "type": "string",
                    "description": (
                        "Names the embedding space `vector` lives in — required whenever it is "
                        "present, and it must equal the seeded AnchorSet's `model_id`. One node "
                        "serves exactly one space; a vector in any other is refused with a 400 "
                        "naming both."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "update_artifact",
        "title": "Update an artifact",
        "description": (
            "Partial update of an artifact or collection by id — only the fields you supply "
            "change, everything else is left as it was. Re-supply `vector` + `space_id` when "
            "the content the vector describes has changed: the semantic arm reindexes with "
            "the rest of the write, and omitting it leaves the old vector describing new "
            "content.\n"
            "Setting `state` to `\"committed\"` is how a draft (an artifact filed inside a "
            "collection via `create_artifact`'s `container_id`) becomes findable by `recall` "
            "without the `state: \"draft\"` argument."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "name": {"type": "string", "description": "Short human-readable label."},
                "description": {
                    "type": "string",
                    "description": "One-line summary. Recallable through the `description:` filter.",
                },
                "content": {
                    "type": "string",
                    "description": "Replacement body. Re-indexed the same way a create is.",
                },
                "content_type": {
                    "type": "string",
                    "description": "MIME label. See `create_artifact` for what an omission vs. an explicit value means.",
                },
                "state": {
                    "type": "string",
                    "description": (
                        "New lifecycle state, e.g. `\"committed\"` to promote a draft. Not the "
                        "`state` argument `recall` takes — this one is written, that one is read."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Metadata beside the content, as a JSON OBJECT ENCODED IN A STRING — "
                        "same shape as `create_artifact`'s `context`. Immutable fields the "
                        "schema protects are silently dropped from this update rather than "
                        "refused."
                    ),
                },
                "vector": {
                    "type": "array", "items": {"type": "number"},
                    "description": "Replacement embedding. Requires `space_id` alongside it.",
                },
                "space_id": {"type": "string"},
            },
            "required": ["artifact_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "delete_artifact",
        "title": "Delete an artifact",
        "description": (
            "Delete an artifact or collection by id. There is no undo.\n"
            "If this is a collection with members, they are DETACHED by default — evicted from "
            "it, not destroyed — so nothing under it is lost unless you say so. Set `cascade: "
            "true` to actually delete them too: a member reachable only through this collection "
            "is destroyed outright (its search index and content dropped with it); one still "
            "reachable through another collection is evicted from this one only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "cascade": {
                    "type": "boolean", "default": False,
                    "description": (
                        "true to also delete a collection's members instead of just detaching "
                        "them. Ignored when the target has no members."
                    ),
                },
            },
            "required": ["artifact_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "recall",
        "title": "Recall artifacts",
        "description": (
            "Find artifacts by what they say, ordered and hydrated — the way to get something "
            "back once it is stored. Not `list_artifacts`, which pages the visible set in id "
            "order and matches on nothing.\n"
            "THE TEXT ARGUMENT IS `query_text`. There is no `query` argument, and a call that "
            "sends one has stated no query at all.\n"
            "Each hit carries `id`, `root_id`, `version_id`, `collection_id`, `title`, "
            "`description`, `content` (the densest spans of the artifact, not a prefix — "
            "length is not fixed and depends on how much of the document actually carries "
            "signal), `highlights` (those same dense spans individually, under the key "
            "`content`), `tags` and a `score`; the "
            "response also carries `total`, `applied_filters` (what actually narrowed it) and "
            "`ordering`. READ `ordering` TO KNOW WHAT `score` MEANS: `coverage` — the usual "
            "answer — makes it the integer count of distinct query stems that hit carries, "
            "which is a count and not a relevance measure, comparable between the hits of one "
            "response and across no two; `semantic` makes it a cosine; `reach` makes it how "
            "far that hit reaches toward what the query is ABOUT, in spreads above what a "
            "hit of its size would reach by nothing (bigger is better; negative is a reading, "
            "not an absence) — and `reach` is the one ordering that CUTS, so `total` counts "
            "what survived the cut rather than every match; `recency` makes it null. "
            "Only artifacts the caller's grants reach are searched, so an empty result means "
            "'nothing you may see matched', never 'nothing exists'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": (
                        "What to look for. The terms NARROW: an artifact must carry them to come "
                        "back. `+term` requires one, `!term` excludes one, `=\"a phrase\"` takes "
                        "it whole.\n"
                        "`field:value` narrows before retrieval, and ONLY FOR THESE TEN FIELDS: "
                        "`id`, `root_id`, `collection_id`, `content_type` (alias `type`), "
                        "`owner_id`, `title`, `description`, `tags` (alias `tag`), `created_at`, "
                        "`updated_at`. ANY OTHER WORD BEFORE A COLON IS AN ORDINARY SEARCH TERM "
                        "— `titel:foo`, `https://example.com` and `16:9` all search for that "
                        "text — so a misspelt field name narrows nothing and comes back empty "
                        "rather than as an error. Check that list first when a `field:value` "
                        "query finds nothing.\n"
                        "Operators: `field:value` (case-insensitive, and `a,b` is any-of), "
                        "`field:=\"Exact Value\"` (case-sensitive, taken whole), "
                        "`!field:value`, and `field:>value` / `field:<value` on `created_at` and "
                        "`updated_at` only. Filters conjoin. A known field carrying an operator "
                        "it cannot take is refused by name, as is `state:` (send the `state` "
                        "argument) and `content:` (encrypted at rest — search for the words "
                        "instead). A query of nothing but filters is refused too: a filter "
                        "narrows a search, it does not constitute one."
                    ),
                },
                "candidates": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Return the RAW narrowed candidate set instead of an ordered, cut page. "
                        "Same authorization, same light cone, same narrowing — what is skipped is "
                        "the server's ordering and its cut.\n"
                        "USE THIS WHEN YOU HAVE A BETTER FRAME THAN THE SERVER DOES. A caller "
                        "holding the query's embedding can rank the candidates itself and decide "
                        "how many to keep by reading their own features; the server, ranking "
                        "inside one request, must decide that from scores alone. Measured on "
                        "71/dev: the server's cut returned ONE hit for a query whose answer sat "
                        "at rank #2 of the same candidate set.\n"
                        "The default stays false because a caller with no ranker of its own is "
                        "better served by an answer than by a pile."
                    ),
                },
                "state": {
                    "type": "string", "enum": ["committed", "draft", "archived"],
                    "default": "committed",
                    "description": (
                        "Which index segment to search. Each artifact state is its own "
                        "separately keyed tree, so this selects the corpus rather than filtering "
                        "one — an artifact created inside a collection is a draft and is found "
                        "only with `draft`."
                    ),
                },
                "sort": {
                    "type": "string", "enum": ["relevance", "recency"], "default": "relevance",
                    "description": (
                        "What order to ask for. `recency` is most-recently-updated first. "
                        "`relevance` asks for the best this recall can produce and cannot "
                        "promise one — `ordering` on the response reports what happened."
                    ),
                },
                "size": {"type": "integer", "default": 20, "minimum": 1,
                         "description": "Page size."},
                "from": {"type": "integer", "default": 0, "minimum": 0,
                         "description": "How many matching artifacts to skip."},
                "vector": {
                    "type": "array", "items": {"type": "number"},
                    "description": (
                        "The query's vector — the reader's half of the semantic arm, and the "
                        "exact counterpart of `vector` on `create_artifact`: a writer supplies "
                        "the vector of what it stores, a reader supplies the vector of what it "
                        "is looking for. Mantle embeds neither.\n"
                        "It ACCOMPANIES `query_text` rather than replacing it — the lexical arm "
                        "reads the text, the semantic arm reads this, and each contributes what "
                        "it found. Sent alone it is a kNN recall.\n"
                        "Requires `space_id`. Answered with `ordering: \"semantic\"` and a "
                        "cosine per hit, or a 400 naming both spaces when this node ranks in a "
                        "different one."
                    ),
                },
                "space_id": {
                    "type": "string",
                    "description": (
                        "Names the embedding space `vector` lives in. Required alongside it, "
                        "because two vectors are comparable only within a named space and "
                        "Mantle cannot infer one from the numbers. Must equal the seeded "
                        "AnchorSet's `model_id`."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
]

_TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}


async def _call_tool(name: str, args: Dict[str, Any], *, auth: AuthContext,
                     store_db: Database, request: Request) -> Any:
    """Dispatch a tool call into the REST handler that owns it.

    Note every branch passes `auth` straight through. That is the whole point: the light-cone sees
    the same principal it would have seen over HTTP, so a caller cannot reach anything through MCP
    that they could not reach through the API.

    This router issues no store call of its own, so it needs no `offload_sync` of its own either:
    dispatching into the REST handler inherits that handler's thread offload along with its
    authorization. A tool that read the store directly here would lose both at once.
    """
    if name == "list_artifacts":
        return await artifacts.list_visible(
            content_type=args.get("content_type"),
            action=args.get("action", "read"),
            limit=int(args.get("limit", 100)),
            offset=int(args.get("offset", 0)),
            auth=auth,
            store_db=store_db,
        )
    if name == "get_artifact":
        return await artifacts.read_artifact(
            artifact_id=args["artifact_id"], auth=auth, store_db=store_db,
        )
    if name == "get_children":
        return await artifacts.list_children(
            artifact_id=args["artifact_id"], request=request,
            content_type=args.get("content_type"), workspace_id=None,
            limit=int(args.get("limit", 100)), offset=int(args.get("offset", 0)),
            auth=auth, store_db=store_db,
        )
    if name == "create_artifact":
        # The arguments are the REST body. `CreateArtifactRequest` is what validates them, one
        # frame down, exactly as it does for a POST — so there is no MCP-side shaping to drift
        # from the API's, and the self-grant, the index enqueue, the event and the state default
        # are the ones `workspace_service` applies to every write.
        #
        # This call bypasses FastAPI entirely and hands the handler its argument directly, so it
        # must track the handler's signature by hand: passing a raw dict where the handler expects
        # `CreateArtifactRequest` (matching `update_artifact` and `recall` two calls below) raises
        # a `TypeError` at call time rather than a type error at import time — this seam has
        # broken that way twice.
        #: `response=` is a throwaway, and it must be passed. The handler lowers its status to
        #: 200 when the write turned out not to create anything/C7) by writing to the
        #: `Response` FastAPI injects. MCP carries no HTTP status, so the object is discarded here
        #: — but the parameter is required, and omitting it is a `TypeError` at call time, not a
        #: type error at import time.
        return await artifacts.create_artifact(
            request=request, response=Response(),
            body=artifacts.CreateArtifactRequest(**dict(args)),
            auth=auth, store_db=store_db,
        )
    if name == "update_artifact":
        # Same split as the REST body: `artifact_id` is a path parameter over HTTP, so it comes
        # out of `args` before what remains is validated as the request model, exactly as
        # FastAPI's own path/body split does for `PATCH /artifacts/{artifact_id}`.
        rest = dict(args)
        artifact_id = rest.pop("artifact_id")
        return await artifacts.update_artifact(
            artifact_id=artifact_id,
            body=artifacts.UpdateArtifactRequest(**rest),
            auth=auth, store_db=store_db,
        )
    if name == "delete_artifact":
        return await artifacts.delete_artifact(
            artifact_id=args["artifact_id"], cascade=bool(args.get("cascade", False)),
            auth=auth, store_db=store_db,
        )
    if name == "recall":
        # Same request model FastAPI builds from the POST body, so `query_text` vs `query`,
        # the `from` alias and the unknown-fields-are-ignored rule all resolve identically —
        # and the handler beyond it is the one that builds the accessor, which is where the
        # field filters, `applied_filters`, the coverage `ordering` and the 503 live.
        return await artifacts.recall_artifacts(
            body=artifacts.ArtifactRecallRequest(**args), auth=auth, store_db=store_db,
        )
    raise KeyError(name)


# ── the endpoint ──────────────────────────────────────────────────────────────────────────────
async def _dispatch(msg: Dict[str, Any], *, auth: AuthContext, store_db: Database,
                    request: Request) -> Optional[Dict[str, Any]]:
    """Handle one JSON-RPC message. Returns None for notifications (nothing may be sent back).

    A batch element need not be an object: `[1, 2]` is well-formed JSON and a well-formed array,
    so a non-object element can reach here, and `msg.get` on it would raise `AttributeError` — a
    500 for a request the server understood well enough to reject.
    """
    if not isinstance(msg, dict):
        return _error(None, INVALID_REQUEST, "each message must be an object")
    if msg.get("jsonrpc") != "2.0":
        return _error(msg.get("id"), INVALID_REQUEST, "jsonrpc must be '2.0'")

    method = msg.get("method")
    req_id = msg.get("id")
    is_notification = "id" not in msg

    if is_notification:
        # `notifications/initialized` and friends are acknowledgements. Silence is the protocol.
        return None

    if method == "initialize":
        return _result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": (
                "Agience Mantle is an encrypted artifact lattice. Every result is filtered by the "
                "caller's grants, so an empty list means 'nothing you may see', not 'nothing exists'. "
                "Write with `create_artifact`, change it with `update_artifact`, remove it with "
                "`delete_artifact`, and find it again with `recall` — `list_artifacts` pages "
                "what is visible and matches on nothing. A collection is an artifact like any "
                "other, so the same five tools cover it too."
            ),
        })

    if method == "ping":
        return _result(req_id, {})

    if method == "tools/list":
        return _result(req_id, {"tools": TOOLS})

    if method == "tools/call":
        # `params` and `arguments` are caller-supplied JSON and need not be objects: an unchecked
        # `params.get(...)` on a string raises `AttributeError` out of the handler and becomes a
        # 500, and a non-object `arguments` reaching `_call_tool` surfaces "AttributeError: 'str'
        # object has no attribute 'get'" to the model as though the tool had failed. Reproduced
        # 2026-08-17 for four shapes; a 500 on a malformed request tells a client the server broke,
        # when the truth is the message did.
        params = msg.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _error(req_id, INVALID_PARAMS, "params must be an object")
        name = params.get("name")
        if name not in _TOOLS_BY_NAME:
            return _error(req_id, INVALID_PARAMS, f"unknown tool {name!r}")
        args = params.get("arguments")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return _error(req_id, INVALID_PARAMS,
                          f"arguments must be an object for {name}")
        violation = _schema_violation(name, args)
        if violation is not None:
            return _error(req_id, INVALID_PARAMS, violation)
        try:
            data = await _call_tool(name, args,
                                    auth=auth, store_db=store_db, request=request)
        except KeyError as e:                       # a required argument was absent
            return _error(req_id, INVALID_PARAMS, f"missing argument: {e}")
        except ValidationError as e:                # an argument the request model refuses
            return _error(req_id, INVALID_PARAMS, f"invalid arguments for {name}: {e}")
        except HTTPException as e:
            # The API's own refusal, carried word for word. Those messages were written to name
            # a remedy — which field to send instead of `state:`, which operator a field cannot
            # take, that a node's AnchorSet is unprovisioned — and an MCP client puts a tool
            # error in front of the model, which is the one reader that can act on it. Restating
            # them here would spend that turn saying less. The status rides along because 400
            # ("fix the request") and 503 ("this node is not configured for that") ask the caller
            # for different things.
            logger.info("mcp tool %s refused: %s %s", name, e.status_code, e.detail)
            return _result(req_id, {
                "content": [{"type": "text", "text": f"{e.status_code}: {e.detail}"}],
                "isError": True,
            })
        except Exception as e:                      # noqa: BLE001 - surfaced as a TOOL error, below
            logger.info("mcp tool %s failed: %s", name, e)
            return _result(req_id, {
                "content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}],
                "isError": True,
            })
        data = _jsonable(data)
        # The observation is recorded here, the one place all seven tools meet.
        # Every tool reaches its answer through `_call_tool` above and leaves through this line, so
        # one emit is complete coverage of the MCP surface by construction — the same argument the
        # change feed makes for `db.doc_boundary.emit_artifact_change`. Per-tool emits would be
        # seven places to forget, and `tests/test_observation_events.py` asserts this one covers
        # all seven rather than trusting that it does.
        #
        # AFTER `_jsonable`, not before: the observation should describe what the caller was
        # actually handed, and that is this shape. It is also why the refusal paths above do not
        # reach here — a refused tool call produced no answer to describe, and the refusal is
        # already the audited event on the access path.
        _observe(name, args, data, auth=auth, store_db=store_db)
        return _result(req_id, {
            "content": [{"type": "text", "text": _as_text(data)}],
            "structuredContent": data if isinstance(data, dict) else {"result": data},
        })

    return _error(req_id, METHOD_NOT_FOUND, f"unknown method {method!r}")


#: The argument each tool carries its question in. Only `recall` has one — the others are
#: addressed by id — and an absent entry means "this tool asked no question", not "look harder".
_QUERY_ARG = {"recall": "query_text"}


def _schema_violation(name: str, args: Dict[str, Any]) -> Optional[str]:
    """Why *args* does not satisfy the tool's declared `inputSchema`, or ``None``.

    `additionalProperties: false` is enforced here rather than merely declared. Every tool schema
    declares it, and unread it lets a misspelled argument be dropped so the tool runs without it.
    The resulting failure is never about the typo: `recall` with `query` instead of
    `query_text` reached the handler with no query at all and came back "query_text or vector is
    required" — a true statement about a request the caller thought they had made.

    An agent is the only reader of these messages and the only thing that can act on one. Naming
    the key it got wrong, beside the keys that exist, is the difference between one corrected
    call and a retry loop.

    Deliberately NOT a JSON Schema implementation. It reads the two assertions the schemas
    actually make — `required` and `additionalProperties: false` — because those are the two an
    agent gets wrong. Types are left to the request models one frame down, which already refuse
    them with a message written for a human and reused verbatim; duplicating that here would be
    a second validator to drift from the first.
    """
    schema = (_TOOLS_BY_NAME.get(name) or {}).get("inputSchema") or {}
    declared = schema.get("properties") or {}
    missing = [k for k in (schema.get("required") or []) if k not in args]
    if missing:
        return "missing required argument(s) %s for %s" % (", ".join(sorted(missing)), name)
    if schema.get("additionalProperties") is False:
        unknown = sorted(k for k in args if k not in declared)
        if unknown:
            return (
                "unknown argument(s) %s for %s; it accepts %s"
                % (", ".join(unknown), name, ", ".join(sorted(declared)) or "no arguments")
            )
    return None


def _results_of(data: Any) -> Any:
    """The artifacts a tool answered with, out of that tool's own response shape.

    Four shapes, because the seven tools genuinely have four: a recall envelope (`hits`), a list
    envelope (`result`), a bare list, and a single artifact. Anything else contributes no
    descriptors rather than guessing — an observation that names the wrong artifacts is worse than
    one that names none.
    """
    if isinstance(data, dict):
        for key in ("hits", "result", "artifacts"):
            inner = data.get(key)
            if isinstance(inner, list):
                return inner
        if data.get("id"):
            return [data]
        return ()
    if isinstance(data, list):
        return data
    return ()


def _observe(name: str, args: Dict[str, Any], data: Any, *, auth: AuthContext,
             store_db: Database) -> None:
    """Record one tool call as an observation. Best-effort by construction — see
    `events.observation.record_observation`, which swallows everything."""
    from mantle.events import observation

    query_arg = _QUERY_ARG.get(name)
    observation.record_observation(
        store_db=store_db,
        # A key acts on its own behalf, not its issuer's — the same subject choice
        # `services/dependencies` makes when it fills the audit context, and for the same reason:
        # attributing a detached credential's reads to the person who minted it would make the log
        # say someone did something they did not do.
        principal_id=getattr(auth, "user_id", None) or getattr(auth, "principal_id", None),
        tool=name,
        query_text=args.get(query_arg) if query_arg else None,
        hits=_results_of(data),
        # Which machine looked. `principal_id` above is whose authority it looked with; for an
        # `mcp_client` those are different, and this column is the whole of what makes "which
        # agent" answerable.
        actor=getattr(auth, "actor", None),
        via=getattr(auth, "principal_type", None),
    )


def _jsonable(data: Any) -> Any:
    """Serialize a handler's return value the way its REST route would.

    A handler may hand back a pydantic model rather than a dict — `recall_artifacts` returns
    `ArtifactRecallResponse`, whose own serializer is what spells `from_` as `from` — and on the
    REST path FastAPI is what turns that into a body. Running FastAPI's own encoder here is why
    a tool result IS the JSON a client would have received over HTTP, field for field, instead
    of a second shape maintained beside it.
    """
    from fastapi.encoders import jsonable_encoder
    return jsonable_encoder(data)


def _as_text(data: Any) -> str:
    import json
    return json.dumps(data, indent=2, default=str)


@mcp_router.post(
    "",
    summary="MCP JSON-RPC endpoint",
    description=(
        "Streamable HTTP transport. Accepts one JSON-RPC 2.0 message or a batch; a batch of "
        "nothing but notifications answers 202 with no body. Every tool dispatches into the "
        "REST handler that owns it, with the caller's own principal, so MCP reaches nothing "
        "the API would not."
    ),
    #: The two answers besides `200`, declared 2026-08-26's scope, applied
    #: where the ratchet said it had not been).
    #:
    #: And the error codes stop there, deliberately. This is JSON-RPC: a tool that fails
    #: answers **200** carrying an error object, not an HTTP error — so the usual sweep of "what
    #: can the handler raise" would document codes this transport never uses and hide the ones it
    #: does. The two below are the transport's own, raised before any message is dispatched.
    responses={
        200: {"model": JsonRpcResponse,
              "description":
              "A JSON-RPC response envelope. ⚠ A FAILING TOOL ARRIVES HERE, not in the 4xx: "
              "`error` is set and `result` is absent, because the transport succeeded even though "
              "the call did not."},
        202: {"description":
              "Accepted. A batch of nothing but notifications has no response to return, so the "
              "body is empty by construction rather than by omission."},
        400: {"description":
              "The envelope itself is unusable — unparseable JSON, or an empty batch. The body is "
              "a JSON-RPC error object. ⚠ A failing TOOL does not arrive here: that answers `200` "
              "with an error member, which is what JSON-RPC specifies."},
    },
)
@mcp_router.post("/", include_in_schema=False)
async def mcp_post(request: Request,
                   auth: AuthContext = Depends(get_auth),
                   store_db: Database = Depends(get_store_db)) -> Response:
    """The Streamable HTTP entry point. Accepts one JSON-RPC message or a batch."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_error(None, PARSE_ERROR, "invalid JSON"), status_code=400)

    headers = {"MCP-Protocol-Version": PROTOCOL_VERSION}

    if isinstance(body, list):
        if not body:
            return JSONResponse(_error(None, INVALID_REQUEST, "empty batch"), status_code=400)
        out = [r for r in
               [await _dispatch(m, auth=auth, store_db=store_db, request=request) for m in body] if r is not None]
        # A batch of nothing but notifications gets 202 and no body — same rule as a single one.
        if not out:
            return Response(status_code=202, headers=headers)
        return JSONResponse(out, headers=headers)

    if not isinstance(body, dict):
        return JSONResponse(_error(None, INVALID_REQUEST, "expected an object or array"),
                            status_code=400)

    resp = await _dispatch(body, auth=auth, store_db=store_db, request=request)
    if resp is None:
        return Response(status_code=202, headers=headers)
    return JSONResponse(resp, headers=headers)


@mcp_router.get(
    "",
    summary="No server-initiated stream",
    description=(
        "405 with `Allow: POST`, which is the spec's own answer for a server that offers no "
        "GET stream. A 200 stream that never emits would leave the client waiting forever."
    ),
    #: It declared a `200` it can never send. FastAPI documents `200` by default, and this
    #: handler has exactly one `return` — `405`. So the spec promised a success this route does
    #: not have, and a generated client had a branch for it and none for the answer it always
    #: gets. `status_code=405` makes the default match the only reality; `responses` then
    #: carries the description a caller can act on.
    status_code=405,
    responses={
        405: {"description":
              "Always. This server originates nothing, so there is no GET stream to join. The "
              "`Allow: POST` header names the verb that works. ⚠ This is the ONLY answer this "
              "operation gives — it is not an error condition, it is the endpoint's whole "
              "contract."},
    },
)
@mcp_router.get("/", include_in_schema=False)
async def mcp_get() -> Response:
    """No server-initiated stream — see the module docstring.

    405 with `Allow` is the spec's own answer for a server that does not offer the GET stream. It
    tells the client to stop waiting, which an empty 200 stream never would.
    """
    return Response(status_code=405, headers={"Allow": "POST", "MCP-Protocol-Version": PROTOCOL_VERSION})
