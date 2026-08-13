"""MCP over Streamable HTTP — the lattice, reachable from anywhere that speaks MCP.

`mantle.agience.ai/mcp` exposes the store as Model Context Protocol tools, so any MCP client
(Claude, an IDE, an agent runtime) can use the lattice without a bespoke integration.

THE SURFACE IS ROUND. A client that can only read what something else put there has no use for
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
from pydantic import ValidationError

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


# ── the tool surface ──────────────────────────────────────────────────────────────────────────
#: Every entry names the REST handler it dispatches into. Adding a tool here must never mean adding
#: a query — if a tool needs data no endpoint exposes, add the endpoint first and call it.
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "list_artifacts",
        "title": "List artifacts",
        "description": (
            "List artifacts the caller may act on, newest first. Returns only what the caller's "
            "grants reach — an artifact absent from this list may exist and simply not be visible."
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
            "only with `state: \"draft\"` until something commits it.\n"
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
                        "artifact. Top-level only — not valid with `container_id`."
                    ),
                },
                "container_id": {
                    "type": "string",
                    "description": (
                        "Artifact id of a collection to file this inside. Omit for a top-level "
                        "artifact; see the draft note above before supplying one."
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
            "response and across no two; `semantic` makes it a cosine; `recency` makes it null. "
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
        # The arguments ARE the REST body. `CreateArtifactRequest` is what validates them, one
        # frame down, exactly as it does for a POST — so there is no MCP-side shaping to drift
        # from the API's, and the self-grant, the index enqueue, the event and the state default
        # are the ones `workspace_service` applies to every write.
        return await artifacts.create_artifact(
            request=request, body=dict(args), auth=auth, store_db=store_db,
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
    """Handle one JSON-RPC message. Returns None for notifications (nothing may be sent back)."""
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
        params = msg.get("params") or {}
        name = params.get("name")
        if name not in _TOOLS_BY_NAME:
            return _error(req_id, INVALID_PARAMS, f"unknown tool {name!r}")
        try:
            data = await _call_tool(name, params.get("arguments") or {},
                                    auth=auth, store_db=store_db, request=request)
        except KeyError as e:                       # a required argument was absent
            return _error(req_id, INVALID_PARAMS, f"missing argument: {e}")
        except ValidationError as e:                # an argument the request model refuses
            return _error(req_id, INVALID_PARAMS, f"invalid arguments for {name}: {e}")
        except HTTPException as e:
            # The API's own refusal, carried WORD FOR WORD. Those messages were written to name
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
        # ⭐ THE OBSERVATION IS RECORDED HERE BECAUSE HERE IS THE ONLY PLACE ALL SEVEN TOOLS MEET.
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
        _observe(name, params.get("arguments") or {}, data, auth=auth, store_db=store_db)
        return _result(req_id, {
            "content": [{"type": "text", "text": _as_text(data)}],
            "structuredContent": data if isinstance(data, dict) else {"result": data},
        })

    return _error(req_id, METHOD_NOT_FOUND, f"unknown method {method!r}")


#: The argument each tool carries its question in. Only `recall` has one — the others are
#: addressed by id — and an absent entry means "this tool asked no question", not "look harder".
_QUERY_ARG = {"recall": "query_text"}


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
        # WHICH MACHINE looked. `principal_id` above is whose authority it looked with; for an
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
)
@mcp_router.get("/", include_in_schema=False)
async def mcp_get() -> Response:
    """No server-initiated stream — see the module docstring.

    405 with `Allow` is the spec's own answer for a server that does not offer the GET stream. It
    tells the client to stop waiting, which an empty 200 stream never would.
    """
    return Response(status_code=405, headers={"Allow": "POST", "MCP-Protocol-Version": PROTOCOL_VERSION})
