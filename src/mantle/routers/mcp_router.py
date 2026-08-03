"""MCP over Streamable HTTP — the lattice, reachable from anywhere that speaks MCP.

`mantle.agience.ai/mcp` exposes the store as Model Context Protocol tools, so any MCP client
(Claude, an IDE, an agent runtime) can read the lattice without a bespoke integration.

⚠ THE ONE RULE: MCP IS A SURFACE, NOT A SECOND IMPLEMENTATION. Every tool below dispatches into the
SAME router handler the REST API uses, with the SAME `AuthContext`, so authorization is computed once
— by the light-cone, in `check_access` — no matter which door the request came through. A second
authorization path is how a read surface quietly becomes a bypass: it starts as "just a thin wrapper"
and drifts the first time one side gains a check the other does not. There is nothing to keep in sync
here because there is only one implementation.

TRANSPORT. Streamable HTTP (the transport that replaced HTTP+SSE): a single endpoint where the client
POSTs JSON-RPC 2.0 and MAY open a GET stream for server-initiated messages. This server is
request/response only — it originates nothing — so `GET /mcp` answers 405 with an `Allow` header,
which the spec provides for precisely this case. Answering 200 with a stream that never emits is
worse than refusing: the client waits on a channel that will never carry anything.

⚠ JSON-RPC NOTIFICATIONS HAVE NO `id` AND MUST NOT BE ANSWERED. A notification (`notifications/…`)
gets HTTP 202 with an empty body — returning a JSON-RPC result for one is a protocol violation that
some clients surface as a hang rather than an error, because they are not waiting for a reply.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

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
            },
            "required": ["artifact_id"],
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
    """
    if name == "list_artifacts":
        # ⚠ Only the arguments `list_visible` actually accepts. An earlier version advertised a
        # `limit` the handler has no parameter for — the schema described a REST endpoint that was
        # imagined rather than read, and every call raised TypeError. The tool surface must be
        # derived from the handler signature, never from what a sensible endpoint would look like.
        return await artifacts.list_visible(
            content_type=args.get("content_type"),
            action=args.get("action", "read"),
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
            auth=auth, store_db=store_db,
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
                "caller's grants, so an empty list means 'nothing you may see', not 'nothing exists'."
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
        except Exception as e:                      # noqa: BLE001 - surfaced as a TOOL error, below
            # ⚠ A failed tool is NOT a failed JSON-RPC call. Per the spec it returns a normal result
            # with `isError: true`, so the model can see what went wrong and adapt. Returning a
            # protocol error instead hides the reason from the model and looks like a broken server.
            logger.info("mcp tool %s failed: %s", name, e)
            return _result(req_id, {
                "content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}],
                "isError": True,
            })
        return _result(req_id, {
            "content": [{"type": "text", "text": _as_text(data)}],
            "structuredContent": data if isinstance(data, dict) else {"result": data},
        })

    return _error(req_id, METHOD_NOT_FOUND, f"unknown method {method!r}")


def _as_text(data: Any) -> str:
    import json
    return json.dumps(data, indent=2, default=str)


@mcp_router.post("", include_in_schema=False)
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


@mcp_router.get("", include_in_schema=False)
@mcp_router.get("/", include_in_schema=False)
async def mcp_get() -> Response:
    """No server-initiated stream — see the module docstring.

    405 with `Allow` is the spec's own answer for a server that does not offer the GET stream. It
    tells the client to stop waiting, which an empty 200 stream never would.
    """
    return Response(status_code=405, headers={"Allow": "POST", "MCP-Protocol-Version": PROTOCOL_VERSION})
