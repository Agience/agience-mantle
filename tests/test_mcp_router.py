"""`/mcp` — Model Context Protocol over Streamable HTTP.

The load-bearing test here is `test_mcp_dispatches_into_the_same_handler_as_rest`. MCP is a second
DOOR onto the store, and the failure that matters is not a malformed JSON-RPC envelope — it is a
second door that reaches further than the first. Every tool must dispatch into the REST handler
that owns the operation, carrying the same `AuthContext`, so the light-cone decides once. A tool
that ran its own query would pass every protocol test in this file and still be a bypass.

The rest of the file pins the protocol edges that clients actually trip on: notifications must get
no response body, a failed tool is a RESULT with `isError` rather than a JSON-RPC error, and the
absent GET stream must refuse rather than hang.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


def _rpc(method: str, params=None, req_id=1):
    msg = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params is not None:
        msg["params"] = params
    return msg


# ── protocol handshake ────────────────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_initialize_reports_protocol_and_tool_capability(client):
    r = await client.post("/mcp", json=_rpc("initialize"))
    assert r.status_code == 200
    body = r.json()
    assert body["jsonrpc"] == "2.0" and body["id"] == 1
    assert body["result"]["protocolVersion"] == "2025-06-18"
    assert body["result"]["capabilities"]["tools"] == {"listChanged": False}
    assert body["result"]["serverInfo"]["name"] == "agience-mantle"


@pytest.mark.anyio
async def test_tools_list_advertises_schemas(client):
    r = await client.post("/mcp", json=_rpc("tools/list"))
    tools = r.json()["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {"list_artifacts", "get_artifact", "get_children"}
    for t in tools:
        assert t["inputSchema"]["type"] == "object"
        # additionalProperties:false is what makes a client's argument typo an error rather than a
        # silently-ignored field that produces a confidently wrong result.
        assert t["inputSchema"]["additionalProperties"] is False


@pytest.mark.anyio
async def test_ping_answers(client):
    r = await client.post("/mcp", json=_rpc("ping"))
    assert r.json()["result"] == {}


@pytest.mark.anyio
async def test_unknown_method_is_a_jsonrpc_error(client):
    r = await client.post("/mcp", json=_rpc("tools/summon"))
    assert r.json()["error"]["code"] == -32601


# ── the rule: one authorization path ──────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_mcp_dispatches_into_the_same_handler_as_rest(client):
    """⛔ THE LOAD-BEARING ONE. A tool must call the REST handler, not query the store itself.

    Patching the handler and asserting it was reached proves the dispatch is real. If someone
    later 'optimises' a tool by talking to the store directly, this test fails — which is the
    only automatic warning that MCP has grown an authorization path of its own.
    """
    with patch("mantle.routers.artifacts_router.read_artifact") as handler:
        async def _ok(**kwargs):
            return {"id": kwargs["artifact_id"], "content_type": "text/plain"}
        handler.side_effect = _ok
        r = await client.post("/mcp", json=_rpc(
            "tools/call", {"name": "get_artifact", "arguments": {"artifact_id": "art-1"}}))

    assert r.status_code == 200
    assert handler.called, "get_artifact did not reach the REST handler — MCP has its own query path"
    # and the caller's identity was handed through rather than re-derived
    assert "auth" in handler.call_args.kwargs
    assert handler.call_args.kwargs["auth"].principal_id == "user-123"


@pytest.mark.anyio
async def test_every_tool_reaches_a_handler(client):
    """No tool may be advertised without a handler behind it — the manifest-vs-implementation gap."""
    from mantle.routers import mcp_router as mcp
    import inspect
    src = inspect.getsource(mcp._call_tool)
    for name in (t["name"] for t in mcp.TOOLS):
        assert f'name == "{name}"' in src, f"{name} is advertised but has no dispatch branch"


# ── tool results ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_a_failing_tool_is_a_result_with_isError_not_a_protocol_error(client):
    """A tool that raises must come back as a RESULT the model can read and adapt to.

    Returning a JSON-RPC error instead hides the reason from the model and reads as a broken
    server rather than a failed call."""
    with patch("mantle.routers.artifacts_router.read_artifact") as handler:
        handler.side_effect = RuntimeError("store unavailable")
        r = await client.post("/mcp", json=_rpc(
            "tools/call", {"name": "get_artifact", "arguments": {"artifact_id": "x"}}))

    body = r.json()
    assert "error" not in body, "a failed TOOL must not become a failed RPC"
    assert body["result"]["isError"] is True
    assert "store unavailable" in body["result"]["content"][0]["text"]


@pytest.mark.anyio
async def test_unknown_tool_is_invalid_params(client):
    r = await client.post("/mcp", json=_rpc("tools/call", {"name": "drop_everything"}))
    assert r.json()["error"]["code"] == -32602


@pytest.mark.anyio
async def test_a_missing_required_argument_is_invalid_params(client):
    r = await client.post("/mcp", json=_rpc("tools/call", {"name": "get_artifact",
                                                           "arguments": {}}))
    assert r.json()["error"]["code"] == -32602


# ── notifications and transport edges ─────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_a_notification_gets_202_and_no_body(client):
    """A notification has no `id`, so the client is not waiting for a reply.

    Answering one is a protocol violation that some clients surface as a hang."""
    r = await client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert r.status_code == 202
    assert r.content == b""


@pytest.mark.anyio
async def test_a_batch_of_only_notifications_also_gets_202(client):
    r = await client.post("/mcp", json=[
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "method": "notifications/cancelled"},
    ])
    assert r.status_code == 202
    assert r.content == b""


@pytest.mark.anyio
async def test_a_batch_returns_one_response_per_request(client):
    r = await client.post("/mcp", json=[_rpc("ping", req_id=1), _rpc("tools/list", req_id=2)])
    body = r.json()
    assert isinstance(body, list) and {m["id"] for m in body} == {1, 2}


@pytest.mark.anyio
async def test_get_refuses_rather_than_holding_a_stream_that_never_emits(client):
    """This server originates nothing, so it must say so.

    A 200 with an idle stream leaves the client waiting on a channel that will never carry a
    message — strictly worse than an honest refusal."""
    r = await client.get("/mcp")
    assert r.status_code == 405
    assert r.headers["allow"] == "POST"


@pytest.mark.anyio
async def test_malformed_json_is_a_parse_error(client):
    r = await client.post("/mcp", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32700


@pytest.mark.anyio
async def test_a_missing_jsonrpc_version_is_rejected(client):
    r = await client.post("/mcp", json={"method": "ping", "id": 1})
    assert r.json()["error"]["code"] == -32600


# ── the root: one URL, two audiences ──────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_root_serves_html_to_a_browser(client):
    r = await client.get("/", headers={"accept": "text/html,application/xhtml+xml"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Agience" in r.text and "/mcp" in r.text


@pytest.mark.anyio
async def test_root_serves_json_to_a_client(client):
    r = await client.get("/", headers={"accept": "application/json"})
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["links"]["mcp"] == "/mcp"


@pytest.mark.anyio
async def test_root_defaults_to_json_when_accept_is_absent(client):
    """A caller that states no preference is a machine — curl, a health check, a probe."""
    r = await client.get("/", headers={"accept": ""})
    assert r.headers["content-type"].startswith("application/json")


@pytest.mark.anyio
async def test_root_varies_on_accept(client):
    """⛔ Without `Vary: Accept` a cache keyed on URL alone can serve the HTML body to a JSON
    client (or the reverse). It only breaks behind a CDN, so nothing local would catch it."""
    for accept in ("text/html", "application/json"):
        r = await client.get("/", headers={"accept": accept})
        assert "accept" in r.headers.get("vary", "").lower()
