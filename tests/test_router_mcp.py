"""Tests for `mcp_router.py`.

The live capability-introspection endpoints (`GET /mcp/servers` and
`GET /mcp/workspaces/{id}/servers`) do not exist. MCP server records
are queried via:

  GET /artifacts?content_type=application/vnd.agience.mcp-server+json

Tool invocation and resource ops flow through the unified artifact surface:

  POST /artifacts/{server_id}/invoke
  POST /artifacts/{server_id}/op/resources_read
  POST /artifacts/{server_id}/op/resources_import
"""

import pytest


class TestMCPRouterNoLiveIntrospection:
    """The live-introspection endpoints must not be reachable.

    Asserted as `== 404`, the code a route that does not exist actually returns. `not in (200, 201)`
    was the claim "not reachable" written so loosely that a 500 from a route that DOES exist and
    crashes satisfies it — so would a 401 from a route that exists behind auth. Both are the
    endpoint being back, which is the one thing these tests are here to notice.
    """

    @pytest.mark.asyncio
    async def test_list_all_servers_endpoint_gone(self, client):
        """`GET /mcp/servers` does not exist; query artifacts by content_type instead."""
        response = await client.get("/mcp/servers")
        assert response.status_code == 404, response.text

    @pytest.mark.asyncio
    async def test_list_workspace_servers_endpoint_gone(self, client):
        """`GET /mcp/workspaces/{id}/servers` does not exist."""
        response = await client.get("/mcp/workspaces/ws_123/servers")
        assert response.status_code == 404, response.text


class TestPhase7DRemovedEndpoints:
    """The four action endpoints under `/mcp/servers/{id}/...` must
    return non-2xx. Their replacements live under `/artifacts/{id}/invoke`
    and `/artifacts/{id}/op/{op_name}`.

    Note: the FastMCP server transport is mounted at the `/mcp` prefix, so
    unrecognized `/mcp/servers/{id}/...` sub-paths fall through to that
    transport which returns 401 (auth required for unknown MCP method).
    Any non-2xx status confirms the dedicated route does not exist.
    """

    _GONE = (401, 404, 405)

    @pytest.mark.asyncio
    async def test_old_tools_call_endpoint_gone(self, client):
        response = await client.post(
            "/mcp/servers/agience-beam/tools/call",
            json={"tool": "search", "arguments": {}},
        )
        assert response.status_code in self._GONE

    @pytest.mark.asyncio
    async def test_old_resources_read_endpoint_gone(self, client):
        response = await client.post(
            "/mcp/servers/agience-beam/resources/read",
            json={"uri": "agience://collections/c1"},
        )
        assert response.status_code in self._GONE

    @pytest.mark.asyncio
    async def test_old_resources_import_endpoint_gone(self, client):
        response = await client.post(
            "/mcp/servers/github_mcp/resources/import",
            json={"workspace_id": "ws_123", "resources": []},
        )
        assert response.status_code in self._GONE

    @pytest.mark.asyncio
    async def test_old_server_info_endpoint_gone(self, client):
        response = await client.get("/mcp/servers/agience-beam/info")
        assert response.status_code in self._GONE

    @pytest.mark.asyncio
    async def test_legacy_upsert_server_endpoint_still_gone(self, client):
        """The user-level POST /mcp/servers (registry upsert) does not exist."""
        response = await client.post(
            "/mcp/servers",
            json={"id": "server_123", "label": "Test Server"},
        )
        assert response.status_code in (401, 404, 405, 422)
