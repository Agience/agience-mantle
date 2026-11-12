"""A server artifact's tool runs as a workflow step through the public "Use a server tool" operator
(John, 2026-11-12, pilot check 5). Its server is an artifact input, so the step links it and the
run's grant reaches it; Iris runs one tools/call with the server's own key."""
from __future__ import annotations

from mantle.services.seed_provisioning import platform_operators as P


def test_use_a_server_tool_is_public_and_takes_the_server_as_an_artifact():
    op = next(o for o in P.OPERATORS if o["slug"] == "op-use-a-server-tool")
    assert op["public"] is True and (op["server"], op["tool"]) == ("iris", "use_server_tool")
    server = op["input_schema"]["properties"]["server"]
    assert server["format"] == "artifact-id"
    assert server["x-content-types"] == ["application/vnd.agience.mcp-server+json"]
    assert op["input_schema"]["required"] == ["server", "tool"]
    assert op["arguments"] == {"server_artifact_id": "$.body.params.server",
                               "tool": "$.body.params.tool", "arguments": "$.body.params.arguments?"}
