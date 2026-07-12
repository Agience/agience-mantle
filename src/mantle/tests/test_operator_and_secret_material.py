"""Foundation for composable operators + secrets-as-artifacts.

1. McpToolHandler resolves a context-ref `dispatch.arguments` (string → dict →
   each value resolved) — this is what lets ONE generic `operator+json` type
   back every tool-bound operation.
2. set_secret_material / fetch_secret_material round-trip — the `secret+json`
   `fetch` operation handler.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from services.handler_registry import McpToolHandler  # noqa: E402
import services.secrets_service as ss  # noqa: E402
from services import secrets_service  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Operator dispatch — context-ref arguments
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_operator_dispatch_resolves_context_arguments():
    """An operator's invoke points dispatch.arguments at its own context spec;
    the handler resolves it to a dict and resolves each value from the body."""
    handler = McpToolHandler()

    # Operator artifact: context.operator carries the execution spec.
    artifact = {
        "_key": "op-1",
        "root_id": "op-1",
        "context": {
            "content_type": "application/vnd.agience.operator+json",
            "operator": {
                "server": "iris",
                "tool": "send_email",
                "arguments": {
                    "to": "$.body.params.to",
                    "subject": "$.body.params.subject",
                    "body_html": "$.body.params.body_html",
                },
            },
        },
    }
    op_spec = SimpleNamespace(dispatch={
        "kind": "mcp_tool",
        "server_ref": "$.context.operator.server",
        "tool_ref": "$.context.operator.tool",
        "arguments": "$.context.operator.arguments",
    })
    body = {"params": {"to": "connect@agience.ai", "subject": "Hi", "body_html": "<p>x</p>"}}
    ctx = SimpleNamespace(user_id="u-1", arango_db=None)

    captured = {}

    def fake_call_tool(server_id, tool_name, arguments, *, user_id):
        captured.update(server=server_id, tool=tool_name, args=arguments, user=user_id)
        return [{"type": "text", "text": "{\"status\": \"sent\"}"}]

    with (
        patch("services.chorus_client.call_tool", side_effect=fake_call_tool),
        patch("services.chorus_client.is_uuid_like", return_value=False),
        patch("services.server_registry.resolve_name_to_id", return_value="iris-uuid"),
    ):
        await handler.run(artifact, op_spec, body, ctx)

    assert captured["server"] == "iris-uuid"
    assert captured["tool"] == "send_email"
    assert captured["args"] == {
        "to": "connect@agience.ai", "subject": "Hi", "body_html": "<p>x</p>",
    }


# ---------------------------------------------------------------------------
# 2. Secret material round-trip
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _cipher():
    ss._cipher = None
    key = Fernet.generate_key().decode()
    with patch("services.secrets_service.get_encryption_key", return_value=key):
        yield
    ss._cipher = None


def test_set_then_fetch_secret_material_round_trip():
    # Custody lives in Origin's vault — fake it as a dict keyed by secret id.
    store: dict = {}
    fake_client = MagicMock()
    fake_client.store_secret_material.side_effect = (
        lambda sid, val, category="secret": (store.__setitem__(sid, val), True)[1]
    )
    fake_client.resolve_secret_material.side_effect = lambda sid: store.get(sid)

    with patch("clients.origin_client.get_origin_client", return_value=fake_client):
        secrets_service.set_secret_material(
            db=None, owner_id="owner-1",
            secret_id="secret-artifact-1",
            value="refresh-token-xyz",
            secret_type="oauth_refresh_token",
            provider="google",
            authorizer_id="authz-1",
        )
        # The secret+json `fetch` op handler resolves material by artifact id.
        artifact = {"root_id": "secret-artifact-1", "created_by": "owner-1"}
        result = secrets_service.fetch_secret_material(
            artifact, {}, SimpleNamespace(arango_db=MagicMock()),
        )

    assert result["secret_id"] == "secret-artifact-1"
    assert result["material"] == "refresh-token-xyz"
    # Stored in Origin keyed by the secret artifact id; the type is the category.
    fake_client.store_secret_material.assert_called_once_with(
        "secret-artifact-1", "refresh-token-xyz", category="oauth_refresh_token"
    )


def test_fetch_secret_material_missing_raises():
    fake_client = MagicMock()
    fake_client.resolve_secret_material.return_value = None
    with patch("clients.origin_client.get_origin_client", return_value=fake_client):
        artifact = {"root_id": "nope", "created_by": "owner-1"}
        with pytest.raises(RuntimeError, match="No material stored"):
            secrets_service.fetch_secret_material(artifact, {}, SimpleNamespace(arango_db=MagicMock()))
