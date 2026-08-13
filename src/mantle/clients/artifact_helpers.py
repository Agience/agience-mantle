"""Shared helpers for working with Core artifact responses on MCP servers.

Core uses ``content_type`` in artifact context; MCP convention uses ``mimeType``.
These helpers standardize the translation so every server reads artifact fields
the same way.

⚠ THIS MODULE EXISTS FOR EXTERNAL MCP CONSUMERS, NOT FOR MANTLE. Nothing in this
repository imports it. Its callers are the `agience-chorus` personas — `astra`,
`aria`, `sage`, `iris`, `seraph`, `ophan`, `lumen` — which call Mantle over the
wire and need its artifact fields in MCP's vocabulary. It sits under ``clients/``
because that package is about the wire between Mantle and someone else, and this
is the consumer's side of it: it reads Mantle's responses and touches no store.

It is a CANDIDATE TO MOVE INTO ``agience-chorus``, where every caller lives. What
holds it here is that all seven personas would import it from one place either
way, and the move is theirs to make — not a deletion this repository can perform,
because nothing here would notice it break.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def register_types_manifest(mcp_instance: Any, server_name: str, server_file: str) -> None:
    """Register a ``types://{server_name}/manifest`` MCP resource.

    Scans the server's ``ui/application/`` directory for ``type.json`` files
    and returns them as a JSON manifest when the resource is read.  This
    enables runtime type discovery by the platform backend --- servers own
    their type definitions and the backend discovers them via standard MCP
    ``resources/read`` at bootstrap.

    Args:
        mcp_instance: The FastMCP server instance to register the resource on.
        server_name: The server persona name (e.g. ``"iris"``).
        server_file: The server's ``__file__`` path, used to locate the
            ``ui/`` directory relative to the server module.
    """
    ui_root = Path(server_file).parent / "ui" / "application"
    uri = f"types://{server_name}/manifest"

    @mcp_instance.resource(uri)
    async def types_manifest() -> str:
        """Return all type definitions owned by this server."""
        types: dict[str, Any] = {}
        if not ui_root.exists() or not ui_root.is_dir():
            return json.dumps(types)
        for type_dir in sorted(ui_root.iterdir()):
            if not type_dir.is_dir():
                continue
            type_json_path = type_dir / "type.json"
            if not type_json_path.exists():
                continue
            try:
                defn = json.loads(type_json_path.read_text(encoding="utf-8"))
                ct = defn.get("content_type", f"application/{type_dir.name}")
                types[ct] = defn
            except Exception:
                continue
        return json.dumps(types)


def parse_artifact_context(artifact: dict) -> dict:
    """Parse artifact context, handling both dict and JSON-string forms."""
    raw = artifact.get("context") or {}
    if isinstance(raw, str):
        raw = json.loads(raw)
    return raw if isinstance(raw, dict) else {}


def get_artifact_content_type(artifact: dict) -> str:
    """Return the lowercased MIME type from a Core artifact response.

    Reads ``context.content_type`` (the Agience canonical field) and strips
    any ``; charset=...`` suffix.  Returns empty string if not set.
    """
    ctx = parse_artifact_context(artifact)
    raw = ctx.get("content_type") or artifact.get("content_type") or ""
    return str(raw).split(";", 1)[0].strip().lower()
