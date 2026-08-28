"""Shared helpers for working with Core artifact responses on MCP servers.

Core uses ``content_type`` in artifact context; MCP convention uses ``mimeType``.
These helpers standardize the translation so every server reads artifact fields
the same way.

This module serves external MCP consumers rather than Mantle itself: nothing in this
repository imports it. Its callers are the `agience-chorus` personas — `astra`,
`aria`, `sage`, `iris`, `seraph`, `ophan`, `lumen` — which call Mantle over the
wire and need its artifact fields in MCP's vocabulary. It sits under ``clients/``
because that package is about the wire between Mantle and someone else, and this
is the consumer's side of it: it reads Mantle's responses and touches no store.

It is a candidate to move into ``agience-chorus``, where every caller lives. What
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


def artifact_url(base: str, artifact_id: str, *suffix: str) -> str:
    """`{base}/artifacts/{id}` with the id encoded as ONE path segment.

    An artifact id is an opaque string and this corpus's ids carry characters that mean something
    in a URL. `canon:best-practices#intro` interpolated into an f-string yields the path
    `/artifacts/canon:best-practices` with `#intro` split off as a fragment — and a fragment is
    never sent to the server, so the request silently asks for a different artifact. Measured
    against `httpx.Request`:

        wn-glacier.n.01              path=/artifacts/wn-glacier.n.01           fragment=''
        canon:best-practices#intro   path=/artifacts/canon:best-practices      fragment='intro'

    Every canon id has a `#`, so every fetch of this project's own documentation asked for a
    truncated id instead. None of the 276 truncated forms is an artifact, so the request 404s: the
    failure is loud, and it covers all 6,480 canon artifacts.

    `safe=""` encodes the whole id, including `:` and `/`, so an id can never end a segment early
    or introduce one. The server decodes the path parameter, so the route matches the id it was
    given rather than a prefix of it.

    `suffix` appends further fixed segments — `artifact_url(base, aid, "children")` — and those are
    NOT encoded, because they are route structure rather than data.
    """
    from urllib.parse import quote

    parts = "/".join(str(p).strip("/") for p in suffix if str(p).strip("/"))
    tail = ("/" + parts) if parts else ""
    return "%s/artifacts/%s%s" % (str(base).rstrip("/"), quote(str(artifact_id), safe=""), tail)


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
