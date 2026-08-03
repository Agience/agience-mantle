"""First-party MCP server registry — DYNAMIC (registration-driven).

Mantle is a database layer and carries NO built-in knowledge of persona servers.
The persona servers (Chorus) are responsible for REGISTERING themselves at startup
via ``POST /servers/register`` — Mantle learns them dynamically; nothing is
pre-known from a manifest file.

Registered records are persisted in ``platform_settings`` (category ``server``)
so they survive Mantle restarts even before a persona re-registers.

Server UUIDs are deterministic (``derive_uuid`` over the ``agience-server-<name>``
slug + the shared instance namespace), so Mantle and Chorus agree on the id used
to address a persona through Chorus's gateway — without exchanging it.

Public API
----------
Metadata:
- ``get_entry(name)`` / ``get_entry_by_client_id`` / ``all_entries`` /
  ``all_names`` / ``all_client_ids``

ID resolution:
- ``get_id(name)`` / ``get_name_by_id(uuid)`` / ``is_builtin_id(uuid)`` /
  ``resolve_name_to_id(name)``

Population / registration:
- ``register(db, *, name, client_id, path, ...)`` — a persona self-registers
- ``load_from_store(db)`` — reload persisted registrations at startup
- ``populate_ids()`` — re-derive UUIDs for current entries (compat)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

_SERVER_SETTING_PREFIX = "server."
_SERVER_SETTING_CATEGORY = "server"


@dataclass(frozen=True)
class ManifestEntry:
    """Immutable record for one first-party MCP server."""
    name: str          # human-readable name — "aria", "verso", …
    title: str         # display title
    path: str          # HTTP path on the unified host (e.g. "/aria/mcp")
    client_id: str     # JWT audience / credential identity
    role: str          # one-line role description
    summary: str       # longer description for seed artifacts


# Module-level state — starts EMPTY (no manifest). Populated by load_from_store()
# at startup and by register() at runtime.
_ENTRIES: List[ManifestEntry] = []
_BY_NAME: Dict[str, ManifestEntry] = {}
_BY_CLIENT_ID: Dict[str, ManifestEntry] = {}
_ALL_CLIENT_IDS: FrozenSet[str] = frozenset()
_ID_BY_NAME: Dict[str, str] = {}
_NAME_BY_ID: Dict[str, str] = {}


def _server_slug(name: str) -> str:
    from mantle.services.bootstrap_types import SERVER_ARTIFACT_SLUG_PREFIX
    return f"{SERVER_ARTIFACT_SLUG_PREFIX}{name}"


def _derive_server_uuid(name: str) -> Optional[str]:
    """Deterministic server UUID (matches the seed loader's derivation), so Mantle
    and Chorus agree on the id without exchanging it. None if the instance
    namespace is unavailable."""
    try:
        from mantle.services.seed_provisioning.loader import derive_uuid, get_instance_namespace
        return derive_uuid(get_instance_namespace(), "agience", _server_slug(name))
    except Exception:
        logger.debug("server uuid derivation failed for %s", name, exc_info=True)
        return None


def _reindex() -> None:
    global _BY_NAME, _BY_CLIENT_ID, _ALL_CLIENT_IDS
    _BY_NAME = {e.name: e for e in _ENTRIES}
    _BY_CLIENT_ID = {e.client_id: e for e in _ENTRIES}
    _ALL_CLIENT_IDS = frozenset(e.client_id for e in _ENTRIES)


def _add_entry(entry: ManifestEntry) -> None:
    """Insert/replace an entry + refresh its UUID indexes."""
    global _ENTRIES
    _ENTRIES = [e for e in _ENTRIES if e.name != entry.name] + [entry]
    _reindex()
    uuid = _derive_server_uuid(entry.name)
    if uuid:
        from mantle.services.platform_topology import register_id
        _ID_BY_NAME[entry.name] = uuid
        _NAME_BY_ID[uuid] = entry.name
        register_id(_server_slug(entry.name), uuid)


# ---------------------------------------------------------------------------
# Registration + persistence
# ---------------------------------------------------------------------------

def register(
    db,
    *,
    name: str,
    client_id: str,
    path: str,
    title: str = "",
    role: str = "",
    summary: str = "",
) -> Optional[str]:
    """Register (or re-register) a persona server. Returns its deterministic UUID.

    Called by ``POST /servers/register`` when a persona (Chorus) declares itself.
    Idempotent — re-registering updates the record. Persists to platform_settings
    so the registration survives a Mantle restart.
    """
    if not name or not client_id:
        return None
    entry = ManifestEntry(
        name=name, title=title or name, path=path or f"/{name}/mcp",
        client_id=client_id, role=role, summary=summary,
    )
    _add_entry(entry)
    try:
        from mantle.services.platform_settings_service import settings as platform_settings
        platform_settings.set_setting(
            db,
            key=f"{_SERVER_SETTING_PREFIX}{name}",
            value=json.dumps(asdict(entry), separators=(",", ":")),
            category=_SERVER_SETTING_CATEGORY,
        )
    except Exception:
        logger.warning("failed to persist server registration for %s", name, exc_info=True)
    logger.info("server registered: %s (client_id=%s)", name, client_id)
    return _ID_BY_NAME.get(name)


def load_from_store(db) -> None:
    """Reload persisted persona registrations into the in-memory registry.

    Reads platform_settings DIRECTLY from the lattice (not the in-memory cache) so it
    is independent of when ``platform_settings.load_all`` runs in the startup
    sequence. Called at startup so persona resolution works across Mantle restarts
    even before a persona re-registers.
    """
    try:
        from mantle.db import identity_backend as identity_store
        rows = identity_store.get_all_platform_settings(db)
    except Exception:
        logger.debug("server registry load_from_store: settings unavailable", exc_info=True)
        return
    loaded = 0
    for row in rows or []:
        key = (row.get("id") if isinstance(row, dict) else "") or ""
        if not key.startswith(_SERVER_SETTING_PREFIX):
            continue
        raw = row.get("value")
        if not raw:
            continue
        try:
            data = json.loads(raw)
            _add_entry(ManifestEntry(
                name=data["name"], title=data.get("title", data["name"]),
                path=data.get("path", ""), client_id=data["client_id"],
                role=data.get("role", ""), summary=data.get("summary", ""),
            ))
            loaded += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("skipping malformed server registration row: %r", raw)
    logger.info("server registry loaded %d persona(s) from store", loaded)


# ---------------------------------------------------------------------------
# Metadata API
# ---------------------------------------------------------------------------

def get_entry(name: str) -> Optional[ManifestEntry]:
    return _BY_NAME.get(name)


def get_entry_by_client_id(client_id: str) -> Optional[ManifestEntry]:
    return _BY_CLIENT_ID.get(client_id)


def all_entries() -> List[ManifestEntry]:
    return list(_ENTRIES)


def all_names() -> List[str]:
    return [e.name for e in _ENTRIES]


def all_client_ids() -> FrozenSet[str]:
    """Frozenset of all registered first-party client_ids (auth/gate fast-path)."""
    return _ALL_CLIENT_IDS


# ---------------------------------------------------------------------------
# ID resolution
# ---------------------------------------------------------------------------

def populate_ids() -> None:
    """Re-derive UUIDs for all current entries from the topology/derivation.

    Kept for compatibility with the startup sequence; ``register`` /
    ``load_from_store`` already populate ids, but this re-syncs after the
    platform topology's ``pre_resolve_platform_ids`` runs.
    """
    from mantle.services.platform_topology import get_id_optional as topo_get_id

    for entry in _ENTRIES:
        slug = _server_slug(entry.name)
        uuid = topo_get_id(slug) or _derive_server_uuid(entry.name)
        if uuid:
            _ID_BY_NAME[entry.name] = uuid
            _NAME_BY_ID[uuid] = entry.name
    logger.info("Server registry populated: %d servers", len(_ID_BY_NAME))


def get_id(name: str) -> Optional[str]:
    return _ID_BY_NAME.get(name)


def get_name_by_id(server_id: str) -> Optional[str]:
    return _NAME_BY_ID.get(server_id)


def is_builtin_id(server_id: str) -> bool:
    return server_id in _NAME_BY_ID


def resolve_name_to_id(name: str) -> str:
    """Resolve a persona name to its UUID. Raises ``ValueError`` if the name is
    not registered (or ids not yet populated)."""
    uuid = _ID_BY_NAME.get(name)
    if uuid is None:
        if name not in _BY_NAME:
            raise ValueError(f"Server name '{name}' is not registered")
        raise ValueError(
            f"Server '{name}' is registered but has no id yet "
            "(instance namespace unavailable / bootstrap incomplete)."
        )
    return uuid


def _reset_for_tests() -> None:
    """Clear all state (test hook)."""
    global _ENTRIES, _ID_BY_NAME, _NAME_BY_ID
    _ENTRIES = []
    _ID_BY_NAME = {}
    _NAME_BY_ID = {}
    _reindex()
