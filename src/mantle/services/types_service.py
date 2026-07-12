from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TypeResolutionResult:
    content_type: str
    definition: Dict[str, Any]
    sources: List[str]
    validation_errors: List[str]


def _normalize_content_type(content_type: str) -> str:
    content_type = (content_type or "").strip().lower()
    if not content_type:
        return ""
    # Drop parameters like charset.
    if ";" in content_type:
        content_type = content_type.split(";", 1)[0].strip()
    return content_type


def _repo_root() -> Path:
    # backend/services/types_service.py -> backend -> repo root
    return Path(__file__).resolve().parents[3]


def _builtin_types_root() -> Path:
    """Builtin type manifests live under `package/types/`."""
    return _repo_root() / "package" / "types"


def _default_server_ui_roots() -> List[Path]:
    """Deprecated — always empty. Mantle no longer scans Chorus's filesystem for
    server-owned types. That coupled Mantle to Chorus and forced the chorus tree
    into Mantle's image. Servers now SELF-REGISTER the types they own (Mantle
    stays passive): each type arrives as a pushed
    ``application/vnd.agience.type+json`` and is held in the in-memory runtime
    registry (``_runtime_types``), persisted via ``type_registry_store`` so a
    Mantle restart rehydrates without contacting any server. See
    ``register_runtime_type``."""
    return []


def get_types_roots() -> List[Path]:
    """Return search roots for the folder-based type system.

    Order matters: earlier roots take precedence.

    Env vars:
    - AGIENCE_TYPES_PATHS: extra roots (os.pathsep-separated)
    - AGIENCE_TYPES_DISABLE_BUILTIN: if truthy, do not include repo `types/`
    """
    roots: List[Path] = []
    seen: set[Path] = set()

    def add_root(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(resolved)

    extra = os.getenv("AGIENCE_TYPES_PATHS", "")
    if extra:
        for raw in extra.split(os.pathsep):
            raw = (raw or "").strip()
            if not raw:
                continue
            p = Path(raw)
            if not p.is_absolute():
                p = _repo_root() / p
            if p.exists() and p.is_dir():
                add_root(p)

    disable_builtin = os.getenv("AGIENCE_TYPES_DISABLE_BUILTIN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if not disable_builtin:
        builtin = _builtin_types_root()
        if builtin.exists() and builtin.is_dir():
            add_root(builtin)

    return roots


def _merge_ordered_roots() -> List[Path]:
    """FILESYSTEM type roots in MERGE precedence (LOWEST first): core
    (``package/types``) is the base/fallback; a local override
    (``AGIENCE_TYPES_PATHS``) overrides core. Deep-merge applies LATER over
    EARLIER.

    Server-owned types are NOT on the filesystem here — they are self-registered
    at runtime and overlaid ON TOP of these roots in :func:`resolve_type_definition`
    (an "Open-with"-style chain: core base < runtime/self-registered < local).
    A type lives canonically in exactly ONE place; higher layers contribute
    partial overrides (e.g. just a viewer), never a full duplicate.
    """
    roots: List[Path] = []
    seen: set[Path] = set()

    def add_root(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not (resolved.exists() and resolved.is_dir()):
            return
        seen.add(resolved)
        roots.append(resolved)

    disable_builtin = os.getenv("AGIENCE_TYPES_DISABLE_BUILTIN", "").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }
    if not disable_builtin:
        add_root(_builtin_types_root())              # core (base)
    # Chorus types are NOT read from the filesystem here. Servers self-register
    # them; resolution overlays the pushed runtime defs (see resolve_type_definition).
    extra = os.getenv("AGIENCE_TYPES_PATHS", "")
    if extra:
        for raw in extra.split(os.pathsep):
            raw = (raw or "").strip()
            if not raw:
                continue
            p = Path(raw)
            if not p.is_absolute():
                p = _repo_root() / p
            add_root(p)                              # local (overrides chorus)
    return roots


def _content_type_to_rel_folder(content_type: str) -> Optional[Path]:
    content_type = _normalize_content_type(content_type)
    if not content_type or "/" not in content_type:
        return None
    top, sub = content_type.split("/", 1)
    if sub == "*":
        sub = "_wildcard"
    return Path(top) / sub


def _find_type_folder(roots: Iterable[Path], content_type: str) -> Optional[Tuple[Path, str]]:
    """Return (folder_path, source_label) for the highest-priority definition.

    Root ordering from ``get_types_roots()`` defines the priority:
    extra roots > builtin ``types/`` > server ``ui/`` overlays.  When a
    builtin type skeleton also has a server viewer overlay, the builtin
    definition wins — this is the expected configuration, not an error.
    """
    rel = _content_type_to_rel_folder(content_type)
    if rel is None:
        return None

    for root in roots:
        candidate = root / rel
        if candidate.exists() and candidate.is_dir():
            return (candidate, str(candidate))

    return None


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        # utf-8-sig tolerates a leading BOM (a common Windows editor artifact).
        # Plain utf-8 raises on a BOM, which previously caused the *entire*
        # type definition to be dropped silently — a phantom 404 at runtime.
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        # The file exists but is unreadable or malformed. Never swallow this:
        # a broken type.json must be visible in logs, not an invisible gap in
        # the type's operations / UI / schema.
        logger.warning("Failed to read JSON file %s: %s", path, exc)
        return None


def _load_handlers(handlers_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not handlers_dir.exists() or not handlers_dir.is_dir():
        return out

    for file in sorted(handlers_dir.glob("*.json")):
        obj = _read_json_file(file)
        if not isinstance(obj, dict):
            continue
        cap = obj.get("capability")
        if isinstance(cap, str) and cap:
            out[cap] = obj
        else:
            # Fallback to filename stem.
            out[file.stem] = obj
    return out


def _deep_merge(parent: Any, child: Any) -> Any:
    """Deterministic merge: objects recurse, child wins; lists replaced by child."""
    if isinstance(parent, dict) and isinstance(child, dict):
        merged = dict(parent)
        for k, v in child.items():
            if k in merged:
                merged[k] = _deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged
    return child


def _resolve_handler_target(handler_obj: Dict[str, Any]) -> Optional[str]:
    """Resolve a callable target from a handler contract object.

    Supports both current and draft forms:
    - {"tool": "extract_text"}
    - {"implementation": {"kind": "builtin", "id": "text.extract_text"}}
    - {"implementation": {"kind": "mcp-tool", "tool": "extract_text"}}
    """
    if not isinstance(handler_obj, dict):
        return None

    direct_tool = handler_obj.get("tool")
    if isinstance(direct_tool, str) and direct_tool.strip():
        return direct_tool.strip()

    impl = handler_obj.get("implementation")
    if not isinstance(impl, dict):
        return None

    # Builtin handlers typically use `id`, while tool handlers may use `tool`.
    for key in ("tool", "id"):
        value = impl.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_handler_binding(handler_obj: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Resolve handler binding metadata from a handler contract object.

    Returns a mapping with:
    - tool: required target identifier
    - server_artifact_id: optional MCP server artifact id/name
    """
    tool = _resolve_handler_target(handler_obj)
    if not tool:
        return None

    binding: Dict[str, str] = {"tool": tool}

    # Allow server declaration at the handler root.
    root_server = handler_obj.get("server") or handler_obj.get("server_artifact_id")
    if isinstance(root_server, str) and root_server.strip():
        binding["server_artifact_id"] = root_server.strip()

    # Allow server declaration inside implementation.
    impl = handler_obj.get("implementation")
    if isinstance(impl, dict):
        impl_server = impl.get("server") or impl.get("server_artifact_id")
        if isinstance(impl_server, str) and impl_server.strip():
            binding["server_artifact_id"] = impl_server.strip()

    return binding


def _load_folder_definition(folder: Path) -> Tuple[Dict[str, Any], List[str]]:
    sources = [str(folder)]

    type_json = _read_json_file(folder / "type.json") or {}
    schema_json = _read_json_file(folder / "schema.json")
    preview_json = _read_json_file(folder / "preview.json")
    behaviors_json = _read_json_file(folder / "behaviors.json")
    handlers = _load_handlers(folder / "handlers")

    # UI metadata lives inside type.json["ui"] (merged format).
    ui_json = type_json.pop("ui", None)
    # `operations` is a top-level block under type.json (Phase 0 — Enterprise
    # Eventing refactor). Promote it so resolve_operation() can find it at
    # `definition["operations"]` without digging through `definition["type"]`.
    operations_json = type_json.pop("operations", None)
    relationships_json = type_json.pop("relationships", None)

    definition: Dict[str, Any] = {
        "type": type_json,
        "handlers": handlers,
    }
    if schema_json is not None:
        definition["schema"] = schema_json
    if ui_json is not None:
        definition["ui"] = ui_json
    if operations_json is not None:
        definition["operations"] = operations_json
    if relationships_json is not None:
        definition["relationships"] = relationships_json
    if preview_json is not None:
        definition["preview"] = preview_json
    if behaviors_json is not None:
        definition["behaviors"] = behaviors_json

    return definition, sources


def _collect_type_validation_errors(definition: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    handlers = definition.get("handlers")
    if not isinstance(handlers, dict):
        handlers = {}

    behaviors = definition.get("behaviors")
    if not isinstance(behaviors, dict):
        return errors

    events = behaviors.get("events")
    if events is None:
        return errors
    if not isinstance(events, dict):
        return ["behaviors.events must be an object"]

    for event_name, event_obj in events.items():
        if not isinstance(event_name, str) or not event_name.strip():
            errors.append("behaviors.events keys must be non-empty strings")
            continue
        if not isinstance(event_obj, dict):
            errors.append(f"behaviors.events.{event_name} must be an object")
            continue

        has_tool = isinstance(event_obj.get("tool"), str) and bool(str(event_obj.get("tool")).strip())
        has_handler = isinstance(event_obj.get("handler"), str) and bool(str(event_obj.get("handler")).strip())

        if has_tool and has_handler:
            errors.append(f"behaviors.events.{event_name} must declare only one of 'tool' or 'handler'")
        if not has_tool and not has_handler:
            errors.append(f"behaviors.events.{event_name} must declare either 'tool' or 'handler'")

        for server_key in ("server", "server_artifact_id"):
            if server_key in event_obj and not (
                isinstance(event_obj.get(server_key), str) and str(event_obj.get(server_key)).strip()
            ):
                errors.append(f"behaviors.events.{event_name}.{server_key} must be a non-empty string")

        if has_handler:
            cap = Path(str(event_obj.get("handler"))).stem
            handler_obj = handlers.get(cap)
            if not isinstance(handler_obj, dict):
                errors.append(
                    f"behaviors.events.{event_name}.handler references missing handler '{cap}'"
                )
                continue
            binding = _resolve_handler_binding(handler_obj)
            if not isinstance(binding, dict) or not binding.get("tool"):
                errors.append(f"handlers.{cap} does not define a valid callable target")

    return errors


def resolve_type_definition(content_type: str, *, roots: Optional[List[Path]] = None) -> Optional[TypeResolutionResult]:
    """Resolve a type definition.

    Resolution is an "Open-with"-style override chain (LOWEST first):
    ``core (package/types) < self-registered (runtime) < local``, then layered
    over any ``inherits`` parents (child wins). Mirrors the Facet content-types
    plugin so the platform and the frontend resolve types identically.

    The runtime overlay (server self-registered types, ``_runtime_types``) is
    applied ONLY on the default-roots path (``roots is None``). When an explicit
    ``roots`` list is passed (tests), resolution is pure-filesystem — no runtime
    overlay — so callers can probe folder semantics in isolation.

    Matching: exact ``top/subtype`` else category wildcard ``top/*``
    (``_wildcard`` folder). The wildcard fallback is whole-resolution, never
    per-layer — otherwise a type with an exact def would be polluted by a `*`.
    """
    content_type = _normalize_content_type(content_type)
    if not content_type:
        return None

    use_runtime = roots is None
    merge_roots = roots if roots is not None else _merge_ordered_roots()

    # Own-definition: core/local filesystem layers (base-first, deep-merged),
    # then the self-registered runtime overlay (highest, deep-merged on top).
    def _gather(target: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        acc: Optional[Dict[str, Any]] = None
        srcs: List[str] = []
        for root in merge_roots:
            match = _find_type_folder([root], target)
            if match is None:
                continue
            folder, _source = match
            folder_def, folder_sources = _load_folder_definition(folder)
            acc = folder_def if acc is None else _deep_merge(acc, folder_def)
            srcs.extend(folder_sources)
        if use_runtime:
            rt = _runtime_types.get(target)
            # Lazy path: only when the type is absent from BOTH the core
            # filesystem and the warm cache do we hit the artifact store — an
            # O(1) on-demand fetch for the long tail of third-party types
            # ("find them via artifacts, in context"). Core types never trigger a
            # DB hit; resolved/missing results are cached so it fires at most once.
            if rt is None and acc is None and _lazy_type_loader is not None:
                try:
                    if _lazy_type_loader(target):
                        rt = _runtime_types.get(target)
                except Exception:
                    logger.warning("Lazy type loader failed for %s", target, exc_info=True)
            if rt is not None:
                acc = rt.definition if acc is None else _deep_merge(acc, rt.definition)
                srcs = srcs + rt.sources
        return acc, srcs

    own, sources = _gather(content_type)
    if own is None and "/" in content_type:
        top, _sub = content_type.split("/", 1)
        own, sources = _gather(f"{top}/*")

    if own is None:
        return None

    # Inheritance: resolve parents (also cross-root merged), layer child on top.
    type_obj = own.get("type") if isinstance(own.get("type"), dict) else {}
    inherits = type_obj.get("inherits") if isinstance(type_obj, dict) else None

    merged = own
    if isinstance(inherits, list) and inherits:
        base: Optional[Dict[str, Any]] = None
        parent_sources: List[str] = []
        for parent_ct in inherits:
            if not isinstance(parent_ct, str):
                continue
            parent_res = resolve_type_definition(parent_ct, roots=roots)
            if parent_res is None:
                continue
            parent_sources = parent_res.sources + parent_sources
            base = parent_res.definition if base is None else _deep_merge(base, parent_res.definition)
        if base is not None:
            merged = _deep_merge(base, own)
            sources = parent_sources + sources

    validation_errors = _collect_type_validation_errors(merged)
    if validation_errors:
        logger.warning("Type '%s' has %d contract validation error(s)", content_type, len(validation_errors))

    return TypeResolutionResult(
        content_type=content_type,
        definition=merged,
        sources=sources,
        validation_errors=validation_errors,
    )


def resolve_capability_target(
    content_type: str,
    capability: str,
    *,
    roots: Optional[List[Path]] = None,
) -> Optional[str]:
    """Resolve the declared target for a type capability.

    Returns a target name suitable for runtime invocation.
    For builtins this is typically a builtin id (e.g. `text.extract_text`),
    for remote handlers this is typically a tool name.
    """
    if not capability:
        return None

    res = resolve_type_definition(content_type, roots=roots)
    if res is None:
        return None

    handlers = res.definition.get("handlers")
    if not isinstance(handlers, dict):
        return None

    handler_obj = handlers.get(capability)
    if not isinstance(handler_obj, dict):
        return None

    return _resolve_handler_target(handler_obj)


def resolve_event_target(
    content_type: str,
    event_name: str,
    *,
    roots: Optional[List[Path]] = None,
) -> Optional[str]:
    """Resolve a declared target for a lifecycle event.

    Supports:
    - Draft event contract: behaviors.events.<event>.tool
    - Existing file-ref form: behaviors.events.<event>.handler -> handlers/<capability>.json
    """
    binding = resolve_event_binding(content_type, event_name, roots=roots)
    if not isinstance(binding, dict):
        return None
    tool = binding.get("tool")
    return tool if isinstance(tool, str) and tool.strip() else None


def resolve_event_binding(
    content_type: str,
    event_name: str,
    *,
    roots: Optional[List[Path]] = None,
) -> Optional[Dict[str, str]]:
    """Resolve a lifecycle event binding (tool + optional server)."""
    if not event_name:
        return None

    res = resolve_type_definition(content_type, roots=roots)
    if res is None:
        return None

    behaviors = res.definition.get("behaviors")
    if not isinstance(behaviors, dict):
        return None

    events = behaviors.get("events")
    if not isinstance(events, dict):
        return None

    event_obj = events.get(event_name)
    if not isinstance(event_obj, dict):
        return None

    direct_tool = event_obj.get("tool")
    if isinstance(direct_tool, str) and direct_tool.strip():
        binding: Dict[str, str] = {"tool": direct_tool.strip()}
        direct_server = event_obj.get("server") or event_obj.get("server_artifact_id")
        if isinstance(direct_server, str) and direct_server.strip():
            binding["server_artifact_id"] = direct_server.strip()
        return binding

    handler_ref = event_obj.get("handler")
    if not isinstance(handler_ref, str) or not handler_ref.strip():
        return None

    # Common form: "handlers/open.json" -> capability "open"
    cap = Path(handler_ref).stem
    handlers = res.definition.get("handlers")
    if not isinstance(handlers, dict):
        return None
    handler_obj = handlers.get(cap)
    if not isinstance(handler_obj, dict):
        return None

    # Event-level server (if provided) overrides handler-level server.
    binding = _resolve_handler_binding(handler_obj)
    if not binding:
        return None

    event_server = event_obj.get("server") or event_obj.get("server_artifact_id")
    if isinstance(event_server, str) and event_server.strip():
        binding["server_artifact_id"] = event_server.strip()

    return binding


# ---------------------------------------------------------------------------
# Operations schema (Phase 0 — Enterprise Eventing refactor)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperationSpec:
    """Normalized view of an `operations.{op_name}` entry in type.json."""

    name: str
    enabled: bool
    requires_grant: str
    dispatch: Dict[str, Any]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    emits: List[Dict[str, Any]]
    observe: Optional[Dict[str, Any]]
    audit: bool


_OP_NAME_TO_GRANT_FLAG = {
    "create": "create",
    "read": "read",
    "update": "update",
    "delete": "delete",
    "invoke": "invoke",
    "add": "add",
    "search": "search",
    "own": "own",
}


# ---------------------------------------------------------------------------
# Runtime type registration (MCP server discovery)
# ---------------------------------------------------------------------------

# Self-registered types — an in-memory CACHE of definitions a server owns, keyed
# by content type. NOT a formal global registry: it holds only what has actually
# been touched (warm platform set + lazily-resolved app types), never the whole
# (eventual millions-strong) universe.
_runtime_types: Dict[str, TypeResolutionResult] = {}

# Optional lazy loader: ``fn(content_type) -> bool``. Called on a resolution MISS
# (type not in the core filesystem and not already cached) to fetch the type's
# definition from its artifact on demand — the "find them via artifacts, lazily,
# in context" path for the long tail of app/third-party types. The loader should
# register what it finds (``register_runtime_type``) and return True, or return
# False if no such type-def artifact exists. Wired at startup (see main.py).
_lazy_type_loader: Optional[Any] = None


def set_lazy_type_loader(loader: Optional[Any]) -> None:
    """Install the lazy type loader (see ``_lazy_type_loader``). Pass None to clear."""
    global _lazy_type_loader
    _lazy_type_loader = loader


def _parse_raw_type_definition(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a raw type.json dict into the standard definition shape.

    Mirrors ``_load_folder_definition`` logic: pops ``ui`` and ``operations``
    from the type block and promotes them to top-level keys.
    """
    type_json = dict(raw)  # don't mutate the caller's dict
    ui_json = type_json.pop("ui", None)
    operations_json = type_json.pop("operations", None)
    relationships_json = type_json.pop("relationships", None)

    definition: Dict[str, Any] = {"type": type_json, "handlers": {}}
    if ui_json is not None:
        definition["ui"] = ui_json
    if operations_json is not None:
        definition["operations"] = operations_json
    if relationships_json is not None:
        definition["relationships"] = relationships_json
    return definition


def register_runtime_type(
    content_type: str,
    raw_definition: Dict[str, Any],
    source: str,
) -> None:
    """Register (or re-register) a type a server OWNS — the self-registration path.

    ``raw_definition`` is the full ``type.json`` the owning server published. It
    is parsed into the standard definition shape and stored as the runtime
    OVERLAY for this content type. Resolution layers it on top of the core
    (``package/types``) base (see :func:`resolve_type_definition`), so a server
    may publish either a full type (chorus-owned, e.g. ``chat``) or a partial
    override of a core type (e.g. a viewer for ``application/json``).

    Upsert semantics: a re-push (server restart / edit) replaces the prior
    overlay — last publish wins — and the per-type cache entry is invalidated.
    """
    key = _normalize_content_type(content_type)
    if not key:
        return

    definition = _parse_raw_type_definition(raw_definition)
    _runtime_types[key] = TypeResolutionResult(
        content_type=key,
        definition=definition,
        sources=[source],
        validation_errors=_collect_type_validation_errors(definition),
    )
    _type_cache.pop(key, None)  # invalidate the merged cache for this type
    logger.info("Registered runtime type '%s' from %s", key, source)


def clear_runtime_types() -> None:
    """Clear all runtime-registered types (for tests)."""
    _runtime_types.clear()


# Process-wide type resolution cache. Keyed by content type. Cleared via invalidate_type_cache().
_type_cache: Dict[str, Optional[TypeResolutionResult]] = {}


def resolve_type_definition_cached(content_type: str) -> Optional[TypeResolutionResult]:
    """Cached variant of `resolve_type_definition` using default roots.

    Resolution = core (``package/types``) base overlaid with the self-registered
    runtime types, then ``inherits`` parents — all handled inside
    :func:`resolve_type_definition`. This wrapper only adds a process-wide cache,
    keyed by content type, invalidated per-type on (re-)registration and wholesale
    via :func:`invalidate_type_cache` (e.g. on an admin reload). Tests that mutate
    type files should call ``invalidate_type_cache()`` in setup.
    """
    key = _normalize_content_type(content_type)
    if key in _type_cache:
        return _type_cache[key]
    res = resolve_type_definition(key)
    _type_cache[key] = res
    return res


def invalidate_type_cache() -> None:
    """Clear the cached type resolutions. Call after hot-reloading type files."""
    _type_cache.clear()


# Recognized index-hint names per Step 1.7. Hints not in this set are ignored
# by the indexer. New hint kinds (e.g. "vector", "fulltext") get added here as
# they're introduced.
INDEX_HINT_KINDS = frozenset({"lexical", "semantic", "geo", "numeric", "temporal"})


def get_field_index_hints(content_type: str) -> Dict[str, List[str]]:
    """Return per-field index hints from a type's ``context_schema``.

    Per Step 1.7 (context-schema-driven indexing), a type may declare which of
    its context fields participate in which kind of index by shaping its
    schema entries as::

        "context_schema": {
          "title":       { "index": ["lexical"] },
          "description": { "index": ["lexical", "semantic"] },
          "offers":      { "index": ["semantic"] },
          "location":    { "index": ["geo"] }
        }

    This helper returns ``{field_name: [hint, ...]}`` for every top-level field
    that declares an ``index`` array. Returns an empty dict when no hints are
    declared — callers should fall back to default indexing in that case.

    Hints not in :data:`INDEX_HINT_KINDS` are dropped silently so the indexer
    never sees an unknown kind. Inheritance is honoured because resolution goes
    through :func:`resolve_type_definition_cached`, which merges parents.
    """
    res = resolve_type_definition_cached(content_type)
    if res is None:
        return {}

    # `_load_folder_definition` keeps top-level type.json fields under
    # `definition["type"]` (only `ui`, `operations`, `relationships`, `preview`,
    # `behaviors` are promoted). `context_schema` stays nested.
    type_block = res.definition.get("type")
    schema = type_block.get("context_schema") if isinstance(type_block, dict) else None
    if not isinstance(schema, dict):
        return {}

    hints: Dict[str, List[str]] = {}
    for field_name, field_def in schema.items():
        if not isinstance(field_def, dict):
            continue
        raw = field_def.get("index")
        if not isinstance(raw, list):
            continue
        accepted = [h for h in raw if isinstance(h, str) and h in INDEX_HINT_KINDS]
        if accepted:
            hints[field_name] = accepted
    return hints


def resolve_operation(content_type: str, op_name: str) -> Optional[OperationSpec]:
    """Resolve an operation specification for a content type + op name.

    Walks `definition["operations"][op_name]` from the (cached) type
    resolution and normalizes it into an `OperationSpec`. Returns `None` if
    the type doesn't declare the operation.
    """
    if not op_name:
        return None

    res = resolve_type_definition_cached(content_type)
    if res is None:
        return None

    ops = res.definition.get("operations")
    if not isinstance(ops, dict):
        return None

    op = ops.get(op_name)
    if not isinstance(op, dict):
        return None

    enabled = bool(op.get("enabled", True))

    requires_grant = op.get("requires_grant")
    if not isinstance(requires_grant, str) or not requires_grant.strip():
        requires_grant = _OP_NAME_TO_GRANT_FLAG.get(op_name, "read")

    dispatch = op.get("dispatch")
    if not isinstance(dispatch, dict):
        dispatch = {"kind": "artifact_crud"} if op_name in _OP_NAME_TO_GRANT_FLAG else {}

    input_schema = op.get("input_schema") if isinstance(op.get("input_schema"), dict) else {}
    output_schema = op.get("output_schema") if isinstance(op.get("output_schema"), dict) else {}

    emits_raw = op.get("emits")
    emits: List[Dict[str, Any]] = []
    if isinstance(emits_raw, list):
        for entry in emits_raw:
            if isinstance(entry, dict) and isinstance(entry.get("event"), str):
                emits.append({
                    "event": entry["event"],
                    "phase": entry.get("phase", "after"),
                    "optional": bool(entry.get("optional", False)),
                })

    observe = op.get("observe") if isinstance(op.get("observe"), dict) else None
    audit = bool(op.get("audit", False))

    return OperationSpec(
        name=op_name,
        enabled=enabled,
        requires_grant=requires_grant,
        dispatch=dispatch,
        input_schema=input_schema,
        output_schema=output_schema,
        emits=emits,
        observe=observe,
        audit=audit,
    )


def list_available_content_types(*, roots: Optional[List[Path]] = None) -> List[str]:
    """List content type patterns available across roots (exact + wildcards)."""
    roots = roots if roots is not None else get_types_roots()
    content_types: List[str] = []
    seen: set[str] = set()

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for top in sorted([p for p in root.iterdir() if p.is_dir()]):
            for sub in sorted([p for p in top.iterdir() if p.is_dir()]):
                if sub.name.startswith("."):
                    continue
                if sub.name == "_wildcard":
                    content_type = f"{top.name}/*"
                else:
                    content_type = f"{top.name}/{sub.name}"
                if content_type not in seen:
                    seen.add(content_type)
                    content_types.append(content_type)

    return content_types


def list_resolved_type_definitions() -> List[Dict[str, Any]]:
    """Resolve every available content type (filesystem roots + runtime-registered)
    to its merged definition. This is the single RUNTIME source the frontend
    hydrates its content-type registry from — so Facet never compiles in (scans)
    server-owned type definitions at build time.

    Mantle is a blind relay: each entry carries the merged ``definition`` (incl.
    the ``ui`` block) verbatim. The viewer pointer (``ui.resource_uri`` /
    ``ui.resource_server``) is declared by the OWNING SERVER in its own
    ``type.json`` — Mantle never infers it.

    Sources are the union of the core filesystem types and the self-registered
    runtime types; each is resolved through the same merge so an overlay (e.g. a
    server viewer on a core type) appears folded into the one entry.
    """
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    content_types = set(list_available_content_types()) | set(_runtime_types.keys())
    for ct in sorted(content_types):
        res = resolve_type_definition_cached(ct)
        if res is None or res.content_type in seen:
            continue
        seen.add(res.content_type)
        out.append({
            "content_type": res.content_type,
            "definition": res.definition,
            "validation_errors": res.validation_errors,
        })
    return out
