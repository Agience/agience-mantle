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
    # `src/mantle/services/types_service.py` -> parents[3] is the agience-mantle repo root.
    return Path(__file__).resolve().parents[3]


# ⭐ MANTLE DOES NOT HOLD A TYPE REGISTRY. CRYSTAL DOES, and says so in its own words:
# "Types and personas are the gateway's own state — personas self-register their type definitions
# and endpoint here (POST /register); Mantle never sees them" (`crystal/main.py`). What survives
# here is one narrow filesystem lookup, kept because `ingest_runner_service` asks it which handler
# extracts text for a content type.
#
# ⛔ A RUNTIME REGISTRY, A LAZY LOADER, CACHES AND BULK LISTERS ALL LIVED HERE AND NONE OF THEM HAD
# A PRODUCTION CALLER. Removed 2026-09-14. The registration and loader entry points were reached
# only from this module's own tests, `main.py` wired neither, and the persistence their docstrings
# promised was never written — so the registry was empty on every node, the loader was never
# installed, and resolution answered `None` for every type.
# The docstrings described a Mantle that does not exist, and cost a session's work to disprove.
#
# ⚠ THE FILESYSTEM BASE IS EMPTY IN A DEFAULT DEPLOYMENT TOO. Resolution reads only the roots named
# by `AGIENCE_TYPES_PATHS`, which no node sets, so `resolve_capability_target` returns `None` and
# its caller takes the fallback branch. That is the current behaviour, unchanged by this cut.




def _merge_ordered_roots() -> List[Path]:
    """FILESYSTEM type roots in MERGE precedence (LOWEST first). Deep-merge applies
    LATER over EARLIER.

    There is exactly one filesystem source — ``AGIENCE_TYPES_PATHS`` — which is unset by
    default, making the filesystem base empty in the default deployment. Server-owned types
    are not here and never were: a persona registers what it owns with the gateway.
    """
    roots: List[Path] = []
    seen: set[Path] = set()

    def add_root(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not (resolved.exists() and resolved.is_dir()):
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
            add_root(p)
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

    Root ordering from ``_merge_ordered_roots()`` defines the priority. There is one
    filesystem source, ``AGIENCE_TYPES_PATHS``, unset by default.
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

    # Tool handlers carry `tool`; builtin handlers carry `id`. `tool` is checked first,
    # so it wins when a handler carries both.
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

    Filesystem roots (``AGIENCE_TYPES_PATHS``, base-first, deep-merged), then layered over
    any ``inherits`` parents (child wins).

    Matching: exact ``top/subtype`` else category wildcard ``top/*``
    (``_wildcard`` folder). The wildcard fallback is whole-resolution, never
    per-layer — otherwise a type with an exact def would be polluted by a `*`.
    """
    content_type = _normalize_content_type(content_type)
    if not content_type:
        return None

    merge_roots = roots if roots is not None else _merge_ordered_roots()

    # Own-definition: filesystem layers, base-first, deep-merged.
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

    Returns a target name suitable for runtime invocation: a builtin id for builtins
    (e.g. `text.extract_text`), a tool name for remote handlers.
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






# ---------------------------------------------------------------------------
# Operations schema (Phase 0 — Enterprise Eventing refactor)
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Runtime type registration (MCP server discovery)
# ---------------------------------------------------------------------------












# Process-wide type resolution cache. Keyed by content type. Cleared via invalidate_type_cache().






# Recognized index-hint names per Step 1.7. Hints not in this set are ignored
# by the indexer. New hint kinds (e.g. "vector", "fulltext") get added here as
# they're introduced.








