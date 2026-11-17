"""What a content type says gives its artifacts their title, description and tags.

A type's `type.json` may carry an ``offer`` block beside its ``describer``:

    "offer": {
      "title":       ["title", "context.title", "name", "lemmas.0"],
      "description": ["description", "context.description"],
      "tags":        ["lemmas"]
    }

Each entry is a list of paths into the artifact. A path's first segment is a top-level field;
``context.<key>`` reads the artifact's structured context; a number indexes a list. ``title`` and
``description`` take the first path that holds a non-empty value; ``tags`` takes every value of
every path. A type that declares a key replaces the inherited list for that key; a key it leaves
out comes from the types it ``inherits``, then its major type's wildcard (``text/*``), then
:data:`DEFAULT_OFFER`.

Crystal holds the type registry. It publishes each persona's definitions here through
``PUT /type-definitions/{server}``, and each one is stored as a system-owned
``application/vnd.agience.type+json`` artifact, so a restart reloads them from the store. Only
system-owned definitions are read: Mantle's create endpoint is label-blind, so a definition
any user wrote would otherwise decide how everyone's artifacts are indexed (the boundary
`services/issuers.py` draws for issuers).
"""
from __future__ import annotations

import copy
import json
import logging
import threading
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mantle.services.bootstrap_types import TYPE_DEFINITION_CONTENT_TYPE

logger = logging.getLogger(__name__)

OFFER_KEYS: Tuple[str, ...] = ("title", "description", "tags")

#: The offer of a type that declares none: what the indexer read before types could say.
DEFAULT_OFFER: Dict[str, Tuple[str, ...]] = {
    "title": ("title", "context.title", "name"),
    "description": ("description", "context.description"),
    "tags": (),
}

_lock = threading.Lock()
#: content type -> its published definition (the whole type.json).
_definitions: Dict[str, dict] = {}


def _normalize(content_type: Any) -> str:
    ct = str(content_type or "").strip().lower()
    return ct.split(";", 1)[0].strip()


def validate_offer(offer: Any) -> List[str]:
    """The problems with an ``offer`` block; empty when it is well formed."""
    if offer is None:
        return []
    if not isinstance(offer, dict):
        return ["offer must be an object"]
    problems = []
    for key, paths in offer.items():
        if key not in OFFER_KEYS:
            problems.append(f"offer.{key} is not one of {', '.join(OFFER_KEYS)}")
        elif not isinstance(paths, list) or not all(isinstance(p, str) and p.strip() for p in paths):
            problems.append(f"offer.{key} must be a list of paths")
    return problems


#: What a type may declare its artifacts do when a create reaches them over an edge
#: (``on_create: {"does": <verb>}``); `services.on_create` runs the verb.
ON_CREATE_VERBS: Tuple[str, ...] = ("number", "create")

#: What an ``on_update`` stamp writes into its field.
STAMPS: Tuple[str, ...] = ("caller", "time")


def validate_on_create(declared: Any) -> List[str]:
    """The problems with a type's own ``on_create`` block; empty when it is well formed."""
    if declared is None:
        return []
    if not isinstance(declared, dict):
        return ["on_create must be an object"]
    problems = []
    unknown = set(declared) - {"does"}
    if unknown:
        problems.append(f"on_create has unknown keys {', '.join(sorted(unknown))}")
    if declared.get("does") not in ON_CREATE_VERBS:
        problems.append(f"on_create.does must be one of {', '.join(ON_CREATE_VERBS)}")
    return problems


def validate_field_rules(fields: Any) -> List[str]:
    """The problems with a ``fields`` block; empty when it is well formed.

    A field may carry ``on_create: {"from_edge": <label>, "key": <field>?}``: on create its value
    comes from the artifact the creating collection's ``from_edge`` edge points to, as that
    artifact's type declares, and ``key``, when named, receives a second value from it. A field
    may carry ``on_update: {"stamp": "caller" | "time", "when": <field>}``: an update that changes
    ``when`` sets it to the caller or the time."""
    if fields is None:
        return []
    if not isinstance(fields, dict):
        return ["fields must be an object"]
    problems = []
    for name, rule in fields.items():
        if not isinstance(rule, dict):
            problems.append(f"fields.{name} must be an object")
            continue
        problems += _stamp_problems(name, rule.get("on_update"))
        on_create = rule.get("on_create")
        if on_create is None:
            continue
        where = f"fields.{name}.on_create"
        if not isinstance(on_create, dict):
            problems.append(f"{where} must be an object")
            continue
        unknown = set(on_create) - {"from_edge", "key"}
        if unknown:
            problems.append(f"{where} has unknown keys {', '.join(sorted(unknown))}")
        edge = on_create.get("from_edge")
        if not isinstance(edge, str) or not edge.strip():
            problems.append(f"{where}.from_edge must name an edge")
        key = on_create.get("key")
        if key is not None and (not isinstance(key, str) or not key.strip() or key == name):
            problems.append(f"{where}.key must name another field")
    return problems


def _stamp_problems(name: str, stamp: Any) -> List[str]:
    if stamp is None:
        return []
    where = f"fields.{name}.on_update"
    if not isinstance(stamp, dict):
        return [f"{where} must be an object"]
    problems = []
    unknown = set(stamp) - {"stamp", "when"}
    if unknown:
        problems.append(f"{where} has unknown keys {', '.join(sorted(unknown))}")
    if stamp.get("stamp") not in STAMPS:
        problems.append(f"{where}.stamp must be one of {', '.join(STAMPS)}")
    when = stamp.get("when")
    if not isinstance(when, str) or not when.strip() or when == name:
        problems.append(f"{where}.when must name another field")
    return problems


# ── reading an artifact ──────────────────────────────────────────────────────────────────────

def _context_dict(artifact: Any) -> dict:
    ctx = getattr(artifact, "context", None)
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx) if ctx else {}
        except ValueError:
            return {}          # prose in `context` is provenance, not an offer (chunking.py)
    return ctx if isinstance(ctx, dict) else {}


def read_path(artifact: Any, path: str) -> Any:
    """The value at ``path`` in ``artifact``, or ``None``."""
    parts = [p for p in path.split(".") if p]
    if not parts:
        return None
    if parts[0] == "context":
        value: Any = _context_dict(artifact)
        parts = parts[1:]
    else:
        value = getattr(artifact, parts[0], None)
        parts = parts[1:]
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, (list, tuple)):
            # An index (`0`) or a slice (`1:`), as a describer's `field_source` path reads them
            # (`agience_chorus/sage/offer.py::_resolve_path`).
            try:
                if ":" in part:
                    a, _, b = part.partition(":")
                    value = list(value[(int(a) if a else None):(int(b) if b else None)])
                else:
                    value = value[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return value


def _strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for v in value:
            out.extend(_strings(v))
        return out
    text = str(value).strip()
    return [text] if text else []


def first_value(artifact: Any, paths: Iterable[str]) -> str:
    """The first non-empty value among ``paths``."""
    for path in paths:
        values = _strings(read_path(artifact, path))
        if values:
            return values[0]
    return ""


def all_values(artifact: Any, paths: Iterable[str]) -> List[str]:
    """Every non-empty value of every path, in order."""
    out: List[str] = []
    for path in paths:
        out.extend(_strings(read_path(artifact, path)))
    return out


# ── resolving a type's offer ─────────────────────────────────────────────────────────────────

def _lineage(content_type: str) -> List[str]:
    """``content_type``, the types it inherits (depth first), its major wildcard, then the root
    ``*/*`` (Artifact). An Agience or ``+json`` data type skips ``application/*`` (File), as Crystal
    and the web app resolve it."""
    seen: List[str] = []

    def walk(ct: str) -> None:
        if not ct or ct in seen:
            return
        seen.append(ct)
        for parent in (_definitions.get(ct) or {}).get("inherits") or []:
            walk(_normalize(parent))

    walk(content_type)
    if "/" in content_type:
        data = content_type.startswith("application/vnd.agience.") or content_type.endswith("+json")
        if not data:
            walk(content_type.split("/", 1)[0] + "/*")
    walk("*/*")
    return seen


def offer_of(content_type: Any) -> Dict[str, Tuple[str, ...]]:
    """The resolved offer for ``content_type``: each key from the nearest type declaring it."""
    ct = _normalize(content_type)
    resolved: Dict[str, Tuple[str, ...]] = {}
    with _lock:
        for link in _lineage(ct):
            declared = (_definitions.get(link) or {}).get("offer")
            if not isinstance(declared, dict) or validate_offer(declared):
                continue
            for key in OFFER_KEYS:
                if key not in resolved and key in declared:
                    resolved[key] = tuple(declared[key])
    for key in OFFER_KEYS:
        resolved.setdefault(key, DEFAULT_OFFER[key])
    return resolved


def _field_rules(content_type: Any, kind: str) -> Dict[str, dict]:
    """``{field: rule}`` of ``kind`` for ``content_type``: each field's rule from the nearest type
    in its lineage that declares the field. Empty for a type with none, the common case."""
    ct = _normalize(content_type)
    if not ct:
        return {}
    resolved: Dict[str, Any] = {}
    with _lock:
        if not _definitions:
            return {}
        for link in _lineage(ct):
            declared = (_definitions.get(link) or {}).get("fields")
            if not isinstance(declared, dict) or validate_field_rules(declared):
                continue
            for name, rule in declared.items():
                if name not in resolved:
                    resolved[name] = rule.get(kind)
    return {name: dict(rule) for name, rule in resolved.items() if rule}


def on_create_rules(content_type: Any) -> Dict[str, dict]:
    """``{field: on_create rule}`` for ``content_type``."""
    return _field_rules(content_type, "on_create")


def on_update_rules(content_type: Any) -> Dict[str, dict]:
    """``{field: on_update stamp}`` for ``content_type``."""
    return _field_rules(content_type, "on_update")


def secret_fields(content_type: Any) -> set:
    """The fields ``content_type`` marks ``"secret": true``, from the nearest type in its lineage
    that declares each field."""
    ct = _normalize(content_type)
    if not ct:
        return set()
    resolved: Dict[str, bool] = {}
    with _lock:
        for link in _lineage(ct):
            declared = (_definitions.get(link) or {}).get("fields")
            if not isinstance(declared, dict):
                continue
            for name, rule in declared.items():
                if name not in resolved and isinstance(rule, dict):
                    resolved[name] = rule.get("secret") is True
    return {name for name, secret in resolved.items() if secret}


def defaults(content_type: Any) -> Dict[str, Any]:
    """``{field: default}`` from ``content_type``'s ``context_schema``: each field's ``default``
    from the nearest type in its lineage that declares the field. A type is the only place a
    field's default lives."""
    ct = _normalize(content_type)
    if not ct:
        return {}
    resolved: Dict[str, Any] = {}
    with _lock:
        for link in _lineage(ct):
            schema = (_definitions.get(link) or {}).get("context_schema")
            if not isinstance(schema, dict):
                continue
            for name, rule in schema.items():
                if name not in resolved and isinstance(rule, dict):
                    resolved[name] = rule
    return {name: copy.deepcopy(rule["default"]) for name, rule in resolved.items()
            if "default" in rule}


def on_create_does(content_type: Any) -> Optional[str]:
    """The verb ``content_type`` declares for a create that reaches one of its artifacts over an
    edge, from the nearest type in its lineage that declares one; ``None`` when none does."""
    ct = _normalize(content_type)
    if not ct:
        return None
    with _lock:
        for link in _lineage(ct):
            declared = (_definitions.get(link) or {}).get("on_create")
            if declared is not None and not validate_on_create(declared):
                return declared["does"]
    return None


def set_definitions(definitions: Dict[str, dict]) -> None:
    """Replace the in-memory definitions (startup load and tests)."""
    with _lock:
        _definitions.clear()
        _definitions.update({_normalize(k): v for k, v in definitions.items() if _normalize(k)})


# ── the store ────────────────────────────────────────────────────────────────────────────────

def _system_principal_id() -> Optional[str]:
    try:
        from mantle.services.peer_signing import get_system_principal_id
        return get_system_principal_id()
    except Exception:
        return None


def _stored(db: Any, system_id: str) -> List[Tuple[Any, dict]]:
    from mantle.db import backend as db_store
    out = []
    for art in db_store.list_live_artifacts_by_context_content_type(
            db, TYPE_DEFINITION_CONTENT_TYPE, created_by=system_id):
        out.append((art, _context_dict(art)))
    return out


#: The source Crystal publishes its own builtin types under. A persona that registers the same
#: content type takes precedence, as it does in Crystal's registry (`TypeRegistry.resolve`).
BUILTIN_SOURCE = "crystal"


def load_definitions(db: Any) -> int:
    """Load every published definition from the store. Returns how many types are defined."""
    system_id = _system_principal_id()
    if db is None or not system_id:
        logger.warning("type offers: system principal unknown; every type reads the default offer")
        set_definitions({})
        return 0
    found: Dict[str, dict] = {}
    from_builtin: set = set()
    for _art, ctx in _stored(db, system_id):
        target = _normalize(ctx.get("target_content_type"))
        if not target or not isinstance(ctx.get("definition"), dict):
            continue
        builtin = ctx.get("source") == BUILTIN_SOURCE
        if target in found and builtin and target not in from_builtin:
            continue                       # a persona's definition outranks the builtin
        found[target] = ctx["definition"]
        if builtin:
            from_builtin.add(target)
        else:
            from_builtin.discard(target)
    set_definitions(found)
    logger.info("type offers: %d type definition(s) loaded", len(found))
    return len(found)


def publish(db: Any, server: str, definitions: List[dict]) -> Dict[str, Any]:
    """Store ``server``'s definitions, replacing what it published before.

    A definition that is unchanged is left alone. One the server no longer publishes is archived.
    Returns ``{"stored", "unchanged", "archived", "refused", "offer_changed"}``, where
    ``offer_changed`` lists the content types whose resolved offer moved, so their artifacts can
    be indexed again.
    """
    from mantle.db import backend as db_store
    from mantle.entities.artifact import Artifact

    system_id = _system_principal_id()
    if not system_id:
        from mantle.services.acting_principal import SystemPrincipalUnavailable
        raise SystemPrincipalUnavailable("system principal unavailable — cannot own type artifacts")

    wanted: Dict[str, dict] = {}
    refused: Dict[str, List[str]] = {}
    for definition in definitions:
        if not isinstance(definition, dict):
            continue
        target = _normalize(definition.get("content_type"))
        if not target:
            continue
        problems = (validate_offer(definition.get("offer"))
                    + validate_field_rules(definition.get("fields"))
                    + validate_on_create(definition.get("on_create")))
        if problems:
            refused[target] = problems
            continue
        wanted[target] = definition

    # Keyed by (source, type): two servers may define one type, and each replaces only its own.
    stored: Dict[Tuple[str, str], Tuple[Any, dict]] = {}
    for art, ctx in _stored(db, system_id):
        target = _normalize(ctx.get("target_content_type"))
        if target:
            stored[(str(ctx.get("source") or ""), target)] = (art, ctx)

    # Every known type, not only this server's: a type inherits keys it does not declare, so a
    # change to `text/*` moves `text/markdown` too. A wildcard in the result stands for every
    # type under it, declared or not.
    before = {ct: offer_of(ct) for ct in set(wanted) | {t for _s, t in stored}}

    counts = {"stored": 0, "unchanged": 0, "archived": 0}
    for target, definition in wanted.items():
        ctx = {"content_type": TYPE_DEFINITION_CONTENT_TYPE, "target_content_type": target,
               "definition": definition, "source": server}
        existing = stored.get((server, target))
        if existing is not None:
            art, old = existing
            if old.get("definition") == definition:
                counts["unchanged"] += 1
                continue
            art.context = json.dumps(ctx)
            art.modified_by = system_id
            db_store.update_artifact(db, art)
        else:
            db_store.create_artifact(db, Artifact(
                id=str(uuid.uuid4()), collection_id="", state="live",
                created_by=system_id, modified_by=system_id, context=json.dumps(ctx),
                content_type=TYPE_DEFINITION_CONTENT_TYPE, name=target,
            ))
        counts["stored"] += 1

    for (source, target), (art, _ctx) in stored.items():
        if source == server and target not in wanted:
            # Written directly: `archive_artifact` asks for a grant on a parent collection, and a
            # type artifact has none.
            art.state = "archived"
            art.modified_by = system_id
            db_store.update_artifact(db, art)
            counts["archived"] += 1

    load_definitions(db)
    changed = sorted(ct for ct, offer in before.items() if offer_of(ct) != offer)
    logger.info("type offers: %s published %d, unchanged %d, archived %d, refused %d",
                server, counts["stored"], counts["unchanged"], counts["archived"], len(refused))
    return {**counts, "refused": refused, "offer_changed": changed}


__all__ = [
    "BUILTIN_SOURCE", "DEFAULT_OFFER", "OFFER_KEYS", "ON_CREATE_VERBS", "STAMPS",
    "all_values", "first_value", "load_definitions", "offer_of", "on_create_does",
    "on_create_rules", "on_update_rules", "publish", "read_path", "set_definitions",
    "validate_field_rules", "validate_offer", "validate_on_create",
]
