"""Export a collection to the declarative seed format the loader consumes.

The inverse of ``loader.seed_from_artifacts``: walk a collection's contents
(artifacts + nested sub-collections + containment edges) and emit the same
``namespace/slug/content_type/context/content/edges`` seed dicts the loader reads
back. This is the basis for collection import/export and a future "seed bank"
(pull a collection from another Agience instance).

CONTENT ONLY — grants are deliberately NOT exported. Grants are per-deployment
access records keyed by user ids that won't exist on another instance; the
importing instance applies its own grants via the user/admin grant seeds.

Read-only and side-effect-free. Exposed as the ``export`` operation on the
collection / workspace types (dispatched to the native handler ``collection.export``)
so it is invokable by agents and workflows, not a built-in CLI feature.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from mantle.db.store import Database

from mantle.db.backend import get_collection_by_id, list_collection_artifacts
from mantle.entities.collection import COLLECTION_CONTENT_TYPE

_WORKSPACE_CONTENT_TYPE = "application/vnd.agience.workspace+json"
# ⛔ `_MAX_DEPTH = 25` as a TRUNCATION removed 2026-07-30. `seen` is the cycle guard (the comment
# here said so itself), so the depth test only cut the walk short — and a truncated export was
# returned as a SUCCESSFUL one. Data loss presented as a completed export is the worst shape a
# bound can take. What remains is an operational guard on Python's own call stack, and it RAISES:
# an export either contains everything or fails loudly.
_RECURSION_CEILING = 500  # stack depth, not a claim about how deep containment goes


def _is_container(content_type: Optional[str]) -> bool:
    return content_type in (COLLECTION_CONTENT_TYPE, _WORKSPACE_CONTENT_TYPE)


def _slugify(name: Optional[str], root_id: str) -> str:
    """Stable, readable slug for an exported artifact: sanitized name + a short
    id suffix for uniqueness. Deterministic for a given source artifact."""
    base = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    short = root_id.replace("-", "")[:8]
    return f"{base}-{short}" if base else f"artifact-{short}"


def _parse_context(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _seed_dict(
    namespace: str,
    slug: str,
    *,
    content_type: Optional[str],
    name: str,
    description: str,
    context: dict,
    content: Any,
) -> dict:
    seed: dict = {
        "namespace": namespace,
        "slug": slug,
        "content_type": content_type,
        "name": name or "",
    }
    if description:
        seed["description"] = description
    if context:
        seed["context"] = context
    if content:
        seed["content"] = content
    return seed


def export_collection(
    store_db: Database,
    collection_id: str,
    *,
    namespace: str = "export",
) -> list[dict]:
    """Return the seed-artifact dicts representing ``collection_id`` and everything
    it contains (recursively). The first entry is the root collection; each other
    entry carries a ``contained_by`` edge to its parent. No grants."""
    root = get_collection_by_id(store_db, collection_id)
    if root is None:
        return []

    seeds: list[dict] = []
    slug_by_root: dict[str, str] = {}
    seen: set[str] = set()

    root_slug = _slugify(getattr(root, "name", None), collection_id)
    slug_by_root[collection_id] = root_slug
    seeds.append(_seed_dict(
        namespace, root_slug,
        content_type=getattr(root, "content_type", COLLECTION_CONTENT_TYPE),
        name=getattr(root, "name", "") or "",
        description=getattr(root, "description", "") or "",
        context=_parse_context(getattr(root, "context", None)),
        content=getattr(root, "content", None),
    ))

    def _walk(container_id: str, parent_slug: str, depth: int) -> None:
        if container_id in seen:
            return
        if depth > _RECURSION_CEILING:
            raise RuntimeError(
                f"seed export exceeded {_RECURSION_CEILING} levels of containment at "
                f"{container_id!r}; refusing to return a truncated export"
            )
        seen.add(container_id)
        for index, member in enumerate(list_collection_artifacts(store_db, container_id) or []):
            root_id = str(member.get("root_id") or member.get("id") or "").strip()
            if not root_id:
                continue
            slug = slug_by_root.get(root_id) or _slugify(member.get("name"), root_id)
            slug_by_root[root_id] = slug
            seed = _seed_dict(
                namespace, slug,
                content_type=member.get("content_type"),
                name=member.get("name") or "",
                description=member.get("description") or "",
                context=_parse_context(member.get("context")),
                content=member.get("content"),
            )
            seed["edges"] = [{
                "rel": "contained_by",
                "to": f"{namespace}/{parent_slug}",
                "origin": True,
                "order_key": member.get("order_key") or f"a{index}",
            }]
            seeds.append(seed)
            if _is_container(member.get("content_type")):
                _walk(root_id, slug, depth + 1)

    _walk(collection_id, root_slug, 0)

    # Second pass: export TYPED relationship edges (rel != contained_by) between
    # exported artifacts, so compositions round-trip — e.g. an operator's
    # `authorizer` edge, an authorizer's `client_secret`/`refresh_token` edges.
    # Targets outside the exported set are skipped (they can't be referenced).
    seed_by_slug = {s["slug"]: s for s in seeds}
    for src_root, src_slug in slug_by_root.items():
        seed = seed_by_slug.get(src_slug)
        if seed is None:
            continue
        for member in list_collection_artifacts(store_db, src_root) or []:
            rel = member.get("relationship")
            if not rel or rel == "contained_by":
                continue
            target_root = str(member.get("root_id") or member.get("id") or "").strip()
            target_slug = slug_by_root.get(target_root)
            if not target_slug:
                continue
            edge = {"rel": rel, "to": f"{namespace}/{target_slug}"}
            seed.setdefault("edges", []).append(edge)

    return seeds


# ---------------------------------------------------------------------------
# Native operation handler — wired as `collection.export` (dispatch kind native)
# ---------------------------------------------------------------------------

async def dispatch_export(artifact: dict, body: dict, ctx) -> dict:
    """Native handler for the ``export`` operation on collection/workspace types.

    Returns ``{namespace, count, seeds: [...]}`` — the declarative representation
    of the collection tree, ready to be written to ``package/seeds/<namespace>``
    or shipped to another instance. Grants are not included."""
    collection_id = artifact.get("root_id") or artifact.get("_key") or artifact.get("id")
    if not collection_id:
        raise ValueError("export: cannot resolve collection id from dispatch target")
    namespace = (body or {}).get("namespace") or "export"
    seeds = export_collection(ctx.store_db, str(collection_id), namespace=namespace)
    return {"namespace": namespace, "count": len(seeds), "seeds": seeds}
