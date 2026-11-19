"""The title an artifact is shown by when its writer gives none.

A new empty artifact with no title is named after its type and numbered against the titles beside
it ("Untitled Markdown", "Untitled Markdown 2", ...), and a new workspace against the person's
other workspaces, so a list of new ones can be told apart. A save that blanks the title keeps a
name: the first line of the content, else the title it had. A client shows the title it reads.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Optional

from mantle.entities.artifact import WORKSPACE_CONTENT_TYPE

#: The name a workspace gets when it is created without one.
DEFAULT_WORKSPACE_NAME = "New workspace"
#: What a title derived from content is cut to.
TITLE_FROM_CONTENT_MAX = 80


def context_dict(context: Any) -> Optional[Dict[str, Any]]:
    """A context as a dict, or None when it is not a JSON object."""
    if isinstance(context, dict):
        return context
    if not context:
        return {}
    try:
        parsed = json.loads(context)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def shown_title(doc: Dict[str, Any]) -> str:
    """The title a document is shown by: its context's title, else its filename, else `name`."""
    ctx = context_dict(doc.get("context")) or {}
    return str(ctx.get("title") or ctx.get("filename") or doc.get("name") or "").strip()


def base_title(content_type: Optional[str]) -> str:
    """The unnumbered title of a new artifact of this type."""
    if content_type == WORKSPACE_CONTENT_TYPE:
        return DEFAULT_WORKSPACE_NAME
    label = ""
    if content_type:
        try:
            from mantle.services.types_service import resolve_type_definition
            resolved = resolve_type_definition(content_type)
            ui = (resolved.definition.get("ui") if resolved else None) or {}
            label = str(ui.get("label") or "").strip()
        except Exception:  # noqa: BLE001 - a missing type is named generically
            label = ""
    return f"Untitled {label or 'Artifact'}"


def next_free(base: str, existing: Iterable[str]) -> str:
    """`base` if no title in `existing` takes it, else `base 2`, `base 3`, ...: the first free one."""
    pattern = re.compile(rf"^{re.escape(base)}(?: (\d+))?$", re.IGNORECASE)
    taken = set()
    for title in existing:
        m = pattern.match(str(title or "").strip())
        if m:
            taken.add(int(m.group(1)) if m.group(1) else 1)
    n = 1
    while n in taken:
        n += 1
    return base if n == 1 else f"{base} {n}"


def needs_title(context_str: Optional[str], name: Optional[str], content: Optional[str]) -> bool:
    """A create is titled here when it is empty and its writer named it nowhere."""
    if (name or "").strip() or (content or "").strip():
        return False
    ctx = context_dict(context_str)
    if ctx is None:
        return False
    return not str(ctx.get("title") or ctx.get("filename") or "").strip()


def with_title(context_str: Optional[str], title: str) -> str:
    ctx = dict(context_dict(context_str) or {})
    ctx["title"] = title
    return json.dumps(ctx)


def title_from_content(content: Any, previous: str) -> str:
    """The first line of the content, headings unmarked, else `previous`, else "Untitled"."""
    text = content if isinstance(content, str) else ""
    for line in text.splitlines():
        line = re.sub(r"^#+\s*", "", line).strip()
        if line:
            return line[:TITLE_FROM_CONTENT_MAX]
    return previous.strip() or "Untitled"


def title_on_save(context_str: Optional[str], content: Optional[str], stored: Dict[str, Any]) -> Optional[str]:
    """`context_str` with a blank `title` replaced by one derived from the content (the one sent,
    else the stored one) or the title it had. Any other context is returned as it came."""
    ctx = context_dict(context_str)
    if not ctx or "title" not in ctx or str(ctx.get("title") or "").strip():
        return context_str
    body = content if content is not None else stored.get("content")
    ctx = dict(ctx)
    ctx["title"] = title_from_content(body, shown_title(stored))
    return json.dumps(ctx)


def name_as_shown(doc: Any) -> Any:
    """Set an outgoing document's `name` to the title it is shown by, in place, so a client that
    lists by `name` (a workspace or collection picker) shows the same title as one that reads the
    context. A document with no title keeps the `name` it has."""
    if isinstance(doc, dict):
        title = shown_title(doc)
        if title:
            doc["name"] = title
    return doc
