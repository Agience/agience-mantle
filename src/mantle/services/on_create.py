"""Field values a type sets on create from an artifact its collection is edged to, and fields an
update stamps.

A type definition marks a field:

    "fields": {"number": {"on_create": {"from_edge": "numbered_by", "key": "key"}}}

Creating an artifact of that type (or of a type that inherits it) finds the edged artifact: the
target of a ``from_edge`` edge out of the collection the artifact is created in, else out of the
collections containing that one, nearest first. The edged artifact's own type says what the create
does there, in its definition's ``on_create: {"does": <verb>}``. The verbs:

- ``number``: the edged artifact's context holds ``prefix``, ``width`` and ``last``. The create
  takes ``last + 1``, writes it back and stamps the field, and the ``key`` field when the rule
  names one (``prefix`` + ``-`` + the number padded to ``width``). Two creates never take one
  number, and a number is never reused.
- ``create``: the edged artifact's context is a spec, ``{"creates": <content type>, "in":
  <collection id>, "link": <edge label>, "fields": {...}}``. The create makes a new artifact of
  type ``creates`` in ``in``, with ``fields`` as its context and the edged artifact as its
  creator, stamps the field with its id, and links it from the new artifact under ``link``. No
  grant is minted for it: who may change it is whoever holds update on ``in``. The spec's owner
  must hold Create on ``in``, so a spec cannot place artifacts in a collection its owner cannot.
  ``conducts``, when given, lists the actions a grant on the new artifact carries to the made one
  (``["read"]``: whoever reads the item reads its part, and nothing else in ``in``). The spec is
  readable configuration, so a field the made type marks ``"secret": true`` refuses the create.

Everything is written in the create's transaction, so a refusal writes nothing.

Refusals: no edged artifact, two at the nearest level, one first written on another node (its
values are allocated on its home node), one whose type declares no verb, or a malformed spec, is a
409; a creator without Invoke on the edged artifact is a 403.

Fields with an ``on_create`` rule keep their value: an update may leave them out (the stored value
carries forward) or repeat them, and a different value is refused (`guard_update`). ``adopt`` is
the one writer that sets them on existing artifacts, for an import.

A field may also carry ``on_update: {"stamp": "caller" | "time", "when": <field>}``. An update that
changes ``when`` sets the field to the caller (``modified_by``) or the time; any other write keeps
the stored value, and a value the caller supplies is overwritten.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class CreateRefused(HTTPException):
    """A create or update the field rules refuse. Raised as the HTTP answer it is."""


#: The caller a create is made for, set around a request's create; a create made without one
#: is checked as its `created_by`.
_creator: contextvars.ContextVar[Any] = contextvars.ContextVar("on_create_creator", default=None)
#: Set while `adopt` writes, so the update guard lets it fill the fields it guards.
_adopting: contextvars.ContextVar[bool] = contextvars.ContextVar("on_create_adopting",
                                                                 default=False)


@contextlib.contextmanager
def creating_as(auth: Any) -> Iterator[None]:
    """Check Invoke on an edged artifact as ``auth`` for the creates made inside this block."""
    token = _creator.set(auth)
    try:
        yield
    finally:
        _creator.reset(token)


# ── context helpers ──────────────────────────────────────────────────────────────────────────

def _ctx_of(value: Any) -> Tuple[Dict[str, Any], bool]:
    """(context as a dict, whether it was a JSON string)."""
    if isinstance(value, dict):
        return dict(value), False
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}, True
        return (parsed if isinstance(parsed, dict) else {}), True
    return {}, isinstance(value, str)


def _put_ctx(entity: Any, ctx: Dict[str, Any], as_string: bool) -> None:
    entity.context = json.dumps(ctx) if as_string else ctx


def _content_type(entity: Any, ctx: Dict[str, Any]) -> str:
    return str(getattr(entity, "content_type", None) or ctx.get("content_type") or "")


def _type_of(entity: Any) -> str:
    ctx, _ = _ctx_of(getattr(entity, "context", None))
    return _content_type(entity, ctx)


def rules_for(entity: Any) -> Dict[str, dict]:
    """The ``on_create`` rules of ``entity``'s type; empty for nearly every type."""
    from mantle.services import type_offers
    return type_offers.on_create_rules(_type_of(entity))


def stamps_for(entity: Any) -> Dict[str, dict]:
    """The ``on_update`` stamps of ``entity``'s type; empty for nearly every type."""
    from mantle.services import type_offers
    return type_offers.on_update_rules(_type_of(entity))


def guarded_fields(rules: Dict[str, dict]) -> List[str]:
    """Every field the rules set: each ruled field and the key field it names."""
    out: List[str] = []
    for name, rule in rules.items():
        out.append(name)
        if rule.get("key"):
            out.append(rule["key"])
    return out


def _stored_context(db: Any, artifact_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """(exists, plaintext context or None when it cannot be opened here)."""
    from mantle.db import doc_boundary
    prior = db.artifacts.get_artifact(artifact_id) if artifact_id else None
    if not prior:
        return False, None
    opened = dict(prior)
    try:
        doc_boundary.decrypt_artifact_content(opened, strict=True)
    except Exception:
        return True, None
    ctx, _ = _ctx_of(opened.get("context"))
    return True, ctx


def _same(a: Any, b: Any) -> bool:
    return str(a) == str(b)


def _blank(value: Any) -> bool:
    return value in (None, "")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── finding the edged artifact ───────────────────────────────────────────────────────────────

def _targets(db: Any, node_id: str, edge: str) -> List[str]:
    """The roots ``node_id``'s ``edge`` edges point to, under the stored label and the plain one."""
    from mantle.db import lattice_api
    found: List[str] = []
    for label in dict.fromkeys((lattice_api._stored_label(edge), edge)):
        for e in db.graph.edges_of(node_id, label=label, direction="out") or []:
            dst = e.get("dst")
            if dst and dst not in found:
                found.append(dst)
    return found


def provider_for(db: Any, collection_id: Optional[str], edge: str) -> str:
    """The edged artifact for an artifact created in ``collection_id``: the ``edge`` target of
    that collection, else of its containing collections, nearest first.

    Where the edge is read from is decided here and nowhere else."""
    from mantle.db import lattice_api
    current, seen = collection_id, set()
    while current and current not in seen:
        seen.add(current)
        targets = _targets(db, current, edge)
        if len(targets) > 1:
            raise CreateRefused(409, f"{current} has {len(targets)} {edge} edges; "
                                     "a collection names one")
        if targets:
            return targets[0]
        parent = lattice_api.get_origin_parent(db, current)
        current = parent[0] if parent else None
    raise CreateRefused(409, f"no {edge} artifact for this collection; nothing was created")


def _creator_auth(entity: Any) -> Any:
    auth = _creator.get()
    if auth is not None:
        return auth
    return _auth_of(getattr(entity, "created_by", None) or "")


def _auth_of(who: str) -> Any:
    from mantle.services.dependencies import AuthContext
    return AuthContext(principal_id=who, principal_type="user", user_id=who)


def _provider_doc(db: Any, provider_id: str) -> Dict[str, Any]:
    from mantle.db.constants import state_of
    doc = db.artifacts.get_artifact(provider_id)
    if not doc or state_of(doc) == "archived":
        raise CreateRefused(409, f"{provider_id} is not live; nothing was created")
    return doc


def _check_home(db: Any, provider_id: str, doc: Dict[str, Any]) -> None:
    """Values are allocated on the node that first wrote the edged artifact."""
    from mantle.db.vertex import LatticeArtifactStore
    home = doc.get(LatticeArtifactStore.TIME_CLAIM_KEY) or doc.get("_origin")
    here = getattr(db, "origin", None) or getattr(db.artifacts, "origin", None)
    if home and here and home != here:
        raise CreateRefused(409, f"values from {provider_id} are allocated on its home node")


# ── the verb `number` ────────────────────────────────────────────────────────────────────────

def _key_of(pctx: Dict[str, Any], number: int) -> str:
    width = pctx.get("width") or 0
    try:
        width = max(0, int(width))
    except (TypeError, ValueError):
        width = 0
    prefix = str(pctx.get("prefix") or "")
    digits = str(number).zfill(width)
    return f"{prefix}-{digits}" if prefix else digits


def _last_of(pctx: Dict[str, Any]) -> int:
    try:
        return max(0, int(pctx.get("last") or 0))
    except (TypeError, ValueError):
        return 0


def _write_provider(db: Any, doc: Dict[str, Any], pctx: Dict[str, Any], as_string: bool) -> None:
    """Write the provider's counter in place and announce it once the write commits. Its context
    is readable configuration and the counter is not content, so no version is cut. A rolled-back
    transaction drops the announcement with the write."""
    from mantle.db import doc_boundary
    out = {k: v for k, v in doc.items() if k not in ("_origin", "_seq")}
    out["context"] = json.dumps(pctx) if as_string else pctx
    db.artifacts.put_artifact(out, new_version=False)
    announced = {**doc, "context": dict(pctx)}
    db.conn.after_commit(lambda: doc_boundary.emit_artifact_change(announced, "artifact.updated"))


def _number(db: Any, entity: Any, field: str, rule: dict, provider_id: str,
            doc: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Callable[[], None]]]:
    pctx, p_string = _ctx_of(doc.get("context"))
    number = _last_of(pctx) + 1
    pctx["last"] = number
    _write_provider(db, doc, pctx, p_string)
    values: Dict[str, Any] = {field: number}
    if rule.get("key"):
        values[rule["key"]] = _key_of(pctx, number)
    return values, []


# ── the verb `create` ────────────────────────────────────────────────────────────────────────

def _spec_of(provider_id: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    from mantle.attenuation import ACTIONS
    from mantle.services import type_offers

    spec, _ = _ctx_of(doc.get("context"))
    problems = [k for k in ("creates", "in", "link")
                if not isinstance(spec.get(k), str) or not spec[k].strip()]
    if not isinstance(spec.get("fields", {}), dict):
        problems.append("fields")
    conducts = spec.get("conducts", [])
    if not isinstance(conducts, list) or not set(conducts) <= set(ACTIONS):
        problems.append("conducts")
    if problems:
        raise CreateRefused(409, f"{provider_id} needs {', '.join(problems)} to create from; "
                                 "nothing was created")
    fields = spec.get("fields") or {}
    secret = sorted(set(fields) & type_offers.secret_fields(spec["creates"]))
    if secret:
        raise CreateRefused(409, f"{provider_id} is readable configuration and may not hold "
                                 f"{', '.join(secret)}; nothing was created")
    return spec



def _create(db: Any, entity: Any, field: str, rule: dict, provider_id: str,
            doc: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Callable[[], None]]]:
    from mantle.db import lattice_api
    from mantle.entities.artifact import Artifact as ArtifactEntity
    from mantle.services import content_crypto, type_offers
    from mantle.services.dependencies import check_access

    spec = _spec_of(provider_id, doc)
    home, link = spec["in"], spec["link"]
    conducts = list(spec.get("conducts") or [])
    owner = str(doc.get("created_by") or "")
    try:
        check_access(_auth_of(owner), home, "create", db)
    except HTTPException as refused:
        if refused.status_code in (401, 403, 404):
            raise CreateRefused(409, f"the owner of {provider_id} cannot create in {home}; "
                                     "nothing was created") from None
        raise
    creator = str(doc.get("root_id") or provider_id)
    now = _now_iso()
    part_id = str(uuid.uuid4())
    part = ArtifactEntity(
        id=part_id, root_id=part_id, collection_id=home,
        # The made type is the one place a default lives; the spec's fields lie over it.
        context={**type_offers.defaults(spec["creates"]), **(spec.get("fields") or {}),
                 "content_type": spec["creates"]},
        content_type=spec["creates"], state=ArtifactEntity.STATE_LIVE,
        created_by=creator, modified_by=creator, created_time=now, modified_time=now)
    # The part is its own seal boundary: a read the link conducts opens the part's key and no
    # other in its home, and home grantees reach it through its home.
    with content_crypto.content_key_artifact(part_id):
        lattice_api.create_artifact(db, part, own_key=True)
    lattice_api.add_artifact_to_collection(db, home, part_id, origin=True)

    def link_from_item() -> None:
        item_root = getattr(entity, "root_id", None) or entity.id
        lattice_api.add_artifact_to_collection(db, item_root, part_id, relationship=link,
                                               origin=False, propagate=[])
        if conducts:
            # Grants walk containment edges only (`lattice_api.light_cone`), so what the link
            # carries rides a containment edge beside it, masked to exactly those actions.
            lattice_api.add_artifact_to_collection(db, item_root, part_id, origin=False,
                                                   propagate=sorted(conducts))
    return {field: part_id}, [link_from_item]


#: What a create does at an edged artifact, by the verb its type declares.
_VERBS: Dict[str, Callable[..., Tuple[Dict[str, Any], List[Callable[[], None]]]]] = {
    "number": _number,
    "create": _create,
}


def _verb_of(provider_id: str, doc: Dict[str, Any]) -> Callable[..., Any]:
    from mantle.services import type_offers
    pctx, _ = _ctx_of(doc.get("context"))
    ct = doc.get("content_type") or pctx.get("content_type") or ""
    does = type_offers.on_create_does(ct)
    if does not in _VERBS:
        raise CreateRefused(409, f"the type of {provider_id} ({ct or 'none'}) declares nothing "
                                 "to do on create; nothing was created")
    return _VERBS[does]


# ── create ───────────────────────────────────────────────────────────────────────────────────

def strip_stamps(db: Any, entity: Any) -> None:
    """Drop the stamped fields a create supplies; a re-create keeps what was stamped."""
    stamps = stamps_for(entity)
    if not stamps:
        return
    ctx, as_string = _ctx_of(getattr(entity, "context", None))
    _, stored = _stored_context(db, getattr(entity, "id", None))
    for field in stamps:
        ctx.pop(field, None)
        if stored and not _blank(stored.get(field)):
            ctx[field] = stored[field]
    _put_ctx(entity, ctx, as_string)


def apply_on_create(db: Any, entity: Any, rules: Dict[str, dict]) -> List[Callable[[], None]]:
    """Stamp ``entity``'s ruled fields, inside the caller's write transaction. Answers the steps
    to run once ``entity`` is written."""
    from mantle.services.dependencies import check_access

    ctx, as_string = _ctx_of(getattr(entity, "context", None))
    exists, stored = _stored_context(db, getattr(entity, "id", None))
    if exists:
        # The same create again (a natural key resolved to the same id): what it was given
        # stays, and nothing is allocated.
        if stored is not None:
            for field in guarded_fields(rules):
                if not _blank(stored.get(field)):
                    ctx[field] = stored[field]
            _put_ctx(entity, ctx, as_string)
        if stored is None or all(not _blank(stored.get(f)) for f in rules):
            return []

    auth = _creator_auth(entity)
    after_put: List[Callable[[], None]] = []
    for field, rule in rules.items():
        if exists and stored and not _blank(stored.get(field)):
            continue
        provider_id = provider_for(db, getattr(entity, "collection_id", None), rule["from_edge"])
        try:
            check_access(auth, provider_id, "invoke", db)
        except HTTPException as refused:
            if refused.status_code in (401, 403, 404):
                raise CreateRefused(403, f"setting {field} needs Invoke on {provider_id}") from None
            raise
        doc = _provider_doc(db, provider_id)
        _check_home(db, provider_id, doc)
        values, steps = _verb_of(provider_id, doc)(db, entity, field, rule, provider_id, doc)
        ctx.update(values)
        after_put.extend(steps)
    _put_ctx(entity, ctx, as_string)
    return after_put


# ── update ───────────────────────────────────────────────────────────────────────────────────

def _apply_stamps(entity: Any, stamps: Dict[str, dict], ctx: Dict[str, Any],
                  stored: Optional[Dict[str, Any]]) -> None:
    for field, stamp in stamps.items():
        when = stamp["when"]
        if stored is None:
            raise CreateRefused(409, f"the stored {when} cannot be read here, so {field} "
                                     "cannot be stamped")
        before, after = stored.get(when), ctx.get(when)
        moved = _blank(before) != _blank(after) or (not _blank(after) and not _same(before, after))
        if moved:
            ctx[field] = (getattr(entity, "modified_by", None) if stamp["stamp"] == "caller"
                          else getattr(entity, "modified_time", None) or _now_iso())
        elif _blank(stored.get(field)):
            ctx.pop(field, None)
        else:
            ctx[field] = stored[field]


def guard_update(db: Any, entity: Any) -> None:
    """Keep the ruled fields as they were set: carry a left-out value forward, refuse a change.
    Stamp the ``on_update`` fields whose ``when`` field the update changes."""
    if _adopting.get():
        return
    rules, stamps = rules_for(entity), stamps_for(entity)
    if not rules and not stamps:
        return
    fields = guarded_fields(rules)
    ctx, as_string = _ctx_of(getattr(entity, "context", None))
    exists, stored = _stored_context(db, getattr(entity, "id", None))
    if not exists:
        return
    if stored is None:
        if any(not _blank(ctx.get(f)) for f in fields):
            raise CreateRefused(409, "the stored values of " + ", ".join(fields)
                                + " cannot be read here, so they cannot be changed")
    else:
        for field in fields:
            before, after = stored.get(field), ctx.get(field)
            if _blank(before):
                continue
            if _blank(after):
                ctx[field] = before
            elif not _same(before, after):
                raise CreateRefused(409, f"{field} is set on create and does not change")
    if stamps:
        _apply_stamps(entity, stamps, ctx, stored)
    _put_ctx(entity, ctx, as_string)


# ── adopt ────────────────────────────────────────────────────────────────────────────────────

def adopt(db: Any, provider_id: str, collection_id: str, pattern: str, *, field: str = "number",
          strip: bool = True, dry_run: bool = True) -> Dict[str, Any]:
    """Give the artifacts in ``collection_id`` the numbers their titles already carry.

    ``pattern`` is a regular expression with a group ``n`` (the number) and, optionally, a group
    ``prefix``; an item whose ``prefix`` differs from the provider's is left alone. Each matching
    item whose ``field`` is empty gets the number and its key, and with ``strip`` the matched text
    is removed from its title. The provider's ``last`` becomes the highest number. Two items with
    one number refuse the whole run, listing them. ``dry_run`` (the default) writes nothing and
    returns the changes."""
    from mantle.db import lattice_api
    from mantle.services import type_offers

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise CreateRefused(400, f"pattern: {exc}") from None
    if "n" not in regex.groupindex:
        raise CreateRefused(400, "pattern needs a group named n")
    doc = _provider_doc(db, provider_id)
    _check_home(db, provider_id, doc)
    pctx, p_string = _ctx_of(doc.get("context"))
    prefix = str(pctx.get("prefix") or "")

    changes: List[Dict[str, Any]] = []
    claimed: Dict[int, List[str]] = {}
    highest = _last_of(pctx)
    for item in lattice_api.list_collection_artifacts(db, collection_id):
        ictx, _ = _ctx_of(item.get("context"))
        rule = type_offers.on_create_rules(
            item.get("content_type") or ictx.get("content_type") or "").get(field)
        if rule is None:
            continue
        if not _blank(ictx.get(field)):
            try:
                held = int(ictx[field])
            except (TypeError, ValueError):
                continue
            claimed.setdefault(held, []).append(item["root_id"])
            highest = max(highest, held)
            continue
        title = str(ictx.get("title") or item.get("name") or "")
        match = regex.search(title)
        if not match:
            continue
        if "prefix" in regex.groupindex and (match.group("prefix") or "") != prefix:
            continue
        number = int(match.group("n"))
        highest = max(highest, number)
        claimed.setdefault(number, []).append(item["root_id"])
        new_title = title
        if strip:
            new_title = (title[:match.start()] + title[match.end():]).strip(" \t:-–—")
        change = {"id": item["id"], "root_id": item["root_id"], "number": number,
                  "title": new_title, "rule": rule}
        if rule.get("key"):
            change["key"] = _key_of(pctx, number)
        changes.append(change)

    duplicates = {n: roots for n, roots in claimed.items() if len(roots) > 1}
    if duplicates:
        raise CreateRefused(409, "two artifacts claim one number: " + "; ".join(
            f"{n}: {', '.join(sorted(roots))}" for n, roots in sorted(duplicates.items())))

    report = {"changes": [{k: v for k, v in c.items() if k != "rule"} for c in changes],
              "last": highest, "dry_run": dry_run}
    if dry_run:
        return report

    token = _adopting.set(True)
    try:
        with db.conn.write():
            for c in changes:
                entity = lattice_api.get_artifact(db, c["id"])
                if entity is None:
                    continue
                ctx, as_string = _ctx_of(entity.context)
                ctx[field] = c["number"]
                if c["rule"].get("key"):
                    ctx[c["rule"]["key"]] = c["key"]
                if strip:
                    ctx["title"] = c["title"]
                _put_ctx(entity, ctx, as_string)
                lattice_api.update_artifact(db, entity)
            pctx["last"] = highest
            _write_provider(db, doc, pctx, p_string)
    finally:
        _adopting.reset(token)
    return report


__all__ = ["CreateRefused", "adopt", "apply_on_create", "creating_as", "guard_update",
           "guarded_fields", "provider_for", "rules_for", "stamps_for", "strip_stamps"]
