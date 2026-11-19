"""The platform's own operators — granted artifacts, not open tools.

An operator (`vnd.agience.operator+json`) is the invokable artifact: it names the tool it runs and
the arguments it takes, and invoking it needs an `invoke` grant on it. The platform provides a core
set this way — reconcile accounts, answer from my notes, raise an Inbox item, list my collections,
ask a model — in a "Platform operators" collection, owned by the platform operator and granted to
every account, one grant per operator, so each can be withheld or metered on its own. Anyone else
offers an operator the same way: a card of their own, bound to their own server or authorizer by a
link, shared by their grants.

Each operator declares its inputs (`context.input_schema`); an input naming an artifact says so
(`"format": "artifact-id"`, with the content types it accepts), so a builder offers a picker and
stores the choice as a link. Its arguments map each input from the invoke request
(`$.body.params.<name>?`, left out when absent).

Idempotent, non-fatal, runs every startup (`main.py`); `grant_to_account` runs at an account's first
sign-in (`user_provisioning`), and `python -m mantle.services.seed_provisioning.platform_operators`
grants every existing account (a release migration).

Where Origin keeps the ledger, what every new account receives is the operator's "New accounts
receive" collection: each edge out of it names a member and conducts read, or read and invoke, to the
person. Its id is derived from the operator's, so Origin finds it with no setting. Mantle links the
operators' collection and each per-user seed into it; Origin makes the grants, as the operator's own
entries, when Mantle names a new person.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from typing import Any, Dict, List

from mantle.db.store import Database

logger = logging.getLogger(__name__)

_NS = "platform"
_COLLECTION_SLUG = "platform-operators"
OPERATOR_CT = "application/vnd.agience.operator+json"
COLLECTION_CT = "application/vnd.agience.collection+json"
WORKSPACE_CT = "application/vnd.agience.workspace+json"
SEARCH_CT = "application/vnd.agience.search+json"


def _artifact(description: str, *content_types: str) -> Dict[str, Any]:
    return {"type": "string", "format": "artifact-id", "x-content-types": list(content_types),
            "description": description}


#: What a step reads or answers from: a note, a file, or a collection or workspace of them. A field
#: naming no type is offered collections alone (BUG-052, 2026-11-10: a single note could not be
#: linked as evidence).
ITEM_TYPES = ("text/markdown", "text/plain", "application/pdf")


def _item(description: str) -> Dict[str, Any]:
    return _artifact(description, *ITEM_TYPES, COLLECTION_CT, WORKSPACE_CT)


#: The core set. `public` operators are granted to every account; one that spends a platform
#: resource is not, and is granted where the operator decides.
OPERATORS: List[Dict[str, Any]] = [
    {"slug": "op-reconcile-accounts", "title": "Reconcile accounts", "public": True,
     "description": "Compare several accounts of one event fact by fact: what all agree on, where they "
                    "differ and what only one states; each difference raised in an Inbox for a person.",
     "server": "lumen", "tool": "reconcile",
     "input_schema": {"type": "object", "required": ["collection_id"], "properties": {
         "collection_id": _artifact("The accounts", COLLECTION_CT, WORKSPACE_CT),
         "inbox_to": _artifact("Where each difference waits for a person", WORKSPACE_CT, COLLECTION_CT),
         "out_to": _artifact("Where the reconciliation is written (the accounts' collection when left out)",
                             COLLECTION_CT, WORKSPACE_CT)}}},
    # In a workflow it answers from the items linked to it, which the run's grant lists (John,
    # 2026-11-10, BUG-052): a grant reaches what it names and never a search across everything, and
    # Sage answers in itself, with no second hop. Ask anything still searches every note.
    {"slug": "op-answer-from-my-notes", "title": "Answer from my notes", "public": True,
     "description": "Answer a question with cited extracts from the item it is given.",
     "server": "sage", "tool": "ask", "definition": "op.retrieve",
     "input_schema": {"type": "object", "required": ["question", "evidence"], "properties": {
         "question": {"type": "string", "description": "The question"},
         "evidence": _item("What to answer from")}},
     # A step made before then names its question `input`.
     "arguments": {"question": "$.body.params.question?", "input": "$.body.params.input?",
                   "evidence": "$.body.params.evidence?"}},
    {"slug": "op-raise-inbox-item", "title": "Raise an Inbox item", "public": True,
     "description": "Put something in front of a person, with why and what it came from.",
     "server": "lumen", "tool": "raise_item",
     "input_schema": {"type": "object", "required": ["container_id", "title", "reason"], "properties": {
         "container_id": _artifact("The Inbox (or any workspace) it waits in", WORKSPACE_CT, COLLECTION_CT),
         "title": {"type": "string"},
         "reason": {"type": "string", "description": "Why it needs a person"},
         "source_id": _artifact("What raised it (optional)")}}},
    # A server artifact's tool as a step (John, 2026-11-12, pilot check 5): the server is an input
    # artifact, so the step links it and the run's grant reaches it; Iris sends one tools/call with
    # the server's own key, never the caller's credential.
    {"slug": "op-use-a-server-tool", "title": "Use a server tool", "public": True,
     "description": "Run one tool on an MCP server added to the workspace as a server artifact.",
     "server": "iris", "tool": "use_server_tool", "definition": "op.mcp.call",
     "input_schema": {"type": "object", "required": ["server", "tool"], "properties": {
         "server": _artifact("The server artifact", "application/vnd.agience.mcp-server+json"),
         "tool": {"type": "string", "description": "The tool's name, as the server lists it"},
         "arguments": {"type": "object", "description": "The tool's arguments"}}},
     "arguments": {"server_artifact_id": "$.body.params.server", "tool": "$.body.params.tool",
                   "arguments": "$.body.params.arguments?"}},
    {"slug": "op-list-my-collections", "title": "List my collections", "public": True,
     "description": "The collections you can reach.",
     "server": "sage", "tool": "browse_collections",
     "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    # One item, named by a link: what a timer's grant reaches, so a scheduled workflow can read with
    # Sage (2026-11-10, BUG-051: Sage's only operator listed everything, which no timer reaches).
    {"slug": "op-read-an-item", "title": "Read an item", "public": True,
     "description": "Read one item you can reach: its title, type and what it holds.",
     "server": "sage", "tool": "get_artifact",
     "input_schema": {"type": "object", "required": ["artifact_id"], "properties": {
         "artifact_id": _item("The item")}}},
    {"slug": "op-ask-a-model", "title": "Ask a model", "public": True,
     "description": "Send a question to a model you have a connection to; its key stays sealed to Lumen.",
     "server": "lumen", "tool": "invoke_connection",
     "input_schema": {"type": "object", "required": ["connection_id", "input"], "properties": {
         "connection_id": _artifact("The model connection", "application/vnd.agience.llm-connection+json"),
         "input": {"type": "string", "description": "The question"},
         "max_tokens": {"type": "integer"}}}},
]


#: The platform's Search: what a Chat runs when it links no Search of its own. Its fields are the
#: recall it makes (`/artifacts/recall` with `search_id`): rarest matched words first, then what the
#: reader made, then the newest; six hits; no scope, so everything the reader may read.
PLATFORM_SEARCH: Dict[str, Any] = {
    "slug": "search-platform", "title": "Platform Search",
    "description": ("Finds what you may read: an exact title first, then the rarest matched words,"
                    " then your own, then the newest."),
    "size": 6, "rank": ["title", "rarity", "own", "recency"],
}


def ids() -> Dict[str, str]:
    """{slug: id} for the collection, every operator and the platform Search, derived from the
    instance namespace."""
    from mantle.services.seed_provisioning.loader import derive_uuid, get_instance_namespace
    ns = get_instance_namespace()
    if not ns:
        return {}
    slugs = [_COLLECTION_SLUG, *(o["slug"] for o in OPERATORS), PLATFORM_SEARCH["slug"]]
    return {slug: derive_uuid(ns, _NS, slug) for slug in slugs}


def platform_search_id() -> str:
    """The platform Search's id on this node, or empty before the instance namespace is known."""
    return ids().get(PLATFORM_SEARCH["slug"], "")


def catalog_title(artifact_id: str) -> str:
    """The title of the platform operator (or the operators' collection) `artifact_id` names, from
    its definition here: what a reader is shown when the node cannot open the sealed title, as on
    staging, which holds none of production's keys (BUG-057). Empty for any other id."""
    if not artifact_id:
        return ""
    titles = {_COLLECTION_SLUG: "Platform operators", **{o["slug"]: o["title"] for o in OPERATORS},
              PLATFORM_SEARCH["slug"]: PLATFORM_SEARCH["title"]}
    for slug, known in ids().items():
        if known == artifact_id:
            return titles.get(slug, "")
    return ""


def _on_rc() -> bool:
    return (os.getenv("AGIENCE_PLANE") or "").strip().lower() == "rc"


def _converge_unsealed(db: Database, artifact_id: str, *, context: dict, name: str,
                       description: str, owner: str) -> bool:
    """On the release candidate only: converge a row whose key production wrapped, which this node
    cannot unwrap, so nothing on it can be sealed here. It is written with no sealed field at all:
    the definition's open keys as they are, no title (readers get the catalog's,
    `catalog_title`), and the copied seal, which no one here can open, dropped. On a row that is
    not readable configuration the name and description are sealed fields too, and go as well."""
    from mantle.db import doc_boundary as boundary
    from mantle.db.backend import get_artifact as db_get_artifact
    from mantle.db.backend import update_artifact as db_update_artifact
    from mantle.services import mint_context as _mc

    existing = db_get_artifact(db, artifact_id)
    if existing is None:
        return False
    stored = existing.context
    stored = json.loads(stored) if isinstance(stored, str) and stored else (stored or {})
    stored = {k: v for k, v in dict(stored).items() if k not in (boundary.SEALED, *boundary.CONTENT_FIELDS)}
    given = {k: v for k, v in context.items() if k not in boundary.CONTENT_FIELDS}
    merged = _mc.with_client_keys(json.dumps(stored), given)
    merged_ctx = json.loads(merged) if isinstance(merged, str) else dict(merged or {})
    readable = boundary._readable({"content_type": existing.content_type}, merged_ctx)
    existing.context = merged
    existing.content = ""
    existing.name = name if readable else ""
    existing.description = description if readable else ""
    existing.modified_by = owner
    db_update_artifact(db, existing)
    return True


def _ensure(db: Database, artifact_id: str, **kw) -> bool:
    """`_ensure_artifact`, and on the release candidate, for a row this node cannot seal,
    `_converge_unsealed` (BUG-057)."""
    from mantle.services.seed_provisioning.platform_email import _ensure_artifact
    try:
        return _ensure_artifact(db, artifact_id, **kw)
    except Exception as exc:
        if not (_on_rc() and _is_unsealable_here(exc)):
            raise
    logger.info("platform_operators: %s is sealed under keys this node does not hold; "
                "converged in the open without its title", artifact_id)
    return _converge_unsealed(db, artifact_id, context=kw["context"], name=kw["name"],
                              description=kw["description"], owner=kw["owner"])


def _search_context() -> Dict[str, Any]:
    return {"content_type": SEARCH_CT, "title": PLATFORM_SEARCH["title"],
            "description": PLATFORM_SEARCH["description"], "platform": True,
            "size": PLATFORM_SEARCH["size"], "rank": list(PLATFORM_SEARCH["rank"])}


def _context(op: Dict[str, Any]) -> Dict[str, Any]:
    props = (op["input_schema"] or {}).get("properties") or {}
    context = {
        "content_type": OPERATOR_CT, "title": op["title"], "description": op["description"],
        "platform": True,
        "input_schema": op["input_schema"],
        "operator": {"server": op["server"], "tool": op["tool"],
                     "arguments": op.get("arguments") or {name: f"$.body.params.{name}?" for name in props}},
    }
    if op.get("definition"):
        # The operator definition it runs, by id; the definition is an artifact in this collection.
        context["definition"] = op["definition"]
    return context


#: What the definitions' publisher is given on the collection: create only. Create does not carry
#: read, so it adds definitions and lists only what its own creation grants reach.
DEFINITIONS_GRANT_NAME = "Operator definitions (create)"


def ensure_definitions_grant(db: Database, operator: str, collection_id: str) -> bool:
    """Grant the platform's system principal create on the operators' collection, as the operator,
    so the operator definitions the services carry are published into it. Idempotent; True when a
    grant was made.

    Where Origin keeps the ledger, Origin makes the grant as the operator's own entry, on the
    collection it derives from the instance namespace, and this store applies it from the ledger.
    Without a ledger, the grant is written here."""
    from mantle.db.backend import upsert_user_collection_grant
    from mantle.services.peer_signing import get_system_principal_id

    origin = _origin_grants()
    if origin is not None and hasattr(origin, "definitions_grant"):
        answer = origin.definitions_grant() or {}
        if answer.get("granted"):
            apply_remote(db)
        return bool(answer.get("granted"))
    system = get_system_principal_id()
    if not system or not collection_id or str(system) == str(operator):
        return False
    _grant, changed = upsert_user_collection_grant(
        db, user_id=system, collection_id=collection_id, granted_by=operator,
        can_create=True, can_read=False, name=DEFINITIONS_GRANT_NAME)
    return bool(changed)


def _as_operator(operator: str):
    """Act as the operator for the provisioning. When the operator is already acting, on their own
    request through the admin seed route, keep that principal and the credential it carries: under
    custody a bare principal of the same id is given no key, and the provisioning fails
    (2026-11-11, the home test node; John chose the admin route over a job grant)."""
    from contextlib import nullcontext

    from mantle.services.acting_principal import acting_as, current_acting_principal

    current = current_acting_principal()
    if (current is not None and str(getattr(current, "principal_id", "")) == str(operator)
            and getattr(current, "principal_type", "") == "user"):
        return nullcontext()
    return acting_as(operator, principal_type="user", scope="platform.operators-provisioning")


def ensure_platform_operators(db: Database) -> bool:
    """Create or converge the collection and its operators. False when not yet possible."""
    from mantle.db.backend import add_artifact_to_collection as db_add_edge
    from mantle.services.platform_topology import register_id
    from mantle.services.seed_provisioning.platform_email import _operator_id

    operator = _operator_id(db)
    known = ids()
    if not operator or not known:
        logger.info("platform_operators: no operator or instance namespace yet — will retry")
        return False
    with _as_operator(operator):
        coll = known[_COLLECTION_SLUG]
        # Made as any creator makes a container: the root and the creator's own grant in one batch,
        # which Origin co-signs. The release candidate, a copy of production that cannot seal under
        # its keys, converges the row in the open (BUG-057).
        if not _exists(db, coll) and not _on_rc():
            from mantle.services import workspace_service
            workspace_service.create_container(
                db, operator, content_type=COLLECTION_CT, name="Platform operators",
                description="Operators the platform provides; each is granted to every account.",
                context=json.dumps({"content_type": COLLECTION_CT, "title": "Platform operators"}),
                artifact_id=coll)
        else:
            _ensure(db, coll, content_type=COLLECTION_CT, name="Platform operators",
                    description="Operators the platform provides; each is granted to every account.",
                    context={"content_type": COLLECTION_CT, "title": "Platform operators"},
                    content="", owner=operator)
        register_id(_COLLECTION_SLUG, coll)
        for op in OPERATORS:
            op_id = known[op["slug"]]
            existed = _ensure(db, op_id, content_type=OPERATOR_CT, name=op["title"],
                              description=op["description"], context=_context(op),
                              content="", owner=operator)
            register_id(op["slug"], op_id)
            if not existed:
                # Its creation edge in the collection, which the operator owns at Origin.
                db_add_edge(db, coll, op_id, origin=True)
        search_id = known[PLATFORM_SEARCH["slug"]]
        existed = _ensure(db, search_id, content_type=SEARCH_CT, name=PLATFORM_SEARCH["title"],
                          description=PLATFORM_SEARCH["description"], context=_search_context(),
                          content="", owner=operator)
        register_id(PLATFORM_SEARCH["slug"], search_id)
        if not existed:
            db_add_edge(db, coll, search_id, origin=True)
        ensure_definitions_grant(db, operator, coll)
        if not _on_rc():
            from mantle.services.role_presets import ensure_roles
            ensure_roles(db, operator)
            ensure_new_accounts_receive(db, operator, coll)
    logger.info("platform_operators: %d operators in %s", len(OPERATORS), known[_COLLECTION_SLUG])
    return True


#: The mask the "New account ceiling" edge is seeded with. Once seeded, the edge is the bound:
#: Origin reads it from the ledger, and the operator changes it by editing the edge.
CEILING_SEED = ("invoke", "read")


def new_accounts_receive_id(operator: str) -> str:
    """The operator's "New accounts receive" collection; Origin derives the same id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agience://new-accounts-receive/{operator}"))


def new_account_ceiling_id(operator: str) -> str:
    """The operator's "New account ceiling"; its edge into "New accounts receive" bounds every
    edge of the list. Origin derives the same id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agience://new-account-ceiling/{operator}"))


def _ceiling(db: Database, ceiling: str, nar: str) -> List[str]:
    """The actions the ceiling edge conducts now."""
    from mantle.db import lattice_api as la
    edge = la._find_membership(db, ceiling, nar)
    if edge is None:
        return []
    propagate = edge.get("propagate")
    if isinstance(propagate, str):
        propagate = json.loads(propagate) if propagate.startswith("[") else [propagate]
    return sorted(str(a) for a in (propagate or ()))


def _per_user_seeds(ceiling: List[str]) -> List[tuple]:
    """(collection id, actions) for each per-user grant seed (`<seeds>/user/grants/*.yaml`, principal
    `{{user.id}}`): the seeded collections every account is given."""
    import yaml

    from mantle.services.seed_provisioning.loader import derive_uuid, get_instance_namespace
    from mantle.services.seed_provisioning.user_provisioning import _seeds_base

    base = _seeds_base()
    ns = get_instance_namespace()
    if base is None or ns is None:
        return []
    out = []
    for f in sorted((base / "user" / "grants").glob("*.yaml")):
        doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        if doc.get("type") != "grant" or doc.get("principal") != "{{user.id}}":
            continue
        space, _, slug = str(doc.get("resource") or "").partition("/")
        actions = sorted(set(doc.get("actions") or []) & set(ceiling))
        if space and slug and actions:
            out.append((derive_uuid(ns, space, slug), actions))
    return out


def ensure_new_accounts_receive(db: Database, operator: str, operators_collection: str) -> str:
    """Create the operator's "New accounts receive" collection, and its "New account ceiling"
    with an edge into it seeded `CEILING_SEED`, and link into the list what every account is
    given: the operators' collection and each per-user seed, each within the ceiling. A member not
    in the store yet is linked on a later pass; an existing edge is left as it is."""
    from mantle.db import lattice_api as la
    from mantle.services import workspace_service

    nar = new_accounts_receive_id(operator)
    if not _exists(db, nar):
        workspace_service.create_container(
            db, operator, content_type=COLLECTION_CT, name="New accounts receive",
            description="What every new account receives: on each member, what its edge conducts.",
            context=json.dumps({"content_type": COLLECTION_CT, "title": "New accounts receive"}),
            artifact_id=nar)
    ceiling = new_account_ceiling_id(operator)
    if not _exists(db, ceiling):
        workspace_service.create_container(
            db, operator, content_type=COLLECTION_CT, name="New account ceiling",
            description="The most a new account receives on any one member: the mask on its "
                        "edge into New accounts receive.",
            context=json.dumps({"content_type": COLLECTION_CT, "title": "New account ceiling"}),
            artifact_id=ceiling)
    if la._find_membership(db, ceiling, nar) is None:
        la.add_artifact_to_collection(db, ceiling, nar, origin=False, propagate=list(CEILING_SEED))
    bound = _ceiling(db, ceiling, nar)
    members = [(operators_collection, bound), *_per_user_seeds(bound)]
    for member, actions in members:
        if actions and _exists(db, member) and la._find_membership(db, nar, member) is None:
            la.add_artifact_to_collection(db, nar, member, origin=False, propagate=actions)
    return nar


def apply_remote(db: Database) -> int:
    """Apply Origin's new entries now, so the grants it just made are here at once."""
    from mantle.services import authority_ledger
    try:
        return authority_ledger.apply_remote(db)
    except Exception:  # noqa: BLE001 — the feed pass applies them shortly
        return 0


def _exists(db: Database, artifact_id: str) -> bool:
    """The artifact is in the store, whether or not this caller may open it."""
    from mantle.db import lattice_api as la
    try:
        return la.get_artifact(db, artifact_id) is not None
    except Exception:  # noqa: BLE001 — present, but sealed to this caller
        return True


def _origin_grants() -> object:
    """Origin's client, when this store's changes to reach go to Origin's ledger, else None."""
    from mantle.db import reach_journal
    sub = getattr(reach_journal, "_authority", None)
    return getattr(sub, "custodian", None) if sub is not None else None


def grant_to_account(db: Database, user_id: str, skip: frozenset = frozenset()) -> int:
    """Grant `user_id` read and invoke on each public operator, and read on the platform Search.
    Idempotent; returns how many.

    Where Origin keeps the ledger, Mantle names the person and Origin makes what "New accounts
    receive" lists, as the operator's own entries, which this store then applies from the ledger.
    Without a ledger, the grants are written here."""
    from mantle.db.backend import upsert_user_collection_grant
    from mantle.services.seed_provisioning.platform_email import _operator_id

    origin = _origin_grants()
    if origin is not None and hasattr(origin, "new_person_grants"):
        answer = origin.new_person_grants(user_id) or {}
        if answer.get("granted"):
            apply_remote(db)
        return int(answer.get("granted") or 0)
    operator = _operator_id(db)
    known = ids()
    if not operator or not known or not user_id or user_id == operator:
        return 0
    n = 0
    for op in OPERATORS:
        # Only an operator that exists: a grant naming one not yet made would stop its creation,
        # whose first key is minted only while no grant names it (71/test, 2026-11-12: an account
        # that signed in before the seed). The seed grants the rest to every account afterwards.
        if (op.get("public") and (str(user_id), known[op["slug"]]) not in skip
                and _exists(db, known[op["slug"]])):
            upsert_user_collection_grant(db, user_id=user_id, collection_id=known[op["slug"]],
                                         granted_by=operator, can_read=True, can_invoke=True,
                                         name=f"Platform operator: {op['title']}")
            n += 1
    search_id = known.get(PLATFORM_SEARCH["slug"])
    if search_id and (str(user_id), search_id) not in skip and _exists(db, search_id):
        upsert_user_collection_grant(db, user_id=user_id, collection_id=search_id,
                                     granted_by=operator, can_read=True,
                                     name=f"Platform Search: {PLATFORM_SEARCH['title']}")
        n += 1
    return n


def grant_to_every_account(db: Database) -> Dict[str, int]:
    """Grant every account the store knows (every grantee of a user grant) the public operators."""
    from mantle.db import lattice_api as la
    from mantle.services.peer_signing import get_system_principal_id
    # The platform's own system principal holds principal-typed grants on the platform's
    # collections (`ensure_platform_principal_grant`) and is not an account; the operator account
    # owns the operators and is never granted them.
    from mantle.services.seed_provisioning.platform_email import _operator_id
    not_accounts = {get_system_principal_id(), _operator_id(db) or ""} - {""}
    accounts = set()
    for doc in la._grant_docs(db, state="active"):
        if doc.get("grantee_type") == "user" and doc.get("grantee_id"):
            accounts.add(str(doc["grantee_id"]))
    accounts -= not_accounts
    granted = left = 0
    on_rc = (os.getenv("AGIENCE_PLANE") or "").strip().lower() == "rc"
    for a in sorted(accounts):
        try:
            granted += 1 if grant_to_account(db, a) else 0
        except Exception as exc:
            # The release candidate runs on a copy of production's store with a key encryption key
            # of its own: a grant written under a collection key production wrapped cannot be
            # sealed there (2026-11-09, release 2026.11.2: the accounts made since 2026.11.0's
            # promote). It is left, and counted; on any other plane it is a fault.
            if on_rc and _is_unsealable_here(exc):
                left += 1
                continue
            raise
    out = {"accounts": len(accounts), "granted": granted}
    if left:
        out["left_on_rc"] = left
    return out


def _is_unsealable_here(exc: BaseException) -> bool:
    """A write refused because a key production wrapped will not unwrap here: the master key, or
    the content encryption that needs it, anywhere in the exception's chain."""
    from mantle.db.doc_boundary import ContentEncryptionError
    from mantle.search.mantle.custody import MasterKeyUnavailable

    seen = exc
    while seen is not None:
        if isinstance(seen, (ContentEncryptionError, MasterKeyUnavailable)):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    from prism.trust.key_manager import init_encryption_key, init_nonce_secret
    init_encryption_key()
    init_nonce_secret()
    from mantle.services import peer_signing
    peer_signing.init()
    from mantle.services.dependencies import get_store_db
    from mantle.services.system_identity import system_acting_context

    db = next(get_store_db())
    # The node's settings, loaded as Mantle's startup loads them, so the operator resolves from the
    # store itself (`platform.operator_id`) rather than only by asking Origin, which a release
    # candidate that redeploys Mantle alone does not run. 2026-10-22: the migration failed there.
    from mantle import config
    from mantle.services.platform_settings_service import settings as platform_settings
    platform_settings.load_all(db)
    config.set_settings_provider(platform_settings.get)
    config.load_settings_from_db()
    with system_acting_context(scope="platform.operators-seed"):
        try:
            provisioned = ensure_platform_operators(db)
        except Exception as exc:
            # The release candidate cannot seal under a collection key production wrapped
            # (2026-11-09, release 2026.11.2: the operators' collection itself); there the
            # convergence is left as production has it. On any other plane it is a fault.
            if (os.getenv("AGIENCE_PLANE") or "").strip().lower() == "rc" and _is_unsealable_here(exc):
                logger.info("release candidate: the operators' collection is sealed under a key it "
                            "does not hold; left as production has it")
                provisioned = True
            else:
                raise
        if not provisioned:
            logger.error("platform operators could not be provisioned")
            return 1
        counts = grant_to_every_account(db)
    logger.info("granted the platform operators: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
