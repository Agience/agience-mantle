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
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List

from mantle.db.store import Database

logger = logging.getLogger(__name__)

_NS = "platform"
_COLLECTION_SLUG = "platform-operators"
OPERATOR_CT = "application/vnd.agience.operator+json"
COLLECTION_CT = "application/vnd.agience.collection+json"
WORKSPACE_CT = "application/vnd.agience.workspace+json"


def _artifact(description: str, *content_types: str) -> Dict[str, Any]:
    return {"type": "string", "format": "artifact-id", "x-content-types": list(content_types),
            "description": description}


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
    {"slug": "op-answer-from-my-notes", "title": "Answer from my notes", "public": True,
     "description": "Answer a question with cited extracts from what you can read.",
     "server": "lumen", "tool": "synthesize",
     "input_schema": {"type": "object", "required": ["input"], "properties": {
         "input": {"type": "string", "description": "The question"},
         "workspace_id": _artifact("Look only here (optional)", WORKSPACE_CT, COLLECTION_CT)}}},
    {"slug": "op-raise-inbox-item", "title": "Raise an Inbox item", "public": True,
     "description": "Put something in front of a person, with why and what it came from.",
     "server": "lumen", "tool": "raise_item",
     "input_schema": {"type": "object", "required": ["container_id", "title", "reason"], "properties": {
         "container_id": _artifact("The Inbox (or any workspace) it waits in", WORKSPACE_CT, COLLECTION_CT),
         "title": {"type": "string"},
         "reason": {"type": "string", "description": "Why it needs a person"},
         "source_id": _artifact("What raised it (optional)")}}},
    {"slug": "op-list-my-collections", "title": "List my collections", "public": True,
     "description": "The collections you can reach.",
     "server": "sage", "tool": "browse_collections",
     "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    {"slug": "op-ask-a-model", "title": "Ask a model", "public": True,
     "description": "Send a question to a model you have a connection to; its key stays sealed to Lumen.",
     "server": "lumen", "tool": "invoke_connection",
     "input_schema": {"type": "object", "required": ["connection_id", "input"], "properties": {
         "connection_id": _artifact("The model connection", "application/vnd.agience.llm-connection+json"),
         "input": {"type": "string", "description": "The question"},
         "max_tokens": {"type": "integer"}}}},
]


def ids() -> Dict[str, str]:
    """{slug: id} for the collection and every operator, derived from the instance namespace."""
    from mantle.services.seed_provisioning.loader import derive_uuid, get_instance_namespace
    ns = get_instance_namespace()
    if not ns:
        return {}
    return {slug: derive_uuid(ns, _NS, slug) for slug in [_COLLECTION_SLUG, *(o["slug"] for o in OPERATORS)]}


def _context(op: Dict[str, Any]) -> Dict[str, Any]:
    props = (op["input_schema"] or {}).get("properties") or {}
    return {
        "content_type": OPERATOR_CT, "title": op["title"], "description": op["description"],
        "platform": True,
        "input_schema": op["input_schema"],
        "operator": {"server": op["server"], "tool": op["tool"],
                     "arguments": {name: f"$.body.params.{name}?" for name in props}},
    }


def _own(upsert, db: Database, operator: str, artifact_id: str, title: str) -> None:
    """The platform operator's own grant on what it provisions. `created_by` is provenance and
    confers nothing, so without it the operator could not read, change or share its own operators,
    nor open their titles, which are sealed with their content."""
    upsert(db, user_id=operator, collection_id=artifact_id, granted_by=operator,
           can_create=True, can_read=True, can_update=True, can_delete=True, can_evict=True,
           can_invoke=True, can_add=True, can_share=True, can_admin=True, name=f"Owner: {title}")


def ensure_platform_operators(db: Database) -> bool:
    """Create or converge the collection and its operators. False when not yet possible."""
    from mantle.db.backend import add_artifact_to_collection as db_add_edge
    from mantle.db.backend import upsert_user_collection_grant
    from mantle.services.acting_principal import acting_as
    from mantle.services.platform_topology import register_id
    from mantle.services.seed_provisioning.platform_email import _ensure_artifact, _operator_id

    operator = _operator_id(db)
    known = ids()
    if not operator or not known:
        logger.info("platform_operators: no operator or instance namespace yet — will retry")
        return False
    with acting_as(operator, principal_type="user", scope="platform.operators-provisioning"):
        coll = known[_COLLECTION_SLUG]
        _ensure_artifact(db, coll, content_type=COLLECTION_CT, name="Platform operators",
                         description="Operators the platform provides; each is granted to every account.",
                         context={"content_type": COLLECTION_CT, "title": "Platform operators"},
                         content="", owner=operator)
        register_id(_COLLECTION_SLUG, coll)
        _own(upsert_user_collection_grant, db, operator, coll, "Platform operators")
        for op in OPERATORS:
            op_id = known[op["slug"]]
            existed = _ensure_artifact(db, op_id, content_type=OPERATOR_CT, name=op["title"],
                                       description=op["description"], context=_context(op),
                                       content="", owner=operator)
            register_id(op["slug"], op_id)
            _own(upsert_user_collection_grant, db, operator, op_id, op["title"])
            if not existed:
                db_add_edge(db, coll, op_id, origin=True)
    logger.info("platform_operators: %d operators in %s", len(OPERATORS), known[_COLLECTION_SLUG])
    return True


def grant_to_account(db: Database, user_id: str) -> int:
    """Grant `user_id` read and invoke on each public operator. Idempotent; returns how many."""
    from mantle.db.backend import upsert_user_collection_grant
    from mantle.services.seed_provisioning.platform_email import _operator_id

    operator = _operator_id(db)
    known = ids()
    if not operator or not known or not user_id or user_id == operator:
        return 0
    n = 0
    for op in OPERATORS:
        if op.get("public"):
            upsert_user_collection_grant(db, user_id=user_id, collection_id=known[op["slug"]],
                                         granted_by=operator, can_read=True, can_invoke=True,
                                         name=f"Platform operator: {op['title']}")
            n += 1
    return n


def grant_to_every_account(db: Database) -> Dict[str, int]:
    """Grant every account the store knows (every grantee of a user grant) the public operators."""
    from mantle.db import lattice_api as la
    accounts = set()
    for doc in la._grant_docs(db, state="active"):
        if doc.get("grantee_type") == "user" and doc.get("grantee_id"):
            accounts.add(str(doc["grantee_id"]))
    granted = sum(1 for a in sorted(accounts) if grant_to_account(db, a))
    return {"accounts": len(accounts), "granted": granted}


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
        if not ensure_platform_operators(db):
            logger.error("platform operators could not be provisioned")
            return 1
        counts = grant_to_every_account(db)
    logger.info("granted the platform operators: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
