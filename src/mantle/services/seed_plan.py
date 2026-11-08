"""A seed pass's plan: the ledger entries it will make, known before its credential is minted.

Origin takes a seed credential's entries only when each one's digest is in the plan the operator
minted it with (`POST /oracle/seed-credential` `{"plan": [...]}`, review 25). A plan can only be
written ahead if the pass's entries can be known ahead, and a new grant's id was a random UUID, so
they could not (review 26, open item c). Now:

  * Inside a plan (`planned_ids(salt)`), a new grant's id is derived from the plan's salt and the
    grant's structure (`grant_id`): the same pass over the same store makes the same ids, and a
    new plan, with a new salt, makes new ones, so no id is reused across passes.
  * Planning (`planning(salt)`) runs the pass with an authority that records each entry instead of
    submitting it. `mantle.system.seed_plan` runs it against a copy of the store, prints each
    entry's body and digest for the operator to read, and writes the plan file.
  * The pass itself (`MANTLE_SEED_PLAN_FILE`, read by `system_identity.seed_pass`) uses the plan's
    salt and refuses, before Origin is asked, any entry whose digest is not in the plan
    (`enforcing`); when it ends it reports any planned entry it did not make.

So minting with the plan, and running the pass with it, make exactly the planned entries: one
outside the plan is refused here and at Origin, and one planned but not made is reported.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import uuid
from contextlib import contextmanager
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

PLAN_FILE_SETTING = "MANTLE_SEED_PLAN_FILE"
PLAN_FORMAT = "agience-seed-plan-v1"

#: The namespace a planned grant's id is derived in.
_GRANT_NS = uuid.uuid5(uuid.NAMESPACE_URL, "agience://seed-plan/grant")

_salt: contextvars.ContextVar = contextvars.ContextVar("seed_plan_salt", default=None)


def digest(entry: dict) -> str:
    """The digest Origin checks a seed's entry against (`authority_ledger.body_digest`), as Mantle
    names it on every submission."""
    from mantle.clients.origin_key_share_client import entry_digest
    return entry_digest(entry)


def grant_id(resource_id, grantee_type, grantee_id) -> Optional[str]:
    """A new grant's id inside a plan, derived from the plan's salt and what the grant is on and
    for; None outside one (a random id is used)."""
    salt = _salt.get()
    if not salt:
        return None
    return str(uuid.uuid5(_GRANT_NS, f"{salt}\n{resource_id}\n{grantee_type}\n{grantee_id}"))


@contextmanager
def planned_ids(salt: str) -> Iterator[None]:
    if not isinstance(salt, str) or len(salt) < 16:
        raise ValueError("a plan's salt is at least 16 characters")
    token = _salt.set(salt)
    try:
        yield
    finally:
        _salt.reset(token)


@contextmanager
def planning(salt: str) -> Iterator[List[dict]]:
    """Run a pass with each ledger entry recorded, not submitted, and new grant ids derived from
    `salt`. Yields the list the entries are appended to, in order. Only for a copy of the store:
    the writes commit, with no Origin to co-sign them."""
    from mantle.db import reach_journal

    made: List[dict] = []

    def record(entry, *, proof=None, compensates=None, batch=None):
        if compensates is None:
            made.append(json.loads(json.dumps(entry)))
        return None

    prior = reach_journal._authority
    reach_journal.install(record)
    try:
        with planned_ids(salt):
            yield made
    finally:
        if prior is None:
            reach_journal.uninstall()
        else:
            reach_journal.install(prior)


class EntryNotPlanned(Exception):
    pass


def read_plan(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        plan = json.load(fh)
    if plan.get("format") != PLAN_FORMAT or not isinstance(plan.get("entries"), list):
        raise ValueError(f"{path} is not a seed plan ({PLAN_FORMAT})")
    for e in plan["entries"]:
        if digest(e["body"]) != e["digest"]:
            raise ValueError(f"{path}: an entry's digest does not match its body")
    return plan


@contextmanager
def enforcing(plan: dict) -> Iterator[dict]:
    """Run the pass on `plan`: its salt for new grant ids, and every entry checked against its
    digests before it is submitted. Yields `{"made": [...], "missing": [...]}`, filled when the
    block ends."""
    from mantle.db import reach_journal

    wanted = {e["digest"] for e in plan["entries"]}
    report = {"made": [], "missing": []}
    inner = reach_journal._authority

    def checked(entry, *, proof=None, compensates=None, batch=None):
        if compensates is None:
            d = digest(entry)
            if d not in wanted:
                raise reach_journal.ReachChangeRefused(
                    f"this seed pass's {entry.get('kind')} entry is not in its plan ({d[:12]}); "
                    f"nothing outside the plan is submitted")
            report["made"].append(d)
        if inner is None:
            return None
        return inner(entry, proof=proof, compensates=compensates, batch=batch)

    if inner is not None:
        reach_journal.install(checked)
    try:
        with planned_ids(plan["salt"]):
            yield report
    finally:
        if inner is not None:
            reach_journal.install(inner)
        report["missing"] = sorted(wanted - set(report["made"]))
        if report["missing"]:
            logger.error("seed pass: %d planned entr(ies) were not made: %s",
                         len(report["missing"]), [d[:12] for d in report["missing"]])


def plan_from_env() -> Optional[dict]:
    path = (os.getenv(PLAN_FILE_SETTING) or "").strip()
    return read_plan(path) if path else None


def write_plan(path: str, pass_name: str, salt: str, entries: List[dict]) -> dict:
    plan = {"format": PLAN_FORMAT, "pass": pass_name, "salt": salt,
            "entries": [{"body": e, "digest": digest(e)} for e in entries]}
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=1, sort_keys=True)
    return plan
