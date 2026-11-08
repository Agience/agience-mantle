"""Co-sign the authority Mantle already holds at Origin, once, on the operator's bootstrap
credential.

WHY. Origin decides key release by its own replica of the authority ledger, and its replica holds
only entries it co-signed. Changes made from now on are journaled as they happen
(`db.reach_journal`); what was granted before has never been seen by Origin. This pass submits the
current state of every containment edge, every key principal's origin root and every grant as
bootstrap entries, and keeps each receipt in the store's `authority_receipt` table. It runs before
`MANTLE_MASTER_KEY_CUSTODY=threshold` is turned on, and before `mantle.system.split_master_keys`,
whose check read needs Origin to know each key.

The credential is a dedicated bootstrap credential the operator obtains at Origin
(`POST /oracle/authority-ledger/bootstrap`): an hour at most, good for bootstrap entries and
nothing else, and withdrawn there with `DELETE`. The operator's ordinary access token is refused
for bootstrap entries.

Order: edges, then roots (Origin checks a root against its own walk, which the edges decide), then
grants. Structure only goes to Origin: ids, actions, effect, state, expiry, an invite's target
digest and claim limit, masks.

A member with more than one creation edge (written by a move before the store kept one creation
edge per member) is submitted with the one `get_origin_parent` reads as its creation edge, and the
others as secondary containment, which conducts nothing upward. The pass reports how many.

Idempotent: Origin answers an entry already in its replica `unchanged`, with that entry's receipt,
and appends nothing. Run it again after any interruption.

    python -m mantle.system.cosign_authority --dry-run
    python -m mantle.system.cosign_authority --bootstrap-token-file /path/bootstrap.token

The bootstrap credential is read from the file (or `MANTLE_BOOTSTRAP_CREDENTIAL_FILE`), used for
this run only, and never written anywhere. `--dry-run` counts what would be submitted and sends
nothing.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Iterator, Optional

logger = logging.getLogger("mantle.system.cosign_authority")

_CONTAINS = "contains"


def _contains_edges(db) -> Iterator[dict]:
    """Every containment edge, straight from the table: a collection's member list is not read
    through `edges_of`, whose cap a large collection exceeds."""
    rows = db.graph.db.read().execute(
        "SELECT src, dst, propagate, is_origin, props FROM edge WHERE label = ? "
        "ORDER BY edge_key", (_CONTAINS,))
    for r in rows:
        props = {}
        try:
            props = json.loads(r["props"]) if r["props"] else {}
        except ValueError:
            pass
        prop = r["propagate"] if r["propagate"] is not None else props.get("propagate")
        is_origin = r["is_origin"] if r["is_origin"] is not None else props.get("is_origin")
        yield {"src": r["src"], "dst": r["dst"], "propagate": prop, "is_origin": bool(is_origin)}


def _key_principals(db) -> list:
    from mantle.search.mantle.oracle import SHARE_CONTENT_TYPE, LatticeMasterKeyStore

    out = set()
    for ct in (LatticeMasterKeyStore.CONTENT_TYPE, SHARE_CONTENT_TYPE):
        for doc in db.artifacts.list_artifacts(content_type=ct):
            rid = str(doc.get("id") or "")
            if rid.startswith("master-key:"):
                out.add(rid[len("master-key:"):])
    return sorted(out)


def _grant_docs(db) -> list:
    from mantle.db.reach_journal import GRANT_CT
    return list(db.artifacts.list_artifacts(content_type=GRANT_CT, include_archived=True))


def plan(db) -> dict:
    """What a run would submit, as entries, in order, and how many members had more than one
    creation edge."""
    from mantle.db import reach_journal as j
    from mantle.db.lattice_api import get_origin_parent, get_origin_root

    edges, extra = [], 0
    parents: dict = {}
    for e in _contains_edges(db):
        origin = e["is_origin"]
        if origin:
            if e["dst"] not in parents:
                p = get_origin_parent(db, e["dst"])
                parents[e["dst"]] = p[0] if p else None
            if parents[e["dst"]] != e["src"]:
                origin, extra = False, extra + 1
        edges.append(j.edge_entry(e["src"], e["dst"], origin=origin, propagate=e["propagate"]))
    edges.sort(key=lambda x: (not x["origin"], x["src"], x["dst"]))
    roots = [j.root_entry(p, str(get_origin_root(db, p))) for p in _key_principals(db)]
    grants = [j.grant_entry(d) for d in _grant_docs(db)]
    return {"edges": edges, "roots": roots, "grants": grants, "secondary": extra}


def _key(entry: dict) -> str:
    if entry["kind"] == "grant":
        return f"grant:{entry['grant_id']}"
    if entry["kind"] == "root":
        return f"root:{entry['principal_id']}"
    return f"edge:{entry['src']} {entry['dst']}"


def cosign(db, custodian, bootstrap_credential: str) -> dict:
    """Submit the plan as bootstrap entries and keep the receipts. Returns the counts. Stops at
    the first refusal, naming it: a partial replica is safe to extend by running again."""
    from mantle.services.authority_ledger import receipt_of

    p = plan(db)
    counts = {"edges": 0, "roots": 0, "grants": 0, "appended": 0, "unchanged": 0,
              "secondary": p["secondary"]}
    for group in ("edges", "roots", "grants"):
        for entry in p[group]:
            answer = custodian.submit_entry(entry, credential=bootstrap_credential,
                                            bootstrap=True)
            outcome = answer.get("outcome") or "appended"
            counts[outcome] = counts.get(outcome, 0) + 1
            counts[group] += 1
            receipt = receipt_of(answer)
            if receipt:
                with db.graph.db.write() as cur:
                    cur.execute(
                        "INSERT OR REPLACE INTO authority_receipt(subject, kind, seq, entry_hash,"
                        " signature, kid) VALUES(?,?,?,?,?,?)",
                        (_key(entry), entry["kind"], receipt["seq"], receipt["entry_hash"],
                         receipt["signature"], receipt["kid"]))
    return counts


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bootstrap-token-file",
                    default=os.getenv("MANTLE_BOOTSTRAP_CREDENTIAL_FILE", ""),
                    help="the bootstrap credential the operator obtained at Origin")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    credential = ""
    if not args.dry_run:
        if not args.bootstrap_token_file:
            ap.error("the bootstrap credential is required: --bootstrap-token-file or "
                     "MANTLE_BOOTSTRAP_CREDENTIAL_FILE (POST /oracle/authority-ledger/bootstrap "
                     "at Origin)")
        with open(args.bootstrap_token_file, encoding="utf-8") as fh:
            credential = fh.read().strip()
        if not credential:
            ap.error(f"{args.bootstrap_token_file} is empty")

    from prism.trust.key_manager import init_encryption_key, init_nonce_secret
    init_encryption_key()
    init_nonce_secret()
    from mantle.services import peer_signing
    peer_signing.init()
    from mantle.services.dependencies import get_store_db

    db = next(get_store_db())
    if args.dry_run:
        p = plan(db)
        print(f"would co-sign {len(p['edges'])} containment edge(s), {len(p['roots'])} key "
              f"root(s) and {len(p['grants'])} grant(s) at Origin; {p['secondary']} extra "
              f"creation edge(s) go as secondary containment")
        return 0

    from mantle.clients.origin_key_share_client import OriginShareCustodian

    custodian = OriginShareCustodian()
    try:
        counts = cosign(db, custodian, credential)
    except Exception as exc:  # noqa: BLE001 — reported by name; a rerun resumes
        logger.error("co-signing stopped: %s. Nothing already co-signed is lost; run again "
                     "with a new bootstrap credential.", exc)
        return 1
    finally:
        # The credential is good for this batch only, finished or not.
        try:
            custodian.bootstrap_done(credential)
        except Exception:  # noqa: BLE001 — it still lapses at its expiry
            logger.warning("Origin was not told the bootstrap batch is over; the credential "
                           "lapses at its expiry", exc_info=True)
    print(f"co-signed {counts['edges']} edge(s), {counts['roots']} root(s), {counts['grants']} "
          f"grant(s): {counts['appended']} appended, {counts['unchanged']} already held; "
          f"{counts['secondary']} extra creation edge(s) as secondary containment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
