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

Only what Origin has not co-signed goes: an edge or grant whose receipt the store already keeps
(written when the change was journaled, or by an earlier run) is left out (`pending`). Once
changes are journaled as they happen, Origin accepts a bootstrap only for ids it has not seen, and
only for ids the operator listed when opening the window, so a run is told which ids it may state
(`--ids-stdin`, one id per line): an entry touching any other id is left out. `--ids` prints the
ids the pending entries touch, for Origin to filter and list (`origin.scripts.authority_bootstrap
--unseen`, then `--ids-stdin`).

A row Origin refuses as malformed (400: a grant with no grantee, an id longer than Origin keeps)
reaches no one; it is counted by reason and the run goes on. Any other refusal stops the run.

Idempotent: Origin answers an entry already in its replica `unchanged`, with that entry's receipt,
and appends nothing. Run it again after any interruption.

    python -m mantle.system.cosign_authority --dry-run
    python -m mantle.system.cosign_authority --ids                 # the pending entries' ids
    python -m mantle.system.cosign_authority --bootstrap-token-file /path/bootstrap.token [--ids-stdin]

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


def entry_ids(entry: dict) -> list:
    """The ids an entry states, as Origin reads them (`authority_ledger._entry_ids`)."""
    if entry["kind"] == "grant":
        return [entry["grant_id"], entry["resource_id"]]
    if entry["kind"] in ("edge.add", "edge.remove"):
        return [entry["src"], entry["dst"]]
    return [entry["principal_id"]]


def pending(db, p: Optional[dict] = None) -> dict:
    """The plan without the edges and grants whose receipt the store keeps: Origin co-signed
    those already. A key root keeps no receipt, so every root stays; Origin's filter of the ids
    it has seen decides those."""
    from mantle.db import reach_journal

    p = p or plan(db)
    conn = db.graph.db.read()
    held = {g: [e for e in p[g] if reach_journal.receipt_of_row(conn, _key(e))] for g in ("edges", "grants")}
    out = {g: [e for e in p[g] if e not in held.get(g, [])] for g in ("edges", "roots", "grants")}
    out["secondary"] = p["secondary"]
    out["held"] = sum(len(v) for v in held.values())
    return out


def pending_ids(db) -> list:
    p = pending(db)
    return sorted({i for g in ("edges", "roots", "grants") for e in p[g] for i in entry_ids(e)})


def _key(entry: dict) -> str:
    if entry["kind"] == "grant":
        return f"grant:{entry['grant_id']}"
    if entry["kind"] == "root":
        return f"root:{entry['principal_id']}"
    return f"edge:{entry['src']} {entry['dst']}"


def cosign(db, custodian, bootstrap_credential: str, only: Optional[set] = None) -> dict:
    """Submit the pending entries as bootstrap entries and keep the receipts; with `only`, just
    the entries whose ids are all in it. Returns the counts. Stops at the first refusal, naming
    it: a partial replica is safe to extend by running again."""
    from mantle.clients.origin_key_share_client import AuthorityRefused
    from mantle.services.authority_ledger import receipt_of

    p = pending(db)
    counts = {"edges": 0, "roots": 0, "grants": 0, "appended": 0, "unchanged": 0,
              "secondary": p["secondary"], "held": p["held"], "left_out": 0, "malformed": 0,
              "left_out_by_kind": {"edges": 0, "roots": 0, "grants": 0}, "left_out_grantees": {},
              "malformed_by_reason": {}, "malformed_by_kind": {"edges": 0, "roots": 0, "grants": 0}}
    for group in ("edges", "roots", "grants"):
        for entry in p[group]:
            if only is not None and not set(entry_ids(entry)) <= only:
                counts["left_out"] += 1
                counts["left_out_by_kind"][group] += 1
                if group == "grants":
                    who = str(entry.get("grantee_id") or "")
                    counts["left_out_grantees"][who] = counts["left_out_grantees"].get(who, 0) + 1
                continue
            try:
                answer = custodian.submit_entry(entry, credential=bootstrap_credential,
                                                bootstrap=True)
            except AuthorityRefused as exc:
                # A malformed row (a grant with no grantee, an id past Origin's length) is refused
                # as a 400 and reaches no one: it is counted by reason and the run goes on. Any
                # other refusal is Origin's authority speaking, and stops the run.
                if exc.status != 400:
                    raise
                counts["malformed"] += 1
                reason = str(exc.detail)[:120]
                counts["malformed_by_reason"][reason] = counts["malformed_by_reason"].get(reason, 0) + 1
                counts["malformed_by_kind"][group] += 1
                continue
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
    ap.add_argument("--ids", action="store_true", help="print the ids the pending entries touch")
    ap.add_argument("--ids-stdin", action="store_true",
                    help="state only entries whose ids are all among those on stdin, one per line")
    ap.add_argument("--bootstrap-token-file",
                    default=os.getenv("MANTLE_BOOTSTRAP_CREDENTIAL_FILE", ""),
                    help="the bootstrap credential the operator obtained at Origin")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    credential = ""
    if not args.dry_run and not args.ids:
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
    if args.ids:
        for i in pending_ids(db):
            print(i)
        return 0
    if args.dry_run:
        p = plan(db)
        left = pending(db, p)
        print(f"would co-sign {len(p['edges'])} containment edge(s), {len(p['roots'])} key "
              f"root(s) and {len(p['grants'])} grant(s) at Origin; {p['secondary']} extra "
              f"creation edge(s) go as secondary containment; {left['held']} already co-signed")
        return 0
    only = {ln.strip() for ln in sys.stdin if ln.strip()} if args.ids_stdin else None

    from mantle.clients.origin_key_share_client import OriginShareCustodian

    custodian = OriginShareCustodian()
    try:
        counts = cosign(db, custodian, credential, only)
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
          f"{counts['secondary']} extra creation edge(s) as secondary containment; "
          f"{counts['held']} co-signed before; "
          f"left out, touching ids Origin has seen: {counts['left_out_by_kind']['edges']} edge(s), "
          f"{counts['left_out_by_kind']['roots']} root(s), {counts['left_out_by_kind']['grants']} grant(s); "
          f"malformed, refused: {counts['malformed_by_kind']['edges']} edge(s), "
          f"{counts['malformed_by_kind']['roots']} root(s), {counts['malformed_by_kind']['grants']} grant(s)")
    for who, n in sorted(counts["left_out_grantees"].items()):
        print(f"left out: {n} grant(s) to {who}")
    for reason, n in sorted(counts["malformed_by_reason"].items()):
        print(f"malformed: {n} x {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
