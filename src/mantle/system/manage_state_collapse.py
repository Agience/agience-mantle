"""Collapse the legacy `committed` and `draft` artifact states into `live`.

An artifact version is `live` or `archived` (`db/constants`). A store that has not been migrated
carries the legacy `committed` on nearly every row and `draft` on members that were never
promoted. Every reader
already treats both as `live` (`db/constants.state_of`, `live_states`), so this pass changes what
the rows RECORD, not what any read returns; afterwards the `state:committed` and `state:draft`
counters read 0 and nothing depends on the legacy values.

What it does, in order:

  1. **This node's legacy drafts**, one group per ``(root_id, collection_id)`` holding one,
     classified by the drafts' own ``_origin``:

       A  draft-only lineage: one draft and nothing else live in that collection.
       B  forked pair: one draft beside one or more committed/live rows in the same collection.
       C  several drafts under one root in one collection.
       D  every draft in the group was authored by another node. Not settled here: settling
          archives rows and writes new versions, which are authored writes, and this node does
          not author a peer's rows. Its drafts are relabelled in place in step 2.
       E  an A whose context still carries an unfinished ``upload`` section.

     The newest row of the group — last in ``(_origin, _seq)`` order, the rule every reader
     applies — is the one that stays live. When this node authored it, it is written ``live``
     and announced on the change feed as ``artifact.updated``. Each older row this node
     authored is archived with ``superseded_by`` naming the newest; an older row another node
     authored is left to step 2, and the readers' newest-row rule already resolves past it.
     These are ordinary authored writes (a fresh ``_seq``), so peers receive them. The newest row
     is then reindexed into the live segment with ``replace=True`` and its root is vacated from
     the legacy draft segment.

  2. **Every remaining legacy row is relabelled in place**, whoever authored it — first the
     drafts other nodes authored, then every `committed` row — by
     `LatticeArtifactStore.relabel_state`: keyset batches, the candidates read without the write
     lock and rewritten by id under it, the `state:*` counters moved by what was rewritten in
     the same transaction. It allocates no ``_seq`` and changes no merkle leaf, so nothing is
     republished; a relabel is this node's copy of a value that reads the same either way, which
     is also what `mesh/sync` does to a legacy row on ingest. See that method for why the
     counters stay exact.

  3. **Verify**: `state:committed` and `state:draft` both read 0, and no draft this node authored
     remains.

The legacy draft index segment (``LEGACY_DRAFT_PREFIXES``) is read by no recall and written by no
index job. Its size is reported, and the commands that would delete it are PRINTED for the operator
— this pass never deletes it: deleting an operator's data is the operator's decision.

``--report`` is the default and writes nothing. ``--apply`` is explicit. Re-running ``--apply``
after a complete run finds nothing to do.

    python -m mantle.system.manage_state_collapse
    python -m mantle.system.manage_state_collapse --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from mantle.db.constants import LEGACY_LIVE_STATES, STATE_ARCHIVED, STATE_LIVE, state_of

logger = logging.getLogger("manage_state_collapse")

#: The recorded states the report counts.
REPORTED_STATES = (STATE_LIVE,) + LEGACY_LIVE_STATES + (STATE_ARCHIVED,)
CLASSES = ("A", "B", "C", "D", "E")
#: Rows per relabel transaction: 5.6M rows is ~1,100 transactions, and each holds the write lock
#: only for its own batch, so request traffic interleaves between them.
DEFAULT_BATCH = 5000


def _context_of(doc: Dict[str, Any]) -> Dict[str, Any]:
    ctx = doc.get("context")
    if isinstance(ctx, dict):
        return ctx
    if isinstance(ctx, str) and ctx:
        try:
            parsed = json.loads(ctx)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _upload_unfinished(doc: Dict[str, Any]) -> bool:
    """A draft whose upload never completed: `update_upload_status` drops the ``upload`` section
    from the context on `complete`, so its presence is the unfinished state."""
    return bool(_context_of(doc).get("upload"))


def classify(db) -> List[Dict[str, Any]]:
    """One record per ``(root_id, collection_id)`` holding a legacy draft.

    Each record: ``{"class", "root_id", "collection_id", "drafts": [ids], "local_drafts": [ids],
    "members": [ids], "winner": id, "winner_is_local": bool, "losers": [ids],
    "foreign_losers": [ids]}``. ``members`` is every non-archived row of the root in that
    collection, oldest first; ``winner`` is the last of them. ``losers`` are the older members
    this node authored — the rows it archives — and ``foreign_losers`` the ones it leaves."""
    keys = []
    seen = set()
    # `state="draft"` matches the recorded value exactly (`live_states`): legacy drafts only.
    for doc in db.artifacts.list_artifacts(state="draft"):
        key = (str(doc.get("root_id") or doc.get("id")), str(doc.get("collection_id") or ""))
        if key not in seen:
            seen.add(key)
            keys.append(key)

    out = []
    for root_id, collection_id in keys:
        members = [v for v in (db.artifacts.versions_of(root_id) or [])
                   if str(v.get("collection_id") or "") == collection_id
                   and state_of(v) != "archived"]
        drafts = [v for v in members if v.get("state") == "draft"]
        local_drafts = [v for v in drafts if v.get("_origin") == db.origin]
        if not local_drafts:
            cls = "D"
        elif len(drafts) >= 2:
            cls = "C"
        elif len(members) > len(drafts):
            cls = "B"
        elif _upload_unfinished(drafts[0]):
            cls = "E"
        else:
            cls = "A"
        older = members[:-1]
        out.append({
            "class": cls, "root_id": root_id, "collection_id": collection_id,
            "drafts": [v["id"] for v in drafts],
            "local_drafts": [v["id"] for v in local_drafts],
            "members": [v["id"] for v in members],
            "winner": members[-1]["id"] if members else None,
            "winner_is_local": bool(members) and members[-1].get("_origin") == db.origin,
            "losers": [v["id"] for v in older if v.get("_origin") == db.origin],
            "foreign_losers": [v["id"] for v in older if v.get("_origin") != db.origin],
        })
    return out


def _dir_bytes(path: str) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def measure_draft_prefixes() -> Dict[str, Any]:
    """Bytes held under the legacy draft index prefixes, locally and in the edge bucket.

    Local: the cell tree ``<MANTLE_CELL_DIR>/mantle-cells-draft``, the posting database
    ``<MANTLE_SSE_DIR>/mantle-sse-draft.db`` with its WAL and shared-memory files, and the retired
    posting tree ``<MANTLE_SSE_DIR>/mantle-sse-draft``. S3: every object under each prefix, when
    edge storage is configured and reachable; otherwise the reason it was not measured."""
    from mantle.search.mantle import wiring

    cell_prefix, sse_prefix = wiring.LEGACY_DRAFT_PREFIXES
    local: Dict[str, int] = {}
    cells = os.path.join(wiring.local_cell_root(), cell_prefix)
    if os.path.isdir(cells):
        local[cells] = _dir_bytes(cells)
    sse_db = wiring.local_sse_path(sse_prefix)
    for path in (sse_db, sse_db + "-wal", sse_db + "-shm"):
        if os.path.isfile(path):
            local[path] = os.path.getsize(path)
    tree = os.path.join(wiring.local_sse_root(), sse_prefix)
    if os.path.isdir(tree):
        local[tree] = _dir_bytes(tree)

    s3: Dict[str, Any] = {}
    bucket = None
    try:
        if not wiring.edge_object_storage_is_configured():
            s3 = {"unmeasured": "edge object storage is not configured"}
        else:
            client, bucket = wiring.edge_s3_if_reachable("state collapse report")
            if client is None:
                s3 = {"unmeasured": "edge object storage is configured and unreachable"}
            else:
                for prefix in (cell_prefix, sse_prefix):
                    n = size = 0
                    for page in client.get_paginator("list_objects_v2").paginate(
                            Bucket=bucket, Prefix=prefix + "/"):
                        for obj in page.get("Contents") or ():
                            n += 1
                            size += int(obj.get("Size") or 0)
                    s3[prefix] = {"objects": n, "bytes": size}
    except Exception as exc:                  # noqa: BLE001 — a report must not fail on a probe
        s3 = {"unmeasured": "%s: %s" % (type(exc).__name__, exc)}
    return {"local": local, "s3": s3, "bucket": bucket}


def deletion_commands(prefixes: Dict[str, Any]) -> List[str]:
    """The commands that would delete the legacy draft index. Printed, never run."""
    from mantle.search.mantle import wiring

    cmds = []
    for path in sorted(prefixes.get("local") or {}):
        cmds.append(("rm -rf %s" if os.path.isdir(path) else "rm -f %s") % path)
    bucket = prefixes.get("bucket")
    if bucket and "unmeasured" not in (prefixes.get("s3") or {}):
        for prefix in wiring.LEGACY_DRAFT_PREFIXES:
            cmds.append("aws s3 rm s3://%s/%s/ --recursive" % (bucket, prefix))
    return cmds


def report(db, *, measure_prefixes: bool = True) -> Dict[str, Any]:
    """Everything the migration would act on. Reads only."""
    groups = classify(db)
    by_class = {c: [g for g in groups if g["class"] == c] for c in CLASSES}
    return {
        "origin": db.origin,
        "counters": {s: db.artifacts.count_recorded(s) for s in REPORTED_STATES},
        "classes": {c: len(v) for c, v in by_class.items()},
        "groups": groups,
        "foreign_draft_rows": sum(len(g["drafts"]) - len(g["local_drafts"]) for g in groups),
        "committed_to_relabel": db.artifacts.count_recorded("committed"),
        "draft_prefixes": measure_draft_prefixes() if measure_prefixes else None,
    }


def _default_reindex(artifact) -> None:
    """Index the live row into the live segment in place and vacate its root from the legacy
    draft segment, in one job."""
    from mantle.search.ingest.pipeline_unified import enqueue_index_artifact
    enqueue_index_artifact(artifact, artifact.collection_id or artifact.id,
                           vacate=["draft"], replace=True)


def _changed(raw: Dict[str, Any], **changes: Any) -> Dict[str, Any]:
    """A stored doc with ``changes`` applied, ready for an authored write. The doc goes back as
    stored — content stays sealed exactly as it was — so the write changes only the fields
    named; ``_origin`` and ``_seq`` are dropped because the write stamps its own."""
    doc = {k: v for k, v in raw.items() if k not in ("_origin", "_seq")}
    doc.update(changes)
    return doc


def apply(db, groups: List[Dict[str, Any]], *, batch: int = DEFAULT_BATCH,
          reindex: Optional[Callable[[Any], None]] = None) -> Dict[str, Any]:
    """Settle every A, B, C and E group, then relabel every remaining legacy row in place — the
    drafts other nodes authored, then every `committed` row. Returns counts.

    ``reindex`` receives the live row's entity after its write; the default enqueues the index
    job. A reindex that fails is logged and counted and does not stop the pass: the row is
    already live, and a later reindex puts it in the index.

    Each group is settled inside one write transaction that re-reads the lineage first, so no
    other writer can change a row between the read and the write that rests on it. A group whose
    newest row is no longer the one the report named — the lineage moved since — is left alone
    and counted under ``changed``; the next report classifies it afresh."""
    from mantle.db import doc_boundary as _boundary
    from mantle.db import lattice_api

    reindex = reindex or _default_reindex
    done = {c: 0 for c in CLASSES if c != "D"}
    skipped = archived = reindex_failed = changed = 0
    for g in groups:
        if g["class"] == "D":
            skipped += 1                      # relabelled in place below, never authored here
            continue
        if not g["winner"]:
            continue
        written = None
        with db.conn.write():
            members = [v for v in (db.artifacts.versions_of(g["root_id"]) or [])
                       if str(v.get("collection_id") or "") == g["collection_id"]
                       and state_of(v) != "archived"]
            if not members or members[-1].get("id") != g["winner"]:
                changed += 1
                continue
            winner = members[-1]
            for loser in members[:-1]:
                if loser.get("id") in g["losers"]:
                    db.artifacts.put_artifact(
                        _changed(loser, state="archived", superseded_by=g["winner"]))
                    archived += 1
            # A newest row another node authored is not written here; step 2 relabels it in
            # place.
            if g["winner_is_local"]:
                written = db.artifacts.put_artifact(_changed(winner, state=STATE_LIVE))
        # Announced for the version that is now current. The rows archived above are older
        # versions of the same root, which is what a subscriber keys on.
        if written is not None:
            _boundary.emit_artifact_change(lattice_api._feed_doc(written), _boundary.UPDATED)
        try:
            reindex(lattice_api.get_artifact(db, g["winner"]))
        except Exception:                     # noqa: BLE001 — see the docstring
            reindex_failed += 1
            logger.warning("reindex of %s FAILED — it is live and not yet in the live segment",
                           g["winner"], exc_info=True)
        done[g["class"]] += 1
    foreign = db.artifacts.relabel_state("draft", STATE_LIVE, batch=batch,
                                         exclude_origin=db.origin)
    relabel = db.artifacts.relabel_state("committed", STATE_LIVE, batch=batch)
    return {"settled": done, "skipped_d": skipped, "archived": archived, "changed": changed,
            "reindex_failed": reindex_failed,
            "foreign_drafts_relabelled": foreign["relabelled"],
            "relabelled": relabel["relabelled"], "relabel_batches": relabel["batches"]}


def verify(db) -> Dict[str, Any]:
    """After an apply: no `committed` row, no `draft` row, and none of this node's own drafts
    left unsettled — named separately, because that is the failure a skipped group produces."""
    committed = db.artifacts.count_recorded("committed")
    draft = db.artifacts.count_recorded("draft")
    local = [d["id"] for d in db.artifacts.list_artifacts(state="draft")
             if d.get("_origin") == db.origin]
    return {"committed": committed, "draft": draft, "local_drafts": local,
            "ok": committed == 0 and draft == 0 and not local}


def _print_report(rep: Dict[str, Any]) -> None:
    print("origin              %s" % rep["origin"])
    for s in REPORTED_STATES:
        print("state:%-14s %d" % (s, rep["counters"][s]))
    print("")
    print("legacy draft groups by class")
    for c in CLASSES:
        print("  %s  %d" % (c, rep["classes"][c]))
    print("  (D = only other nodes' drafts; relabelled in place, %d foreign draft row(s) in all)"
          % rep["foreign_draft_rows"])
    print("committed rows to relabel  %d" % rep["committed_to_relabel"])
    pre = rep.get("draft_prefixes")
    if pre is not None:
        print("")
        print("legacy draft index prefixes")
        for path, size in sorted((pre.get("local") or {}).items()):
            print("  local %s  %d bytes" % (path, size))
        if not pre.get("local"):
            print("  local: none present")
        for key, val in sorted((pre.get("s3") or {}).items()):
            print("  s3 %s  %s" % (key, val))


def main(argv: Optional[List[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="migrate (default: report only, write nothing)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help="rows per relabel transaction (default %d)" % DEFAULT_BATCH)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    from mantle.db.backend import store_handle
    db = store_handle()

    rep = report(db)
    _print_report(rep)
    if not args.apply:
        print("")
        print("REPORT ONLY — nothing was written. Re-run with --apply to migrate.")
        return 0

    result = apply(db, rep["groups"], batch=args.batch)
    print("")
    print("applied: settled %s, %d D group(s) left to the relabel, archived %d, foreign drafts "
          "relabelled %d, committed relabelled %d in %d batch(es), reindex failed %d"
          % (result["settled"], result["skipped_d"], result["archived"],
             result["foreign_drafts_relabelled"], result["relabelled"],
             result["relabel_batches"], result["reindex_failed"]))
    ver = verify(db)
    print("verify: state:committed %d, state:draft %d, unsettled local drafts %d -> %s"
          % (ver["committed"], ver["draft"], len(ver["local_drafts"]),
             "OK" if ver["ok"] else "NOT CLEAN"))
    cmds = deletion_commands(rep.get("draft_prefixes") or {})
    if cmds:
        print("")
        print("Recall reads the live and archived segments only; these prefixes hold the legacy "
              "draft index. To delete them — the operator's call, not run here:")
        for c in cmds:
            print("    " + c)
    return 0 if ver["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
