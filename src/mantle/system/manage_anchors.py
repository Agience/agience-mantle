#!/usr/bin/env python3
"""manage_anchors.py — load and inspect the MANTLE AnchorSet.

The AnchorSet is the shared coordinate system (see
`.dev/features/mantle-canonical-architecture.md` §3). Every vector is routed against it: a chunk
lands in the cell of its nearest anchor, a query fans out to its nearest anchors. It is what makes
one node's cells comparable with another's, which is why a **client seeds it** — the set arrives
as a file the client authored, and every node admits the same one.

Mantle does not derive, grow, reconcile or crosswalk a coordinate system, and that is deliberate.
Anchors are vectors, this process runs no model (the no-models rule), and anchors fitted locally
would mint region ids no peer computes. So the whole contract is three steps: seed a set, supply a
query vector in that set's space, get ranked results. This command is step one.

What stays dark without a set. The AnchorSet is the semantic arm's on switch, and a node nobody
has seeded has none. Until one is loaded:

    an artifact write        indexed LEXICALLY; the vector arm returns `skipped` and warns
    POST /artifacts/recall   narrows on the terms, then answers in QUERY-COVERAGE order —
                             `ordering: "coverage"`, each score the count of query stems matched
    the same call + `vector` the vector validates, reaches the ranker, and routes nowhere

Every one of those is a node reporting itself healthy. `--action inspect` below is how you tell
the two states apart before anything depends on the answer.

Run as `python -m mantle.system.manage_anchors --action <action>`. KEYS_DIR must already hold a
keyset — every action runs as the platform system principal, which is derived from it.

Actions
-------
load     Admit an AnchorSet from its single-file JSON form (`--path`), preserving every anchor
         id. Verifies each id against its own content and refuses the file whole if any
         disagrees. Idempotent — re-running writes nothing.
inspect  Show the current live AnchorSet (count, model, fingerprint, sample anchors), or report
         that this node has none.

"""

import argparse
import logging

from mantle.search.anchors import (
    AnchorSet,
    AnchorSetCorrupt,
    get_anchor_repo,
    get_live_anchorset,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("manage_anchors")

#: How many anchors a listing names individually before summarising the rest. One number, for
#: every listing in this CLI. Purely how much scrolls past a human; a different value would be
#: right only if these reports were being parsed by something, in which case the answer is a file,
#: not a cap.
_LISTED = 12


def _fingerprint(aset) -> str:
    """The set's fingerprint — imported here so `--help` costs nothing but argparse."""
    from mantle.search.anchors import anchorset_fingerprint
    return anchorset_fingerprint(aset)


def action_replace(path: str, dry_run: bool) -> None:
    """Make the set in `path` THE set: remove what is there, then admit this one.

    `load` adds, which grows a set rather than correcting one: a corrected 16-anchor set loaded
    over an existing 21 produces 37, with the superseded regions still routing, the command
    reporting "37 anchors in the store (37 new)", and every later query answering 200 out of a
    coordinate system nobody chose. This action is the way back from a mis-seed.

    It invalidates every cell. An anchor id is the cluster id, so a different set is a different
    coordinate system: existing cells were written under the old regions and are not comparable
    with anything routed under the new ones. `/status` reports it — `matches_cells` goes false and
    `indexed_fingerprint` stops equalling `fingerprint` — and the fix is to reindex, the same
    instruction `load` ends with and a heavier one here.

    The file is verified BEFORE anything is removed. `AnchorSet.load` recomputes every id from
    its own content and refuses the file whole if one disagrees, so a corrupt file cannot leave
    the node with no set at all — the failure mode a delete-then-load ordering would invite.
    """
    if not path:
        raise SystemExit(
            "--path is required for replace. It is the AnchorSet JSON your client wrote with "
            "`AnchorSet.save`, e.g. `--action replace --path anchors.json`."
        )
    try:
        aset = AnchorSet.load(path)
    except AnchorSetCorrupt as e:
        raise SystemExit(f"REFUSED: {e}") from None
    except (OSError, ValueError) as e:
        raise SystemExit(f"REFUSED: {path} is not a readable AnchorSet file: {e}") from None

    repo = get_anchor_repo()
    before = repo.count()
    logger.info("AnchorSet in %s: %d anchors, model=%s, dim=%d", path, len(aset), aset.model_id,
                aset.dim)
    logger.info("  fingerprint %s", _fingerprint(aset))
    logger.info("  this REPLACES %d anchor(s) already in the store", before)
    if dry_run:
        logger.info("[DRY-RUN] verified; nothing removed and nothing written.")
        return

    removed = repo.clear()
    logger.info("  removed %d", removed)

    def _progress(done: int, total: int) -> None:
        logger.info("  ... %d/%d anchors", done, total)

    try:
        repo.bulk_add(aset.anchors, progress=_progress)
    except AnchorSetCorrupt as e:
        raise SystemExit(f"REFUSED: {e}") from None

    from mantle.search.anchors import reset_anchorset
    reset_anchorset()
    logger.info("Replaced: %d anchors in the store (was %d). Ids preserved from the file.",
                repo.count(), before)
    logger.info("Send queries with space_id=%s; any other space is refused by name.",
                aset.model_id)
    logger.info("⚠ REINDEX NOW. Every cell was written under the old regions and routes into a "
                "coordinate system this set does not have; `/status` reports matches_cells false "
                "until they are rewritten.")


def action_load(path: str, dry_run: bool) -> None:
    """Admit an AnchorSet from `path`, ids intact.

    The file is the single-file JSON form `AnchorSet.save`/`load` emits and `ember ingest
    --anchors PATH` consumes. There is no second serialisation, so the set a client holds is the
    set every consumer of it reads.

    `AnchorSet.load` recomputes `uuid5(_ANCHOR_NS, sha256(label ‖ model_id ‖ embedding))` for
    every record and refuses the file whole when any stated id disagrees. That check is the
    reason this command exists rather than a documented `curl` loop: POST /artifacts has no `id`
    field and `workspace_service` assigns a fresh uuid4, so the documented route silently
    replaces exactly the value that makes two nodes' cells comparable.
    """
    if not path:
        raise SystemExit(
            "--path is required for load. It is the AnchorSet JSON your client wrote with "
            "`AnchorSet.save`, e.g. `--action load --path anchors.json`."
        )
    try:
        aset = AnchorSet.load(path)
    except AnchorSetCorrupt as e:
        # SystemExit, not a traceback: this is an answer about the file, and the operator's next
        # move is to get a good copy of it, not to read Mantle's stack.
        raise SystemExit(f"REFUSED: {e}") from None
    except (OSError, ValueError) as e:
        raise SystemExit(
            f"REFUSED: {path} is not a readable AnchorSet file: {e}. It must be the single-file "
            f"JSON `AnchorSet.save` writes — a `model_id`/`dim` header and an `anchors` list."
        ) from None

    logger.info("AnchorSet in %s: %d anchors, model=%s, dim=%d", path, len(aset), aset.model_id,
                aset.dim)
    logger.info("  fingerprint %s", _fingerprint(aset))
    if dry_run:
        logger.info("[DRY-RUN] verified; nothing written.")
        return

    repo = get_anchor_repo()
    before = repo.count()

    def _progress(done: int, total: int) -> None:
        logger.info("  ... %d/%d anchors", done, total)

    try:
        repo.bulk_add(aset.anchors, progress=_progress)
    except AnchorSetCorrupt as e:
        raise SystemExit(f"REFUSED: {e}") from None

    from mantle.search.anchors import reset_anchorset
    reset_anchorset()
    after = repo.count()
    logger.info("Loaded: %d anchors in the store (%d new). Ids preserved from the file.",
                after, after - before)
    logger.info("Send queries with space_id=%s; any other space is refused by name.",
                aset.model_id)
    logger.info("Reindex so already-stored artifacts reach the vector cells.")


def action_inspect(_dry_run: bool) -> None:
    aset = get_live_anchorset()
    if aset is None:
        # The answer an operator came here for, so it states the consequence rather than only
        # the fact: "no AnchorSet" alone reads as inventory, and the thing worth knowing is
        # that recall on this node is lexical and will stay lexical.
        logger.info("No live AnchorSet on this node. The semantic arm is OFF: writes index "
                    "lexically only, and POST /artifacts/recall answers lexically only.")
        logger.info("It is seeded, not generated here — no command in Mantle mints anchors, "
                    "and no later boot will produce one.")
        logger.info("TO FIX: `--action load --path anchors.json` with the set your client "
                    "wrote, then reindex. That route preserves the anchor ids, which are the "
                    "cluster ids; POST /artifacts assigns a fresh uuid4 and cannot.")
        return
    logger.info("Live AnchorSet: %d anchors, model=%s, dim=%d", len(aset), aset.model_id, aset.dim)
    logger.info("  queries must arrive with space_id=%s and dim=%d", aset.model_id, aset.dim)
    # The fingerprint is the whole reason two operators can compare nodes at all: it is a hash
    # over the anchor ids, so it says "same coordinate system" or "not" without either node
    # exporting an anchor, a label or a vector.
    fp = _fingerprint(aset)
    logger.info("  fingerprint %s", fp)
    from mantle.search.anchors import indexed_geometry
    rec = indexed_geometry()
    if not rec:
        logger.info("  cells: no indexing geometry recorded — nothing has been indexed under any "
                    "AnchorSet on this node yet.")
    elif rec.get("fingerprint") == fp:
        logger.info("  cells: written under this same set.")
    elif rec.get("model_id") != aset.model_id or int(rec.get("dim") or -1) != aset.dim:
        logger.error("  cells: written under %s (model %s, dim %s) — a DIFFERENT SPACE. The "
                     "semantic arm refuses until the sets agree; restore that set or drop the "
                     "cells and reindex.",
                     rec.get("fingerprint"), rec.get("model_id"), rec.get("dim"))
    else:
        logger.warning("  cells: written under %s (%s anchors) — the set has moved within the "
                       "same space since. Existing cells stay readable; REINDEX to cover the "
                       "anchors added since.", rec.get("fingerprint"), rec.get("anchors"))
    logger.info("  (manifold analysis is available via the Beacon add-on)")
    shown = aset.anchors[:_LISTED]
    for a in shown:
        logger.info("  [%s] %s  (%s)", a.tier, a.label, a.anchor_id[:12])
    if len(aset) > len(shown):
        logger.info("  ... and %d more", len(aset) - len(shown))


def main() -> None:
    # `--help` is where an operator meets this tool, and the module docstring is the only place
    # that says what an AnchorSet is for and what is dark without one. Handing it to argparse
    # verbatim means the two cannot drift apart, and it means the person who typed `--help`
    # because recall was returning lexical results reads the answer here.
    parser = argparse.ArgumentParser(
        prog="python -m mantle.system.manage_anchors",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--action", choices=["load", "replace", "inspect"], required=True,
        help=(
            "load: admit an AnchorSet file (--path), ids preserved and verified. "
            "inspect: whether this node has a live AnchorSet at all — the semantic arm is off "
            "when it does not. Neither MINTS anchors; nothing in Mantle does."
        ),
    )
    parser.add_argument("--path", help="AnchorSet JSON to admit (load).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="load: verify the file and report it without writing anything.",
    )
    args = parser.parse_args()
    # CLI: no request context. Anchor admission writes anchor artifacts through the
    # ordinary artifact path.
    from mantle.services.system_identity import system_acting_context

    with system_acting_context(scope="platform.manage-anchors"):
        if args.action == "replace":
            action_replace(args.path, args.dry_run)
        elif args.action == "load":
            action_load(args.path, args.dry_run)
        elif args.action == "inspect":
            action_inspect(args.dry_run)


if __name__ == "__main__":
    main()
