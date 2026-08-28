"""Move a local MANTLE-SSE index from the retired file tree into its SQLite database.

Why this exists. `FilePostingStore` was replaced by `SqlitePostingStore` (see
`search/mantle/sse/sqlite_stores.py` for the four measured reasons). An install that searched
before the change has its index as a directory tree of `.enc` blobs, and the new store looks for a
database file — so without this pass the corpus reads as empty and the only recovery is a full
reindex.

It copies ciphertext and derives nothing. Every blob in the old tree is already
`nonce ‖ ciphertext ‖ tag` sealed under a key this pass cannot compute, bound to a slot AAD it
never constructs. The blind tokens and artifact ids are recoverable from the path (`decode_component`
is the inverse of the escaping the tree used), and that is the whole of what this needs: a slot's
identity and its bytes. So it needs **no oracle, no grant, and no acting principal** — unlike the
owner-index rebuild it replaces, which had to derive each owner's SSE key to invert posting lists and
therefore could not run without an identity that already read the corpus.

That is also why it is safe to re-run: copying a sealed blob to the row it belongs in is idempotent,
and `put_posting` is an upsert.

It does not delete the old tree. Deleting an operator's data is an operator's decision, and the
tree is the only copy until this has been verified. `--report` tells you what would move; the tree
can go once a recall answers.

    python -m mantle.system.manage_sse_index --report
    python -m mantle.system.manage_sse_index
    python -m mantle.system.manage_sse_index --segment draft
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, Iterator, Tuple

logger = logging.getLogger("manage_sse_index")

#: The layout the retired file store wrote, relative to `<root>/<prefix>/`:
#:
#:     {principal}/sse/posting/{aa}/{bb}/{blind_token}.enc
#:     {principal}/sse/manifests/{aa}/{bb}/{artifact_id}.enc
#:
#: A third path used to sit beside them — `{principal}/sse/stats.enc`, the per-owner BM25 corpus
#: aggregates — and a fourth, `{principal}/sse/index.enc`, the owner-index accelerator. Nothing
#: reads either any more, so neither is carried across: BM25 is gone and the accelerator is
#: unnecessary once a probe is an indexed lookup. They are left in the tree, not deleted.
_KIND_DIRS = {"posting": "posting", "manifests": "manifests"}


def _walk_slots(owner_dir: str, kind: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(decoded_name, path)`` for every ``.enc`` blob under one owner's ``kind`` directory.

    A name that will not decode is skipped with a warning rather than raising: it is not something
    the file store wrote, and one stray file must not stop the migration reaching the thousands of
    real slots behind it.
    """
    from mantle.search.mantle.sse.file_stores import decode_component

    base = os.path.join(owner_dir, "sse", _KIND_DIRS[kind])
    for dirpath, _dirnames, filenames in os.walk(base):
        for filename in filenames:
            if not filename.endswith(".enc"):
                continue                      # mkstemp leftovers, stats.enc, index.enc
            try:
                name = decode_component(filename[: -len(".enc")])
            except ValueError:
                logger.warning("skipping undecodable %s name %r under %s", kind, filename, dirpath)
                continue
            yield name, os.path.join(dirpath, filename)


def _owner_dirs(root_prefix: str) -> list:
    """Every principal directory in the old tree, decoded back to its principal id."""
    from mantle.search.mantle.sse.file_stores import decode_component

    try:
        names = os.listdir(root_prefix)
    except OSError:
        return []
    out = []
    for name in names:
        if not os.path.isdir(os.path.join(root_prefix, name)):
            continue
        try:
            out.append((decode_component(name), os.path.join(root_prefix, name)))
        except ValueError:
            logger.warning("skipping undecodable owner directory %r", name)
    return out


def migrate(root_prefix: str, store, *, report_only: bool = False) -> Dict[str, int]:
    """Copy every slot in the tree at *root_prefix* into *store*. Returns a count per kind."""
    counts = {"owners": 0, "postings": 0, "manifests": 0, "unreadable": 0}
    for principal_id, owner_dir in _owner_dirs(root_prefix):
        counts["owners"] += 1
        for kind, put in (("posting", "put_posting"), ("manifests", "put_manifest")):
            for name, path in _walk_slots(owner_dir, kind):
                if report_only:
                    counts["postings" if kind == "posting" else "manifests"] += 1
                    continue
                try:
                    with open(path, "rb") as fh:
                        blob = fh.read()
                except OSError:
                    logger.warning("owner %s: cannot read %s; skipped", principal_id, path,
                                   exc_info=True)
                    counts["unreadable"] += 1
                    continue
                if not blob:
                    # An empty file is not a sealed blob. Copying it would put a row in the
                    # database that fails GCM authentication on read — indistinguishable from
                    # tampering — where leaving it out simply means the slot is absent.
                    counts["unreadable"] += 1
                    continue
                getattr(store, put)(principal_id, name, blob)
                counts["postings" if kind == "posting" else "manifests"] += 1
    return counts


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true",
                    help="count what would move; write nothing")
    ap.add_argument("--segment", default="committed",
                    help="which index segment to migrate (committed / draft / archived)")
    ap.add_argument("--generation", action="store_true",
                    help="report which analyzer generation wrote this segment, then exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    from mantle.search.mantle.sse.sqlite_stores import SqlitePostingStore
    from mantle.search.mantle.wiring import _segment_prefixes, local_sse_path, local_sse_root

    _, sse_prefix = _segment_prefixes(args.segment)
    root_prefix = os.path.join(local_sse_root(), sse_prefix)
    db_path = local_sse_path(sse_prefix)

    # Answered before the old-tree check, because "which analysis wrote this index" is a question
    # about the DATABASE and has nothing to do with whether a pre-SQLite tree is still lying around.
    # `wiring` already logs a mismatch once at open; that line is easy to miss and impossible to ask
    # for again, which is the whole reason this flag exists.
    if args.generation:
        from mantle.search.mantle.sse.posting import analyzer_generation_of
        from mantle.search.mantle.sse.tokenizer import ANALYZER

        if not os.path.isfile(db_path):
            print("segment %s: no index at %s" % (args.segment, db_path))
            return 0
        store = SqlitePostingStore(db_path)
        try:
            found = analyzer_generation_of(store)
            owners = len(store.list_owners())
        finally:
            store.close()
        print("segment    %s" % args.segment)
        print("index      %s" % db_path)
        print("owners     %d" % owners)
        print("written by %s" % ("generation %d" % found if found is not None
                                 else ("generation 1 (unstamped, and populated)" if owners
                                       else "nothing yet (unstamped and empty)")))
        print("this build generation %d" % ANALYZER)
        if found == ANALYZER or (found is None and not owners):
            print("")
            print("CURRENT — nothing to do.")
            return 0
        print("")
        print("STALE. A blind token is an HMAC of an analysed term, so there is no in-place")
        print("migration, and content indexed under the old analysis is unreachable — silently,")
        print("as an empty result. Rebuild:")
        print('    python -c "from mantle.search.init_search import reindex_all_artifacts; '
              'reindex_all_artifacts()"')
        return 1

    if not os.path.isdir(root_prefix):
        logger.info("no old index tree at %s — nothing to migrate for segment %r",
                    root_prefix, args.segment)
        return 0

    logger.info("segment %s: tree %s -> database %s", args.segment, root_prefix, db_path)
    store = None if args.report else SqlitePostingStore(db_path)
    try:
        counts = migrate(root_prefix, store, report_only=args.report)
    finally:
        if store is not None:
            store.close()

    verb = "would move" if args.report else "moved"
    logger.info("%s %d posting slot(s) and %d manifest(s) across %d owner(s); %d unreadable",
                verb, counts["postings"], counts["manifests"], counts["owners"],
                counts["unreadable"])
    if not args.report:
        logger.info("the old tree at %s is LEFT IN PLACE — verify a recall answers, then remove it",
                    root_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
