#!/usr/bin/env python3
"""
manage_erasure.py -- inventory, then remove, everything grounded at one person.

    python -m mantle.scripts.manage_erasure --person <principal>              # report only
    python -m mantle.scripts.manage_erasure --person <principal> --apply      # delete it
    python -m mantle.scripts.manage_erasure --person <principal> --include-identity --apply

The command-line front door to `shard/erasure.py`; `POST /system/erasure/{person_id}` is the
other one, and both call the same primitive with the same defaults, so the report a human reads
here is the report the API would have produced.

A worked example — reset a member, keeping their identity:

    $ export MANTLE_LATTICE_PATH=/var/lib/mantle/lattice.db KEYS_DIR=/var/lib/mantle/keys
    $ python -m mantle.scripts.manage_erasure --person 4c1f...9a

      store  : /var/lib/mantle/lattice.db
      person : 4c1f...9a
      also known as: 4c1f...9a, 6b20...11

      DRY RUN -- nothing below has been deleted.

      private          12   artifacts in your private collection
      authored         41   artifacts you authored
      conversation      7   messages and conversations
      identity          0   your person artifact itself
      -------------------
      total            60   of 3,914 vertices scanned

      not yours         9   registry/commons artifacts this person touched, left in place

      Re-run with --apply to delete the 60 artifact(s) above.

    $ python -m mantle.scripts.manage_erasure --person 4c1f...9a --apply
      ...
      APPLIED: 60 removed, 0 failed -- COMPLETE.

**Dry run is the default and `--apply` is the only thing that changes it.** The inventory is the
product; the deletion is a separate decision made by a human who has read it. `--dry-run` exists
so a scripted invocation can say what it means rather than relying on an absent flag.

Two different acts, and the tool makes you pick. Without `--include-identity` this is a RESET:
the person stays and everything grounded at them goes. With it, the person artifact goes too, and
an applied run also drops the standalone identity record (email, provider subject, password hash),
which lives on the identity plane rather than in the artifact graph and so is out of the erasure
primitive's reach.

Erasure is defined positively -- what is provably grounded at this person -- never by exclusion,
because an exclusion list misses whatever class nobody thought to name. What it finds and
deliberately leaves alone (the commons, the registry, the operators) is reported under
"not yours" rather than passed over in silence.

Run it where the store is. This opens the lattice directly (`MANTLE_LATTICE_PATH`), so run it on
the node -- in the container, or anywhere that env and `KEYS_DIR` resolve to the real ones. The
path it opened is printed before anything is read, because erasing the wrong store is silent.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger("mantle.erasure")


def connect():
    """Open THE store — the standalone lattice (MANTLE_LATTICE_PATH / MANTLE_ORIGIN env)."""
    from mantle.db import backend
    return backend.store_handle()


def _print_report(report: dict, *, applied: bool) -> None:
    from mantle.shard.erasure import CLASSES

    ids = report.get("ids") or []
    if len(ids) > 1:
        print("  also known as: %s" % ", ".join(ids))
    print()
    if not applied:
        print("  DRY RUN -- nothing below has been deleted.\n")

    counts = report.get("counts") or {}
    for key, description in CLASSES:
        print("  %-14s %5d   %s" % (key, counts.get(key, 0), description))
    print("  " + "-" * 19)
    print("  %-14s %5d   of %d vertices scanned"
          % ("total", report.get("total", 0), report.get("scanned", 0)))

    not_yours = report.get("not_yours") or []
    if not_yours:
        print("\n  %-14s %5d   registry/commons artifacts this person touched, left in place"
              % ("not yours", len(not_yours)))

    unresolved = report.get("unresolved") or []
    if unresolved:
        # Named, not counted: an artifact the sweep could not classify is the one case where the
        # positive definition cannot say "this is not yours", so it must not read as "none found".
        print("\n  UNRESOLVED (%d) -- neither claimed nor cleared by this run:" % len(unresolved))
        for aid in unresolved:
            print("    %s" % aid)


def erase(db, person: str, *, apply: bool, include_identity: bool) -> int:
    from mantle.shard import erasure

    report = erasure.erase(db, person, apply=apply, include_identity=include_identity)
    _print_report(report, applied=apply)

    if not apply:
        total = report.get("total", 0)
        print()
        if total:
            print("  Re-run with --apply to delete the %d artifact(s) above." % total)
        else:
            print("  Nothing is grounded at this person. --apply would delete nothing.")
        return 0

    failed = report.get("failed") or []
    complete = bool(report.get("complete"))

    identity_note = ""
    if include_identity:
        # The identity record is a separate plane, so an applied full erasure reports it
        # separately -- claiming the sweep complete while the email row survives is the one
        # thing this tool must never do.
        from mantle.db import identity_backend as identity_store
        try:
            removed = bool(identity_store.delete_person(db, person))
            identity_note = ("identity record removed" if removed
                             else "no identity record for this id")
        except Exception as e:
            complete = False
            identity_note = "identity record delete FAILED (%s: %s)" % (type(e).__name__, e)

    print("\n  APPLIED: %d removed, %d failed -- %s."
          % (report.get("removed", 0), len(failed), "COMPLETE" if complete else "INCOMPLETE"))
    if identity_note:
        print("  %s" % identity_note)
    for line in failed:
        print("    ! %s" % line)
    if not complete:
        # A partial erasure that is reported is recoverable; one reported as complete is not.
        print("\n  This person is NOT fully erased. Re-run to retry the remainder.", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, built apart from `main` so the defaults are directly assertable.

    Which flag is the default is the whole safety property of this tool, and a default is only
    a claim until something reads it back off the parser."""
    parser = argparse.ArgumentParser(
        description="Inventory or erase everything grounded at one person. "
                    "Reports only unless --apply is given — see the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--person", required=True,
        help="The principal to erase. Either the raw claim or the resolved person artifact id — "
             "both are resolved to the same higgs and both halves are swept.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true",
                      help="Actually delete. Without this the run reports and changes nothing.")
    mode.add_argument("--dry-run", action="store_true",
                      help="Report what would happen; write nothing. The default — this flag "
                           "only lets a scripted call say so explicitly.")
    parser.add_argument("--include-identity", action="store_true",
                        help="Full erasure rather than a reset: the person artifact goes too, and "
                             "an applied run also drops the standalone identity record.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Say which store, before touching it. An unset MANTLE_LATTICE_PATH resolves to a relative
    # default and would open (or CREATE) a store nobody meant — which then reports a clean,
    # entirely fictional "nothing is grounded at this person".
    path = os.getenv("MANTLE_LATTICE_PATH")
    print(f"\n  store  : {os.path.abspath(path) if path else '<MANTLE_LATTICE_PATH UNSET — refusing>'}")
    if not path:
        print("\n  Set MANTLE_LATTICE_PATH. Unset, mantle opens a relative default, and this tool\n"
              "  would report on a store nobody asked about.", file=sys.stderr)
        return 2
    print(f"  person : {args.person}")

    db = connect()

    # No acting principal is established, and none is needed. Erasure works on the store's own
    # rows: the inventory is a read over `vertex` and the deletion is a row delete. Nothing here
    # derives a content key or resolves a grant, so a store that cannot yet name a system
    # principal — the shape a half-provisioned or already-emptied node is in, and exactly the
    # shape someone runs this against — can still be inventoried and cleared.
    return erase(db, args.person, apply=args.apply,
                 include_identity=args.include_identity)


if __name__ == "__main__":
    sys.exit(main())
