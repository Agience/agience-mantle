"""Rebuild the search index once for each change to what it holds.

WHY. Search finds an artifact by the terms its index entries were written with, and an entry is
written when the artifact is saved. A change to how an artifact is indexed reaches only what is
saved after it; everything stored before keeps its old entries until it is indexed again. 0.1.6
splits a hyphenated name into its parts (`quasar-lantern-731` is found by `quasar lantern`), so an
artifact saved before it is found only by the whole word.

:data:`GENERATION` names the last such change. The release runs this module as a migration on every
promote; it rebuilds the index (`search.init_search.reindex_all_artifacts`, forced past the marker
that keeps a boot from rebuilding twice) when the store records an older generation, then records
this one, so each change is rebuilt once. Bump it with any change that needs the corpus indexed
again.

A pass with failures records nothing and exits non-zero, so the release stops and the next one tries
again. An artifact that cannot be read on this node (its body held in an object store the node
cannot reach, as on the release candidate's copy) is counted as unreadable, not as a failure.

    python -m mantle.system.reindex --check
    python -m mantle.system.reindex
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Callable

logger = logging.getLogger("reindex")

#: The index generation this build writes.
#:
#:   1  hyphenated names indexed by their parts as well (0.1.6)
GENERATION = 1

#: The platform setting recording the generation the stored index was last rebuilt for.
SETTING = "search.reindex.generation"


def reindex_once(db, settings, *, rebuild: Callable[[], dict] | None = None,
                 check: bool = False) -> dict:
    """Rebuild the index when the store records an older generation. Returns what was done."""
    if rebuild is None:
        from mantle.search.init_search import reindex_all_artifacts
        rebuild = lambda: reindex_all_artifacts(force=True)  # noqa: E731
    recorded = settings.get_int(SETTING, 0)
    if recorded >= GENERATION:
        return {"generation": recorded, "rebuilt": False}
    if check:
        return {"generation": recorded, "rebuilt": False, "would_rebuild_to": GENERATION}
    counts = rebuild()
    if counts.get("failed"):
        return {"generation": recorded, "rebuilt": False, **counts}
    settings.set_setting(db, SETTING, str(GENERATION), category="search")
    return {"generation": GENERATION, "rebuilt": True, **counts}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="say whether a rebuild is due; change nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    from prism.trust.key_manager import init_encryption_key, init_nonce_secret
    init_encryption_key()
    init_nonce_secret()
    from mantle.services import peer_signing
    peer_signing.init()
    from mantle.services.dependencies import get_store_db
    from mantle.services.platform_settings_service import settings

    db = next(get_store_db())
    settings.load_all(db)
    result = reindex_once(db, settings, check=args.check)
    logger.info("reindex: %s", result)
    return 1 if result.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
