"""Rebuild the search index once for each change to what it holds.

WHY. Search finds an artifact by the terms its index entries were written with, and an entry is
written when the artifact is saved. A change to how an artifact is indexed reaches only what is
saved after it; everything stored before keeps its old entries until it is indexed again. 0.1.6
splits a hyphenated name into its parts (`quasar-lantern-731` is found by `quasar lantern`), so an
artifact saved before it is found only by the whole word.

:data:`GENERATION` names the last such change. The release runs this module as a migration on every
promote; it indexes every live artifact again, each as its creator (:func:`reindex_as_owners`), when
the store records an older generation, then records this one, so each change is rebuilt once. Bump
it with any change that needs the corpus indexed again.

A pass with failures records nothing and exits non-zero, so the release stops and the next one tries
again. An artifact that cannot be read on this node (its body held in an object store the node
cannot reach, as on the release candidate's copy), or that its creator's grants no longer reach, is
counted apart, not as a failure.

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


def reindex_as_owners(db) -> dict:
    """Index every live artifact again, as a principal whose grants reach it. Returns the counts.

    Grants are keys: an artifact is read, and its index entries written, under keys issued only to a
    principal its grants reach, and the platform's own principal holds no grant to a person's
    content. 2026-10-29, on the release candidate: the full pass (`reindex_all_artifacts`, which
    runs as the platform) was refused 273 times and indexed nothing.

    Each artifact is indexed as its creator, else as the owner of its root: entries are written under
    the root's key (`resolve_cell_principal`), and a contribution to someone else's workspace can be
    written only by that workspace's owner (69 of them on the release candidate). Writing an entry
    twice is harmless, so the second try costs nothing.

    An artifact whose body cannot be read here (held only in an object store this node cannot
    reach) is `unreadable`; one whose index write every candidate is refused is `denied` (on the
    release candidate, 69 of the platform's own artifacts: type definitions, operators, platform
    collections). Neither is a failure: retrying cannot change either."""
    from mantle.db.doc_boundary import ContentDecryptionError as SealError
    from mantle.db.lattice_api import from_lattice_doc
    from mantle.entities.artifact import Artifact
    from mantle.search.ingest.pipeline_unified import index_artifact, is_indexable
    from mantle.search.mantle.oracle import KeyCustodyDenied
    from mantle.services.acting_principal import acting_as
    from mantle.services.content_service import ContentDecryptionError, ContentStoreUnavailable
    from mantle.services.system_identity import system_acting_context

    counts = {"indexed": 0, "skipped": 0, "failed": 0, "unreadable": 0, "denied": 0, "ownerless": 0}
    root_owners: dict[str, str | None] = {}

    def candidates(raw: dict) -> list[str]:
        root = raw.get("origin_root")
        if root and root not in root_owners:
            doc = db.artifacts.get_artifact(root)
            root_owners[root] = doc.get("created_by") if isinstance(doc, dict) else None
        found = [raw.get("created_by"), root_owners.get(root) if root else None]
        return [str(c) for i, c in enumerate(found) if c and c not in found[:i]]

    class _Refusals(logging.Handler):
        """The indexer reports a refused write as a failed arm and logs the exception; this reads
        that exception's type, so a key refusal is told from a fault."""
        def __init__(self):
            super().__init__(logging.WARNING)
            self.refused = self.other = False

        def emit(self, record):
            exc = record.exc_info[1] if record.exc_info else None
            if isinstance(exc, KeyCustodyDenied):
                self.refused = True
            elif exc is not None:
                self.other = True

    def index_as(owner: str, kind: str, raw: dict) -> str:
        """`indexed`, `skipped`, `failed`, `unreadable` or `denied`."""
        watch = _Refusals()
        pipeline_log = logging.getLogger("mantle.search.ingest.pipeline_unified")
        pipeline_log.addHandler(watch)
        try:
            with (system_acting_context(scope="platform.reindex") if kind == "platform"
                  else acting_as(owner, principal_type=kind)):
                artifact = from_lattice_doc(dict(raw), Artifact)
                if artifact is None or not is_indexable(artifact):
                    return "skipped"
                outcome = index_artifact(artifact, artifact.collection_id or artifact.id, is_head=True)
        except KeyCustodyDenied:
            return "denied"
        except (SealError, ContentDecryptionError, ContentStoreUnavailable):
            return "unreadable"
        except Exception as exc:  # noqa: BLE001 — per artifact; the rest of the corpus still indexes
            logger.warning("  cannot index %s as %s (%s: %s)", str(raw.get("id"))[:8], owner[:8],
                           type(exc).__name__, exc)
            return "failed"
        finally:
            pipeline_log.removeHandler(watch)
        if outcome.failed:
            return "denied" if watch.refused and not watch.other else "failed"
        return "skipped" if outcome.wrote_nothing else "indexed"

    for raw in db.artifacts.list_artifacts():
        owners = candidates(raw) if isinstance(raw, dict) else []
        if not owners:
            counts["ownerless"] += 1
            continue
        result = "failed"
        for owner in owners:
            # A person's artifact is theirs as a user; the platform's (type definitions, operators,
            # its own collections) belongs to a service principal.
            for kind in ("user", "service"):
                result = index_as(owner, kind, raw)
                if result in ("indexed", "skipped", "unreadable"):
                    break
            if result in ("indexed", "skipped", "unreadable"):
                break
        if result not in ("indexed", "skipped", "unreadable"):
            # The platform's own artifacts (type definitions, operators, its collections) are
            # written under the platform principal's grants.
            result = index_as("", "platform", raw)
        if result in ("failed", "denied"):
            logger.warning("  cannot index %s (%s) as %s", str(raw.get("id"))[:8], raw.get("content_type"),
                           " or ".join(o[:8] for o in owners))
        counts[result] += 1
    return counts


def reindex_once(db, settings, *, rebuild: Callable[[], dict] | None = None,
                 check: bool = False) -> dict:
    """Rebuild the index when the store records an older generation. Returns what was done."""
    if rebuild is None:
        rebuild = lambda: reindex_as_owners(db)  # noqa: E731
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
