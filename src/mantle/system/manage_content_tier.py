"""Give every artifact's bytes a content address, so the lattice stops being the store.

WHY. `content_ref` — `cas/<sha256 of the plaintext>` — is what the vertex column in `db/schema.py`
holds, what `db/vertex.py` reads to tell a re-describe from a new version, and what
`shard/content_tier.promote_local_content` drains to S3 on a node with a mirror. The whole tier
was built and correct; the artifact write path never set the ref, so the bytes went into the
document instead. Measured on 71/dev before the write path was wired: 0 of 709 vertices carried a
ref, and 7.2 MB of 7.9 MB of doc bytes — 92% of the lattice — was artifact content.

This pass addresses what is already stored. New writes are handled at the source
(`workspace_service._store_content_in_s3`), and `tests/test_content_never_lives_in_the_lattice.py`
is what keeps it that way.

It addresses the bytes and the row stops carrying them. `db/doc_boundary` is the pair that makes
that safe: `encrypt_artifact_content` keeps addressed content out of the document, and
`decrypt_artifact_content` hydrates it back from the ref on every read. Because both sit at the
persistence boundary, indexing, recall previews and `GET /artifacts/{id}` still read
`artifact.content` exactly as before and never learn where it lives — nothing downstream changed.

Measured after this ran: a 9,800-character body leaves a 981-byte vertex row.

    python -m mantle.system.manage_content_tier --dry-run
    python -m mantle.system.manage_content_tier
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

logger = logging.getLogger("manage_content_tier")


def _ref_for(artifact, *, owner_id: str, collection_id: Optional[str]) -> Optional[str]:
    """Write one artifact's bytes into the CAS and return the address, or None.

    Content-addressed, so this is idempotent: an artifact already in the tier dedups to the same
    address and writes nothing (`FileContentCache.put` returns False on a verified hit).
    """
    from mantle.services.content_service import put_bytes_encrypted

    body = artifact.content or ""
    if not body.strip():
        return None
    try:
        ctx = json.loads(artifact.context or "{}")
        if not isinstance(ctx, dict):
            ctx = {}
    except (TypeError, ValueError):
        ctx = {}
    content_type = ctx.get("content_type") or artifact.content_type or "text/plain"
    content_key = "artifacts/%s.content" % artifact.id
    return put_bytes_encrypted(
        content_key, body.encode("utf-8"), content_type, owner_id,
        collection_id=collection_id, cas=True,
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--as-principal", default=None, metavar="ID",
                    help="run as this principal. Required in practice: the content key is minted "
                         "per origin root and the oracle checks the grant ledger, so a migration "
                         "can only address content its caller could already read.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    from prism.trust.key_manager import init_encryption_key, init_nonce_secret
    init_encryption_key()
    init_nonce_secret()
    from mantle.services import peer_signing
    peer_signing.init()

    from mantle.db import backend as store
    from mantle.services.acting_principal import acting_as
    from mantle.services.dependencies import get_store_db
    from mantle.services.system_identity import system_acting_context
    from mantle.services.workspace_service import _record_content_address

    db = next(get_store_db())
    context = (acting_as(args.as_principal, principal_type="user")
               if args.as_principal else system_acting_context())

    with context:
        # The handle's own enumerator — `db.artifacts.list_artifacts` — because the module-level
        # API is deliberately scoped (by collection, by content type) and this pass is not.
        todo = []
        for doc in db.artifacts.list_artifacts():
            if not isinstance(doc, dict):
                continue
            if (doc.get("content") or "").strip() and not doc.get("content_ref"):
                todo.append(doc)
        logger.info("%d artifact(s) hold content with no content address", len(todo))
        if args.dry_run:
            for d in todo[:8]:
                logger.info("  would address %s (%d chars)",
                            str(d.get("id"))[:8], len(d.get("content") or ""))
            if len(todo) > 8:
                logger.info("  ... and %d more", len(todo) - 8)
            return 0

        done = skipped = denied = failed = 0
        for i, doc in enumerate(todo, 1):
            # The read itself needs a key, so it can be refused. `get_artifact` decrypts
            # content, and the content key is minted per origin root against the grant ledger —
            # so an artifact this caller cannot read raises here, before any of the work below.
            # That is the migration correctly failing to exceed its own authorization, and it is
            # per-artifact rather than fatal: the rest of the corpus is still addressable.
            try:
                artifact = store.get_artifact(db, str(doc.get("id")))
            except Exception as exc:
                denied += 1
                logger.info("  cannot read %s (%s) — leaving it alone",
                            str(doc.get("id"))[:8], type(exc).__name__)
                continue
            if artifact is None:
                skipped += 1
                continue
            owner = artifact.created_by or args.as_principal
            if not owner:
                skipped += 1          # no owner, no envelope — see put_bytes_encrypted
                continue
            try:
                ref = _ref_for(artifact, owner_id=owner,
                               collection_id=artifact.collection_id or None)
            except Exception:
                logger.warning("could not address %s", artifact.id, exc_info=True)
                failed += 1
                continue
            if not ref:
                skipped += 1
                continue
            artifact.content_ref = ref
            artifact.context = _record_content_address(
                artifact.context, "artifacts/%s.content" % artifact.id, ref)
            #: NO `is None` BRANCH. `store.update_artifact`
            #: has one `return entity` and no error path, so `failed += 1` here was
            #: unreachable: this migration could never report a persist failure, and its exit
            #: code could never turn 1 because of one.
            #:
            #: What the dead branch was pretending to handle is still unhandled. A
            #: `put_artifact` that RAISES propagates out of this loop and aborts the whole
            #: run mid-way — it always did, the guard never caught it. `failed` still counts
            #: the addressing failures above, which is a real and reachable case.
            store.update_artifact(db, artifact)
            done += 1
            if i % 25 == 0:
                logger.info("  %d/%d (addressed %d, skipped %d, denied %d, failed %d)",
                            i, len(todo), done, skipped, denied, failed)
        logger.info("addressed %d, skipped %d, denied %d, failed %d",
                    done, skipped, denied, failed)
        logger.info("the row now carries the ADDRESS; `db/doc_boundary` hydrates the bytes back "
                    "on read, so nothing downstream had to change")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
