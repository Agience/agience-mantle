"""Seal the titles already stored in plain text.

WHY. A title is part of an artifact's content (declared by the root type, `*/*`): the write path
seals it under the body's key (`db/doc_boundary.seal_content_fields`) and every authorized read
restores it. Rows written before that carry their title in plain text in the artifact row. This
pass reads each such row through the ordinary path and writes it back, so the write boundary seals
it. New writes need nothing; `tests/test_a_title_is_content_and_sealed_at_rest.py` keeps them
sealed.

Each row is handled as its creator, else as the owner of its root: a title is sealed under its
root's key, and a creator whose grant was revoked (a contributor removed from a shared workspace,
whose note stays) no longer reaches it, while the root's owner does. A row whose key the oracle
hands to neither is reported and left alone, never forced. Only the title is sealed: the
body is left as it is stored.

`--unseal` is the reverse, which the release runs before rolling Mantle back to a version that
does not read sealed titles; then the forward pass seals them again on the next promote.

    python -m mantle.system.seal_titles --dry-run
    python -m mantle.system.seal_titles
    python -m mantle.system.seal_titles --unseal
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

logger = logging.getLogger("seal_titles")


def _plain_title(doc: dict) -> bool:
    """Whether the row holds anything the store seals in plaintext: a sealable context key or a
    writer-supplied top-level field (`doc_boundary.sealable_keys`, `SEALED_DOC_FIELDS`)."""
    from mantle.db.doc_boundary import sealable_doc_fields, sealable_keys
    raw = doc.get("context")
    try:
        ctx = raw if isinstance(raw, dict) else json.loads(raw or "{}")
    except (TypeError, ValueError):
        return False
    if not isinstance(ctx, dict):
        return False
    return bool(sealable_keys(doc, ctx)) or bool(sealable_doc_fields(doc, ctx))


def _still_sealed(doc: dict) -> bool:
    from mantle.db.doc_boundary import SEALED
    raw = doc.get("context")
    try:
        ctx = raw if isinstance(raw, dict) else json.loads(raw or "{}")
    except (TypeError, ValueError):
        return False
    return isinstance(ctx, dict) and SEALED in ctx


def _holders(db, doc: dict, roots: dict) -> list[str]:
    """Who may hold the row's key: its creator, then the owner of its root. 2026-10-29, on the
    release candidate: a note whose contributor had been revoked could not be unsealed as them."""
    root = doc.get("origin_root")
    if root and root not in roots:
        r = db.artifacts.get_artifact(root)
        roots[root] = r.get("created_by") if isinstance(r, dict) else None
    found = [doc.get("created_by"), roots.get(root) if root else None]
    return [str(h) for i, h in enumerate(found) if h and h not in found[:i]]


def seal_stored_titles(db, *, dry_run: bool = False) -> dict:
    """Seal every stored plain-text title. Returns the counts.

    Each title is sealed under the row's key, as the write boundary seals it, and the row is written
    as it is: the body is neither read nor rewritten, so a body held only in the object store needs
    no object store here."""
    from mantle.db.doc_boundary import seal_content_fields
    from mantle.services.acting_principal import acting_as

    todo = [d for d in db.artifacts.list_artifacts() if isinstance(d, dict) and _plain_title(d)]
    counts = {"plain": len(todo), "sealed": 0, "denied": 0, "ownerless": 0}
    if dry_run:
        return counts
    roots: dict = {}
    for doc in todo:
        holders = _holders(db, doc, roots)
        if not holders:
            counts["ownerless"] += 1
            continue
        why = ""
        for holder in holders:
            raw = dict(doc)
            try:
                with acting_as(holder, principal_type="user"):
                    seal_content_fields(raw)
            except Exception as exc:  # noqa: BLE001 — per row; the next holder, then the next row
                why = type(exc).__name__
                continue
            db.artifacts.put_artifact(raw)
            counts["sealed"] += 1
            break
        else:
            counts["denied"] += 1
            logger.info("  cannot seal %s (%s) — leaving it alone", str(doc.get("id"))[:8], why)
    return counts


def unseal_stored_titles(db, *, dry_run: bool = False) -> dict:
    """The reverse, run before a rollback to a Mantle that does not read sealed titles: each sealed
    title put back in plain text, read as the row's creator and written to the row as it is, past
    the write boundary that would seal it again. Returns the counts."""
    from mantle.db.doc_boundary import SEALED, unseal_content_fields
    from mantle.services.acting_principal import acting_as

    def sealed(doc: dict) -> bool:
        raw = doc.get("context")
        try:
            ctx = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        except (TypeError, ValueError):
            return False
        return isinstance(ctx, dict) and SEALED in ctx

    todo = [d for d in db.artifacts.list_artifacts() if isinstance(d, dict) and sealed(d)]
    counts = {"sealed": len(todo), "unsealed": 0, "denied": 0, "ownerless": 0}
    if dry_run:
        return counts
    roots: dict = {}
    for doc in todo:
        holders = _holders(db, doc, roots)
        if not holders:
            counts["ownerless"] += 1
            continue
        for holder in holders:
            raw = dict(doc)
            with acting_as(holder, principal_type="user"):
                unseal_content_fields(raw)
            if not _still_sealed(raw):
                db.artifacts.put_artifact(raw)
                counts["unsealed"] += 1
                break
        else:
            counts["denied"] += 1      # the oracle would open it for neither; left sealed, not emptied
            logger.info("  cannot unseal %s — leaving it sealed", str(doc.get("id"))[:8])
    return counts


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unseal", action="store_true",
                    help="the reverse: put sealed titles back in plain text, before a rollback")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    from prism.trust.key_manager import init_encryption_key, init_nonce_secret
    init_encryption_key()
    init_nonce_secret()
    from mantle.services import peer_signing
    peer_signing.init()
    from mantle.services.dependencies import get_store_db

    db = next(get_store_db())
    if args.unseal:
        counts = unseal_stored_titles(db, dry_run=args.dry_run)
        logger.info("%s: %s", "would unseal" if args.dry_run else "unsealed", counts)
    else:
        counts = seal_stored_titles(db, dry_run=args.dry_run)
        logger.info("%s: %s", "would seal" if args.dry_run else "sealed", counts)
    return 0 if not counts["denied"] else 1


if __name__ == "__main__":
    sys.exit(main())
