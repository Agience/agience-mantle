"""Give each stored upload the type of its file.

WHY. An uploaded file is an artifact of the file's type: the type is what a reader renders it as and
what a `type:` filter matches (`search/field_filters.py`). The upload paths set it
(`workspace_service.initiate_upload_and_create_artifact`, and the app for a small text file it
reads in whole); uploads stored before that carry the type in their context only, and show as a
generic card that no `type:` filter finds. This pass copies the context's type to the artifact, in
place: the content is unchanged, so no version is kept. New uploads need nothing; `tests/test_an_upload_is_typed_as_its_file.py` keeps them typed.

Additive and idempotent. An older Mantle reads a typed upload as it reads any typed artifact, so a
rollback needs no reverse.

    python -m mantle.system.type_uploads --dry-run
    python -m mantle.system.type_uploads
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

logger = logging.getLogger("type_uploads")


def _file_type(doc: dict) -> str:
    """The context's type of an untyped upload, else ""."""
    if doc.get("content_type"):
        return ""
    raw = doc.get("context")
    try:
        ctx = raw if isinstance(raw, dict) else json.loads(raw or "{}")
    except (TypeError, ValueError):
        return ""
    # An uploaded file names itself: stored ones (`content_source`) and small text files read in
    # whole by the app both carry the `filename` they were uploaded under.
    if not isinstance(ctx, dict) or not (ctx.get("filename") or ctx.get("content_source") == "agience-content"):
        return ""
    ct = ctx.get("content_type")
    return ct.strip() if isinstance(ct, str) else ""


def type_stored_uploads(db, *, dry_run: bool = False) -> dict:
    """Type every untyped upload from its context. Returns the counts."""
    todo = [(d, _file_type(d)) for d in db.artifacts.list_artifacts() if isinstance(d, dict)]
    todo = [(d, ct) for d, ct in todo if ct]
    counts = {"untyped": len(todo), "typed": 0}
    if dry_run:
        return counts
    for doc, ct in todo:
        row = dict(doc)
        row["content_type"] = ct
        db.artifacts.put_artifact(row)
        counts["typed"] += 1
    return counts


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
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
    counts = type_stored_uploads(db, dry_run=args.dry_run)
    logger.info("%s: %s", "would type" if args.dry_run else "typed", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
