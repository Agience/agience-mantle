"""Seal the titles already stored in plain text.

WHY. A title is part of an artifact's content (declared by the root type, `*/*`): the write path
seals it under the body's key (`db/doc_boundary.seal_content_fields`) and every authorized read
restores it. Rows written before that carry their title in plain text in the artifact row. This
pass reads each such row through the ordinary path and writes it back, so the write boundary seals
it. New writes need nothing; `tests/test_a_title_is_content_and_sealed_at_rest.py` keeps them
sealed.

Each row is handled as its creator, else as the owner of its root: a title is sealed under its
root's key, and a creator whose grant was revoked (a contributor removed from a shared workspace,
whose note stays) no longer reaches it, while the root's owner does. The platform's own collections
are first granted to the platform system principal (`ensure_platform_principal_grant`), which the
pass acts as last. A row whose key the oracle hands to none of them is reported and left alone,
never forced. Only the title is sealed: the body is left as it is stored.

A grant is not content: its name, notes and invite address are sealed under the node's key
(`db/node_seal`, `lattice_api.GRANT_SEALED_FIELDS`), which needs no actor.

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


def _is_grant(doc: dict) -> bool:
    from mantle.db.lattice_api import _GRANT_CT
    return doc.get("content_type") == _GRANT_CT


def _plain_grant(doc: dict) -> bool:
    from mantle.db.lattice_api import GRANT_SEALED_FIELDS
    return any(doc.get(k) not in (None, "") for k in GRANT_SEALED_FIELDS)


def _plain_title(doc: dict) -> bool:
    """Whether the row holds anything the store seals in plaintext: a sealable context key or a
    writer-supplied top-level field (`doc_boundary.sealable_keys`, `SEALED_DOC_FIELDS`), or, on a
    grant, a field the node's key seals."""
    from mantle.db.doc_boundary import sealable_doc_fields, sealable_keys
    if _is_grant(doc):
        return _plain_grant(doc)
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


def _seal_grant(db, doc: dict) -> bool:
    """Seal a grant's fields under the node's key, written only if they open again."""
    from mantle.db import node_seal
    from mantle.db.lattice_api import GRANT_SEALED_FIELDS
    raw = dict(doc)
    node_seal.seal_fields(raw, GRANT_SEALED_FIELDS)
    check = dict(raw)
    node_seal.unseal_fields(check)
    if any(check.get(k) != doc.get(k) for k in GRANT_SEALED_FIELDS):
        return False
    db.artifacts.put_artifact(raw)
    return True


def _grant_the_platform_its_collections(db, rows: list) -> int:
    """Every collection the platform created, granted to the platform system principal."""
    from mantle.config import AGIENCE_PLATFORM_USER_ID
    from mantle.entities.collection import COLLECTION_CONTENT_TYPE
    from mantle.services.seed_provisioning.user_provisioning import ensure_platform_principal_grant
    return sum(1 for d in rows
               if d.get("content_type") == COLLECTION_CONTENT_TYPE
               and d.get("created_by") == AGIENCE_PLATFORM_USER_ID
               and ensure_platform_principal_grant(db, str(d.get("id"))))


def _acting_as_each(holders: list):
    """Each way to act for a row, in order: every holder as a person, then as a service, then the
    platform's own principal. The platform's own rows (operators, its collections, type definitions)
    belong to service principals or to no one, and the key service opens them for those, not for a
    person of the same id. 2026-10-30, on the release candidate: the rollback pass, acting only as a
    person, could open 199 of 242 rows; 347 rows with no owner were left in plain text. The same
    order as `mantle.system.reindex`."""
    from mantle.services.acting_principal import acting_as
    from mantle.services.system_identity import system_acting_context
    for holder in holders:
        yield lambda h=holder: acting_as(h, principal_type="user")
        yield lambda h=holder: acting_as(h, principal_type="service")
    yield lambda: system_acting_context(scope="platform.seal")


def seal_stored_titles(db, *, dry_run: bool = False) -> dict:
    """Seal every stored plain-text title. Returns the counts.

    Each title is sealed under the row's key, as the write boundary seals it, and the row is written
    as it is: the body is neither read nor rewritten, so a body held only in the object store needs
    no object store here."""
    from mantle.db.doc_boundary import seal_content_fields, unseal_content_fields

    rows = [d for d in db.artifacts.list_artifacts() if isinstance(d, dict)]
    todo = [d for d in rows if _plain_title(d)]
    counts = {"plain": len(todo), "sealed": 0, "left_plain": 0, "ownerless": 0}
    if dry_run:
        return counts
    counts["platform_grants"] = _grant_the_platform_its_collections(db, rows)
    roots: dict = {}
    for doc in todo:
        if _is_grant(doc):
            if _seal_grant(db, doc):
                counts["sealed"] += 1
            else:
                counts["left_plain"] += 1
                logger.info("  cannot seal grant %s — left in plain text", str(doc.get("id"))[:8])
            continue
        holders = _holders(db, doc, roots)
        if not holders:
            counts["ownerless"] += 1
        why = ""
        for acting in _acting_as_each(holders):
            raw = dict(doc)
            try:
                with acting():
                    seal_content_fields(raw)
                    # Written only if the same actor opens it again: a seal nobody this pass can
                    # act as can read back would leave the row unreadable. 2026-10-30, on the
                    # release candidate: 43 of the platform's rows were sealed and then could not be
                    # opened by any actor, person, service or platform.
                    check = dict(raw)
                    unseal_content_fields(check)
            except Exception as exc:  # noqa: BLE001 — per row; the next way, then the next row
                why = type(exc).__name__
                continue
            if _still_sealed(check):
                why = "not readable back"
                continue
            db.artifacts.put_artifact(raw)
            counts["sealed"] += 1
            break
        else:
            # Left as it is stored today, in plain text, and reported: the key service opens this
            # row for no actor this pass has. Not a failure of the release; an open item.
            counts["left_plain"] += 1
            logger.info("  cannot seal %s (%s) — left in plain text", str(doc.get("id"))[:8], why)
    return counts


def unseal_stored_titles(db, *, dry_run: bool = False) -> dict:
    """The reverse, run before a rollback to a Mantle that does not read sealed titles: each sealed
    title put back in plain text, read as the row's creator and written to the row as it is, past
    the write boundary that would seal it again. Returns the counts."""
    from mantle.db import node_seal
    from mantle.db.doc_boundary import SEALED, unseal_content_fields

    def sealed(doc: dict) -> bool:
        if doc.get(node_seal.SEALED):
            return True
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
        if doc.get(node_seal.SEALED):
            raw = dict(doc)
            node_seal.unseal_fields(raw)
            if raw.get(node_seal.SEALED):         # sealed under another node's key; left sealed
                counts["denied"] += 1
                logger.info("  cannot unseal grant %s — leaving it sealed", str(doc.get("id"))[:8])
                continue
            db.artifacts.put_artifact(raw)
            counts["unsealed"] += 1
            continue
        holders = _holders(db, doc, roots)
        if not holders:
            counts["ownerless"] += 1
        for acting in _acting_as_each(holders):
            raw = dict(doc)
            try:
                with acting():
                    unseal_content_fields(raw)
            except Exception:  # noqa: BLE001 — an actor that cannot be assumed; the next way
                continue
            if not _still_sealed(raw):
                db.artifacts.put_artifact(raw)
                counts["unsealed"] += 1
                break
        else:
            counts["denied"] += 1      # the oracle would open it for neither; left sealed, not emptied
            logger.info("  cannot unseal %s — leaving it sealed", str(doc.get("id"))[:8])
    return counts


def _named_edges(db, *, tokens: bool) -> list:
    """Every edge carrying a name a person gave it: in plain text, or (`tokens`) as a keyed token."""
    import json
    from mantle.db import lattice_api as la
    from mantle.db import node_seal
    system = tuple(la._SYSTEM_LABELS)
    marks = ",".join("?" * len(system))
    rows = db.graph.db.read().execute(
        f"SELECT src, dst, label, force, propagate, is_origin, order_key, props FROM edge "
        f"WHERE label NOT IN ({marks})", system).fetchall()
    out = []
    for r in rows:
        if node_seal.is_token(r["label"]) != tokens:
            continue
        try:
            props = json.loads(r["props"] or "{}")
        except (TypeError, ValueError):
            props = {}
        for k in ("force", "propagate", "is_origin", "order_key"):
            if r[k] is not None:
                props[k] = r[k]
        out.append((r["src"], r["dst"], r["label"], props))
    return out


def _relabel(db, src: str, dst: str, old: str, new: str, props: dict) -> None:
    db.graph.add_edge(src, dst, new, props)
    db.graph.delete_edge(src, dst, old)


def seal_stored_labels(db, *, dry_run: bool = False) -> dict:
    """Each link name stored in plain text, rewritten as `add_artifact_to_collection` now writes it:
    a keyed token, the name sealed beside it, and a token per dotted prefix."""
    from mantle.db import lattice_api as la
    todo = _named_edges(db, tokens=False)
    counts = {"plain_labels": len(todo), "labels_sealed": 0}
    if dry_run:
        return counts
    for src, dst, label, props in todo:
        props = {k: v for k, v in props.items() if k not in ("label_sealed", "label_prefixes")}
        _relabel(db, src, dst, label, la._stored_label(label), {**props, **la._label_props(label)})
        counts["labels_sealed"] += 1
    if counts["labels_sealed"]:
        # A label's extent counter is named by the label (`edge:label:<label>`), so the emptied
        # counter of a plain name still spells it.
        with db.graph.db.write() as cur:
            cur.execute("DELETE FROM counter WHERE name LIKE 'edge:label:%' AND n <= 0")
    return counts


def unseal_stored_labels(db, *, dry_run: bool = False) -> dict:
    """The reverse, before a rollback: an older Mantle shows `edge.label` as the link's name, so each
    token goes back to the name it stands for. A name that will not open here is left as it is."""
    from mantle.db import node_seal
    todo = _named_edges(db, tokens=True)
    counts = {"token_labels": len(todo), "labels_unsealed": 0, "labels_denied": 0}
    if dry_run:
        return counts
    for src, dst, label, props in todo:
        sealed = props.get("label_sealed")
        try:
            name = node_seal.open_value(sealed) if sealed else ""
        except Exception:  # noqa: BLE001 — sealed under another node's key
            name = ""
        if not name:
            counts["labels_denied"] += 1
            continue
        props = {k: v for k, v in props.items() if k not in ("label_sealed", "label_prefixes")}
        _relabel(db, src, dst, label, name, props)
        counts["labels_unsealed"] += 1
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
        counts.update(unseal_stored_labels(db, dry_run=args.dry_run))
        logger.info("%s: %s", "would unseal" if args.dry_run else "unsealed", counts)
    else:
        counts = seal_stored_titles(db, dry_run=args.dry_run)
        counts.update(seal_stored_labels(db, dry_run=args.dry_run))
        logger.info("%s: %s", "would seal" if args.dry_run else "sealed", counts)
    return 1 if counts.get("denied") else 0


if __name__ == "__main__":
    sys.exit(main())
