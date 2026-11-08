"""Give each existing collection its own random key, and re-encrypt its content under it.

WHY. Every collection's content is keyed under its own collection key (John, 2026-10-31), so a
grantee on one collection can open that collection and no sibling, and the master key is released
only to its principal. Content written before collections had keys is under the principal's master
key (`MEC1`). This pass, run once as part of switching custody on, re-encrypts it under the key of
the collection it is scoped to (`MEC3`), which it mints on first use.

What it re-encrypts, per artifact:
  * the inline body (`content`, when `content_encrypted`);
  * the sealed fields (`context.sealed`);
  * a body held in the content tier (`content_ref`), re-stored under the collection key with the
    row pointed at the new copy, and the old copy removed once no row names it.
Prior versions are rows of their own (archived snapshots sharing the head's `root_id`), and are
moved the same way. The rewrite is in place: the same content under a new key is not a new
version, so the pass never snapshots. An artifact with no collection scope stays under the master
key; the report names every row it leaves unscoped or under the master key (`left`).

Then the search index (`rekey_index`): each search cell is re-encrypted from its master-derived
key to its collection's, in place, and each artifact's postings are filed again under its
collection's SSE key and removed from under the principal-wide key they had.

Safety, in order:
  * Readable throughout. The collection key is stored (and under threshold custody split, Origin's
    share stored) before anything is written under it. Each new blob is decrypted and compared with
    the plaintext before the row is written. The row is replaced in one write, so a reader sees the
    old form or the new one, never neither. An old content-tier copy is removed only after the
    row naming the new one is written, and only when no other row still names it.
  * Content written during the pass stays readable. A row is rewritten only if it is exactly the
    row this pass read (compared inside the write); a row that changed is left for the next run.
    New writes are collection-keyed already, and the pass leaves those alone.
  * Resumable and idempotent. What is already collection-keyed is skipped, so an interrupted run
    is finished by running it again, and a finished run does nothing.

Each artifact is opened as its own principal (`acting_as(principal, source="system")`), the way
the per-owner passes run: under threshold custody that shows Origin a delegation minted for each
key, which needs the operator's background authority.

    python -m mantle.system.rekey_collections --dry-run
    python -m mantle.system.rekey_collections --mantle-stopped [--limit N]

Progress is logged every `--every` artifacts and the counts are printed at the end. Charged roots'
keys are rotated (`step21`) only with Mantle's server stopped (`--mantle-stopped`), with this
store's ledger feed at Origin's head, and with Origin's answer to who is who; the run exits
non-zero without them, or when any charged root is left.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

logger = logging.getLogger("mantle.system.rekey_collections")

_SKIP_TYPES = ("application/vnd.agience.grant+json", "application/vnd.agience.master-key+json",
               "application/vnd.agience.master-key-share+json",
               "application/vnd.agience.drop-key+json", "application/vnd.agience.add-seal+json")


@dataclass
class Cas:
    """How the pass reads and re-stores a content-tier body. Defaults to the content tier."""

    read: Callable[[dict, str, str], bytes] = None
    raw: Callable[[dict], Optional[bytes]] = None
    write: Callable[[dict, str, str, bytes], Optional[str]] = None
    #: Remove an old copy no row names any more.
    drop: Callable[[str], None] = None
    #: Remove an object-store copy (by its content key) no row names any more.
    drop_object: Callable[[str], None] = None

    @classmethod
    def default(cls) -> "Cas":
        from mantle.services import content_service as cs

        def _ctx(doc):
            ctx = doc.get("context")
            if isinstance(ctx, str):
                try:
                    ctx = json.loads(ctx)
                except ValueError:
                    ctx = {}
            return ctx if isinstance(ctx, dict) else {}

        def read(doc, principal, scope):
            return cs.get_bytes_decrypted(_ctx(doc).get("content_key") or "", principal,
                                          cas_ref=doc.get("content_ref"), collection_id=scope)

        def raw(doc):
            try:
                tier = cs.local_content_tier()
                return tier.get_raw(doc.get("content_ref")) if tier else None
            except Exception:  # noqa: BLE001 — no raw view: decided by the marker instead
                return None

        def write(doc, principal, scope, data):
            ctx = _ctx(doc)
            return cs.put_bytes_encrypted(ctx.get("content_key") or doc.get("id"), data,
                                          ctx.get("content_type") or "text/plain", principal,
                                          collection_id=scope, cas=True)

        def drop(ref):
            import os

            tier = cs.local_content_tier()
            cache = getattr(tier, "cache", None) if tier else None
            if cache is not None:
                try:
                    os.unlink(cache.path_of(ref))
                except OSError:
                    pass

        def drop_object(key):
            try:
                cs.delete_object(key)
            except Exception:  # noqa: BLE001 — logged by the store; the copy stays
                logger.warning("rekey: object-store copy %s not removed", key, exc_info=True)

        return cls(read=read, raw=raw, write=write, drop=drop, drop_object=drop_object)


@dataclass
class Report:
    seen: int = 0
    rekeyed: int = 0
    already: int = 0
    unscoped: int = 0
    changed: int = 0
    failed: int = 0
    old_copies_removed: int = 0
    collections: set = field(default_factory=set)
    #: Every row the pass leaves under the master key, and why: `unscoped`, `changed`, `failed`.
    left: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"seen": self.seen, "rekeyed": self.rekeyed, "already": self.already,
                "unscoped": self.unscoped, "changed": self.changed, "failed": self.failed,
                "old_copies_removed": self.old_copies_removed,
                "collections": len(self.collections),
                "left": dict(sorted(self.left.items()))}


#: The marker a content-tier body re-stored under its collection key leaves on the row.
CAS_MARK = "content_collection_keyed"


def _b64(s) -> bytes:
    return base64.b64decode(s or "")


def _plan(doc: dict, cas: Cas, *, drops: bool = True) -> dict:
    """What in this row is still to re-key onto its collection's key. `drops=False` asks only
    what is under a master key (review 31): a drop (MED1) is sealed to its collection's drop key,
    under no master key, and is re-keyed by its first reader, not by a master key's pass."""
    from mantle.db.doc_boundary import SEALED, _context_dict
    from mantle.services import content_crypto as cc

    def pending(blob: bytes) -> bool:
        return (cc.is_encrypted(blob) and not cc.is_collection_keyed(blob)
                and (drops or not cc.is_drop(blob)))

    todo = {}
    if doc.get("content_encrypted") and doc.get("content"):
        blob = _b64(doc["content"])
        if pending(blob):
            todo["content"] = blob
    ctx, _as_string = _context_dict(dict(doc))
    sealed = (ctx or {}).get(SEALED) if ctx else None
    if sealed and sealed.get("blob"):
        blob = _b64(sealed["blob"])
        if pending(blob):
            todo["sealed"] = blob
    if not drops and doc.get("content_dropped"):
        return todo                     # a dropped content-tier body: under no master key
    if doc.get("content_ref") and (not doc.get(CAS_MARK) or doc.get("content_dropped")):
        raw = cas.raw(doc) if cas.raw else None
        if raw is None or not cc.is_collection_keyed(raw):
            todo["cas"] = True
    return todo


def rekey_one(db, doc: dict, cas: Cas, report: Optional["Report"] = None, *,
              as_caller: bool = False, refs: Optional[Dict[str, int]] = None) -> str:
    """`as_caller`: re-key as the acting caller (a reader re-keying a drop it just read), not as
    the content's principal."""
    """Re-encrypt one row under its collection's key. Returns `rekeyed`, `already`, `unscoped`
    or `changed`."""
    from mantle.db import doc_boundary as b
    from mantle.services import content_crypto as cc
    from mantle.services.acting_principal import acting_as

    scope = b.content_key_scope(doc)
    principal = doc.get(b.CONTENT_KEY_PRINCIPAL) or b.content_key_principal(doc) \
        or doc.get("created_by")
    todo = _plan(doc, cas)
    if not todo:
        return "already"
    if not scope or not principal:
        return "unscoped"
    new = {k: v for k, v in doc.items() if not k.startswith("_")}
    new.pop("content_dropped", None)              # re-keyed under the collection key now
    written = written_key = old_key = None
    import contextlib as _contextlib
    as_principal = (_contextlib.nullcontext() if as_caller else
                    acting_as(str(principal), principal_type="user", source="system",
                              scope="platform.rekey"))
    with as_principal:
        with cc.content_key_artifact(doc.get("root_id") or doc.get("id")):
            # The collection key exists (stored, and split under threshold custody) before any
            # content is written under it: minting is the first thing the encrypt does.
            def reencrypt(blob: bytes) -> bytes:
                plain = cc.decrypt_content(principal, blob, collection_id=scope,
                                           require_encrypted=True)
                fresh = cc.encrypt_content(principal, plain, collection_id=scope,
                                           creator_id=doc.get("created_by"))
                if cc.decrypt_content(principal, fresh, collection_id=scope) != plain:
                    raise RuntimeError("a re-encrypted blob did not read back; row left as it was")
                return fresh

            if "content" in todo:
                new["content"] = base64.b64encode(reencrypt(todo["content"])).decode("ascii")
            if "sealed" in todo:
                ctx, as_string = b._context_dict(new)
                ctx[b.SEALED] = {**ctx[b.SEALED],
                                 "blob": base64.b64encode(reencrypt(todo["sealed"])).decode()}
                b._put_context(new, ctx, as_string)
            if "cas" in todo:
                data = cas.read(doc, principal, scope)
                # Each rewrite has its own object-store key, so a prior version can never write
                # over the head's copy, and a copy for a row someone rewrote meanwhile is removed
                # without touching theirs.
                ctx, as_string = b._context_dict(new)
                ctx = dict(ctx or {})
                old_key = ctx.get("content_key")
                ctx["content_key"] = written_key = (
                    f"{old_key or doc['id']}.rk-{uuid.uuid4().hex[:12]}")
                b._put_context(new, ctx, as_string)
                ref = cas.write(new, principal, scope, data)
                if not ref:          # not landed: nothing to point at, so the row is not done
                    raise RuntimeError("the content tier took no new copy; row left as it was")
                new["content_ref"] = written = ref
                new[CAS_MARK] = True

    # One write, and only over the row this pass read: a row someone wrote since is left for
    # the next run, so their write stands.
    # In place: the same content under a new key is not a new version, so nothing is snapshotted.
    conn = db.artifacts.db
    old_ref = doc.get("content_ref")
    with conn.write() as cur:
        r = cur.execute("SELECT _seq, _origin FROM vertex WHERE id = ?", (doc["id"],)).fetchone()
        if r is None or (r["_seq"], r["_origin"]) != (doc.get("_seq"), doc.get("_origin")):
            changed = True
        else:
            changed = False
            db.artifacts.put_artifact(new, snapshot=False)
    if changed:
        if written and written != old_ref and cas.drop:
            cas.drop(written)        # the copy made for a row someone rewrote: no orphan
        if written_key and cas.drop_object:
            cas.drop_object(written_key)
        return "changed"
    still_named = old_ref and new.get("content_ref") != old_ref and conn.read().execute(
        "SELECT 1 FROM vertex WHERE content_ref = ? LIMIT 1", (old_ref,)).fetchone()
    if old_ref and new.get("content_ref") != old_ref and not still_named and cas.drop:
        cas.drop(old_ref)
        if report is not None:
            report.old_copies_removed += 1
    if written_key and old_key and cas.drop_object:
        # Which rows name the old key, decided by parsing each row's context, never by text.
        if refs is None:
            refs = key_refs(db)
        else:
            refs[old_key] = max(0, refs.get(old_key, 1) - 1)
            refs[written_key] = refs.get(written_key, 0) + 1
        # The count was taken when the pass started; a version written since may name the key
        # too. A content key belongs to one artifact, so its own versions are checked, parsed.
        if not refs.get(old_key) and not _versions_name(db, doc.get("root_id") or doc["id"],
                                                        old_key):
            cas.drop_object(old_key)     # the old object-store copy, once no version names it
    return "rekeyed"


def _content_key_of(doc: dict) -> Optional[str]:
    from mantle.db import doc_boundary as b

    ctx, _as_string = b._context_dict(dict(doc))
    key = (ctx or {}).get("content_key") if isinstance(ctx, dict) else None
    return str(key) if key else None


def _versions_name(db, root: str, key: str) -> bool:
    """Does any row of this artifact (the head or a prior version) name `key` now, by its parsed
    context?"""
    rows = db.artifacts.db.read().execute(
        "SELECT id FROM vertex WHERE root_id = ? OR id = ?", (root, root)).fetchall()
    for r in rows:
        doc = db.artifacts.get_artifact(r[0])
        if doc and _content_key_of(doc) == key:
            return True
    return False


def key_refs(db) -> Dict[str, int]:
    """How many rows (every version) name each object-store content key, from their parsed
    contexts."""
    refs: Dict[str, int] = {}
    for doc in db.artifacts.list_artifacts(include_archived=True):
        key = _content_key_of(doc)
        if key:
            refs[key] = refs.get(key, 0) + 1
    return refs


def _rows(db) -> Iterable[dict]:
    for doc in db.artifacts.list_artifacts(include_archived=True):
        if doc.get("content_type") in _SKIP_TYPES or str(doc.get("id", "")).startswith(
                "master-key:"):
            continue
        yield doc


def rekey(db, *, cas: Optional[Cas] = None, dry_run: bool = False, limit: Optional[int] = None,
          every: int = 500, progress: Optional[Callable[[Report], None]] = None) -> Report:
    """Run the pass over every artifact. A row that fails is counted, logged and left as it was;
    the pass goes on and a rerun picks it up."""
    from mantle.db import doc_boundary as b

    cas = cas or Cas.default()
    report = Report()
    refs = None if dry_run else key_refs(db)
    for doc in _rows(db):
        if limit is not None and report.seen >= limit:
            break
        report.seen += 1
        try:
            if dry_run:
                todo = _plan(doc, cas)
                outcome = "already" if not todo else (
                    "unscoped" if not b.content_key_scope(doc) else "rekeyed")
            else:
                outcome = rekey_one(db, doc, cas, report, refs=refs)
        except Exception as exc:  # noqa: BLE001 — one row's failure does not stop the pass
            report.failed += 1
            report.left[str(doc.get("id"))] = "failed"
            logger.warning("rekey: %s left as it was: %s", doc.get("id"), exc)
            continue
        setattr(report, outcome, getattr(report, outcome) + 1)
        if outcome in ("unscoped", "changed"):
            report.left[str(doc.get("id"))] = outcome
        if outcome == "rekeyed":
            report.collections.add(b.content_key_scope(doc))
        if every and report.seen % every == 0:
            logger.info("rekey progress: %s", report.as_dict())
            if progress:
                progress(report)
    if not dry_run and report.rekeyed:
        # What the pass rewrote leaves no bytes under the master key in the store file.
        from mantle.db.scrub import scrub_freed
        try:
            scrub_freed(db)
        except AttributeError:             # a store with no SQLite file behind it
            pass
    logger.info("rekey %s: %s", "planned" if dry_run else "done", report.as_dict())
    if progress:
        progress(report)
    return report


# -- the search index ---------------------------------------------------------------------------

@dataclass
class IndexReport:
    cells_seen: int = 0
    cells_moved: int = 0
    cells_already: int = 0
    postings_seen: int = 0
    postings_moved: int = 0
    postings_already: int = 0
    failed: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


def _key_principals(db) -> list:
    from mantle.search.mantle.oracle import SHARE_CONTENT_TYPE, LatticeMasterKeyStore

    out = set()
    for ct in (LatticeMasterKeyStore.CONTENT_TYPE, SHARE_CONTENT_TYPE):
        for doc in db.artifacts.list_artifacts(content_type=ct):
            rid = str(doc.get("id") or "")
            if rid.startswith("master-key:") and not rid.startswith("master-key:ck:"):
                out.add(rid[len("master-key:"):])
    return sorted(out)


def _own_request(principal: str):
    from mantle.search.mantle.oracle import KeyPurpose, KeyRequest
    return KeyRequest(requester_id=principal, purpose=KeyPurpose.SELF, action="create")


def rekey_cells(db, oracle, cells, report: IndexReport) -> None:
    """Re-encrypt each search cell from its master-derived key to its collection's key, in place.

    One put per cell, after the new blob is read back: a cell is under the old key or the new
    one, never neither. A cell that already opens under the new key is left alone."""
    from mantle.search.mantle import cell as cell_mod
    from mantle.services.acting_principal import acting_as

    for principal in _key_principals(db):
        with acting_as(principal, principal_type="user", source="system",
                           scope="platform.rekey"):
            req = _own_request(principal)
            for coll in cells.list_cells(principal):
                for cluster in cells.list_clusters(principal, coll):
                    report.cells_seen += 1
                    try:
                        blob = cells.get(principal, coll, cluster)
                        if blob is None:
                            continue
                        aad = cell_mod.cell_aad(coll, cluster)
                        new_key = oracle.derive_cell_key(principal, coll, cluster, req)
                        try:
                            cell_mod.unpack_cell(blob, new_key, collection_id=aad)
                            report.cells_already += 1
                            continue
                        except cell_mod.CellTampered:
                            pass
                        old_key = oracle.derive_legacy_cell_key(principal, coll, cluster, req)
                        chunks = cell_mod.unpack_cell(blob, old_key, collection_id=aad)
                        fresh = cell_mod.pack_cell(chunks, new_key, collection_id=aad)
                        if cell_mod.unpack_cell(fresh, new_key, collection_id=aad) != chunks:
                            raise RuntimeError("a re-encrypted cell did not read back")
                        cells.put(principal, coll, fresh, cluster)
                        report.cells_moved += 1
                    except Exception as exc:  # noqa: BLE001 — counted; the cell is as it was
                        report.failed += 1
                        logger.warning("rekey: cell %s/%s/%s left as it was: %s",
                                       principal, coll, cluster, exc)


def rekey_postings(db, oracle, indexer, report: IndexReport) -> None:
    """File each artifact's postings under its collection's SSE key, then remove the copy under
    the principal-wide key it had. Blind tokens are keyed, so postings cannot be re-encrypted in
    place; they are written again from the artifact, as an index write does.

    New first, old second: interrupted between the two, a rerun finds the old manifest, writes the
    new (the same tokens again) and removes the old. An artifact with no old manifest is done."""
    from mantle.db.lattice_api import ArtifactEntity, from_lattice_doc
    from mantle.search.ingest.pipeline_unified import _extract_artifact_fields
    from mantle.search.mantle.principal import resolve_cell_principal
    from mantle.services.acting_principal import acting_as

    for doc in _rows(db):
        coll = doc.get("collection_id")
        root = doc.get("root_id") or doc.get("id")
        if not coll or not root:
            continue
        if str(doc.get("id")) != str(root):
            continue        # a prior version: postings are the head's, filed by its root
        report.postings_seen += 1
        try:
            principal = resolve_cell_principal(db, coll)
            if not principal or indexer._postings.get_manifest(principal, root) is None:
                report.postings_already += 1
                continue
            with acting_as(principal, principal_type="user", source="system",
                           scope="platform.rekey"):
                req = _own_request(principal)
                fields = _extract_artifact_fields(from_lattice_doc(dict(doc), ArtifactEntity))
                if fields:
                    indexer.index_artifact(principal, coll, root, fields, req)
                indexer.remove_artifact(principal, root, req,
                                        sse_key=oracle.derive_legacy_sse_key(principal, req))
            report.postings_moved += 1
        except Exception as exc:  # noqa: BLE001 — counted; a rerun picks it up
            report.failed += 1
            logger.warning("rekey: postings of %s left as they were: %s", root, exc)


def rekey_index(db, *, oracle, cells=None, indexer=None) -> IndexReport:
    """Move the search index onto collection keys: cells, then postings."""
    report = IndexReport()
    if cells is not None:
        rekey_cells(db, oracle, cells, report)
    if indexer is not None:
        rekey_postings(db, oracle, indexer, report)
    logger.info("rekey index done: %s", report.as_dict())
    return report


# -- a charged root's key is rotated (reviews 30 and 31) ------------------------------------------

#: The field of a master-key row that says step 21 rotated it (review 31): the rotated key's
#: key-check value, and when. Written with the new key, in the rotation's own transaction; the
#: split refuses a charged root whose key does not carry it, and the operator records the same
#: kcv at Origin before granting the split.
ROTATION_MARK = "rotated"


class _Changed(Exception):
    """A row the rotation read changed before its transaction: nothing of the root is written."""


def rotated_key_row(kek, root: str, key: bytes, *, kcv: Optional[str] = None) -> dict:
    """The master-key row for `root`'s rotated `key`, carrying its rotation mark."""
    from datetime import datetime, timezone

    from mantle.search.mantle.oracle import LatticeMasterKeyStore, _frame_master_key
    from mantle.search.mantle.threshold_custody import key_check_value

    return {"id": LatticeMasterKeyStore._id(root),
            "content_type": LatticeMasterKeyStore.CONTENT_TYPE,
            "token": kek.wrap(_frame_master_key(root, key)),
            ROTATION_MARK: {"kcv": kcv or key_check_value(root, key),
                            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}}


def rotation_mark(row: Optional[dict], root: str, key: bytes) -> bool:
    """Does this key row carry step 21's rotation mark for this very key?"""
    import hmac

    from mantle.search.mantle.threshold_custody import key_check_value

    mark = (row or {}).get(ROTATION_MARK)
    kcv = mark.get("kcv") if isinstance(mark, dict) else None
    return isinstance(kcv, str) and hmac.compare_digest(kcv, key_check_value(root, key))


def off_root_rows(db, root: str) -> list:
    """The rows under `root`'s master key whose root, by this store's own creation edges, is no
    longer `root`: a removed part's (review 31, I2). Mantle's own keep/shred test, run over every
    root at step 21, so a removal from before the ledger is caught though Origin charges none."""
    from mantle.db.lattice_api import get_origin_root
    from mantle.services.owner_rekey import rows_under_master_key

    return [d for d, _who in rows_under_master_key(db, root)
            if get_origin_root(db, str(d.get("root_id") or d.get("id"))) != root]


def _opens_under(doc: dict, root: str, key: bytes) -> bool:
    """Does any blob of this row open under `key`?"""
    from mantle.db import doc_boundary as b
    from mantle.services import content_crypto as cc

    blobs = []
    if doc.get("content_encrypted") and doc.get("content"):
        blobs.append(_b64(doc["content"]))
    ctx, _s = b._context_dict(dict(doc))
    sealed = (ctx or {}).get(b.SEALED) if ctx else None
    if sealed and sealed.get("blob"):
        blobs.append(_b64(sealed["blob"]))
    for blob in blobs:
        try:
            cc.decrypt_content(root, blob, collection_id=b.content_key_scope(doc),
                               master_key_provider=lambda _p: key, require_encrypted=True)
            return True
        except Exception:  # noqa: BLE001 — not this key's
            continue
    return False


def rotate_charged_roots(db, kek, roots, *, again=()) -> Dict[str, dict]:
    """Rotate each charged root's master key, under lattice custody, at step 21 (John,
    2026-11-08), with Mantle's server stopped. A root is charged at Origin once a part of it was
    removed, for good: its master key opens for its own principal alone. What is still under its
    old key:

      * a row whose root, by this store's own creation edges (the edges Origin co-signed), is
        still the root is re-encrypted under a fresh key for the root;
      * a row whose root is no longer the root belonged to the removed part: it stays under the
        old key, which is then deleted, so the row is shredded;
      * the new key, carrying its rotation mark (`ROTATION_MARK`), and every rewritten row are
        written in one transaction, each blob read back first, and only if every row and the key
        row are still exactly as read (`_seq`, `_origin`, and the same rows under the old key),
        checked inside that transaction; otherwise the root is left as it was (`changed`) for a
        re-run. The old key's bytes are then settled out of the store file;
      * a root with nothing of its own left has its whole key removed instead.

    Each root stands alone: one that fails is reported (`left`: `failed: ...`) and the run goes on.
    The old key is opened only here, where the node holds every key anyway; under threshold
    custody it no longer exists, so no one is ever handed it. A key already split is left (a
    re-run after step 24 rotates nothing). A content-tier copy under the old key is not rotated:
    that root is left as it was and reported. A key already rotated, with nothing of a removed
    part under it, is left with its mark."""
    out: Dict[str, dict] = {}
    again = {str(r) for r in again}
    for root in sorted({str(r) for r in roots}):
        report = {"rotated": 0, "shredded": 0, "removed": False}
        out[root] = report
        try:
            _rotate_one(db, kek, root, report, again=root in again)
        except _Changed as exc:
            report["left"] = f"changed: {exc}; left as it was for a re-run"
            logger.warning("rotation of charged root %s left: %s", root, exc)
        except Exception as exc:  # noqa: BLE001 — one root's failure leaves the others alone
            report["left"] = f"failed: {type(exc).__name__}: {exc}"
            logger.error("rotation of charged root %s failed; left as it was: %s", root, exc)
    return out


def _rotate_one(db, kek, root: str, report: dict, *, again: bool = False) -> None:
    import os as _os

    from mantle.db import doc_boundary as b
    from mantle.db.scrub import settle_replaced
    from mantle.search.mantle.oracle import LatticeMasterKeyStore, SHARE_CONTENT_TYPE
    from mantle.search.mantle.threshold_custody import key_check_value, stored_form
    from mantle.services import content_crypto as cc
    from mantle.services.owner_rekey import rows_under_master_key

    store = LatticeMasterKeyStore(kek, lambda: db)
    key_id = LatticeMasterKeyStore._id(root)
    row = db.artifacts.get_artifact(key_id)
    if row is None:
        return
    if row.get("content_type") == SHARE_CONTENT_TYPE:
        report["left"] = "split: rotated only under lattice custody, before step 23"
        return
    old = store.get(root)
    if old is None:
        return
    rows = [d for d, _who in rows_under_master_key(db, root)]
    gone = {str(d["id"]) for d in off_root_rows(db, root)}
    keep = [d for d in rows if str(d["id"]) not in gone]
    if any(d.get("content_ref") for d in keep):
        report["left"] = "a content-tier copy under the old key: not rotated"
        return
    report["shredded"] = len(rows) - len(keep)
    if not keep:
        db.artifacts.delete_artifact(key_id)
        settle_replaced(db)
        report["removed"] = True
        return
    # `again` (review 33): the operator's `--rotate-again`, past a split that stored Origin's
    # share of this key and failed its read-back; the mark is ignored and a fresh key minted.
    if not again and rotation_mark(row, root, old) and not any(
            _opens_under(d, root, old) for d in rows if str(d["id"]) in gone):
        # Rotated before (review 32): the removed part's rows are still listed under the root's
        # principal, but none opens under this key, which is the one the mark names. The kcv is
        # the one the operator recorded from the run that rotated it.
        report["kcv"] = key_check_value(root, old)
        report["already"] = True
        return
    new = _os.urandom(32)

    def move(blob: bytes, scope) -> bytes:
        plain = cc.decrypt_content(root, blob, collection_id=scope,
                                   master_key_provider=lambda _p: old)
        fresh = cc.encrypt_content(root, plain, collection_id=scope,
                                   master_key_provider=lambda _p: new)
        if cc.decrypt_content(root, fresh, collection_id=scope,
                              master_key_provider=lambda _p: new) != plain:
            raise RuntimeError("a rotated blob did not read back; nothing written")
        return fresh

    rewritten = []
    for doc in keep:
        scope = b.content_key_scope(doc)
        new_doc = {k: v for k, v in doc.items() if not k.startswith("_")}
        if doc.get("content_encrypted") and doc.get("content"):
            new_doc["content"] = base64.b64encode(
                move(_b64(doc["content"]), scope)).decode("ascii")
        ctx, as_string = b._context_dict(dict(new_doc))
        sealed = (ctx or {}).get(b.SEALED) if ctx else None
        if sealed and sealed.get("blob"):
            ctx = dict(ctx)
            ctx[b.SEALED] = {**sealed, "blob": base64.b64encode(
                move(_b64(sealed["blob"]), scope)).decode("ascii")}
            b._put_context(new_doc, ctx, as_string)
        rewritten.append(new_doc)
    key_row = rotated_key_row(kek, root, new)
    read_ids = {str(d["id"]) for d in rows}
    with db.artifacts.db.write() as cur:              # the new key and its rows land together
        # Inside the write, as `rekey_one` checks: a row someone wrote since this pass read it,
        # or a row that came under the old key meanwhile, leaves the root as it was.
        for d in rows:
            r = cur.execute("SELECT _seq, _origin FROM vertex WHERE id = ?",
                            (d["id"],)).fetchone()
            if r is None or (r["_seq"], r["_origin"]) != (d.get("_seq"), d.get("_origin")):
                raise _Changed(f"row {d['id']} changed since it was read")
        if {str(d["id"]) for d, _w in rows_under_master_key(db, root)} != read_ids:
            raise _Changed("the rows under the old key changed since they were read")
        if stored_form(db.artifacts.get_artifact(key_id)) != stored_form(row):
            raise _Changed("the key row changed since it was read")
        db.artifacts.put_artifact(key_row)
        for new_doc in rewritten:
            db.artifacts.put_artifact(new_doc, snapshot=False)
    settle_replaced(db)
    report["rotated"] = len(rewritten)
    report["kcv"] = key_row[ROTATION_MARK]["kcv"]
    logger.warning("rotated the master key of charged root %s: %d row(s) moved, %d shredded",
                   root, len(rewritten), report["shredded"])


def step21(db, kek, custodian, *, mantle_stopped: bool, retire_member_keys: bool = False,
           rotate_again=()) -> tuple:
    """Step 21's key work after the content and the index are moved: which keys are whose, by
    Origin (`classify`); each charged root's key rotated; each emptied whole non-person key
    removed. Returns `(report, failed)`; `failed` makes the run exit non-zero (review 31):

      * `no_kinds`: Origin did not say who is who while key rows exist, so nothing is rotated or
        removed;
      * `no_charged`: Origin did not say which roots are charged;
      * `feed_behind`: this store's ledger feed is not at Origin's head, so its creation edges
        may not be the ones Origin co-signed; nothing is rotated;
      * `mantle_running`: the run was not stated to have Mantle's server stopped
        (`--mantle-stopped`); a server holding the old key in its cache could write under it
        after the rotation commits, and that write would be lost; nothing is rotated;
      * any rotated root reporting `left`.

    The roots rotated are those Origin names charged and, by Mantle's own keep/shred test
    (`off_root_rows`), every root with a row of a removed part under its key: a removal from
    before the ledger (before step 14) is caught though Origin charges none."""
    from mantle.services.owner_rekey import rows_under_master_key
    from mantle.system.split_master_keys import (
        KINDS_PAGE, _key_rows, _principal, classify, retire_empty_master_keys)

    out: dict = {"refused": []}
    keyrows = _key_rows(db)
    kinds = classify({_principal(d) for d in keyrows}, custodian) if custodian else {}
    if keyrows and not kinds:
        out["refused"].append("no_kinds")
        logger.error("Origin did not say which key principals are whose; nothing is rotated or "
                     "removed")
        return out, True
    roots = sorted(p for p, k in kinds.items() if k == "root")
    charged: list = []
    try:
        for i in range(0, len(roots), KINDS_PAGE):
            charged += custodian.charged_roots(roots[i:i + KINDS_PAGE])
    except Exception as exc:  # noqa: BLE001 — unanswered: nothing rotated, failing closed
        out["refused"].append("no_charged")
        logger.error("Origin did not say which roots are charged (%s); nothing is rotated", exc)
        return out, True
    own = [r for r in roots if r not in set(charged) and off_root_rows(db, r)]
    if own:
        logger.warning("%d root(s) Origin charges none for hold a removed part's rows under their "
                       "key (a removal from before the ledger): rotated too", len(own))
    to_rotate = sorted(set(charged) | set(own))
    out["charged_by_origin"], out["charged_by_this_store"] = sorted(set(charged)), own
    failed = False
    if rotate_again and not to_rotate:
        out["refused"].append("rotate_again_unknown")
        failed = True
    if to_rotate:
        from mantle.services.authority_ledger import feed_at_head
        at_head, cursor, head = feed_at_head(db, custodian)
        if not at_head:
            out["refused"].append("feed_behind")
            logger.error("the ledger feed is at %s and Origin's head at %s: run the feed to the "
                         "head first; nothing is rotated", cursor, head)
        unknown = sorted(set(str(r) for r in rotate_again) - set(to_rotate))
        if unknown:
            out["refused"].append("rotate_again_unknown")
            logger.error("--rotate-again names root(s) this run does not rotate: %s; nothing is "
                         "rotated", ", ".join(unknown))
        # The server's own lock, held for the whole rotation (review 33): no server can start on
        # this store meanwhile, and a running one holds it, so the rotation refuses.
        from mantle.services import server_lock
        lock_path = server_lock.default_path()
        mine = lock_path in server_lock._HELD
        lock = None
        if mantle_stopped:
            try:
                lock = server_lock.hold(lock_path)
            except RuntimeError:
                lock = None
        if lock is None:
            out["refused"].append("mantle_running")
            logger.error("rotation runs only with Mantle's server stopped (its lock free) and "
                         "--mantle-stopped passed; nothing is rotated")
        try:
            if out["refused"]:
                failed = True
            else:
                out["charged_roots"] = rotate_charged_roots(db, kek, to_rotate,
                                                            again=rotate_again)
        finally:
            if lock is not None and not mine:
                server_lock.release(lock)
        if not out["refused"]:
            # A key already split is done with step 21 (review 32): reported, not `left`.
            left = sorted(r for r, rep in out["charged_roots"].items()
                          if "left" in rep and not str(rep["left"]).startswith("split"))
            if left:
                failed = True
                logger.error("%d charged root(s) left: %s", len(left), ", ".join(left))
    out["non_person_keys"] = retire_empty_master_keys(db, kinds=kinds,
                                                      member_keys=retire_member_keys)
    # Per root: what is left under each non-person key after the run (reviews 29 and 30): a
    # root whose leftover reads 0 had its whole key removed in this run.
    per_root: dict = {}
    for _doc, who in rows_under_master_key(db):
        if kinds.get(str(who)) in ("root", "platform", "member", "unknown"):
            per_root[str(who)] = per_root.get(str(who), 0) + 1
    out["left_per_root"] = per_root
    return out, failed


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=500, help="log progress every N artifacts")
    ap.add_argument("--retire-member-keys", action="store_true",
                    help="also remove an emptied whole key Origin answers `member` or `unknown` "
                         "for (the remedy when step 23's gate stays above 0)")
    ap.add_argument("--mantle-stopped", action="store_true",
                    help="Mantle's server is stopped for this run: charged roots' keys are "
                         "rotated only then (review 31)")
    ap.add_argument("--rotate-again", action="append", default=[], metavar="ROOT",
                    help="rotate this charged root onto a fresh key though its key carries a "
                         "valid mark: past a split that stored Origin's share and failed its "
                         "read-back (review 33); repeatable")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    from prism.trust.key_manager import init_encryption_key, init_nonce_secret
    init_encryption_key()
    init_nonce_secret()
    from mantle.services import peer_signing
    peer_signing.init()
    from mantle.services.authority_ledger import install_for_cli
    install_for_cli()
    from mantle.services.dependencies import get_store_db

    db = next(get_store_db())
    report = rekey(db, dry_run=args.dry_run, limit=args.limit, every=args.every)
    out = report.as_dict()
    failed = report.failed or report.changed
    if not args.dry_run:
        from mantle.search.mantle import wiring

        oracle = wiring._build_oracle()
        index = rekey_index(db, oracle=oracle, cells=wiring._build_cell_store(),
                            indexer=wiring.build_sse_indexer(db, segment="live"))
        out["index"] = index.as_dict()
        failed = failed or index.failed
    from mantle import config
    custodian = None
    if (getattr(config, "ORIGIN_URI", "") or "").strip():
        from mantle.clients.origin_key_share_client import OriginShareCustodian
        custodian = OriginShareCustodian()
    from mantle.system.split_master_keys import _key_rows, _principal, classify, exposure_counts
    if not args.dry_run and not failed:
        from mantle.search.mantle.key_provider import build_key_provider
        s21, s21_failed = step21(db, build_key_provider(), custodian,
                                 mantle_stopped=args.mantle_stopped,
                                 retire_member_keys=args.retire_member_keys,
                                 rotate_again=args.rotate_again)
        out.update(s21)
        failed = failed or s21_failed
    else:
        keyrows = _key_rows(db)
        if keyrows and not (classify({_principal(d) for d in keyrows}, custodian)
                            if custodian else {}):
            out["refused"] = ["no_kinds"]
            failed = True
    left = exposure_counts(db, custodian=custodian)
    out["left"] = {k: left[k] for k in ("whole_non_person_master_keys",
                                        "non_person_master_keys_holding_content",
                                        "content_under_master_keys",
                                        "people_with_content_under_master_keys")}
    print(json.dumps(out))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
