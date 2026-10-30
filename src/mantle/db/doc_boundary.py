"""The artifact persistence boundary — content envelope crypto + the change-event chokepoint.

One implementation, shared by every store adapter (`db.store`, `db.lattice_api`), so no adapter
can fork the security behavior: same MEC1 wire format, same origin-root key principal, same
fail-closed rules. Store-agnostic on purpose — everything here operates on plain doc dicts /
entities and knows nothing about where they are persisted.

The load-bearing docstrings below (why the principal is the origin root, why a failed encrypt
fails the write, why a failed decrypt never returns ciphertext) are the contract every adapter
relies on. Read them before touching anything.
"""
from __future__ import annotations

import json
import logging

# Module level, not inside the try blocks that use it: an `except KeyCustodyDenied`
# with a lazily-imported name would raise NameError at the worst possible moment.
try:                                        # plain path first — the canonical import inside the
    from mantle.services.acting_principal import KeyCustodyDenied    # mantle service (matches db.store,
except ImportError:                          # and what the test suite patches)
    from mantle.services.acting_principal import KeyCustodyDenied

logger = logging.getLogger(__name__)


class ContentEncryptionError(RuntimeError):
    """Raised when artifact content cannot be encrypted for storage."""


class ContentDecryptionError(RuntimeError):
    """Raised when stored artifact content cannot be decrypted for a read."""


#: Document field recording which principal the content key was derived from.
#:
#: Written at encrypt time so the read path does not have to re-resolve it (the
#: converters have no db handle). Safe to store and to trust: the collection's
#: origin root is immutable, and the value is also the GCM AAD, so a blob whose
#: recorded root is edited fails to authenticate rather than decrypting under the
#: attacker's choice.
CONTENT_KEY_PRINCIPAL = "content_key_principal"

#: Sealing is the default and structure is the named exception (John, 2026-10-29: "everything related
#: to content is encrypted; a grant is the ability to decrypt"). Every context key an artifact carries
#: is sealed on the way into the store, under the same key as the body (`context.sealed`), and restored
#: on every authorized read, except the keys below, which the store reads before any key is issued or
#: writes itself. The top-level fields a writer supplies (`SEALED_DOC_FIELDS`) are sealed with them.
#: Readers and reasons are measured in `_scratch/CURRENT/MANTLE-SEAL-EVERYTHING.md` ("Reader inventory").
STRUCTURAL_CONTEXT_KEYS = frozenset({
    # The issuer and type loaders select by it before authentication
    # (`lattice_api.list_live_artifacts_by_context_content_type`), and it is the event filter.
    "content_type",
    # Read to hydrate the body from the CAS before the body's decrypt (`_decrypt_body`), and by the
    # mirror drain, which runs with no person's key.
    "content_key", "content_cas_ref",
    # Card-key bindings: an authorization decision reads their grant key ids
    # (`workspace_service.resolve_card_key`).
    "stream", "inbound",
    # Upload bookkeeping the server writes as bytes arrive; it is not what the writer said.
    "upload", "processing", "storage",
})

#: Content types whose whole context is platform configuration, read before any person's key is
#: issued: issuers at token verification (`services/issuers.py`), type definitions at boot
#: (`services/type_offers.py`), anchors and crosswalks at recall routing (`search/anchors`), and
#: subscriptions by the delivery and resume paths (`entities/subscription.py`). Their title is still
#: sealed.
READABLE_CONTEXT_TYPES = frozenset({
    "application/vnd.agience.issuer+json",
    "application/vnd.agience.type+json",
    "application/vnd.agience.anchor+json",
    "application/vnd.agience.crosswalk+json",
    "application/vnd.agience.subscription+json",
})

#: Top-level fields a writer supplies: content, sealed with the context.
SEALED_DOC_FIELDS = ("name", "description")

#: Sealed even on a readable type: a title is part of the content of every artifact (`*/*`).
#: 2026-10-19 (John): "Title is really just part of content".
CONTENT_FIELDS = ("title",)
SEALED = "sealed"

#: The payload format is `{"v": 2, "context": {...}, "doc": {...}}`. A blob without `v` is the
#: title-only format (`{"title": ...}`) sealed before 2026-10-29, and still opens.
_PAYLOAD_VERSION = 2


def _content_type_of(doc: dict, ctx: dict) -> str:
    return str(ctx.get("content_type") or doc.get("content_type") or "").split(";", 1)[0].strip().lower()


def sealable_keys(doc: dict, ctx: dict) -> list:
    """The context keys of `doc` that are sealed: every one but the structural, and on a readable
    type the title alone."""
    if _content_type_of(doc, ctx) in READABLE_CONTEXT_TYPES:
        return [k for k in CONTENT_FIELDS if k in ctx]
    return [k for k in ctx if k not in STRUCTURAL_CONTEXT_KEYS and k != SEALED]


def sealable_doc_fields(doc: dict, ctx: dict) -> list:
    """The writer-supplied top-level fields of `doc` that are sealed: none on a readable type, whose
    name and description are configuration like the rest of it."""
    if _content_type_of(doc, ctx) in READABLE_CONTEXT_TYPES:
        return []
    return [k for k in SEALED_DOC_FIELDS if doc.get(k)]


#: Sealed, but not what a new version is cut for: what the server records about the bytes and the
#: mint (`services/mint_context.py`). They follow the body, and a new body already cuts a version
#: through `content_ref`; counting them as well made every completed upload fork its artifact.
UNVERSIONED_SEALED_KEYS = frozenset({
    "content_sha256", "content_bytes", "size", "addressing",
    "minted", "minted_by", "placement", "provenance",
})


def sealed_view(doc: dict) -> dict:
    """What `doc` would seal, in plaintext: its sealable context keys and its writer-supplied
    top-level fields."""
    ctx, _ = _context_dict(doc)
    ctx = ctx or {}
    return {"context": {k: ctx[k] for k in sealable_keys(doc, ctx)},
            "doc": {k: doc.get(k) for k in sealable_doc_fields(doc, ctx)}}


def versioned_view(doc: dict) -> dict:
    """What a new version is cut for (`lattice_api._content_changed`): the sealed view without the
    server's bookkeeping (`UNVERSIONED_SEALED_KEYS`)."""
    view = sealed_view(doc)
    view["context"] = {k: v for k, v in view["context"].items() if k not in UNVERSIONED_SEALED_KEYS}
    return view


def _context_dict(doc: dict):
    """(context as a dict, whether it was stored as a JSON string)."""
    raw = doc.get("context")
    if isinstance(raw, dict):
        return dict(raw), False
    if not raw:
        return {}, isinstance(raw, str)
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None, True
    return (parsed, True) if isinstance(parsed, dict) else (None, True)


def _put_context(doc: dict, ctx: dict, as_string: bool) -> None:
    doc["context"] = json.dumps(ctx) if as_string else ctx


def _open(principal, blob_b64, doc: dict) -> dict:
    import base64

    from mantle.services import content_crypto
    plain = content_crypto.decrypt_content(
        principal or content_key_principal(doc), base64.b64decode(blob_b64 or ""),
        collection_id=content_key_scope(doc), require_encrypted=True)
    payload = json.loads(plain.decode("utf-8"))
    if isinstance(payload, dict) and payload.get("v") == _PAYLOAD_VERSION:
        return {"context": payload.get("context") or {}, "doc": payload.get("doc") or {}}
    return {"context": payload if isinstance(payload, dict) else {}, "doc": {}}   # title-only format


def seal_content_fields(doc: dict) -> None:
    """Move the context's sealable keys and the writer-supplied top-level fields into
    `context.sealed`, encrypted under the body's key.

    A context that already carries a blob (a reader that could not open it, then saved) is merged:
    what was sealed stays, and what this write sets replaces it. A write that sets content on a row
    whose seal it cannot open is refused rather than allowed to overwrite the seal."""
    ctx, as_string = _context_dict(doc)
    if ctx is None:
        return
    fields = {k: ctx.pop(k) for k in sealable_keys(doc, ctx)}
    top = {k: doc.pop(k) for k in sealable_doc_fields(doc, ctx)}
    if not fields and not top:
        return
    import base64

    from mantle.services import content_crypto
    prior = ctx.get(SEALED)
    if prior:
        try:
            opened = _open(prior.get("p"), prior.get("blob"), doc)
        except KeyCustodyDenied:
            raise
        except Exception as exc:
            raise ContentEncryptionError(
                "this row's sealed fields cannot be opened by this writer; refusing to overwrite them") from exc
        fields = {**opened["context"], **fields}
        top = {**opened["doc"], **top}
    principal = content_key_principal(doc)
    payload = {"v": _PAYLOAD_VERSION, "context": fields, "doc": top}
    try:
        blob = content_crypto.encrypt_content(
            principal, json.dumps(payload).encode("utf-8"),
            collection_id=content_key_scope(doc), creator_id=doc.get("created_by"))
    except KeyCustodyDenied:
        raise
    except Exception as exc:
        logger.error("context field encryption failed (id=%s) — refusing to store plaintext",
                     doc.get("_key") or doc.get("id"), exc_info=True)
        raise ContentEncryptionError("context field encryption unavailable; refusing to persist plaintext") from exc
    ctx[SEALED] = {"p": principal, "blob": base64.b64encode(blob).decode("ascii")}
    _put_context(doc, ctx, as_string)


def unseal_content_fields(raw: dict) -> None:
    """Restore the sealed context keys and top-level fields for an authorized reader.

    Fails soft: a reader the key oracle will not serve (a background path with no acting
    principal, a row keyed elsewhere) gets the context without the fields and never an error — a
    missing title is visible, a failed read of a workspace is an outage. The sealed blob stays in
    that reader's context, so a save of what it read writes the fields back as they were sealed
    (`seal_content_fields` merges into a blob it can open and refuses one it cannot). 2026-10-29:
    the blob was taken out before the decrypt, and a background reader that then saved the row
    erased the sealed fields for good. A blob reveals nothing to a reader the oracle will not serve,
    and the feed and `_artifact_body` drop it."""
    ctx, as_string = _context_dict(raw)
    if not ctx or SEALED not in ctx:
        return
    sealed = ctx.get(SEALED) or {}
    try:
        opened = _open(sealed.get("p"), sealed.get("blob"), raw)
    except Exception:  # noqa: BLE001 — see the docstring
        logger.debug("context fields of %s stay sealed for this reader",
                     raw.get("_key") or raw.get("id"), exc_info=True)
        return
    ctx.pop(SEALED, None)
    for k, v in opened["context"].items():
        ctx.setdefault(k, v)
    for k, v in opened["doc"].items():
        if not raw.get(k):
            raw[k] = v
    _put_context(raw, ctx, as_string)


#: Document field carrying the collection's immutable origin root.
#:
#: Matches the vocabulary the lattice store already uses
#: (``db/content_cache.collection_key``), which takes ``origin_root`` as a
#: supplied value rather than resolving it. Same contract here.
ORIGIN_ROOT = "origin_root"


def content_key_principal(doc: dict) -> str:
    """The principal the content key roots at: the collection's origin root.

    ``db/content_cache.collection_key`` already sets the contract — it takes
    ``origin_root`` as a value and refuses an empty one. This matches it, so the two
    stores derive content keys the same way and neither needs a graph walk.

    Rooting at the origin root rather than ``created_by`` keeps ``created_by`` as pure
    provenance, correctable without touching a key: ``created_by`` is not part of the
    HKDF root or the AAD, so correcting it can never make a stored blob simultaneously
    underivable and unauthenticatable.

    A blob with no ``origin_root``/``content_key_principal`` field stays keyed to
    ``created_by`` — see ``scripts/migrate_encrypt_content``, which re-keys it onto
    the origin root.
    """
    origin_root = doc.get(ORIGIN_ROOT) or doc.get(CONTENT_KEY_PRINCIPAL)

    # A top-level artifact's `origin_root` is its own id (`lattice_api._stamp_origin_root`:
    # "top-level artifact IS the root"), so it is self-rooted. That is safe because
    # `KeyRequest.creator_id` (minting path only) lets the creator hold the key of the
    # artifact it is creating — key custody and the light cone's grant on
    # `(artifact_id, artifact_id)` then answer the same question. Per artifact, the key is
    # exactly as wide as the grant it came from, rather than one key per creator spanning
    # every top-level artifact they own.
    #
    # `decrypt_artifact_content` reads the per-doc `CONTENT_KEY_PRINCIPAL` stamped at write
    # time and never calls this function, so a blob already written under `created_by` keeps
    # that value and stays readable — only new writes take the origin-root path.
    if origin_root:
        return origin_root

    legacy = doc.get("created_by")
    if not legacy:
        # Neither root available: a missing key principal is a failed write.
        raise ContentEncryptionError(
            "no content key principal: the artifact carries neither "
            f"{ORIGIN_ROOT!r} nor 'created_by', so there is nothing stable to key "
            "its content to"
        )
    return legacy


def content_key_scope(doc: dict) -> str:
    """The collection the content-key grant check narrows to.

    One reader, used by both the encrypt and decrypt paths — they must agree. The scope decides
    which branch of `LightConeGrantVerifier.authorized` answers, and a blob written under one scope
    and read under another is a blob nobody can open.

    A member of a collection scopes to that collection, which is what makes sharing a collection
    reach its members' keys. A top-level artifact is its own scope: the light cone resolves a grant
    on it to `(artifact_id, artifact_id)`, so naming the artifact is what lets a grantee's own grant
    reach the key.

    Server-derived, never caller-supplied — every value comes off the stored doc, so a caller
    cannot nominate the scope its key is checked against.
    """
    coll = doc.get("collection_id")
    if coll:
        return coll
    origin_root = doc.get(ORIGIN_ROOT) or doc.get(CONTENT_KEY_PRINCIPAL)
    if origin_root and origin_root == doc.get("id"):
        return origin_root
    return ""


def encrypt_artifact_content(doc: dict) -> None:
    """Envelope-encrypt an artifact doc's inline ``content`` in place.

    Keyed on the collection's origin root — see :func:`content_key_principal`, NOT
    on ``created_by``.

    Storage sees only ciphertext; the in-memory entity (used for indexing + API
    output) is untouched because the doc converters build a fresh dict. Idempotent
    via the ``content_encrypted`` flag.

    Failing the write is the correct trade: content encryption is a security
    control, and skipping it when it is unavailable would be a fail-open shape
    this codebase does not accept.
    """
    seal_content_fields(doc)
    content = doc.get("content")

    # Addressed content never enters the document. The bytes are already in the CAS at
    # `cas/<sha256 of the plaintext>`, envelope-encrypted for their owner by
    # `content_service.put_bytes_encrypted`, so the row carries the ADDRESS and nothing else.
    # `decrypt_artifact_content` hydrates the body back on read, which is why indexing, recall
    # previews and `GET /artifacts/{id}` never had to learn where it lives. Measured on 71/dev
    # before this: 7.2 MB of 7.9 MB of doc bytes were artifact content; a 9,800-character body
    # now leaves a 981-byte row.
    if doc.get("content_ref"):
        doc["content"] = ""
        doc["content_encrypted"] = False
        return

    # Everything below is for rows written before the content tier, whose bytes exist nowhere else.
    # Clearing an unaddressed body here would destroy the only copy, and encrypting it is the
    # protection it has. The write path cannot reach this — `_store_content_in_s3` addresses every
    # body it stores — so it is a read-modify-write of existing data rather than a second way to put
    # content into the lattice.
    if not content or doc.get("content_encrypted"):
        return
    try:
        import base64
        try:
            from mantle.services import content_crypto
        except ImportError:
            from mantle.services import content_crypto
        principal = content_key_principal(doc)
        raw = content if isinstance(content, (bytes, bytearray)) else str(content).encode("utf-8")
        blob = content_crypto.encrypt_content(
            principal, raw,
            collection_id=content_key_scope(doc),
            # The encrypt path only: `decrypt_artifact_content` below does not pass
            # this, because on a read it would let anyone naming themselves the creator obtain a key.
            creator_id=doc.get("created_by"),
        )
        doc["content"] = base64.b64encode(blob).decode("ascii")
        doc["content_encrypted"] = True
        doc[CONTENT_KEY_PRINCIPAL] = principal
    except KeyCustodyDenied:
        raise
    except Exception as exc:
        logger.error(
            "artifact content encryption failed (id=%s) — refusing to store plaintext",
            doc.get("_key") or doc.get("id"),
            exc_info=True,
        )
        raise ContentEncryptionError(
            "content encryption unavailable; refusing to persist plaintext"
        ) from exc

def decrypt_artifact_content(raw: dict, *, strict: bool = True) -> None:
    # The artifact is named to the key oracle, so a grant on it alone opens it (`KeyRequest`).
    from mantle.services.content_crypto import content_key_artifact
    with content_key_artifact(raw.get("root_id") or raw.get("id")):
        unseal_content_fields(raw)
        _decrypt_body(raw, strict=strict)


def _decrypt_body(raw: dict, *, strict: bool = True) -> None:
    """Decrypt an artifact doc's inline ``content`` in place if flagged.

    A failed decrypt must never leave ``content`` holding ciphertext with the
    ``content_encrypted`` flag cleared: a caller that then saves the entity back would have
    ``encrypt_artifact_content`` see unflagged content and encrypt the ciphertext a second
    time, making the original plaintext unrecoverable. Two guards close this: ``Artifact``
    models ``content_encrypted`` so the flag survives a round trip through ``from_dict``,
    and this function never hands ciphertext back as content.

    ``strict=True`` (single-artifact reads) raises, so the caller gets an error
    instead of ciphertext dressed as plaintext. ``strict=False`` (list/stream
    paths, where one bad row must not fail the whole page) drops the content and
    keeps the flag set, so the row is visibly incomplete rather than silently wrong.

    Nor does it hand back *unauthenticated* bytes: the decrypt is made with
    ``require_encrypted=True``, so a flagged doc whose blob carries no MEC1 magic is an
    error rather than a passthrough — see the comment at the call below.
    """
    # The bytes come from the CAS. `encrypt_artifact_content` keeps addressed content out of the
    # document, so a row with a `content_ref` carries no body and this is where it comes back.
    # Hydrating at the boundary rather than at each call site is what keeps indexing, previews and
    # `GET /artifacts/{id}` reading `artifact.content` without learning where it lives.
    #
    # The read is authorized, not merely addressed: `get_bytes_decrypted` demands the envelope
    # this artifact's own write sealed, and `FileContentCache` verifies the bytes against the
    # address on the way out. A ref edited to name another object yields an error, not bytes.
    ref = raw.get("content_ref")
    if ref and not (raw.get("content") or ""):
        # ── the WRITE keys on the collection; the READ must try that too ────────────────────
        # `encrypt_artifact_content` seals under `content_key_principal` — "the collection's
        # origin root ... NOT on `created_by`" — while this read asked for
        # `CONTENT_KEY_PRINCIPAL or created_by`. The two agree only when `origin_root` is stamped.
        #
        # Measured 2026-08-25 on 71/home: the 310,003 bulk-ingested Wikipedia artifacts carry
        # `collection_id` and NO `origin_root`, so the read asked for `created_by` (`54eaa8aa`,
        # ember-source) and the envelope had been sealed under `stage.1.grammar`. Every body failed
        # to hydrate, and the error read `GrantDenied` — a grant on `created_by` was minted to
        # chase it and changed nothing, because that principal has no master key at all. Opening
        # the same blob under the collection returned 16,116 bytes on the first try.
        #
        # Ordered candidates rather than one guess: the stamped principal is authoritative when it
        # is there, the collection scope is what the writer actually used, and `created_by` stays
        # last for the rows `content_key_principal` describes as keyed to `created_by`. A wrong
        # candidate raises and the next is tried; none of them can
        # return the wrong plaintext, because AES-GCM authenticates.
        scope = content_key_scope(raw)
        candidates = []
        for candidate in (raw.get(CONTENT_KEY_PRINCIPAL), scope, raw.get("created_by")):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        try:
            from mantle.services.content_service import get_bytes_decrypted
            context = raw.get("context")
            if isinstance(context, str):
                try:
                    context = json.loads(context)
                except (TypeError, ValueError):
                    context = {}
            content_key = (context or {}).get("content_key") if isinstance(context, dict) else None
            last = None
            for principal in candidates:
                try:
                    # `probing=True`: a miss here is EXPECTED — that is what an ordered
                    # candidate list means. Without it every read of a body whose first
                    # candidate misses logged an ERROR with a traceback, so a node with zero
                    # actual failures still emitted one error per read (744 in a day here).
                    # The real failure is still reported: `last` is re-raised below and the
                    # WARNING under it names the artifact.
                    raw["content"] = get_bytes_decrypted(
                        content_key or "", principal, cas_ref=ref, collection_id=scope,
                        probing=True,
                    ).decode("utf-8")
                    last = None
                    break
                except Exception as attempt:  # noqa: BLE001 — try the next candidate
                    last = attempt
            if last is not None:
                raise last
        except Exception as exc:
            # An unreadable body is not an empty one. `strict` carries the same meaning it does
            # below: a single-artifact read raises rather than hand back a document that looks
            # like it has no content, and a list page drops the body so the row is visibly
            # incomplete instead of silently wrong.
            logger.warning("content hydration failed for %s from %s",
                           raw.get("_key") or raw.get("id"), ref, exc_info=True)
            if strict:
                raise ContentDecryptionError(
                    f"artifact content is addressed at {ref} but could not be read: {exc}"
                ) from exc
            raw["content"] = ""
        return

    if not raw.get("content_encrypted"):
        return

    # Which principal this blob was keyed under. New blobs record it explicitly;
    # a blob with no such field is keyed under `created_by`, so fall back to that.
    #
    principal = raw.get(CONTENT_KEY_PRINCIPAL) or raw.get("created_by")
    try:
        import base64
        try:
            from mantle.services import content_crypto
        except ImportError:
            from mantle.services import content_crypto
        blob = base64.b64decode(raw.get("content") or "")
        raw["content"] = content_crypto.decrypt_content(
            principal, blob, collection_id=content_key_scope(raw),
            # `require_encrypted=True` because this branch is only reached when the doc
            # ITSELF asserts `content_encrypted`. Without it, `decrypt_content`'s
            # "no MEC1 magic → return the bytes unchanged" compatibility path would be an
            # unauthenticated downgrade *here*: an attacker with store-write access could
            # replace authenticated ciphertext with chosen plaintext and have it served
            # as artifact content, with no tag to fail. That compatibility path is
            # legitimate for `content_service.get_bytes_decrypted` — S3 objects predating
            # envelope encryption genuinely carry no flag and no magic — but this caller
            # has already been told the blob is encrypted, so a blob that isn't means
            # tampering or corruption, and either way it must fail rather than be served.
            require_encrypted=True,
        ).decode("utf-8")
        raw["content_encrypted"] = False
    except KeyCustodyDenied:
        raise
    except Exception as exc:
        logger.error("failed to decrypt artifact content (id=%s)",
                     raw.get("_key") or raw.get("id"), exc_info=True)
        if strict:
            raise ContentDecryptionError(
                f"artifact {raw.get('_key') or raw.get('id')} content could not be decrypted"
            ) from exc
        # Non-strict: never surface ciphertext as content, and leave
        # content_encrypted set so a write-back cannot double-encrypt.
        raw["content"] = None
        raw["content_encrypted"] = True


#: The lifecycle event names the change feed is defined over.
#:
#: Enumerated here, at the boundary that emits two of the three, because the set is the contract a
#: subscriber filters on and `tests/test_change_feed_is_complete_by_construction.py` checks the
#: tree against. A fourth name arriving without appearing here means a subscriber cannot know to
#: ask for it.
CREATED = "artifact.created"
UPDATED = "artifact.updated"
DELETED = "artifact.deleted"
CHANGE_EVENTS = (CREATED, UPDATED, DELETED)


def emit_artifact_change(entity, event_name: str) -> None:
    """Emit a change event for an artifact write.

    Every artifact create and update goes through the store adapter's create/update functions, so
    emitting HERE makes those two complete by construction — "everything is an artifact, and
    deserves an event". No such write is silent, including raw ones (the Person card, provisioning,
    seeds).

    **Deletes are emitted one layer up**, by the services: a delete arrives at this boundary as an
    id, after the doc it names is already gone,
    so the container the event has to be addressed to is no longer readable from anything this
    function is given. The service holds that context and emits there
    (`workspace_service._emit_event`, `collection_service._emit`). The consequence is that delete
    coverage is discipline where create/update coverage is construction, which is exactly why the
    guard test enumerates the delete sites rather than trusting them.

    Best-effort: a failed emit never breaks the write. The write is the fact; the event is the
    announcement of it, and an announcement that cannot be made must not undo what it was
    announcing.

    **The announcement carries no body.** `encrypt_artifact_content` above encrypts a fresh doc
    dict, so the entity handed to this function still holds its plaintext `content` — encrypting
    for storage does nothing for the copy that leaves through here. The descriptor is built with
    `event_bus.redacted_artifact`, which is the same rule `event_bus.publish_event` enforces at
    the seam; applying it here as well means the plaintext never leaves this function, so no
    future emit path can reach the bus with it still attached. A subscriber that needs the bytes
    reads them back through `decrypt_artifact_content`, where custody is checked.
    """
    try:
        from mantle.events import event_bus
        d = entity.to_dict() if hasattr(entity, "to_dict") else dict(entity)
        container = d.get("collection_id") or d.get("root_id") or d.get("id") or ""
        event_bus.emit_artifact_event_sync(
            container, event_name, {"artifact": event_bus.redacted_artifact(d)},
            actor_id=d.get("modified_by") or d.get("created_by"),
        )
    except Exception:
        logger.debug("artifact event emit failed (%s)", event_name, exc_info=True)


__all__ = [
    "ContentEncryptionError", "ContentDecryptionError",
    "CONTENT_KEY_PRINCIPAL", "ORIGIN_ROOT",
    "CREATED", "UPDATED", "DELETED", "CHANGE_EVENTS",
    "content_key_principal", "content_key_scope", "encrypt_artifact_content", "decrypt_artifact_content",
    "emit_artifact_change",
]
