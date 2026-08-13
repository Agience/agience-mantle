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
    content = doc.get("content")
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

    **Deletes are emitted one layer up**, by the services, and that is a decision rather than an
    omission: a delete arrives at this boundary as an id, after the doc it names is already gone,
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
