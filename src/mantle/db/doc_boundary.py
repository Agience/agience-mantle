"""The artifact persistence boundary — content envelope crypto + the change-event chokepoint.

ONE implementation, shared by every store adapter (`db.store` while it lives, `db.lattice_api`
going forward). Extracted from `db.store` 2026-07-22 so the lattice path could not fork the
security behavior: same MEC1 wire format, same origin-root key principal, same fail-closed rules.
Store-agnostic on purpose — everything here operates on plain doc dicts / entities and knows
nothing about where they are persisted.

The load-bearing docstrings (WHY the principal is the origin root, WHY a failed encrypt fails the
write, WHY a failed decrypt never returns ciphertext) moved here with the code. Read them before
touching anything.
"""
from __future__ import annotations

import logging

# Module level, NOT inside the try blocks that use it: an `except KeyCustodyDenied`
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


#: Document field recording WHICH principal the content key was derived from.
#:
#: Written at encrypt time so the read path does not have to re-resolve it (the
#: converters have no db handle). Safe to store and to trust: the collection's
#: origin root is immutable, and the value is also the GCM **AAD**, so a blob whose
#: recorded root is edited fails to authenticate rather than decrypting under the
#: attacker's choice.
CONTENT_KEY_PRINCIPAL = "content_key_principal"


#: Document field carrying the collection's immutable origin root.
#:
#: Matches the vocabulary the lattice store already uses
#: (``db/lattice/content_cache.collection_key``), which takes ``origin_root`` as a
#: SUPPLIED VALUE rather than resolving it. Same contract here.
ORIGIN_ROOT = "origin_root"


def content_key_principal(doc: dict) -> str:
    """The principal the content key roots at: the collection's ORIGIN ROOT.

    ⚠ READ FROM THE DOCUMENT — NEVER LOOKED UP. An earlier draft of this called
    ``resolve_cell_principal(next(get_store_db()), ...)``, which was wrong twice
    over: it put a database round trip on every artifact write (42s of connect
    timeouts per test in the suite), and it added the lattice dependency to the one
    layer that is supposed to be store-agnostic while the lattice is being retired.

    ``db/lattice/content_cache.collection_key`` already sets the contract — it takes
    ``origin_root`` as a value and refuses an empty one. This matches it, so the two
    stores derive content keys the same way and neither needs a graph walk.

    ⚠ THIS USED TO BE ``created_by``, AND THAT WAS THE BUG. ``oracle.py`` states the
    rule in as many words: *"The principal is the collection's immutable origin
    root, NOT an 'owner' / created_by. Agience has no owners — access is by
    grant."* The search path obeyed it; artifact content did not, so the same
    artifact had TWO key roots — cells under the origin root, content under a user
    id. Rooting crypto at ``created_by`` reintroduces ownership through the back
    door, in the one place it is hardest to see.

    It also made a DECIDED migration step destructive. ``created_by`` is provenance
    and provenance gets corrected — LATTICE §3 decides John's two forms are folded
    into one — but ``created_by`` was both the HKDF root AND the AAD, so folding an
    identity made every blob written under the dropped value simultaneously
    underivable and unauthenticatable. A metadata correction would have silently
    destroyed readability.

    Rooting at the origin root leaves ``created_by`` as what it should be: pure
    provenance, correctable without touching a key.

    ⛔ THE ``created_by`` FALLBACK IS A TRANSITIONAL SHIM AND MUST BE REMOVED.
    Artifacts do not yet carry ``origin_root`` on every legacy row — until it is
    stamped everywhere, a document without one still keys under ``created_by``,
    exactly as before: no regression, no new failure mode, and no silent change of
    key root for existing rows.

    While that fallback stands, **the identity fold is unsafe** — see
    ``scripts/migrate_encrypt_content``, which re-keys away from it.
    """
    origin_root = doc.get(ORIGIN_ROOT) or doc.get(CONTENT_KEY_PRINCIPAL)
    if origin_root:
        return origin_root

    legacy = doc.get("created_by")
    if not legacy:
        # Neither root available. Previously this returned early and stored the
        # content as PLAINTEXT; a missing key principal is now a failed write.
        raise ContentEncryptionError(
            "no content key principal: the artifact carries neither "
            f"{ORIGIN_ROOT!r} nor 'created_by', so there is nothing stable to key "
            "its content to"
        )
    return legacy


def encrypt_artifact_content(doc: dict) -> None:
    """Envelope-encrypt an artifact doc's inline ``content`` in place.

    Keyed on the collection's origin root — see :func:`content_key_principal`, NOT
    on ``created_by``.

    Storage sees only ciphertext; the in-memory entity (used for indexing + API
    output) is untouched because the doc converters build a fresh dict. Idempotent
    via the ``content_encrypted`` flag.

    ⛔ THIS USED TO SWALLOW THE FAILURE AND STORE PLAINTEXT.
    A write whose encryption failed was persisted unencrypted with no
    ``content_encrypted`` flag — and because a blob without the ``MEC1`` magic
    reads back as "legacy plaintext", the read path returned it happily forever.
    The degradation was therefore SILENT, PERMANENT, and INVISIBLE: nothing
    downstream could tell an intentionally-plaintext legacy row from one that
    should have been encrypted and was not.

    Failing the write is the correct trade. Content encryption is a security
    control, and "the control was unavailable so we skipped it" is the
    fail-open shape this codebase keeps getting bitten by.
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
            principal, raw, collection_id=doc.get("collection_id")
        )
        doc["content"] = base64.b64encode(blob).decode("ascii")
        doc["content_encrypted"] = True
        doc[CONTENT_KEY_PRINCIPAL] = principal
    except KeyCustodyDenied:
        # ⛔ NOT A CRYPTO FAILURE — AN AUTHORIZATION ONE. The handler below reports
        # "encryption unavailable", which for a refusal would be actively
        # misleading: it reads as a broken key service and invites someone to
        # "fix" it by relaxing the control. Let the denial through unchanged so the
        # caller can map it to a 403.
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

    ⛔ ON FAILURE THIS USED TO LEAVE THE CIPHERTEXT IN ``content`` AND RETURN 200.
    That fed a data-destruction chain, not just a bad response:

      1. read fails to decrypt -> ``content`` holds base64 ciphertext
      2. ``Artifact`` did not model ``content_encrypted``, so ``from_dict``
         DROPPED the flag
      3. the caller saves the entity back -> ``encrypt_artifact_content`` sees
         content with no flag and encrypts the CIPHERTEXT a second time
      4. the original plaintext is now unrecoverable

    Two independent guards now close this. ``Artifact`` models
    ``content_encrypted`` so the flag survives the round trip and step 3 becomes
    a no-op; and this function no longer hands ciphertext back as content.

    ``strict=True`` (single-artifact reads) raises, so the caller gets an error
    instead of ciphertext dressed as plaintext. ``strict=False`` (list/stream
    paths, where one bad row must not fail the whole page) drops the content and
    keeps the flag set, so the row is visibly incomplete rather than silently wrong.
    """
    if not raw.get("content_encrypted"):
        return

    # Which principal this blob was keyed under. New blobs record it explicitly;
    # blobs written before the re-root carry no such field and are keyed under
    # `created_by`, so fall back to that.
    #
    # ⚠ THE FALLBACK IS A MIGRATION SHIM, NOT THE DESIGN. It exists so existing MEC1
    # content stays readable through the re-root; it is what `migrate_encrypt_content`
    # re-keys away. It cannot be used to widen access: the recorded principal is also
    # the AAD, so a blob only decrypts under the exact root it was written with, and
    # the grant check runs against that root either way.
    principal = raw.get(CONTENT_KEY_PRINCIPAL) or raw.get("created_by")
    try:
        import base64
        try:
            from mantle.services import content_crypto
        except ImportError:
            from mantle.services import content_crypto
        blob = base64.b64decode(raw.get("content") or "")
        raw["content"] = content_crypto.decrypt_content(
            principal, blob, collection_id=raw.get("collection_id")
        ).decode("utf-8")
        raw["content_encrypted"] = False
    except KeyCustodyDenied:
        # ⛔ NOT A DECRYPTION FAILURE — AN AUTHORIZATION ONE, and the difference
        # matters in BOTH modes below. Strict would relabel a denial as
        # "could not be decrypted" (reads as corruption, not as "you may not read
        # this"). Non-strict is worse: it would swallow the refusal and return the
        # row with `content: None`, which is indistinguishable from a genuinely
        # empty artifact — an authorization failure rendered as ordinary data.
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


def emit_artifact_change(entity, event_name: str) -> None:
    """Emit a change event for an artifact write.

    Every artifact write goes through the store adapter's create/update functions, so
    emitting HERE makes the change-feed complete by construction — "everything is an
    artifact, and deserves an event". No write is silent, including raw ones (the
    Person card, provisioning, seeds). Best-effort: a failed emit never breaks the write.
    """
    try:
        from mantle import event_bus
        d = entity.to_dict() if hasattr(entity, "to_dict") else dict(entity)
        container = d.get("collection_id") or d.get("root_id") or d.get("id") or ""
        event_bus.emit_artifact_event_sync(
            container, event_name, {"artifact": d},
            actor_id=d.get("modified_by") or d.get("created_by"),
        )
    except Exception:
        logger.debug("artifact event emit failed (%s)", event_name, exc_info=True)


__all__ = [
    "ContentEncryptionError", "ContentDecryptionError",
    "CONTENT_KEY_PRINCIPAL", "ORIGIN_ROOT",
    "content_key_principal", "encrypt_artifact_content", "decrypt_artifact_content",
    "emit_artifact_change",
]
