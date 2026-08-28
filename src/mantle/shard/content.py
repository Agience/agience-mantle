"""Content service — CONTENT lives in the content store, ENCRYPTED and content-addressed.

The proper content/context separation (the label-blind mesh model):
  • the artifact store (SQLite lattice) holds the CONTEXT + a REFERENCE (`content_ref` = the
    content's address). Queryable.
  • the content store (local CAS cache / S3 mirror) holds the CONTENT as
    opaque ENCRYPTED bytes. No size limit, no truncation.

Content-addressing: `content_ref = cas/<sha256(plaintext)>`. Identical content -> identical
ref -> stored once (dedup for free) and verifiable (the mesh Merkle root is over these).
Encryption: the stored bytes are ciphertext; only a holder of the content key reads them
(single-owner symmetric key now; per-owner envelope encryption is the multi-tenant upgrade).
"""
from __future__ import annotations

import hashlib
import re
import time as _time
from pathlib import Path
from typing import Optional, Tuple

from cryptography.fernet import Fernet, MultiFernet

CAS_PREFIX = "cas/"


class ContentKeyMissing(RuntimeError):
    """No usable `content.key` — a CONFIGURATION fault, never a missing artifact.

    Raised instead of silently minting a replacement. Distinct from a decrypt failure so callers
    can swallow "this one blob is unreadable" without also swallowing "this node has no key"."""


class ContentStillSealed(RuntimeError):
    """The bytes came back as an unopened MEC1 envelope — a CUSTODY fault, not empty content.

    Content is sealed under its collection's origin root (`doc_boundary.content_key_principal`),
    so opening it needs an acting principal holding a read grant. A caller without one used to
    receive the envelope itself, which `.decode("utf-8", "ignore")` then turned into a lossy
    pseudo-string: measured 2026-08-27, 154,865 bytes of AES-GCM ciphertext arrived 69% printable
    and nothing raised.

    That return value is worse than an error because it is INDISTINGUISHABLE FROM AN EMPTY
    DOCUMENT. Downstream, `sage/describe` found no terms in it, took its documented
    always-terminating fallback (`lemmas = [... or "document"]`), and `describe_dark` — which skips
    anything that already has lemmas — never revisited it. 874 capture artifacts carry the literal
    `['document']` and 473 carry `['module']`: bodies nobody could read, recorded as bodies nobody
    needed to read. `_split`'s own docstring records fixing the same SHAPE once before, when a path
    line made `ast.parse` fail and the file was "still keyed from its stem, so `describe_dark`
    skips it forever with no error at any layer."

    So a caller that cannot open the envelope is told so."""


class ContentIntegrityError(RuntimeError):
    """The blob decrypted but does not hash to the ref it was fetched under.

    Fernet authenticates the ciphertext under the node's content key; it does not — and with no
    AAD parameter cannot — bind the blob to its content address. So a ciphertext moved between two
    refs in the store, or served for the wrong ref by a mirror, decrypts cleanly and returns the
    wrong content under the right name. `cas/<sha256(plaintext)>` is the address, so re-hashing the
    plaintext is the binding Fernet can't express, and it is checked rather than assumed — the same
    verify-on-read contract `db/content_cache.py` already holds itself to.

    Never an empty return: a caller that cannot use the content gets this exception, and
    `resolve_text` degrades to the artifact's inline content rather than to wrong bytes."""


_KEY_CACHE: dict = {}
_KEY_TTL = 60.0


def _content_key(keys_dir: Path, *, create: bool = False) -> MultiFernet:
    """The content/segment cipher. Returns a MultiFernet: `content.key` is primary (used for
    ENCRYPT), and any additional `content.key.*` files are decrypt-only fallbacks. This lets a
    node decrypt peers' segments written under a different key (the mesh keys diverge: t5/tu each
    have their own, 71+45 share one). Single-key nodes are unaffected —
    encrypt still uses their one key, and MultiFernet with one key behaves exactly like Fernet.

    Key-bootstrap properties:

    1. `create` defaults to False. Only `put_content` opts in — a write on a provisioned node may
       legitimately bootstrap the first key. Reads never mint (a read only happens when content
       already exists, so a missing key there is always a fault), and `sync._fernet` never mints:
       publishing under a fresh key is the worst outcome available, because it corrupts the mesh
       for every peer rather than just this node.
    2. No `mkdir`. An absent `keys_dir` means the volume is not mounted; that is a hard error, not
       an invitation to create one. Bootstrapping requires the directory to already exist, which is
       what distinguishes "provisioned but empty" from "not mounted".
    3. Exclusive create (`"xb"`). Under concurrent writers, two threads could otherwise both see no
       key, mint different ones, and the loser's ciphertext become permanently unreadable. On
       collision we re-read the winner's key instead of overwriting it.

    Cached per keys_dir for `_KEY_TTL`, so a stat, a read, a glob and N Fernet constructions run
    at most once per TTL window rather than on every put/get.
    """
    kd = Path(keys_dir)
    ck = str(kd.resolve()) if kd.exists() else str(kd)
    now = _time.time()
    ent = _KEY_CACHE.get(ck)
    if ent is not None and now - ent[0] <= _KEY_TTL:
        return ent[1]

    kp = kd / "content.key"
    if not kp.exists():
        if not create:
            raise ContentKeyMissing(
                f"no content.key in {kd} — refusing to mint one on this path. A generated key "
                f"would silently partition this node's content from the mesh. Check that the keys "
                f"volume is mounted (EMBER_STORE_KEYS_DIR)."
            )
        if not kd.is_dir():
            raise ContentKeyMissing(
                f"keys dir {kd} does not exist — the keys volume is probably not mounted. "
                f"Refusing to create it and bootstrap a key, which would partition this node."
            )
        key = Fernet.generate_key()
        try:
            with open(kp, "xb") as fh:          # exclusive: loser re-reads rather than clobbering
                fh.write(key)
        except FileExistsError:
            pass                                # another thread won the race; fall through and read
    keys = [Fernet(kp.read_bytes().strip())]                 # primary (encrypt) first
    for extra in sorted(kd.glob("content.key.*")):           # decrypt-only fallbacks
        try:
            keys.append(Fernet(extra.read_bytes().strip()))
        except Exception:
            pass
    mf = MultiFernet(keys)
    _KEY_CACHE[ck] = (now, mf)
    return mf


def content_ref(plaintext: bytes) -> str:
    """The content address — sha256 of the PLAINTEXT, so identical content dedupes regardless
    of the (nonce'd) ciphertext."""
    return CAS_PREFIX + hashlib.sha256(plaintext).hexdigest()


def _put_wants_collection(content_store) -> bool:
    """Does this store's `put` require a `collection`? Answers for both content-store shapes.

    There are two, with incompatible contracts:

        FsContentStore.put(ref, ciphertext)                     — the caller encrypts, no scope
        TieredContentStore.put(ref, plaintext, *, collection)   — the store encrypts, and scopes

    `db.backend.content_handle()` — what a deployed node has — returns the second, so a caller
    built against the first shape raises `TypeError: put() missing 1 required keyword-only
    argument: 'collection'`, and passing ciphertext once that is satisfied would double-encrypt.

    Detected from the signature rather than by an isinstance check: the store is injected (the
    platform is a function argument), so this file must not know the concrete classes. A store that
    grows the parameter later is handled without editing this function.
    """
    try:
        import inspect
        params = inspect.signature(content_store.put).parameters
    except (TypeError, ValueError, AttributeError):
        return False
    p = params.get("collection")
    return p is not None and p.default is p.empty


def put_content(content_store, keys_dir, data: bytes, *, collection: str | None = None) -> Tuple[str, int]:
    """Store content-addressed in the content store. Returns (content_ref, size).

    Idempotent: identical content is written once.

    `collection` is required by the stores that take it: `mantle/oci/__init__.py` states the
    mapping — repository is a collection — and this is where it is supplied: an image's blobs name
    their repository, federated bodies name the artifact's own collection.

    It is not a storage scope. `FileContentCache.path_of` derives the object path from the ref
    alone; per-collection keying is deliberately unused for writes, because addressing objects
    globally while keying them per collection would let one shared ref overwrite another root's
    copy. The parameter selects the legacy per-collection key on read, for objects predating the
    shared-key scheme — passing the true value is correct and matters for those, but it does not
    partition storage and is not authorization.

    The encryption side flips with it, which a signature change alone would miss: a scoping store
    takes plaintext and encrypts internally, so handing it ciphertext would encrypt twice and store
    something that never decrypts to its own ref. `FileContentCache.put` additionally verifies
    `sha256(plaintext)` against the ref — a check that only means anything when it really is given
    plaintext.

    Refused, not defaulted, when a scoping store is handed no collection: a plausible default
    ("public", the node id) is an authorization decision made by accident, and the cost lands on a
    reader who is later surprised by what a grant reaches.
    """
    ref = content_ref(data)
    if content_store.exists(ref):
        return ref, len(data)

    if _put_wants_collection(content_store):
        if not collection:
            raise ValueError(
                "this content store scopes writes by collection and none was given for %s. "
                "Pass the repository (for an image or a git object) or the artifact's own "
                "collection_id (for federated content) — guessing one here would be an "
                "authorization decision taken by accident." % ref)
        content_store.put(ref, data, collection=collection)
        return ref, len(data)

    # The caller-encrypts shape. The ONLY caller allowed to bootstrap: a write on a provisioned node
    # may legitimately mint the first key. Still refuses if the keys dir is absent.
    content_store.put(ref, _content_key(keys_dir, create=True).encrypt(data))
    return ref, len(data)


_CAS_HEX = re.compile(r"^[0-9a-f]{64}$")


def _verify_ref(ref: str, plaintext: bytes) -> bytes:
    """Return `plaintext` iff it hashes to `ref`. See :class:`ContentIntegrityError`.

    Only a well-formed `cas/<64 hex>` ref carries a hash to check against. Anything else is an
    addressing scheme this function was not given evidence for, and inventing a check for it
    would be guessing — those pass through, exactly as before."""
    if not ref.startswith(CAS_PREFIX) or not _CAS_HEX.match(ref[len(CAS_PREFIX):]):
        return plaintext
    got = hashlib.sha256(plaintext).hexdigest()
    if got != ref[len(CAS_PREFIX):]:
        raise ContentIntegrityError(
            "content fetched as %s decrypted to bytes hashing to cas/%s — the ciphertext is "
            "authentic under this node's content key but is NOT the content this ref names. "
            "Refusing to return it." % (ref, got))
    return plaintext


def _get_wants_collection(content_store) -> bool:
    """The read-side twin of `_put_wants_collection`. Same two shapes, same reason.

        FsContentStore.get(ref) -> CIPHERTEXT          — the caller decrypts
        TieredContentStore.get(ref, *, collection) -> PLAINTEXT

    This asymmetry means the write side alone is not enough: a caller that writes through the
    scoping store's `put` and then reads back what it just wrote — including `routers/oci_router`
    serving a blob from a real node — needs the same collection-aware detection here, or it raises
    `TypeError: get() missing 1 required keyword-only argument: 'collection'`, a 500 on every pull.
    """
    try:
        import inspect
        params = inspect.signature(content_store.get).parameters
    except (TypeError, ValueError, AttributeError):
        return False
    p = params.get("collection")
    return p is not None and p.default is p.empty


def get_content(content_store, keys_dir, ref: str, *, collection: str | None = None) -> bytes:
    """Fetch content by ref and VERIFY it against that ref.

    Fernet gives no AAD parameter, so the ciphertext cannot be cryptographically bound to its
    address the way `cell.py` and `content_cache.py` bind theirs. The content address is the
    binding available here, and re-hashing is what enforces it — the check is a pure addition, it
    changes no stored byte, and it is the difference between "authentic" and "authentic AND the
    content this ref names".

    Two store shapes, and the verify is the one thing both paths share. A scoping
    store returns plaintext and needs the collection; the caller-decrypts store returns ciphertext.
    Decrypting what is already plaintext would fail as an InvalidToken and read as a key problem,
    so the branch is on the store's contract rather than on a guess about the bytes. `_verify_ref`
    runs either way — it is the check that makes a read mean "this ref's content" rather than
    "something the store handed back".
    """
    if _get_wants_collection(content_store):
        if not collection:
            raise ValueError(
                "this content store scopes reads by collection and none was given for %s. "
                "Pass the repository (for an image or a git object) or the artifact's own "
                "collection_id — guessing one here would read from a scope the caller never "
                "named." % ref)
        return _verify_ref(ref, content_store.get(ref, collection=collection))
    return _verify_ref(ref, _content_key(keys_dir).decrypt(content_store.get(ref)))


def resolve_text(store_bundle, artifact: dict) -> str:
    """The full text of an artifact — from the content store via `content_ref`, or inline
    `content` (legacy / small artifacts like WordNet defs). Never truncated."""
    ref = artifact.get("content_ref")
    content = store_bundle.content
    # One path on the lattice backend: the tiered read seam (mantle `TieredContentStore`) — local
    # FileContentCache first, then the S3/CDN mirror with sha256 verify-on-pull. The tier is built
    # unconditionally whenever the store's content is the lattice cache (local-only when
    # air-gapped), so there is no separate FileContentCache branch: the tier already tried the
    # local cache internally, and on any tier failure the only remaining fallback is inline
    # `content` (never a wrong key). `ContentKeyMissing` propagates — a node-wide configuration
    # fault must not read as an empty artifact.
    tier = getattr(store_bundle, "content_tier", None)
    if ref and tier is not None:
        try:
            return _as_text(tier.get(ref, collection=artifact.get("collection_id")),
                            ref, artifact)
        except ContentStillSealed:
            raise
        except Exception as e:
            if type(e).__name__ == "ContentKeyMissing":
                raise
        return artifact.get("content") or ""
    if ref and content is not None:
        # The BOOTSTRAP seam: a seed `FsContentStore` (Fernet ciphertext under the single
        # content.key) on a shard with no migrated cas/ cache yet. The one legacy read left;
        # it folds into the tier when the write path moves onto it.
        try:
            return _as_text(get_content(content, store_bundle.keys_dir, ref), ref, artifact)
        except (ContentKeyMissing, ContentStillSealed):
            raise
        except Exception:
            pass
    return artifact.get("content") or ""


def _as_text(blob, ref, artifact) -> str:
    """Bytes to text — but an unopened envelope is a custody fault, not a document.

    `ignore` is doing a second, legitimate job that this keeps: genuine content is not always
    clean UTF-8, and a stray byte must not fail a read. What changes is only the sealed case,
    which `content_crypto.is_encrypted` identifies by the MEC1 magic rather than by guessing at
    how printable the bytes look."""
    if isinstance(blob, (bytes, bytearray)):
        from mantle.services import content_crypto
        if content_crypto.is_encrypted(blob):
            raise ContentStillSealed(
                "content at %r (artifact %r) is still sealed: opening it needs an acting "
                "principal holding a read grant on its collection's origin root. Background work "
                "must declare one — `op.describe.*` is an OPERATOR, so INVOKE it as one rather "
                "than calling it as a bare function. Do NOT escalate to the system principal: "
                "that authorizes a user's content as the platform (see ember/custody.py)."
                % (ref, artifact.get("id")))
        return blob.decode("utf-8", "ignore")
    return blob if isinstance(blob, str) else ""


# One clean sentence of an article's own text — the `summary` the type template projects. NOT a
# hardcoded layout: the type's `offer_template` decides how title + summary read; this only supplies
# the summary value from the document itself, title-stripped so it is never repeated. Lives here (with
# `resolve_text`, content projection) so BOTH the ember viewer (`ember.browse`) and the retrieval
# tekton (sage's `content_search._describe`) read one implementation — grounding stays Apache-side.
_SENT_END = re.compile(r"[.!?](?:\s|$)")


def _summary(content: str, title: str, *, budget: Optional[int] = None) -> str:
    """The document's own text, title-stripped — NOT cut to a chosen length.

    So the text is returned WHOLE, ending at its first paragraph break when the document declares one
    (a unit the source itself supplies), and otherwise entire. `budget` survives only as a FLAGGED
    SEAM for a caller that must genuinely bound a runaway blob; it defaults to None — no bound, the
    honest default — and when supplied it ends on a sentence, never mid-word."""
    raw = content or ""
    t = (title or "").strip()
    # Paragraph structure BEFORE whitespace is collapsed — the document's own unit, if it has one.
    para = raw.split(chr(10)*2, 1)[0] if chr(10)*2 in raw else raw
    text = " ".join(para.split())
    if t and text.lower().startswith(t.lower()):
        text = text[len(t):].lstrip(" -—:.–")
    if not text:
        return ""
    if budget is None:
        return text                                      # WHOLE — the source decides its own length
    m = _SENT_END.search(text, 0, budget + 60)           # seam path: end on a real sentence if near
    if m and m.end() <= budget + 60:
        return text[:m.end()].strip()
    if len(text) > budget:
        return text[:budget].rsplit(" ", 1)[0].rstrip() + "…"
    return text
