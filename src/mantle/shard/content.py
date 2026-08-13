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


class ContentIntegrityError(RuntimeError):
    """The blob decrypted but does not hash to the ref it was fetched under.

    Fernet authenticates the CIPHERTEXT under the node's content key; it does not — and with no
    AAD parameter cannot — bind the blob to its content address. So a ciphertext moved between two
    refs in the store, or served for the wrong ref by a mirror, decrypts cleanly and returns the
    WRONG CONTENT under the right name. `cas/<sha256(plaintext)>` is the address, so re-hashing the
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


def put_content(content_store, keys_dir, data: bytes) -> Tuple[str, int]:
    """Encrypt + store content-addressed in the content store. Returns (content_ref, size).
    Idempotent: identical content is written once."""
    ref = content_ref(data)
    if not content_store.exists(ref):
        # The ONLY caller allowed to bootstrap: a write on a provisioned node may
        # legitimately mint the first key. Still refuses if the keys dir is absent.
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


def get_content(content_store, keys_dir, ref: str) -> bytes:
    """Fetch + decrypt content by ref, then VERIFY it against that ref.

    Fernet gives no AAD parameter, so the ciphertext cannot be cryptographically bound to its
    address the way `cell.py` and `content_cache.py` bind theirs. The content address is the
    binding available here, and re-hashing after decrypt is what enforces it — the check is a pure
    addition, it changes no stored byte, and it is the difference between "authentic" and
    "authentic AND the content this ref names"."""
    return _verify_ref(ref, _content_key(keys_dir).decrypt(content_store.get(ref)))


def resolve_text(store_bundle, artifact: dict) -> str:
    """The full text of an artifact — from the content store via `content_ref`, or inline
    `content` (legacy / small artifacts like WordNet defs). Never truncated."""
    ref = artifact.get("content_ref")
    content = store_bundle.content
    # ONE PATH on the lattice backend: the tiered read seam (mantle `TieredContentStore`) — local
    # FileContentCache first, then the S3/CDN mirror with sha256 verify-on-pull. The tier is built
    # unconditionally whenever the store's content is the lattice cache (local-only when
    # air-gapped), so there is no separate FileContentCache branch: the tier already tried the
    # local cache internally, and on ANY tier failure the only remaining fallback is inline
    # `content` (never a wrong key). `ContentKeyMissing` propagates — a node-wide configuration
    # fault must not read as an empty artifact.
    tier = getattr(store_bundle, "content_tier", None)
    if ref and tier is not None:
        try:
            return tier.get(ref, collection=artifact.get("collection_id")).decode("utf-8", "ignore")
        except Exception as e:
            if type(e).__name__ == "ContentKeyMissing":
                raise
        return artifact.get("content") or ""
    if ref and content is not None:
        # The BOOTSTRAP seam: a seed `FsContentStore` (Fernet ciphertext under the single
        # content.key) on a shard with no migrated cas/ cache yet. The one legacy read left;
        # it folds into the tier when the write path moves onto it.
        try:
            return get_content(content, store_bundle.keys_dir, ref).decode("utf-8", "ignore")
        except ContentKeyMissing:
            raise
        except Exception:
            pass
    return artifact.get("content") or ""


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
