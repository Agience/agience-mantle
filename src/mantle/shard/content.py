"""Content service — CONTENT lives in the content store, ENCRYPTED and content-addressed.

The proper content/context separation (the label-blind mesh model):
  • the artifact store (SQLite lattice) holds the CONTEXT + a REFERENCE (`content_ref` = the
    content's address). Queryable.
  • the content store (local CAS cache / S3 mirror; formerly Garage) holds the CONTENT as
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


_KEY_CACHE: dict = {}
_KEY_TTL = 60.0


def _content_key(keys_dir: Path, *, create: bool = False) -> MultiFernet:
    """The content/segment cipher. Returns a MultiFernet: `content.key` is primary (used for
    ENCRYPT), and any additional `content.key.*` files are decrypt-only fallbacks. This lets a
    node decrypt peers' segments that were written under a DIFFERENT key (the mesh keys diverged
    historically: t5/tu each have their own, 71+45 share one). Single-key nodes are unaffected —
    encrypt still uses their one key, and MultiFernet with one key behaves exactly like Fernet.

    ⛔⛔ THIS USED TO MINT A BRAND-NEW KEY WHENEVER `content.key` WAS ABSENT — ON THE READ PATH.
    It is the one entry point for `put_content`, `get_content` AND `sync._fernet`, and it could not
    tell "first-ever write, bootstrap me" from "the keys volume is not mounted yet". In the second
    case it fabricated a key, wrote it as the new PRIMARY, and returned it. Then:
      • the pending `decrypt` failed with `InvalidToken`, which `resolve_text` swallowed → `""`;
      • every SUBSEQUENT write encrypted under a key **no peer holds**;
      • published mesh segments became undecryptable fleet-wide, stalling every peer's cursor;
      • and row counts, `keyed_coverage` and ρ all stayed perfectly healthy throughout.
    The project's own `content-encryption` note names this exact failure — "a missing key is a
    SILENT partition, fingerprint keys first" — and the code still implemented the partition.
    It also `mkdir`'d the directory, which is precisely what made an unmounted volume look normal.

    Three changes, each aimed at one half of the confusion:

    1. `create` DEFAULTS TO FALSE. Only `put_content` opts in — a write on a provisioned node may
       legitimately bootstrap the first key. Reads never mint (a read only happens when content
       already exists, so a missing key there is always a fault), and `sync._fernet` never mints:
       publishing under a fresh key is the worst outcome available, because it corrupts the mesh
       for every peer rather than just this node.
    2. NO `mkdir`. An absent `keys_dir` means the volume is not mounted; that is a hard error, not
       an invitation to create one. Bootstrapping requires the directory to already exist, which is
       what distinguishes "provisioned but empty" from "not mounted".
    3. EXCLUSIVE CREATE (`"xb"`). The old check-then-write was racy: with a 64-worker pool two
       threads could both see no key, mint different ones, and the loser's ciphertext became
       permanently unreadable. On collision we re-read the winner's key instead of overwriting it.

    Cached per keys_dir for `_KEY_TTL` — this ran a stat, a read, a glob and N Fernet constructions
    on EVERY put/get, and the re-entry is also what made the old create-branch self-perpetuating.
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


def get_content(content_store, keys_dir, ref: str) -> bytes:
    """Fetch + decrypt content by ref."""
    return _content_key(keys_dir).decrypt(content_store.get(ref))


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
            # ⛔ DELIBERATELY NOT SWALLOWED. Everything else here is a per-artifact problem (blob
            # evicted, ref stale) where returning the inline fallback is right. A missing content
            # key is a NODE-WIDE configuration fault: swallowing it makes every artifact resolve to
            # "" and the whole corpus re-describe as empty, with no error at any layer. Note the
            # inline fallback below is GUARANTEED absent for exactly these artifacts — `content` is
            # popped once a `content_ref` is written — so "" here is never a real value.
            raise
        except Exception:
            pass
    return artifact.get("content") or ""


# One clean sentence of an article's own text — the `summary` the type template projects. NOT a
# hardcoded layout: the type's `offer_template` decides how title + summary read; this only supplies
# the summary value from the document itself, title-stripped so it is never repeated. Lives here (with
# `resolve_text`, content projection) so BOTH the ember viewer (`ember.browse`) and the moved retrieval
# tekton (sage's `content_search._describe`) read one implementation — grounding stays Apache-side.
_SENT_END = re.compile(r"[.!?](?:\s|$)")


def _summary(content: str, title: str, *, budget: Optional[int] = None) -> str:
    """The document's own text, title-stripped — NOT cut to a chosen length.

    ⛔ THIS USED TO TRUNCATE AT A CHOSEN 200 CHARACTERS and append "…", which silently amputated
    definitions mid-clause — e.g. *"the part of calculus that deals with the variation of a function
    with respect to changes in the independent variable (or variables) by means of the concepts of
    derivative and…"*. A character budget is an ARBITRARY CAP ([[no-arbitrary-caps]]): a bound must be
    DERIVED, and a definition's natural bound is THE DOCUMENT'S OWN STRUCTURE — it ends where its
    author ended it, not where a constant here decided.

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
