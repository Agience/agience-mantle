"""The path law for a filesystem-backed index tree — escaping, sharding, atomic publication.

This module holds no posting store. :class:`~.sqlite_stores.SqlitePostingStore` is the posting
store, and that module records the four measured problems a one-file-per-slot layout produces. What
lives here is the part that is not about postings: how to turn a caller-supplied id into a safe path
component, how to fan out so no directory holds millions
of entries, and how to publish a blob without a reader ever seeing half of it.

It remains because :mod:`..file_cell_store` — the VECTOR arm's local store, a different arm with a
different access pattern — is a directory tree and needs exactly these rules. Two trees on one disk
disagreeing about how to escape an id is a bug that shows up as one owner's data in another owner's
directory, so the rules live in one place and are imported, not restated.

Two things a filesystem needs that an object store does not:

1. **Path components are escaped, not interpolated.** ``principal_id`` / ``collection_id`` /
   ``cluster_id`` are strings from the caller. Interpolating one containing ``/`` or ``..``
   straight into a path is a traversal out of the index root, so every component is escaped into
   ``[a-z0-9_-]`` with a reversible ``~hh`` escape. Lowercase-only is deliberate: on a
   case-insensitive filesystem (Windows, default macOS) two ids differing only in case would
   otherwise share one file, and one owner's data would silently overwrite another's.
2. **Two levels of fan-out** on ``sha256`` of the escaped name. One directory holding a file per
   item is a directory with millions of entries, which is pathological on every filesystem that
   matters — the same reason the content CAS shards.

Writes are atomic (`mkstemp` + :func:`os.replace`). A torn write would produce a short blob that
fails GCM authentication on read, which is indistinguishable from tampering; interruption must not
look like an attack.

Nothing here encrypts, decrypts, inspects or summarises a blob. A local disk is treated as just
another untrusted server, the same premise :class:`~mantle.db.content_cache.FileContentCache` is
built on.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from typing import List, Optional

logger = logging.getLogger(__name__)

#: Characters allowed to appear literally in a path component. Lowercase only — see the module
#: docstring on case-insensitive filesystems. ``.`` is excluded so ``..`` is unconstructible.
_SAFE = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_")

#: The same alphabet as a string, for :func:`encode_component`'s fast path. `str.strip` takes a
#: character set as a string and runs in C; kept beside `_SAFE` so the two cannot drift.
_SAFE_STR = "abcdefghijklmnopqrstuvwxyz0123456789-_"

#: Escape introducer. Not in :data:`_SAFE`, so it always escapes itself and can never be ambiguous.
_ESC = "~"

#: Windows device names, which stay reserved even with an extension (``nul.enc`` IS ``NUL``). An
#: escaped component is lowercase by construction, so this compares lowercase.
_RESERVED = (
    {"con", "prn", "aux", "nul"}
    | {"com%d" % i for i in range(1, 10)}
    | {"lpt%d" % i for i in range(1, 10)}
)


def encode_component(component: str) -> str:
    """One caller-supplied id → one safe path component. Reversible by :func:`decode_component`.

    Every character outside :data:`_SAFE` becomes ``~hh`` per UTF-8 byte. The empty string maps to
    a lone ``~``, which no escape can produce (escapes are always three characters).
    """
    if component == "":
        return _ESC
    # Fast path for the only two things this is called with. A blind token is 64 hex characters
    # and a principal id is a lowercase UUID; `-` and the hex digits are all in `_SAFE`, so the
    # loop below would run 64 iterations to hand back its own argument. `str.strip` with a
    # character set is one C-level pass and returns "" exactly when every character is
    # safe, which is the same test the loop performs one character at a time.
    #
    # Measured on 71/dev: a ten-term recall calls this 9,040 times (once per principal and once
    # per blind token, for 202 principals), and it was 0.121 s cumulative — comparable to the
    # 0.128 s of actual `open()` in the same query. The escape path below is unchanged and still
    # handles anything that is not already safe.
    if not component.strip(_SAFE_STR):
        encoded = component
    else:
        out: List[str] = []
        for ch in component:
            if ch in _SAFE:
                out.append(ch)
            else:
                out.extend("%s%02x" % (_ESC, b) for b in ch.encode("utf-8"))
        encoded = "".join(out)
    if encoded in _RESERVED:
        # Escape the first character; the name stops being a device name and still decodes.
        encoded = "%s%02x%s" % (_ESC, ord(encoded[0]), encoded[1:])
    return encoded


def decode_component(name: str) -> str:
    """Inverse of :func:`encode_component`. Raises ``ValueError`` on a malformed escape.

    An escape is exactly two hex digits. `int()` on the two-character slice accepts a
    one-character tail, so ``~7`` would decode to ``"\\x07"`` — a name :func:`encode_component`
    writes as ``~07`` and therefore never wrote. Both callers of this are enumerating a store to
    decide what to migrate or re-key
    (`mantle.system.manage_sse_index`, `FileCellStore.list_cells`) and both skip on `ValueError`, so
    accepting a name this module could not have produced meant reporting a principal id that does
    not exist rather than skipping a directory that is not ours. Tightening it is safe by
    construction: the encoder emits ``%02x``, so no well-formed name is affected.
    """
    if name == _ESC:
        return ""
    buf = bytearray()
    i = 0
    while i < len(name):
        ch = name[i]
        if ch == _ESC:
            hexits = name[i + 1:i + 3]
            if len(hexits) != 2:
                raise ValueError(
                    "truncated escape in path component %r — an escape is %s plus exactly two "
                    "hex digits" % (name, _ESC)
                )
            try:
                buf.append(int(hexits, 16))
            except ValueError:
                raise ValueError("malformed escape in path component %r" % name) from None
            i += 3
        else:
            buf.append(ord(ch))
            i += 1
    return buf.decode("utf-8")


def _shard(encoded: str) -> tuple[str, str]:
    """Two fan-out directories for one escaped leaf name.

    Derived from ``sha256`` rather than the name's own leading characters so the split stays
    balanced for ids that are not uniformly distributed (a blind token is; an artifact id is not).
    """
    digest = hashlib.sha256(encoded.encode("ascii")).hexdigest()
    return digest[:2], digest[2:4]


def _atomic_write(path: str, blob: bytes) -> None:
    """Publish ``blob`` at ``path`` atomically — a reader never observes a partial blob."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read(path: str) -> Optional[bytes]:
    """The blob at ``path``, or None if it is not there. Any other error is raised."""
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except FileNotFoundError:
        return None
    except IsADirectoryError:
        return None


def _unlink(path: str) -> None:
    """Remove ``path``. Absent is not an error — the Protocol says delete is a no-op if absent."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


__all__ = [
    "decode_component",
    "encode_component",
]
