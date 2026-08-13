"""File-backed :class:`PostingStore` adapter — the standalone index.

The S3 adapter in :mod:`.s3_stores` is MANTLE-SSE's only other production storage. A standalone
install — "one SQLite file plus a filesystem CAS, opened in-process, nothing external to
provision" — needs somewhere to put an index without provisioning S3. This adapter gives that
install a durable place, on the same disk the lattice already lives on, against the same
Protocol. No Protocol method is added, removed, or re-typed.

Layout mirrors the S3 key layout one-for-one, so the two are a `cp -r` apart::

    {root}/{prefix}/{principal_id}/sse/posting/{aa}/{bb}/{blind_token}.enc
    {root}/{prefix}/{principal_id}/sse/manifests/{aa}/{bb}/{artifact_id}.enc

A third path used to sit beside them — ``{root}/{prefix}/{principal_id}/sse/stats.enc``, the
per-owner BM25 corpus aggregates. Recall computes no corpus statistic, so the store that wrote
that file is gone. An existing index tree keeps its `stats.enc` files; nothing opens them, and
nothing here removes them, because deleting an operator's data is an operator's decision.

This store sees ciphertext only. It is a dictionary over a filesystem. The bytes handed
to :meth:`put_posting` / :meth:`put_manifest` are already `nonce ‖ ciphertext ‖ tag`
from :func:`posting.pack_posting` / :func:`posting.pack_manifest`,
AES-256-GCM under a key this module never sees and cannot derive. Nothing here encrypts, decrypts,
inspects, or summarises a blob, and nothing here writes a readable side-car: a local disk is treated
as just another untrusted server, the same premise
:class:`~mantle.db.content_cache.FileContentCache` is built on.

Two things a filesystem needs that an object store does not:

1. **Path components are escaped, not interpolated.** ``principal_id`` / ``blind_token`` /
   ``artifact_id`` are strings from the caller. Interpolating one containing ``/`` or ``..``
   straight into a path is a traversal out of the index root, so every component is escaped into
   ``[a-z0-9_-]`` with a reversible ``~hh`` escape. Lowercase-only is deliberate: on a
   case-insensitive filesystem (Windows, default macOS) two ids differing only in case would
   otherwise share one file, and one owner's postings would silently overwrite another's.
2. **Two levels of fan-out** on ``sha256`` of the escaped name. One directory per owner holding a
   posting file per vocabulary term is a directory with millions of entries, which is pathological
   on every filesystem that matters — the same reason the content CAS shards.

Writes are atomic (`mkstemp` + :func:`os.replace`). A torn write would produce a short blob that
fails GCM authentication on read, which is indistinguishable from tampering; interruption must not
look like an attack.
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
    """Inverse of :func:`encode_component`. Raises ``ValueError`` on a malformed name."""
    if name == _ESC:
        return ""
    buf = bytearray()
    i = 0
    while i < len(name):
        ch = name[i]
        if ch == _ESC:
            try:
                buf.append(int(name[i + 1:i + 3], 16))
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


class _FileStoreBase:
    """Shared root/prefix handling. ``root`` is created eagerly so a bad path fails at wiring
    time, where the caller can still answer 503, rather than on the first index write."""

    def __init__(self, root: str, prefix: str) -> None:
        if not root:
            raise ValueError(
                "%s: root directory is required — an empty root would put the encrypted index in "
                "the process's current working directory, which is not a location anyone chose"
                % type(self).__name__
            )
        self._root = os.path.abspath(os.path.expanduser(str(root)))
        self._prefix = prefix.strip("/")
        os.makedirs(self._owner_base_dir(), exist_ok=True)

    @property
    def root(self) -> str:
        """The resolved absolute index root."""
        return self._root

    def _owner_base_dir(self) -> str:
        return os.path.join(self._root, *[p for p in self._prefix.split("/") if p])

    def _owner_dir(self, principal_id: str) -> str:
        return os.path.join(self._owner_base_dir(), encode_component(principal_id), "sse")


class FilePostingStore(_FileStoreBase):
    """:class:`~.posting.PostingStore` over a local directory tree.

    Args:
        root: Index root directory. Created if absent.
        prefix: Sub-tree under ``root``, mirroring the S3 key prefix so the per-segment
            (``committed`` / ``draft`` / ``archived``) index trees stay physically separate.
            Defaults to ``"mantle-sse"``.
    """

    def __init__(self, root: str, prefix: str = "mantle-sse") -> None:
        super().__init__(root, prefix)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _posting_dir(self, principal_id: str) -> str:
        return os.path.join(self._owner_dir(principal_id), "posting")

    def _posting_path(self, principal_id: str, blind_token: str) -> str:
        name = encode_component(blind_token)
        return os.path.join(self._posting_dir(principal_id), *_shard(name), name + ".enc")

    def _manifest_path(self, principal_id: str, artifact_id: str) -> str:
        name = encode_component(artifact_id)
        return os.path.join(
            self._owner_dir(principal_id), "manifests", *_shard(name), name + ".enc",
        )

    # ------------------------------------------------------------------
    # PostingStore Protocol — postings
    # ------------------------------------------------------------------

    def get_posting(self, principal_id: str, blind_token: str) -> Optional[bytes]:
        return _read(self._posting_path(principal_id, blind_token))

    def put_posting(self, principal_id: str, blind_token: str, blob: bytes) -> None:
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("FilePostingStore.put_posting expects bytes")
        _atomic_write(self._posting_path(principal_id, blind_token), bytes(blob))

    def delete_posting(self, principal_id: str, blind_token: str) -> None:
        _unlink(self._posting_path(principal_id, blind_token))

    def list_tokens_for_owner(self, principal_id: str) -> List[str]:
        base = self._posting_dir(principal_id)
        out: List[str] = []
        for dirpath, _dirnames, filenames in os.walk(base):
            for filename in filenames:
                if not filename.endswith(".enc"):
                    continue        # mkstemp leftovers etc. — not index objects
                try:
                    out.append(decode_component(filename[: -len(".enc")]))
                except ValueError:
                    logger.warning(
                        "FilePostingStore: skipping unreadable index filename %r under %s",
                        filename, dirpath,
                    )
        return out

    # ------------------------------------------------------------------
    # PostingStore Protocol — manifests
    # ------------------------------------------------------------------

    def get_manifest(self, principal_id: str, artifact_id: str) -> Optional[bytes]:
        return _read(self._manifest_path(principal_id, artifact_id))

    def put_manifest(self, principal_id: str, artifact_id: str, blob: bytes) -> None:
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("FilePostingStore.put_manifest expects bytes")
        _atomic_write(self._manifest_path(principal_id, artifact_id), bytes(blob))

    def delete_manifest(self, principal_id: str, artifact_id: str) -> None:
        _unlink(self._manifest_path(principal_id, artifact_id))


__all__ = [
    "FilePostingStore",
    "decode_component",
    "encode_component",
]
