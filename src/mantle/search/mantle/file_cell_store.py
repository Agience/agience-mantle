"""File-backed :class:`CellStore` adapter — the standalone vector index.

:mod:`.s3_cell_store` is the vector arm's only other production storage, and the
lexical arm already has this file's counterpart in :mod:`.sse.file_stores`. Without
one here, ``None`` for S3 is a working configuration for one arm and a dark arm for
the other — so the air-gap claim ("one SQLite file plus a filesystem CAS, opened
in-process, nothing external to provision") held for lexical recall and quietly did
not hold for semantic recall. This is the missing half.

Layout mirrors the S3 cell key, with the fan-out a filesystem needs::

    {root}/{prefix}/{principal_id}/{collection_id}/{aa}/{bb}/{cluster_id}.cell

One cell per ``(principal_id, collection_id, cluster_id)``, the cluster being the
routing anchor — one path, no flat cell, exactly as in S3.

**This store sees ciphertext only, and it is the same ciphertext.** The bytes handed to
:meth:`put` are already `nonce ‖ ciphertext ‖ tag` from :func:`cell.pack_cell`,
AES-256-GCM under a key derived by the oracle and bound by :func:`cell.cell_aad` to the
``collection:cluster`` slot they are being written to. Nothing here encrypts, decrypts,
inspects or summarises a blob, and nothing here writes a readable side-car — the
envelope is built above this layer and is byte-identical whichever backend holds it. A
local disk is treated as just another untrusted server, the same premise
:class:`~mantle.db.content_cache.FileContentCache` and :mod:`.sse.file_stores` are built
on.

The path law is not restated here. ``encode_component`` / ``decode_component`` /
``_shard`` / ``_atomic_write`` are imported from :mod:`.sse.file_stores`, because two
copies of an escaping rule is precisely how one tree's traversal guard and another's
drift apart while both look right. That import runs the other way from the usual
dependency (vector arm → lexical arm) and costs nothing: those helpers are stdlib-only,
so it cannot pull numpy into the lexical surface.

Two things a filesystem needs that an object store does not — see
:mod:`.sse.file_stores` for the full argument:

1. **Path components are escaped, not interpolated**, so a ``principal_id`` /
   ``collection_id`` / ``cluster_id`` containing ``/`` or ``..`` cannot climb out of the
   index root, and two ids differing only in case cannot share one file.
2. **Two levels of fan-out on the cluster leaf.** A collection holds one cell per
   routing anchor and the anchor set grows continuously with the manifold, so the
   cluster axis is the one that becomes a directory with too many entries. The owner
   and collection levels stay literal single directories — they are enumerable, which
   is what :meth:`list_cells` reads.

Writes are atomic (`mkstemp` + :func:`os.replace`). A torn write would produce a short
blob that fails GCM authentication on read, which is indistinguishable from tampering;
interruption must not look like an attack.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from .sse.file_stores import (
    _atomic_write,
    _read,
    _shard,
    _unlink,
    decode_component,
    encode_component,
)

logger = logging.getLogger(__name__)

#: Same suffix as the S3 layout, so the two trees stay readable as one thing.
_SUFFIX = ".cell"


class FileCellStore:
    """:class:`~.stores.CellStore` Protocol implementation over a local directory tree.

    Args:
        root: Index root directory. Created eagerly, so a bad path fails at wiring time
            where the caller can still answer 503, rather than on the first index write.
        prefix: Sub-tree under ``root``, mirroring the S3 key prefix so the per-segment
            (``committed`` / ``draft`` / ``archived``) index trees stay physically
            separate. Defaults to ``"mantle-cells"``.
    """

    def __init__(self, root: str, prefix: str = "mantle-cells") -> None:
        if not root:
            raise ValueError(
                "FileCellStore: root directory is required — an empty root would put the "
                "encrypted cells in the process's current working directory, which is not a "
                "location anyone chose"
            )
        self._root = os.path.abspath(os.path.expanduser(str(root)))
        self._prefix = prefix.strip("/")
        os.makedirs(self._base_dir(), exist_ok=True)

    @property
    def root(self) -> str:
        """The resolved absolute index root."""
        return self._root

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _base_dir(self) -> str:
        return os.path.join(self._root, *[p for p in self._prefix.split("/") if p])

    def _owner_dir(self, principal_id: str) -> str:
        return os.path.join(self._base_dir(), encode_component(principal_id))

    def _collection_dir(self, principal_id: str, collection_id: str) -> str:
        return os.path.join(self._owner_dir(principal_id), encode_component(collection_id))

    def _cell_path(self, principal_id: str, collection_id: str, cluster_id: str) -> str:
        name = encode_component(cluster_id)
        return os.path.join(
            self._collection_dir(principal_id, collection_id),
            *_shard(name),
            name + _SUFFIX,
        )

    # ------------------------------------------------------------------
    # CellStore Protocol
    # ------------------------------------------------------------------

    def get(
        self, principal_id: str, collection_id: str, cluster_id: str = "",
    ) -> Optional[bytes]:
        return _read(self._cell_path(principal_id, collection_id, cluster_id))

    def put(
        self, principal_id: str, collection_id: str, blob: bytes, cluster_id: str = "",
    ) -> None:
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("FileCellStore.put expects bytes")
        _atomic_write(
            self._cell_path(principal_id, collection_id, cluster_id), bytes(blob),
        )

    def delete(
        self, principal_id: str, collection_id: str, cluster_id: str = "",
    ) -> None:
        _unlink(self._cell_path(principal_id, collection_id, cluster_id))

    def list_cells(self, principal_id: str) -> List[str]:
        """Distinct ``collection_id`` strings under ``principal_id``.

        Read from the directory names, which is why the collection level is not
        sharded: the alternative would be walking every cell of every collection to
        recover a set the tree already spells out.
        """
        base = self._owner_dir(principal_id)
        out: List[str] = []
        try:
            entries = os.listdir(base)
        except (FileNotFoundError, NotADirectoryError):
            return out
        for entry in entries:
            if not os.path.isdir(os.path.join(base, entry)):
                continue
            try:
                out.append(decode_component(entry))
            except ValueError:
                logger.warning(
                    "FileCellStore: skipping unreadable collection directory %r under %s",
                    entry, base,
                )
        return out

    def list_clusters(self, principal_id: str, collection_id: str) -> List[str]:
        """The ``cluster_id`` strings stored for one context.

        Walks the fan-out, so removal and admin paths can reach every cell of a
        collection the same way they can in S3.
        """
        base = self._collection_dir(principal_id, collection_id)
        out: List[str] = []
        for dirpath, _dirnames, filenames in os.walk(base):
            for filename in filenames:
                if not filename.endswith(_SUFFIX):
                    continue        # mkstemp leftovers etc. — not index objects
                try:
                    out.append(decode_component(filename[: -len(_SUFFIX)]))
                except ValueError:
                    logger.warning(
                        "FileCellStore: skipping unreadable cell filename %r under %s",
                        filename, dirpath,
                    )
        return out


__all__ = ["FileCellStore"]
