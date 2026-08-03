"""Cell-key principal resolution — the master-key root for MANTLE cells.

A MANTLE cell's master key roots at the collection's **origin root** (the
immutable top of its creation chain), NOT at provenance (``created_by``) or the
mutable grant set. Index and query both resolve the principal the same way, so
they derive the same key for the same collection. There is no "owner" — access
is by grant, not ownership.

See :func:`db.lattice.get_origin_root` and
``.dev/features/anchors-and-anchorsets.md`` §5.
"""

from __future__ import annotations

import logging

from mantle.db import backend as db_store

logger = logging.getLogger(__name__)


class CellPrincipalUnresolved(RuntimeError):
    """The collection's origin root could not be walked, so its principal is unknown.

    Distinct from "the collection has no ancestors" — that case legitimately
    resolves to the collection itself. This means the LOOKUP FAILED and we do not
    know which principal owns the cells. See :func:`resolve_cell_principal`.
    """


def resolve_cell_principal(db, collection_id: str) -> str:
    """The encrypted-search principal for ``collection_id`` — its origin root.

    Stable and single-valued for the collection's whole sub-tree, so a cell
    encrypted at index time is decryptable at query time under the same key.
    Returns ``""`` for an empty input (callers skip empty principals).

    ⛔ THIS USED TO SWALLOW THE LOOKUP FAILURE AND RETURN ``collection_id``.
    That is not a fallback — it is a DIFFERENT, PLAUSIBLE-LOOKING PRINCIPAL, and
    therefore a different HKDF master key, for every collection that is not its
    own origin root. Nothing downstream can tell the substitute from the real
    answer: it is a well-formed id that flows straight into key derivation.

    The damage is that the substitution is TRANSIENT and therefore INCONSISTENT.
    Index time and query time each call this independently, so the lattice blip
    during indexing writes cells under `collection_id` while queries later seek
    them under the true origin root (or the reverse). The cells exist, the search
    finds nothing, and every metric reads healthy. This has already cost us one
    production defect — a deletion that removed nothing while reporting success
    (see ``pipeline_unified.delete_artifact_from_index``), which is exactly this
    bug: the wrong principal made both gates pass and the removal a no-op.

    Now it RAISES. A caller that cannot resolve the principal must fail or skip,
    never guess — guessing writes ciphertext under a key nobody will look for.
    """
    if not collection_id:
        return ""
    try:
        root = db_store.get_origin_root(db, collection_id)
    except Exception as exc:
        logger.error(
            "cell principal unresolved for collection %s: %s", collection_id, exc,
        )
        raise CellPrincipalUnresolved(
            f"could not resolve the origin root of collection {collection_id!r}; "
            f"refusing to substitute the collection id, which would derive a "
            f"different master key than the one its cells were written under: {exc}"
        ) from exc
    # No ancestors is a REAL answer: the collection is its own origin root.
    return root or collection_id
