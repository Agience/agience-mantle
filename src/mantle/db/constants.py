"""Verbatim constants — LATTICE-CONTRACT.md §3. PORT, DO NOT RE-DERIVE.

The functions are byte-compatible with `agience-mantle/src/mantle/mesh/merkle.py`
(which needs zero changes); they are restated here rather than imported so that
`mantle.db` has no cross-repo dependency and stands alone.
"""
from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Union


def natural_leaves(n: int) -> int:
    """The number of Merkle leaves DERIVED from the corpus — not a chosen constant.

    A reconcile round costs, in the worst case of one changed row: `L` to publish the L-leaf tree
    plus `N/L` to pull the one differing leaf's rows. That sum `L + N/L` is minimised at **L = sqrt(N)**
    — the point where publishing the tree and pulling a leaf cost exactly the same. So the leaf count
    IS the square root of the corpus: it grows with the store (a fixed count degrades as the corpus
    grows) and there is no magic number to justify.

    Rounded to a power of two on purpose: `leaf_of` is `hash(id) mod 2^k`, so the low k bits nest, and
    two nodes whose corpus sizes gave them different leaf counts compare by XOR-FOLDING the finer tree
    onto the coarser (a leaf at 2^k maps to `leaf mod 2^k'` at any k' < k). k = round(log2(N)/2)."""
    n = max(1, int(n))
    k = max(0, round(math.log2(n) / 2.0))
    return 1 << k


# A fallback value only — the real leaf count is `natural_leaves(corpus)`, resolved
# per-store and re-derived as the corpus grows. This is used where a corpus size is not yet known,
# and it is the resolution used by stores that predate per-store derivation. A power of two, so
# folding still applies.
DEFAULT_LEAVES = 4096

CAS_PREFIX = "cas/"

COLLECTION_CONTENT_TYPE = "application/vnd.agience.collection+json"

#: How many rows a typed content-type fetch will pull before it stops claiming to be exhaustive.
CT_FETCH_CAP = 10_000

CRUDEASIO = ("create", "read", "update", "delete",
             "evict", "invoke", "add", "share", "admin")

# ── artifact lineage state ───────────────────────────────────────────────────
#
# THE ONE ANSWER TO "WHAT STATE IS A DOC WITH NO `state` FIELD IN".
#
# `state` partitions the search index into three separately-keyed encrypted trees, one per state,
# so this is not a stylistic default: two readers that answer it differently move an artifact
# between encrypted trees on a write that changed nothing. Every reader derives from here.
#
# ABSENCE IS COMMITTED. Three facts fix the value:
#
#   * `vertex.revise()` — the store's own versioning primitive — writes `state = "committed"`
#     unconditionally, so the raw write path's output is committed.
#   * Both head-only SQL predicates are `json_extract(doc,'$.state') IS NOT 'archived'`, which is
#     null-safe on purpose: a stateless row is LIVE, not filtered out.
#   * `draft` is an affirmative claim — "an unpublished edit exists". Absence of the field cannot
#     make that claim. (`Artifact()` still constructs a draft; creating an artifact and reading a
#     stored doc are different questions.)
#
# It lives at the store layer because the fact is about a STORED DOC, and because this is the
# layer both sides can reach: `db/vertex.py` stands alone and cannot import `entities`.
STATE_WHEN_ABSENT = "committed"


def state_of(doc: Dict[str, Any]) -> str:
    """The lineage state of a doc, with absence resolved to `STATE_WHEN_ABSENT`.

    Falsy, not `is None` — an empty string is absence spelled differently, and it is not one of
    the three states, so resolving the two the same way is what keeps a doc from landing in no
    segment at all.

    NOT for the `state:*` counters or the `state=` SQL filter: those partition on the value a row
    RECORDS, over a mixed plane (people, grants, commits, markers all live in `vertex`), and are
    documented at their call sites.
    """
    return str(doc.get("state") or STATE_WHEN_ABSENT)

# ── edge forces (contract §2, `edge.force`) ──────────────────────────────────
EDGE_FORCES = ("grant", "temporal", "semantic", "lifecycle", "derivation")


class NullAuditField(str, Enum):
    """The CLOSED SET of fields the provenance null-audit may count.

    A closed set, not a free string: two independent derivations of the same value are
    only guaranteed to agree if there is exactly one place either can be defined, and a
    free string would let a new field silently be miscounted as `provenance`.

    Inherits `str` deliberately: `NullAuditField.PROVENANCE == "provenance"` is
    True, so callers holding a plain string from `genesis._NULL_AUDIT_FIELDS`
    keep working while anything outside the set is rejected."""

    CITED_FROM = "cited_from"
    PROVENANCE = "provenance"

    @classmethod
    def coerce(cls, field: Union[str, "NullAuditField"]) -> "NullAuditField":
        # A member is returned as-is. NOT via `cls(str(field))`: on Python 3.11+
        # `str()` of a str-mixin enum member yields "NullAuditField.PROVENANCE",
        # not "provenance", so that spelling rejects this enum's OWN members.
        # (`StrEnum` fixes this but is 3.11+, and this package targets older.)
        if isinstance(field, cls):
            return field
        try:
            return cls(field)
        except ValueError:
            raise ValueError(
                "%r is not an audited provenance field. The audited set is %r. "
                "Counting an arbitrary field would require either a scan or a new "
                "incremental counter — add it here AND to the counter maintenance "
                "in vertex._counter_deltas, so the two cannot disagree."
                % (field, [f.value for f in cls])) from None


NULL_AUDIT_FIELDS = tuple(f.value for f in NullAuditField)


def is_missing(value: Any) -> bool:
    """Whether a doc field counts as MISSING for the null audit.

    FALSY, not `is None` — matching `genesis._scan_missing_field`'s fallback walk
    (`if not d.get(field)`). An empty string or empty list is an absent
    provenance, and the typed count and the fallback count MUST agree or the audit
    silently changes answer depending on which backend serves it."""
    return not value


def _h64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def leaf_of(artifact_id: str, leaves: int = DEFAULT_LEAVES) -> int:
    """Which merkle leaf an id belongs to. HASHED, never the raw id.

    Ids are wildly non-uniform (`wn-*`, wiki titles, `op.*`), so range-splitting
    raw ids piles most of the corpus into a few leaves and defeats the point."""
    return _h64(artifact_id.encode("utf-8")) % leaves


def row_hash(artifact_id: str, seq: Any) -> int:
    """Identity of one row: `blake2b(id || \\0 || str(_seq), digest_size=8)`.

    A NULL/absent `_seq` hashes as 0 so unversioned rows still get a stable,
    comparable value instead of being silently skipped."""
    s = seq if isinstance(seq, int) else 0
    return _h64(artifact_id.encode("utf-8") + b"\x00" + str(s).encode("ascii"))


def edge_key(src: str, dst: str, label: str) -> bytes:
    """`blake2b(src || \\0 || dst || \\0 || label, digest_size=16)`.
    byte that cannot appear in a UTF-8 id."""
    return hashlib.blake2b(
        src.encode("utf-8") + b"\x00" + dst.encode("utf-8") + b"\x00" + label.encode("utf-8"),
        digest_size=16,
    ).digest()


def edge_hash(key: bytes, content: Dict[str, Any]) -> int:
    """The Merkle contribution of ONE edge — NODE-INVARIANT, and that is the whole point.

    A vertex is versioned, so its `row_hash` keys on `(id, _seq)`: a new version changes the leaf and
    replicates. An edge is an IDENTITY-FACT — it exists or it does not — and its `(_origin, _seq)`
    differs per node (the author allocates a real seq; a consumer stamps the reserved
    `_local:edge:<digest>` origin with seq 1). Keying an edge's Merkle hash on `_seq`, as vertices are
    keyed, would make the SAME edge hash differently on two nodes, so their edge leaves could NEVER
    converge. So an edge is hashed on its `edge_key` plus its SHIPPED CONTENT (label + the promoted
    grant/order columns + the blob props) and NOTHING per-node — the exact bytes two nodes agree on
    when they both hold the edge. `_origin`/`_seq` are excluded because they are the per-node fields.

    `content` is canonicalised (sorted keys, tight separators) so serialization order cannot make two
    equal edges hash differently. A NULL/absent value is dropped rather than encoded, so an edge with
    an explicit `force=None` hashes the same as one that never set it — both are 'no force'."""
    clean = {k: v for k, v in (content or {}).items()
             if v is not None and k not in ("_origin", "_seq", "src", "dst", "label")}
    canon = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return _h64(key + b"\x00" + canon.encode("utf-8"))


# ── version comparison results ───────────────────────────────────────────────
# RESOLVED-3 (contract §4): there is NO shared clock and none will be introduced.
# `(_origin, _seq)` is a PARTIAL order. Two events from different observers are
# genuinely unordered; synthesizing a tiebreak invents information. The API must
# therefore be able to SAY "unordered" — that is the honest answer, not a failure.
NEWER = "newer"
OLDER = "older"
SAME = "same"
UNORDERED = "unordered"

VERSION_ORDERS = (NEWER, OLDER, SAME, UNORDERED)


def compare_version(a_origin: Any, a_seq: Any, b_origin: Any, b_seq: Any) -> str:
    """Compare version identity `(_origin, _seq)` of A against B.

    Returns one of `NEWER` / `OLDER` / `SAME` / `UNORDERED`.

    Single-writer-per-vertex means a higher `_seq` from the SAME `_origin` wins.
    **Different origins are simply UNORDERED.** When cross-observer ordering is
    genuinely needed it is already in the graph as a provenance edge — read it
    off the structure (`is there a path X ⇝ Y?`), never off a clock.

    An unversioned row (`_origin` or `_seq` NULL) is UNORDERED against
    everything except an identically-unversioned row."""
    if not isinstance(a_seq, int) or not isinstance(b_seq, int):
        return SAME if (a_origin == b_origin and a_seq == b_seq) else UNORDERED
    if a_origin != b_origin:
        return UNORDERED
    if a_seq > b_seq:
        return NEWER
    if a_seq < b_seq:
        return OLDER
    return SAME
