"""Merkle anti-entropy — reconcile two stores by exchanging hashes, not data.

WHY THIS EXISTS. The cursor-based log feed converges only if every node reads every other node's
entire log: per-node work is O(N x fleet write rate), so total work is O(N^2). At 5 nodes that is
survivable; at 500 it is 10,000x the work to move the same corpus. Worse, a cursor is a single
fragile scalar per (peer, feed) — it silently skipped failed segments (permanent data loss), it
could not express "how far behind am I", and a node whose store was rebuilt had no way back except
replaying history from zero.

Merkle anti-entropy (Dynamo section 4.7; Cassandra repair; Riak) replaces all of that with one
idea: BOTH SIDES HASH THEIR DATA THE SAME WAY, COMPARE HASHES, AND MOVE ONLY WHAT DIFFERS.

  - Recovery from ANY divergence is the SAME operation as steady state. 8 GB behind or 8 bytes
    behind runs identical code. There is no "catch-up mode" to get wrong, and no reseed script.
  - Cost is proportional to the DIFFERENCE, not the corpus. Two converged nodes exchange ONE hash
    and stop. That is what makes 500 nodes affordable.
  - There is no cursor, so the entire cursor-skip data-loss class cannot occur. A failed transfer
    just leaves a hash mismatch that the next round retries. Self-healing is the default state.
  - It is order-independent and idempotent, so it composes with a CRDT-shaped store: applying the
    same artifact twice, or out of order, changes nothing.

DESIGN NOTES (why it is built this way)

Leaf assignment is by a HASH of the artifact id, not by the id itself. Ids are wildly non-uniform
(`wn-*` synsets, wiki titles, `op.*` operators), so range-splitting raw ids would pile most of the
corpus into a few leaves and defeat the point. blake2b of the id spreads them evenly regardless of
naming, so every leaf holds roughly corpus/leaves rows and a mismatch localizes tightly.

Leaf digests are XOR-of-row-hashes, which makes a leaf's value INDEPENDENT OF THE ORDER rows are
read in. That matters because different nodes scan their store in different orders and we must not
require a sort over millions of rows to compare — XOR is associative and commutative, so it can be
computed streaming, in any order, and incrementally maintained later.

A row hashes as (id, _rev): `_rev` changes on every insert and in-place update, so a mutation
(compaction, is->was decay) changes the leaf just as a new artifact does. Content is NOT hashed —
it is content-addressed in `cas/` already, so id+rev is a complete identity for the graph row.

MEASURED LIMIT — INCREMENTAL MAINTENANCE IS REQUIRED, NOT OPTIONAL (71, 2026-07-19)
A full-rescan publish is too slow to be the steady-state mechanism. On the live corpus while the
node was concurrently consuming, the scan ran at 6,286 rows/sec — ~7 minutes for 2.7M rows — and
the corpus itself grew 2.23M -> 2.73M during the same session. A tree built by full rescan is
therefore STALE BEFORE IT FINISHES on a node that is actively catching up, and publishing it would
advertise a root that never matched any real state.

The consequence is a division of labour, not a contest:
  - WHILE CATCHING UP (large, known divergence) the log feed does the bulk transfer. It streams and
    needs no consistent snapshot.
  - AT STEADY STATE (small, unknown divergence) Merkle is the right tool: two converged nodes
    exchange one 32 KB tree and stop, which the log feed can never do.

To make Merkle cheap enough to run continuously, leaf digests must be maintained INCREMENTALLY on
write — XOR out the row's old hash, XOR in the new one — which the XOR construction already
supports for free (that is a large part of why it was chosen over a sorted hash). Full rescan then
becomes a bootstrap/verification path, not the hot path. This is what Cassandra and Riak do, and
it is the next piece of work before wiring Merkle into the aggregator.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple


def natural_leaves(n: int) -> int:
    """Leaf count DERIVED from the corpus (not a chosen constant). Byte-compatible with
    `mantle.db.lattice.constants.natural_leaves`. A round costs `L + N/L` (publish the L-leaf tree +
    pull one differing leaf), minimised at L = sqrt(N); rounded to a power of two so different-sized
    nodes compare by XOR-folding the finer tree onto the coarser. k = round(log2(N)/2)."""
    n = max(1, int(n))
    return 1 << max(0, round(math.log2(n) / 2.0))


# A fallback / legacy value only, for a corpus size not yet known. Real count = natural_leaves(corpus).
DEFAULT_LEAVES = 4096


def _h64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def leaf_of(artifact_id: str, leaves: int = DEFAULT_LEAVES) -> int:
    """Which leaf an id belongs to. Hashed, never the raw id — see the module note on skew."""
    return _h64(artifact_id.encode("utf-8")) % leaves


def row_hash(artifact_id: str, rev: Any) -> int:
    """Identity of one row: id + _rev. A NULL/absent rev is hashed as 0 so pre-_rev rows still get a
    stable, comparable value instead of being silently skipped (that omission is exactly what made
    the legacy corpus unpublishable through the `_rev` feed)."""
    r = rev if isinstance(rev, int) else 0
    return _h64(artifact_id.encode("utf-8") + b"\x00" + str(r).encode("ascii"))


def build(rows: Iterable[Tuple[str, Any]], leaves: int = DEFAULT_LEAVES) -> List[int]:
    """Leaf digests from a stream of (id, _rev). Streaming and order-independent: XOR is
    associative and commutative, so callers may scan in any order, in pages, without sorting."""
    acc = [0] * leaves
    for aid, rev in rows:
        if not aid:
            continue
        acc[leaf_of(aid, leaves)] ^= row_hash(aid, rev)
    return acc


def root(leaf_digests: List[int]) -> int:
    """Single value summarizing the whole store. Two nodes with equal roots are converged and need
    exchange NOTHING further — this is the common case at steady state and the reason the scheme
    costs almost nothing when there is no work to do."""
    h = hashlib.blake2b(digest_size=8)
    for d in leaf_digests:
        h.update(d.to_bytes(8, "big"))
    return int.from_bytes(h.digest(), "big")


def fold(digests: List[int], target_len: int) -> Optional[List[int]]:
    """Collapse a 2^k-leaf digest array onto a 2^k' one (k' <= k), by XOR. EXACT, because `leaf_of`
    is `hash(id) mod 2^k`: a row in leaf j at resolution k lands in leaf `j mod 2^k'` at k', and the
    digest is an order-free XOR accumulator. Returns None if `target_len` is not a clean divisor
    (the two are not both powers of two nesting into each other)."""
    L = len(digests)
    t = int(target_len)
    if t <= 0 or L % t != 0:
        return None
    out = [0] * t
    for j, d in enumerate(digests):
        out[j % t] ^= d
    return out


def common(a: List[int], b: List[int]) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Fold both arrays to their common (smaller) resolution so they are comparable. Two nodes at
    different corpus sizes derived different leaf counts; the finer folds onto the coarser."""
    t = min(len(a), len(b))
    fa = a if len(a) == t else fold(a, t)
    fb = b if len(b) == t else fold(b, t)
    return fa, fb


def diff(mine: List[int], theirs: List[int]) -> List[int]:
    """Leaf indices that differ, at the two trees' COMMON resolution. This is the whole
    reconciliation decision: everything not listed here is provably identical on both sides and is
    never read, transferred, or considered again. When the two trees have different (power-of-two)
    leaf counts, both are XOR-folded onto the smaller before comparing — so a node never has to
    re-shard to talk to a peer, it just compares at the coarser of the two resolutions."""
    a, b = common(mine, theirs)
    if a is None or b is None:
        # Not power-of-two multiples of each other — a genuinely incompatible shape. Treat every
        # coarse leaf as differing rather than silently converging.
        n = min(len(mine), len(theirs))
        return list(range(max(len(mine), len(theirs)))) if n == 0 else list(range(n))
    return [i for i in range(len(a)) if a[i] != b[i]]


def summary(leaf_digests: List[int]) -> Dict[str, Any]:
    """Compact publishable form: the root plus the leaves, for a peer to compare against its own."""
    return {"leaves": len(leaf_digests), "root": root(leaf_digests),
            "digests": [d for d in leaf_digests]}


def load(doc: Dict[str, Any]) -> Optional[List[int]]:
    """Parse a published summary back to leaf digests; None if it is malformed or a different shape."""
    try:
        digs = [int(x) for x in doc["digests"]]
    except Exception:
        return None
    return digs if len(digs) == int(doc.get("leaves", len(digs))) else None
