"""An inbound grant is not applied while nothing on the sync path can verify it.

The peering daemon has exactly one sync mechanism — its own comment reads "the one path … no
second mechanism" — and it verifies nothing: `_apply_artifacts` filters inbound rows on `id` +
content-type and upserts them, and `_is_replicated` returns True for grants. A grant published by
anyone able to write to the plane becomes a grant in this lattice — write access to the directory
would equal full lattice authority.

Refusing at apply time, rather than dropping grants from `_is_replicated`: a type quietly removed
from the replicated set is indistinguishable from a type nobody publishes, and the day signing
lands nobody would know to put it back. Refusing here keeps the intent (grants replicate) and
states the precondition (when they can be verified), counts every refusal, and logs.

Safe to make strict: no node declares `MANTLE_RUN_MESH` or `MESH_ROLE` across either peer tier —
the mesh runs nowhere — so this closes the hole before it opens. The same change once peering is
live would be a behaviour change to a running system.

This is not the end of trust: the verifier is deliberately not wired here. `mesh/node.py::
sync_from` takes an `authority_pub` and raises on tamper, but it belongs to the shard/region
subsystem and is not an alternative wiring of this daemon. Until that work lands, "verify what it
applies" can only mean "refuse what it cannot verify".
"""
from __future__ import annotations

from mantle.mesh import sync


GRANT_CT = "application/vnd.agience.grant+json"
PLAIN_CT = "application/vnd.agience.artifact+json"


def _doc(i, ct):
    return {"id": "row-%s" % i, "content_type": ct, "_origin": "99", "_seq": 1000 + i}


# ── what counts as authority ─────────────────────────────────────────────────────────────────────

def test_a_grant_confers_authority():
    assert sync._confers_authority(GRANT_CT)
    assert sync._confers_authority("application/vnd.agience.grant+json; charset=utf-8")
    assert sync._confers_authority("APPLICATION/VND.AGIENCE.GRANT+JSON")


def test_an_ordinary_artifact_does_not():
    """The refusal must be NARROW. Withholding ordinary rows would stop replication working at
    all, which is the opposite of the ruling — it asked to make peering work."""
    for ct in (PLAIN_CT, "application/vnd.agience.collection+json", "text/markdown",
               "application/vnd.agience.operator+json", None, ""):
        assert not sync._confers_authority(ct), ct


# ── the split ────────────────────────────────────────────────────────────────────────────────────

def test_grants_are_withheld_and_everything_else_passes():
    batch = [_doc(0, PLAIN_CT), _doc(1, GRANT_CT), _doc(2, PLAIN_CT), _doc(3, GRANT_CT)]
    kept, refused = sync._withhold_unverified_authority(batch)
    assert refused == 2
    assert [d["id"] for d in kept] == ["row-0", "row-2"]


def test_the_refusal_is_counted_for_the_caller():
    """Counted, not silent. A refusal nobody can see is the same as an apply nobody checked."""
    stats: dict = {}
    sync._withhold_unverified_authority([_doc(1, GRANT_CT)], stats=stats)
    assert stats.get("refused_unverified_authority") == 1


def test_an_empty_batch_is_not_a_special_case():
    assert sync._withhold_unverified_authority([]) == ([], 0)


# ── the cursor guard, which is the part that could lose data ─────────────────────────────────────

def test_a_refused_row_counts_as_HANDLED():
    """The mesh's only data-loss guard runs on this number: `_apply_artifacts` returns what it
    handled, and its caller advances the consume cursor only when that equals the segment size — a
    segment recorded as applied moves `last_key` behind a monotone marker, so anything uncounted is
    unrecoverable. A refused row was examined and a decision was recorded, exactly like an
    LWW-declined one, so it must count. Returning only the written rows would stall every segment
    containing a grant, forever."""
    import inspect

    src = inspect.getsource(sync._apply_artifacts)
    assert "return refused_authority" in src, (
        "an all-grant segment returns 0 handled, so the caller raises 'partial apply' and the "
        "cursor never advances — the mesh stalls on the first segment carrying a grant")
    assert "n += refused_authority" in src, (
        "refused rows are not added to the handled count, so any segment containing a grant looks "
        "like a short write")


def test_the_short_write_guard_still_measures_the_WRITTEN_rows():
    """The subtle half: `n` now includes refusals, so the 'did the store write everything'
    comparison must subtract them again — otherwise a genuine partial write is masked by the
    refusals that happened to be in the same segment."""
    import inspect

    src = inspect.getsource(sync._apply_artifacts)
    assert "if n - refused_authority < len(batch):" in src, (
        "the partial-apply guard compares the refusal-inflated count against the batch, so a real "
        "short write inside a segment carrying grants would go unnoticed")


# ── the reason it is refused, not dropped ────────────────────────────────────────────────────────

def test_grants_are_still_declared_replicated():
    """The intent is preserved: grants remain in the replicated set; what changed is that
    applying one now has a precondition. Dropping them from `_is_replicated` would erase the intent
    along with the risk, and nothing would record that they are meant to travel."""
    assert sync._is_replicated(GRANT_CT), (
        "grants were removed from the replicated set instead of gated at apply time — the "
        "precondition is now invisible and nobody will know to restore replication when signing "
        "lands")
