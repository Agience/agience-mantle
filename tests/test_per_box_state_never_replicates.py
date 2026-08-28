"""Operational state describes one box and must never reach another.

The set's own first line is the rule — `_OP_EXCLUDE`: "operational state is per-box, never
replicated (would pollute the mesh)". What this file adds is that membership is checked rather than
remembered, because the failure mode is silent and arrives late.

Every miss in this set is a defect that waits: it does nothing at all until the mesh is switched
on, and then it is confidently wrong rather than missing — which is worse, because an operator acts
on it. The set records two that were found exactly that way:

  · `materialized-marker+json`, added 2026-08-25 — a marker written on node A makes node B skip
    indexing an artifact B has never indexed, leaving nothing to see but a search result that is not
    there.
  · `sensor+json`, added 2026-08-26 — this node's services, store, authority, certificate and code.
    Replicated, a reading from node A claims B's services are whatever A measured, B's disk is A's
    disk, and B's certificate expires when A's does.

The second one had been written down as a warning first: `agience-cloud/scripts/sensor_common.py`
says "this content type must not be minted on a node with MESH_ROLE set" — a rule stated in a
comment beside the code that would violate it, enforced by nothing. This file is what makes the
difference between a note and a rule.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from mantle.mesh.sync import _OP_EXCLUDE

#: Every content type a `sensor_*.py` may mint. Read from the source rather than restated, so a
#: sensor that changes its type cannot quietly leave the deny-list behind.
_SENSOR_COMMON = (pathlib.Path(__file__).resolve().parents[2]
                  / "agience-cloud" / "scripts" / "sensor_common.py")


def test_the_sensor_content_type_is_not_replicated() -> None:
    """A reading from another box is not a weaker answer, it is a wrong one."""
    assert "application/vnd.agience.sensor+json" in _OP_EXCLUDE, (
        "sensor readings would replicate: a reading written on one node would describe another "
        "node's services, disk, certificate and code as if they were its own")


@pytest.mark.skipif(not _SENSOR_COMMON.is_file(), reason="agience-cloud is not beside this repo")
def test_the_type_the_sensors_actually_mint_is_the_one_excluded() -> None:
    """The deny-list and the writer must name the same string, and nothing else connects them.

    They live in different repositories, so a rename on either side is invisible to the other — and
    the symptom of a mismatch is not an error anywhere, it is replication quietly resuming.
    """
    text = _SENSOR_COMMON.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'SENSOR_CONTENT_TYPE\s*=\s*"([^"]+)"', text)
    assert m, "sensor_common.py no longer declares SENSOR_CONTENT_TYPE"
    minted = m.group(1)
    assert minted in _OP_EXCLUDE, (
        "the sensors mint %r and `_OP_EXCLUDE` does not contain it — the two sides have drifted, "
        "and the only symptom will be per-box state replicating once the mesh is on" % minted)


def test_the_exclusion_is_what_the_replication_check_reads() -> None:
    """Membership in a set nobody consults is not an exclusion. This is the assertion that stops
    the two tests above from passing against a set that has been disconnected from the decision."""
    from mantle.mesh import sync

    src = pathlib.Path(sync.__file__).read_text(encoding="utf-8", errors="replace")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert re.search(r"if\s+ct\s+in\s+_OP_EXCLUDE\s*:", code), (
        "nothing tests membership of `_OP_EXCLUDE` any more; the set has become documentation")


@pytest.mark.parametrize("ct", sorted(_OP_EXCLUDE))
def test_every_excluded_type_is_a_plausible_content_type(ct: str) -> None:
    """A typo in this set excludes nothing and says nothing. Membership is an exact string match on
    a document's `content_type`, so `applicaton/...` or a stray space is a silent no-op — the entry
    looks present in every review and matches no artifact ever written."""
    assert ct == ct.strip(), "%r has surrounding whitespace and will never match" % ct
    assert re.match(r"^application/(vnd\.agience\.[a-z0-9-]+\+json|x-[a-z0-9-]+)$", ct), (
        "%r does not look like a content type this platform mints; an entry that matches nothing "
        "excludes nothing" % ct)
