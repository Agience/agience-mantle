"""An artifact's collection is one fact, recorded by one write.

## The two halves of that fact

`collection_id` on the document is what a listing filters on (`list_artifacts(collection_id=…)`).
The `contains` edge is what authorization walks: `LightConeResolver.resolve` seeds from a
principal's grants and expands through `list_origin_descendants`, a BFS over exactly those edges.

Writing them from different code paths lets them disagree, and each side stays self-consistent while
they do — an ingest counts its own members with `list_artifacts(collection_id=…)` — so nothing
reconciles until a recall asks a question that crosses between them. Measured on a live
2.94M-artifact store in that state: 2,933,349 documents carried the field and 155 carried the edge,
so a grant on `stage.0.lexicon` reached the collection artifact and none of its 1,841,335 members.

`_put_one` is the single point every artifact write passes through — `put_artifact` and `put_many`
both reach it — so the edge is written there, on the caller's cursor, inside the caller's savepoint.
An artifact and its membership commit or roll back together.

## What each test pins

    the edge is written           setting the field IS joining the collection
    is_origin is set              `list_origin_descendants` skips any edge without it, so its
                                  absence is invisible from the graph and fatal to the light cone
    atomic with the artifact      one savepoint, so there is no window with one and not the other
    the collection must exist     a collection IS an artifact; an edge to a vertex that is not
                                  here is a relation to nothing
    mesh consume places too       a replicated artifact is in its collection locally, or two nodes
                                  disagree about one principal's light cone
"""
from __future__ import annotations

import pytest

from mantle.db import open_lattice


@pytest.fixture()
def lattice(tmp_path):
    return open_lattice(str(tmp_path / "lattice.db"), origin="test")


def _collection(lattice, cid="col-1"):
    lattice.artifacts.put_artifact(
        {"id": cid, "content_type": "application/vnd.agience.collection+json"})
    return cid


def _contains(lattice, cid):
    return lattice.graph.edges_of(cid, label="contains", direction="out") or []


def test_setting_the_field_is_joining_the_collection(lattice):
    cid = _collection(lattice)
    lattice.artifacts.put_artifact({"id": "a-1", "collection_id": cid})

    edges = _contains(lattice, cid)
    assert [e.get("dst") for e in edges] == ["a-1"], (
        "the document names a collection and no containment edge records it; a grant on %r will "
        "reach the collection and none of its members" % cid)


def test_the_edge_carries_is_origin(lattice):
    """`list_origin_descendants` does `if not _eprop(e, "is_origin"): continue`. Without the flag
    the edge exists, `edges_of` returns it, and authorization expands through nothing — which is
    why its absence cannot be seen by looking at the graph."""
    from mantle.db.backend import list_origin_descendants

    cid = _collection(lattice)
    lattice.artifacts.put_artifact({"id": "a-1", "collection_id": cid})

    assert _contains(lattice, cid)[0].get("is_origin"), "the membership edge has no is_origin"

    class _Db:
        graph = lattice.graph
    assert list_origin_descendants(_Db(), [cid], "read") == {"a-1"}


def test_put_many_places_every_document(lattice):
    cid = _collection(lattice)
    n = lattice.artifacts.put_many(
        [{"id": "a-%d" % i, "collection_id": cid} for i in range(5)])

    assert n == 5
    assert sorted(e.get("dst") for e in _contains(lattice, cid)) == ["a-%d" % i for i in range(5)]


def test_the_membership_rolls_back_with_the_artifact(lattice):
    """One savepoint. A document that fails to write leaves no membership behind, because the two
    are one write rather than two that happen to run in order."""
    cid = _collection(lattice)
    with pytest.raises(Exception):
        lattice.artifacts.put_artifact({"collection_id": cid})      # no id: refused

    assert _contains(lattice, cid) == []


def test_a_collection_that_is_not_an_artifact_records_nothing(lattice):
    """A collection IS an artifact and is a vertex for that reason. Measured on the live store,
    774,915 artifacts name a parent that is not there — experiment outputs whose parent was never
    created — and an edge to a vertex that does not exist is a relation to nothing."""
    lattice.artifacts.put_artifact({"id": "a-1", "collection_id": "never-created"})
    assert _contains(lattice, "never-created") == []


def test_an_artifact_is_not_its_own_container(lattice):
    lattice.artifacts.put_artifact(
        {"id": "col-self", "content_type": "application/vnd.agience.collection+json",
         "collection_id": "col-self"})
    assert _contains(lattice, "col-self") == []


def test_a_document_with_no_collection_is_not_placed(lattice):
    """Not every artifact belongs to a collection, and inventing one would be a claim the document
    does not make."""
    lattice.artifacts.put_artifact({"id": "loose"})
    assert lattice.graph.edges_of("loose", label="contains", direction="in") == []


def test_the_root_travels_with_the_edge(lattice):
    """A membership is a statement about a lineage rather than about one version of it, which is
    what `add_artifact_to_collection` records on the API path too."""
    cid = _collection(lattice)
    lattice.artifacts.put_artifact({"id": "v2", "root_id": "v1", "collection_id": cid})

    import json
    props = _contains(lattice, cid)[0].get("props")
    props = json.loads(props) if isinstance(props, str) else (props or {})
    assert props.get("root_id") == "v1"


def test_a_consumed_artifact_is_placed_too(lattice):
    """`stamp_rev=False` is mesh consume. A replicated artifact has to be in its collection on this
    node as well, or two nodes disagree about what one principal's light cone contains."""
    cid = _collection(lattice)
    lattice.artifacts.put_many(
        [{"id": "remote-1", "collection_id": cid, "_origin": "peer", "_seq": 7}],
        stamp_rev=False)

    assert [e.get("dst") for e in _contains(lattice, cid)] == ["remote-1"]


def test_replacing_an_artifact_does_not_duplicate_its_membership(lattice):
    """Upsert on `edge_key`, so re-writing a document leaves exactly one membership."""
    cid = _collection(lattice)
    for _ in range(3):
        lattice.artifacts.put_artifact({"id": "a-1", "collection_id": cid})

    assert len(_contains(lattice, cid)) == 1
