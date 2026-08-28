"""AnchorSet — the seeded coordinate system, and what routes against it.

The set arrives whole, from a client, and Mantle stores it and compares vectors to it. Nothing
here derives, grows, reconciles or maps a coordinate system, and the last two tests in this file
are the guard that keeps it that way.
"""

import numpy as np
import pytest

from mantle.search.anchors import AnchorSet
from mantle.search.anchors import store as _store
from mantle.search.anchors.anchorset import l2norm
from mantle.search.anchors.repo import InMemoryAnchorRepo
from .helpers import make_oracle, req, self_request

D = 32


@pytest.fixture
def anchor_repo():
    """Inject an in-memory AnchorRepo as the live store (anchors are artifacts;
    this keeps the geometry tests db-free), restoring the default repo after."""
    repo = InMemoryAnchorRepo()
    _store.set_anchor_repo(repo)
    try:
        yield repo
    finally:
        _store.set_anchor_repo(None)


def _clusters(rng, n_clusters=4, per=40, d=D, spread=0.12):
    centers = l2norm(rng.standard_normal((n_clusters, d)).astype(np.float32))
    items, truth = [], []
    for c in range(n_clusters):
        pts = l2norm(centers[c] + spread * rng.standard_normal((per, d)).astype(np.float32))
        for j in range(per):
            items.append((f"c{c}_{j}", pts[j]))
            truth.append(c)
    return items, truth


def _provision(items, k, model_id="hf:test@1.0", d=D):
    """A provisioned AnchorSet: the first real item of each of the first ``k`` clusters, admitted
    explicitly. No fitting — anchors are not derived here or anywhere else, because locally-derived
    anchors would mint region ids no peer computes. In production the canonical set arrives as an
    artifact; in this test the test is that outside authority, which is exactly the point: someone
    else decides, and every node admits the same set. Anchors stay real items (fully-disclosed
    artifacts, §3), which is what the routing assertions below actually depend on."""
    aset = AnchorSet(model_id=model_id, dim=d)
    seen = []
    for lab, vec in items:
        c = lab.split("_")[0]
        if c not in seen and len(seen) < k:
            seen.append(c)
            aset.add_text(lab, vec)
    return aset


def test_anchors_are_real_items():
    rng = np.random.default_rng(7)
    items, _ = _clusters(rng)
    aset = _provision(items, 4)
    assert len(aset) == 4
    labelset = {lab for lab, _ in items}
    # anchors are real items (fully-disclosed artifacts), not synthetic centers
    assert all(a.label in labelset for a in aset.anchors)


def test_store_roundtrip(anchor_repo):
    from mantle.search.anchors import store

    rng = np.random.default_rng(2)
    items, _ = _clusters(rng, n_clusters=4, per=20)
    aset = _provision(items, 4, model_id="hf:bge-m3@1.0")
    store.save_live_anchorset(aset)             # persists each anchor via the repo

    loaded = store.get_live_anchorset()
    assert loaded is not None
    assert len(loaded) == len(aset)
    assert loaded.model_id == aset.model_id
    assert loaded.dim == aset.dim
    # The round-trip has to preserve the geometry, not merely the count: the ids ARE the cluster
    # ids, so a set that came back with the same anchors in a different arrangement would still
    # route every vector somewhere else.
    for lab, vec in items[:8]:
        before = aset.nearest(vec, k=4)
        after = loaded.nearest(vec, k=4)
        assert [a.anchor_id for a, _ in before] == [a.anchor_id for a, _ in after], lab
        assert all(abs(s - t) < 1e-6 for (_, s), (_, t) in zip(before, after)), lab


def test_require_live_anchorset_raises_when_not_provisioned(anchor_repo):
    """One path, and it cannot manufacture its own start.
    """
    from mantle.search.anchors import AnchorSetNotProvisioned
    from mantle.search.anchors import store

    assert store.get_live_anchorset() is None          # nothing provisioned yet
    with pytest.raises(AnchorSetNotProvisioned):
        store.require_live_anchorset()


def test_require_live_anchorset_returns_the_provisioned_set(anchor_repo):
    """And when the canonical set HAS been provisioned, it is returned — no derivation involved."""
    from mantle.search.anchors import store

    canonical = AnchorSet("hf:test@1.0", 8)
    canonical.add_text("seed-anchor", l2norm(np.eye(8)[0]))
    store.save_live_anchorset(canonical)

    got = store.require_live_anchorset()
    assert got is not None and len(got) == 1
    assert [a.label for a in got.anchors] == ["seed-anchor"]


def test_model_id_is_commons_format():
    from mantle.search.embeddings import model_id

    mid = model_id()
    assert mid.count("@") == 1 and ":" in mid  # <ns>:<path>@<ver>


# --------------------------------------------------------------------------
# P2 — anchor routing + cluster-aware cells
# --------------------------------------------------------------------------

def test_route_vector_and_query():
    from mantle.search.anchors import routing

    rng = np.random.default_rng(5)
    items, _ = _clusters(rng, n_clusters=4, per=25)
    aset = _provision(items, 4)

    cid = routing.route_vector(aset, items[0][1])
    assert cid in {a.anchor_id for a in aset.anchors}

    cands = routing.route_query(aset, items[0][1], nprobe=3)
    assert len(cands) == 3
    assert cands[0] == cid          # the index cell is the top query candidate
    assert cands[0] != cands[1]

    # One path: a vector that can't be placed (wrong dimension) is an error,
    # not a flat fallback.
    import pytest
    with pytest.raises(ValueError):
        routing.route_vector(aset, np.ones(D + 1, dtype=np.float32))
    with pytest.raises(ValueError):
        routing.route_query(aset, np.ones(D + 1, dtype=np.float32))


def test_oracle_cluster_keying_separates_anchors():
    from cryptography.fernet import Fernet

    from mantle.search.mantle.oracle import FernetMasterKeyStore, OracleService

    svc = make_oracle()
    # One formula: info = collection ‖ 0x00 ‖ cluster. Each routing anchor gets
    # an independent key; re-derivation is deterministic. There is no flat key.
    anchor_x = svc.derive_cell_key("owner1", "colA", "anchorX", self_request("owner1", "update"))
    anchor_x2 = svc.derive_cell_key("owner1", "colA", "anchorX", self_request("owner1", "update"))
    anchor_y = svc.derive_cell_key("owner1", "colA", "anchorY", self_request("owner1", "update"))

    assert anchor_x == anchor_x2                 # deterministic per (owner, col, anchor)
    assert anchor_x != anchor_y                  # per-anchor cells are independent
    assert len({len(k) for k in (anchor_x, anchor_y)}) == 1 and len(anchor_x) == 32


def test_cell_aad_binding():
    from mantle.search.mantle.cell import cell_aad

    # One formula: AAD = "collection:cluster" (canonical plan §5.1).
    assert cell_aad("colA", "anchorX") == "colA:anchorX"
    assert cell_aad("colA", "anchorY") == "colA:anchorY"


def test_cell_store_is_cluster_aware():
    from mantle.search.mantle.stores import InMemoryCellStore

    s = InMemoryCellStore()
    s.put("o", "c", b"a1", "anchorX")
    s.put("o", "c", b"a2", "anchorY")
    s.put("o", "c2", b"a3", "anchorX")

    assert s.get("o", "c", "anchorX") == b"a1"
    assert s.get("o", "c", "anchorY") == b"a2"
    assert s.get("o", "c", "missing") is None

    assert set(s.list_cells("o")) == {"c", "c2"}                  # distinct collections
    assert set(s.list_clusters("o", "c")) == {"anchorX", "anchorY"}

    s.delete("o", "c", "anchorX")
    assert s.get("o", "c", "anchorX") is None
    assert s.get("o", "c", "anchorY") == b"a2"   # deleting one cluster leaves others


def test_anchor_routing_end_to_end(anchor_repo):
    """P2.2: chunks land in per-anchor cells; a query routes to the right cell."""
    from cryptography.fernet import Fernet

    from mantle.search.anchors import store
    from mantle.search.mantle.engine import MantleQueryEngine
    from mantle.search.mantle.indexer import MantleIndexer
    from mantle.search.mantle.oracle import FernetMasterKeyStore, OracleService
    from mantle.search.mantle.stores import InMemoryCellStore

    d = 8
    e0, e1, e2 = (l2norm(np.eye(d)[i]) for i in range(3))

    def near(base, i):
        v = base.copy()
        v[3] += 0.01 * i
        return l2norm(v).tolist()

    aset = AnchorSet("hf:test@1.0", d)
    a0 = aset.add_text("anchor0", e0)
    a1 = aset.add_text("anchor1", e1)
    a2 = aset.add_text("anchor2", e2)
    store.save_live_anchorset(aset)             # get_live_anchorset() now returns this

    oracle = make_oracle()
    cells = InMemoryCellStore()
    indexer = MantleIndexer(oracle, cells)
    principal, coll = "principal1", "coll1"

    chunks = [
        {"artifact_id": "art1", "chunk_id": 0, "embedding": near(e0, 1)},
        {"artifact_id": "art1", "chunk_id": 1, "embedding": near(e0, 2)},
        {"artifact_id": "art2", "chunk_id": 0, "embedding": near(e1, 1)},
    ]
    touched = indexer.index_artifact(principal, coll, chunks, self_request(principal, "update"))
    assert touched == 2                          # two anchor cells touched (a0, a1)

    clusters = set(cells.list_clusters(principal, coll))
    assert a0.anchor_id in clusters and a1.anchor_id in clusters
    assert "" not in clusters                    # nothing flat — all routed
    assert a2.anchor_id not in clusters
    assert {c["chunk_id"] for c in indexer.chunks_in_cell(principal, coll, a0.anchor_id, self_request(principal, "update"))} == {0, 1}

    # Query near anchor0 with nprobe=1 → only a0's cell is searched.
    engine = MantleQueryEngine(oracle, cells, nprobe=1)
    ids = {h.artifact_id for h in engine.search(near(e0, 1), [(principal, coll)], req(), top_k=10)}
    assert "art1" in ids and "art2" not in ids   # routing limited the search to the right cell

    # Removal strips from every cluster.
    assert indexer.remove_artifact(principal, coll, "art1", self_request(principal, "update")) == 1
    assert indexer.chunks_in_cell(principal, coll, a0.anchor_id, self_request(principal, "update")) == []


# ── the anchor set is seeded, and that is enforced ──────────────────────────────────────────────
# These guard against a lifecycle being reinstated: a clusterer added back on a fresh deployment
# would let region ids diverge between nodes that fit different anchors from different data, and a
# projection between spaces would let a node answer in a basis its cells were never written in.
def test_no_module_owns_the_coordinate_system_s_lifecycle():
    """Not "raises" — gone. A function that exists only to raise is an invitation with a docstring.

    A client seeds the set; Mantle loads it and routes against it. Every name below is a way of
    deriving, growing or reconciling one, and none of them resolves.

    `crosswalk` / `crosswalk_artifact` are not on this list: they are in the package because
    `ember/embed.py`'s `Aligner` imports them, and 27 ember tests plus 2 whole modules need them.

    The risk this list is written against is real, and for those two it is accepted rather than
    absent:

        a projection between spaces lets a node answer in a basis its cells were never
        written in.

    What still holds the line, and what a reviewer should check instead of this test:
      * `Crosswalk.is_isometry` MEASURES ‖MᵀM − I‖ rather than trusting `method`, so a stored
        matrix that is not a gauge is detectable at the point of use.
      * `FitResidual` defines no `__float__` and no ordering, so an in-sample residual cannot be
        read as a fidelity guarantee — that exact misreading is why `error_bound` was retired.
      * `null_residual` computes the shared-nothing null (E[cos]=0, sd=1/√D), so
        `z_below_null == 0` says the walk carries no information. On the one real measurement in
        the tree (6 anchors, 16→64 dim) the held-out residual was 1.026 against a null of 1.000 —
        i.e. AT the null. A caller that ignores that number is the failure mode, not the module.

    The clustering half of the original guard is untouched and still enforced by
    `test_no_clustering_remains_in_the_anchor_layer` below: anchors are still seeded, never fitted.
    """
    import importlib

    import mantle.search.anchors as anchors_pkg

    assert not hasattr(anchors_pkg, "bootstrap_anchorset")
    for gone in ("bootstrap", "seed_corpus", "grow", "reconciler", "density", "activate"):
        with pytest.raises(ImportError):
            importlib.import_module(f"mantle.search.anchors.{gone}")
        assert not hasattr(anchors_pkg, gone), f"anchors.{gone} is re-exported"


def test_the_crosswalk_cannot_pass_itself_off_as_a_gauge():
    """The guard that replaces the import ban: a projection must PROVE it is an isometry.

    `crosswalk` is admitted to the package (see above), so the question is whether it can
    misreport what it is. These are the three properties the accepted risk
    rests on; if any of them regresses, the ban was load-bearing after all.
    """
    import numpy as np

    from mantle.search.anchors.crosswalk import (
        LINEAR, PROCRUSTES, Crosswalk, FitResidual, null_residual,
    )

    # 1. A non-orthogonal matrix is not an isometry, whatever `method` claims.
    liar = Crosswalk(
        source_space_id="a", target_space_id="b", method=PROCRUSTES,
        matrix=np.diag([2.0, 1.0]), dim_in=2, dim_out=2,
        residual=FitResidual(in_sample=0.0, held_out=None, n_pairs=0, n_held_out=0,
                             dim_in=2, dim_out=2, closed_form=True),
    )
    assert liar.is_isometry is False, (
        "a matrix that scales an axis by 2 was accepted as an isometry — `is_isometry` has "
        "stopped measuring and started trusting `method`.")
    honest = Crosswalk(
        source_space_id="a", target_space_id="b", method=PROCRUSTES,
        matrix=np.eye(2), dim_in=2, dim_out=2,
        residual=FitResidual(in_sample=0.0, held_out=None, n_pairs=0, n_held_out=0,
                             dim_in=2, dim_out=2, closed_form=True),
    )
    assert honest.is_isometry is True, "the identity is an isometry and must measure as one"

    # 2. A residual cannot be silently read as a number.
    r = FitResidual(in_sample=0.01, held_out=1.0, n_pairs=6, n_held_out=2,
                    dim_in=16, dim_out=64, closed_form=False)
    with pytest.raises(TypeError):
        _ = r < 0.05          # noqa: B015 — the point is that this raises
    with pytest.raises(TypeError):
        float(r)

    # 3. A walk that carries no information says so, against a COMPUTED null.
    null, se = null_residual(64, 2)
    assert null == 1.0 and se > 0.0
    assert abs(r.z_below_null) < 1.0, (
        "a held-out residual sitting AT the null must not report as informative — that is the "
        "measured case (6 anchors, 16->64 dim, held_out 1.026 vs null 1.000).")
    assert r.underdetermined is True, "6 pairs fitting 16 input dims is underdetermined"


def test_no_clustering_remains_in_the_anchor_layer():
    """Walks the whole anchors package with the AST for any clustering-shaped definition or
    import, rather than checking a single module: a clusterer introduced in any file, not only
    one expected module, would defeat a narrower guard."""
    import ast
    import importlib
    import inspect
    import pkgutil

    import mantle.search.anchors as anchors_pkg

    # AnchorSet must not regrow a fitting constructor.
    assert not hasattr(AnchorSet, "bootstrap")

    mods = [anchors_pkg]
    for m in pkgutil.walk_packages(anchors_pkg.__path__, prefix=anchors_pkg.__name__ + "."):
        mods.append(importlib.import_module(m.name))
    assert len(mods) >= 5, f"anchor package discovery found only {[m.__name__ for m in mods]}"
    assert anchors_pkg in mods, "the package __init__ itself must be scanned"

    SHAPES = ("kmeans", "k_means", "cluster", "centroid", "medoid")
    offenders = []
    defs_seen = 0
    for mod in mods:
        src = inspect.getsource(mod)
        tree = ast.parse(src)
        # Guard the guard, at the scan level not per module: a re-export `__init__.py` legitimately
        # has no definitions of its own, so requiring some in every module fails on a correct file.
        # What must hold is that the scan as a whole saw real code.
        defs_seen += sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                         for n in ast.walk(tree))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if any(s in node.name.lower() for s in SHAPES):
                    offenders.append(f"{mod.__name__}.{node.name} (definition)")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if "sklearn" in a.name or any(s in a.name.lower() for s in SHAPES):
                        offenders.append(f"{mod.__name__}: import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                m = node.module or ""
                names = [a.name for a in node.names]
                if "sklearn" in m or any(s in (m + " " + " ".join(names)).lower() for s in SHAPES):
                    offenders.append(f"{mod.__name__}: from {m} import {names}")

    assert defs_seen >= 40, (
        f"the scan parsed only {defs_seen} definitions across {len(mods)} modules — too few to "
        f"believe it read the package")
    assert not offenders, (
        "clustering re-entered the anchor layer: "
        + "; ".join(offenders)
        + " -- Anchors are SEEDED. A locally-derived set mints region ids no peer computes."
    )
