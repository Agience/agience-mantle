"""AnchorSet + Reconciler — the native-language geometry layer.

Validates the load-bearing claim of the canonical plan (§4): the anchor-relative
code is **model-unbiased** — a gauge change (rotation = "a different embedding
model") leaves the native code unchanged.
"""

import numpy as np
import pytest

from mantle.search.anchors import AnchorSet, Reconciler
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
    """A PROVISIONED AnchorSet: the first real item of each of the first ``k`` clusters, admitted
    explicitly. No fitting — anchors are not derived here or anywhere else (the k-means bootstrap
    was removed 2026-07-31 because locally-derived anchors mint region ids no peer computes). In
    production the canonical set arrives as an artifact; in this test the test IS that outside
    authority, which is exactly the point: someone else decides, and every node admits the same set.
    Anchors stay REAL items (fully-disclosed artifacts, §3), which is what the routing assertions
    below actually depend on."""
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
    # anchors are REAL items (fully-disclosed artifacts), not synthetic centers
    assert all(a.label in labelset for a in aset.anchors)


def test_routing_lands_in_own_cluster():
    rng = np.random.default_rng(7)
    items, truth = _clusters(rng)
    aset = _provision(items, 4)
    rec = Reconciler(aset, top_m=4)
    label_by_id = {a.anchor_id: a.label for a in aset.anchors}
    correct = 0
    for (_, vec), c in zip(items, truth):
        code = rec.to_native(vec)
        top_cluster = label_by_id[code.top_anchor_id].split("_")[0]
        correct += int(top_cluster == f"c{c}")
    assert correct / len(items) > 0.9


def test_same_cluster_more_similar_than_cross():
    rng = np.random.default_rng(3)
    items, truth = _clusters(rng)
    aset = _provision(items, 4)
    rec = Reconciler(aset, top_m=4)
    codes = [rec.to_native(v) for _, v in items]
    same = [i for i, c in enumerate(truth) if c == truth[0]]
    diff = [i for i, c in enumerate(truth) if c != truth[0]]
    s = codes[same[0]].dot(codes[same[1]])
    d = codes[same[0]].dot(codes[diff[0]])
    assert s > d


def test_native_code_is_model_invariant_under_gauge_change():
    """The core option-B claim: rotate the space (a different 'model'), provision anchors from the
    same items in the rotated frame, and the native code is unchanged."""
    rng = np.random.default_rng(11)
    items, _ = _clusters(rng, n_clusters=4, per=30)
    labels = [lab for lab, _ in items]
    X = l2norm(np.vstack([v for _, v in items]))

    R, _ = np.linalg.qr(rng.standard_normal((D, D)))   # random orthogonal gauge
    Xr = (X @ R).astype(np.float32)

    A = _provision(list(zip(labels, X)), 4, model_id="hf:m1@1.0")
    B = _provision(list(zip(labels, Xr)), 4, model_id="hf:m2@1.0")
    recA, recB = Reconciler(A, top_m=4), Reconciler(B, top_m=4)

    for i in range(0, len(items), 7):
        ca = recA.to_native(X[i], model_id="hf:m1@1.0")
        cb = recB.to_native(Xr[i], model_id="hf:m2@1.0")
        la = sorted((A.anchors[int(p)].label, round(float(w), 4)) for p, w in zip(ca.indices, ca.weights))
        lb = sorted((B.anchors[int(p)].label, round(float(w), 4)) for p, w in zip(cb.indices, cb.weights))
        assert la == lb


def test_cross_model_without_crosswalk_fails_loudly():
    aset = AnchorSet("hf:m1@1.0", D)
    aset.add_text("a", l2norm(np.ones(D, dtype=np.float32)))
    rec = Reconciler(aset)
    try:
        rec.to_native(np.ones(D, dtype=np.float32), model_id="hf:other@1.0")
        assert False, "expected cross-walk error"
    except ValueError as exc:
        assert "cross-walk" in str(exc)


def test_sparsecode_dict_roundtrip():
    from mantle.search.anchors.reconciler import SparseCode

    rng = np.random.default_rng(1)
    items, _ = _clusters(rng, n_clusters=3, per=20)
    aset = _provision(items, 3)
    code = Reconciler(aset, top_m=3).to_native(items[0][1])
    back = SparseCode.from_dict(code.to_dict())
    assert list(back.indices) == list(code.indices)
    assert back.anchor_ids == code.anchor_ids
    assert back.dim == code.dim
    assert abs(back.dot(code) - 1.0) < 1e-5  # unit-norm code, self-cosine ~ 1


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
    # a loaded anchor reconciles identically to the original
    a = Reconciler(aset, top_m=4).to_native(items[0][1])
    b = Reconciler(loaded, top_m=4).to_native(items[0][1])
    assert abs(a.dot(b) - 1.0) < 1e-5


def test_require_live_anchorset_raises_when_not_provisioned(anchor_repo):
    """One path, and it cannot manufacture its own start.

    This used to assert the opposite — that the first call "light-trains the set from the seed
    corpus" — and it only passed because it monkeypatched a `bootstrap_anchorset` with a fake that
    returned anchors. With the k-means gone there is nothing to fake: an AnchorSet that is not
    provisioned is a provisioning failure, and the routed path must say so rather than invent cells
    no peer shares.
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
    from mantle.embeddings import model_id

    mid = model_id()
    assert mid.count("@") == 1 and ":" in mid  # <ns>:<path>@<ver>


def test_bootstrap_corpus_finds_platform_seeds():
    from mantle import manage_anchors

    corpus = manage_anchors.gather_seed_corpus()
    # the platform seed tree has 40+ artifacts (agents, servers, tools, docs)
    assert len(corpus) >= 20
    assert all(text for _, text in corpus)


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


# --------------------------------------------------------------------------
# P3 — density-zoom
# (Manifold-structure analysis lives in the Beacon add-on, not in core.)
# --------------------------------------------------------------------------


def test_density_layer_common_vs_novel():
    from mantle.search.anchors.anchorset import l2norm
    from mantle.search.anchors.density import DensityZoom

    rng = np.random.default_rng(2)
    d = 16
    base = l2norm(rng.standard_normal(d))
    aset = AnchorSet("hf:t@1.0", d)
    for i in range(8):
        aset.add_text(f"a{i}", l2norm(base + 0.03 * rng.standard_normal(d)))
    dz = DensityZoom(aset)

    common = dz.layer(l2norm(base + 0.01 * rng.standard_normal(d)))   # at the cluster
    orth = rng.standard_normal(d)
    orth = l2norm(orth - (orth @ base) * base)                        # orthogonal → far
    novel = dz.layer(orth)

    assert common[0] == "L2"
    assert novel[0] == "L0"
    assert common[1] > novel[1]


def test_density_layer_frame_invariant():
    from mantle.search.anchors.anchorset import l2norm
    from mantle.search.anchors.density import DensityZoom

    rng = np.random.default_rng(4)
    d = 16
    pts = l2norm(rng.standard_normal((10, d)))
    a = AnchorSet("hf:m1@1.0", d)
    for i, p in enumerate(pts):
        a.add_text(f"a{i}", p)
    R, _ = np.linalg.qr(rng.standard_normal((d, d)))
    b = AnchorSet("hf:m2@1.0", d)
    for i, p in enumerate(pts):
        b.add_text(f"a{i}", l2norm(p @ R))

    q = l2norm(rng.standard_normal(d))
    la, da = DensityZoom(a).layer(q)
    lb, db = DensityZoom(b).layer(q @ R)
    assert la == lb and abs(da - db) < 1e-4


# --------------------------------------------------------------------------
# P4 — cross-walk / AlignmentRegistry
# --------------------------------------------------------------------------

def test_crosswalk_enables_cross_model_reconcile():
    from mantle.search.anchors.crosswalk import CrosswalkRegistry, fit_crosswalk

    rng = np.random.default_rng(7)
    d = 16
    items = l2norm(rng.standard_normal((30, d)))        # concepts in TARGET space
    R, _ = np.linalg.qr(rng.standard_normal((d, d)))    # SOURCE = TARGET gauge-rotated
    items_src = l2norm(items @ R)

    aset = AnchorSet("hf:target@1.0", d)
    for i in range(6):
        aset.add_text(f"a{i}", items[i])

    reg = CrosswalkRegistry()
    cw = reg.register(fit_crosswalk(
        items_src, items,
        source_model_id="hf:source@1.0", target_model_id="hf:target@1.0",
        method="procrustes",
    ))
    assert cw.method == "procrustes" and cw.error_bound < 1e-3

    rec_x = Reconciler(aset, top_m=4, crosswalks=reg)
    rec_t = Reconciler(aset, top_m=4)
    q = 10
    code_src = rec_x.to_native(items_src[q], model_id="hf:source@1.0")
    code_tgt = rec_t.to_native(items[q])
    assert code_src.dot(code_tgt) > 0.99          # cross-walked code ≈ native code

    # without a registry, a foreign model still fails loudly
    try:
        Reconciler(aset, top_m=4).to_native(items_src[q], model_id="hf:source@1.0")
        assert False, "expected cross-walk error"
    except ValueError as exc:
        assert "cross-walk" in str(exc)


def test_crosswalk_linear_cross_dimension():
    from mantle.search.anchors.crosswalk import fit_crosswalk

    rng = np.random.default_rng(2)
    n, d_in, d_out = 60, 12, 20
    src = l2norm(rng.standard_normal((n, d_in)))
    M = rng.standard_normal((d_in, d_out))
    tgt = l2norm(src @ M)

    cw = fit_crosswalk(src, tgt, source_model_id="a@1", target_model_id="b@1")
    assert cw.method == "linear" and (cw.dim_in, cw.dim_out) == (d_in, d_out)
    assert cw.error_bound < 0.05

    x = l2norm(rng.standard_normal(d_in))
    assert float(cw.apply(x) @ l2norm(x @ M)) > 0.95


# --------------------------------------------------------------------------
# Anchor growth (RG-flow) — the AnchorSet grows as the manifold grows
# --------------------------------------------------------------------------

def test_propose_anchor_admits_novel_rejects_covered(anchor_repo):
    from mantle.search.anchors import store
    from mantle.search.anchors.anchorset import CANDIDATE
    from mantle.search.anchors.grow import propose_anchor

    d = 8
    base = l2norm(np.eye(d)[0])
    aset = AnchorSet("hf:t@1.0", d)
    for i in range(5):
        aset.add_text(f"a{i}", l2norm(base + 0.02 * np.random.default_rng(i).standard_normal(d)))
    store.save_live_anchorset(aset)
    before = len(store.get_live_anchorset())

    # a novel (orthogonal) signal in an uncovered region → admitted as CANDIDATE
    grown = propose_anchor("novel-concept", l2norm(np.eye(d)[4]))
    assert grown is not None and grown.tier == CANDIDATE
    assert len(store.get_live_anchorset()) == before + 1

    # a near-duplicate of the existing cluster → already covered → rejected
    assert propose_anchor("dup", l2norm(base + 0.001 * np.ones(d, dtype=np.float32))) is None
    assert len(store.get_live_anchorset()) == before + 1


def test_propose_anchor_no_anchorset_is_noop(anchor_repo):
    from mantle.search.anchors.grow import propose_anchor

    # Empty repo → no live AnchorSet → propose is a no-op.
    assert propose_anchor("x", [0.1] * 8) is None


# ── the anchor set is PROVISIONED, and that is enforced ─────────────────────────────────────────
# These fail if someone reinstates a local derivation. FAILURE MODE FIRST: without them, "we removed
# k-means" is a claim about one commit, and the next person who needs anchors on a fresh deployment
# rebuilds the clusterer — which is precisely how the region ids diverge silently again.
def test_the_derivation_entry_point_is_gone_entirely():
    """Not "raises" — GONE. A function that exists only to raise is an invitation with a docstring.

    The `bootstrap` module is renamed to `seed_corpus` (it gathers the corpus that gets indexed and
    nothing more), and the module that remains must expose no way to manufacture an AnchorSet.
    """
    import mantle.search.anchors as anchors_pkg
    from mantle.search.anchors import seed_corpus

    assert not hasattr(anchors_pkg, "bootstrap_anchorset")
    assert not hasattr(seed_corpus, "bootstrap_anchorset")
    assert not hasattr(seed_corpus, "DEFAULT_K")     # the anchor count is not a knob anywhere
    with pytest.raises(ImportError):
        from mantle.search.anchors import bootstrap  # noqa: F401 — must not resolve


def test_no_clustering_remains_in_the_anchor_layer():
    """⛔ THIS USED TO ASSERT `"KMeans" not in src` AND `"sklearn" not in src`, NEITHER OF WHICH THE
    REMOVED CODE EVER CONTAINED.

    The clusterer was a hand-rolled `def _kmeans_cosine(X, k, *, iters=25, seed=0)` built on numpy
    alone — `git show e3a9430^:.../anchorset.py` has zero occurrences of `sklearn` or `KMeans`. So
    both assertions were checks that could not fail against the removal they were written to guard,
    and an earlier "proof" that this test had teeth only worked because the revert used to test it
    seeded an `import sklearn` line that was never in the original.

    It also scoped `_kmeans_cosine` to ONE of the three modules, so restoring the clusterer in
    `seed_corpus.py` — the natural home, since that file IS the old `bootstrap.py` — passed cleanly.

    Now: walk the WHOLE anchors package with the AST for any clustering-shaped definition or import.

    ⚠ **Stated limit.** This matches on NAMES — definitions and imports whose identifier contains a
    clustering word. It is a naming guard, not a structural one: `def _lloyd_fit(X, k)` containing
    the identical iteration, or the same loop written inline inside another function, is NOT caught.
    An earlier version of this docstring claimed it catches clustering "whatever it is built on",
    which was an overclaim. Detecting a Lloyd iteration structurally is not something this test does;
    what it does is make the obvious reinstatement loud, and make the package boundary explicit so a
    new module cannot arrive outside the scan.
    """
    import ast
    import importlib
    import inspect
    import pkgutil

    import mantle.search.anchors as anchors_pkg

    # AnchorSet must not regrow a fitting constructor.
    assert not hasattr(AnchorSet, "bootstrap")

    # ⛔ THIS SCANNED THREE MODULES (`anchorset`, `seed_corpus`, `store`) OUT OF TEN. `grow.py` is the
    # natural home for a reinstated derivation — it already holds `propose_anchor`, the anchor
    # admission entry point — and it was not scanned. Walk the whole package instead, discovered
    # rather than listed, so a NEW module cannot arrive outside the guard.
    # ⛔ `iter_modules` SKIPS `__init__.py` AND DOES NOT RECURSE, so one file inside the boundary was
    # outside the scan — and `__init__.py` is a perfectly ordinary place to land a helper. Include the
    # package module itself and walk subpackages too, so "the WHOLE package" is true rather than
    # nearly true.
    mods = [anchors_pkg]
    for m in pkgutil.walk_packages(anchors_pkg.__path__, prefix=anchors_pkg.__name__ + "."):
        mods.append(importlib.import_module(m.name))
    assert len(mods) >= 9, f"anchor package discovery found only {[m.__name__ for m in mods]}"
    assert anchors_pkg in mods, "the package __init__ itself must be scanned"

    SHAPES = ("kmeans", "k_means", "cluster", "centroid", "medoid")
    offenders = []
    defs_seen = 0
    for mod in mods:
        src = inspect.getsource(mod)
        tree = ast.parse(src)
        # Guard the guard, at the SCAN level not per module: a re-export `__init__.py` legitimately
        # has no definitions of its own, so requiring some in EVERY module fails on a correct file.
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

    assert defs_seen >= 20, (
        f"the scan parsed only {defs_seen} definitions across {len(mods)} modules — too few to "
        f"believe it read the package")
    assert not offenders, (
        "clustering re-entered the anchor layer: "
        + "; ".join(offenders)
        + " -- Anchors are PROVISIONED. A locally-derived set mints region ids no peer computes."
    )
