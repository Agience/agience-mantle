"""One rounding law, and no second copy of it in mantle's search surface.

`prism.rounding` holds the forward-error bound for floating-point summation, `ε · Σ · n`, on
prism's dependency-free base — stdlib only, no numpy, no extra. `mantle/search/beacon/instrument.py`
consumes it rather than carrying its own copy, since beacon runs on numpy alone and a local copy of
the derivation risks drifting from `prism.rounding`'s.

Naming a duplicate does not stop it drifting, so this suite sweeps the live implementation against
a frozen reference copy kept as an oracle (Section 1) across a wide range of dtypes, shapes and
magnitudes, and guards against a new copy of the law reappearing anywhere else under
`src/mantle/search` (Section 2, via AST rather than grep, since several docstrings here discuss the
formula in prose).
"""
from __future__ import annotations

import ast
import itertools
import math
import pathlib
import struct

import numpy as np
import pytest

from mantle.search.beacon.instrument import _float_noise

SEARCH = pathlib.Path(__file__).resolve().parents[1] / "src" / "mantle" / "search"


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 1 · The oracle — a frozen reference copy, never imported
# ═════════════════════════════════════════════════════════════════════════════════════════════

def _deleted_float_noise(W: np.ndarray, energy: float, *, splits: int = 1) -> float:
    """An independent implementation of `mantle.search.beacon.instrument._float_noise`, kept
    verbatim and never wired to `prism.rounding`.

    Do not refactor it and do not make it call `prism.rounding`: an oracle that calls the thing it
    checks proves only that a function equals itself. If `prism.rounding` changes deliberately, this
    body changes in the same commit, so every moved number is explained."""
    eps = float(np.finfo(W.dtype).eps) if np.issubdtype(W.dtype, np.floating) \
        else float(np.finfo(float).eps)
    terms = 1 + 3 * max(1, int(splits))                    # the additions this verdict assembles
    elems = int(W.size) * (1 + 2 * max(1, int(splits)))    # every product each ‖·‖² performed
    return eps * max(float(energy), 0.0) * float(terms + elems)


def _bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def _frames():
    rng = np.random.default_rng(20260804)
    shapes = [(1, 1), (1, 2), (2, 1), (1, 64), (64, 1), (3, 7), (16, 16), (64, 256), (466, 195),
              (4, 16), (128, 3), (1000, 2)]
    mags = [0.0, 1e-300, 1e-8, 1.0, 1e8, 1e150]
    dtypes = [np.float64, np.float32, np.float16, np.longdouble, np.int64, np.complex128]
    for shape, mag, dt in itertools.product(shapes, mags, dtypes):
        with np.errstate(over="ignore", invalid="ignore"):
            base = rng.normal(size=shape)
            if dt is np.complex128:
                yield ((base + 1j * rng.normal(size=shape)) * mag).astype(dt)
            elif np.issubdtype(np.dtype(dt), np.integer):
                yield (base * min(mag, 1e17)).astype(dt)
            else:
                yield (base * mag).astype(dt)
    for dt in (np.float64, np.float32):
        yield np.zeros((8, 8), dtype=dt)          # all-zero: a real frame carrying zero energy
        yield np.zeros((0, 4), dtype=dt)          # empty: no elements at all
        yield np.zeros((4, 0), dtype=dt)
        yield np.full((4, 4), np.inf, dtype=dt)
    yield np.array([[True, False], [False, True]])   # a dtype with no `finfo` — the fallback branch


_ENERGIES = [0.0, -0.0, -1.0, -1e12, 5e-324, 1e-12, 1.0, 3.7, 1e6, 1e300, math.inf, math.nan]
_SPLITS = [-3, 0, 1, 2, 7, 1000]


def test_the_merged_law_agrees_with_the_deleted_implementation_on_every_input() -> None:
    """Compares the two implementations bit-for-bit — never `approx` — since comparing two
    tolerances with a tolerance would be the same defect one level up.

    Beacon's routing floor decides whether a tekton is coupled at all, so a disagreement in some
    corner nobody exercises directly would change that answer while `screen_read_vectors.json`
    still reports parity."""
    n = 0
    bad = []
    for W in _frames():
        for energy, splits in itertools.product(_ENERGIES, _SPLITS):
            n += 1
            got = _float_noise(W, energy, splits=splits)
            want = _deleted_float_noise(W, energy, splits=splits)
            if not ((math.isnan(got) and math.isnan(want)) or _bits(got) == _bits(want)):
                bad.append((W.shape, W.dtype.name, energy, splits, got, want))
    assert n >= 3000, f"the sweep collapsed to {n} inputs and would prove almost nothing"
    assert not bad, f"{len(bad)} of {n} inputs disagree with the deleted body; first five: {bad[:5]}"


def test_the_band_tracks_its_inputs_rather_than_being_a_constant_in_disguise() -> None:
    """A derivation that returns the same number regardless of its inputs is a constant in disguise.
    Each dependence the bound claims is asserted, one strict inequality each — the energy dependence
    is asserted as proportionality rather than monotonicity, because `ε · Σ · n` is what the
    derivation says."""
    small, large = np.ones((4, 16)), np.ones((2048, 16))
    assert _float_noise(small, 1.0) < _float_noise(large, 1.0), "size did not widen the band"
    assert _float_noise(small, 1.0, splits=1) < _float_noise(small, 1.0, splits=3), \
        "a longer walk did not widen the band"
    assert _float_noise(small, 100.0) == pytest.approx(100.0 * _float_noise(small, 1.0)), \
        "the band is not proportional to the energy it bounds, so it is not a relative bound"
    assert _float_noise(small.astype(np.float32), 1.0) > _float_noise(small, 1.0), \
        "a float32 frame must earn a wider band than a float64 one, from its own dtype"


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 2 · The reappearance guard — AST, not grep
# ═════════════════════════════════════════════════════════════════════════════════════════════

#: Every site under `src/mantle/search` allowed to touch a machine epsilon, with what it models.
#:
#: The annotation is the point, not the membership: these are not all the same bound, and merging
#: them would be worse than leaving them separate — a derivation can be exactly as wrong as a
#: constant if it models the wrong error. A new entry means stating the model.
ALLOWED = {
    ("beacon/instrument.py", "_float_noise"):
        "READ ONLY — reads the frame's dtype epsilon and hands it to prism.rounding, which holds "
        "the law and states the model (ACCUMULATION, valid only because every summed term is a "
        "non-negative ‖·‖²). No arithmetic happens here any more.",
    ("beacon/engine.py", "_permutation_core"):
        "RANK / REPRESENTATION — `max(N, F) * eps * s_max`, numpy's own `matrix_rank` tolerance, "
        "deciding whether the mean direction is a direction or centring residue. Nothing "
        "accumulates: there is no running total, only a zero that has been rounded.",
}

_EPS_NAMES = {"eps", "_eps", "epsilon", "_epsilon", "EPS", "EPSILON"}


def _enclosing(tree: ast.AST) -> dict[int, str]:
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(line, node.name)
    return owner


def _mentions_epsilon(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in _EPS_NAMES:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in _EPS_NAMES:
            return True
    return False


def _epsilon_sites(root: pathlib.Path) -> dict[tuple[str, str], list[int]]:
    """Two detectors over one tree, because a copy of the law can arrive either way:

      · a multiplication mentioning an epsilon anywhere inside it — what writing the law out looks
        like, whatever the local variable happens to be called;
      · a read of machine epsilon (`finfo(...).eps`) — where the value a hand-written law needs can
        enter this subtree at all.

    What it cannot catch: a copy that receives `eps` as a parameter under some other name and is
    fed from an already-allowed read. The read sites are enumerated, so such a copy still has to
    appear in a diff beside one of them.

    Scoped to `src/mantle/search` rather than the whole package."""
    found: dict[tuple[str, str], list[int]] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing(tree)
        rel = str(path.relative_to(root)).replace("\\", "/")
        for node in ast.walk(tree):
            hit = False
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                hit = _mentions_epsilon(node)
            elif isinstance(node, ast.Attribute) and node.attr in ("eps", "epsilon"):
                hit = True
            if hit:
                found.setdefault((rel, owner.get(node.lineno, "<module>")), []).append(node.lineno)
    return found


def test_no_second_implementation_of_the_rounding_law_exists_in_the_search_surface() -> None:
    """The guard is AST rather than grep, because several docstrings here discuss `ε · Σ · n` in
    prose — a textual search would match the prose, and tuning it until it stops matching the prose
    would leave it guarding nothing.

    What it guards against: someone needing a noise band, not knowing `prism.rounding` exists (or
    not wanting the import), and writing `eps * energy * n` locally. Such a copy can agree with the
    law on the day it is written and drift later."""
    sites = _epsilon_sites(SEARCH)
    unexpected = {k: v for k, v in sites.items() if k not in ALLOWED}
    assert not unexpected, (
        "an epsilon expression appeared outside the enumerated sites:\n"
        + "\n".join(f"  {f}:{ls} in {fn}()" for (f, fn), ls in sorted(unexpected.items()))
        + "\n\nIf it is the accumulation bound, call `prism.rounding` instead of writing it again. "
          "If it is a DIFFERENT bound, add it to ALLOWED with the error model it assumes — "
          "accumulation, cancellation, rank or representation.")
    assert set(ALLOWED) <= set(sites), (
        f"ALLOWED names sites that no longer exist: {sorted(set(ALLOWED) - set(sites))}. An "
        f"allow-list that has drifted from the tree stops being evidence about it.")


def test_the_guard_fires_on_a_seeded_copy(tmp_path: pathlib.Path) -> None:
    """The control: a guard that has never been seen to fire is indistinguishable from no guard,
    and this one concludes from an absence, so it must be shown that it can speak."""
    (tmp_path / "sneaky.py").write_text(
        "def _float_noise(W, energy, splits=1):\n"
        "    eps = 2.220446049250313e-16\n"
        "    terms = 1 + 3 * max(1, int(splits))\n"
        "    elems = int(W.size) * (1 + 2 * max(1, int(splits)))\n"
        "    return eps * max(float(energy), 0.0) * float(terms + elems)\n",
        encoding="utf-8")
    sites = _epsilon_sites(tmp_path)
    assert ("sneaky.py", "_float_noise") in sites, (
        f"the scanner did not find a verbatim copy of the deleted law, so its silence on the real "
        f"tree means nothing. It saw: {sorted(sites)}")
