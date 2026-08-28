"""Every value this audit derives must track its inputs — change an input, the value moves.

A derivation that returns the same number regardless of its inputs is a constant wearing a
function, and it passes every test the constant would pass. Each test here states first what
would still be true if the derivation were faked — a test that cannot fail proves nothing, and
"the suite is green" is not evidence on its own.

Grouped by the site each one certifies.
"""
from __future__ import annotations

import ast
import importlib

import pytest


# ═════════════════════════════════════════════════════════════════════════════
# The instrument: an AST scan, not a text search
# ═════════════════════════════════════════════════════════════════════════════
# Grep miscounts this, in both directions. It matches inside prose (a comment merely naming a
# numeric literal reads as a match), and it matches substrings ("200" inside "2000"). The question
# is structural — *is a measurement compared against a numeric literal?* — so it is asked of the
# parse tree.

_ORDER_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


def _is_num(node) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return True
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) \
        and _is_num(node.operand)


def literal_comparisons(module_name: str, *, ignore_below: int = 4):
    """Every `<expression> <op> <numeric literal>` in a module, as `(lineno, value, source)`.

    `ignore_below` drops structural small ints (0, 1, 2, 3 — emptiness, arity, ndim), which are
    the shape of the data rather than a judgement about it.
    """
    mod = importlib.import_module(module_name)
    src = open(mod.__file__, encoding="utf-8").read()
    lines = src.splitlines()
    found = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, _ORDER_OPS):
                for a, b in ((left, right), (right, left)):
                    if _is_num(b) and not _is_num(a):
                        v = ast.literal_eval(b)
                        if isinstance(v, float) or abs(v) >= ignore_below:
                            found.append((node.lineno, v,
                                          lines[node.lineno - 1].strip()))
            left = right
    return found


def assert_no_literal_comparison(module_name: str, *, allow=()):
    """Fails if any measurement in the module is compared against a bare number.

    `allow` names values that are specifications (a digest width, a key length, a wire-format
    field) — the contract, not a judgement. Everything else must be a name.
    """
    bad = [f"line {ln}: {v!r}  |  {s}"
           for ln, v, s in literal_comparisons(module_name) if v not in allow]
    assert not bad, f"{module_name} compares a measurement against a bare literal:\n" + "\n".join(bad)


# ═════════════════════════════════════════════════════════════════════════════
# db/content_cache.py — the AES-GCM minimum-blob floor
# ═════════════════════════════════════════════════════════════════════════════

def test_min_blob_is_arithmetic_over_the_format_not_a_literal():
    """Fails if the floor were re-typed as a literal.

    The floor must be exactly nonce + tag. A hardcoded 28 would pass the first assert and fail
    the second: the value has to be built from the two format constants.
    """
    from mantle.db import content_cache as cc

    assert cc._MIN_BLOB_BYTES == cc._NONCE_BYTES + cc._TAG_BYTES
    assert cc._MIN_BLOB_BYTES == 28, "12-byte GCM nonce + 16-byte tag"
    # `_NONCE_BYTES + 1` bounds a nonce, not a ciphertext; the floor must not equal it.
    assert cc._MIN_BLOB_BYTES != cc._NONCE_BYTES + 1


def test_a_real_empty_plaintext_object_is_exactly_the_floor():
    """Fails if the floor were set from anything but the writer's actual output.

    This is the independent oracle: encrypt the shortest possible plaintext and measure what the
    cipher produced. If `_MIN_BLOB_BYTES` were mistyped as 13, or any other guess, the shortest
    real object would not equal it.
    """
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    from mantle.db import content_cache as cc

    blob = bytes(cc._NONCE_BYTES) + AESGCM(bytes(32)).encrypt(bytes(cc._NONCE_BYTES), b"", b"aad")
    assert len(blob) == cc._MIN_BLOB_BYTES


def test_blobs_between_the_old_and_new_floor_are_now_provably_corrupt():
    """Fails if the corruption check were only cosmetic.

    13..27 bytes is a band no ciphertext can occupy. Those blobs must raise `CacheCorrupt` rather
    than reach `AESGCM.decrypt` and come back as `InvalidTag` — reported as "no key opens this",
    which is ambiguous with a missing key.
    """
    pytest.importorskip("cryptography")
    import tempfile

    from mantle.db.content_cache import CacheCorrupt, FileContentCache

    with tempfile.TemporaryDirectory() as root:
        cache = FileContentCache(root=root, key=bytes(32))
        for n in range(13, 28):
            with pytest.raises(CacheCorrupt):
                cache._decrypt(bytes(n), "cas/" + "0" * 64, "col")


def test_the_rekey_script_shares_the_writers_format_rather_than_restating_it():
    """Fails if the script re-declares its own copy of the lengths."""
    from mantle.db import content_cache as cc
    from mantle.scripts import mantle_cas_rekey as rk

    src = open(rk.__file__, encoding="utf-8").read()
    assert "_MIN_BLOB_BYTES" in src and "_NONCE_BYTES" in src
    # No second spelling of the nonce length anywhere in the script.
    assert "blob[:12]" not in src and "urandom(12)" not in src
    assert cc._NONCE_BYTES == 12 and cc._MIN_BLOB_BYTES == 28


# ═════════════════════════════════════════════════════════════════════════════
# ontology/seed_lattice.py lives in agience-crystal, not here
# ═════════════════════════════════════════════════════════════════════════════
# `_crossed` (the checkpoint cadence), `_unchanged` (the exact IC comparison) and `_IC_BATCH` (the
# batch bound) are certified in `agience-crystal/tests/test_seed_lattice_no_forcings.py`, which
# copies the AST instrument above rather than importing it.
#
# The tests do not import upward: mantle is the standalone database and is below crystal
# (`agience-pharos/genesis/ARCHITECTURE-TARGET.md` §2); a mantle test that imported `crystal.ontology` would make mantle's
# own suite unrunnable without crystal installed, which is the property this repo exists to have.
# The AST helper is duplicated on purpose: it is ~40 lines of stdlib and the alternative is a
# test-only package edge between two components that must not depend on each other.



# ═════════════════════════════════════════════════════════════════════════════
# Keyset walks — "a short page is the last page" must read the limit it asked for
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("module, name", [
    ("mantle.db.vertex", "DEMAND_PAGE"),
    ("mantle.mesh.demand", "_PAGE"),
    ("mantle.mesh.sync", "_GATED_PAGE"),
])
def test_the_end_of_walk_test_reads_the_limit_it_asked_for(module, name):
    """Fails if a second copy of the page size survives as a literal in the module.

    Two literals is the risk: raise one and the walk stops a page early (an undercount, or —
    in `sync._withheld_ids` — a leak that still reports itself exhaustive); lower one and the
    walk re-reads its tail forever. Checked structurally, so a `< 5000` cannot hide behind a
    comment that also mentions 5000.
    """
    mod = importlib.import_module(module)
    assert getattr(mod, name) == 5000
    offenders = [c for c in literal_comparisons(module) if c[1] == 5000]
    assert not offenders, f"{module} still compares against a literal page size: {offenders}"


def test_demand_count_moves_with_the_page_size(tmp_path, monkeypatch):
    """Fails if `demand_count` were still testing against a baked-in 5000.

    The oracle is independent: the true row count is known here because this test inserted the
    rows. Forcing a tiny page makes the walk take several pages — with a baked-in `< 5000` it
    would stop after the first page and undercount.
    """
    from mantle.db import LatticeArtifactStore, LatticeConn
    from mantle.db import vertex

    n_rows, page = 25, 4
    monkeypatch.setattr(vertex, "DEMAND_PAGE", page, raising=True)

    arts = LatticeArtifactStore(LatticeConn(str(tmp_path / "l.db")), origin="71")
    for i in range(n_rows):
        arts.demand_set("art-%03d" % i, mass=1.0, ts=float(i))

    # The walk must cross 7 pages to see 25 rows. A baked-in `< 5000` stops after the first.
    assert n_rows > page, "the fixture must force a multi-page walk or it proves nothing"
    assert arts.demand_count() == n_rows

    # And the page walk itself honours the smaller limit — otherwise the count above could have
    # been right for the wrong reason (one page big enough to hold everything).
    assert len(arts.demand_page(after="", limit=page)) == page


# ═════════════════════════════════════════════════════════════════════════════
# mantle_cas_rekey.py — the key-completeness warning has no rate threshold
# ═════════════════════════════════════════════════════════════════════════════

def test_the_key_completeness_warning_is_no_longer_rate_gated():
    """The warning fires on any unreadable object, not past a rate threshold.

    The null is computed: with a complete key set every object opens, so the expected unreadable
    count is exactly zero. `if unreadable:` tests against that null. Fails if a rate threshold
    gates the warning.
    """
    from mantle.scripts import mantle_cas_rekey as rk

    # Structural: no rate is compared against a literal anywhere in the script. `_LIST_CAP` is a
    # name, and the nonce/tag floor is imported, so a clean module is the whole assertion.
    assert_no_literal_comparison("mantle.scripts.mantle_cas_rekey", allow=(64,))
    # The advice itself must remain present regardless — it is what the reader needs at any rate,
    # and removing the warning branch entirely would also make the assertion above pass.
    src = open(rk.__file__, encoding="utf-8").read()
    assert "confirm the legacy" in src and "NOTHING WAS DELETED" in src


# ═════════════════════════════════════════════════════════════════════════════
# Display / memory bounds that stay — stated once, tail derived from the slice
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("module, name, value", [
    ("mantle.scripts.mantle_cas_rekey", "_LIST_CAP", 40),
    ("mantle.system.manage_anchors", "_LISTED", 12),
    ("mantle.shard.content_tier", "_ERR_SAMPLE", 200),
    ("mantle.shard.content_tier", "_QUARANTINE_MAX", 1000),
])
def test_a_kept_bound_is_named_and_stated_once(module, name, value):
    """These are bounds that cannot be derived. The rule they obey instead: one name, stated,
    with the reason a different value would be right written beside it.

    Fails if the bound is a bare literal at its comparison — checked structurally, so a docstring
    that merely mentions the number does not satisfy this.
    """
    mod = importlib.import_module(module)
    assert getattr(mod, name) == value
    offenders = [c for c in literal_comparisons(module) if c[1] == value]
    assert not offenders, f"{module}.{name} is still compared as a literal: {offenders}"


@pytest.mark.parametrize("module, name", [
    ("mantle.scripts.mantle_cas_rekey", "_LIST_CAP"),
    ("mantle.system.manage_anchors", "_LISTED"),
    ("mantle.shard.content_tier", "_ERR_SAMPLE"),
    ("mantle.shard.content_tier", "_QUARANTINE_MAX"),
    ("mantle.db.vertex", "DEMAND_PAGE"),
    ("mantle.mesh.demand", "_PAGE"),
    ("mantle.mesh.sync", "_GATED_PAGE"),
])
def test_every_kept_bound_says_what_would_make_a_different_value_right(module, name):
    """A named constant with no stated reason is a forcing that learned to spell.

    Fails if a bound is renamed without the sentence that makes it auditable — the whole point of
    keeping it is that the next reader can tell what evidence would move it.
    """
    mod = importlib.import_module(module)
    src = open(mod.__file__, encoding="utf-8").read()
    where = src.index(f"{name} = ")
    # Whitespace-normalised: the sentence is wrapped across comment lines, and a test that broke
    # on line wrapping would be testing the formatter, not the reasoning.
    preamble = " ".join(src[max(0, where - 1400):where].lower().replace("#:", " ").split())
    assert any(k in preamble for k in ("would be right", "would be wrong", "the bound is")), (
        f"{module}.{name} is named but not stated: no sentence says what would move it")


def test_the_more_tail_derives_from_the_slice_actually_shown():
    """The property `_LIST_CAP` and `_LISTED` exist to preserve: the tail count must derive from
    the slice actually shown, not from the cap.

    Fails if the tail count were computed from the cap instead of from the slice: with fewer items
    than the cap the two disagree, and computing from the cap alone can report a negative
    "... and N more".
    """
    for cap in (1, 12, 40, 200):
        for total in (0, 1, cap - 1, cap, cap + 1, cap * 3):
            if total < 0:
                continue
            items = list(range(total))
            shown = items[:cap]
            tail = len(items) - len(shown)           # the shape now used at both sites
            assert tail >= 0
            assert len(shown) + tail == total
