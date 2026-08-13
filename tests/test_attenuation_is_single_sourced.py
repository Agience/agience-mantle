"""One attenuation operator, and no second copy of it anywhere under `src/mantle`.

`mantle.attenuation` holds the meet — the intersection of CRUDEASIO authority that the
bundle ceiling, the light cone and the `edge.propagate` column all compose with. It exists
because the algebra used to be implemented twice, in two encodings, and the two disagreed
about the zero element: `Grant.masked_by` knew deny was absorbing and the light-cone path
did not, which is audit finding S1. A second copy would not have to be wrong on the day it
is written; it only has to drift later.

Naming a duplicate does not stop it reappearing, so this file measures instead of asking:

1. **The operator is defined once.** Only `attenuation.py` may define a meet.
2. **The two shapes a re-implementation takes** are swept for by AST, with every surviving
   site enumerated and annotated. Grep would be the wrong tool — several docstrings here and
   in `attenuation.py` discuss intersection in prose, and tuning a text search until it
   stopped matching the prose would leave it guarding nothing.
3. **The guard is shown to fire**, on a seeded copy of each shape. A guard that concludes
   from an absence and has never been seen to speak is indistinguishable from no guard.

Modelled on `tests/test_rounding_law_is_single_sourced.py`, which does the same job for the
floating-point rounding law, down to the annotated ALLOWED table: the annotation is the
point, not the membership. Adding an entry means stating what the site does and why it is
not the operator.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from mantle.attenuation import FLAG_OF

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "mantle"

#: The module that owns the algebra. Nothing here may be allow-listed away.
OWNER = "attenuation.py"


# ═════════════════════════════════════════════════════════════════════════════════════════
# 0 · Reading the tree
# ═════════════════════════════════════════════════════════════════════════════════════════

def _enclosing(tree: ast.AST) -> dict:
    owner: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(line, node.name)
    return owner


def _names(node: ast.AST) -> set:
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr)
    return out


def _modules(root: pathlib.Path):
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                                   # pragma: no cover - not our tree
            continue
        yield str(path.relative_to(root)).replace("\\", "/"), tree


# ═════════════════════════════════════════════════════════════════════════════════════════
# 1 · The operator is defined once
# ═════════════════════════════════════════════════════════════════════════════════════════

#: Every spelling a meet arrives under. `__and__`/`__rand__` because that is the operator
#: form; the words because that is what someone writes when they do not know the operator
#: exists — which is the case this whole file is about.
_MEET_NAMES = {"__and__", "__rand__", "intersect", "intersection", "attenuate", "meet",
               "narrow", "narrowed_by"}


def test_the_meet_is_defined_in_exactly_one_module() -> None:
    """`masked_by` is deliberately NOT in this set: it is a consumer that calls the meet on
    two masks and writes the result back onto an entity, not a second definition of it. If
    it ever stops delegating, section 2's sweep is what notices."""
    found = {}
    for rel, tree in _modules(SRC):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _MEET_NAMES:
                found.setdefault(rel, []).append(f"{node.name}:{node.lineno}")
    assert OWNER in found, (
        f"{OWNER} defines no meet at all — the single source is missing, so everything "
        f"below is guarding an empty claim. Saw: {found}")
    strays = {k: v for k, v in found.items() if k != OWNER}
    assert not strays, (
        "a second attenuation operator appeared:\n"
        + "\n".join(f"  {f}: {', '.join(v)}" for f, v in sorted(strays.items()))
        + f"\n\nIf it composes CRUDEASIO authority, call `mantle.{OWNER[:-3]}` instead. If it "
          "is a meet over something else entirely, rename it so it does not read as this one.")


def test_the_owner_exports_the_whole_algebra_the_consumers_need() -> None:
    """A single source that does not cover the consumers' needs is how the second copy gets
    written. These are the exact entry points `Grant.masked_by`, `lightcone` and the
    `edge.propagate` readers use."""
    from mantle import attenuation

    for name in ("Mask", "DENY", "TOP", "NOTHING", "ACTIONS", "FLAG_OF",
                 "attenuate", "compose", "propagates"):
        assert hasattr(attenuation, name), f"attenuation.{name} is missing"
    for name in ("from_propagate", "to_propagate", "from_flags", "to_flags",
                 "allows", "carries", "intersect"):
        assert hasattr(attenuation.Mask, name), f"Mask.{name} is missing"


# ═════════════════════════════════════════════════════════════════════════════════════════
# 2 · The reappearance sweep — AST, not grep
# ═════════════════════════════════════════════════════════════════════════════════════════
#
# Two detectors, because a re-implementation can arrive in either encoding:
#
#   `flag-fold`  — a loop or comprehension driven by the CRUDEASIO flag tuple. That is what
#                  writing the bundle ceiling out by hand looks like, whatever the local
#                  variable is called.
#   `mask-in`    — a membership test against something named like a propagate mask. That is
#                  what writing the light-cone prune out by hand looks like.
#   `bare-flag`  — `getattr(g, flag, False)` inside a function that never mentions the
#                  effect. Not an intersection, but the OTHER half of the same defect: it is
#                  the exact expression S1 was, and re-pointing a site at the operator is
#                  what removes it. Enumerated so the remaining ones cannot grow quietly.
#   `dict-flag`  — the same read against a raw grant DOCUMENT, `d.get("can_read")`. The
#                  lattice-side paths never load an entity, so `bare-flag` cannot see them;
#                  without this detector the deny-blind sites under `db/` would be invisible
#                  to a file whose whole subject is deny-blindness.

_FLAG_TUPLES = {"PERMISSION_FLAGS", "ACTION_FLAGS", "_ACTION_FLAGS", "_ACTION_FLAG_MAP",
                "FLAG_OF"}
_MASK_NAMES = {"mask", "masks", "propagate", "propagate_mask", "prop_mask", "_prop_mask"}
#: The nine stored column names, taken from the owner so this cannot go stale.
_FLAG_ATTRS = frozenset(FLAG_OF.values())
#: Anything that shows the author was thinking about `effect` in that function.
_EFFECT_NAMES = {"grant_is_allow", "grant_is_deny", "is_allow", "is_deny", "effect",
                 "mask_of", "allows", "EFFECT_DENY", "EFFECT_ALLOW"}


def _sites(root: pathlib.Path) -> dict:
    """`{(module, function, detector): [lines]}` for every hit in the tree."""
    found: dict = {}

    def hit(rel, fn, kind, line):
        found.setdefault((rel, fn, kind), []).append(line)

    for rel, tree in _modules(root):
        owner = _enclosing(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.AsyncFor)) and _names(node.iter) & _FLAG_TUPLES:
                hit(rel, owner.get(node.lineno, "<module>"), "flag-fold", node.lineno)
            elif isinstance(node, ast.comprehension) and _names(node.iter) & _FLAG_TUPLES:
                hit(rel, owner.get(node.iter.lineno, "<module>"), "flag-fold", node.iter.lineno)
            elif (isinstance(node, ast.Compare)
                  and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
                  and any(_names(c) & _MASK_NAMES for c in node.comparators)):
                hit(rel, owner.get(node.lineno, "<module>"), "mask-in", node.lineno)
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            if _names(fn) & _EFFECT_NAMES:
                continue                       # the author paired the bit with the effect
            for call in [c for c in ast.walk(fn) if isinstance(c, ast.Call)]:
                if (isinstance(call.func, ast.Name) and call.func.id == "getattr"
                        and len(call.args) > 1 and isinstance(call.args[1], ast.Name)
                        and "flag" in call.args[1].id.lower()):
                    hit(rel, fn.name, "bare-flag", call.lineno)
                elif (isinstance(call.func, ast.Attribute) and call.func.attr == "get"
                      and call.args and isinstance(call.args[0], ast.Constant)
                      and isinstance(call.args[0].value, str)
                      and call.args[0].value in _FLAG_ATTRS):
                    hit(rel, fn.name, "dict-flag", call.lineno)
    return found


#: Every surviving site, with what it actually does. A new entry means stating that.
#:
#: The six RE-POINT entries this table used to carry are GONE — every one of them has been
#: re-pointed at `mantle.attenuation` and its detector no longer fires:
#:
#:   * `lattice_api.list_origin_descendants` and `dependencies.check_access` (both `mask-in`)
#:     now call `attenuation.propagates(column_value, action)`.
#:   * `artifacts_router.list_visible` (`bare-flag`) and `grant_service.user_has_any_flag`
#:     (`bare-flag`) now ask `mask_of(g).allows(action)`.
#:   * `access.invokable_resources` (`dict-flag`) — the one that failed OPEN — now asks
#:     `allows`, and subtracts the deny reach.
#:   * `lattice_api.get_active_collection_ids_for_user` (`dict-flag`) — the read light cone one
#:     layer below `access.reachable_collections`, and the second site that failed OPEN — now
#:     asks `allows` and subtracts the deny reach too.
#:   * `access.gated_collections` / `access.gated_owner_map` (`dict-flag`) now read the bit
#:     through `Mask.carries`, which is effect-blind BY DESIGN there: gating asks "is this
#:     administered", not "is this authorized", and a deny grant must keep a collection
#:     private. See their docstrings — reading them as `allows` would have been the widening.
#:
#: Removing an entry is the point of fixing one: the `stale` assertion below is what forces a
#: RE-POINT to be deleted from this table in the same commit that closes it, so the table can
#: never claim a defect is still there once it is not.
ALLOWED = {
    ("routers/grants_router.py", "_explicit_bits", "flag-fold"):
        "NOT A GRANT — folds over the REQUEST BODY to find which CRUDEASIO fields the caller "
        "actually set. There is no second grant and nothing is intersected; a request model "
        "has no effect to honour.",
    ("routers/grants_router.py", "_explicit_bits", "bare-flag"):
        "NOT A GRANT — same site, reading the request body's fields.",
    ("routers/grants_router.py", "add_bundle_member_endpoint", "flag-fold"):
        "NOT A GRANT — marshals the request body into the `flags` dict `add_member` takes. "
        "The narrowing happens later, in `masked_by`, against the real ceiling.",
    ("routers/grants_router.py", "add_bundle_member_endpoint", "bare-flag"):
        "NOT A GRANT — same site, reading the request body's fields.",
    ("services/grant_key_service.py", "_flags_from", "flag-fold"):
        "NORMALISATION, NOT COMPOSITION — completes a partial permission spec to a full "
        "CRUDEASIO map so a `Grant` built from it cannot inherit the constructor's "
        "can_read=True default. One input, so there is nothing to intersect with.",
    ("services/grant_key_service.py", "_open_ceiling", "flag-fold"):
        "CONSTRUCTION, NOT COMPOSITION — builds the all-True bundle root. It is the value "
        "`attenuation.TOP` denotes, expressed in the entity's flag encoding at the point a "
        "Grant is constructed; the ceiling it later acts as is applied by `masked_by`.",
    ("entities/grant.py", "from_dict", "dict-flag"):
        "DESERIALISATION, NOT A DECISION — reads the stored columns back onto the entity. It "
        "must be effect-blind: a deny grant's bits say WHICH actions it denies, so dropping "
        "them here would turn every stored deny into a deny of nothing.",
    ("services/grant_store.py", "upsert_user_grant", "dict-flag"):
        "SERIALISATION, NOT A DECISION — marshals a caller-supplied flags dict into the "
        "upsert's keyword arguments. Nothing is read from a grant and nothing is composed.",
}


def test_no_second_implementation_of_the_attenuation_operator_exists() -> None:
    sites = _sites(SRC)
    unexpected = {k: v for k, v in sites.items() if k not in ALLOWED}
    assert not unexpected, (
        "CRUDEASIO authority was composed outside the enumerated sites:\n"
        + "\n".join(f"  {f}:{ls} in {fn}()  [{kind}]"
                    for (f, fn, kind), ls in sorted(unexpected.items()))
        + "\n\nIf it intersects authority, call `mantle.attenuation` — `a & b`, or "
          "`propagates(edge_value, action)` for the column form. If it reads a CRUDEASIO bit "
          "to make a decision, use `mask_of(grant).allows(action)` so the effect is not "
          "dropped. If it is genuinely neither, add it to ALLOWED saying which.")
    stale = sorted(set(ALLOWED) - set(sites))
    assert not stale, (
        f"ALLOWED names sites that no longer exist: {stale}. An allow-list that has drifted "
        f"from the tree stops being evidence about it — delete the entries, and if a "
        f"RE-POINT one was fixed, say so in the commit.")


def test_the_owner_module_is_not_itself_caught_by_the_sweep() -> None:
    """The single source must be invisible to its own detectors, or ALLOWED would have to
    exempt it and the exemption would cover any future copy pasted beside it."""
    assert not [k for k in _sites(SRC) if k[0] == OWNER], (
        f"{OWNER} tripped its own sweep; the detectors would have to allow-list the owner, "
        f"which weakens them")


# ═════════════════════════════════════════════════════════════════════════════════════════
# 3 · The control — the guard is shown to fire
# ═════════════════════════════════════════════════════════════════════════════════════════

_SEEDS = {
    "flag-fold": (
        "def masked_by(self, ceiling):\n"
        "    out = copy(self)\n"
        "    for flag in Grant.PERMISSION_FLAGS:\n"
        "        if not getattr(ceiling, flag, False):\n"
        "            setattr(out, flag, False)\n"
        "    return out\n"),
    "mask-in": (
        "def walk(edge, action):\n"
        "    mask = _prop_mask(edge)\n"
        "    if mask is not None and action not in mask:\n"
        "        return None\n"
        "    return edge\n"),
    "bare-flag": (
        "def visible(grant, flag_attr):\n"
        "    return bool(getattr(grant, flag_attr, False))\n"),
    "dict-flag": (
        "def reachable(docs, user_id):\n"
        "    return [d['resource_id'] for d in docs\n"
        "            if d.get('grantee_id') == user_id and d.get('can_read')]\n"),
}


@pytest.mark.parametrize("detector", sorted(_SEEDS))
def test_the_guard_fires_on_a_seeded_copy(detector: str, tmp_path: pathlib.Path) -> None:
    """Each seed is the shape actually deleted or actually still in the tree, not a
    caricature: `flag-fold` is `masked_by` as it read before this phase, `mask-in` is the
    origin-BFS prune verbatim, `bare-flag` is the S1 expression itself."""
    (tmp_path / "sneaky.py").write_text(_SEEDS[detector], encoding="utf-8")
    found = _sites(tmp_path)
    assert [k for k in found if k[0] == "sneaky.py" and k[2] == detector], (
        f"the {detector} detector did not find a verbatim copy, so its silence on the real "
        f"tree means nothing. It saw: {sorted(found)}")


def test_the_meet_uniqueness_check_fires_on_a_seeded_copy(tmp_path: pathlib.Path) -> None:
    (tmp_path / "sneaky.py").write_text(
        "class Perms:\n"
        "    def __and__(self, other):\n"
        "        return Perms(self.bits & other.bits)\n",
        encoding="utf-8")
    found = {rel for rel, tree in _modules(tmp_path)
             for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name in _MEET_NAMES}
    assert found == {"sneaky.py"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
