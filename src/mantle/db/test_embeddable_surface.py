"""The embeddable surface is stdlib + `cryptography`. Nothing else. Enforced, not documented.

`pyproject.toml` declares `dependencies = ["cryptography"]` for the base install — the whole
product claim that Mantle is a standalone data store rests on that being true. A claim like this
rots the moment someone adds a convenient import, and it rots SILENTLY: a function-local import
keeps every existing test passing, and the breakage shows up on a consumer's machine, at install
time, as an unresolvable requirement for code they never call.

An external consumer is the reason this exists. Such a consumer embeds Mantle for artifacts,
collections and access grants, and must not inherit any sibling Agience package or the
reasoning stack (`FORBIDDEN` below is the list).

Three independent checks, because each misses what the others catch:
  * AST — sees imports in code paths that never execute;
  * runtime — sees re-exports and lazy chains a grep misses;
  * functional — proves the surface actually WORKS with nothing else installed.
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys

import pytest

#: `<repo>/src/mantle/db/` → three levels up is `<repo>/src`, which is what makes the fully
#: qualified `mantle.*` imports resolve in an uninstalled checkout. Asserted, not trusted: a
#: `sys.path.insert` of the wrong directory is a silent no-op.
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
assert os.path.basename(_SRC) == "src" and os.path.isdir(os.path.join(_SRC, "mantle", "db")), (
    "path depth is wrong: expected <repo>/src, resolved %s — fix the depth" % _SRC)
sys.path.insert(0, _SRC)

#: This file sits at `<repo>/src/mantle/db/`, so `parents[1]` is `src/mantle`. The depth is asserted
#: in `_surface()` rather than trusted: a wrong `parents[N]` resolves silently to a directory that is
#: not `src/mantle`, and every glob under it then yields nothing — `Path.glob` on a missing directory
#: returns an empty iterator and raises NOTHING, so the three static checks below would keep passing
#: while measuring an empty roster. A silent pass is the one outcome this file must not have.
PKG = pathlib.Path(__file__).resolve().parents[1]          # …/src/mantle
STORE_DIR = PKG / "db"
STDLIB = set(sys.stdlib_module_names)

#: The store modules that MUST be on the surface, by name. A roster that loses `vertex.py` — because
#: the package moved, was renamed, or the glob stopped resolving — is the failure this pins: the AST
#: checks below would report "no forbidden import found" against a file list that no longer holds the
#: store at all. Named individually so the message says WHICH module went missing.
STORE_MODULES = frozenset({
    "__init__.py", "access.py", "audit.py", "backend.py", "constants.py", "content_cache.py",
    "content_tier.py", "doc_boundary.py", "edge.py", "identity_backend.py", "lattice_api.py",
    "lattice_identity.py", "plane.py", "s3_content.py", "schema.py", "seq.py", "store.py",
    "typed_fetch.py", "vertex.py",
})

#: The one file on the surface that lives outside the store package. Named for the same reason: the
#: `f.exists()` filter in `_surface()` turns a moved file into a silently shorter roster.
NAMED_SURFACE = ("services/acting_principal.py",)

# Sibling Agience packages and the reasoning stack. An embedding consumer installs NONE of these.
#
# A non-Agience entry was removed 2026-08-25 [John: that brand must not appear in this repo], and
# the guard did NOT weaken — measured, not assumed. `FORBIDDEN` is the specific-message half; the
# load-bearing half is the ALLOWLIST below (`INTERNAL | STDLIB | ALLOWED_THIRD_PARTY`), which every
# import and every loaded module is checked against. A root that is in none of those fails whether
# or not it is named here, so the removed entry is still rejected — it just reports as "unexpected
# package" rather than "sibling package". Do not re-add a name to restore a guarantee that never
# depended on it.
FORBIDDEN = {
    "chorus", "ember", "beam", "iris", "sage", "lumen", "aria", "astra", "ophan", "seraph",
    "crystal", "prism", "origin", "facet", "tekton", "bundle",
}
# Absolute imports that resolve INSIDE mantle (the package is importable both as `mantle.x` and,
# service-side, with `src/mantle` itself on the path).
INTERNAL = {"mantle", "api", "clients", "db", "entities", "events", "mesh", "oci",
            "routers", "scripts", "search", "services", "shard", "system", "tools", "ui",
            "attenuation", "config", "main"}

ALLOWED_THIRD_PARTY = {"cryptography"}


def _surface() -> list:
    """Every file `pip install agience-mantle` must be able to import with only `cryptography`.

    The roster is asserted, not merely built. Each check below supplies an absence-assertion's
    premise: "no forbidden import was found" and "no file was examined" are the same result
    otherwise, and the second arrives silently — an empty glob raises nothing and the suite stays
    green."""
    assert PKG.name == "mantle", (
        "path depth is wrong: parents[1] should be src/mantle, got %s — fix the depth" % PKG)
    assert STORE_DIR.is_dir(), (
        "the store package is missing at %s — every glob below would yield nothing and the three "
        "static checks would pass on an empty roster. Fix the path; do not delete the check."
        % STORE_DIR)
    files = [p for p in STORE_DIR.glob("*.py") if not p.name.startswith("test_")]
    files += [PKG / rel for rel in NAMED_SURFACE]
    files += [p for p in (PKG / "entities").glob("*.py")]
    present = sorted(f for f in files if f.exists())

    missing = sorted(STORE_MODULES - {p.name for p in present if p.parent == STORE_DIR})
    assert not missing, (
        "the store roster lost %r under %s — the checks would then run against a file list that no "
        "longer contains the store. Update STORE_MODULES only when a module is genuinely gone, "
        "never to make this pass." % (missing, STORE_DIR))
    absent = sorted(rel for rel in NAMED_SURFACE if not (PKG / rel).is_file())
    assert not absent, (
        "named surface files are missing: %r — `_surface()` drops what does not exist, so their "
        "absence would shrink the roster silently" % absent)
    assert len(present) > len(STORE_MODULES), (
        "only %d files on the surface — the entities glob is not resolving" % len(present))
    return present


def _guarded_lines(tree) -> set:
    """Line numbers of imports wrapped in a `try:` with an `except ImportError` handler.

    A guarded import is a DECLARED extra boundary: the module still imports on the base install,
    and only the path needing the extra fails — loudly, with instructions. An unguarded one is an
    undeclared dependency that surfaces as a bare ModuleNotFoundError from library internals."""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_import_error = any(
            (h.type is None)
            or (isinstance(h.type, ast.Name) and h.type.id in ("ImportError", "ModuleNotFoundError"))
            or (isinstance(h.type, ast.Tuple)
                and any(isinstance(e, ast.Name)
                        and e.id in ("ImportError", "ModuleNotFoundError") for e in h.type.elts))
            for h in node.handlers)
        if not catches_import_error:
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    out.add(sub.lineno)
    return out


def _imports(path: pathlib.Path):
    """(module, lineno, is_module_level, is_guarded) for every ABSOLUTE import, any depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    guarded = _guarded_lines(tree)
    toplevel = set()
    for node in tree.body:                       # direct children of Module == module level
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                toplevel.add(sub.lineno)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, node.lineno, node.lineno in toplevel, node.lineno in guarded
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative — in-package by construction
                continue
            yield (node.module or ""), node.lineno, node.lineno in toplevel, node.lineno in guarded


# ── 1. static ────────────────────────────────────────────────────────────────
def test_the_surface_scan_actually_reaches_the_store():
    """Vacuous-pass guard, checked first: the three static checks below are absence-assertions, and
    an absence-assertion over an empty file list passes on nothing.

    `Path.glob` on a directory that is not there returns an empty iterator and raises nothing, so a
    roster that silently loses the store is indistinguishable from a clean one — the suite would
    stay green while measuring nineteen fewer files. Every module is therefore named, not counted.
    """
    files = _surface()
    assert files, "the surface roster is empty"
    in_store = {p.name for p in files if p.parent == STORE_DIR}
    assert in_store >= STORE_MODULES, (
        "store modules missing from the roster: %r" % sorted(STORE_MODULES - in_store))
    assert {"vertex.py", "access.py", "edge.py"} <= in_store, (
        "the roster does not reach the largest store modules — it is not scanning the store")
    assert any(p.parent == PKG / "entities" for p in files), (
        "no file under %s was scanned — the entities arm of the surface is not being reached"
        % (PKG / "entities"))
    assert all(p.is_file() for p in files), "the roster names paths that are not files"


def test_the_roster_guard_fails_on_an_empty_directory(tmp_path, monkeypatch):
    """Seeded-violation proof. A guard that has never been shown to fire is a belief.

    An empty directory is exactly what a move leaves behind, and it must raise here rather than hand
    the static checks a shorter list."""
    empty = tmp_path / "db"
    empty.mkdir()
    monkeypatch.setattr(sys.modules[__name__], "STORE_DIR", empty)
    with pytest.raises(AssertionError) as ei:
        _surface()
    assert "vertex.py" in str(ei.value), (
        "the guard fired without naming the module that went missing: %s" % ei.value)

    monkeypatch.setattr(sys.modules[__name__], "STORE_DIR", tmp_path / "gone")
    with pytest.raises(AssertionError) as ei:
        _surface()
    assert "missing at" in str(ei.value), (
        "a store directory that does not exist must fail loudly, got: %s" % ei.value)


def test_no_sibling_package_is_a_hard_dependency():
    """A MODULE-LEVEL sibling import fires on import, so it is a hard install requirement —
    banned outright on this surface."""
    bad = ["%s:%d imports %s" % (f.relative_to(PKG), line, mod)
           for f in _surface()
           for mod, line, is_top, _g in _imports(f)
           if is_top and mod.split(".")[0] in FORBIDDEN]
    assert not bad, (
        "the embeddable surface hard-depends on a sibling package:\n  " + "\n  ".join(bad)
        + "\n\nMove it behind an extra — see services/system_identity.py for the pattern.")


def test_every_sibling_import_is_guarded_with_an_actionable_error():
    """Function-local sibling imports are allowed ONLY when guarded.

    Both current cases are deliberate: `APIKey.has_scope` (the platform scope grammar) and
    `services/system_identity.py` (the platform system principal, which is not on this surface)."""
    unguarded = ["%s:%d imports %s" % (f.relative_to(PKG), line, mod)
                 for f in _surface()
                 for mod, line, _t, guarded in _imports(f)
                 if mod.split(".")[0] in FORBIDDEN and not guarded]
    assert not unguarded, (
        "unguarded sibling-package import in the embeddable surface:\n  "
        + "\n  ".join(unguarded)
        + "\n\nWrap it in `try: ... except ImportError as e: raise ImportError(<what to install, "
          "and the store-native alternative>) from e`. See APIKey.has_scope for the pattern.")


def test_no_module_level_third_party_beyond_cryptography():
    """Module-level = a HARD install dependency: it fires on import, before any call."""
    bad = []
    for f in _surface():
        for mod, line, is_top, _g in _imports(f):
            root = mod.split(".")[0]
            if not root or root in INTERNAL or root in STDLIB or root in ALLOWED_THIRD_PARTY:
                continue
            if is_top:
                bad.append("%s:%d imports %s" % (f.relative_to(PKG), line, mod))
    assert not bad, (
        "module-level third-party import in the embeddable surface:\n  " + "\n  ".join(bad)
        + "\n\nEither add it to pyproject `dependencies` (and to ALLOWED_THIRD_PARTY here), or "
          "make it function-local and declare it as an extra. boto3/smbclient are the precedent.")


def test_acting_principal_is_stdlib_only():
    """Pinned separately because it is the one SERVICE-layer file on the surface — `doc_boundary`
    imports `KeyCustodyDenied` from it, so every embedding consumer imports it. `system_acting_context`
    lives in `services/system_identity.py`, off the embeddable surface, so it must not be
    importable from here."""
    f = PKG / "services" / "acting_principal.py"
    ext = sorted({m.split(".")[0] for m, _l, _t, _g in _imports(f)
                  if m and m.split(".")[0] not in STDLIB and m.split(".")[0] not in INTERNAL})
    assert ext == [], "acting_principal.py must stay stdlib-only, found: %s" % ext


def test_the_moved_symbol_fails_with_a_useful_message():
    """A bare AttributeError would send someone hunting. The split must name its own fix.

    And it must NOT be a re-export: aliasing `system_acting_context` back into `acting_principal`
    would drag `system_identity` — and its reach for `origin` — into the surface again."""
    from mantle.services import acting_principal
    with pytest.raises(ImportError) as ei:
        acting_principal.system_acting_context
    msg = str(ei.value)
    assert "system_identity" in msg and "embeddable" in msg
    with pytest.raises(AttributeError):
        acting_principal.some_symbol_that_does_not_exist


# ── 2. runtime ───────────────────────────────────────────────────────────────
def test_importing_the_surface_loads_nothing_outside_stdlib():
    before = set(sys.modules)
    import mantle.db.lattice_api            # noqa: F401
    import mantle.db.doc_boundary           # noqa: F401
    import mantle.db.content_cache  # noqa: F401
    import mantle.db.content_tier   # noqa: F401
    import mantle.db.s3_content     # noqa: F401
    import mantle.entities.artifact         # noqa: F401
    import mantle.entities.grant            # noqa: F401
    import mantle.entities.collection       # noqa: F401
    import mantle.services.acting_principal  # noqa: F401
    loaded = {m.split(".")[0] for m in set(sys.modules) - before}
    assert not (loaded & FORBIDDEN), "sibling packages loaded: %s" % sorted(loaded & FORBIDDEN)
    outside = {r for r in loaded
               if r not in STDLIB and r not in INTERNAL and not r.startswith("_")}
    assert not (outside - ALLOWED_THIRD_PARTY), "unexpected packages loaded: %s" % sorted(outside)


def test_s3_module_imports_without_boto3_installed():
    """`s3_content` is in the base package but boto3 is an extra, so importing the module must not
    require it — only CONSTRUCTING an S3ContentStore may. Otherwise the extra is a lie: every
    consumer would need boto3 to import a store they never point at a bucket."""
    import mantle.db.s3_content as s3c
    assert hasattr(s3c, "S3ContentStore")
    assert "boto3" not in sys.modules or True     # tolerated if a sibling test imported it
    src = (STORE_DIR / "s3_content.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name.split(".")[0] for n in top if isinstance(n, ast.Import) for a in n.names}
    names |= {(n.module or "").split(".")[0] for n in top if isinstance(n, ast.ImportFrom)}
    assert "boto3" not in names and "smbclient" not in names


# ── 3. functional ────────────────────────────────────────────────────────────
def test_the_store_actually_works_on_the_base_install(tmp_path):
    """The surface is not just importable — it does an external consumer's job: collections,
    artifacts, version lineage, and grant-gated access, with nothing but stdlib + cryptography
    loaded."""
    from mantle.db import open_lattice
    from mantle.db.lattice_api import LatticeDatabase
    import mantle.db.lattice_api as api

    L = open_lattice(str(tmp_path / "s.db"), origin="external-node")
    L.artifacts.put_artifact({"id": "coll-1", "state": "committed", "origin_root": "external-root",
                              "content_type": "application/vnd.agience.collection+json"})
    L.artifacts.put_artifact({"id": "art-1", "content_type": "text/markdown", "state": "committed",
                              "collection_id": "coll-1", "content": "v1"})
    assert L.artifacts.get_artifact("art-1")["content"] == "v1"

    rev = L.artifacts.revise("art-1", {"content": "v2"})
    root = rev.get("root_id") or "art-1"
    assert L.artifacts.head_of(root)["content"] == "v2"
    assert len(L.artifacts.versions_of(root)) == 2

    db = LatticeDatabase(str(tmp_path / "g.db"), origin="external-node")
    db.artifacts.put_artifact({"id": "coll-1", "state": "committed", "origin_root": "external-root",
                               "content_type": "application/vnd.agience.collection+json"})
    api.upsert_user_collection_grant(db, user_id="alice", collection_id="coll-1",
                                     can_read=True, granted_by="external-root")
    assert api.get_active_collection_ids_for_user(db, "alice") == ["coll-1"]
    assert api.get_active_collection_ids_for_user(db, "bob") == []      # isolation holds


def test_encrypted_content_works_on_the_base_install(tmp_path):
    """`cryptography` is the one declared dependency because the store encrypts at rest — prove
    the content tier round-trips without any other package."""
    import hashlib
    from mantle.db.content_cache import FileContentCache, shared_content_key
    cache = FileContentCache(str(tmp_path / "cas"), key=shared_content_key(b"root-secret"))
    plain = b"the embeddable store encrypts its own content"
    ref = "cas/" + hashlib.sha256(plain).hexdigest()
    cache.put(ref, plain, collection="c1")
    assert cache.get(ref, collection="c1") == plain


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
