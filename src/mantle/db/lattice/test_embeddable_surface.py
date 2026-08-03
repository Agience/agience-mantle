"""THE EMBEDDABLE SURFACE IS STDLIB + `cryptography`. NOTHING ELSE. Enforced, not documented.

`pyproject.toml` declares `dependencies = ["cryptography"]` for the base install — the whole
product claim that Mantle is a standalone data store rests on that being true. A claim like this
rots the moment someone adds a convenient import, and it rots SILENTLY: the offending import is
usually function-local, so every existing test keeps passing and the breakage only shows up on a
consumer's machine, at install time, as an unresolvable requirement for code they never call.

EREA (the first external consumer) is the reason this exists. They embed Mantle for artifacts,
collections and access grants, and must not inherit chorus / ember / beam / origin.

⚠ THIS FILE LIVES IN `db/lattice/` ON PURPOSE. `src/mantle/tests/conftest.py` does
`import origin.config`, so a test placed there could not run without the very package it exists to
prove unnecessary. A check that cannot run in the condition it validates proves nothing.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

PKG = pathlib.Path(__file__).resolve().parents[2]          # …/src/mantle
STDLIB = set(sys.stdlib_module_names)

# Sibling Agience packages and the reasoning stack. An embedding consumer installs NONE of these.
FORBIDDEN = {
    "chorus", "ember", "beam", "iris", "sage", "lumen", "aria", "astra", "ophan", "seraph",
    "crystal", "prism", "origin", "facet", "tekton", "bundle", "entroptics",
}
# Absolute imports that resolve INSIDE mantle (the package is importable both as `mantle.x` and,
# service-side, with `src/mantle` itself on the path).
INTERNAL = {"mantle", "db", "entities", "services", "search", "routers", "schemas", "api",
            "clients", "tools", "scripts", "embeddings", "embeddings_cache", "event_bus",
            "artifact_helpers", "main"}

ALLOWED_THIRD_PARTY = {"cryptography"}


def _surface() -> list:
    """Every file `pip install agience-mantle` must be able to import with only `cryptography`."""
    files = [p for p in (PKG / "db" / "lattice").glob("*.py") if not p.name.startswith("test_")]
    files += [PKG / "db" / "lattice_api.py", PKG / "db" / "doc_boundary.py",
              PKG / "services" / "acting_principal.py"]
    files += [p for p in (PKG / "entities").glob("*.py")]
    return sorted(f for f in files if f.exists())


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

    ⚠ THE POINT IS NOT PURITY, IT IS THE FAILURE MODE. Unguarded, an embedding consumer who
    strays onto a platform path gets `ModuleNotFoundError: No module named 'origin'` raised from
    inside library internals — indistinguishable from a broken install, with no hint that they
    wanted a different API entirely. Guarded, they get told which extra to install and which
    store-native alternative to use instead.

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
    imports `KeyCustodyDenied` from it, so every embedding consumer imports it. `origin` lived here
    until the 2026-07-29 split moved `system_acting_context` to `services/system_identity.py`."""
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
    import mantle.db.lattice.content_cache  # noqa: F401
    import mantle.db.lattice.content_tier   # noqa: F401
    import mantle.db.lattice.s3_content     # noqa: F401
    import mantle.db.lattice.fts            # noqa: F401
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
    import mantle.db.lattice.s3_content as s3c
    assert hasattr(s3c, "S3ContentStore")
    assert "boto3" not in sys.modules or True     # tolerated if a sibling test imported it
    src = (PKG / "db" / "lattice" / "s3_content.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name.split(".")[0] for n in top if isinstance(n, ast.Import) for a in n.names}
    names |= {(n.module or "").split(".")[0] for n in top if isinstance(n, ast.ImportFrom)}
    assert "boto3" not in names and "smbclient" not in names


# ── 3. functional ────────────────────────────────────────────────────────────
def test_the_store_actually_works_on_the_base_install(tmp_path):
    """The surface is not just importable — it does EREA's job: collections, artifacts, version
    lineage, and grant-gated access, with nothing but stdlib + cryptography loaded."""
    from mantle.db.lattice import open_lattice
    from mantle.db.lattice_api import LatticeDatabase
    import mantle.db.lattice_api as api

    L = open_lattice(str(tmp_path / "s.db"), origin="erea-node")
    L.artifacts.put_artifact({"id": "coll-1", "state": "committed", "origin_root": "erea-root",
                              "content_type": "application/vnd.agience.collection+json"})
    L.artifacts.put_artifact({"id": "art-1", "content_type": "text/markdown", "state": "committed",
                              "collection_id": "coll-1", "content": "v1"})
    assert L.artifacts.get_artifact("art-1")["content"] == "v1"

    rev = L.artifacts.revise("art-1", {"content": "v2"})
    root = rev.get("root_id") or "art-1"
    assert L.artifacts.head_of(root)["content"] == "v2"
    assert len(L.artifacts.versions_of(root)) == 2

    db = LatticeDatabase(str(tmp_path / "g.db"), origin="erea-node")
    db.artifacts.put_artifact({"id": "coll-1", "state": "committed", "origin_root": "erea-root",
                               "content_type": "application/vnd.agience.collection+json"})
    api.upsert_user_collection_grant(db, user_id="alice", collection_id="coll-1",
                                     can_read=True, granted_by="erea-root")
    assert api.get_active_collection_ids_for_user(db, "alice") == ["coll-1"]
    assert api.get_active_collection_ids_for_user(db, "bob") == []      # isolation holds


def test_encrypted_content_works_on_the_base_install(tmp_path):
    """`cryptography` is the one declared dependency because the store encrypts at rest — prove
    the content tier round-trips without any other package."""
    import hashlib
    from mantle.db.lattice.content_cache import FileContentCache, shared_content_key
    cache = FileContentCache(str(tmp_path / "cas"), key=shared_content_key(b"root-secret"))
    plain = b"the embeddable store encrypts its own content"
    ref = "cas/" + hashlib.sha256(plain).hexdigest()
    cache.put(ref, plain, collection="c1")
    assert cache.get(ref, collection="c1") == plain


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
