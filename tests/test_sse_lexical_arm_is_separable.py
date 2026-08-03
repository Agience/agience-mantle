"""THE ENCRYPTED LEXICAL ARM MUST IMPORT WITHOUT THE VECTOR ARM OR KEY CUSTODY. Enforced.

EREA §5 (2026-07-28) is blocked on shipping `sse/` — blind tokens, encrypted posting lists, BM25 —
as the retrieval story for an embedded store. Their alternative, `db/lattice/fts.py`, is FTS5
*contentless* but still holds PLAINTEXT term postings, which discloses the vocabulary of every
collection and confirms whether any given term appears. Not acceptable for customer data, and there
is no acceptable interim: a temporary plaintext index leaks exactly what the migration prevents.

Ten of the eleven `sse/` modules were already stdlib + `cryptography`. What pinned the arm
service-side was NOT the lexical code — it was two `__init__.py` files importing eagerly:

  * `search/mantle/__init__.py` imported `.engine`/`.indexer`/`.lightcone`/`.oracle`, so `import
    …sse.tokenizer` executed it first and dragged in numpy, embeddings and `services`;
  * `sse/__init__.py` imported `.router_accessor` and `.unified` — the vector-arm and custody
    integration EREA explicitly does not want.

Plus one real interface leak: `indexer`/`query` typed against `OracleService` (703 lines of grant
verifier + lattice master-key store + CRUDEASIO mint policy) for ONE method call. That is now
`sse.keys.SseKeyProvider`.

⚠ THIS RUNS IN A SUBPROCESS, AND THAT IS THE WHOLE POINT.
In-process, `sys.modules` is already polluted by every other test in the session — `oracle` is
certainly loaded by the time this runs. An in-process assertion would either fail spuriously or,
worse, be written as a `sys.modules` delta that passes vacuously because the module was imported
before the snapshot. A check that cannot fail proves nothing, so this pays for a clean interpreter.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

# ⚠ DEPTH FIXED 2026-07-31: the suite moved from `src/mantle/tests/` to `<repo>/tests/`, so the
# package is reached from the repo root now, not from a parent. Both of these failed with a
# FileNotFoundError naming the wrong path — loud, which is the right way for a path bug to fail.
SRC = str(pathlib.Path(__file__).resolve().parents[1] / "src")   # …/agience-mantle/src

# The modules an embedding consumer takes. `router_accessor` and `unified` are deliberately absent —
# they ARE the vector-arm/custody integration, and EREA does not want them ("a data store does no
# reasoning").
LEXICAL_CORE = [
    "keys", "tokenizer", "blind_tokens", "posting", "scorer", "stats", "s3_stores",
    "indexer", "query",
]

# cryptography drags its own C extension in; that is the one declared dependency.
ALLOWED_ROOTS = {"cryptography", "_cffi_backend", "_openssl", "cffi", "pycparser"}


def _run(body: str, *, service_path: bool = False) -> subprocess.CompletedProcess:
    """Execute `body` in a CLEAN interpreter with only <mantle>/src importable.

    `service_path=True` additionally puts `<mantle>/src/mantle` on the path — required to import
    `oracle`, which does `from services.acting_principal import …`. That absolute intra-package
    import is the SERVICE surface's two-path convention; the lexical arm deliberately needs only the
    outer path, which is the property the other tests here measure."""
    env = dict(os.environ)
    if service_path:
        # oracle/engine need the inner path (`from services...`) and `origin` (embeddings/config).
        origin_src = os.path.join(os.path.dirname(os.path.dirname(SRC)), "agience-origin", "src")
        env["PYTHONPATH"] = os.pathsep.join([SRC, os.path.join(SRC, "mantle"), origin_src])
    else:
        env["PYTHONPATH"] = SRC
    return subprocess.run([sys.executable, "-c", textwrap.dedent(body)],
                          capture_output=True, text=True, env=env, timeout=120)


def test_lexical_core_imports_without_oracle_engine_or_embeddings():
    # ⚠ DEDENT THE TEMPLATE BEFORE SUBSTITUTING. The interpolated import block is multi-line and
    # lands at column 0, so a post-substitution `dedent` finds no common prefix and strips nothing —
    # the surviving indentation is then an IndentationError in the child.
    template = textwrap.dedent("""
        import sys
        # BASELINE FIRST. Measuring all of sys.modules would count interpreter-startup noise —
        # editable-install .pth finders, _distutils_hack, pywin32 — which this code never imports.
        _base = set(sys.modules)
        %s
        loaded = set(sys.modules) - _base
        forbidden = sorted(m for m in loaded if m.endswith(
            ("oracle", "engine", "lightcone", "router_accessor", "unified", "embeddings")))
        roots = sorted({m.split(".")[0] for m in loaded}
                       - set(sys.stdlib_module_names) - {"mantle"} - %r)
        print("FORBIDDEN=%%s" %% forbidden)
        print("ROOTS=%%s" %% roots)
    """)
    imports = "\n".join("import mantle.search.mantle.sse.%s" % m for m in LEXICAL_CORE)
    r = _run(template % (imports, ALLOWED_ROOTS))
    assert r.returncode == 0, (
        "the lexical core does not import on a bare <mantle>/src path:\n" + r.stderr)
    assert "FORBIDDEN=[]" in r.stdout, (
        "the lexical arm pulled in the vector arm or key custody:\n" + r.stdout
        + "\nThis is what makes it unshippable — check for a new eager import in "
          "`search/mantle/__init__.py` or `sse/__init__.py`.")
    assert "ROOTS=[]" in r.stdout, (
        "the lexical arm gained a third-party dependency beyond cryptography:\n" + r.stdout)


def test_the_key_contract_file_is_stdlib_only():
    """`sse/keys.py` decides whether the arm can travel, so it takes nothing — not `cryptography`,
    not a sibling package.

    ⚠ CHECKED BY AST, NOT BY IMPORTING IT. Importing `…sse.keys` runs `sse/__init__.py`, which
    eagerly imports the other lexical modules — and those legitimately use `cryptography`. A runtime
    `sys.modules` check would therefore measure the PACKAGE, not this file, and report a dependency
    `keys.py` does not have. The file's own import list is the thing under test."""
    import ast
    src = pathlib.Path(SRC) / "mantle" / "search" / "mantle" / "sse" / "keys.py"
    tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and not node.level:
            roots.add((node.module or "").split(".")[0])
    external = sorted(roots - set(sys.stdlib_module_names) - {""})
    assert external == [], (
        "sse/keys.py must stay stdlib-only — it is the file that decides whether the lexical arm "
        "can be extracted. Found: %s" % external)


def test_master_key_exceptions_are_one_class_not_two():
    """`oracle` RE-EXPORTS the contract's exceptions; it must never redefine them.

    `pipeline_unified.py` and `test_key_custody_bypasses.py` both `except MasterKeyMissing` off the
    oracle path while `query.py` raises through the contract path. Two same-named classes would stop
    matching each other — silently, and only on the failure path, which is the worst place to find
    out."""
    r = _run("""
        from mantle.search.mantle.oracle import MasterKeyMissing as O, MasterKeyUnavailable as OU
        from mantle.search.mantle.sse.keys import MasterKeyMissing as K, MasterKeyUnavailable as KU
        from mantle.search.mantle.sse.query import MasterKeyMissing as Q
        print("SAME=%s" % ((O is K) and (OU is KU) and (Q is K)))
        print("HIER=%s" % (issubclass(K, KU) and issubclass(KU, RuntimeError)))
    """, service_path=True)
    assert r.returncode == 0, r.stderr
    assert "SAME=True" in r.stdout, "MasterKeyMissing was redefined, not re-exported:\n" + r.stdout
    assert "HIER=True" in r.stdout, "the exception hierarchy changed:\n" + r.stdout


def test_the_lazy_init_still_exports_the_entangled_names():
    """Making `router_accessor`/`unified` lazy must not remove them from the public API — several
    call sites do `from search.mantle.sse import MantleUnifiedAccessor`."""
    r = _run("""
        import mantle.search.mantle.sse as sse
        names = ["MantleSseSearchAccessor", "MantleUnifiedAccessor", "HitSource", "UnifiedHit",
                 "SseIndexer", "SseQueryEngine"]
        print("MISSING=%s" % [n for n in names if not hasattr(sse, n)])
        import mantle.search.mantle as sm
        top = ["OracleService", "KeyRequest", "KeyPurpose", "GrantDenied", "GrantVerifier",
               "LightConeGrantVerifier", "LightConeResolver", "MantleIndexer", "MantleQueryEngine"]
        print("MISSING_TOP=%s" % [n for n in top if not hasattr(sm, n)])
        print("ALL_OK=%s" % (sorted(sm.__all__) == sorted(top)))
    """, service_path=True)
    # These resolve the vector arm, so they need numpy etc. — skip rather than fail if a lean
    # environment lacks them; the point of the test is the NAME surface, not the vector deps.
    if r.returncode != 0 and ("numpy" in r.stderr or "No module named" in r.stderr):
        pytest.skip("vector-arm dependencies absent in this environment: %s" % r.stderr.strip()[-160:])
    assert r.returncode == 0, r.stderr
    assert "MISSING=[]" in r.stdout, "lazy export dropped a public name:\n" + r.stdout
    assert "MISSING_TOP=[]" in r.stdout, "lazy export dropped a package-root name:\n" + r.stdout
    assert "ALL_OK=True" in r.stdout, "__all__ drifted from the resolvable exports:\n" + r.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
