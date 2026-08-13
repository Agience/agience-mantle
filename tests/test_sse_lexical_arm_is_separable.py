"""The encrypted lexical arm must import without pulling in the vector arm or key custody.

`sse/` — blind tokens, encrypted posting lists, blind-token narrowing — is the retrieval story for an embedded
store. It is blind: the index holds HMAC'd tokens and encrypted postings, never plaintext terms, so
the server cannot read the vocabulary of a collection or confirm whether any given term appears in
it — which is what makes it acceptable for customer data.

Most `sse/` modules depend on nothing beyond stdlib + `cryptography`. Importing the
lexical core must not import:

  * `search/mantle/__init__.py`'s `.engine`/`.indexer`/`.lightcone`/`.oracle` — importing
    `…sse.tokenizer` would otherwise execute it first and drag in numpy, embeddings and `services`;
  * `sse/__init__.py`'s `.router_accessor` and `.unified` — the vector-arm and custody integration
    a lexical-only consumer does not want.

`indexer`/`narrowing` depend on `sse.keys.SseKeyProvider` for their one method call rather than the
full `OracleService` (grant verifier + lattice master-key store + CRUDEASIO mint policy).
`narrowing` additionally names one custody class — the refusal it catches, which an `except` matches
by class object rather than by name — and it takes that class from `..custody`, a module holding
those two names and nothing else. So the statement here is unconditional: every module in
`LEXICAL_CORE` reaches stdlib, `cryptography`, and other `mantle` modules only. There is no
per-module exemption list, because there is no module needing one.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

SRC = str(pathlib.Path(__file__).resolve().parents[1] / "src")   # …/agience-mantle/src

# The modules an embedding consumer takes. `router_accessor` and `unified` are deliberately absent —
# they ARE the vector-arm/custody integration, and a lexical-only consumer does not want them ("a
# data store does no reasoning").
LEXICAL_CORE = [
    "keys", "tokenizer", "blind_tokens", "posting", "s3_stores", "file_stores",
    "indexer", "narrowing",
]

# cryptography drags its own C extension in; that is the one declared dependency.
ALLOWED_ROOTS = {"cryptography", "_cffi_backend", "_openssl", "cffi", "pycparser"}


def _run(body: str, *, service_path: bool = False) -> subprocess.CompletedProcess:
    """Execute `body` in a clean interpreter with only <mantle>/src importable.

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
    template = textwrap.dedent("""
        import sys
        # Baseline first: measuring all of sys.modules would count interpreter-startup noise —
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

    # Each module measured ALONE as well as together: importing the set at once lets one clean
    # module's dependency-free reading be produced by another having already loaded what it
    # needed. Per-module is the reading that would catch a single new edge.
    for module in LEXICAL_CORE:
        r = _run(template % ("import mantle.search.mantle.sse.%s" % module, ALLOWED_ROOTS))
        assert r.returncode == 0, (
            "sse/%s.py does not import on a bare <mantle>/src path:\n" % module + r.stderr)
        assert "FORBIDDEN=[]" in r.stdout, (
            "sse/%s.py reaches a custody or vector-arm module:\n%s" % (module, r.stdout))
        assert "ROOTS=[]" in r.stdout, (
            "sse/%s.py gained a third-party dependency beyond cryptography:\n" % module
            + r.stdout)


#: The two files that exist to BE a seam, and so may not carry one themselves. `sse/keys.py` is
#: the key contract; `search/mantle/custody.py` is the refusal vocabulary the custodian raises and
#: the arm catches. A dependency in either is a dependency in everything on both sides of it.
_SEAM_FILES = (
    ("mantle", "search", "mantle", "sse", "keys.py"),
    ("mantle", "search", "mantle", "custody.py"),
)


@pytest.mark.parametrize("parts", _SEAM_FILES, ids=lambda p: p[-1])
def test_the_seam_files_are_stdlib_only(parts):
    """A seam file decides whether the arm can travel, so it takes nothing — not `cryptography`,
    not a sibling package. The file's own import list is the thing under test.

    `custody.py` is held to this because it is what makes the arm's one custody name free: the
    whole point of moving the refusals out of `oracle` is that catching them costs no dependency,
    and an import added here would silently reintroduce exactly the coupling it removed.
    """
    import ast
    src = pathlib.Path(SRC).joinpath(*parts)
    tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and not node.level:
            roots.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level:
            roots.add("<relative:%s>" % (node.module or ""))
    external = sorted(roots - set(sys.stdlib_module_names) - {""})
    assert external == [], (
        "%s must stay stdlib-only — it is a file that decides whether the lexical arm can be "
        "extracted. Found: %s" % ("/".join(parts), external))


def test_master_key_exceptions_are_one_class_not_two():
    """`custody` defines the refusals once; every consumer catches that same class OBJECT.

    `pipeline_unified.py` and `test_key_custody_bypasses.py` both `except MasterKeyMissing`, and
    `sse/narrowing.py` catches it around the provider call that raises it. Two same-named classes would
    stop matching each other — silently, and only on the failure path, which is the worst place to
    find out. Identity is therefore the assertion, and a sweep of every loaded module is what
    turns "one definition" into a measurement rather than a claim.

    The definitions live in `search/mantle/custody.py`, which imports nothing, rather than in
    `oracle` beside the raises. Both give one class object per name; only the first lets the
    lexical arm name the class without importing the custodian, which is why `HOME` is asserted
    and not merely `SAME` — `oracle` re-exports these, so identity alone would still hold if they
    moved back."""
    r = _run("""
        import sys
        import mantle.search.mantle.custody as C
        import mantle.search.mantle.oracle as O
        import mantle.search.mantle.sse.narrowing as N
        print("SAME=%s" % (N.MasterKeyMissing is O.MasterKeyMissing is C.MasterKeyMissing))
        print("HOME=%s" % (C.MasterKeyMissing.__module__,))
        print("HIER=%s" % (issubclass(O.MasterKeyMissing, O.MasterKeyUnavailable)
                           and issubclass(O.MasterKeyUnavailable, RuntimeError)))
        # Any module holding a DIFFERENT object under either name is a second definition.
        dupes = sorted(
            name for name, mod in list(sys.modules.items())
            if mod is not None and any(
                getattr(mod, attr, klass) is not klass
                for attr, klass in (("MasterKeyMissing", O.MasterKeyMissing),
                                    ("MasterKeyUnavailable", O.MasterKeyUnavailable))
            )
        )
        print("DUPES=%s" % dupes)
    """, service_path=True)
    assert r.returncode == 0, r.stderr
    assert "SAME=True" in r.stdout, "MasterKeyMissing was redefined, not imported:\n" + r.stdout
    assert "HOME=mantle.search.mantle.custody" in r.stdout, (
        "the refusals moved back into a module that carries other things; the lexical arm has to "
        "import whatever that module imports in order to catch them:\n" + r.stdout)
    assert "HIER=True" in r.stdout, "the exception hierarchy changed:\n" + r.stdout
    assert "DUPES=[]" in r.stdout, (
        "a second class object is live under one of these names:\n" + r.stdout)

    # The ingest catcher, in this interpreter: it must catch what the oracle raises.
    from mantle.search.ingest import pipeline_unified
    from mantle.search.mantle import oracle

    assert pipeline_unified.MasterKeyMissing is oracle.MasterKeyMissing


def test_the_lazy_init_still_exports_the_entangled_names():
    """Making `router_accessor` lazy must not remove it from the public API — several call sites
    do `from search.mantle.sse import MantleSseSearchAccessor`."""
    r = _run("""
        import mantle.search.mantle.sse as sse
        names = ["MantleSseSearchAccessor", "SseIndexer", "TokenNarrower", "Coverage"]
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
