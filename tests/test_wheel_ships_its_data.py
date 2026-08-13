# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
# ---------------------------------------------------------------------------

"""The built wheel is opened and read, because the failure this guards is silent.

`pyproject.toml`'s `[tool.setuptools.package-data]` globs are a DECLARATION, and a declaration is
not evidence. The specific way it goes wrong here has no symptom at build time and no symptom at
install time:

    a glob under `mantle = [...]` matches files directly in `src/mantle/`. It does not descend
    into subpackages. `src/mantle/system/` has its own `__init__.py`, so it is a package of its
    own, and `uvicorn_log_config.json` — which `system/logging_utils.py` loads as
    `Path(__file__).parent / "uvicorn_log_config.json"` — needs its own entry or it is simply
    left out. A wheel built without it BUILDS, INSTALLS AND RUNS, with default log formatting
    and nothing to say why.

No check that reads the source tree can see that, and no check that reads `pyproject.toml` can
see it either — the declaration would have to be interpreted, which is exactly the step that got
it wrong. So this file builds a real wheel with the real backend and reads the archive.

## How the wheel is built

The source tree is COPIED to a temporary directory and built there. Building in place would
create `build/lib/` and `build/bdist.*/` inside `build/`, which in this repo is a tracked
directory holding the Dockerfile — a test that leaves artifacts beside a checked-in file is a test
that eventually gets blamed for a dirty tree.

The PEP 517 backend is called directly rather than through `python -m build`, which would
construct an isolated virtualenv and fetch its build requirements from an index. `setuptools` is
this project's declared backend and is present wherever the suite runs, so calling it is both
faster and offline.

## The control

Every assertion below is of the form "X is in the archive". If `X` had simply been deleted from
the source tree, those assertions would fail for a reason that has nothing to do with packaging
and the failure message would send the reader to the wrong file. So each asset is confirmed to
exist in the tree first, and the archive is confirmed to be a real wheel (it contains modules)
before its contents are judged.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

import pytest

#: `<repo>/tests/` → parents[1] is the repo root. Asserted rather than trusted: a wrong depth
#: here would copy the wrong tree and fail somewhere unrecognisable.
_REPO = Path(__file__).resolve().parents[1]
assert (_REPO / "pyproject.toml").is_file(), (
    "path depth is wrong: parents[1] should be the agience-mantle repo root, got %s" % _REPO)

#: What a build needs from the tree. `src/` is the package; the rest are files `pyproject.toml`
#: names (`readme`) or that setuptools picks up for the dist-info licence directory.
_BUILD_INPUTS = ("pyproject.toml", "README.md", "LICENSE", "NOTICE")

def _beside(dotted: str, filename: str) -> str:
    """The repo-relative path of `filename` sitting beside module `dotted`.

    Derived from the module's own location, never spelled out: each of these assets is loaded as
    `Path(__file__).parent / filename` by the module named here, so asking the import system where
    that module is asks the same question the loader does. A literal path would instead pin the
    package layout, and moving the reader into a subpackage would fail this file for a reason that
    has nothing to do with packaging.
    """
    spec = importlib.util.find_spec(dotted)
    assert spec and spec.origin, (
        "%s is not importable, so the asset it loads cannot be located — fix the module name here "
        "or retire this asset in pyproject.toml too" % dotted)
    found = (Path(spec.origin).resolve().parent / filename)
    assert _REPO in found.parents, (
        "%s resolves to %s, outside this checkout — the suite is reading an installed copy, so a "
        "wheel built from this tree would not be the thing under test" % (dotted, found))
    return found.relative_to(_REPO).as_posix()


#: The runtime assets whose absence is silent. Each entry is (module that loads it, filename beside
#: that module, what reads it) — the third element goes into the failure message, because the
#: reader of that message needs to know what breaks, not just what is missing.
_ASSET_READERS = (
    ("mantle.system.logging_utils", "uvicorn_log_config.json",
     "system/logging_utils.py reads it as Path(__file__).parent / 'uvicorn_log_config.json'"),
)

#: The same three facts as paths: (path inside the wheel, path in the source tree, what reads it).
#: `src/` is what setuptools strips, so the wheel path is the tree path with that prefix removed.
_ASSETS = tuple(
    (_beside(dotted, name).split("src/", 1)[1], _beside(dotted, name), read_by)
    for dotted, name, read_by in _ASSET_READERS
)


def _ignore(_dir: str, names: list) -> set:
    """Skip caches and previous build output — they are large, and copying them into the build
    directory is how a stale artifact ends up inside a fresh wheel."""
    return {n for n in names if n in ("__pycache__", ".pytest_cache", "build", "dist")
            or n.endswith((".pyc", ".egg-info"))}


@lru_cache(maxsize=1)
def _wheel() -> Path:
    """Build one wheel for the whole module and return the path to it.

    Cached because the build is the expensive part and every test below asks about the same
    archive. The temporary tree is deliberately NOT cleaned up inside the session — the wheel is
    read from it by later tests — and lives under the OS temp directory, so it is not the repo's
    problem.
    """
    root = Path(tempfile.mkdtemp(prefix="mantle-wheel-"))
    src = root / "src-tree"
    out = root / "wheelhouse"
    out.mkdir()
    src.mkdir()

    for name in _BUILD_INPUTS:
        origin = _REPO / name
        if origin.is_file():
            shutil.copy2(origin, src / name)
    shutil.copytree(_REPO / "src", src / "src", ignore=_ignore)

    # The backend, in a subprocess with cwd inside the copy. In-process would mean chdir'ing the
    # whole test session and importing a build backend into it.
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, setuptools.build_meta as b; print(b.build_wheel(sys.argv[1]))",
         str(out)],
        cwd=src, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        "the wheel did not build, so nothing below is a statement about packaging:\n"
        f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")

    wheels = sorted(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {[w.name for w in wheels]}"
    return wheels[0]


@lru_cache(maxsize=1)
def _names() -> tuple:
    with zipfile.ZipFile(_wheel()) as z:
        return tuple(z.namelist())


def test_the_wheel_is_a_real_wheel() -> None:
    """The control for everything else. An empty or malformed archive would make every
    "X is present" assertion below fail with a message about X, when the real fact is that the
    build produced nothing worth inspecting."""
    names = _names()
    assert "mantle/main.py" in names, (
        f"the built wheel does not contain mantle/main.py — it is not a mantle wheel: "
        f"{sorted(names)[:20]}")
    assert sum(1 for n in names if n.endswith(".py")) > 100, (
        f"the wheel contains only {sum(1 for n in names if n.endswith('.py'))} modules; the "
        "package is ~190, so the build captured a fraction of the tree")


@pytest.mark.parametrize("in_wheel,in_tree,read_by", _ASSETS,
                         ids=[a[0].rsplit("/", 1)[-1] for a in _ASSETS])
def test_a_runtime_asset_is_inside_the_wheel(in_wheel: str, in_tree: str, read_by: str) -> None:
    """Each asset exists in the tree first, then in the archive.

    Order matters: without the tree check, deleting the source file would fail this test with a
    packaging message and send the reader to `pyproject.toml` for a problem that is not there."""
    source = _REPO / in_tree
    assert source.is_file(), (
        f"{in_tree} is not in the source tree, so its absence from the wheel is not a packaging "
        "finding — fix the tree, or retire this asset here and in pyproject.toml")

    assert in_wheel in _names(), (
        f"{in_wheel} is MISSING from the built wheel. It is loaded at runtime by {read_by}, and "
        "its absence is silent: the wheel installs and the service starts, just without it. Add "
        f"the package that owns it to [tool.setuptools.package-data] — a glob on a parent package "
        "does not descend into a subpackage.")

    with zipfile.ZipFile(_wheel()) as z:
        assert z.getinfo(in_wheel).file_size > 0, f"{in_wheel} shipped empty"


def test_the_console_scripts_are_declared_in_the_wheel() -> None:
    """`pip install agience-mantle[service]` must yield a command.

    The entry points live in `[project.scripts]`, and the only place they are observable as
    something a consumer gets is the built dist-info — which is what this reads."""
    with zipfile.ZipFile(_wheel()) as z:
        entry_points = [n for n in z.namelist() if n.endswith("dist-info/entry_points.txt")]
        assert entry_points, (
            "the wheel declares no entry points, so an install yields no runnable command and the "
            "consumer must know `uvicorn mantle.main:app` to start the service")
        text = z.read(entry_points[0]).decode()

    # `mantle-token` belongs in this list for the same reason the other two do, and for one more:
    # an installed node that boots and answers 401 to everything is indistinguishable from a broken
    # one, and this is the only thing in the distribution that mints a credential it accepts.
    for command, target in (("mantle-serve", "mantle.scripts.serve:main"),
                            ("mantle-init-keys", "mantle.scripts.dev_init_keys:main"),
                            ("mantle-token", "mantle.scripts.dev_mint_token:main")):
        assert f"{command} = {target}" in text, (
            f"`{command}` is not in the wheel's entry points:\n{text}")


#: Non-`.py` files under `src/mantle/` that are deliberately NOT in the wheel, each with the
#: reason. An allowlist rather than a pattern: the whole failure this module guards is a
#: declaration that looked right, and "everything except what we listed, with a reason" is the one
#: shape that cannot quietly grow a hole.
_NOT_SHIPPED = {
    "requirements.txt": "the runtime install list for build/Dockerfile — an image input, not a "
                        "package resource; the wheel's dependencies come from [project] instead",
}


def test_every_non_python_file_is_shipped_or_explicitly_excused() -> None:
    """The GENERAL form, so the next one is caught before a release.

    `_ASSETS` above names the runtime assets we know about, which means it can only ever
    re-detect a regression in those. The bug it exists for is a file that exists in the tree
    with nobody having added it anywhere — and the wheel that omits it builds, installs and
    runs.

    So this asserts over the tree instead of over a list: every non-`.py` file under `src/mantle/`
    is either inside the wheel or named in `_NOT_SHIPPED` with a reason. Adding a data file and
    forgetting `[tool.setuptools.package-data]` fails here, at the commit that adds it, naming the
    file — rather than silently at a consumer's install.
    """
    skip_dirs = {"__pycache__", ".pytest_cache"}
    tree = {
        p.relative_to(_REPO / "src" / "mantle").as_posix()
        for p in (_REPO / "src" / "mantle").rglob("*")
        if p.is_file() and p.suffix != ".py"
        and not any(part in skip_dirs or part.endswith(".egg-info") for part in p.parts)
    }
    assert tree, "found no non-.py files at all under src/mantle — the walk is wrong, not the tree"

    names = _names()
    missing = sorted(
        rel for rel in tree
        if rel not in _NOT_SHIPPED and f"mantle/{rel}" not in names
    )
    assert not missing, (
        "these non-.py files live under src/mantle/ but are NOT in the built wheel:\n"
        + "\n".join("    " + m for m in missing)
        + "\n\nEither add the owning package to [tool.setuptools.package-data] in pyproject.toml "
          "(a glob on a parent package does NOT descend into a subpackage — each subpackage needs "
          "its own entry), or, if the file is genuinely not a package resource, add it to "
          "_NOT_SHIPPED in this file with the reason. Do not leave it undecided: a runtime asset "
          "missing from a wheel fails SILENTLY — the install works and the feature is just empty.")

    # The allowlist is held to the same standard: an entry for a file that no longer exists is a
    # stale excuse that would silently cover a future file of the same name.
    stale = sorted(set(_NOT_SHIPPED) - tree)
    assert not stale, (
        f"_NOT_SHIPPED excuses files that are no longer in the tree: {stale}. Remove them.")


def test_the_package_data_globs_name_only_patterns_that_match() -> None:
    """A glob that matches nothing is a claim that such files exist.

    Read in BOTH directions, because a package-data table can be wrong either way and the two
    failures look nothing alike:

    · A declared pattern that matches no file in the tree. It ships nothing, and it reads to the
      next person as evidence that this package carries data — so a real file added later looks
      already covered when the pattern's package no longer holds anything. Every pattern in the
      table is resolved against the tree here.

    · A file in the tree that no pattern names. `*.sql` is the standing example: the lattice
      schema is created by inline Python on open, and there has never been a `.sql` file to ship,
      so there is no `*.sql` entry. If one is ever added, the omission surfaces here rather than
      at a customer's install.
    """
    import tomllib

    with (_REPO / "pyproject.toml").open("rb") as fh:
        package_data = tomllib.load(fh)["tool"]["setuptools"]["package-data"]
    assert package_data, (
        "[tool.setuptools.package-data] is empty or absent — either the table moved, in which "
        "case fix the path read here, or the declaration was dropped and every runtime asset is "
        "now missing from the wheel silently")

    empty = []
    for package, patterns in package_data.items():
        pkg_dir = _REPO / "src" / Path(*package.split("."))
        assert pkg_dir.is_dir(), (
            f"[tool.setuptools.package-data] names `{package}`, which is not a package in this "
            f"tree ({pkg_dir} does not exist). setuptools does not error on this; it just ships "
            "nothing. Remove the entry or fix the package name.")
        empty += [f"{package} = {p!r}" for p in patterns if not list(pkg_dir.glob(p))]
    assert not empty, (
        "these [tool.setuptools.package-data] patterns match no file in the tree:\n"
        + "\n".join("    " + e for e in empty)
        + "\n\nA pattern that ships nothing is a claim that such files exist. Remove it, or add "
          "the file it was declared for.")

    sql = sorted(p.relative_to(_REPO).as_posix()
                 for p in (_REPO / "src").rglob("*.sql"))
    assert sql == [], (
        f"there are now .sql files in the package: {sql}. They are NOT declared in "
        "[tool.setuptools.package-data], so a built wheel omits them silently — add the pattern "
        "and an entry to _ASSETS above.")
