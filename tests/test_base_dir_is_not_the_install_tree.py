"""`BASE_DIR` is the root every derived default hangs off, and it must not be site-packages.

`KEYS_DIR`, `MANTLE_LATTICE_PATH`, `MANTLE_SSE_DIR`, `MANTLE_CELL_DIR` and the embeddings cache all
default to `<BASE_DIR>/.data/…`. Derived from the directory the `mantle` package sits in, that is
`<venv>/Lib/site-packages` on an installed node — a tree the next `pip install --upgrade` rewrites,
holding indexes the README calls data rather than cache and whose rebuild is measured in
days-to-weeks.

The question `_derive_base_dir` asks is not "which install shape" but "is this directory mine to
write in", and the shapes answer differently: a checkout and a PEP 660 editable install own a tree
beside the source, a wheel install does not, a zip import has no directory at all, and a frozen
build's is deleted on exit. The negative cases are the substance — a rule that only ever answered
"the repo root" would pass a checkout-only test while being exactly the defect.
"""
from __future__ import annotations

import importlib
import sys
import sysconfig
from pathlib import Path

import mantle.config as cfg


# ── a source tree is still a source tree ────────────────────────────────────────────────────────

def test_a_src_layout_checkout_still_derives_the_repo_root(tmp_path):
    """Nothing about developing here changes: `.data/` stays beside `src/`."""
    root = tmp_path / "agience-mantle" / "src"
    root.mkdir(parents=True)
    assert cfg._derive_base_dir(root) == root.parent


def test_a_source_tree_that_is_not_src_layout_derives_itself(tmp_path):
    """The `src/` step up is a layout detail, not the rule. A flat tree is still a tree that owns
    the directory it sits in."""
    root = tmp_path / "flat-checkout"
    root.mkdir()
    assert cfg._derive_base_dir(root) == root


def test_an_editable_install_is_a_checkout_and_is_treated_as_one(tmp_path):
    """A PEP 660 editable install is the same tree seen through a finder: `__file__` resolves into
    the real `<repo>/src` on disk, which is not an install directory. Asserted against THIS repo's
    own source tree rather than a synthetic one, because "the real path is what matters" is the
    whole claim."""
    real_src = Path(cfg.__file__).resolve().parent.parent
    assert real_src.name == "src", "this repo is no longer src-layout; the premise moved"
    assert cfg._derive_base_dir(real_src) == real_src.parent


# ── the trees that are not ours to write in ─────────────────────────────────────────────────────

def test_this_interpreters_own_install_directory_is_refused(monkeypatch, tmp_path):
    """The substance. `purelib` is where pip is TOLD to put things — asked of the interpreter
    rather than guessed from a name, because there is no one name: `site-packages` in a venv,
    `dist-packages` on Debian, neither under `pip install --target`."""
    monkeypatch.chdir(tmp_path)
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    assert cfg._derive_base_dir(purelib) == Path.cwd()
    assert cfg._derive_base_dir(purelib) != purelib


def test_a_directory_named_site_packages_is_refused_even_unknown_to_this_interpreter(
        monkeypatch, tmp_path):
    """The name floor, for an interpreter that answers neither `sysconfig` nor `site` — an embedded
    or `-S` build. `tmp_path` is nowhere this interpreter installs to, so only the name can catch
    it."""
    monkeypatch.chdir(tmp_path)
    for name in ("site-packages", "dist-packages"):
        root = tmp_path / "elsewhere" / name
        root.mkdir(parents=True)
        assert cfg._derive_base_dir(root) == Path.cwd()


def test_a_zip_import_has_no_directory_to_write_beside(monkeypatch, tmp_path):
    """`<…>/app.zip/mantle` resolves as a path and is not a directory. Writing an index "next to"
    it would create a directory inside nothing."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "app.zip"          # deliberately never created
    assert not root.is_dir()
    assert cfg._derive_base_dir(root) == Path.cwd()


def test_a_frozen_build_never_derives_from_its_bundle(monkeypatch, tmp_path):
    """PyInstaller unpacks to a temp directory and DELETES it on exit — the one place worse than
    site-packages. The bundle here is a perfectly good `src`-shaped directory, so only `sys.frozen`
    can tell it apart."""
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "_MEI123456" / "src"
    bundle.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert cfg._derive_base_dir(bundle) == Path.cwd()


# ── the override still outranks all of it ───────────────────────────────────────────────────────

def test_an_explicit_base_dir_wins_over_every_derivation(monkeypatch, tmp_path):
    """`AGIENCE_BASE_DIR` is how an installed node stops deriving, and the README's Quick Start
    sets it. Nothing above may reach it."""
    monkeypatch.setenv("AGIENCE_NO_DOTENV", "1")
    monkeypatch.setenv("AGIENCE_BASE_DIR", str(tmp_path / "node"))
    reloaded = importlib.reload(cfg)
    try:
        assert reloaded.BASE_DIR == (tmp_path / "node").resolve()
        assert reloaded.DEFAULT_LATTICE_PATH == (tmp_path / "node").resolve() / ".data" / "mantle-lattice.db"
    finally:
        monkeypatch.delenv("AGIENCE_BASE_DIR", raising=False)
        importlib.reload(cfg)


# ── what it all hangs off ───────────────────────────────────────────────────────────────────────

def test_no_derived_default_lands_in_an_install_tree(monkeypatch, tmp_path):
    """The defect stated end to end, in the four paths that carry it. Checking `_derive_base_dir`
    alone would pass while a consumer read `_MANTLE_ROOT` for itself."""
    monkeypatch.setenv("AGIENCE_NO_DOTENV", "1")
    monkeypatch.delenv("AGIENCE_BASE_DIR", raising=False)
    monkeypatch.delenv("KEYS_DIR", raising=False)
    monkeypatch.delenv("MANTLE_SSE_DIR", raising=False)
    monkeypatch.delenv("MANTLE_CELL_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    installed = Path(sysconfig.get_paths()["purelib"]).resolve()
    monkeypatch.setattr(cfg, "_MANTLE_ROOT", installed, raising=False)
    reloaded = importlib.reload(cfg)
    # `_MANTLE_ROOT` is recomputed by the reload, so pin the derivation instead of the module state.
    base = reloaded._derive_base_dir(installed)
    try:
        from mantle.search.mantle import wiring
        monkeypatch.setattr(reloaded, "BASE_DIR", base, raising=False)

        derived = [
            base / ".data" / "keys",
            base / ".data" / "mantle-lattice.db",
            base / ".data" / "mantle" / "mantle.embeddings_cache.sqlite",
            Path(wiring.local_sse_root()),
            Path(wiring.local_cell_root()),
        ]
        for path in derived:
            assert installed not in path.parents and path != installed, (
                "%s is inside the install tree — `pip install --upgrade` deletes it" % path)
            assert base in path.parents
    finally:
        importlib.reload(cfg)
