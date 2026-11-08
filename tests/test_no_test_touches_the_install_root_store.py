"""No test opens this checkout's `.data/` (the developer's own node store): the root
`conftest.py` pins every default under a temporary base, and fails the session on any open
there."""
import os
import sys


def test_the_defaults_resolve_under_the_test_base():
    from mantle import config
    from mantle.search.mantle import wiring

    _data, base, _touched = sys._mantle_install_data
    assert str(config.BASE_DIR).startswith(base)
    assert wiring.local_sse_root().startswith(base)
    assert os.environ["MANTLE_LATTICE_PATH"].startswith(base)


def test_the_detector_sees_an_open_under_the_install_root_store():
    """An audit event for a path under `.data/` is recorded (raised here without opening
    anything), and one under the test base is not."""
    data, base, touched = sys._mantle_install_data
    before = len(touched)
    sys.audit("open", os.path.join(base, ".data", "x.db"), "r", 0)
    assert len(touched) == before
    import pytest
    with pytest.raises(PermissionError):      # refused in place, in any process
        sys.audit("open", os.path.join(data.upper(), "mantle-lattice.db"), "r", 0)
    try:
        assert len(touched) == before + 1     # and recorded, compared case-insensitively
    finally:
        del touched[before:]                  # the probe itself is no violation


# -- sixth review: the ways round the guard -------------------------------------------------------

def _guard():
    import types
    return types.SimpleNamespace(under=sys._mantle_guard.under)


def test_a_sqlite_uri_naming_the_store_is_refused():
    import pytest
    data, _base, touched = sys._mantle_install_data
    before = len(touched)
    db = os.path.join(data, "mantle-lattice.db").replace(os.sep, "/")
    for uri in (f"file:{db}?mode=ro", f"file:///{db.lstrip('/')}?mode=ro&cache=shared"):
        with pytest.raises(PermissionError):
            sys.audit("sqlite3.connect", uri)
    del touched[before:]


def test_a_link_resolving_into_a_guarded_root_is_seen(tmp_path):
    """A junction (or symlink) whose target is a guarded root is the root. Tried against a root
    made for the test, so nothing under the real `.data/` is resolved."""
    import subprocess
    root = tmp_path / "guarded"
    root.mkdir()
    link = tmp_path / "innocent"
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(root)], check=True,
                       capture_output=True)
    else:
        link.symlink_to(root, target_is_directory=True)
    assert _guard().under(str(link / "x.db"), str(root))
    assert not _guard().under(str(tmp_path / "elsewhere" / "x.db"), str(root))


def test_a_subprocess_aimed_at_the_store_is_refused():
    import pytest
    data, _base, touched = sys._mantle_install_data
    before = len(touched)
    target = os.path.join(data, "mantle-lattice.db")
    with pytest.raises(PermissionError):
        sys.audit("subprocess.Popen", "sqlite3", ["sqlite3", target], None, None)
    with pytest.raises(PermissionError):
        sys.audit("subprocess.Popen", "x", ["x"], data, None)        # run in it
    del touched[before:]


def test_a_python_child_carries_the_guard(tmp_path):
    """A child interpreter inherits the guard (`sitecustomize` on its path) and refuses an open
    under a guarded root. Tried against a root made for the test."""
    import subprocess
    root = tmp_path / "guarded"
    root.mkdir()
    env = dict(os.environ)
    env["MANTLE_TEST_GUARDED"] = os.pathsep.join([env.get("MANTLE_TEST_GUARDED", ""), str(root)])
    r = subprocess.run([sys.executable, "-c",
                        f"open({str(root / 'x.db')!r}, 'w')"], env=env, capture_output=True,
                       text=True)
    assert r.returncode != 0 and "PermissionError" in r.stderr, r.stderr
    assert not (root / "x.db").exists()
