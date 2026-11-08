"""Every test runs against a temporary base directory, never this checkout's `.data/`.

`.data/` is the install-root store a developer's own node uses (`mantle-lattice.db`, the SSE
index, keys, the embeddings cache). The defaults that resolve there (`config.BASE_DIR`,
`MANTLE_LATTICE_PATH`, `MANTLE_SSE_DIR`) are pinned here, before anything imports `mantle`, to a
directory made for this run. An audit hook refuses, where it happens, any touch of a path under a
guarded root (`.data/`, and any root named in `MANTLE_TEST_GUARDED`), and the session fails if
there was one (`test_no_test_touches_the_install_root_store` checks the detector).

What it sees:
  * an open, a `sqlite3.connect` (a plain path or a `file:` URI), and the `os` and `shutil` calls
    that make, remove, rename, list or enter a path;
  * a path that reaches a guarded root through a junction or symlink (it is resolved);
  * a subprocess whose arguments name a guarded path, or that is started in one;
  * a child Python interpreter: `sitecustomize` (`_guard_child/`, put first on `PYTHONPATH`)
    installs the same hook there, so the child refuses too.

What it cannot see: a non-Python child that reaches a guarded path by a name not in its argument
list (read from a file, an environment variable, or built at run time); native code that opens a
file without a Python audit event (a C extension calling `fopen` directly); and a link created
after a path was checked (a check-then-use race). A link is resolved without touching its target's
contents, but resolving one does read the link itself.
"""
import os as _nb_os
_nb_os.environ.setdefault("AGIENCE_NOT_BEFORE_NO_POLL", "1")   # no Origin poller in tests
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
INSTALL_DATA = os.path.join(REPO, ".data")
TEST_BASE = tempfile.mkdtemp(prefix="mantle-test-base-")

os.environ["AGIENCE_BASE_DIR"] = TEST_BASE
os.environ["MANTLE_LATTICE_PATH"] = os.path.join(TEST_BASE, ".data", "mantle-lattice.db")
os.environ["MANTLE_SSE_DIR"] = os.path.join(TEST_BASE, ".data", "mantle-sse")
# Threshold custody is claimed only by isolated production (`services.custody_plane`); a test
# run exercises it anyway. The override is honoured only while pytest is imported.
os.environ.setdefault("MANTLE_CUSTODY_PLANE_OVERRIDE", "1")

_GUARD_DIR = os.path.join(REPO, "tests", "_guard_child")
os.environ["MANTLE_TEST_GUARDED"] = os.pathsep.join(
    p for p in (os.environ.get("MANTLE_TEST_GUARDED", ""), INSTALL_DATA) if p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (_GUARD_DIR, os.environ.get("PYTHONPATH", "")) if p)

#: (event, path) for every touch of a path under a guarded root.
TOUCHED = []

sys.path.insert(0, _GUARD_DIR)
import mantle_test_guard as _g  # noqa: E402

under = _g.under


def _record(event, path):
    TOUCHED.append((event, path))


_g.install(on_refuse=_record)
#: Where a test reads these, whatever has been imported or unloaded since.
sys._mantle_install_data = (INSTALL_DATA, TEST_BASE, TOUCHED)
sys._mantle_guard = _g


def under_install_data(path) -> bool:
    return _g.guarded(path)


def pytest_sessionfinish(session, exitstatus):
    if TOUCHED:
        seen = sorted(set(TOUCHED))
        print("\nA test touched the install-root store under .data/:", file=sys.stderr)
        for event, path in seen[:20]:
            print(f"  {event}: {path}", file=sys.stderr)
        session.exitstatus = 1
