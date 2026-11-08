"""The test suite's guard over the install-root store, shared by the root `conftest.py` and, via
`sitecustomize`, by every child Python interpreter a test starts. See the root `conftest.py` for
what it sees and what it cannot."""
import os
import sys
from urllib.parse import unquote, urlsplit

_WATCHED = {"open", "sqlite3.connect", "os.mkdir", "os.remove", "os.rename", "os.replace",
            "os.rmdir", "os.listdir", "os.scandir", "os.chdir", "os.truncate", "os.symlink",
            "os.link", "shutil.rmtree", "shutil.copyfile", "shutil.copytree", "shutil.move"}


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _real(path: str) -> str:
    try:
        return os.path.normcase(os.path.realpath(path))
    except (OSError, ValueError):
        return _norm(path)


def _roots():
    out = []
    for r in (os.environ.get("MANTLE_TEST_GUARDED") or "").split(os.pathsep):
        if r:
            out.append(r)
    return out


def _path_of(value):
    """A path from an audit argument: a path, bytes, a PathLike, or a sqlite `file:` URI."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if isinstance(value, bytes):
        try:
            value = os.fsdecode(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("file:"):
        parts = urlsplit(value)
        p = unquote(parts.path or "")
        if os.name == "nt" and len(p) > 2 and p[0] == "/" and p[2] == ":":
            p = p[1:]
        return p.replace("/", os.sep) if os.name == "nt" else p
    return value


def under(path, root) -> bool:
    """Is `path` at or below `root`, as written or once links are resolved?"""
    p = _path_of(path)
    if p is None:
        return False
    for rp, pp in ((_norm(root), _norm(p)), (_real(root), _real(p))):
        if pp == rp or pp.startswith(rp.rstrip(os.sep) + os.sep):
            return True
    return False


def guarded(path) -> bool:
    return any(under(path, r) for r in _roots())


def _names_guarded(args) -> bool:
    if isinstance(args, (str, bytes, os.PathLike)):
        args = [args]
    for a in args or ():
        p = _path_of(a)
        if p and (guarded(p) or any(r and _norm(r) in _norm(p) for r in _roots()
                                    if os.path.isabs(p))):
            return True
    return False


def install(on_refuse=None):
    def audit(event, args):
        hit = None
        if event in _WATCHED and args:
            for a in args[:2] if event in ("os.rename", "os.replace", "os.symlink", "os.link",
                                           "shutil.copyfile", "shutil.copytree",
                                           "shutil.move") else args[:1]:
                if guarded(a):
                    hit = _path_of(a)
                    break
        elif event == "subprocess.Popen" and args:
            _exe, argv, cwd = (list(args) + [None, None, None])[:3]
            if _names_guarded(argv) or (cwd is not None and guarded(cwd)):
                hit = f"subprocess: {argv!r} in {cwd!r}"
        if hit is not None:
            if on_refuse is not None:
                on_refuse(event, hit)
            raise PermissionError(f"a test may not touch the install-root store: {hit}")

    sys.addaudithook(audit)
