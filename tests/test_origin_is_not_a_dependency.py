"""Mantle does not import `origin`. Proven by blocking it, not by grepping for it.

A grep is no better on its own: it sees `from origin import config` and misses a lazy
`importlib.import_module("origin.…")`, a re-export through a third package, or a function-local
import in a branch nothing in the suite executes.

So this blocks the module at the import system and imports mantle for real. Anything that reaches
for `origin` — at module level, lazily, or through a chain — raises here and nowhere else.

Run: pytest tests/test_origin_is_not_a_dependency.py
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

#: Installed into a child interpreter. A meta-path finder is used rather than
#: `sys.modules["origin"] = None`, because the latter raises only for a plain `import origin` and
#: lets `from origin.config import X` resolve through the already-imported submodule in some paths.
_BLOCKER = """
import sys
class _Blocked(Exception):
    pass
class _Blocker:
    def find_module(self, fullname, path=None):
        return self.find_spec(fullname, path)
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "origin" or fullname.startswith("origin."):
            raise _Blocked("BLOCKED: something imported %r" % fullname)
        return None
sys.meta_path.insert(0, _Blocker())
# Drop anything already resolved, so a pre-import cannot mask the edge.
for _m in [m for m in sys.modules if m == "origin" or m.startswith("origin.")]:
    del sys.modules[_m]
"""


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(body)],
        capture_output=True, text=True, timeout=180,
    )


def test_the_blocker_actually_blocks():
    """The negative control. If this passes trivially, every other assertion here is worthless."""
    proc = _run("import origin\n")
    assert proc.returncode != 0, (
        "the blocker did not fire on a direct `import origin` — every other test in this file is "
        "then vacuous, because it would pass with the edge fully intact"
    )
    assert "BLOCKED" in proc.stderr, proc.stderr


@pytest.mark.parametrize("module", [
    "mantle.main",              # the whole service, every router and service it wires
    "mantle.config",            # the module that replaced origin.config
    "mantle.services.oidc",     # token verification — the path an Entra deployment stands on
    "mantle.services.principal",  # a plausible route to origin.person
])
def test_mantle_imports_with_origin_blocked(module):
    proc = _run(f"import {module}\n")
    assert proc.returncode == 0, (
        f"{module} still reaches `origin`:\n{proc.stderr}"
    )
