"""Every router module on disk is mounted in the app — or listed here with the reason it is not.

⚠ WHY THIS EXISTS. An unmounted router is not inert. It is a file that reads like a live API: it
declares paths, it has auth checks, it has a docstring describing what it serves — and it serves
nothing. Two were found on 2026-07-30, both left behind by the same Phase 2a migration whose comment
in `main.py` says the routers were "removed":

  · `stream_router.py` (440 lines) — superseded by astra's live `stream_routes.py`, which is what
    actually handles the SRS webhooks. It survived only because a structural-duplication scan
    matched its `_parse_stream_key` against astra's. DELETED.
  · `server_credentials_router.py` (63 lines) — see the allow-list entry below. NOT deleted, because
    it is the only implementation of something that is missing, and deleting it would erase the
    evidence.

Dead code in a router directory is worse than dead code elsewhere: the thing it misleads you about
is the SHAPE OF THE API. Reading `routers/` and concluding "mantle serves /stream" was a reasonable
inference and a wrong one.
"""
from __future__ import annotations

import pathlib
import re

import pytest

# ⚠ DEPTH CHANGED 2026-07-31: the suite moved from `src/mantle/tests/` to `<repo>/tests/`, so the
# package is no longer the parent — it is `src/mantle` from the repo root. The `test_there_is_
# something_to_check` guard below caught this immediately ("only 0 router modules found"), which is
# exactly why a sweep must assert its corpus is non-empty: a wrong path would otherwise have made
# every parametrized case vanish and the module report success.
_PKG = pathlib.Path(__file__).resolve().parents[1] / "src" / "mantle"
ROUTERS_DIR = _PKG / "routers"
MAIN_PY = _PKG / "main.py"

#: Modules in `routers/` that are deliberately NOT mounted, each with the reason. Stated here rather
#: than deleted or silently tolerated — an unmounted router is a claim about the API that is not
#: true, and the reason is the only thing that makes it reviewable.
UNMOUNTED_WITH_REASON = {
    "server_credentials_router":
        "⛔ ORPHANED WRITE PATH — A LIVE READ PATH DEPENDS ON IT. This module holds the ONLY "
        "`PUT /{client_id}/key` in the workspace: the endpoint a server calls to register the RSA "
        "public key that mantle wraps its secrets to. It is not mounted here, and origin's own "
        "`server_credentials_router` has no key endpoint either (POST/GET/PATCH/DELETE/rotate "
        "only). So `lattice_api.upsert_server_jwk` is reachable from no HTTP surface anywhere and "
        "is called by exactly one test.\n"
        "        The consequence is live: `secrets_router.py:166` calls `get_server_jwk(...)`, gets "
        "None because nothing can ever have written one, and raises 403 'No public key registered "
        "for server'. That message blames the caller's configuration for a missing endpoint — the "
        "feature is not misconfigured, it is unfinished. Its own docstring says it 'disappears "
        "entirely once server JWK storage migrates to Origin'; that migration did not happen.\n"
        "        Kept rather than deleted so the only implementation survives the decision. "
        "RESOLVE BY: mounting it, moving the endpoint to origin, or removing the read path too — "
        "not by deleting this line.",
}


def _router_modules():
    return sorted(p.stem for p in ROUTERS_DIR.glob("*.py") if p.stem != "__init__")


def _mounted_names():
    """Names passed to `app.include_router(...)` in main.py, resolved back to their module.

    Reads the IMPORT line rather than assuming `include_router(x)` implies a module called `x` —
    routers are imported `as` an alias (`from mantle.routers.mcp_router import mcp_router`), so
    matching on the alias alone would quietly pass a module that is imported and never included.
    """
    src = MAIN_PY.read_text(encoding="utf-8")
    included = set(re.findall(r"include_router\(\s*([A-Za-z_][A-Za-z0-9_]*)", src))
    mounted = set()
    for module, alias in re.findall(
            r"from mantle\.routers\.([a-z_]+) import (?:router as )?([A-Za-z_][A-Za-z0-9_]*)", src):
        if alias in included:
            mounted.add(module)
    return mounted


@pytest.mark.parametrize("module", _router_modules())
def test_router_module_is_mounted_or_explained(module):
    if module in UNMOUNTED_WITH_REASON:
        pytest.skip(f"deliberately unmounted: {UNMOUNTED_WITH_REASON[module][:80]}…")
    assert module in _mounted_names(), (
        f"routers/{module}.py is not mounted in main.py. It declares paths that nothing serves, "
        f"which misrepresents the API to anyone reading the directory. Mount it, delete it, or add "
        f"it to UNMOUNTED_WITH_REASON with the reason.")


def test_the_allow_list_is_not_stale():
    """⛔ The skip above is the hole. An entry naming a file that no longer exists is a permission
    nobody uses and a reason nobody re-reads — and it would make the skip permanent and invisible."""
    missing = [m for m in UNMOUNTED_WITH_REASON if not (ROUTERS_DIR / f"{m}.py").exists()]
    assert not missing, f"UNMOUNTED_WITH_REASON names modules that are gone: {missing}"


def test_the_mount_detector_actually_detects(tmp_path, monkeypatch):
    """⛔ NEGATIVE CONTROL. `_mounted_names` parses source with a regex; if that regex stopped
    matching, every router would look unmounted (loud) — but if it matched too eagerly, every router
    would look mounted and this whole module would pass while checking nothing. The second failure
    is the silent one, so prove the detector distinguishes imported-and-included from
    imported-only."""
    fake_main = tmp_path / "main.py"
    fake_main.write_text(
        "from mantle.routers.alpha_router import router as alpha_router\n"
        "from mantle.routers.beta_router import router as beta_router\n"
        "from mantle.routers.gamma_router import gamma_router\n"
        "app.include_router(alpha_router)\n"
        "app.include_router(gamma_router)\n",
        encoding="utf-8")
    monkeypatch.setattr(f"{__name__}.MAIN_PY", fake_main)
    got = _mounted_names()
    assert got == {"alpha_router", "gamma_router"}, got
    assert "beta_router" not in got, "imported but never included must NOT count as mounted"


def test_there_is_something_to_check():
    """A glob bug would empty the roster and turn every parametrized case green by not existing."""
    mods = _router_modules()
    assert len(mods) >= 8, f"only {len(mods)} router modules found — the glob is wrong"
    assert _mounted_names(), "no mounted routers detected — the parser is wrong"
