"""Lazy indexing (latent → materialized) — Phase 1 config/hint contract.

Pure-logic tests for the write-time decision (`resolve_lazy`) and the valence
derivation. The default MUST be eager (today's behaviour) so enabling lazy is an
explicit opt-in; the per-write `index` hint always wins over the deployment
default.
"""
import importlib
import os


def _fresh(monkeyenv=None):
    """Reload the module so the env-driven default is read fresh per case."""
    if monkeyenv is None:
        os.environ.pop("MANTLE_LAZY_INDEX", None)
    else:
        os.environ["MANTLE_LAZY_INDEX"] = monkeyenv
    import mantle.search.lazy as lazy
    return importlib.reload(lazy)


def test_default_is_eager_optin():
    lazy = _fresh()
    assert lazy.lazy_index_default() is False
    assert lazy.resolve_lazy(None) is False          # unset default → eager (unchanged)


def test_hint_overrides_default():
    lazy = _fresh()  # default off
    assert lazy.resolve_lazy("lazy") is True
    assert lazy.resolve_lazy("eager") is False
    lazy = _fresh("true")  # default on
    assert lazy.resolve_lazy(None) is True
    assert lazy.resolve_lazy("eager") is False        # hint still wins


def test_env_truthiness():
    for on in ("1", "true", "YES", "on"):
        assert _fresh(on).lazy_index_default() is True
    for off in ("0", "false", "", "no"):
        assert _fresh(off).lazy_index_default() is False


def test_valence_is_two_or_five():
    lazy = _fresh()
    assert lazy.valence(False) == lazy.VALENCE_LATENT == 2
    assert lazy.valence(True) == lazy.VALENCE_MATERIALIZED == 5


def teardown_module(_m):
    os.environ.pop("MANTLE_LAZY_INDEX", None)
