"""Unit tests for services.platform_settings_service.

Covers the in-memory cache + DEFAULTS fallback chain, type coercion helpers,
secret encryption, needs_setup gate, batch writes, and per-category grouping.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from mantle.services.platform_settings_service import (
    DEFAULTS,
    PlatformSettingsService,
)


@pytest.fixture
def svc():
    """A clean service with the test encryption key wired in."""
    s = PlatformSettingsService()
    key = Fernet.generate_key().decode()
    with patch(
        "mantle.services.platform_settings_service.get_encryption_key", return_value=key
    ):
        yield s


# ---------------------------------------------------------------------------
# load_all + cache priority
# ---------------------------------------------------------------------------

class TestLoadAll:
    def test_load_all_populates_cache_and_secret_flags(self, svc):
        rows = [
            {"id": "ai.openai.api_key", "value": "ciphertext", "is_secret": True},
            {"id": "branding.title", "value": "MyApp", "is_secret": False},
        ]
        with patch(
            "mantle.db.identity_backend.get_all_platform_settings", return_value=rows
        ):
            svc.load_all(MagicMock())

        assert svc.is_loaded is True
        assert svc._cache["branding.title"] == "MyApp"
        assert svc._secret_flags["ai.openai.api_key"] is True

    def test_load_all_skips_rows_with_no_key(self, svc):
        rows = [
            {"value": "x", "is_secret": False},  # no id
            {"id": "platform.log_level", "value": "debug"},
        ]
        with patch(
            "mantle.db.identity_backend.get_all_platform_settings", return_value=rows
        ):
            svc.load_all(MagicMock())
        assert "platform.log_level" in svc._cache
        assert len(svc._cache) == 1



# ---------------------------------------------------------------------------
# get / fallback chain
# ---------------------------------------------------------------------------

class TestGet:
    def test_cache_hit_returns_cached_value(self, svc):
        svc._cache["branding.title"] = "Override"
        svc._secret_flags["branding.title"] = False
        assert svc.get("branding.title") == "Override"

    def test_cache_miss_falls_back_to_DEFAULTS(self, svc):
        # branding.title is in DEFAULTS as "Agience"
        assert svc.get("branding.title") == DEFAULTS["branding.title"]

    def test_cache_miss_and_no_default_returns_user_default(self, svc):
        assert svc.get("nonexistent.key", default="fallback") == "fallback"

    def test_cache_miss_returns_none_when_no_default(self, svc):
        assert svc.get("nonexistent.key") is None

    def test_secret_value_is_decrypted_on_get(self, svc):
        cipher = Fernet(
            svc._encrypt.__self__._encrypt.__globals__["get_encryption_key"]().encode()
        )
        del cipher  # not actually using it; encrypt-then-get round-trips below

        encrypted = svc._encrypt("hunter2")
        svc._cache["api.token"] = encrypted
        svc._secret_flags["api.token"] = True
        assert svc.get("api.token") == "hunter2"


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------

class TestCoercion:
    def test_get_bool_truthy_values(self, svc):
        for v in ("true", "1", "yes", "TRUE"):
            svc._cache["k"] = v
            assert svc.get_bool("k") is True

    def test_get_bool_falsy_values(self, svc):
        for v in ("false", "0", "no", ""):
            svc._cache["k"] = v
            assert svc.get_bool("k") is False

    def test_get_bool_default_when_missing(self, svc):
        assert svc.get_bool("missing", default=True) is True

    def test_get_int_parses_valid(self, svc):
        svc._cache["k"] = "42"
        assert svc.get_int("k") == 42

    def test_get_int_default_on_garbage(self, svc):
        svc._cache["k"] = "not-a-number"
        assert svc.get_int("k", default=99) == 99

    def test_get_float_parses_valid(self, svc):
        svc._cache["k"] = "3.14"
        assert svc.get_float("k") == 3.14

    def test_get_float_default_on_garbage(self, svc):
        svc._cache["k"] = "x"
        assert svc.get_float("k", default=1.0) == 1.0

    def test_get_csv_list_splits_and_strips(self, svc):
        svc._cache["k"] = "a, b ,c,,d"
        assert svc.get_csv_list("k") == ["a", "b", "c", "d"]

    def test_get_csv_list_default_when_empty(self, svc):
        assert svc.get_csv_list("missing", default=["x"]) == ["x"]


# ---------------------------------------------------------------------------
# needs_setup
# ---------------------------------------------------------------------------

class TestNeedsSetup:
    def test_true_when_setup_complete_not_set(self, svc):
        # Default in DEFAULTS is "false"
        assert svc.needs_setup() is True

    def test_false_when_setup_complete_true(self, svc):
        svc._cache["platform.setup_complete"] = "true"
        svc._secret_flags["platform.setup_complete"] = False
        assert svc.needs_setup() is False


# ---------------------------------------------------------------------------
# set_setting / set_many / delete_keys
# ---------------------------------------------------------------------------

class TestWriters:
    def test_set_setting_encrypts_secret_before_persisting(self, svc):
        captured = {}

        def fake_set(db, **kwargs):
            captured.update(kwargs)

        with patch(
            "mantle.db.identity_backend.set_platform_setting", side_effect=fake_set
        ):
            svc.set_setting(
                MagicMock(),
                key="api.token",
                value="hunter2",
                category="api",
                is_secret=True,
            )

        # Persisted value is NOT plaintext.
        assert captured["value"] != "hunter2"
        # Cache is updated with the encrypted blob.
        assert svc._cache["api.token"] == captured["value"]
        # get() round-trips through decryption.
        assert svc.get("api.token") == "hunter2"

    def test_set_setting_plain_passthrough_for_non_secret(self, svc):
        with patch(
            "mantle.db.identity_backend.set_platform_setting"
        ):
            svc.set_setting(
                MagicMock(),
                key="branding.title",
                value="MyApp",
                category="branding",
                is_secret=False,
            )
        assert svc._cache["branding.title"] == "MyApp"

    def test_set_many_writes_each_and_reloads(self, svc):
        with (
            patch("mantle.db.identity_backend.set_platform_setting") as set_one,
            patch(
                "mantle.db.identity_backend.get_all_platform_settings", return_value=[]
            ),
        ):
            count = svc.set_many(
                MagicMock(),
                [
                    {
                        "key": "branding.title",
                        "value": "X",
                        "category": "branding",
                    },
                    {
                        "key": "ai.openai.api_key",
                        "value": "sk-1",
                        "category": "ai",
                        "is_secret": True,
                    },
                ],
            )
        assert count == 2
        assert set_one.call_count == 2
        # The reload was triggered (cache cleared by load_all on empty rows).
        assert svc.is_loaded is True

    def test_delete_keys_removes_from_db_and_reloads(self, svc, tmp_path):
        # Real temp lattice: settings live on the identity plane of the ONE store.
        from mantle.db import lattice_api as api
        from mantle.db import lattice_identity

        db = api.open_database(str(tmp_path / "t.db"), origin="test-settings")
        lattice_identity.set_platform_setting(db, key="k1", value="v1", category="c")
        lattice_identity.set_platform_setting(db, key="k2", value="v2", category="c")

        count = svc.delete_keys(db, ["k1", "k2", "missing"])
        # Only the keys that existed count as deleted.
        assert count == 2
        assert lattice_identity.get_platform_setting(db, "k1") is None
        assert lattice_identity.get_platform_setting(db, "k2") is None
        # The cache was reloaded from the (now empty) store.
        assert svc.is_loaded is True
        assert svc._cache == {}


# ---------------------------------------------------------------------------
# Encryption round-trip + invalid token tolerance
# ---------------------------------------------------------------------------

class TestEncryption:
    def test_round_trip(self, svc):
        ct = svc._encrypt("plaintext")
        assert ct != "plaintext"
        assert svc._decrypt(ct) == "plaintext"

    def test_decrypt_invalid_token_raises(self, svc):
        import pytest
        with pytest.raises(Exception):
            svc._decrypt("not-a-fernet-token")


# ---------------------------------------------------------------------------
# get_all_by_category
# ---------------------------------------------------------------------------

class TestGetAllByCategory:
    def test_db_values_take_precedence_over_defaults(self, svc):
        svc._cache["branding.title"] = "Custom"
        svc._secret_flags["branding.title"] = False
        grouped = svc.get_all_by_category()
        branding = next(g for g in grouped["branding"] if g["key"] == "branding.title")
        assert branding["value"] == "Custom"

    def test_secret_values_are_masked(self, svc):
        svc._cache["api.token"] = "ciphertext"
        svc._secret_flags["api.token"] = True
        grouped = svc.get_all_by_category(category="api")
        token = next(g for g in grouped["api"] if g["key"] == "api.token")
        assert token["value"] is None
        assert token["is_secret"] is True

    def test_defaults_filled_in_for_missing_keys(self, svc):
        grouped = svc.get_all_by_category(category="branding")
        keys = {g["key"] for g in grouped["branding"]}
        # branding.title comes from DEFAULTS even with empty cache.
        assert "branding.title" in keys

    def test_invalidate_cache_resets(self, svc):
        svc._cache["x"] = "y"
        svc._loaded = True
        svc.invalidate_cache()
        assert svc._cache == {}
        assert svc.is_loaded is False


class TestSettingMapAgreesWithDefaults:
    """`config._SETTING_MAP` names keys in this table. A key spelled differently in the
    two places resolves to None on every lookup — the DB row is never consulted, the
    module variable keeps its Phase-1 default forever, and nothing reports it. That is a
    silent failure, so it is asserted rather than left to review."""

    def test_every_mapped_key_has_a_default(self):
        from mantle import config

        missing = sorted(k for k in config._SETTING_MAP if k not in DEFAULTS)
        assert missing == [], (
            f"_SETTING_MAP keys absent from DEFAULTS (unreachable from the store): {missing}")

    def test_every_mapped_key_targets_a_real_module_variable(self):
        from mantle import config

        absent = sorted(
            var for var, _conv in config._SETTING_MAP.values() if not hasattr(config, var))
        assert absent == [], f"_SETTING_MAP targets non-existent config variables: {absent}"

    def test_csv_list_keys_are_mapped_keys(self):
        from mantle import config

        assert config._CSV_LIST_KEYS <= set(config._SETTING_MAP)


def _phase1_config():
    """`config` as an UNCONFIGURED node imports it — Phase-1 defaults, nothing rebound.

    Executed into a private module namespace rather than reloaded, because reloading
    `mantle.config` would rebind the singleton every other module already holds a
    reference to. `config.py` calls nothing at import (`load_env` is explicit, from
    `main.py`), so this reads the environment and returns.
    """
    import importlib.util

    from mantle import config

    spec = importlib.util.spec_from_file_location("_config_phase1", config.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTheStoreNeverOverridesAConfigDefault:
    """The layering is `env var > stored row > config's Phase-1 default`, and the third
    term is the one with no natural defence.

    `get()` falls back to DEFAULTS and so NEVER returns None, which means
    `config.load_settings_from_db()` takes the store's answer even on a node that has
    configured nothing at all. A literal default written in this table is therefore not a
    fallback — it overwrites `config.py`'s own default on every such node, silently and
    at every boot. Nothing surfaces the substitution: the variable simply holds a value
    no module claims to have chosen.

    So DEFAULTS mirrors the Phase-1 attributes instead of restating them, and that is
    what these assert. This class of failure is invisible at runtime — it is legible only
    by comparing two files, or by reading a boot log closely enough to notice a URI
    change between the identity line and the first outbound call.
    """

    def _unconfigured(self, monkeypatch):
        from mantle import config

        for var, _conv in config._SETTING_MAP.values():
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("AUTHORITY_ISSUER", raising=False)
        return _phase1_config()

    def test_every_stored_default_equals_the_config_default_it_would_replace(
            self, monkeypatch):
        from mantle import config

        phase1 = self._unconfigured(monkeypatch)
        svc = PlatformSettingsService()          # empty cache: every get() hits DEFAULTS

        for key, (var, converter) in config._SETTING_MAP.items():
            raw = svc.get(key)
            assert raw is not None, f"{key!r} is unreachable from the store"
            if key in config._CSV_LIST_KEYS:
                value = config._csv_list(raw)
            elif converter is not None:
                value = converter(raw)
            else:
                value = raw
            assert value == getattr(phase1, var), (
                f"the store's default for {key!r} resolves to {value!r} and would "
                f"overwrite config's own default for {var} ({getattr(phase1, var)!r}) "
                f"on any node where {var} is unset")

    def test_a_standalone_node_does_not_name_itself_as_its_token_authority(
            self, monkeypatch):
        """`AUTHORITY_ISSUER` derives from `ORIGIN_URI`, so an `ORIGIN_URI` default
        pointing at Mantle's own port makes a default node demand user tokens issued
        by itself — and sends its Origin client to its own door."""
        phase1 = self._unconfigured(monkeypatch)
        svc = PlatformSettingsService()

        assert svc.get("branding.origin_uri") != phase1.MANTLE_URI
        assert phase1.ORIGIN_URI != phase1.MANTLE_URI
        assert phase1.AUTHORITY_ISSUER != phase1.MANTLE_URI
