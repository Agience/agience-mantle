"""
services/platform_settings_service.py

DB-backed platform configuration with in-memory cache.

All runtime settings are stored in the platform_settings table (the lattice).
Settings marked is_secret=True are encrypted at rest using the Fernet
encryption key from key_manager.

Usage:
    from mantle.services.platform_settings_service import settings

    # After load_all() has been called at startup:
    value = settings.get("platform.index_queue_max_workers")
    secret = settings.get_secret("auth.google.client_secret")

This table is shared across services: `agience-origin` reads `branding.title`,
`auth.password.*`, `auth.invite_only` and `platform.allow_local_mcp_servers` from the
same rows. A key with no reader HERE may still have one there, so removing an entry is
a cross-repo decision, not a local one.
"""

import logging
from typing import Optional

from cryptography.fernet import Fernet
from mantle.db.store import Database

from prism.trust.key_manager import get_encryption_key
from mantle.db import identity_backend as identity_store
from mantle import config as _config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Default values — infrastructure settings
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, str] = {
    # No db.postgres.*, ai.*, or embeddings.* defaults: Origin runs its own SQLite and
    # Mantle its lattice (opened in-process from MANTLE_LATTICE_PATH), and model/embedding
    # capability is not configured here. Defaults are only defined for settings that
    # configure something in this deployment.

    # No search tuning either. A default is a promise that something reads the key, and
    # nothing reads a refresh interval, a BM25 window, or kNN candidate counts: Mantle's
    # search runs over encrypted MANTLE/SSE blobs in the content store, so there is no
    # OpenSearch-shaped engine for those numbers to reach. Chunk geometry is likewise
    # absent — it is a free parameter of `search/ingest/chunking.py`, defaulted at its
    # function signatures, and changing it without a full reindex only makes the stored
    # chunks disagree with the new ones. A knob that changes nothing is worse than no
    # knob: `get_all_by_category` surfaces every DEFAULTS key to the settings UI, so a
    # dead default is an operator control that visibly accepts a value and does nothing.

    # ── Keys `config._SETTING_MAP` rebinds: the default lives in `config.py` ──────────
    #
    # For these keys this table MIRRORS config rather than declaring a second default,
    # and the mirror is the module attribute itself so the two cannot be written apart.
    #
    # `get()` falls back to DEFAULTS and therefore never returns None, so a literal
    # written here is not a fallback the store may or may not supply — it is a value
    # that OVERWRITES config's own default on every node whose env var is unset and
    # whose store holds no row (`config.load_settings_from_db`). Two independently
    # authored defaults for one variable cannot coexist under that rule: the one here
    # always wins, and a config default that disagrees is silently unreachable. So
    # there is exactly one default per variable, and it is config's.
    #
    # The layering that remains, unchanged and now the whole of it:
    #   env var  >  stored row  >  config's Phase-1 default (reflected here).
    #
    # Reflecting the attribute also fixes the moment of resolution: an `os.getenv` call
    # here would read the environment at THIS module's import, which need not be the
    # same environment `config` read at its own.

    # Content storage (S3/MinIO)
    "storage.content_uri": _config.CONTENT_URI,
    "storage.content_bucket": _config.CONTENT_BUCKET,
    "storage.content_download_url_expiry": str(_config.CONTENT_DOWNLOAD_URL_EXPIRY),
    "storage.content_upload_url_expiry": str(_config.CONTENT_UPLOAD_URL_EXPIRY),
    "storage.content_multipart_part_url_expiry": str(_config.CONTENT_MULTIPART_PART_URL_EXPIRY),

    # Branding. No favicon: the browser clients take theirs from their own runtime
    # config (`VITE_FAVICON`), never from this table.
    "branding.facet_uri": _config.FACET_URI,
    "branding.origin_uri": _config.ORIGIN_URI,

    # Platform. `seed_collection_slugs` is the CSV form of config's list — the same
    # value in the encoding `config._CSV_LIST_KEYS` decodes.
    "platform.log_level": _config.BACKEND_LOG_LEVEL,
    "platform.index_queue_max_workers": str(_config.INDEX_QUEUE_MAX_WORKERS),
    "platform.seed_collection_slugs": ",".join(_config.SEED_COLLECTION_SLUGS),

    # ── Keys this table alone owns ───────────────────────────────────────────────────
    # Nothing in `config._SETTING_MAP` reads these, so the default below IS the default.

    "branding.title": "Agience",

    # Auth
    "auth.password.enabled": "true",
    "auth.password.min_length": "12",
    "auth.password.pbkdf2_iters": "200000",
    "auth.invite_only": "false",

    # Platform
    "platform.allow_local_mcp_servers": "false",
    "platform.setup_complete": "false",

    # Email (not configured by default)
    "email.provider": "",
    "email.from_address": "",
    "email.from_name": "Agience",
}


class PlatformSettingsService:
    """DB-backed platform configuration with in-memory cache."""

    def __init__(self):
        self._cache: dict[str, str] = {}
        self._secret_flags: dict[str, bool] = {}
        self._loaded = False

    def load_all(self, db: Database) -> None:
        """Load all settings from the lattice platform_settings collection into memory cache."""
        rows = identity_store.get_all_platform_settings(db)
        cache = {}
        secret_flags = {}
        for row in rows:
            key = row.get("id")
            if not key:
                continue
            cache[key] = row.get("value", "")
            secret_flags[key] = row.get("is_secret", False)
        self._cache = cache
        self._secret_flags = secret_flags
        self._loaded = True
        logger.info("Platform settings loaded: %d entries", len(cache))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a setting value. Returns from cache, then DEFAULTS, then default arg."""
        value = self._cache.get(key)
        if value is not None:
            # Auto-decrypt if it's a secret
            if self._secret_flags.get(key, False):
                return self._decrypt(value)
            return value
        return DEFAULTS.get(key, default)

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a setting value, decrypting if stored as secret."""
        value = self._cache.get(key)
        if value is not None:
            if self._secret_flags.get(key, False):
                return self._decrypt(value)
            return value
        return DEFAULTS.get(key, default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        value = self.get(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes")

    def get_int(self, key: str, default: int = 0) -> int:
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_csv_list(self, key: str, default: Optional[list[str]] = None) -> list[str]:
        value = self.get(key)
        if not value:
            return default or []
        return [item.strip() for item in value.split(",") if item.strip()]

    def needs_setup(self) -> bool:
        """Check if the platform setup wizard has been completed."""
        return self.get("platform.setup_complete") != "true"

    def set_setting(
        self,
        db: Database,
        key: str,
        value: str,
        category: str,
        is_secret: bool = False,
        updated_by: Optional[str] = None,
    ) -> None:
        """Write a single setting to the lattice and update the cache."""
        stored_value = self._encrypt(value) if is_secret else value

        identity_store.set_platform_setting(
            db,
            key=key,
            value=stored_value,
            category=category,
            is_secret=is_secret,
            updated_by=updated_by,
        )

        # Update cache
        self._cache[key] = stored_value
        self._secret_flags[key] = is_secret

    def set_many(
        self,
        db: Database,
        settings: list[dict],
        updated_by: Optional[str] = None,
    ) -> int:
        """
        Batch write settings. Each dict must have: key, value, category.
        Optional: is_secret (default False).
        Returns count of settings written.
        """
        count = 0
        for s in settings:
            key = s["key"]
            value = s["value"]
            category = s["category"]
            is_secret = s.get("is_secret", False)
            stored_value = self._encrypt(value) if is_secret else value

            identity_store.set_platform_setting(
                db,
                key=key,
                value=stored_value,
                category=category,
                is_secret=is_secret,
                updated_by=updated_by,
            )
            count += 1

        # Reload cache
        self.load_all(db)
        return count

    def delete_keys(self, db: Database, keys: list[str]) -> int:
        """Delete platform settings by key. Reloads the cache afterwards."""
        count = 0
        from mantle.db import lattice_identity
        for key in keys:
            if lattice_identity.delete_platform_setting(db, key):
                count += 1
        self.load_all(db)
        return count

    def get_all_by_category(self, category: Optional[str] = None) -> dict[str, list[dict]]:
        """
        Return all settings grouped by category.
        Secret values are returned as None (masked).
        DB values take precedence; DEFAULTS fill in any key not yet in the DB so
        the settings UI always shows the current effective value rather than blank fields.
        """
        grouped: dict[str, list[dict]] = {}
        seen_keys: set[str] = set()

        # DB values first (authoritative)
        for key, value in self._cache.items():
            is_secret = self._secret_flags.get(key, False)
            cat = key.split(".")[0] if "." in key else "platform"
            if category and cat != category:
                continue
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append({
                "key": key,
                "value": None if is_secret else value,
                "is_secret": is_secret,
            })
            seen_keys.add(key)

        # DEFAULTS for keys not yet written to DB (always non-secret)
        for key, default_value in DEFAULTS.items():
            if key in seen_keys:
                continue
            cat = key.split(".")[0] if "." in key else "platform"
            if category and cat != category:
                continue
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append({
                "key": key,
                "value": default_value,
                "is_secret": False,
            })

        return grouped

    def invalidate_cache(self) -> None:
        """Clear the in-memory cache. Next get() will use defaults only."""
        self._cache = {}
        self._secret_flags = {}
        self._loaded = False

    # -- Encryption helpers ------------------------------------------------

    def _encrypt(self, plaintext: str) -> str:
        cipher = Fernet(get_encryption_key().encode())
        return cipher.encrypt(plaintext.encode()).decode()

    def _decrypt(self, ciphertext: str) -> str:
        cipher = Fernet(get_encryption_key().encode())
        return cipher.decrypt(ciphertext.encode()).decode()


# Module-level singleton
settings = PlatformSettingsService()
