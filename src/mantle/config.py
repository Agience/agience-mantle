"""Mantle's own configuration. No platform import, no identity-service import.

A settings module must not require another service's source tree to import: the README's claim
that "Mantle IS the database" — one SQLite file plus a filesystem CAS, opened in-process, nothing
external to provision — depends on this module resolving on its own. `agience-prism` is the one
declared cross-package edge (`[project.optional-dependencies].service`), imported in ~28 modules.
"""
from __future__ import annotations

import json as _json
import os
import site as _site
import sys as _sys
import sysconfig as _sysconfig
import uuid as _uuid
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse as _urlparse, urlunparse as _urlunparse

try:                                    # python-dotenv is optional at import time
    from dotenv import load_dotenv as _load_dotenv
except ImportError:                     # pragma: no cover - only when the extra is absent
    def _load_dotenv(*_a, **_k):        # type: ignore[misc]
        return False


def _origin_only(uri: str) -> str:
    """Return scheme+host+port (strip path, query, fragment)."""
    p = _urlparse(uri)
    return _urlunparse((p.scheme, p.netloc, "", "", "", ""))


# ---------------------------------------------------------------------------
#  Phase 0: load .env into os.environ (before any os.getenv call)
# ---------------------------------------------------------------------------
_MANTLE_ROOT = Path(__file__).resolve().parent.parent      # …/src


def load_env(base_dir: Optional[Path] = None) -> Optional[Path]:
    """Load mantle's `.env` — CALLED EXPLICITLY from `main.py`, never at import.

    A `.env` only ever provides DEFAULTS: the shell/container environment always wins
    (`override=False`, standard 12-factor). In Docker the container sets env directly and there is
    no `.env`, so this is a no-op there.
    """
    # Tests set AGIENCE_NO_DOTENV=1 so a real dev `.env` can never leak in: `load_dotenv` mutates
    # os.environ outside monkeypatch's restore, so it would persist past the test that caused it.
    if os.getenv("AGIENCE_NO_DOTENV"):
        return None
    root = Path(base_dir).resolve() if base_dir else _MANTLE_ROOT
    for candidate in (root / ".env", root.parent / ".env"):
        if candidate.is_file():
            _load_dotenv(candidate, override=False)
            return candidate
    return None


# ---------------------------------------------------------------------------
#  Phase 1: static constants (safe at import, never change)
# ---------------------------------------------------------------------------
def _install_locations() -> set:
    """Every directory THIS interpreter installs packages into, resolved.

    Asked of the interpreter rather than assumed, because the answer is not one path and not one
    spelling: a venv on Windows uses `Lib/site-packages` and on POSIX `lib/pythonX.Y/site-packages`,
    Debian ships `dist-packages`, `pip install --target`/`--prefix` uses neither, and a user install
    lands somewhere else again. `sysconfig`'s purelib/platlib are where the installer is told to put
    things; `site`'s lists are where the interpreter looks for them.

    Every source is optional: an embedded or `-S` interpreter may have no `site` at all, and a
    stripped `sysconfig` may not answer. A source that raises contributes nothing rather than
    breaking the import of a settings module.
    """
    found = set()
    def _paths():
        p = _sysconfig.get_paths()
        return [p.get("purelib"), p.get("platlib")]
    for source in (_paths, getattr(_site, "getsitepackages", None),
                   getattr(_site, "getusersitepackages", None)):
        if source is None:
            continue
        try:
            got = source()
        except Exception:               # pragma: no cover - interpreter without `site`
            continue
        if isinstance(got, str):
            got = [got]
        for raw in got or ():
            if not raw:
                continue
            try:
                found.add(Path(raw).resolve())
            except OSError:             # pragma: no cover - unresolvable path
                pass
    return found


def _is_installed_tree(root: Path) -> bool:
    """Is `root` a directory pip installs INTO — the tree `pip install --upgrade` rewrites?

    The name check is the floor, for interpreters that answer neither of the questions above; the
    interpreter's own answer is the substance.
    """
    if root.name in ("site-packages", "dist-packages"):
        return True
    return any(root == p or p in root.parents for p in _install_locations())


def _derive_base_dir(root: Path) -> Path:
    """Where this node's data goes when `AGIENCE_BASE_DIR` says nothing. `root` is the directory
    the `mantle` package sits in — `<repo>/src` in a checkout.

    THE QUESTION IS NOT "WHICH INSTALL SHAPE" BUT "IS THIS DIRECTORY MINE TO WRITE IN". Four
    shapes reach this line and only one of them owns a writable tree next to the source:

    * a **checkout** (and a **PEP 660 editable install**, which is the same thing seen through a
      finder — `root` is the real `<repo>/src` on disk either way): the repo root is the node's
      working tree, and the derived default is where a developer already expects `.data/` to be;
    * a **plain wheel install**: `root` is `<venv>/Lib/site-packages`, which belongs to the package
      manager. `MANTLE_SSE_DIR`, `MANTLE_CELL_DIR` and the embeddings cache defaulting in there put
      a node's INDEXES — data, not cache; a full rebuild is measured in days-to-weeks — inside a
      directory the next `pip install --upgrade` rewrites;
    * a **zip import**: there is no directory behind `__file__` at all, so nothing can be written
      beside it;
    * a **frozen build**: `__file__` is inside a bundle that PyInstaller unpacks to a temp
      directory and DELETES on exit — the one place worse than site-packages.

    So the three that do not own a tree fall back to the working directory, which is a place the
    person who started the node chose. That is a weaker guarantee than an absolute install root —
    a node started from a different directory opens a different index, the failure
    `MANTLE_LATTICE_PATH` and `local_sse_root()` both warn about — and it is the right trade: a
    node that loses its index to `pip install --upgrade` has lost it for good, while a node started
    from the wrong directory is one `cd` from being right. `AGIENCE_BASE_DIR` remains the way to
    say it outright, and a pip-installed node should set it.
    """
    if getattr(_sys, "frozen", False):
        return Path.cwd()
    try:
        on_disk = root.is_dir()
    except OSError:                     # pragma: no cover - exotic loader
        on_disk = False
    if not on_disk:
        return Path.cwd()
    if _is_installed_tree(root):
        return Path.cwd()
    return root.parent if root.name == "src" else root


_BASE_DIR_OVERRIDE = os.getenv("AGIENCE_BASE_DIR")
BASE_DIR = (
    Path(_BASE_DIR_OVERRIDE).resolve()
    if _BASE_DIR_OVERRIDE
    else _derive_base_dir(_MANTLE_ROOT)
)
KEYS_DIR = Path(os.getenv("KEYS_DIR", str(BASE_DIR / ".data" / "keys")))

#: THE STORE'S DEFAULT LOCATION — the one SQLite file that IS the lattice, when
#: `MANTLE_LATTICE_PATH` is unset. Absolute, and derived from `BASE_DIR` exactly as `KEYS_DIR`
#: above, the embeddings cache (`.data/mantle/`) and the SSE index (`.data/mantle-sse`,
#: `search/mantle/wiring.py`) are. Everything this node writes therefore lands under one `.data/`
#: directory, which is what `.gitignore` and `.dockerignore` exclude as a unit.
#:
#: A BARE RELATIVE NAME WOULD RESOLVE AGAINST THE WORKING DIRECTORY EACH TIME IT WAS READ, so the
#: same node would open a different store depending on where it was started from — and come up
#: healthy serving an empty universe. This is absolute: `BASE_DIR` is fixed once at import, and on
#: an installed node that fixes it to the working directory the node was STARTED in rather than to
#: `site-packages` (see `_derive_base_dir`). An explicit `MANTLE_LATTICE_PATH` still wins, and a
#: relative one still resolves as the caller wrote it; this is only what happens when nobody says.
#:
#: `db/lattice_api.py` re-exports this and `db/backend.py` reads it from there, so the two callers
#: cannot drift apart.
DEFAULT_LATTICE_PATH = BASE_DIR / ".data" / "mantle-lattice.db"

#: Platform identity — deterministic UUID, never changes. Must match origin's, because it names the
#: same principal in a store both can read. Derived identically rather than copied as a literal.
AGIENCE_PLATFORM_USER_ID = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "agience://platform"))

# -- service URIs -----------------------------------------------------------
# Three different things on three different ports, and the defaults must not blur them:
# the facet is the BROWSER APP (Vite, :5173) that `grant_service.build_claim_url` sends a
# human to, Origin is the identity SERVICE (:8080) this node dials and derives
# `AUTHORITY_ISSUER` from, and Mantle is this node (:8081). Defaulting any of them to
# another's port makes an unconfigured node name the wrong service — and, for Origin,
# name ITSELF as the issuer its user tokens must carry.
#
# These are Phase-1 defaults and they are the ONLY ones: `_SETTING_MAP` below rebinds
# them from the settings store, and that store's `DEFAULTS` mirrors these values rather
# than declaring its own. See the note on `_SETTING_MAP`.
FACET_URI: str = os.getenv("FACET_URI", "http://localhost:5173")
#: Named, so `authorization_servers()` below can tell this value from one an operator DECLARED.
#: Every other consumer of `ORIGIN_URI` wants a usable URL and this is the usable one; only the
#: question "is there actually an authorization server there" needs the provenance.
_ORIGIN_URI_PHASE1_DEFAULT = "http://localhost:8080"
ORIGIN_URI: str = os.getenv("ORIGIN_URI", _ORIGIN_URI_PHASE1_DEFAULT)
MANTLE_URI: str = os.getenv("MANTLE_URI", "http://localhost:8081")

OPERATOR_ID: str = os.getenv("AGIENCE_OPERATOR_ID", "")

# -- identity ---------------------------------------------------------------
AUTHORITY_ISSUER: str = _origin_only(os.getenv("AUTHORITY_ISSUER") or ORIGIN_URI)
AUTHORITY_DOMAIN: str = _urlparse(AUTHORITY_ISSUER).hostname or "localhost"

#: Did a STORED `branding.origin_uri` row declare the authority? Set by `load_settings_from_db`,
#: read by `authorization_servers()`. False until then, and False on a node with no store — which
#: is every consumer of this module that is not the running app.
_AUTHORITY_DECLARED_IN_STORE: bool = False

# ── the browser's own OAuth client ──────────────────────────────────────────────────────────────
# THIS IS THE WHOLE OF THE SIGN-IN CONFIGURATION. There is no second setting naming an external
# application to send a human to: `MANTLE_SIGNIN_URI` was that setting, read only by a landing page
# `ui/browse_page.py` superseded, and it was removed rather than rewired — a documented control
# reaching no live code accepts a value and does nothing, which is indistinguishable from working
# until someone depends on it.
#
# The page does not link out to a sign-in; it RUNS one. Authorization Code + PKCE against the
# issuer's own discovery document, with this node's `redirect_uri` — so what a deployment must
# supply is a client registration, below, not a URL to redirect to.
# Authorization Code + PKCE (RFC 7636), the standards-track flow for a browser client. A PUBLIC
# client: there is no secret here and there must not be one, because anything shipped to a browser
# is not a secret. PKCE is what replaces it — the client proves it started the flow by presenting
# the verifier whose S256 hash it committed to up front.
#
# An unset client id is a configuration fact, not a failure. A node with no browser client still
# serves its API to token-bearing callers; only the human door is unavailable, and the page says so
# by name. Guessing a client_id would produce a login button that fails at the IdP.
OIDC_CLIENT_ID: str = (os.getenv("MANTLE_OIDC_CLIENT_ID") or "").strip()

# What the browser asks for. `openid` is required to get an id_token at all; the rest is per-IdP —
# Entra needs a scope that resolves to THIS api's audience or the access token comes back for
# Graph instead, and `aud` then fails verification here with "Invalid token audience".
OIDC_SCOPE: str = (os.getenv("MANTLE_OIDC_SCOPE") or "openid profile email").strip()

# External OIDC IdPs whose tokens mantle verifies directly. JSON array via
# AGIENCE_TRUSTED_ISSUERS; each entry {"issuer", "audience", "jwks_uri" | "jwks", "namespace"?,
# "role"?}. Empty = platform-only. This is the seam that makes auth IdP-agnostic — Microsoft Entra,
# Auth0, Okta. `services/oidc.py` verifies signature + iss + aud against the issuer's JWKS entirely
# locally, and derives a stable user id as uuid5(namespace, (issuer, sub)): an external user's
# identity is computed, never fetched, so no identity service need be reachable.
TRUSTED_ISSUERS: list = []
_raw_trusted = os.getenv("AGIENCE_TRUSTED_ISSUERS", "").strip()
if _raw_trusted:
    try:
        _parsed_ti = _json.loads(_raw_trusted)
        if isinstance(_parsed_ti, list):
            TRUSTED_ISSUERS = [i for i in _parsed_ti if isinstance(i, dict) and i.get("issuer")]
    except ValueError:
        pass

# -- storage / content ------------------------------------------------------
CONTENT_URI: str = "http://localhost:9000"
CONTENT_BUCKET: str = "agience-content"
CONTENT_DOWNLOAD_URL_EXPIRY: int = 300
CONTENT_UPLOAD_URL_EXPIRY: int = 900
CONTENT_MULTIPART_PART_URL_EXPIRY: int = 300

# -- search / indexing ------------------------------------------------------
# Chunk size and overlap are not here: they are free parameters of `search/ingest/chunking.py`,
# defaulted at its function signatures. A setting is for something an operator must be able to
# change per deployment; chunk geometry is a property of the index, and changing it without a
# reindex only makes the stored chunks disagree with the new ones.
#
# Nor is a search refresh interval: Mantle's search reads encrypted MANTLE/SSE blobs from the
# content store directly, so there is no index to make freshly-visible and nothing to time.
INDEX_QUEUE_MAX_WORKERS: int = 16
SEED_COLLECTION_SLUGS: list = ["agience-inbox-seeds"]

# -- features / misc --------------------------------------------------------
BACKEND_LOG_LEVEL: str = "info"

# -- set in Phase 1.5 from key files, never from env ------------------------
PLATFORM_ENCRYPTION_KEY: str = ""
INBOUND_NONCE_SECRET: str = ""


# ---------------------------------------------------------------------------
#  Phase 1.5: bootstrap settings from key files (after key_manager init)
# ---------------------------------------------------------------------------
def load_bootstrap_settings() -> None:
    """Load the encryption key and inbound nonce secret from key files.

    """
    global PLATFORM_ENCRYPTION_KEY, INBOUND_NONCE_SECRET
    from prism.trust.key_manager import get_encryption_key, get_nonce_secret

    PLATFORM_ENCRYPTION_KEY = get_encryption_key()
    INBOUND_NONCE_SECRET = get_nonce_secret()


# ---------------------------------------------------------------------------
#  Phase 2: rebind from the app's settings cache
# ---------------------------------------------------------------------------
# The settings provider seam. The platform-settings cache is an app service
# (`services.platform_settings_service`); this module never reaches into the app to find it —
# that would be a dependency pointing the wrong way through a side door. The app hands its getter
# down:
#
#     config.set_settings_provider(platform_settings.get)   # main.py, Phase 2
#     config.load_settings_from_db()
#
# No provider injected fails loudly as a boot-order bug, rather than silently leaving every
# DB-backed setting on its Phase-1 default.
_SETTINGS_PROVIDER: Optional[Callable[[str], Optional[str]]] = None

#: setting key -> (module variable, converter). Mantle's slice only — see the module docstring.
#:
#: THE DEFAULT FOR EVERY VARIABLE NAMED HERE LIVES ABOVE, IN PHASE 1, AND NOWHERE ELSE.
#: `services/platform_settings_service.DEFAULTS` mirrors these Phase-1 values by reading the
#: module attribute, so the store cannot answer with a default of its own invention.
#:
#: The mirror is not decoration. `settings.get()` falls back to `DEFAULTS` and so never returns
#: None, which means the store-backed branch below runs on a node that has configured NOTHING:
#: no env var and no stored row. A literal written in `DEFAULTS` would therefore not be a
#: fallback — it would OVERWRITE the Phase-1 default on every such node, silently, and the value
#: fifteen lines above this one would be dead text. Reflecting the attribute makes the two
#: physically the same value.
#:
#: Layering, entire:  env var  >  stored row  >  Phase-1 default.
#:
#: Every key here must also EXIST in `DEFAULTS`: a key spelled any other way is invisible to
#: `get_all_by_category`, so it never appears in the settings UI and an operator has no control
#: to set the row with.
_SETTING_MAP: dict = {
    "branding.facet_uri": ("FACET_URI", str),
    "branding.origin_uri": ("ORIGIN_URI", str),
    "platform.log_level": ("BACKEND_LOG_LEVEL", lambda v: v.lower()),
    "platform.index_queue_max_workers": ("INDEX_QUEUE_MAX_WORKERS", int),
    "platform.seed_collection_slugs": ("SEED_COLLECTION_SLUGS", None),
    "storage.content_uri": ("CONTENT_URI", str),
    "storage.content_bucket": ("CONTENT_BUCKET", str),
    "storage.content_download_url_expiry": ("CONTENT_DOWNLOAD_URL_EXPIRY", int),
    "storage.content_upload_url_expiry": ("CONTENT_UPLOAD_URL_EXPIRY", int),
    "storage.content_multipart_part_url_expiry": ("CONTENT_MULTIPART_PART_URL_EXPIRY", int),
}

#: CSV-list keys need special handling.
_CSV_LIST_KEYS = {"platform.seed_collection_slugs"}


def set_settings_provider(get: Callable[[str], Optional[str]]) -> None:
    """Inject the app's settings getter (key -> value or None). Called at boot, after the app has
    loaded its settings cache and before `load_settings_from_db()`."""
    global _SETTINGS_PROVIDER
    _SETTINGS_PROVIDER = get


def _csv_list(value: Optional[str]) -> list:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_settings_from_db() -> None:
    """Rebind module-level variables from the settings cache. Call after the cache is loaded."""
    global FACET_URI, ORIGIN_URI, AUTHORITY_ISSUER, AUTHORITY_DOMAIN
    global _AUTHORITY_DECLARED_IN_STORE
    global BACKEND_LOG_LEVEL, INDEX_QUEUE_MAX_WORKERS
    global SEED_COLLECTION_SLUGS
    global CONTENT_URI, CONTENT_BUCKET
    global CONTENT_DOWNLOAD_URL_EXPIRY, CONTENT_UPLOAD_URL_EXPIRY, CONTENT_MULTIPART_PART_URL_EXPIRY

    if _SETTINGS_PROVIDER is None:
        raise RuntimeError(
            "load_settings_from_db() called with no settings provider injected — "
            "call config.set_settings_provider(<settings.get>) after loading the app's "
            "settings cache (see main.py Phase 2)")
    settings_get = _SETTINGS_PROVIDER

    # Recomputed from scratch on every pass, not accumulated: this function is idempotent by
    # contract, and a flag that only ever latched True would make a second call report a
    # declaration the store no longer holds.
    _AUTHORITY_DECLARED_IN_STORE = False

    def _apply(var_name: str, setting_key: str, converter, value) -> None:
        """Assign one setting, converting as the map requires. Both the env and DB paths must
        route through here — a converter on one path only lets the two drift."""
        if setting_key in _CSV_LIST_KEYS:
            globals()[var_name] = _csv_list(value)
        elif converter is not None:
            try:
                globals()[var_name] = converter(value)
            except (ValueError, TypeError):
                pass                        # keep the default
        else:
            globals()[var_name] = value

    for setting_key, (var_name, converter) in _SETTING_MAP.items():
        # Environment overrides the DB (operator .env wins).
        #
        env_val = os.getenv(var_name)
        # An EMPTY assignment (the usual `.env` template line) is treated as UNSET rather than as a
        # decision to blank the value — otherwise a stock template silently suppresses a key that is
        # correctly stored in the DB.
        if env_val is not None and env_val.strip() != "":
            _apply(var_name, setting_key, converter, env_val)
            continue

        value = settings_get(setting_key)
        if value is None:
            continue
        # A STORED ROW IS A DECLARATION; the DEFAULTS mirror is not. `settings.get()` falls back to
        # `platform_settings_service.DEFAULTS`, which reflects this module's own Phase-1 attribute,
        # so it never returns None and this branch runs on a node that has configured nothing. The
        # two cases are told apart by the only thing that distinguishes them: a value equal to the
        # Phase-1 default is what an absent row produces, and a row repeating that default says
        # nothing the default did not already say. See `authorization_servers()`.
        if setting_key == "branding.origin_uri" and value != _ORIGIN_URI_PHASE1_DEFAULT:
            _AUTHORITY_DECLARED_IN_STORE = True
        _apply(var_name, setting_key, converter, value)

    # Derived. The issuer is the PUBLIC origin URI (explicit env wins) — NOT the internal ORIGIN_URI
    # used to reach a service, which may carry a path prefix that must never appear in `iss`/`aud`.
    AUTHORITY_ISSUER = _origin_only(os.getenv("AUTHORITY_ISSUER") or ORIGIN_URI)
    try:
        AUTHORITY_DOMAIN = _urlparse(AUTHORITY_ISSUER).hostname or "localhost"
    except ValueError:
        AUTHORITY_DOMAIN = "localhost"


# ---------------------------------------------------------------------------
# Which issuers this node sends a human to — one definition, three consumers
# ---------------------------------------------------------------------------
def authority_is_declared() -> bool:
    """Did anyone SAY where this node's authority is, or is it sitting on the Phase-1 default?

    Three ways to say it, and the environment is read live rather than at import so a `.env` loaded
    after this module was imported (`main.py`'s lifespan does exactly that) still counts:

    * `AUTHORITY_ISSUER` in the environment, non-empty,
    * `ORIGIN_URI` in the environment, non-empty,
    * a stored `branding.origin_uri` row that differs from the Phase-1 default
      (`_AUTHORITY_DECLARED_IN_STORE`, set in `load_settings_from_db`).

    An EMPTY value is a decision to say nothing, not a declaration, which is the same reading
    `load_settings_from_db` already applies to an empty env var: a stock `.env` template line is not
    an operator suppressing a key.
    """
    if (os.getenv("AUTHORITY_ISSUER") or "").strip():
        return True
    if (os.getenv("ORIGIN_URI") or "").strip():
        return True
    return _AUTHORITY_DECLARED_IN_STORE


def declared_public_uri() -> str:
    """Where this node SAYS it answers, or "" when nobody has said. Trailing slash stripped.

    The same distinction `authority_is_declared()` draws, drawn the same way and for the same
    reason: `MANTLE_URI`'s Phase-1 default is `http://localhost:8081`, a developer default rather
    than a statement about where this node answers, and a consumer that treats it as a declaration
    hands an IdP a redirect to localhost from a production host.

    WHAT DECIDES IS WHETHER IT WAS SAID, NOT WHAT IT SAYS. Rejecting the value `http://localhost…`
    is the reading that cannot be argued with: a node that genuinely answers on localhost — a
    laptop, a sidecar, a `docker run -p 8081:8081` — has no way to say so, because the only
    sentence that is true of it is the one being refused. `MANTLE_URI=http://localhost:8081` set on
    purpose is a statement; the same string arrived at by saying nothing is not.

    One clause where the authority has three: `MANTLE_URI` has no `_SETTING_MAP` entry, so there is
    no stored row that could declare it and the environment is the only place it can be said. Read
    LIVE rather than at import for the reason `authority_is_declared()` gives — `main.py`'s lifespan
    calls `load_env()` after this module has been imported, so a `.env` that names it would
    otherwise be invisible. That also makes this, not `MANTLE_URI`, the accurate VALUE for a caller
    that needs the declaration: the module attribute was bound before the `.env` was read.

    An EMPTY value is a decision to say nothing rather than a declaration — the reading
    `load_settings_from_db` already applies to a stock `.env` template line.
    """
    return (os.getenv("MANTLE_URI") or "").strip().rstrip("/")


def authorization_servers() -> list:
    """Every issuer this node accepts a user token from, most-preferred first.

    One definition because three things ask the same question and must not disagree:
    `browse_page._boot` (which issuer the sign-in button goes to), the RFC 9728 protected-resource
    metadata (which authorization server an MCP client should use), and the `WWW-Authenticate`
    challenge that points at that metadata. A node whose login button and whose machine-readable
    metadata named different issuers would send humans and clients to different places — and only
    one of the two would ever be noticed.

    The rule is the one `_boot` already applied: an external trusted issuer wins when configured
    (`AGIENCE_TRUSTED_ISSUERS[0]` is the browser's IdP — see the two-slot note in `services/oidc.py`),
    otherwise the platform authority. Order is load-bearing: index 0 is the one a browser is sent
    to, and a browser can be sent to exactly one provider.

    THE FALLBACK REQUIRES AN EXPLICIT VALUE, not merely a non-empty one. `ORIGIN_URI`'s Phase-1
    default is `http://localhost:8080` — a developer default, not a statement that an authorization
    server answers there — and `AUTHORITY_ISSUER` derives from it, so a standalone node with no
    Origin used to publish `authorization_servers: ["http://localhost:8080"]` in its RFC 9728
    metadata and send its sign-in button to the same place. An MCP client that reads that document
    dials `http://localhost:8080/.well-known/oauth-authorization-server`, gets nothing, and the flow
    dies at the one step whose whole purpose was to tell it where to go. Advertising a server that
    is not there is worse than advertising none: the empty answer is actionable and the wrong one
    is not. `declared_public_uri()` above draws the same line for `MANTLE_URI`, for the same reason
    and in the same words — with the same rule, too: what decides is whether an operator SAID it,
    not what the value happens to be.

    So a standalone node now names no authorization server, and `main.py` omits the key entirely
    rather than publishing an empty list. That IS the true state of such a node: it serves
    `/.well-known/oauth-protected-resource` and no other part of the OAuth surface — no
    `/.well-known/oauth-authorization-server`, no `/authorize`, no `/token`, no dynamic client
    registration — so no standards-compliant OAuth flow can complete against it, and a static
    `Authorization` header (see `scripts/dev_mint_token.py`, `mantle-token`) is the supported path.
    """
    external = [
        str(i["issuer"]).rstrip("/")
        for i in (TRUSTED_ISSUERS or [])
        if isinstance(i, dict) and i.get("issuer")
    ]
    if external:
        return external
    if not authority_is_declared():
        return []
    fallback = (AUTHORITY_ISSUER or ORIGIN_URI or "").rstrip("/")
    return [fallback] if fallback else []
