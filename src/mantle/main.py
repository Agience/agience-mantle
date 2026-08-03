import os
import sys

# ⛔ A `sys.path.insert(0, <mantle dir>)` BOOTSTRAP LIVED HERE AND IS DELETED (2026-07-29).
#
# It existed so bare imports (`routers`, `services`, …) resolved when the app was launched with
# `src/mantle` as the root. Every import in this package is now package-qualified (`mantle.routers`,
# `mantle.services`), so the shim has nothing left to rescue — and while it stayed, it was the thing
# KEEPING the dual-path hazard reachable: with both `<src>` and `<src>/mantle` on the path, the same
# module imports as two DISTINCT objects, and `except SomeError` silently stops matching across the
# seam because the two classes are not the same class.
#
# One shape now: `PYTHONPATH=<src>`, `uvicorn mantle.main:app`. The wheel and the running service are
# the same artifact. [John: "whatever is easiest for a user. least friction"]
from datetime import datetime, timezone
import asyncio
import json
import logging
from contextlib import asynccontextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

from mantle.db.backend import init_store, check_store_health  # noqa: E402  (backend-selected: lattice | lattice)
from mantle.search.init_search import init_search, reindex_in_background, shutdown_search  # noqa: E402
from mantle.search.ingest.index_queue import start_worker as start_index_worker, stop_worker as stop_index_worker  # noqa: E402
from mantle.routers.secrets_router import router as secrets_router  # noqa: E402
from mantle.routers.downloads_router import router as downloads_router  # noqa: E402
from mantle.routers.artifacts_router import router as artifacts_router  # noqa: E402
from mantle.routers.gate_router import gate_router  # noqa: E402
from mantle.routers.search_router import search_router  # noqa: E402
from mantle.routers.events_router import router as events_router  # noqa: E402
from mantle.routers.issuers_router import issuers_router  # noqa: E402
from mantle.routers.grants_router import router as grants_router  # noqa: E402
from mantle.routers.api_keys_router import router as api_keys_router  # noqa: E402
from mantle.routers.platform_router import router as platform_router  # noqa: E402
from mantle.routers.servers_router import router as servers_router  # noqa: E402
from mantle.routers.mcp_router import mcp_router  # noqa: E402
# Removed app/platform routers (Phase 2a — not database-layer surface; they
# belong to Agience/Origin or the persona services):
#   server_credentials_router, beacon_router, internal_personas_router,
#   stream_router. (events_router restored in 2b — DB data-change stream.)
from origin import config  # noqa: E402
from origin.logging_utils import build_log_config, configure_logging  # noqa: E402

# ----------------------------
# Logging setup (pre-Phase 2 — uses hardcoded defaults until config loads)
# ----------------------------
debug_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
# Apply the shared logging config in-process so timestamps land on uvicorn's
# own startup + access lines regardless of the --log-config CLI flag.
configure_logging()
logger = logging.getLogger("agience.api")

logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class _MCPClosedResourceFilter(logging.Filter):
    """Suppress benign ClosedResourceError noise from MCP SDK."""
    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            exc = record.exc_info[1]
            if type(exc).__name__ == "ClosedResourceError":
                return False
        return True


class _EventsEmitAccessFilter(logging.Filter):
    """Suppress POST /events/emit from uvicorn access log.

    Chat streaming emits dozens of delta events per turn; logging every one
    drowns out useful access log entries.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "POST /events/emit" in msg:
            return False
        return True


logging.getLogger("mcp.server.streamable_http").addFilter(_MCPClosedResourceFilter())
logging.getLogger("uvicorn.access").addFilter(_EventsEmitAccessFilter())
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx._client").setLevel(logging.ERROR)
# the legacy lexical index retired in Step 2.6.9 — no library logger to silence.

# ----------------------------
# Build Info
# ----------------------------
BUILD_INFO_PATH = os.getenv("BUILD_INFO_PATH", "/app/build_info.json")

def _load_build_info():
    for candidate in [BUILD_INFO_PATH, str(Path(__file__).resolve().parent.parent / "build_info.json")]:
        try:
            return json.loads(Path(candidate).read_text(encoding="utf-8"))
        except Exception:
            continue
    return {"version": "", "build_time": ""}

BUILD_INFO = _load_build_info()

# Setup mode is gone after 1.1e — Origin owns the setup wizard. The flag is
# kept at False so legacy references (root status, MCP discovery) keep
# returning "ok". The setup_mode_middleware below is now a no-op.
_setup_mode = False


# NOTE: platform provisioning (email sender, default LLM connection, operator
# bootstrap, seed loading) used to live here. It is a platform concern owned by
# Agience/Origin (platform mode), not the database layer, and has been removed
# from Mantle. See .dev/features/mantle-beacon-saas-split.md.


def _run_phase4_core_sync(loop) -> None:
    """Run core Phase 4 initialization (the store, seeding, operator bootstrap).

    Called from run_phase4_after_setup() via asyncio.to_thread(). Completing
    this function is sufficient to unblock platform routes (_setup_mode = False).
    Search initialization runs separately in _run_phase4_search_sync.

    """
    from prism.trust.key_manager import delete_setup_token as _delete_setup_token
    _delete_setup_token()

    store_db = init_store()

    try:
        from mantle.services import server_registry
        server_registry.load_from_store(store_db)  # reload persisted persona registrations
        from mantle.services.platform_topology import pre_resolve_platform_ids
        pre_resolve_platform_ids(store_db)
        server_registry.populate_ids()
    except Exception:
        logger.exception("Platform ID pre-resolution failed at startup (fatal)")
        raise

    # Start the index worker before seeding (same reason as in the main lifespan
    # path) so that enqueue_index_artifact() is async.  Idempotent if already running.
    try:
        start_index_worker()
    except Exception as e:
        logger.error("Failed to start index worker before post-setup seeding: %s", e)

    # Load platform settings so the bootstrap flag is readable.  Safe to call
    # again — platform_settings.load_all() overwrites the in-memory cache from DB.
    from mantle.services.platform_settings_service import settings as platform_settings
    platform_settings.load_all(store_db)

    # Mantle does not seed itself (database layer). Platform data is provisioned
    # by the application on top via the API — see the main lifespan note and
    # .dev/features/mantle-beacon-saas-split.md.
    _platform_seeded = platform_settings.get_bool("platform.bootstrap.seeded", default=False)

    # Platform provisioning (operator bootstrap, email, default LLM) is owned by
    # Agience/Origin in platform mode — not the database layer. See main lifespan.

    from mantle import event_bus
    event_bus.set_event_loop(loop)

    # Operation dispatch removed (Phase 2b) — Mantle is the database layer; the
    # operation/type runtime lives in the Chorus `core` gateway.

    # Ensure the content bucket exists post-setup (config.CONTENT_URI is now set
    # from the settings written by the setup wizard).
    try:
        from mantle.services.content_service import reinit_edge_clients, ensure_content_bucket
        reinit_edge_clients()
        ensure_content_bucket()
    except Exception:
        logger.warning("Content bucket provisioning after setup failed (non-fatal)", exc_info=True)


def _run_phase4_search_sync(*, is_post_setup: bool = False) -> None:
    """Run search init + index-worker startup.

    Post-the lexical-backend retirement (Step 2.6.9): no index creation step,
    no security provisioning. The encrypted MANTLE + SSE indexes
    bootstrap lazily on first commit. The only lifecycle work is the
    async index queue's worker.
    """
    init_search()
    try:
        start_index_worker()
    except Exception as e:
        logger.error("Failed to start indexing worker: %s", e)

    # Post-setup reindex: any seed content created before the indexer
    # came online needs a one-shot pass through the encrypted indexes.
    if is_post_setup:
        logger.info("Post-setup: reindexing all artifacts (background)...")
        reindex_in_background()


async def run_phase4_after_setup() -> None:
    """Run Phase 4 initialization in the background after setup completes.

    Called from the setup complete endpoint as a background task instead of
    restarting the process. The MCP server session manager is already running
    from Phase 3 so that step is skipped.

    Sets _setup_mode = False after core init (the lattice + seeding) so the
    frontend redirects promptly. Search init runs in background.
    """
    global _setup_mode
    logger.info("Phase 4 initialization starting (post-setup).")
    try:
        loop = asyncio.get_running_loop()
        await asyncio.to_thread(_run_phase4_core_sync, loop)
        _setup_mode = False
        logger.info("Phase 4 core complete — platform routes unblocked.")
        # Background reindex of any seed content created during setup.
        asyncio.create_task(_run_phase4_search_async(is_post_setup=True))
    except Exception:
        logger.exception("Phase 4 initialization failed after setup.")


async def _run_phase4_search_async(*, is_post_setup: bool = False) -> None:
    """Run search initialization in background without blocking platform routes."""
    try:
        await asyncio.to_thread(_run_phase4_search_sync, is_post_setup=is_post_setup)
        logger.info("Phase 4 search initialization complete.")
    except Exception:
        logger.exception("Phase 4 search initialization failed (non-fatal).")


# _ensure_operator_bootstrapped() and _run_platform_seed() removed — operator
# bootstrap and seed loading are platform concerns owned by Agience/Origin
# (platform mode), not the Mantle database layer. Mantle persists whatever the
# application provisions through its API. See
# .dev/features/mantle-beacon-saas-split.md.


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _setup_mode

    # Load MANTLE's own `.env` at STARTUP (defaults only; the environment already set wins). Explicit,
    # at startup — importing the shared `origin.config` must not pull origin's `.env` into this process.
    config.load_env(Path(__file__).resolve().parent.parent)

    # ⚠ LOG THE RESOLVED IDENTITY, BECAUSE GETTING IT WRONG IS SILENT UNTIL A CROSS-SERVICE CALL.
    # `AUTHORITY_ISSUER` is what inbound tokens are validated against — both `iss` and `aud`. If it
    # disagrees with what Origin stamps, mantle boots perfectly, serves unauthenticated routes
    # perfectly, and rejects every authenticated request with "Invalid token audience" — a message
    # that points at the CALLER when the fault is local configuration.
    #
    # It bit exactly that way on 2026-07-30: this service's `.env` carried
    # `AUTHORITY_ISSUER=http://localhost:8081` (mantle's OWN address, not Origin's). A shell that
    # set only `ORIGIN_URI` did not displace it — `load_env` is correctly `override=False`, so the
    # `.env` filled the variable the shell had left unset, and `AUTHORITY_ISSUER` outranks the
    # `ORIGIN_URI` fallback. Partially overriding a precedence chain is worse than overriding none
    # of it: the result is a blend of two configurations that neither side wrote.
    logger.info(
        "Identity resolved: AUTHORITY_ISSUER=%s (tokens must carry this as both iss and aud) "
        "ORIGIN_URI=%s MANTLE_URI=%s",
        config.AUTHORITY_ISSUER, config.ORIGIN_URI, getattr(config, "MANTLE_URI", "<unset>"),
    )

    # ------------------------------------------------------------------
    # Phase 0: Initialize all key material (filesystem)
    # ------------------------------------------------------------------
    try:
        # ⛔ `init_licensing_keys` REMOVED 2026-07-30 (John: "licensing does not matter at all").
        # It was a hard boot requirement for an Ed25519 keypair that NOTHING in mantle's request
        # path ever used — the only other references are a scope STRING in `entities/api_key.py`
        # and two standalone reporting CLIs under `scripts/`, none of which sign anything here.
        # So a service that serves the lattice refused to start over a key it would never reach
        # for. Deleting the call deletes a whole class of key material from every deployment.
        from prism.trust.key_manager import (
            init_encryption_key,
            init_setup_token,
            init_nonce_secret,
        )
        from mantle.services import peer_signing

        init_encryption_key()
        init_nonce_secret()
        init_setup_token()
        # Mantle signs its own outbound peer JWTs with mantle.private.pem via its
        # self-contained signer (no shared trust library). It does NOT sign user
        # tokens — only Origin does; inbound peer JWTs are verified generically.
        peer_signing.init()
    except Exception:
        logger.exception("Key initialization failed at startup (fatal)")
        raise

    # ------------------------------------------------------------------
    # Phase 1.5: Load bootstrap settings from key files
    # ------------------------------------------------------------------
    config.load_bootstrap_settings()
    from mantle.services.content_service import reinit_edge_clients
    reinit_edge_clients()

    # ------------------------------------------------------------------
    # Phase 2: Connect to databases, load platform settings
    # ------------------------------------------------------------------
    # Initialize the lattice (creates collections/indexes) — needed early
    # because platform settings now live in the lattice.
    store_db = init_store()

    # Load platform settings from the lattice into cache
    from mantle.services.platform_settings_service import settings as platform_settings
    platform_settings.load_all(store_db)

    # Rebind config module variables from DB settings. The provider is INJECTED
    # (2026-07-22): beam no longer imports the app's settings service — the app
    # hands its getter down (the core→app inversion, closed).
    config.set_settings_provider(platform_settings.get)
    config.load_settings_from_db()

    # Re-initialize edge S3 clients now that config.CONTENT_URI is set from DB,
    # then eagerly ensure the content bucket exists.
    from mantle.services.content_service import reinit_edge_clients, ensure_content_bucket
    reinit_edge_clients()
    try:
        ensure_content_bucket()
    except Exception:
        logger.warning("Content bucket check at startup failed (non-fatal)", exc_info=True)

    # OAuth provider registration moved to Origin in 1.1a-ii. Mantle no longer
    # serves /auth/authorize or /auth/callback.

    # Reconfigure logging with DB-loaded log level
    log_level = (config.BACKEND_LOG_LEVEL or "info").upper()
    logging.getLogger().setLevel(debug_level_map.get(log_level, logging.INFO))

    # ------------------------------------------------------------------
    # Phase 3 — setup gate is gone after 1.1e. Origin owns the setup wizard;
    # Mantle just starts up. Operators see setup state via Origin's /setup/status.
    # ------------------------------------------------------------------
    _setup_mode = False

    # init_setup_token() in Phase 0 recreates the file if it was deleted by
    # delete_setup_token() during setup completion.  Clean it up now that we
    # know setup is done so no orphan token lingers on disk between restarts.
    from prism.trust.key_manager import delete_setup_token as _delete_setup_token
    _delete_setup_token()

    # store_db already initialized in Phase 2

    # Pre-resolve platform singleton IDs (in-memory cache, needed on every startup)
    try:
        from mantle.services import server_registry
        server_registry.load_from_store(store_db)  # reload persisted persona registrations
        from mantle.services.platform_topology import pre_resolve_platform_ids
        pre_resolve_platform_ids(store_db)
        server_registry.populate_ids()
    except Exception:
        logger.exception("Platform ID pre-resolution failed at startup (fatal)")
        raise

    # Bootstrap-seed the platform's own trust (manifest service anchors +
    # AUTHORITY_ISSUER as role=platform) and env AGIENCE_TRUSTED_ISSUERS (role=external)
    # into governable issuer artifacts (#3 P1, idempotent create-if-missing). Runs
    # BEFORE the verifier load below so the refresh picks them up; before the event
    # loop is set, so it emits no events (no watcher noise). Non-fatal — the verifier
    # falls back to the manifest for anything not seeded.
    #
    # Runs as the platform system principal: startup has no request context, and
    # issuer artifacts are written (and read back) through the ordinary artifact
    # path, which now requires an identity to obtain a content key. The existing
    # non-fatal try/except is kept deliberately — a missing system principal must
    # not stop the process booting, and the seed then fails closed on its own.
    try:
        from mantle.services.issuers import seed_platform_issuer_artifacts
        from mantle.services.system_identity import system_acting_context
        with system_acting_context(scope="platform.issuer-seed"):
            seed_platform_issuer_artifacts(store_db)
    except Exception:
        logger.warning("Platform issuer seed failed (non-fatal; manifest fallback)", exc_info=True)

    # Load trusted-issuer artifacts into the token verifier. The authority manifest
    # + AGIENCE_TRUSTED_ISSUERS env are the bootstrap SEED; governable issuer
    # artifacts in the store are the source of truth going forward (the verifier
    # reads artifacts first, manifest fills only the gaps). Non-fatal.
    try:
        from mantle.services.oidc import get_oidc_verifier
        from mantle.services.system_identity import system_acting_context
        with system_acting_context(scope="platform.issuer-load"):
            get_oidc_verifier().refresh_from_db(store_db)
    except Exception:
        logger.warning("Trusted-issuer load at startup failed (non-fatal)", exc_info=True)

    # Start the index worker before seeding so that enqueue_index_artifact()
    # dispatches to the async queue rather than falling back to synchronous
    # per-artifact blocking. init_search() is a no-op post-the legacy lexical index; the
    # later _run_phase4_search_async task will skip re-starting an already
    # running worker (IndexQueue.start() is idempotent).
    try:
        start_index_worker()
    except Exception as e:
        logger.error("Failed to start index worker before seeding: %s", e)

    # Mantle is a database layer — it does NOT seed itself. The application on
    # top (Agience/Origin) provisions platform collections, grants and type
    # definitions into Mantle over its API at bootstrap (seeds are not Mantle's
    # concern). See .dev/features/mantle-beacon-saas-split.md. The flag below is
    # retained only to decide whether a post-boot reindex is warranted.
    _platform_seeded = platform_settings.get_bool("platform.bootstrap.seeded", default=False)

    # Operator bootstrap, platform email and default-LLM provisioning are platform
    # concerns — they live in Agience/Origin (platform mode), NOT in the database
    # layer. Mantle authenticates by verifying issuer-signed JWTs against the
    # authority JWKS and enforces cryptographic keyed access; it does not provision
    # platform identities. See .dev/features/mantle-beacon-saas-split.md.

    # Event bus — must be set before requests are served
    from mantle import event_bus
    event_bus.set_event_loop(asyncio.get_event_loop())

    # Access-audit "force": start the background flusher that drains buffered access
    # events (recorded in the authz layer) into the access_events edge collection.
    # Needs the running event loop; drained on shutdown so no witnessed access is lost.
    from mantle.services.audit_service import start_audit_worker
    start_audit_worker(store_db)

    # Refresh the verifier's trust set the instant an issuer artifact changes
    # (create/update/delete), so a governed issuer edit takes effect immediately.
    # The throttled refresh-on-miss in resolve_auth is the bounded fallback.
    from mantle.services.issuers import watch_issuer_changes
    from mantle.services.system_identity import system_acting_context

    # `create_task` snapshots the CURRENT context, so creating the task inside this
    # block gives the long-lived watcher the system identity for its whole lifetime
    # — it re-reads issuer artifacts on every change event and would otherwise have
    # no principal at all. The context manager exits immediately; the task keeps the
    # snapshot it took.
    try:
        with system_acting_context(scope="platform.issuer-watch"):
            issuer_watch_task = asyncio.create_task(watch_issuer_changes())
    except Exception:
        logger.warning("Issuer watcher not started (non-fatal)", exc_info=True)
        issuer_watch_task = None

    # Operation dispatch + type-definition resolution removed (Phase 2b). Mantle
    # stores content_type as an opaque label and emits change events; the
    # type/operation runtime (resolution, dispatch, warming) lives in the Chorus
    # `core` gateway. Type definitions remain ordinary artifacts in the store.

    # Search init in background (lazy bootstrap; won't block).
    # Pass is_post_setup=True when seeding just ran so reindex_in_background()
    # is triggered — seed sub-collection containers are indexed at creation time
    # but any that were skipped (e.g. S3 not ready) will be caught here.
    # ⛔ A FULL REINDEX IS NOT A BOOT TASK. This was gated on `not _platform_seeded` — a flag with no
    # writer, so it was always True and every start began a fresh 1,450,252-artifact pass.
    #
    # MEASURED 2026-08-01 on 71, after letting one run for 6.5 hours with S3 reachable and working:
    #
    #     4.8 CPU-hours burned, still inside the "a"s of the lexicon (cn-agitational)
    #     30 s sample: 19 indexed, 32 failed   <- MORE FAILURES THAN SUCCESSES
    #     ~38 artifacts/min  ->  ~636 hours (26 DAYS) for a full pass
    #
    # and the failures are `ClientError (OperationAborted): a conflicting conditional operation is
    # in progress` — 16 workers issuing PutObject against the SAME per-collection SSE manifest key.
    # The work is not merely slow, it is CONTENDING WITH ITSELF, and a pass that cannot finish
    # between restarts restarts from zero forever.
    #
    # So it is opt-in. `reindex_all_artifacts()` remains callable directly, which is what a rebuild
    # of a 2.15M-artifact index should be: a decision someone makes, scheduled, once — not something
    # a service quietly attempts every time it starts. Set MANTLE_REINDEX_ON_BOOT=1 to restore the
    # old behaviour. The write contention is the real defect and is NOT fixed by this line.
    _reindex_on_boot = os.getenv("MANTLE_REINDEX_ON_BOOT", "").strip().lower() in ("1", "true", "yes")
    if not _reindex_on_boot and not _platform_seeded:
        logger.info(
            "Boot reindex NOT started (a full pass measured ~26 days with majority S3 write "
            "conflicts). Set MANTLE_REINDEX_ON_BOOT=1 to restore, or call reindex_all_artifacts() "
            "explicitly. Search serves whatever is already indexed.")
    search_init_task = asyncio.create_task(_run_phase4_search_async(is_post_setup=_reindex_on_boot))

    yield

    # Cleanup on shutdown
    # May be None if the watcher failed to start (see above) — shutdown must not
    # raise AttributeError on the way out and mask the real startup failure.
    if issuer_watch_task is not None:
        issuer_watch_task.cancel()
        try:
            await issuer_watch_task
        except (asyncio.CancelledError, Exception):
            pass
    search_init_task.cancel()
    try:
        await search_init_task
    except (asyncio.CancelledError, Exception):
        pass
    try:
        stop_index_worker(drain=True, timeout=5.0)
    except Exception as e:
        logger.error(f"Failed to stop indexing worker: {e}")
    try:
        from mantle.services.audit_service import stop_audit_worker
        await stop_audit_worker(store_db, drain=True)
    except Exception as e:
        logger.error(f"Failed to stop audit worker: {e}")
    shutdown_search()

# ----------------------------
# Create app
# ----------------------------
# /docs (interactive Swagger console) is OFF by default — a live API console is an
# unnecessary surface in production. Dev/local enables it via AGIENCE_EXPOSE_DOCS=1.
# (/openapi.json stays available for tooling + health checks — schema, not a console.)
_EXPOSE_DOCS = os.getenv("AGIENCE_EXPOSE_DOCS", "").lower() in ("1", "true", "yes")

app = FastAPI(
    title="Agience API",
    version=BUILD_INFO.get("version") or "unknown",
    docs_url="/docs" if _EXPOSE_DOCS else None,
    redoc_url=None,
    lifespan=lifespan,
    swagger_ui_init_oauth={
        "clientId": "agience-docs-client",
        "usePkceWithAuthorizationCodeGrant": True,
    },
)

try:
    app.router.redirect_slashes = False
except Exception:
    pass

# ----------------------------
# Setup mode middleware
# ----------------------------
_SETUP_ALLOWED_PREFIXES = ("/setup", "/version", "/.well-known", "/docs", "/openapi.json", "/auth/token", "/server-credentials")

@app.middleware("http")
async def setup_mode_middleware(request: Request, call_next):
    """When in setup mode, only allow setup-related routes."""
    if _setup_mode:
        path = request.url.path
        if path == "/" or any(path.startswith(prefix) for prefix in _SETUP_ALLOWED_PREFIXES):
            return await call_next(request)
        return JSONResponse(
            status_code=503,
            content={"detail": "Setup required", "setup_url": "/setup"},
        )
    return await call_next(request)

# ----------------------------
# Error logging handlers
# ----------------------------

# Fields whose values must never appear in logs.
_REDACT_KEYS = frozenset({"password", "secret", "token", "api_key", "apikey",
                          "access_token", "refresh_token", "credential",
                          "passkey_credential", "passkey_challenge"})

def _redact_body(raw: bytes, max_len: int = 2048) -> str:
    """Return a log-safe representation of a request body.

    JSON bodies have sensitive fields replaced with '***'. Non-JSON
    bodies are truncated to *max_len* bytes.
    """
    if not raw:
        return ""
    try:
        import json as _json
        obj = _json.loads(raw)
        if isinstance(obj, dict):
            obj = {k: ("***" if k.lower() in _REDACT_KEYS else v)
                   for k, v in obj.items()}
        return _json.dumps(obj, ensure_ascii=False)[:max_len]
    except Exception:
        return repr(raw[:max_len])


@app.exception_handler(HTTPException)
async def http_exception_logger(request: Request, exc: HTTPException):
    try:
        body = await request.body()
    except Exception:
        body = b""
    logger.warning(
        "HTTP %s %s %s user=%s ws=%s artifact=%s detail=%r body=%s",
        exc.status_code,
        request.method,
        request.url.path,
        getattr(request.state, "user_id", None),
        request.path_params.get("workspace_id"),
        request.path_params.get("artifact_id"),
        exc.detail,
        _redact_body(body),
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def unhandled_exception_logger(request: Request, exc: Exception):
    try:
        body = await request.body()
    except Exception:
        body = b""
    logger.exception(
        "HTTP 500 %s %s user=%s ws=%s artifact=%s body=%s error=%s",
        request.method,
        request.url.path,
        getattr(request.state, "user_id", None),
        request.path_params.get("workspace_id"),
        request.path_params.get("artifact_id"),
        _redact_body(body),
        repr(exc),
    )
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# ----------------------------
# CORS & Session
# ----------------------------
# allow_origins=["*"] is safe: all authenticated API calls use Bearer tokens,
# not cookies, so cross-origin requests without credentials pose no CSRF risk.
# The OAuth redirect flow uses session cookies but those are same-site
# (browser navigations, not XHR), so CORS doesn't apply to them.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Security headers — cheap hardening every scanner expects.
# ----------------------------
@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    # Enforced only over HTTPS (harmless over the in-network HTTP hop); the edge terminates TLS.
    response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response

# ----------------------------
# Rate limiting — per-client in-memory sliding window (single worker). Protects the
# whole surface (incl. API-key verification) from floods / credential-stuffing. Set
# AGIENCE_RATE_LIMIT_PER_MIN=0 to disable. Behind the edge, X-Forwarded-For is the client.
# ----------------------------
_RL_WINDOW_S = 60.0
_RL_MAX = int(os.getenv("AGIENCE_RATE_LIMIT_PER_MIN", "600"))
_RL_EXEMPT = frozenset(("/status", "/version", "/openapi.json", "/docs"))
_RL_HITS: dict = {}

@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    if _RL_MAX <= 0 or request.url.path in _RL_EXEMPT:
        return await call_next(request)
    import time as _time
    from collections import deque as _deque
    xff = request.headers.get("x-forwarded-for")
    ip = (xff.split(",")[0].strip() if xff else (request.client.host if request.client else "unknown"))
    now = _time.monotonic()
    dq = _RL_HITS.get(ip)
    if dq is None:
        dq = _deque()
        _RL_HITS[ip] = dq
    cutoff = now - _RL_WINDOW_S
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) >= _RL_MAX:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
            headers={"Retry-After": str(int(_RL_WINDOW_S))},
        )
    dq.append(now)
    return await call_next(request)

# SessionMiddleware moved to Origin alongside the OAuth flows. Mantle no longer
# carries the OAuth PKCE state cookie because it no longer serves /auth/authorize.

# ----------------------------
# Routers
# ----------------------------
# Database-layer surface. App/platform routers (server_credentials, beacon,
# internal_personas, stream) were removed in Phase 2a. events_router is the
# outbound data-change stream (WS /events), restored in 2b.
app.include_router(secrets_router)
app.include_router(downloads_router)
app.include_router(artifacts_router)
app.include_router(gate_router)
app.include_router(search_router)
app.include_router(events_router)
app.include_router(issuers_router)
app.include_router(grants_router)
app.include_router(api_keys_router)
app.include_router(platform_router)
app.include_router(servers_router)
app.include_router(mcp_router)
# types_router (/types type-definition API) unmounted in 2b cleanup — type
# definitions are ordinary artifacts; type resolution lives in the Chorus gateway.

# ----------------------------
# MCP surface moved to chorus's universal gateway. Clients address Mantle's
# platform ops at chorus.example.com/{platform_artifact_id}/mcp via the
# `core` persona registered in chorus/manifest.json. Mantle itself no
# longer publishes /mcp — see .dev/features/mantle-mcp-consolidation.md.

# ----------------------------
# Basic routes
# ----------------------------
def utcnow_z() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _explorer_html(payload: dict) -> str:
    """The human door: a self-contained status + entry page, no build step and no external asset.

    Deliberately inline. A separate CSS/JS bundle would make the ONE page that must render when
    everything else is broken depend on another request succeeding.
    """
    status = payload["status"]
    ok = status == "ok"
    return f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agience · Mantle</title>
<style>
:root{{color-scheme:light dark;--bg:#0b0d12;--panel:#151924;--line:#222839;--txt:#e8eaf0;
--mut:#9aa4bf;--accent:#5b6cff;--ok:#3fb950;--warn:#d29922}}
*{{box-sizing:border-box}}html,body{{height:100%;margin:0}}
body{{font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
background:var(--bg);color:var(--txt);display:flex;justify-content:center;padding:3rem 1.25rem}}
main{{width:100%;max-width:46rem}}
h1{{font-size:1.5rem;margin:0 0 .25rem;letter-spacing:-.01em}}
.sub{{color:var(--mut);margin:0 0 2rem}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:1.25rem;
margin-bottom:1rem}}
.row{{display:flex;justify-content:space-between;gap:1rem;padding:.4rem 0;border-bottom:1px solid var(--line)}}
.row:last-child{{border-bottom:0}}
.k{{color:var(--mut)}} .v{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.dot{{display:inline-block;width:.55rem;height:.55rem;border-radius:50%;margin-right:.45rem;
background:{'var(--ok)' if ok else 'var(--warn)'}}}
a{{color:var(--accent);text-decoration:none}} a:hover{{text-decoration:underline}}
ul{{margin:.5rem 0 0;padding-left:1.1rem}} li{{margin:.3rem 0}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#0f1320;
border:1px solid var(--line);border-radius:5px;padding:.1rem .35rem}}
</style></head><body><main>
<h1>Agience &middot; Mantle</h1>
<p class="sub">The lattice — an encrypted artifact store where authorization <em>is</em> the encryption.</p>
<div class="card">
  <div class="row"><span class="k">Status</span><span class="v"><span class="dot"></span>{status}</span></div>
  <div class="row"><span class="k">Version</span><span class="v">{payload['version']}</span></div>
  <div class="row"><span class="k">Server time</span><span class="v">{payload['server_time']}</span></div>
</div>
<div class="card">
  <strong>Interfaces</strong>
  <ul>
    <li><code>/mcp</code> — Model Context Protocol over Streamable HTTP. Point any MCP client here.</li>
    <li><code>/artifacts</code> — the REST API. Every read is filtered by your grants.</li>
    <li><code>/status</code> — store health.</li>
  </ul>
</div>
<div class="card">
  <strong>Access</strong>
  <p style="color:var(--mut);margin:.5rem 0 0">Sign in through
  <a href="https://origin.agience.ai">origin.agience.ai</a>. What you can see here is exactly what
  your grants reach — an empty result means &ldquo;nothing you may see&rdquo;, not
  &ldquo;nothing exists&rdquo;.</p>
</div>
</main></body></html>"""

@app.get("/", include_in_schema=False)
def read_root(request: Request):
    """One URL, two audiences — content-negotiated.

    A browser asks for `text/html` and gets the explorer; a client asks for JSON (or anything
    else) and gets the machine payload. Serving HTML unconditionally breaks every script that
    parses this endpoint; serving JSON unconditionally means a human who opens the hostname sees
    a wall of braces and no way in. The `Accept` header already carries the answer, so neither
    audience needs a different URL to remember.

    ⚠ `Vary: Accept` is not decoration. Without it a CDN or browser cache can serve the HTML
    body to a JSON client (or the reverse) from a cache entry keyed only on the URL — a failure
    that appears only under a cache and is invisible in local testing.
    """
    payload = {
        "status": "ok" if not _setup_mode else "setup_required",
        "version": BUILD_INFO.get("version") or "unknown",
        "server_time": utcnow_z(),
        "links": {
            "self": "/",
            "mcp": "/mcp",
            "service-doc": "/docs",
            "service-desc": "/openapi.json",
        },
    }
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(
            content=_explorer_html(payload),
            status_code=200,
            headers={"Cache-Control": "no-store", "Vary": "Accept"},
        )
    return JSONResponse(
        content=payload,
        status_code=200,
        headers={"Cache-Control": "no-store", "Vary": "Accept"},
    )

@app.get("/status", include_in_schema=False)
def check_backend_status():
    status = {
        **check_store_health(),
    }
    return status

@app.get("/version", include_in_schema=False)
def version():
    return {
        "version": BUILD_INFO.get("version") or "",
        "build_time": BUILD_INFO.get("build_time") or ""
    }

# ----------------------------
# Entrypoint (dev)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8081,
        reload=True,
        reload_dirs=["mantle", "platform"],
        reload_excludes=["**/__pycache__/*", "**/*.pyc"],
        log_level="info",
        log_config=build_log_config(),
        workers=1,
        server_header=False,
        timeout_graceful_shutdown=3,
    )
