import os
import sys

# Ensure bare imports (`mcp_server`, `routers`, `services`, ...) resolve when
# launched as `python -m mantle.main` from the repo root. The repo root is
# already on sys.path via -m semantics; this adds the mantle/ directory itself.
_MANTLE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MANTLE_DIR not in sys.path:
    sys.path.insert(0, _MANTLE_DIR)

# E402: the sys.path bootstrap above must run before these package imports.
from datetime import datetime, timezone  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from schemas.arango.loader import init_arango_db, check_arango_health  # noqa: E402
from search.init_search import init_search, reindex_in_background, shutdown_search  # noqa: E402
from search.ingest.index_queue import start_worker as start_index_worker, stop_worker as stop_index_worker  # noqa: E402
from routers.secrets_router import router as secrets_router  # noqa: E402
from routers.downloads_router import router as downloads_router  # noqa: E402
from routers.artifacts_router import router as artifacts_router  # noqa: E402
from routers.gate_router import gate_router  # noqa: E402
from routers.search_router import search_router  # noqa: E402
from routers.events_router import router as events_router  # noqa: E402
from routers.issuers_router import issuers_router  # noqa: E402
# Removed app/platform routers (Phase 2a — not database-layer surface; they
# belong to Agience/Origin or the persona services):
#   server_credentials_router, beacon_router, internal_personas_router,
#   stream_router. (events_router restored in 2b — DB data-change stream.)
from agience_core import config  # noqa: E402
from agience_core.logging_utils import build_log_config, configure_logging  # noqa: E402

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
# OpenSearch retired in Step 2.6.9 — no library logger to silence.

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
    """Run core Phase 4 initialization (ArangoDB, seeding, operator bootstrap).

    Called from run_phase4_after_setup() via asyncio.to_thread(). Completing
    this function is sufficient to unblock platform routes (_setup_mode = False).
    OpenSearch initialization runs separately in _run_phase4_search_sync.
    """
    from agience_core.key_manager import delete_setup_token as _delete_setup_token
    _delete_setup_token()

    arango_db = init_arango_db()

    try:
        from services.platform_topology import pre_resolve_platform_ids
        pre_resolve_platform_ids(arango_db)
        from services import server_registry
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
    from services.platform_settings_service import settings as platform_settings
    platform_settings.load_all(arango_db)

    # Mantle does not seed itself (database layer). Platform data is provisioned
    # by the application on top via the API — see the main lifespan note and
    # .dev/features/mantle-beacon-saas-split.md.
    _platform_seeded = platform_settings.get_bool("platform.bootstrap.seeded", default=False)

    # Platform provisioning (operator bootstrap, email, default LLM) is owned by
    # Agience/Origin in platform mode — not the database layer. See main lifespan.

    from agience_core import event_bus
    event_bus.set_event_loop(loop)

    # Operation dispatch removed (Phase 2b) — Mantle is the database layer; the
    # operation/type runtime lives in the Chorus `core` gateway.

    # Ensure the content bucket exists post-setup (config.CONTENT_URI is now set
    # from the settings written by the setup wizard).
    try:
        from services.content_service import reinit_edge_clients, ensure_content_bucket
        reinit_edge_clients()
        ensure_content_bucket()
    except Exception:
        logger.warning("Content bucket provisioning after setup failed (non-fatal)", exc_info=True)


def _run_phase4_search_sync(*, is_post_setup: bool = False) -> None:
    """Run search init + index-worker startup.

    Post-OpenSearch retirement (Step 2.6.9): no index creation step,
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

    Sets _setup_mode = False after core init (ArangoDB + seeding) so the
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

    # ------------------------------------------------------------------
    # Phase 0: Initialize all key material (filesystem)
    # ------------------------------------------------------------------
    try:
        from agience_core.key_manager import (
            init_licensing_keys,
            init_encryption_key,
            init_setup_token,
            init_arango_password, init_minio_password,
            init_nonce_secret,
        )
        from services import peer_signing

        init_licensing_keys()
        init_encryption_key()
        init_nonce_secret()
        init_setup_token()
        init_arango_password()
        init_minio_password()
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
    from services.content_service import reinit_edge_clients
    reinit_edge_clients()

    # ------------------------------------------------------------------
    # Phase 2: Connect to databases, load platform settings
    # ------------------------------------------------------------------
    # Initialize ArangoDB (creates collections/indexes) — needed early
    # because platform settings now live in ArangoDB.
    arango_db = init_arango_db()

    # Load platform settings from ArangoDB into cache
    from services.platform_settings_service import settings as platform_settings
    platform_settings.load_all(arango_db)

    # Rebind config module variables from DB settings
    config.load_settings_from_db()

    # Re-initialize edge S3 clients now that config.CONTENT_URI is set from DB,
    # then eagerly ensure the content bucket exists.
    from services.content_service import reinit_edge_clients, ensure_content_bucket
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
    from agience_core.key_manager import delete_setup_token as _delete_setup_token
    _delete_setup_token()

    # arango_db already initialized in Phase 2

    # Pre-resolve platform singleton IDs (in-memory cache, needed on every startup)
    try:
        from services.platform_topology import pre_resolve_platform_ids
        pre_resolve_platform_ids(arango_db)
        from services import server_registry
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
    try:
        from services.issuers import seed_platform_issuer_artifacts
        seed_platform_issuer_artifacts(arango_db)
    except Exception:
        logger.warning("Platform issuer seed failed (non-fatal; manifest fallback)", exc_info=True)

    # Load trusted-issuer artifacts into the token verifier. The authority manifest
    # + AGIENCE_TRUSTED_ISSUERS env are the bootstrap SEED; governable issuer
    # artifacts in the store are the source of truth going forward (the verifier
    # reads artifacts first, manifest fills only the gaps). Non-fatal.
    try:
        from services.oidc import get_oidc_verifier
        get_oidc_verifier().refresh_from_db(arango_db)
    except Exception:
        logger.warning("Trusted-issuer load at startup failed (non-fatal)", exc_info=True)

    # Start the index worker before seeding so that enqueue_index_artifact()
    # dispatches to the async queue rather than falling back to synchronous
    # per-artifact blocking. init_search() is a no-op post-OpenSearch; the
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
    from agience_core import event_bus
    event_bus.set_event_loop(asyncio.get_event_loop())

    # Refresh the verifier's trust set the instant an issuer artifact changes
    # (create/update/delete), so a governed issuer edit takes effect immediately.
    # The throttled refresh-on-miss in resolve_auth is the bounded fallback.
    from services.issuers import watch_issuer_changes
    issuer_watch_task = asyncio.create_task(watch_issuer_changes())

    # Operation dispatch + type-definition resolution removed (Phase 2b). Mantle
    # stores content_type as an opaque label and emits change events; the
    # type/operation runtime (resolution, dispatch, warming) lives in the Chorus
    # `core` gateway. Type definitions remain ordinary artifacts in the store.

    # Search init in background (lazy bootstrap; won't block).
    # Pass is_post_setup=True when seeding just ran so reindex_in_background()
    # is triggered — seed sub-collection containers are indexed at creation time
    # but any that were skipped (e.g. S3 not ready) will be caught here.
    search_init_task = asyncio.create_task(_run_phase4_search_async(is_post_setup=not _platform_seeded))

    yield

    # Cleanup on shutdown
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
    shutdown_search()

# ----------------------------
# Create app
# ----------------------------
app = FastAPI(
    title="Agience API",
    version=BUILD_INFO.get("version") or "unknown",
    docs_url="/docs",
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

@app.get("/", include_in_schema=False)
def read_root():
    payload = {
        "status": "ok" if not _setup_mode else "setup_required",
        "version": BUILD_INFO.get("version") or "unknown",
        "server_time": utcnow_z(),
        "links": {
            "self": "/",
            "service-doc": "/docs",
            "service-desc": "/openapi.json",
        },
    }
    return JSONResponse(
        content=payload,
        status_code=200,
        headers={"Cache-Control": "no-store"}
    )

@app.get("/status", include_in_schema=False)
def check_backend_status():
    status = {
        **check_arango_health(),
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
