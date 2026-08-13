"""The Mantle ASGI application — the whole HTTP surface of the lattice, in one FastAPI app.

Mantle is the database layer: an encrypted-by-default artifact store where authorization *is*
the encryption. This module owns the process, not the domain. It boots key material and the
store, mounts the routers, installs the middleware, and serves the handful of routes that
belong to the service itself rather than to any router — `/`, `/auth/callback`, `/status`,
`/version`, `/.well-known/oauth-protected-resource`, and `/docs` + `/openapi.json`, which are
always served because the schema is the API's contract and every route behind it enforces its
own authorization.

Lifespan — the order is the contract
------------------------------------
Each step needs what the step before it produced, which is why they are numbered in the source
and why none of them may be reordered casually.

``load_env``
    Mantle's own ``.env``, defaults only: the environment already set wins.
Phase 0 — key material
    ``init_encryption_key`` / ``init_nonce_secret`` / ``init_setup_token`` from the trust floor,
    then ``peer_signing.init()``. Fatal on failure: nothing below can run without keys.
Phase 1.5 — bootstrap settings
    ``config.load_bootstrap_settings()`` reads what the key files carry, then the edge S3
    clients are rebuilt from it.
Phase 2 — the store, then settings
    ``init_store()`` opens the lattice; platform settings load out of it and are injected back
    into ``config`` as a provider, so the settings surface is complete before any request. Edge
    clients are rebuilt a second time against the DB-supplied ``CONTENT_URI`` and the content
    bucket is ensured; the log level is reapplied from the DB value.
Phase 3 — setup
    Origin owns the setup wizard, so this only clears the setup token.
Phase 4 — platform identity, trust, workers
    Platform singleton ids are pre-resolved (fatal on failure); trusted-issuer artifacts are
    seeded and loaded into the token verifier (non-fatal — the authority manifest is the
    fallback); the index worker starts; the event bus is given the loop, its durable log and any
    configured back-plane, and refuses a multi-worker boot without one; the audit flusher starts;
    the issuer watcher and background search initialization are launched as tasks.

Shutdown reverses it: the watcher and search tasks are cancelled, the index worker and audit
worker drain, and search shuts down. Boot is the only path into Phase 4 — Origin owns the setup
wizard, so there is no post-setup callback that re-enters it here.

Registration
------------
Everything below the ``lifespan`` definition is wiring, in source order: the ``FastAPI`` app,
then the ``HTTPException`` and unhandled-exception handlers (which also attach the RFC 9728
``WWW-Authenticate`` challenge to every 401), CORS, security headers, the per-client rate
limiter, and then the routers — ``artifacts``, ``events``,
``grants``, ``system``, ``mcp``. Those five are the whole mounted surface;
``tests/test_every_router_is_mounted.py`` holds it that way, and
``tests/test_route_reshape.py`` pins the paths.
"""

import os

# No `sys.path` manipulation here, and that is the point of the next paragraph.
# Launched as `PYTHONPATH=<src> uvicorn mantle.main:app`. Every import in this package is
# package-qualified (`mantle.routers`, `mantle.services`), so there is exactly one path by which
# any module resolves — with both `<src>` and `<src>/mantle` on the path, the same module would
# import as two distinct objects, and `except SomeError` would silently stop matching across the
# seam because the two classes are not the same class. The wheel and the running service are the
# same artifact.
from datetime import datetime, timezone
import asyncio
import ipaddress
import json
import logging
import time
from collections import OrderedDict, deque  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

from mantle.db.backend import init_store, check_store_health  # noqa: E402  (lattice-backed store)
from mantle.events import event_backplane
from mantle.search.init_search import init_search, reindex_in_background, shutdown_search  # noqa: E402
from mantle.search.ingest.index_queue import start_worker as start_index_worker, stop_worker as stop_index_worker  # noqa: E402
from mantle.routers.artifacts_router import router as artifacts_router  # noqa: E402
from mantle.routers.events_router import router as events_router  # noqa: E402
from mantle.routers.grants_router import router as grants_router  # noqa: E402
from mantle.routers.system_router import router as system_router  # noqa: E402
from mantle.routers.mcp_router import mcp_router  # noqa: E402
# beacon_router, internal_personas_router, and stream_router are not mounted here — they are not
# database-layer surface; they belong to Agience/Origin or the persona services.
# events_router is the outbound data-change stream (WS /events).
from mantle.ui import browse_page as _browse_page  # noqa: E402
from mantle import config  # noqa: E402
from mantle.system.logging_utils import build_log_config, configure_logging  # noqa: E402

# ----------------------------
# Logging setup — uses hardcoded defaults until config loads from the database in Phase 2 below.
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


# No access-log filter for the MCP surface: `routers/mcp_router` speaks JSON-RPC over
# Streamable HTTP directly, so no SDK logger (`mcp.server.*`) exists in this process to filter.
# `/events` is a WebSocket — one accept line per connection, not a per-message log.
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx._client").setLevel(logging.ERROR)

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

def _run_phase4_search_sync(*, is_post_setup: bool = False) -> None:
    """Run search init + index-worker startup.

    No index-creation step and no security provisioning: the encrypted MANTLE + SSE indexes
    bootstrap lazily on first commit. The only lifecycle work here is starting the async index
    queue's worker.
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


async def _run_phase4_search_async(*, is_post_setup: bool = False) -> None:
    """Run search initialization in background without blocking platform routes."""
    try:
        await asyncio.to_thread(_run_phase4_search_sync, is_post_setup=is_post_setup)
        logger.info("Phase 4 search initialization complete.")
    except Exception:
        logger.exception("Phase 4 search initialization failed (non-fatal).")


# Operator bootstrap and seed loading are platform concerns owned by Agience/Origin (platform
# mode), not the Mantle database layer. Mantle persists whatever the application provisions
# through its API. See .dev/features/mantle-beacon-saas-split.md.


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load MANTLE's own `.env` at startup (defaults only; the environment already set wins).
    # Explicit, at startup — importing the shared `origin.config` must not pull origin's `.env`
    # into this process.
    config.load_env(Path(__file__).resolve().parent.parent)

    # `load_env` is `override=False`, so `.env` only fills variables the shell left unset, and
    # `AUTHORITY_ISSUER` outranks the `ORIGIN_URI` fallback. A shell that sets only `ORIGIN_URI`
    # therefore does not displace an `AUTHORITY_ISSUER` already carried in `.env` — partially
    # overriding a precedence chain produces a blend of configurations that neither side wrote.
    logger.info(
        "Identity resolved: AUTHORITY_ISSUER=%s (tokens must carry this as both iss and aud) "
        "ORIGIN_URI=%s MANTLE_URI=%s",
        config.AUTHORITY_ISSUER, config.ORIGIN_URI, getattr(config, "MANTLE_URI", "<unset>"),
    )

    # ------------------------------------------------------------------
    # Phase 0: Initialize all key material (filesystem)
    # ------------------------------------------------------------------
    try:
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

    # Rebind config module variables from DB settings. The provider is injected: the app hands its
    # settings getter down rather than the core importing the app's settings service directly.
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

    # OAuth provider registration (the /auth/authorize authorization endpoint) lives in Origin.
    # Mantle serves /auth/callback only as the browser document below — the code exchange happens
    # client-side against Origin's token endpoint.

    # Reconfigure logging with DB-loaded log level
    log_level = (config.BACKEND_LOG_LEVEL or "info").upper()
    logging.getLogger().setLevel(debug_level_map.get(log_level, logging.INFO))

    # ------------------------------------------------------------------
    # Phase 3 — Origin owns the setup wizard; Mantle just starts up. Operators
    # see setup state via Origin's /setup/status.
    # ------------------------------------------------------------------
    # init_setup_token() in Phase 0 recreates the file if it was deleted by
    # delete_setup_token() during setup completion.  Clean it up now that we
    # know setup is done so no orphan token lingers on disk between restarts.
    from prism.trust.key_manager import delete_setup_token as _delete_setup_token
    _delete_setup_token()

    # store_db already initialized in Phase 2

    # Pre-resolve platform singleton IDs (in-memory cache, needed on every startup)
    try:
        from mantle.services.platform_topology import pre_resolve_platform_ids
        pre_resolve_platform_ids(store_db)
    except Exception:
        logger.exception("Platform ID pre-resolution failed at startup (fatal)")
        raise

    # Bootstrap-seeds the platform's own trust (manifest service anchors +
    # AUTHORITY_ISSUER as role=platform) and env AGIENCE_TRUSTED_ISSUERS (role=external)
    # into governable issuer artifacts (idempotent create-if-missing). Runs before the
    # verifier load below so the refresh picks them up, and before the event loop is
    # set so it emits no watcher events. Non-fatal — the verifier falls back to the
    # manifest for anything not seeded.
    #
    # Runs as the platform system principal: startup has no request context, and
    # issuer artifacts are written and read back through the ordinary artifact path,
    # which requires an identity to obtain a content key. The non-fatal try/except
    # means a missing system principal does not stop the process booting; the seed
    # then fails closed on its own.
    try:
        from mantle.services.issuers import seed_platform_issuer_artifacts
        from mantle.services.system_identity import system_acting_context
        with system_acting_context(scope="platform.issuer-seed"):
            seed_platform_issuer_artifacts(store_db)
    except Exception:
        logger.warning("Platform issuer seed failed (non-fatal; manifest fallback)", exc_info=True)

    # Load trusted-issuer artifacts into the token verifier. The authority manifest
    # + AGIENCE_TRUSTED_ISSUERS env are the bootstrap seed; governable issuer
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
    # per-artifact blocking. init_search() is a no-op here (no index to create); the
    # later _run_phase4_search_async task will skip re-starting an already
    # running worker (IndexQueue.start() is idempotent).
    try:
        start_index_worker()
    except Exception as e:
        logger.error("Failed to start index worker before seeding: %s", e)

    # Mantle is a database layer — it does not seed itself. The application on
    # top (Agience/Origin) provisions platform collections, grants and type
    # definitions into Mantle over its API at bootstrap (seeds are not Mantle's
    # concern). See .dev/features/mantle-beacon-saas-split.md. The flag below is
    # retained only to decide whether a post-boot reindex is warranted.
    _platform_seeded = platform_settings.get_bool("platform.bootstrap.seeded", default=False)

    # Operator bootstrap, platform email and default-LLM provisioning are platform
    # concerns — they live in Agience/Origin (platform mode), not in the database
    # layer. Mantle authenticates by verifying issuer-signed JWTs against the
    # authority JWKS and enforces cryptographic keyed access; it does not provision
    # platform identities. See .dev/features/mantle-beacon-saas-split.md.

    # Event bus — must be set before requests are served.
    #
    # The durable log and the back-plane are both optional tiers over an in-process core, the same
    # shape the content tier uses for S3: the bus is fully functional with neither. The back-plane
    # only becomes required when more than one worker serves the app, because an in-process fanout
    # cannot reach a subscriber attached to a different process — `require_backplane_for_workers`
    # refuses that combination at boot rather than letting the change feed lose events silently.
    from mantle.events import event_bus
    event_bus.set_event_loop(asyncio.get_event_loop())
    event_bus.set_event_log(event_bus.open_event_log(store_db))
    event_bus.set_backplane(event_backplane.backplane_from_env())
    event_backplane.require_backplane_for_workers(
        os.getenv("MANTLE_WORKERS", "1"), event_bus.backplane())

    # Access-audit "force": start the background flusher that drains buffered access
    # events (recorded in the authz layer) into the access_events edge collection.
    # Needs the running event loop; drained on shutdown so no witnessed access is lost.
    from mantle.services.audit_service import start_audit_worker
    start_audit_worker(store_db)

    # The mirror drain: retry the object-store leg this node still owes for content whose upload
    # succeeded locally while the mirror was unreachable (`op.content.mirror`, recorded by
    # `routers/artifacts_router._record_mirror_pending`).
    #
    # HERE, and not in `mesh/daemon.py` or a CLI, because the work is bound to this process's
    # disk: the task names a local CAS ref, so only the node that took the upload can complete it,
    # and inside that node only a process holding the same content tier and the same configured S3
    # client can do it without resolving all of that a second time. The mesh daemon is a per-box
    # ember process that reaches operators through a wired runner mantle does not have; a CLI would
    # be a second claimant on the same pool, which would oblige mantle to grow the stale-claim
    # reclamation it currently has nowhere.
    #
    # `start_mirror_drain` DECLINES on a node with no object store, so an air-gapped node creates
    # no task and schedules no wake-up. When it does start, it never polls: it sleeps until the
    # queue's own earliest `next_retry_at` and is woken by the upload path.
    from mantle.services.mirror_drain import start_mirror_drain
    try:
        start_mirror_drain(store_db)
    except Exception:
        logger.warning("Mirror drain not started (non-fatal)", exc_info=True)

    # Refresh the verifier's trust set the instant an issuer artifact changes
    # (create/update/delete), so a governed issuer edit takes effect immediately.
    # The throttled refresh-on-miss in resolve_auth is the bounded fallback.
    from mantle.services.issuers import watch_issuer_changes
    from mantle.services.system_identity import system_acting_context

    # `create_task` snapshots the current context, so creating the task inside this
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

    # Mantle stores content_type as an opaque label and emits change events; no
    # type/operation runtime (resolution, dispatch, warming) runs here. Type
    # definitions remain ordinary artifacts in the store.

    # Search init in background (lazy bootstrap; won't block).
    # Pass is_post_setup=True when seeding just ran so reindex_in_background()
    # is triggered — seed sub-collection containers are indexed at creation time
    # but any that were skipped (e.g. S3 not ready) will be caught here.
    #
    # A full reindex-on-boot is opt-in rather than automatic: many workers issuing PutObject against
    # the same per-collection SSE manifest key contend with each other, so throughput is dominated
    # by `ClientError (OperationAborted): a conflicting conditional operation is in progress` rather
    # than by indexing work, and a pass that cannot finish between restarts restarts from zero every
    # time.
    #
    # `reindex_all_artifacts()` remains callable directly, which is what a full-index rebuild should
    # be: a decision someone makes, scheduled, once — not something a service attempts every time it
    # starts. Set MANTLE_REINDEX_ON_BOOT=1 to restore that behavior. The write contention itself is
    # unaffected by this flag.
    _reindex_on_boot = os.getenv("MANTLE_REINDEX_ON_BOOT", "").strip().lower() in ("1", "true", "yes")
    if not _reindex_on_boot and not _platform_seeded:
        logger.info(
            "Boot reindex NOT started (a full pass can take weeks under S3 write "
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
    try:
        # Cancelled, not drained: an in-flight mirror write holds a claim, and `settle` compares on
        # `claimed_by`, so an attempt killed mid-flight simply never settles. That row is recovered
        # by the next boot — `_reclaim_own` returns any task still claimed under this worker's own
        # `<origin>:<pid>` to `pending` before its first pass — so nothing is lost by stopping,
        # while waiting would mean waiting on a WAN write to the store that is, by construction,
        # the one not answering.
        from mantle.services.mirror_drain import stop_mirror_drain
        await stop_mirror_drain()
    except Exception as e:
        logger.error(f"Failed to stop mirror drain: {e}")
    shutdown_search()

# ----------------------------
# Create app
# ----------------------------
# `/docs` and `/openapi.json` are always served. The schema is the API's contract, not a
# secret: every route behind it enforces its own authorization, so a reader learns the shape
# and still gets a 401. Serving it unconditionally is what makes the landing page's
# `service-doc` / `service-desc` links true on every node, and lets a client generate against
# the node it is actually talking to rather than a copy of the schema from somewhere else.
app = FastAPI(
    title="Agience API",
    version=BUILD_INFO.get("version") or "unknown",
    docs_url="/docs",
    openapi_url="/openapi.json",
    redoc_url=None,
    lifespan=lifespan,
    swagger_ui_init_oauth={
        "clientId": config.OIDC_CLIENT_ID or "agience-docs-client",
        "usePkceWithAuthorizationCodeGrant": True,
    },
)

try:
    app.router.redirect_slashes = False
except Exception:
    pass

# ----------------------------
# Error logging handlers
# ----------------------------

# Fields whose values must never appear in logs.
# `key` covers the once-only raw token in the POST /grants/keys response body.
# `api_key`/`apikey` stay on the redaction list so a stale client that still sends
# one never gets it written to the log.
_REDACT_KEYS = frozenset({"password", "secret", "token", "api_key", "apikey",
                          "access_token", "refresh_token", "credential",
                          "passkey_credential", "passkey_challenge",
                          "key", "grant_key"})

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


_PROTECTED_RESOURCE_PATH = "/.well-known/oauth-protected-resource"


def _resource_identifier(request: Request) -> str:
    """This node's canonical public URL — what it calls itself to a client.

    `MANTLE_URI` is the declared answer and is used when set. The request URL is the fallback for
    any node where it is unset, rather than an error: a resource that could not describe itself
    because one variable was unset would be strictly less useful than one that describes itself
    from the request it just answered.

    "Declared" is `config.declared_public_uri()` — the same definition `ui/browse_page._public_base`
    reads, and deliberately not a second one. Two readings of what counts as a declared
    `MANTLE_URI` is how the `resource` a client is handed and the `redirect_uri` a human is sent
    through come to disagree about the node's own name, and only the human's failure is noticed.
    """
    declared = config.declared_public_uri()
    if declared:
        return declared
    return str(request.base_url).rstrip("/")


def _bearer_challenge(request: Request) -> str:
    """The `WWW-Authenticate` value: Bearer, pointing at this resource's metadata."""
    return f'Bearer resource_metadata="{_resource_identifier(request)}{_PROTECTED_RESOURCE_PATH}"'


@app.get(_PROTECTED_RESOURCE_PATH, include_in_schema=False)
def oauth_protected_resource(request: Request):
    """RFC 9728 protected-resource metadata — how a client learns where to authenticate.

    This is the missing half of delegation, and is why it lives here rather than in a caller. A
    client acts for a user by exchanging that user's token for an RFC 8693 delegation
    (`prism.trust.server_auth` → Origin `/internal/delegation-token`), and a caller acting on a
    caller-supplied resource id has no authority to act without one. The user's token has to come
    from somewhere, so a client needs a way to discover the authorization server; publishing this
    document makes the whole chain reachable by any standards-compliant MCP client, not only by a
    browser that was told the answer in advance.

    Public and unauthenticated by design: it is discovery metadata, and a document a client must
    already be authenticated to read cannot tell it how to authenticate. It carries no secret — the
    issuer URL, resource URL, and scope names are all things a client is about to be told by the
    redirect anyway.

    It describes, it does not decide. Nothing here grants anything, and nothing reads it back:
    verification stays with `services/oidc.py` against the issuer's own JWKS. A node that published
    a friendly issuer here would not thereby trust it.

    ⚠ ON A STANDALONE NODE THIS DOCUMENT IS THE WHOLE OAUTH SURFACE, and it names no authorization
    server. There is no `/.well-known/oauth-authorization-server`, no `/authorize`, no `/token`, and
    no dynamic client registration anywhere in this app — the endpoints named in
    `services/dependencies.oauth2_scheme` are Origin's, on another host. A standards-compliant MCP
    OAuth flow therefore CANNOT complete against a standalone Mantle, whatever this document says,
    and the supported credential is a static `Authorization: Bearer` header holding a token this
    node's manifest anchor already trusts (`mantle-token`, `scripts/dev_mint_token.py`). Publishing
    an authorization server such a node cannot serve was the defect; `config.authorization_servers`
    now requires the authority to have been DECLARED, so an unconfigured node omits the key and
    says nothing rather than pointing a client at `http://localhost:8080`.
    """
    servers = config.authorization_servers()
    doc = {
        "resource": _resource_identifier(request),
        "authorization_servers": servers,
        "bearer_methods_supported": ["header"],
    }
    scope = (getattr(config, "OIDC_SCOPE", "") or "").split()
    if scope:
        doc["scopes_supported"] = scope
    # Omitted, not empty, when this node has no issuer configured. `"authorization_servers": []`
    # is a positive claim that there is nowhere to authenticate; the absence of the key says the
    # node has not been configured to say. A client can act on the second and only despair at the
    # first.
    if not servers:
        doc.pop("authorization_servers")
    return JSONResponse(content=doc, headers={"Cache-Control": "public, max-age=300"})


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
    # `exc.headers` is carried through: `HTTPException(headers=…)` is part of the API, and rebuilding
    # the response without them would silently drop any header a raise site set — a `Retry-After` or
    # a `WWW-Authenticate` would simply not arrive, with the status code alone looking correct.
    headers = dict(getattr(exc, "headers", None) or {})

    # One challenge for every 401 site. RFC 9728 §5.1 / the MCP authorization spec: a protected
    # resource answers 401 with `WWW-Authenticate: Bearer resource_metadata="<url>"`, and that URL
    # is how a client discovers which authorization server to use. Attached here rather than at a
    # raise site, deliberately: `services/dependencies.py` alone raises 401 from fourteen places,
    # and adding the header at only some of them would make discoverability depend on which path
    # failed — worse than uniformly absent, because the first client to hit the wrong path
    # concludes the server is broken.
    #
    # The test for "already set" is `resource_metadata`, not the header's presence, because
    # FastAPI's own `HTTPBearer` raises 401 with a bare `WWW-Authenticate: Bearer`. Keying on the
    # header's mere presence would suppress the informative challenge on exactly the routes that
    # use the security scheme.
    if exc.status_code == 401:
        key = next((k for k in headers if k.lower() == "www-authenticate"), None)
        existing = (headers.pop(key) if key else "").strip()
        if "resource_metadata=" not in existing:
            pointer = f'resource_metadata="{_resource_identifier(request)}{_PROTECTED_RESOURCE_PATH}"'
            # Preserve any parameters already present (`error="invalid_token"` and friends) and add
            # the pointer beside them, rather than discarding what a raise site deliberately said.
            params = existing[len("Bearer"):].strip().lstrip(",").strip() if existing[:6].lower() == "bearer" else ""
            headers["WWW-Authenticate"] = f"Bearer {params}, {pointer}" if params else f"Bearer {pointer}"
        else:
            headers["WWW-Authenticate"] = existing

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=headers)

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
# AGIENCE_RATE_LIMIT_PER_MIN=0 to disable.
#
# ⚠ THE CLIENT IS THE SOCKET PEER UNLESS A TRUSTED PROXY SAYS OTHERWISE. `X-Forwarded-For`
# is a request header, so the caller writes it. Reading it from any peer gives every caller
# a private bucket for the asking — a different value per request is a different window, and
# a limiter that can be opted out of by naming yourself differently is not a limiter. The
# same value was also a permanent key in the store below, so the bypass and the memory-growth
# path were one bug: an unauthenticated caller chose both how much it was allowed to send and
# how much the process remembered about it.
#
# AGIENCE_TRUSTED_PROXIES names the peers whose word about the client is worth taking, as
# addresses or CIDR blocks. UNSET MEANS TAKE NOBODY'S — a node with no edge in front of it
# must not be talked out of the address it can actually see. The failure direction of getting
# this wrong is deliberate: an edge that is not listed makes every client share the edge's one
# address, so the limiter over-counts and refuses traffic, rather than under-counting and
# admitting a flood. A limiter that fails open is the thing this section exists to prevent.
# ----------------------------
_RL_WINDOW_S = 60.0
_RL_MAX = int(os.getenv("AGIENCE_RATE_LIMIT_PER_MIN", "600"))
_RL_EXEMPT = frozenset(("/status", "/version", "/openapi.json", "/docs"))


def _parse_trusted_proxies(raw: str) -> tuple:
    """Parse AGIENCE_TRUSTED_PROXIES into networks. A malformed entry is dropped, loudly.

    Dropped rather than fatal, and dropped rather than widened: a typo in one entry must not
    stop the node booting, and it must not be read as "trust everything" either. What is left
    is the entries that parsed, which is the conservative reading of a half-understood list.
    """
    nets = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning(
                "AGIENCE_TRUSTED_PROXIES: ignoring %r — not an IP address or CIDR block. "
                "X-Forwarded-For from that peer is not honoured.", entry,
            )
    return tuple(nets)


_RL_TRUSTED_PROXIES = _parse_trusted_proxies(os.getenv("AGIENCE_TRUSTED_PROXIES", ""))

#: One sliding window per client, ORDERED BY LAST TOUCH — an `OrderedDict`, not a `dict`, and
#: that is what bounds it. See `_rl_evict_expired`.
_RL_HITS: "OrderedDict[str, deque]" = OrderedDict()


def _is_trusted_proxy(host) -> bool:
    """True when `host` is one of the configured proxies. Anything unparseable is not."""
    if not host or not _RL_TRUSTED_PROXIES:
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(addr in net for net in _RL_TRUSTED_PROXIES)


def _rate_limit_client(request: Request) -> str:
    """The identity the sliding window is kept against.

    The socket peer, unless the socket peer is a configured trusted proxy — in which case the
    RIGHT-MOST address in `X-Forwarded-For` that is not itself a trusted proxy, which is the
    last hop the trusted chain actually observed. Reading the LEFT-most entry instead stays
    spoofable even behind a real edge: a proxy APPENDS, so whatever the client sent arrives
    ahead of what the proxy added, and the left-most entry is the client's own claim about
    itself. Walking from the right and stopping at the first hop no trusted proxy vouched for
    is the only part of the header a trusted peer's presence actually attests to.

    An entry that is not an IP address is skipped rather than used: a proxy emitting one is a
    misconfiguration, and a string that names no host must not become a bucket of its own.
    """
    peer = request.client.host if request.client else None
    if not _is_trusted_proxy(peer):
        return peer or "unknown"
    for candidate in reversed((request.headers.get("x-forwarded-for") or "").split(",")):
        candidate = candidate.strip()
        if not candidate or _is_trusted_proxy(candidate):
            continue
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            continue
        return candidate
    # A trusted proxy that forwarded nothing usable is still a real peer to limit.
    return peer or "unknown"


def _rl_evict_expired(cutoff: float) -> None:
    """Drop every window whose newest hit has aged out. This is what bounds the store.

    `_RL_HITS` is ordered by last touch, so the dead entries are a PREFIX of it: an entry
    further along was touched more recently by definition, so the first live entry ends the
    sweep. Popping from the front is therefore exact and O(1) amortized — each key is created
    once and dropped once — with no periodic scan and no cap to tune.

    THE BOUND IS DERIVED, NOT CHOSEN. What survives is one entry per client seen within the
    last `_RL_WINDOW_S`, so the store holds at most as many entries as this process accepted
    requests in a window, and it drains to nothing on its own when traffic stops. The number
    of timestamps inside them is bounded the same way and by `_RL_MAX` per client. Before
    this, a key was created and never removed: the store's size was the number of distinct
    client identities the process had EVER seen, which — with the header trusted from any peer
    — was the number of requests an anonymous caller cared to send.
    """
    while _RL_HITS:
        dq = next(iter(_RL_HITS.values()))
        if dq and dq[-1] >= cutoff:
            return
        _RL_HITS.popitem(last=False)


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    if _RL_MAX <= 0 or request.url.path in _RL_EXEMPT:
        return await call_next(request)
    ip = _rate_limit_client(request)
    now = time.monotonic()
    cutoff = now - _RL_WINDOW_S
    _rl_evict_expired(cutoff)
    dq = _RL_HITS.get(ip)
    if dq is None:
        dq = deque()
        _RL_HITS[ip] = dq
    else:
        # Re-seat at the back so the eviction sweep above keeps its prefix property.
        _RL_HITS.move_to_end(ip)
        while dq and dq[0] < cutoff:
            dq.popleft()
    if len(dq) >= _RL_MAX:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
            headers={"Retry-After": str(int(_RL_WINDOW_S))},
        )
    dq.append(now)
    return await call_next(request)

# SessionMiddleware lives in Origin alongside the OAuth flows. Mantle carries no OAuth
# PKCE state cookie: it does not serve /auth/authorize.

# ----------------------------
# Routers
# ----------------------------
# Database-layer surface. App/platform routers (beacon, internal_personas, stream) are
# not mounted here. events_router is the outbound data-change stream (WS /events).
# system_router is the whole admin namespace — issuers, users, seed, erasure — behind
# one predicate. A server is an ordinary artifact, so it needs no router of its own;
# so is a secret — its value is artifact content, encrypted at rest by the envelope and
# authorized by the light cone, so there is no `/secrets` surface either.
app.include_router(artifacts_router)
app.include_router(events_router)
app.include_router(grants_router)
app.include_router(system_router)
app.include_router(mcp_router)

# ----------------------------
# `/mcp` above is Mantle's own database-layer MCP surface: the whole of it, addressed
# directly. There is no second gateway in front of it.

# ----------------------------
# Basic routes
# ----------------------------
def utcnow_z() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@app.get("/auth/callback", include_in_schema=False)
def auth_callback():
    """The OAuth redirect target — serves the same document as `/`, deliberately.

    The page is the handler. This is a single-page browser: the authorization code arrives as
    `?code=…` in the query string, and the JavaScript that exchanges it lives in the page. The
    callback route returns the document, which reads the code, posts it to the token endpoint with
    the PKCE verifier, and `replaceState`s back to `/`.

    A dedicated path rather than a redirect to `/` keeps the auth handler off the busiest route,
    keeps a single-use code out of the address bar of the page people bookmark, carries its own
    cache policy, and lets the registered redirect URI grant one path instead of the whole origin
    root. `/auth/callback` is already this system's convention — Origin's own provider callbacks
    land there (`auth_router`, prefix `/auth`, route `/callback`).

    Never cached: this response is reached with a live authorization code in the URL.
    """
    return HTMLResponse(
        content=_browse_page.render(),
        status_code=200,
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )


@app.get("/", include_in_schema=False)
def read_root(request: Request):
    """One URL, two audiences — content-negotiated.

    A browser asks for `text/html` and gets the explorer; a client asks for JSON (or anything
    else) and gets the machine payload. Serving HTML unconditionally breaks every script that
    parses this endpoint; serving JSON unconditionally means a human who opens the hostname sees
    a wall of braces and no way in. The `Accept` header already carries the answer, so neither
    audience needs a different URL to remember.
    """
    payload = {
        # Origin owns the setup wizard, so this node has no setup state to report: it is either
        # serving or it is not up. `/status` is where store health lives.
        "status": "ok",
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
        # The browser is the human door, not a status page: sign in, land back here, search and
        # read what your grants reach.
        #
        # The status payload stays the default. Only a caller that explicitly asks for `text/html`
        # gets the page; every script parsing this endpoint sees the JSON payload below.
        return HTMLResponse(
            content=_browse_page.render(),
            status_code=200,
            headers={"Cache-Control": "no-store", "Vary": "Accept"},
        )
    return JSONResponse(
        content=payload,
        status_code=200,
        headers={"Cache-Control": "no-store", "Vary": "Accept"},
    )

def _anchorset_status() -> dict:
    """The node's coordinate system, as two hashes and a verdict.

    `/status` is the only surface on which two operators can establish that their nodes route
    into the same regions, and the fingerprint lets them do it without either node exporting an
    anchor, a label or a vector — it is a hash over ids that are themselves hashes of public
    geometry. `indexed` is the fingerprint the cells were written under: equal means the store
    and the live set agree, and unequal is the state in which recall answers 200 with nothing
    from the semantic arm.

    Never raises. A node with no AnchorSet reports `provisioned: false` — that is a real and
    common state (the semantic arm is simply off), not a health failure.
    """
    try:
        from mantle.search.anchors import get_live_anchorset, indexed_geometry, live_fingerprint
        aset = get_live_anchorset()
        if aset is None or len(aset) == 0:
            return {"anchorset": {"provisioned": False}}
        fp = live_fingerprint()
        rec = indexed_geometry() or {}
        return {"anchorset": {
            "provisioned": True,
            "anchors": len(aset),
            "model_id": aset.model_id,
            "dim": aset.dim,
            "fingerprint": fp,
            "indexed_fingerprint": rec.get("fingerprint"),
            "matches_cells": None if not rec else (rec.get("fingerprint") == fp),
        }}
    except Exception as e:
        # An AnchorSet that cannot be read is itself the answer, and `/status` is where an
        # operator looks for it — so it is reported rather than swallowed into absence.
        return {"anchorset": {"provisioned": None, "error": str(e)}}


def _work_pool_status() -> dict:
    """Queue depth on this node's work pool — five counter reads, no scan, no `count(*)`.

    **Reported now, and not before, because a count only becomes a measurement once something
    decrements it.** Until `services/mirror_drain.py` existed, `pending` for this content type was
    a number that could only ever go up; publishing it would have been publishing a leak dressed
    as a queue. With a drain, it is a backlog: it rises when the object store is unreachable and
    falls when it comes back, which is exactly the fact an operator needs from `/status`.

    **It IS the mirror backlog, exactly, and it is named that way.** The counters are keyed
    `(content_type, status)` — `db/schema.py::c_task_status` — and `mirror_drain.MIRROR_TASK_CT`
    is written by one route for one operator (`artifacts_router._record_mirror_pending`), so
    every row it counts is an `op.content.mirror` task and there is no second population folded
    into the number.

    That is a property of the NAME, not of a filter, which is what makes it free. An
    operator-exact count used to be declined here on cost: while these tasks shared
    `application/vnd.agience.task+json` with every other operator's, narrowing to one operator
    meant walking a status bucket on `operator`, a column and not a counter — and the bucket to
    walk was the pending one, unbounded precisely when the number is interesting. Naming the pool
    for the work dissolved that: the counter is already the answer, so `/status` still costs five
    O(1) reads and now means one operator's backlog rather than a pool's. `operator` is reported
    alongside the type so the reading is on the response and not only here.

    `dead` is in the tally and is the one to watch: a dead task is an obligation the node has
    stopped attempting, and each one is a piece of content that is readable here and nowhere else.
    A non-zero `dead` is where an operator starts; the row itself is at
    `task-op.content.mirror:<blake2b-64 of the content key>`, carrying `operator`, `last_error`,
    `dead_reason` and the `arguments` naming the stranded artifact, and is projected into the
    `task` sidecar (`operator`, `status`, `completed_at`). It is NOT reachable through
    `GET /artifacts/{id}` — `mesh/sync._put_op` writes operational rows without minting a grant,
    so `check_access` answers 404 for every principal, the same for this row as for every cursor
    and watermark written the same way.

    Never raises: a store that cannot answer reports that, and does not take `/status` down.
    """
    try:
        from mantle.db.backend import store_handle
        from mantle.services.mirror_drain import MIRROR_OPERATOR, MIRROR_TASK_CT
        counts = store_handle().artifacts.count_by_status(MIRROR_TASK_CT)
        return {"work_pool": {"content_type": MIRROR_TASK_CT,
                              "operator": MIRROR_OPERATOR, **counts}}
    except Exception as e:
        return {"work_pool": {"error": str(e)}}


@app.get("/status", include_in_schema=False)
def check_backend_status():
    status = {
        **check_store_health(),
        **_anchorset_status(),
        **_work_pool_status(),
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
