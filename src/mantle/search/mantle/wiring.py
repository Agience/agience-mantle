"""Production wiring for MANTLE encrypted search.

Centralizes construction of the MANTLE + SSE pipeline so the router and
the commit-hook indexer share one definition. Each builder pulls the
same Oracle + storage adapters; the choice of accessor vs indexer is
the only fork.

Wiring decisions:

- **Master key store**: :class:`~.oracle.LatticeMasterKeyStore` — per-principal
  DEKs wrapped by the platform KEK and persisted in the lattice so they survive
  restarts. KEK custody is pluggable via :mod:`.key_provider` (local file |
  cloud KMS | Vault); a future Shamir-threshold backend is just another provider.
- **Cell storage**: object storage under the ``mantle-cells/`` prefix when the operator
  configured edge object storage, and a local directory tree of the same shape when they
  did not — see :func:`_build_cell_store`. Cells live at
  ``mantle-cells/{owner}/{collection}/{cluster}.cell`` (cluster = routing anchor).
- **SSE posting store**: object storage under the ``mantle-sse/`` prefix when the
  operator configured edge object storage, and a local directory tree of the same shape
  when they did not — see :func:`_build_sse_store`. Both are the same Protocol;
  both see ciphertext only.

Each arm answers "where do the bytes go" through its own env var (``MANTLE_CELL_STORE`` /
``MANTLE_SSE_STORE``) with the same three answers, and both default to ``auto``: S3 where the
operator configured it, local disk where they did not. That is what makes an unconfigured S3 a
complete configuration rather than half of one — both arms have somewhere to write, or neither
does.

That is a claim about STORAGE, and it is the only claim this module makes. A cell store the
vector arm can write to is not the vector arm running: routing needs a provisioned AnchorSet,
which is not a store, not an env var and not something a builder here can construct. Every
builder below can succeed on a node whose semantic arm answers nothing — see
:mod:`mantle.search.anchors.store`.

Builders return ``None`` if production stores cannot be constructed
(missing S3 client, missing encryption key). The router treats ``None``
as 503 — there is no plaintext fallback.

The QUERY builders require all three of oracle, posting store and cell store, and none of
them is best-effort: recall narrows on the postings and ranks on the cells, so a missing
either is a node that cannot answer rather than one that answers less well. See
:func:`_build_query_stack`.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Any, NamedTuple, Optional

from .engine import MantleQueryEngine
from .indexer import MantleIndexer
from .lightcone import LightConeResolver
from .oracle import OracleService
from .s3_cell_store import S3CellStore
from .sse import (
    MantleSseSearchAccessor,
    PostingStore,
    S3PostingStore,
    SqlitePostingStore,
    SseIndexer,
    TokenNarrower,
)
from .stores import CellStore

logger = logging.getLogger(__name__)

# Process-level oracle singleton — caches unwrapped master keys for the process
# lifetime. The keys themselves are persisted durably by LatticeMasterKeyStore, so
# they survive restarts; the singleton only avoids re-unwrapping on every call
# within one process.
_oracle_singleton: Optional[OracleService] = None
_oracle_lock = threading.Lock()


def _build_oracle() -> Optional[OracleService]:
    """Resolve the platform encryption key and return the process-level Oracle.

    Returns None if the encryption key isn't available — happens during
    setup before key_manager has run.
    """
    global _oracle_singleton
    if _oracle_singleton is not None:
        return _oracle_singleton

    with _oracle_lock:
        if _oracle_singleton is not None:
            return _oracle_singleton

        try:
            from .key_provider import build_key_provider
        except Exception as exc:
            logger.warning("MANTLE oracle imports failed: %s", exc)
            return None

        try:
            # KEK custody is pluggable (local file | KMS | Vault) via
            # MANTLE_KEK_PROVIDER — default 'local' = the platform encryption.key.
            kek = build_key_provider()
        except Exception as exc:
            logger.warning("MANTLE oracle: KEK provider unavailable: %s", exc)
            return None

        # Durable master key store: per-principal DEKs are Fernet-wrapped by the
        # platform KEK and persisted in the lattice so they survive a mantle restart
        # (an in-process-only store would lose them on every restart, orphaning all
        # encrypted cells — search would go empty).
        from mantle.db import backend as _backend
        from .oracle import LatticeMasterKeyStore as _KeyStore

        def _master_key_db():
            return _backend.store_handle()

        # Key issuance is coupled to the grant check (canonical plan §5.3): the
        # oracle verifies every request against the same light cone the query path
        # uses to build its authorized context list. Wired here, at the single
        # place the process-wide oracle is constructed, so there is no oracle in
        # this process that can issue keys without a grant.
        #
        # The verifier gets its own store handle from the same factory as the key
        # store; if that handle is unavailable the verifier cannot answer, and
        # `_authorize` fails closed rather than assuming authorization.
        from .oracle import LightConeGrantVerifier

        _oracle_singleton = OracleService(
            _KeyStore(kek, _master_key_db),
            grant_verifier=LightConeGrantVerifier(
                _LazyDb(_master_key_db), ttl_s=_verifier_ttl_s(),
            ),
        )
        return _oracle_singleton


# ---------------------------------------------------------------------------
# The grant-decision memo is process-local, and so is its invalidation
# ---------------------------------------------------------------------------
#
# `LightConeGrantVerifier` memoizes an authorization decision, and every key derivation
# and every cell decryption reads that memo. `invalidate_grant_cache` reaches the memo in
# the process that calls it and no other. With N workers, `DELETE /grants/{id}` is served
# by one of them while the remaining N-1 keep issuing content keys against the revoked
# grant until their own entries lapse.
#
# This is the same hazard `event_backplane.require_backplane_for_workers` refuses for the
# event bus — an in-process fan-out cannot reach a sibling process — arriving on the
# authorization path, where the cost of the gap is a live key rather than a missed UI
# update. The signalling back-plane that would close it properly already exists, but
# carrying invalidations on it means a second delivery path through `event_bus`, and it is
# optional besides: `MANTLE_BACKPLANE_KIND` unset is a supported multi-worker
# configuration, so a back-plane ride could not be the whole answer even once written.
#
# DECISION, until invalidation can cross a process boundary: the memo is not held where it
# cannot be invalidated. More than one worker turns the TTL to 0 — no memo, every decision
# re-read from the ledger — and says so once, loudly, naming the setting. An operator who
# would rather pay a bounded staleness window than the re-walk states that by setting
# MANTLE_GRANT_CACHE_TTL explicitly, which is a decision someone made rather than one that
# happened to them.
#
# The cost is real and is the reason this is stated rather than assumed: under N>1 workers
# a search re-walks the light cone per cell instead of once per 30s. Correct and slow is
# recoverable; fast and issuing keys against revoked grants is not.

#: Seconds the verifier may memoize an authorization decision. Explicit operator override;
#: unset means the shape below decides.
GRANT_CACHE_TTL_SETTING = "MANTLE_GRANT_CACHE_TTL"

#: The TTL for the single-worker deployment, where invalidation reaches the only memo there
#: is. Also `LightConeGrantVerifier`'s own default — stated here because this is the module
#: that decides it in production.
DEFAULT_GRANT_CACHE_TTL_S = 30.0

_ttl_reported = False


def _verifier_ttl_s() -> float:
    """The memo TTL for THIS deployment. See the block above for why the shape decides it."""
    global _ttl_reported
    from mantle.events.event_backplane import WORKERS_SETTING

    raw = (os.getenv(GRANT_CACHE_TTL_SETTING) or "").strip()
    if raw:
        try:
            ttl = max(float(raw), 0.0)
        except ValueError:
            logger.error(
                "%s=%r is not a number of seconds — falling back to %s, which is the safe "
                "answer rather than the fast one.", GRANT_CACHE_TTL_SETTING, raw, 0.0)
            return 0.0
        if not _ttl_reported:
            _ttl_reported = True
            logger.info(
                "MANTLE grant-decision memo: TTL %.1fs (set by %s). Revocation takes effect "
                "immediately in the process that serves it and within %.1fs elsewhere.",
                ttl, GRANT_CACHE_TTL_SETTING, ttl)
        return ttl

    try:
        workers = int(os.getenv(WORKERS_SETTING, "1"))
    except (TypeError, ValueError):
        workers = 1

    if workers > 1:
        if not _ttl_reported:
            _ttl_reported = True
            logger.warning(
                "%s=%d: the grant-decision memo is OFF (TTL 0) because invalidating it does "
                "not cross a process boundary — a grant revoked on one worker would keep "
                "issuing content keys on the other %d until their entries lapsed. Every "
                "authorization decision is now re-read from the ledger, which costs a "
                "light-cone walk per encrypted cell. Run a single worker to get the memo back, "
                "or set %s=<seconds> to accept a bounded window of post-revocation access.",
                WORKERS_SETTING, workers, workers - 1, GRANT_CACHE_TTL_SETTING)
        return 0.0

    return DEFAULT_GRANT_CACHE_TTL_S


def invalidate_grant_cache(requester_id: Optional[str] = None) -> None:
    """Drop this process's memoized light-cone decisions. Call after every grant mutation.

    ``requester_id`` is the ACTING-PRINCIPAL id the memo is keyed on, which for a bearer key
    is the root grant's id and not its ``grantee_id``; `grant_key_service.principal_ids_for`
    is the one place that translation lives. ``None`` clears everything.

    Both directions need it. A freshly granted principal keeps being denied keys until the
    TTL lapses — on the fast path (create workspace → immediately write content) that is a
    hard failure, not a delay. A revoked one keeps being ISSUED them, which is the direction
    that matters.

    Process-local, by construction: it touches `_oracle_singleton`, and there is one of those
    per worker. See the block above for what that means and what the wiring does about it.
    Best-effort and layered: services call this; the db layer never imports search."""
    o = _oracle_singleton
    if o is not None:
        try:
            v = getattr(o, "_verifier", None)
            if v is not None and hasattr(v, "invalidate"):
                v.invalidate(requester_id)
        except Exception:
            logger.debug("grant-cache invalidation failed", exc_info=True)


class _LazyDb:
    """Defers the store handle until the verifier actually needs it.

    ``LightConeGrantVerifier`` is constructed while the oracle singleton is being
    built, which can happen before the database is reachable. Connecting eagerly
    there would make oracle construction fail (→ `None` → search silently
    unwired), so the handle is resolved on first attribute access instead.
    """

    __slots__ = ("_factory",)

    def __init__(self, factory) -> None:
        self._factory = factory

    def __getattr__(self, name):
        return getattr(self._factory(), name)


# ---------------------------------------------------------------------------
# Per-state index segments
#
# Each artifact state indexes into its own physically separate index tree, keyed
# by the storage prefix. `committed` keeps the original prefixes so the existing
# committed index needs NO migration; `draft` and `archived` are sibling trees
# alongside it. A query opts into a non-committed segment explicitly (search is
# committed-only by default). The per-principal master keys + cell AAD are shared
# across segments — separation is purely at the object-storage namespace.
# ---------------------------------------------------------------------------
VALID_SEGMENTS = ("committed", "draft", "archived")


def _segment_prefixes(segment: str) -> tuple[str, str]:
    """Return ``(cell_prefix, sse_prefix)`` for an artifact-state index segment."""
    if segment not in VALID_SEGMENTS:
        raise ValueError(
            f"invalid index segment {segment!r} (expected one of {VALID_SEGMENTS})"
        )
    if segment == "committed":
        return "mantle-cells", "mantle-sse"
    return f"mantle-cells-{segment}", f"mantle-sse-{segment}"


_EDGE_S3_REACHABLE: dict = {}          # (id(client), bucket) -> bool, probed once per process


def edge_s3_if_reachable(what: str):
    """The edge S3 `(client, bucket)` if it can actually be reached, else `(None, None)`.

    Reachability is measured, not inferred from configuration: one `head_bucket` call
    answers "can this client reach this bucket", and the result is memoised per
    (client, bucket) so the cost is one round trip per process, not one per call.
    A False answer is an honest state — the arm is off, the caller skips it, and the
    log says so once rather than once per artifact."""
    try:
        from mantle.services import content_service
    except Exception as exc:
        logger.warning("%s: content_service import failed: %s", what, exc)
        return None, None

    s3_client = getattr(content_service, "_s3_edge_internal", None)
    bucket = getattr(content_service, "_EDGE_BUCKET", None)
    if s3_client is None or not bucket:
        logger.warning("%s: edge S3 client or bucket not initialized", what)
        return None, None

    key = (id(s3_client), str(bucket))
    ok = _EDGE_S3_REACHABLE.get(key)
    if ok is None:
        try:
            s3_client.head_bucket(Bucket=bucket)
            ok = True
        except Exception as exc:
            ok = False
            logger.warning(
                "%s: edge S3 bucket %r is NOT reachable (%s: %s) — this index arm is OFF. "
                "Search stays lexical-only until the bucket is reachable; this is reported once, "
                "not once per artifact.", what, bucket, type(exc).__name__, exc)
        _EDGE_S3_REACHABLE[key] = ok
    return (s3_client, bucket) if ok else (None, None)


#: How the cell index is stored, from ``MANTLE_CELL_STORE``. Same three answers and the same
#: factory shape as :data:`VALID_SSE_STORES` next door — one env var naming a backend — because
#: the two arms answer the same question about the same deployment and must not answer it
#: differently.
VALID_CELL_STORES = ("auto", "s3", "file")


def _cell_store_kind() -> str:
    return (os.getenv("MANTLE_CELL_STORE", "auto") or "auto").strip().lower()


def local_cell_root() -> str:
    """The local cell root — ``MANTLE_CELL_DIR``, else ``<BASE_DIR>/.data/mantle-cells``.

    Derived from ``config.BASE_DIR`` (absolute) rather than the process's working directory, for
    the reason :func:`local_sse_root` states: a store that moves when mantle is started from a
    different place is a store that silently loses its index.
    """
    from mantle import config

    raw = (os.getenv("MANTLE_CELL_DIR") or "").strip()
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    return str(config.BASE_DIR / ".data" / "mantle-cells")


def _build_file_cell_store(cell_prefix: str) -> Optional[CellStore]:
    """The local cell store, or ``None`` if the root cannot be created.

    ``None`` is the same honest refusal the S3 path makes: there is nowhere to put the cells, so
    the caller runs without the vector arm rather than pretending to have one.
    """
    from .file_cell_store import FileCellStore

    root = local_cell_root()
    try:
        store = FileCellStore(root, prefix=cell_prefix)
    except OSError as exc:
        logger.warning(
            "MANTLE cell store: local root %r is not usable (%s: %s) — the vector arm is OFF "
            "until it is. Set MANTLE_CELL_DIR to a writable directory.",
            root, type(exc).__name__, exc,
        )
        return None
    logger.info("MANTLE cell store: local cells at %s (prefix %s)", root, cell_prefix)
    return store


def _build_cell_store(segment: str = "committed") -> Optional[CellStore]:
    """Construct the cell store for one index segment.

    Both backends implement the same :class:`CellStore` Protocol and both receive the same
    ciphertext — `nonce ‖ ciphertext ‖ tag` from :func:`cell.pack_cell`, bound by
    :func:`cell.cell_aad` to its ``collection:cluster`` slot. The choice is where the bytes land,
    never what they are.

    ``MANTLE_CELL_STORE`` selects, with the same vocabulary as ``MANTLE_SSE_STORE``:

    - ``auto`` (default) — S3 if the operator configured edge object storage, else the local
      cell tree. This is what leaves the SEMANTIC arm with somewhere to put its cells when S3
      is ``None``, and not only the lexical one, and what leaves an S3 install byte-identical.
      It does not make the arm answer: that also needs a provisioned AnchorSet, which no value
      of this variable supplies.
    - ``s3`` — object storage only. Unreachable → ``None`` → the vector arm is off.
    - ``file`` — the local tree only, even where S3 is configured.

    The S3 path reuses Mantle's existing edge client + bucket so cells share the same MinIO/S3
    endpoint as content blobs, under a distinct prefix so listing the content bucket doesn't
    tangle with artifact uploads.
    """
    cell_prefix, _ = _segment_prefixes(segment)
    kind = _cell_store_kind()

    if kind not in VALID_CELL_STORES:
        logger.error(
            "MANTLE_CELL_STORE=%r is not one of %s — the vector arm is OFF rather than guessing "
            "which backend was meant.", kind, ", ".join(VALID_CELL_STORES),
        )
        return None

    if kind == "file" or (kind == "auto" and not edge_object_storage_is_configured()):
        return _build_file_cell_store(cell_prefix)

    s3_client, bucket = edge_s3_if_reachable("MANTLE cell store")
    if s3_client is None:
        return None

    return S3CellStore(s3_client, bucket=bucket, prefix=cell_prefix)


def cell_storage_available(segment: str = "committed") -> bool:
    """Is there anywhere to write the cell index right now?

    The vector-arm counterpart of :func:`sse_index_storage_available`, and measured the same
    way — a ``head_bucket`` for S3, a created root directory for the local store.
    """
    try:
        return _build_cell_store(segment) is not None
    except Exception:  # noqa: BLE001 — a builder that raises is also "not available"
        logger.warning("MANTLE cell store: availability probe failed", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_indexer(
    store_db: object, *, segment: str = "committed"
) -> Optional[MantleIndexer]:
    """Construct a production-wired :class:`MantleIndexer` for one index segment."""
    oracle = _build_oracle()
    cells = _build_cell_store(segment)
    if oracle is None or cells is None:
        return None
    return MantleIndexer(oracle, cells)


# ---------------------------------------------------------------------------
# MANTLE-SSE wiring
# ---------------------------------------------------------------------------


#: How the SSE index is stored, from ``MANTLE_SSE_STORE``. Same shape as
#: ``MANTLE_KEK_PROVIDER`` in :mod:`.key_provider` — one env var naming a backend, resolved by
#: one factory.
VALID_SSE_STORES = ("auto", "s3", "file")


def _sse_store_kind() -> str:
    return (os.getenv("MANTLE_SSE_STORE", "auto") or "auto").strip().lower()


def edge_object_storage_is_configured() -> bool:
    """Did the operator configure edge object storage — as distinct from it being reachable?

    The two are different questions. ``content_service._s3_edge_internal`` is always a boto3
    client object, built unconditionally at import regardless of whether credentials are
    present, and ``_EDGE_BUCKET`` always falls back to ``config.CONTENT_BUCKET``. Neither
    object's existence answers "did anyone ask for S3".

    What answers it is the credential/endpoint material the operator supplies. This reads it
    from ``content_service`` rather than re-reading the environment here, since that module
    already owns the env vocabulary (``CONTENT_EDGE_*``/``CONTENT_ROOT_*``/``AWS_ENDPOINT_URL*``)
    and a second copy of it would be a seam that drifts.

    This is why an S3-configured install is untouched by the local backend: configured-and-
    unreachable still resolves to S3, still fails the reachability measurement, and still
    returns ``None`` → 503. It never silently starts a second, local, divergent index.
    """
    try:
        from mantle.services import content_service
    except Exception as exc:
        logger.warning("MANTLE-SSE stores: content_service import failed: %s", exc)
        return False
    return any(
        bool(getattr(content_service, name, None))
        for name in (
            "_EDGE_ACCESS_KEY_ID",
            "_EDGE_SECRET_ACCESS_KEY",
            "_EDGE_ENDPOINT_URL_INTERNAL",
        )
    )


def local_sse_root() -> str:
    """The local index root — ``MANTLE_SSE_DIR``, else ``<BASE_DIR>/.data/mantle-sse``.

    The default is derived from ``config.BASE_DIR`` (absolute) rather than from the process's
    working directory, so it cannot quietly become a different store when mantle is started from
    a different place — the failure ``MANTLE_LATTICE_PATH`` documents next door.
    """
    from mantle import config

    raw = (os.getenv("MANTLE_SSE_DIR") or "").strip()
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    return str(config.BASE_DIR / ".data" / "mantle-sse")


def local_sse_path(sse_prefix: str) -> str:
    """The local index database file for one segment: ``<root>/<prefix>.db``.

    One file per segment, which is what the file store's `prefix` did with one directory tree per
    segment. The root stays :func:`local_sse_root` so ``MANTLE_SSE_DIR`` keeps meaning the same
    thing and an operator who pointed it somewhere writable does not have to move it.
    """
    return os.path.join(local_sse_root(), "%s.db" % sse_prefix)


def _build_file_sse_store(sse_prefix: str) -> Optional[SqlitePostingStore]:
    """The local posting store, or ``None`` if it cannot be opened.

    SQLite, not a directory tree. `SqlitePostingStore` is the only posting store; see that module
    for the four measured problems a one-file-per-slot layout produces (write cost linear in corpus
    size, the object explosion, an accelerator blob that has to exist and brings its own
    corpus-losing failure mode, and no atomicity across a posting's read-modify-write).

    ``None`` here is the same honest refusal the S3 path makes: there is nowhere to put an index,
    so the caller answers 503 rather than pretending. `sqlite3.Error` joins `OSError` because the
    ways a database file can be unusable are a superset of the ways a directory can — a corrupt
    header, a read-only WAL, an unsupported filesystem lock — and every one of them means the same
    thing to this caller.
    """
    path = local_sse_path(sse_prefix)
    try:
        store = SqlitePostingStore(path)
    except (OSError, sqlite3.Error) as exc:
        logger.warning(
            "MANTLE-SSE stores: local index %r is not usable (%s: %s) — encrypted search is "
            "OFF until it is. Set MANTLE_SSE_DIR to a writable directory.",
            path, type(exc).__name__, exc,
        )
        return None
    logger.info("MANTLE-SSE stores: local index at %s", path)
    _report_analyzer_generation(store, path)
    return store


def _report_analyzer_generation(store: Any, where: str) -> None:
    """Say, once at open, whether this index was written by the analysis this build runs.

    A blind token is an HMAC of an ANALYSED term, so the pipeline is part of the index format and
    no in-place migration exists: the store holds hashes and cannot re-derive a term it never saw.
    A client whose analysis has moved queries terms the store was never filed under and gets an
    empty answer that is indistinguishable from a correct one — well-formed query, healthy store,
    nothing found.

    Reported rather than refused, and that is a judgement about blast radius rather than
    squeamishness. Generation 2 added ASCII folding, which is the identity on ASCII, so the terms
    that move are exactly those containing combining marks and everything else still matches.
    Refusing here would take a working index offline over a subset of its content; staying silent
    would leave the subset unreachable with nothing to read. Naming it does neither.

    An unstamped store reads as generation 1 only when it holds something. An EMPTY unstamped store
    is not stale — nothing has been written under any analysis yet — and the first index write
    stamps it. Telling those apart is why this asks the store for its owners.
    """
    from mantle.search.mantle.sse.posting import analyzer_generation_of
    from mantle.search.mantle.sse.tokenizer import ANALYZER

    found = analyzer_generation_of(store)
    if found == ANALYZER:
        return
    if found is None:
        try:
            populated = bool(store.list_owners())
        except Exception:
            populated = False
        if not populated:
            return                      # empty and unstamped: the first write claims it
        found = 1                       # written before stamping existed, which is generation 1

    logger.warning(
        "MANTLE-SSE stores: %s was written by analyzer generation %s and this build is generation "
        "%s. Terms are HMAC'd after analysis, so there is no in-place migration and content "
        "indexed under the old analysis is unreachable — silently, as an empty result. Rebuild "
        "with `mantle.search.init_search.reindex_all_artifacts`.",
        where, found, ANALYZER,
    )


def _build_sse_store(segment: str = "committed") -> Optional[PostingStore]:
    """Construct the posting store for one index segment.

    One store, where there were two. The second was the per-owner BM25 corpus-statistics blob;
    nothing computes a corpus statistic, so nothing writes or reads it and no backend for it
    is built. Existing `stats.enc` objects in a bucket are inert and are left alone.

    Both backends implement the same Protocol and both receive ciphertext only; the choice
    is where the bytes land, never what they are.

    ``MANTLE_SSE_STORE`` selects:

    - ``auto`` (default) — S3 if the operator configured edge object storage, else the local
      file-backed index. This is what makes a standalone install searchable with no configuration
      at all, and what leaves an S3 install byte-identical to before.
    - ``s3`` — object storage only. Unreachable → ``None`` → 503.
    - ``file`` — the local index only, even where S3 is configured.

    Returns ``None`` when the selected backend has nowhere to write — same refusal policy as
    :func:`_build_cell_store`.
    """
    _, sse_prefix = _segment_prefixes(segment)
    kind = _sse_store_kind()

    if kind not in VALID_SSE_STORES:
        logger.error(
            "MANTLE_SSE_STORE=%r is not one of %s — encrypted search is OFF rather than guessing "
            "which backend was meant.", kind, ", ".join(VALID_SSE_STORES),
        )
        return None

    if kind == "file" or (kind == "auto" and not edge_object_storage_is_configured()):
        return _build_file_sse_store(sse_prefix)

    s3_client, bucket = edge_s3_if_reachable("MANTLE-SSE stores")
    if s3_client is None:
        return None

    store = S3PostingStore(s3_client, bucket=bucket, prefix=sse_prefix)
    _report_analyzer_generation(store, "s3://%s/%s" % (bucket, sse_prefix))
    return store


def sse_index_storage_available(segment: str = "committed") -> bool:
    """Is there anywhere to write the SSE index right now?

    The question a full reindex has to ask before it scans anything. It measures the
    selected backend — a ``head_bucket`` for S3, a created root directory for the local store —
    rather than inferring the answer from an object existing.
    """
    try:
        return _build_sse_store(segment) is not None
    except Exception:  # noqa: BLE001 — a builder that raises is also "not available"
        logger.warning("MANTLE-SSE stores: availability probe failed", exc_info=True)
        return False


def build_sse_indexer(
    store_db: object, *, segment: str = "committed"
) -> Optional[SseIndexer]:
    """Construct a production-wired :class:`SseIndexer` for one index segment.

    Returns ``None`` if any prerequisite is missing — Oracle, S3, or
    bucket. The caller (commit-path hook) skips SSE indexing on ``None``
    rather than silently using in-memory stores.
    """
    oracle = _build_oracle()
    posting_store = _build_sse_store(segment)
    if oracle is None or posting_store is None:
        return None
    return SseIndexer(oracle, posting_store)


def build_digest_refresher(
    members_provider, *, read, engine_id: str, segment: str = "committed"
):
    """Construct the collection-digest refresher, or ``None`` if a prerequisite is missing.

    `read` and `engine_id` are the caller's, and neither has a default: mantle does not import
    any spectral library, so the instrument that takes the read cannot be resolved here. A host
    passes `read=<probe>.mp_deviation,
    engine_id=<probe>.ENGINE_ID_PROXIMITY`. This builder supplies the two
    collaborators mantle owns, the key provider and the posting store.

    `engine_id` travels with every digest because a digest taken against one instrument is not
    comparable with one taken against another — `collection_proximity` refuses a cross-engine
    comparison rather than silently mixing them.

    Returns ``None`` on a missing oracle or posting store, matching every other builder here: a
    node that cannot hold digests declines to keep them rather than keeping them somewhere else.
    """
    oracle = _build_oracle()
    posting_store = _build_sse_store(segment)
    if oracle is None or posting_store is None:
        return None
    from mantle.search.ingest.digest_refresh import CollectionDigestRefresher
    from mantle.search.mantle.collection_proximity import DigestSlot

    return CollectionDigestRefresher(
        DigestSlot(oracle, posting_store), members_provider, read=read, engine_id=engine_id,
    )


def build_collection_proximity_narrower(
    members_of, *, probe_factory, segment: str = "committed"
):
    """Construct the query-side proximity narrower, or ``None`` if a prerequisite is missing.

    `probe_factory` is injected for the same reason `read` is above: the probe's contract is
    `<probe>.SpectrumProbe`'s — exact against a full scan, not an approximation —
    and mantle cannot name it.

    The narrower compiles to the same `lookup(pairs) -> set[artifact_id]` shape the blind-token
    narrowing produces, so it meets the light cone through the single `ids &= …` line rather than
    as a second authorization path. That is what makes it safe to add to a recall: it can only
    ever narrow, and it narrows a set the light cone has already decided.
    """
    oracle = _build_oracle()
    posting_store = _build_sse_store(segment)
    if oracle is None or posting_store is None:
        return None
    from mantle.search.mantle.collection_proximity import CollectionProximityNarrower

    return CollectionProximityNarrower(
        oracle, posting_store, members_of, probe_factory=probe_factory,
    )


class _QueryStack(NamedTuple):
    """The two collaborators a recall runs on, built once from one prerequisite check."""

    narrower: TokenNarrower
    ranker: MantleQueryEngine


def _build_query_stack(segment: str) -> Optional[_QueryStack]:
    """Everything a recall needs, or ``None`` if ANY of it is missing.

    Three prerequisites, all hard: the oracle, the posting store and the cell store. Recall
    narrows on the blind-token postings and ranks what survives on the cells, so a node
    missing either store cannot answer — and the two ways it could pretend to are both worse
    than a 503.

    Without a POSTING STORE there is no narrowing, and a narrowing has only two possible
    stand-ins. "Everything authorized" widens every query into a dump of the caller's whole
    light cone, which breaks the property the narrowing exists to hold. "Nothing matched" is a
    200 with an empty list for a query that matched — silent, and indistinguishable from a
    correct empty answer. Neither is visible to the caller, so neither is allowed to happen.

    Without a cell store nothing can rank by cosine, so that store is required on the same terms.
    Treating it as best-effort reads backwards: a node with no SSE store answers 503 while a node
    with no cell store answers 200 with an empty list.

    A cell store is not a working semantic arm — routing also needs a provisioned AnchorSet,
    which is not a store and which nothing here can build. That node returns an accessor, and
    every recall on it narrows correctly and comes back ordered by how much of the query each
    hit matched. See :mod:`mantle.search.anchors.store`.
    """
    oracle = _build_oracle()
    posting_store = _build_sse_store(segment)
    cells = _build_cell_store(segment)
    if oracle is None or posting_store is None or cells is None:
        return None
    return _QueryStack(TokenNarrower(oracle, posting_store), MantleQueryEngine(oracle, cells))


def build_sse_search_accessor(
    store_db: object,
    *,
    segment: str = "committed",
) -> Optional[MantleSseSearchAccessor]:
    """Construct a router-shape (``SearchQuery → SearchResult``) SSE accessor.

    ``segment`` selects which per-state index to query — ``committed`` (default),
    ``draft``, or ``archived``. Composes the narrower, the ranker and the light-cone resolver
    with a ``SearchHit`` hydrator over the lattice. This is the canonical search backend the
    artifacts router uses, on both of its entry points.

    ``field_boosts`` and ``rrf_k`` USED TO BE PARAMETERS HERE. A field boost was a BM25 weight
    and a rank-fusion constant was RRF's; there is no scorer to weight and nothing to fuse, so
    neither names anything a caller could set. A field is now a place the narrowing looks, and
    it looks in all four.

    Returns ``None`` when any of the three prerequisites is missing — the router converts that
    to 503 (no plaintext fallback by design). An accessor that comes back from here always has
    both a narrower and a ranker, which is what lets
    :meth:`~.sse.router_accessor.MantleSseSearchAccessor.search` treat a missing one as a
    programming error rather than a state to degrade through.
    """
    stack = _build_query_stack(segment)
    if stack is None:
        return None
    return MantleSseSearchAccessor(
        LightConeResolver(store_db),
        store_db=store_db,
        narrower=stack.narrower,
        ranker=stack.ranker,
        # The accessor hydrates by lineage, so it needs the same segment the stack queries:
        # a hit names a `root_id`, and this is what says which of that root's versions was
        # the one indexed here.
        segment=segment,
    )


__all__ = [
    "VALID_CELL_STORES",
    "VALID_SEGMENTS",
    "VALID_SSE_STORES",
    "build_indexer",
    "cell_storage_available",
    "edge_object_storage_is_configured",
    "local_cell_root",
    "local_sse_root",
    "sse_index_storage_available",
    "build_sse_indexer",
    "build_sse_search_accessor",
]
