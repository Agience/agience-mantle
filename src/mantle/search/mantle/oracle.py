"""OracleService — in-process key custodian for MANTLE encrypted search.

Step 2.2a implementation. Holds per-principal 256-bit master keys in memory,
loaded lazily from Fernet-wrapped storage on first access. Derives per-cell
AES-256-GCM keys via HKDF on demand — cell keys are never persisted.

The **principal** is the collection's immutable origin root (see
``search.mantle.principal``), NOT an "owner" / ``created_by``. Agience has no
owners — access is by grant — so the master-key root is the stable creation-lineage
root, which the index and query paths resolve identically (same key both ends).

Key derivation hierarchy:

    Principal master key (256 bits, Fernet-wrapped at rest)
      └─ HKDF-Extract+Expand(IKM=master, salt=fixed,
                             info=collection_id ‖ 0x00 ‖ cluster_id, len=32)
      → cell key (256-bit AES-GCM)

One cell per ``(principal_id, collection_id, cluster_id)`` where ``cluster_id`` is
the routing anchor (canonical plan §5.1: the AnchorSet IS the partition). There
is one path — every cell is anchor-routed; there is no flat / unpartitioned key.

Determinism: re-derivation always produces the same cell key for the same
(master_key, collection_id, cluster_id) tuple, which is essential for query-path
decryption.

See `.dev/features/mantle-mvp.md` § Layer 2a.
"""

from __future__ import annotations

import enum
import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Protocol

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# The authenticated caller, and the base class for refusals. `services` has an empty
# __init__ and this module imports nothing back, so there is no cycle.
from mantle.services.acting_principal import KeyCustodyDenied, require_acting_principal

# ⛔ RE-EXPORT, NOT REDEFINITION. `MasterKeyUnavailable`/`MasterKeyMissing` moved to `sse/keys.py` so
# the encrypted lexical arm can ship without this 703-line custody module (EREA §5). They are
# re-exported here because `pipeline_unified.py` and `test_key_custody_bypasses.py` catch them off
# THIS import path — and because a second class of the same name would silently stop matching
# `except MasterKeyMissing` wherever the two paths met. There is exactly ONE class object.
#
# The dependency points implementation → interface, never the reverse: `sse/keys.py` is stdlib-only
# and knows nothing of grants, the lattice, or Fernet. `SseKeyProvider` is the Protocol this module's
# `OracleService` satisfies.
from .sse.keys import (MasterKeyMissing, MasterKeyUnavailable,  # noqa: F401
                       SseKeyProvider)

if TYPE_CHECKING:  # pragma: no cover
    from mantle.db.store import Database

    from .key_provider import KeyProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Crypto parameters
# ---------------------------------------------------------------------------

_MASTER_KEY_BYTES = 32          # 256-bit master keys
_CELL_KEY_BYTES = 32            # 256-bit AES-GCM cell keys
_SSE_KEY_BYTES = 32             # 256-bit per-principal SSE key (MANTLE-SSE)

# Fixed HKDF salt — versioned so a future v2 derivation scheme can coexist
# with v1-encrypted cells during a migration. Cell keys derived under
# different salts are independent.
_HKDF_SALT_V1 = b"agience-mantle-cell-key-v1"

# HKDF info string for the per-principal SSE key (MANTLE-SSE encrypted lexical).
# Distinct from cell-key derivation so the two key trees stay
# cryptographically independent — same master, different info, different key.
_HKDF_SSE_INFO = b"sse"


# ---------------------------------------------------------------------------
# Master key storage
# ---------------------------------------------------------------------------

class MasterKeyStore(Protocol):
    """Persistence boundary for principal master keys.

    The OracleService is agnostic to where keys live — the lattice, KMS,
    or a Shamir share quorum. Each backend implements ``get`` and ``put``.
    """

    def get(self, principal_id: str) -> bytes | None:
        """Return the unwrapped 256-bit master key for ``principal_id``, or None."""

    def put(self, principal_id: str, master_key: bytes) -> None:
        """Persist the master key for ``principal_id``. Implementations are
        responsible for at-rest encryption (Fernet wrapping, KMS, etc.)."""


class FernetMasterKeyStore:
    """Default master key store: Fernet-wraps each key with the platform
    encryption key, persists the wrapped token in a backing dict (production
    uses `LatticeMasterKeyStore`; tests use this with plain dicts).

    The MVP single-node implementation lives in process — there's no separate
    oracle node. Keys are unwrapped on read and never paged out.
    """

    def __init__(self, fernet: Fernet, persist: Mapping[str, str] | None = None) -> None:
        self._fernet = fernet
        # Persistence is delegated to caller-provided dict-like; production
        # uses LatticeMasterKeyStore, tests use plain dicts.
        self._persist: dict[str, str] = dict(persist or {})

    def get(self, principal_id: str) -> bytes | None:
        token = self._persist.get(principal_id)
        if not token:
            return None
        try:
            return self._fernet.decrypt(token.encode())
        except Exception as exc:
            logger.error("Failed to unwrap master key for %s: %s", principal_id, exc)
            return None

    def put(self, principal_id: str, master_key: bytes) -> None:
        token = self._fernet.encrypt(master_key).decode()
        self._persist[principal_id] = token

    @property
    def storage(self) -> dict[str, str]:
        """Read-only view of the wrapped storage. Intended for tests / inspection."""
        return dict(self._persist)


# `MasterKeyUnavailable` was defined here; it now lives in `sse/keys.py` and is imported at the top
# of this module. It means: a master key EXISTS (or may exist) but could not be read or unwrapped —
# deliberately distinct from "no key yet", because a caller must never treat it as first use. The
# recovery for first use is to generate and persist a new key, which overwrites the only copy of the
# old one. Failing a request is recoverable; that is not.


class LatticeMasterKeyStore:
    """The durable master key store over the standalone lattice (THE production backend).

    Envelope contract: only WRAPPED DEKs touch storage — KEK custody stays with the KeyProvider
    (local file | KMS | Vault) — persisted as a typed plane in the one store, ids namespaced so
    a principal id can never collide with an artifact id.

    ⛔ `get` IS FAIL-CLOSED, AND THE DISTINCTION IS LOAD-BEARING: absence returns None (genuine
    first use — safe to generate); a READ OR UNWRAP FAILURE RAISES `MasterKeyUnavailable`. The
    lattice-era predecessor once returned None for all three conditions, and
    `get_or_create_master_key` reads None as "first use" and OVERWRITES the stored wrapped DEK —
    so a transient read error or a rotated KEK destroyed the only copy of the key and, with it,
    every cell and content blob encrypted under it, silently, presented as "no results".
    Refusing to serve is recoverable; overwriting the only copy of the key is not."""

    CONTENT_TYPE = "application/vnd.agience.master-key+json"

    def __init__(self, key_provider: "KeyProvider", db_factory: Callable[[], object]) -> None:
        self._kek = key_provider
        self._db_factory = db_factory

    @staticmethod
    def _id(principal_id: str) -> str:
        return "master-key:" + principal_id

    def get(self, principal_id: str) -> bytes | None:
        try:
            doc = self._db_factory().artifacts.get_artifact(self._id(principal_id))
        except Exception as exc:
            logger.error("Master key read failed for %s: %s", principal_id, exc)
            raise MasterKeyUnavailable(
                f"could not read the master key for {principal_id!r}; refusing to continue, "
                f"because generating a replacement would overwrite it: {exc}"
            ) from exc
        if doc is not None and doc.get("content_type") != self.CONTENT_TYPE:
            doc = None                       # id-scoping: never read a non-key doc as a key
        token = (doc or {}).get("token")
        if not token:
            return None                      # genuinely absent — first use, safe to generate
        try:
            return self._kek.unwrap(token)
        except Exception as exc:
            logger.error("Failed to unwrap master key for %s: %s", principal_id, exc)
            raise MasterKeyUnavailable(
                f"a wrapped master key EXISTS for {principal_id!r} but could not be unwrapped "
                f"(wrong KEK / rotated provider?); refusing to overwrite it with a new key: {exc}"
            ) from exc

    def put(self, principal_id: str, master_key: bytes) -> None:
        token = self._kek.wrap(master_key)
        self._db_factory().artifacts.put_artifact({
            "id": self._id(principal_id),
            "content_type": self.CONTENT_TYPE,
            "token": token,
        })


# ---------------------------------------------------------------------------
# Authorization — key issuance is coupled to the grant check
# ---------------------------------------------------------------------------
#
# Canonical plan §5.3 ("Now (single node)"): *couple key/anchor-position issuance
# to the grant check immediately so the trust boundary matches the target.*
#
# Before this, `get_or_create_master_key(principal_id)` minted or unwrapped ANY
# principal's key for ANY caller: the identity came from the OBJECT BEING READ,
# never from the REQUESTER. Access was therefore a permission CHECK somewhere up
# the call stack — bypassable by reaching the oracle directly — rather than a
# property of key custody. The required property is "if you don't have a grant,
# you simply cannot", and that can only hold if the grant is verified INSIDE
# issuance, where no call site can route around it.
#
# ⚠ THAT PARAGRAPH WAS TRUE ABOUT THE INTENT AND FALSE ABOUT THE RESULT, AND THE
# GAP WENT UNNOTICED BECAUSE IT WAS WRITTEN AS THOUGH ALREADY ACHIEVED. Verifying
# "inside issuance" is necessary but NOT sufficient: the verifier can only be as
# trustworthy as the requester identity handed to it, and that identity was a
# caller-supplied string. Coupling the check to issuance while leaving the
# requester assertable moved the bypass, it did not close it — `KeyPurpose.SELF`
# let any caller name any principal and skip the verifier entirely.
#
# The property now rests on TWO legs, and it holds only while both stand:
#
#   1. WHO is asking is authenticated, not asserted — `requester_id` must equal the
#      acting principal resolved at the request boundary (`services.acting_principal`).
#   2. WHETHER they may is decided by the grant ledger, inside issuance, on EVERY
#      arm and EVERY call, cached or not.
#
# Leg 1 is the one that was missing. If a future change reintroduces a path where
# `requester_id` comes from data rather than from the caller — a document field, a
# collection's lineage root, a queue payload — leg 1 is gone and the check silently
# becomes an identity comparison again, whatever leg 2 does.


#: Actions that may bring a principal's master key into existence. Drawn from
#: CRUDEASIO (see ``entities.grant``); everything else — notably ``read`` — may only
#: USE a key that already exists.
#:
#: An allow-list, not a deny-list: an action nobody thought about must fail to create
#: rather than silently qualify. A new verb that genuinely needs to mint has to be
#: added here deliberately.
_WRITE_ACTIONS = frozenset({"create", "update", "delete", "add", "share", "invoke", "admin"})


# `MasterKeyMissing` was defined here; it now lives in `sse/keys.py` and is imported at the top of
# this module (same class object, so every `except MasterKeyMissing` in the tree still matches).
# It means: no master key EXISTS for this principal, and this request may not create one. It
# subclasses `MasterKeyUnavailable` so existing handling — which already treats "the key is not
# usable" as a hard stop rather than an empty result — applies unchanged. The distinction is *why*:
# absent rather than unreadable.


class GrantDenied(KeyCustodyDenied):
    """The requester holds no grant reaching the requested context — no key is issued.

    Deliberately an exception rather than a ``None`` return: a caller that gets no
    key must fail, not silently continue with a degraded/absent key and report
    "no results". This is the same lesson as :class:`MasterKeyUnavailable`.
    """


class KeyPurpose(str, enum.Enum):
    """Why a key is being requested. Narrows the check — it can never skip one.

    ⚠ HISTORY, BECAUSE THE OLD DOCSTRING HERE WAS FALSE. It claimed *"SELF is not a
    bypass: the oracle checks requester_id == principal_id"*. That check compared two
    values the CALLER supplied, so it was an identity, not a check — and the SELF arm
    returned before the grant verifier was consulted at all. A verifier hard-wired to
    ``return False`` was defeated and four principals' master keys were harvested with
    the verifier recording zero calls.

    Two things changed. ``requester_id`` is now bound to the authenticated acting
    principal (:mod:`services.acting_principal`), so a caller can no longer name a
    requester it is not; and **both arms now run the grant verifier**. ``SELF`` is
    therefore strictly NARROWER than ``GRANT``, never wider — selecting it can only
    add a constraint, so it is no longer a purpose worth choosing to gain anything.
    """

    #: The requester is a third party and must hold a grant reaching the context.
    #: Verified against the grant ledger via the light cone. This is the query path.
    GRANT = "grant"

    #: The requester additionally asserts it IS the context principal. Verified by
    #: identity equality AGAINST THE AUTHENTICATED CALLER, *and* by the same grant
    #: check ``GRANT`` runs.
    #:
    #: ⚠ RETAINED FOR COMPATIBILITY; production paths use ``GRANT``. Because it now
    #: adds a constraint rather than replacing one, it grants nothing extra — which
    #: is the whole point. Do not reintroduce an early return here.
    SELF = "self"


@dataclass(frozen=True)
class KeyRequest:
    """Who is asking for a key, and under what authority.

    Required — there is no default and no "anonymous" construction. An OPTIONAL
    requester would default to the previous insecure behaviour at every call site
    that simply didn't pass one, and nobody would notice; a required argument makes
    an unauthenticated key request a ``TypeError`` at the call site instead.
    """

    requester_id: str
    purpose: KeyPurpose
    requester_type: str = "user"
    action: str = "read"

    def __post_init__(self) -> None:
        if not self.requester_id:
            raise ValueError("KeyRequest.requester_id is required")
        if not isinstance(self.purpose, KeyPurpose):
            raise ValueError("KeyRequest.purpose must be a KeyPurpose")


class GrantVerifier(Protocol):
    """Decides whether ``requester_id`` may obtain keys for a context.

    Kept a protocol so the single-node in-process verifier can later be replaced by
    the distributed one (signed hash-chained ledger, independently checked by each
    Shamir peer) without touching the oracle.
    """

    def authorized(
        self,
        *,
        requester_id: str,
        requester_type: str,
        principal_id: str,
        collection_id: Optional[str],
        action: str,
    ) -> bool:
        """True iff the requester holds a grant reaching ``(principal_id, collection_id)``."""


class LightConeGrantVerifier:
    """Grant verification via the existing :class:`LightConeResolver`.

    Deliberately does NOT reimplement grant resolution. It calls
    ``resolve_authorized_contexts`` — the exact function the query path already
    uses to decide which contexts a principal may search — so the key the oracle
    issues and the contexts the search may touch are decided by one piece of code.
    Two implementations would be two chances to disagree, and the disagreement
    would be silent.

    ``requester_type`` REACHES THE LOOKUP. It was previously accepted, used as a
    cache-key component, and then dropped, so every resolution ran as the ledger's
    default grantee kind regardless of who asked. The two vocabularies differ; the
    mapping is in ``lightcone.ledger_grantee_type`` — read that comment before
    changing this line.

    Results are memoized per ``(requester, type, action)`` for ``ttl_s`` seconds.
    Without it a single query re-walks the whole light cone once per cell. The TTL
    is short and bounds post-revocation validity in the same way the target
    design's signed key TTL will; it is a cache of an AUTHORIZATION DECISION, so it
    is deliberately much shorter-lived than the master-key cache.
    """

    def __init__(self, db, *, resolver=None, ttl_s: float = 30.0) -> None:
        self._db = db
        self._resolver = resolver
        self._ttl_s = float(ttl_s)
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str, str], tuple[float, frozenset]] = {}

    def _contexts(self, requester_id: str, requester_type: str, action: str) -> frozenset:
        ck = (requester_id, requester_type, action)
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(ck)
            if hit is not None and hit[0] > now:
                return hit[1]

        from .lightcone import LightConeResolver
        from .sse.router_accessor import resolve_authorized_contexts

        resolver = self._resolver or LightConeResolver(self._db)
        pairs = frozenset(
            resolve_authorized_contexts(
                self._db, requester_id, lightcone=resolver, action=action,
                principal_type=requester_type,
            )
        )
        with self._lock:
            self._cache[ck] = (now + self._ttl_s, pairs)
        return pairs

    def authorized(
        self,
        *,
        requester_id: str,
        requester_type: str,
        principal_id: str,
        collection_id: Optional[str],
        action: str,
    ) -> bool:
        pairs = self._contexts(requester_id, requester_type, action)
        if collection_id:
            return (principal_id, collection_id) in pairs
        # Principal-scoped key (e.g. the SSE key, which spans a principal's whole
        # corpus): authorized iff the requester reaches AT LEAST ONE collection
        # under that principal. Scoping below the principal is not possible for a
        # key that is derived per-principal by construction.
        return any(p == principal_id for p, _ in pairs)

    def invalidate(self, requester_id: Optional[str] = None) -> None:
        """Drop memoized authorization decisions (grant change / revocation)."""
        with self._lock:
            if requester_id is None:
                self._cache.clear()
            else:
                for k in [k for k in self._cache if k[0] == requester_id]:
                    self._cache.pop(k, None)


# ---------------------------------------------------------------------------
# OracleService
# ---------------------------------------------------------------------------

class OracleService:
    """Single-node, in-process key custodian. MVP implementation.

    Every key-issuing method takes a REQUIRED :class:`KeyRequest` and verifies it
    before any key material is read, derived or returned. See the module's
    "Authorization" section for why the check lives here and not at the call site.
    """

    def __init__(
        self,
        store: MasterKeyStore,
        *,
        grant_verifier: Optional[GrantVerifier] = None,
    ) -> None:
        self._store = store
        self._verifier = grant_verifier
        self._lock = threading.RLock()
        # Cache unwrapped master keys for the process lifetime. Trade-off:
        # crypto round-trip cost vs. RAM. 32 bytes per principal is cheap.
        #
        # ⚠ THIS CACHE IS KEYED BY PRINCIPAL ONLY, AND THAT IS DELIBERATE — BUT IT
        # IS ONLY SAFE BECAUSE THE GRANT CHECK RUNS *BEFORE* THE CACHE IS READ.
        # Keyed by principal alone and consulted first, caller A would ride in on
        # caller B's authorized fetch: B's grant populates the entry, then A's
        # unauthorized request finds it warm and is served without ever being
        # checked — the cache would silently become an authorization bypass with a
        # process-lifetime TTL. Keying it by (requester, principal) instead would
        # look safer and be worse: it caches the same 32 bytes once per requester
        # and still never re-checks a revoked grant. So: the cache stores KEY
        # MATERIAL (whose value genuinely does not depend on who asks) and the
        # authorization decision is made on EVERY call, cached or not, by
        # `_authorize` at the top of `get_or_create_master_key`.
        self._cache: dict[str, bytes] = {}

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    def _authorize(
        self, principal_id: str, collection_id: Optional[str], request: KeyRequest
    ) -> None:
        """Raise :class:`GrantDenied` unless ``request`` may obtain this key.

        Runs before the master-key cache is consulted, before the store is read,
        and before any derivation — so there is no ordering under which key
        material is produced for an unverified requester.
        """
        if not isinstance(request, KeyRequest):
            raise TypeError(
                "a KeyRequest is required to obtain key material; key issuance is "
                "coupled to the grant check and there is no unauthenticated path"
            )

        # ⚠ STEP 1 — THE REQUESTER MUST BE AUTHENTICATED, NOT ASSERTED.
        #
        # This is the load-bearing line, and its absence was the root defect. Before
        # it, `requester_id` was just a string the caller chose, so `requester_id ==
        # principal_id` compared two caller-supplied values — an identity dressed as
        # a check. Binding it to the acting principal means a caller can only ever
        # ask as ITSELF; naming someone else is now a denial rather than a
        # promotion. `require_acting_principal` raises when nothing is in scope, so
        # an unauthenticated path gets no key instead of an unchecked one.
        actor = require_acting_principal()
        if request.requester_id != actor.principal_id:
            raise GrantDenied(
                f"requester_id {request.requester_id!r} is not the authenticated "
                f"acting principal {actor.principal_id!r}; a caller may only request "
                f"keys as itself"
            )

        # STEP 2 — SELF NARROWS, IT DOES NOT SKIP. Falls through to the grant check
        # below rather than returning, so this arm is strictly stronger than GRANT.
        if request.purpose is KeyPurpose.SELF and request.requester_id != principal_id:
            raise GrantDenied(
                f"self-issuance requires requester == principal, but "
                f"{request.requester_id!r} != {principal_id!r}"
            )

        if self._verifier is None:
            # Fail CLOSED. An oracle with no way to check grants cannot establish
            # that the requester has one, and "we could not check, so we allowed
            # it" is precisely the fail-open shape this change exists to remove.
            raise GrantDenied(
                f"no grant verifier is wired into this oracle; refusing to issue a "
                f"key for principal {principal_id!r} to {request.requester_id!r}"
            )

        if not self._verifier.authorized(
            requester_id=request.requester_id,
            requester_type=request.requester_type,
            principal_id=principal_id,
            collection_id=collection_id,
            action=request.action,
        ):
            raise GrantDenied(
                f"{request.requester_id!r} holds no {request.action!r} grant reaching "
                f"principal {principal_id!r} / collection {collection_id!r}"
            )

    def authorize(
        self, principal_id: str, collection_id: Optional[str], request: KeyRequest
    ) -> None:
        """Run the authorization check WITHOUT deriving or returning key material.

        For callers that hold a plaintext CACHE and must decide whether to serve it.
        A cache of decrypted data is a key-equivalent: serving from it without a
        check grants exactly what handing over the key would, so it needs the same
        gate — but calling a key-deriving method just to re-authorize would pay for
        an HKDF on every cache hit, and a caller wanting to avoid that cost is
        precisely how a cache ends up being read before the oracle.

        So the check is available on its own. Raises :class:`GrantDenied` /
        :class:`NoActingPrincipal` exactly as the issuing methods do; returns
        ``None`` when authorized.

        See ``engine._load_cell``, whose plaintext cell cache is gated by this.
        """
        self._authorize(principal_id, collection_id, request)

    # ------------------------------------------------------------------
    # Master key lifecycle
    # ------------------------------------------------------------------

    def get_or_create_master_key(
        self,
        principal_id: str,
        request: KeyRequest,
        *,
        collection_id: Optional[str] = None,
    ) -> bytes:
        """Return the principal's master key, generating + persisting on first call.

        ``request`` is REQUIRED and is verified before any key material is touched;
        see :meth:`_authorize`. ``collection_id`` narrows the grant check to one
        context when the caller has one (cell keys do; the per-principal SSE key
        does not).

        Raises :class:`GrantDenied` when the requester holds no grant reaching the
        context, and :class:`MasterKeyMissing` when no key exists and this request is
        not entitled to create one. Thread-safe: concurrent first-access calls won't
        generate duplicate keys for the same principal.

        ⚠ ONLY A WRITE CREATES A KEY — see :data:`_WRITE_ACTIONS`. A read that finds
        no key now FAILS instead of minting one, for two independent reasons:

        1. *Mint-ahead.* ``get_or_create`` let a caller name a principal that does
           not exist yet, cause a key to be generated and persisted, and keep the
           bytes — so every artifact later written under that principal was readable
           by whoever pre-seeded it. Authenticating the requester (see
           :meth:`_authorize`) already makes this hard, since no grant can reach a
           principal that does not exist; refusing to create on a read closes it
           outright rather than relying on that.
        2. *A read that mints is a silent data-loss detector that stays silent.* If a
           principal's key is missing because the store lost it, minting a fresh one
           on read returns a VALID key that decrypts nothing — surfacing as "no
           results" rather than "the key is gone". That is the same failure
           :class:`MasterKeyUnavailable` exists to prevent, arriving by a different
           door.
        """
        if not principal_id:
            raise ValueError("principal_id is required")

        # ⚠ ORDERING IS LOAD-BEARING: authorize FIRST, unconditionally, before the
        # cache read below. See the `_cache` comment in __init__.
        self._authorize(principal_id, collection_id, request)

        # Fast path: already cached.
        cached = self._cache.get(principal_id)
        if cached is not None:
            return cached

        with self._lock:
            # Double-check after acquiring the lock.
            cached = self._cache.get(principal_id)
            if cached is not None:
                return cached

            existing = self._store.get(principal_id)
            if existing is not None:
                if len(existing) != _MASTER_KEY_BYTES:
                    raise RuntimeError(
                        f"Master key for {principal_id} is {len(existing)} bytes, "
                        f"expected {_MASTER_KEY_BYTES}"
                    )
                self._cache[principal_id] = existing
                return existing

            # No key exists. Only a WRITE may bring one into being.
            if request.action not in _WRITE_ACTIONS:
                raise MasterKeyMissing(
                    f"no master key exists for principal {principal_id!r} and "
                    f"action {request.action!r} does not create one; refusing to mint "
                    f"a key on a read (a fresh key would decrypt nothing and report "
                    f"'no results' instead of surfacing a missing key)"
                )

            # First WRITE by this principal — generate.
            master_key = os.urandom(_MASTER_KEY_BYTES)
            self._store.put(principal_id, master_key)
            self._cache[principal_id] = master_key
            logger.info("Generated new MANTLE master key for principal=%s", principal_id)
            return master_key

    # ------------------------------------------------------------------
    # Cell key derivation
    # ------------------------------------------------------------------

    def derive_cell_key(
        self, principal_id: str, collection_id: str, cluster_id: str, request: KeyRequest
    ) -> bytes:
        """HKDF(master_key, info=collection_id ‖ 0x00 ‖ cluster_id) → 256-bit AES key.

        ``request`` is REQUIRED; the grant check runs against the specific
        ``(principal_id, collection_id)`` context and raises :class:`GrantDenied`
        when the requester cannot reach it.

        ``cluster_id`` is the routing anchor of the cell (canonical plan §5.1:
        the AnchorSet IS the partition; one cell per ``(principal, collection,
        anchor)``) and is required — routing has no flat fallback, so there is no
        anchor-less key. Deterministic; cell keys are never persisted — callers
        re-derive on demand.
        """
        if not principal_id or not collection_id:
            raise ValueError("principal_id and collection_id are required")

        master_key = self.get_or_create_master_key(
            principal_id, request, collection_id=collection_id
        )
        return self._derive(master_key, collection_id, cluster_id)

    # ------------------------------------------------------------------
    # SSE key derivation (MANTLE-SSE encrypted lexical, Step 2.6)
    # ------------------------------------------------------------------

    def derive_sse_key(self, principal_id: str, request: KeyRequest) -> bytes:
        """HKDF(master_key, info='sse') → 256-bit principal SSE key.

        ``request`` is REQUIRED. The SSE key is per-principal, so the grant check
        is principal-scoped: the requester must reach at least one collection under
        ``principal_id``. Raises :class:`GrantDenied` otherwise.

        The SSE key is derived per-principal (not per-collection) because SSE
        posting lists span a principal's entire corpus. Per-blind-token
        encryption keys are subsequently derived from the SSE key inside
        the posting list manager (:mod:`mantle.search.mantle.sse.posting`).

        Deterministic — re-derivation yields the same key.
        """
        if not principal_id:
            raise ValueError("principal_id is required")

        master_key = self.get_or_create_master_key(principal_id, request)
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=_SSE_KEY_BYTES,
            salt=_HKDF_SALT_V1,
            info=_HKDF_SSE_INFO,
        )
        return hkdf.derive(master_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive(master_key: bytes, collection_id: str, cluster_id: str) -> bytes:
        """Run the HKDF-SHA256 derivation for one cell.

        Info = ``collection_id ‖ 0x00 ‖ cluster_id`` — one formula, binding the
        key to exactly one ``(master_key, collection_id, cluster_id)`` tuple.
        ``cluster_id`` is always a real routing anchor; there is no anchor-less
        key.
        """
        info = collection_id.encode("utf-8") + b"\x00" + cluster_id.encode("utf-8")
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=_CELL_KEY_BYTES,
            salt=_HKDF_SALT_V1,
            info=info,
        )
        return hkdf.derive(master_key)

    # ------------------------------------------------------------------
    # Cache management (mainly for tests + admin reload)
    # ------------------------------------------------------------------

    def evict(self, principal_id: str | None = None) -> None:
        """Drop cached master keys. Pass ``principal_id`` to evict one principal;
        omit to clear the whole cache."""
        with self._lock:
            if principal_id is None:
                self._cache.clear()
            else:
                self._cache.pop(principal_id, None)
