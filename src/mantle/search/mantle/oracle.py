"""OracleService — in-process key custodian for MANTLE encrypted search.

Holds per-principal 256-bit master keys in memory, loaded lazily from
Fernet-wrapped storage on first access. Derives per-cell AES-256-GCM keys via
HKDF on demand — cell keys are never persisted.

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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Optional, Protocol

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# The authenticated caller, and the base class for refusals. `services` has an empty
# __init__ and this module imports nothing back, so there is no cycle.
from mantle.services.acting_principal import KeyCustodyDenied, require_acting_principal

# The refusals this module RAISES, defined next door rather than here. One class object per
# name either way; the difference is that `sse/narrowing.py` can catch them without importing the
# custody implementation. See `custody.py` for why that direction matters. Re-exported below,
# so `from ..oracle import MasterKeyMissing` keeps resolving to the same object it always did.
from .custody import MasterKeyMissing, MasterKeyUnavailable

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



#: The framed master-key payload: ``MK1:`` ‖ len(principal)(2, big-endian) ‖ principal ‖ DEK.
#:
#: The wrap names who the key is for. The KEK wrap protects the DEK's confidentiality and
#: integrity and nothing else — `Fernet.encrypt` takes no associated data, `kms.encrypt` is called
#: with no `EncryptionContext`, and Vault's `context` is unused — so the only thing tying a
#: wrapped key to its principal was the document id it happened to be stored under, which is a
#: plaintext, mutable database field.
#:
#: That made keys relocatable: copy the `token` out of `master-key:alice` into
#: `master-key:mallory`, and the unwrap succeeds — nothing in the ciphertext disagrees. Mallory then
#: receives Alice's master key through the self-custody base case (a principal may always fetch its
#: own key), and derives Alice's content key, SSE key and every cell key offline, with the grant
#: ledger never consulted.
#:
#: Framing the principal INSIDE the authenticated plaintext fixes that for all three custody models
#: at once, without changing the `KeyProvider` interface: every provider already authenticates the
#: bytes it wraps, so a moved token now decrypts to a payload that names the wrong principal, and
#: the equality check below rejects it.
_MK_MAGIC = b"MK1:"

#: Counts unframed (pre-binding) master keys still being read, so the fallback below has a
#: measurable end. Counted only on SUCCESS — "un-migrated keys really out there", not failures.
legacy_unbound_master_keys: int = 0


def _frame_master_key(principal_id: str, dek: bytes) -> bytes:
    pid = principal_id.encode("utf-8")
    if len(pid) > 0xFFFF:
        raise ValueError("principal id too long to frame")
    return _MK_MAGIC + len(pid).to_bytes(2, "big") + pid + dek


def _unframe_master_key(principal_id: str, raw: bytes) -> bytes:
    """Return the DEK, or raise if the payload names a different principal.

    An unframed payload is a key written before the binding existed. Those are returned unchanged
    and counted — breaking every existing install to close a database-write-access attack would be
    the wrong trade, and re-wrapping happens on the next `put`.
    """
    global legacy_unbound_master_keys
    if not raw.startswith(_MK_MAGIC):
        legacy_unbound_master_keys += 1
        return raw
    pid_len = int.from_bytes(raw[4:6], "big")
    bound_to = raw[6:6 + pid_len].decode("utf-8", "replace")
    if bound_to != principal_id:
        raise MasterKeyUnavailable(
            f"the wrapped master key stored for {principal_id!r} is bound to {bound_to!r} — "
            f"refusing to unwrap a key that was minted for a different principal"
        )
    return raw[6 + pid_len:]


class LatticeMasterKeyStore:
    """The durable master key store over the standalone lattice — the production backend.

    Envelope contract: only wrapped DEKs touch storage — KEK custody stays with the KeyProvider
    (local file | KMS | Vault) — persisted as a typed plane in the one store, ids namespaced so
    a principal id can never collide with an artifact id.
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
            return _unframe_master_key(principal_id, self._kek.unwrap(token))
        except MasterKeyUnavailable:
            raise                            # already precise: the key names another principal
        except Exception as exc:
            logger.error("Failed to unwrap master key for %s: %s", principal_id, exc)
            raise MasterKeyUnavailable(
                f"a wrapped master key EXISTS for {principal_id!r} but could not be unwrapped "
                f"(wrong KEK / rotated provider?); refusing to overwrite it with a new key: {exc}"
            ) from exc

    def put(self, principal_id: str, master_key: bytes) -> None:
        token = self._kek.wrap(_frame_master_key(principal_id, master_key))
        self._db_factory().artifacts.put_artifact({
            "id": self._id(principal_id),
            "content_type": self.CONTENT_TYPE,
            "token": token,
        })


# ---------------------------------------------------------------------------
# Authorization — key issuance is coupled to the grant check
# ---------------------------------------------------------------------------
#
# Canonical plan §5.3 ("Now (single node)"): key/anchor-position issuance is coupled
# to the grant check so the trust boundary matches the target. Every key-issuing call
# is verified inside issuance, where no call site can route around it — access is a
# property of key custody, not a permission check somewhere up the call stack.
#
# The property rests on two legs, and it holds only while both stand:
#
#   1. Who is asking is authenticated, not asserted — `requester_id` must equal the
#      acting principal resolved at the request boundary (`services.acting_principal`).
#   2. Whether they may is decided by the grant ledger, inside issuance, on every
#      arm and every call, cached or not.
#
# Leg 1 is load-bearing: if `requester_id` ever comes from data rather than from the
# caller — a document field, a collection's lineage root, a queue payload — the check
# silently becomes an identity comparison again, whatever leg 2 does.


#: Actions that may bring a principal's master key into existence. Drawn from
#: CRUDEASIO (see ``entities.grant``); everything else — notably ``read`` — may only
#: use a key that already exists.
#:
#: An allow-list, not a deny-list: an action nobody thought about must fail to create
#: rather than silently qualify. A new verb that genuinely needs to mint has to be
#: added here deliberately.
_WRITE_ACTIONS = frozenset({"create", "update", "delete", "add", "share", "invoke", "admin"})


class GrantDenied(KeyCustodyDenied):
    """The requester holds no grant reaching the requested context — no key is issued.

    Deliberately an exception rather than a ``None`` return: a caller that gets no
    key must fail, not silently continue with a degraded/absent key and report
    "no results". This is the same lesson as :class:`MasterKeyUnavailable`.
    """


class KeyPurpose(str, enum.Enum):
    """Why a key is being requested. Narrows the check — it can never skip one.

    ``requester_id`` is bound to the authenticated acting principal
    (:mod:`services.acting_principal`), so a caller cannot name a requester it is
    not, and both arms run the grant verifier. ``SELF`` is therefore strictly
    narrower than ``GRANT``, never wider — selecting it can only add a constraint.
    """

    #: The requester is a third party and must hold a grant reaching the context.
    #: Verified against the grant ledger via the light cone. This is the query path.
    GRANT = "grant"

    #: The requester additionally asserts it is the context principal. Verified by
    #: identity equality against the authenticated caller, *and* by the same grant
    #: check ``GRANT`` runs.
    #:
    SELF = "self"


@dataclass(frozen=True)
class KeyRequest:
    """Who is asking for a key, and under what authority.

    Required — there is no default and no "anonymous" construction. An optional
    requester would let a call site that simply didn't pass one silently go
    unauthenticated, and nobody would notice; a required argument makes an
    unauthenticated key request a ``TypeError`` at the call site instead.
    """

    requester_id: str
    purpose: KeyPurpose
    requester_type: str = "user"
    action: str = "read"
    #: The `created_by` of the document whose key is being minted — set only by the encrypt path.
    #:
    #: The creator holds the key of the thing it is creating. A self-rooted (top-level) artifact
    #: keys to its own id, and at the instant it is written no grant can yet reach that id — the id
    #: was minted microseconds earlier. Rooting such artifacts at `created_by` instead would let the
    #: first write succeed while leaving the artifact permanently unshareable: a grantee's grant
    #: resolves to `(artifact_id, artifact_id)` while the key asks about the creator, so
    #: `/grants/my-access` would say allowed while the read still 404'd.
    #:
    #: Honoured on the minting path only. On a read this would let anyone who can name themselves
    #: as the creator obtain a key; `may_create` is what confines it to the moment of creation.
    creator_id: Optional[str] = None
    #: May this request bring a key into being that does not exist yet?
    #:
    #: Separate from `action`, because they answer different questions. `action` says which right
    #: the requester must hold; this says whether the call is entitled to mint. They coincide for
    #: every write, but the encrypt path demands `read` — holding a content key is the ability to
    #: read it, and demanding `update` would reject a legitimate *create* — so it needs `may_create`
    #: to be able to create the very key it is about to encrypt with.
    #:
    #: Defaults to False so the read protection below is what a caller gets by not thinking about
    #: it — the same allow-list reasoning `_WRITE_ACTIONS` states.
    may_create: bool = False

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


def _seconds_until(expires_at: Optional[str], now: datetime) -> Optional[float]:
    """Seconds from *now* until an ISO-8601 ``expires_at``, or ``None`` if there isn't one.

    Tolerates a trailing ``Z`` and a naive timestamp (read as UTC, which is what every
    writer in this codebase produces). An unparseable value returns ``0.0`` — a grant
    whose expiry cannot be read must not be memoized on the strength of it.
    """
    if not expires_at:
        return None
    raw = str(expires_at).strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed - now).total_seconds()


class LightConeGrantVerifier:
    """Grant verification via the existing :class:`LightConeResolver`.

    Deliberately does NOT reimplement grant resolution. It calls
    ``resolve_authorized_contexts`` — the exact function the query path already
    uses to decide which contexts a principal may search — so the key the oracle
    issues and the contexts the search may touch are decided by one piece of code.
    Two implementations would be two chances to disagree, and the disagreement
    would be silent.
    ``requester_type`` reaches the lookup.

    Results are memoized per ``(requester, type, action)`` for ``ttl_s`` seconds.
    Without it a single query re-walks the whole light cone once per cell. The TTL
    is short and bounds post-revocation validity in the same way the target
    design's signed key TTL will; it is a cache of an authorization decision, so it
    is deliberately much shorter-lived than the master-key cache.

    Two properties of the memo are load-bearing and neither is optional:

    * ``ttl_s <= 0`` means **no memo at all** — nothing is stored and every call
      re-reads the ledger. That is the honest configuration for a deployment where
      :meth:`invalidate` cannot reach every process holding one (see
      ``wiring._verifier_ttl_s``), not a degenerate case to be rounded up to a
      small positive number.
    * An entry may not outlive the grants it was derived from. The store filters
      expired grants at read; the memo would not, so its deadline is clamped to the
      earliest ``expires_at`` among the requester's grants (:meth:`_memo_ttl`).
      Expiry is revocation that arrives on a clock, and a cache that ignores it is
      stale in the direction that grants access.

    ``clock`` is the monotonic time source, injectable so a test can advance a TTL
    without sleeping through it — the production arithmetic still runs for real.
    """

    def __init__(self, db, *, resolver=None, ttl_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._db = db
        self._resolver = resolver
        self._ttl_s = float(ttl_s)
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str, str], tuple[float, frozenset]] = {}

    def _contexts(self, requester_id: str, requester_type: str, action: str) -> frozenset:
        ck = (requester_id, requester_type, action)
        now = self._clock()
        if self._ttl_s > 0:
            with self._lock:
                hit = self._cache.get(ck)
                if hit is not None and hit[0] > now:
                    return hit[1]

        from .lightcone import LightConeResolver, resolve_authorized_contexts

        resolver = self._resolver or LightConeResolver(self._db)
        pairs = frozenset(
            resolve_authorized_contexts(
                self._db, requester_id, lightcone=resolver, action=action,
                principal_type=requester_type,
            )
        )
        ttl = self._memo_ttl(resolver, requester_id, requester_type)
        if ttl > 0:
            with self._lock:
                self._cache[ck] = (now + ttl, pairs)
        return pairs

    def _memo_ttl(self, resolver, requester_id: str, requester_type: str) -> float:
        """How long this entry may be held: ``ttl_s``, capped by the next grant expiry.

        The cap is what stops an entry warmed one second before a grant's ``expires_at``
        from answering with reach the ledger has already withdrawn. It reads the
        requester's SEED grants — the same set step 1 of the light cone starts from, for
        the same principal kind, via the resolver that just ran — rather than restating
        the "which grants does this principal hold" rule, which for a grant key means
        walking a bundle and would be a second copy free to disagree with the first.

        Costs one grant lookup, on the memo MISS only, on a path that has just walked the
        whole light cone. Grants without an ``expires_at`` — the common case — cap
        nothing, so the common case pays only that lookup.

        Falls back to the full TTL when the seed set cannot be read: the TTL is still a
        bound, and shortening it on a failed read would turn a store hiccup into an
        un-memoized authorization hot path.
        """
        ttl = self._ttl_s
        if ttl <= 0:
            return ttl

        # A resolver that does not expose its seed grants is a test double or a future
        # implementation; the plain TTL still bounds it.
        seed_grants = getattr(resolver, "_grants_for", None)
        if seed_grants is None:
            return ttl
        try:
            now = datetime.now(timezone.utc)
            for grant in seed_grants(requester_id, requester_type) or ():
                remaining = _seconds_until(getattr(grant, "expires_at", None), now)
                if remaining is not None and remaining < ttl:
                    ttl = max(remaining, 0.0)
        except Exception:  # noqa: BLE001 — store reads can raise broadly
            logger.debug("grant-expiry clamp: seed grant read failed", exc_info=True)
            return self._ttl_s
        return ttl

    def authorized(
        self,
        *,
        requester_id: str,
        requester_type: str,
        principal_id: str,
        collection_id: Optional[str],
        action: str,
    ) -> bool:
        # BASE CASE: a principal has custody of its own key. The light cone below resolves pairs
        # whose principal is a collection's origin root — and a user's first collections are
        # themselves self-rooted, so no pair ever carries the user as principal. Without this base
        # case, a principal could never obtain the key that encrypts its own content: a brand-new
        # principal could not write a first top-level artifact on any store, because the check would
        # be asking someone to be granted access to themselves.
        #
        # This is not a bypass. `_authorize` has already bound `requester_id` to the authenticated
        # acting principal and rejects any request naming someone else ("a caller may only request
        # keys as itself"), so reaching this line means the caller IS `principal_id`. It therefore
        # widens nothing: it cannot yield another principal's key. What it removes is a
        # circularity, not a check.
        #
        # Checked before the collection scope, and the position is what makes it reachable. Below
        # `if collection_id: return (principal, collection) in pairs` it is unreachable the moment a
        # caller names a scope, and the C10 repair names one on
        # every content write (`content_service.put_bytes_encrypted` now passes `collection_id`).
        # The effect was exactly the failure the paragraph above predicts: a first
        # `create_artifact` on a virgin store answered `ContentEncryptionError: content encryption
        # unavailable`, reproduced 2026-08-17 by following the README quickstart on a clean install.
        #
        # C10 is right and stays. It bound the AAD to the scope, which decides which BLOB opens —
        # a different question from whose key this is. The scope still narrows every request for
        # SOMEONE ELSE'S key, which is what C10's finding was about ("reach any one collection under
        # an owner, get the key to all of them"). A master key is per-principal by construction, so
        # on your OWN key there is nothing for a collection to narrow: refusing it there denies a
        # caller a key they already hold, and denies it in a way no grant can repair.
        if requester_id == principal_id:
            return True

        # ── one collection, walked rather than enumerated ────────────────────────────────────
        # The question is "may this requester reach THIS collection", and answering it does not
        # require the requester's whole light cone. Enumerating means materialising every
        # descendant of every grant — `list_origin_descendants` holds them all in a set — and that
        # does not survive a corpus: `stage.0.lexicon` has 1,841,335 members, so `edges_of` raises
        # `EdgesTruncated` at its cap and the cone cannot be computed at all. One large grant then
        # denied keys for every OTHER collection the requester held, because the failure was in
        # computing the answer rather than in the answer.
        #
        # `origin_chain` walks the other way: the collection, its root, then each origin ancestor,
        # stopping at the first edge whose propagate mask does not carry the action. It is the same
        # traversal `check_access` performs for an artifact and the same attenuation
        # `list_origin_descendants` applies going down, in one implementation — which is the point.
        # Two readers of one question is how they came to disagree at scale.
        #
        # It is not wider than the cone. A collection is authorized here only when the requester
        # holds a grant on it or on an ancestor that still conducts the action, which is exactly
        # what descending from those grants would have reached.
        if collection_id:
            return self._reaches(requester_id, requester_type, collection_id, action)

        # Principal-scoped key (e.g. the SSE key, which spans a principal's whole corpus):
        # authorized iff the requester reaches at least one collection under that principal.
        # Scoping below the principal is not possible for a key derived per-principal by
        # construction.
        #
        # Enumerating answered this by building every `(cell_principal, collection)` pair the
        # requester reaches and asking whether any names this principal — and that is the same
        # materialisation that raises on a 1.8-million-member collection. The SSE key IS
        # principal-scoped, so an index write took exactly this branch: the collection case could
        # be fixed and indexing would still fail.
        #
        # It has an exact answer that enumerates nothing. A cell principal is a collection's ORIGIN
        # ROOT, and `principal.resolve_cell_principal` states the property this turns on: it is
        # "stable and single-valued for the collection's whole sub-tree". So every collection
        # reachable THROUGH a granted resource shares that resource's root, and
        #
        #     any(p == principal_id for p, _ in pairs)   ==   any(root_of(r) == principal_id
        #                                                        for r in granted resources)
        #
        # which is `O(grants x depth)` walks up instead of one walk down over everything below.
        return self._roots_include(requester_id, requester_type, principal_id, action)

    def _roots_include(self, requester_id: str, requester_type: str,
                       principal_id: str, action: str) -> bool:
        """Does any resource this requester holds sit under `principal_id`'s origin root?

        A set membership over :meth:`_granted_roots`, which is where the walking happens and where
        it is memoized. `principal_id` is only ever COMPARED to the result of that walk — it does
        not steer it — which is what makes the walk shareable across every principal asked about.
        """
        return str(principal_id) in self._granted_roots(requester_id, requester_type, action)

    def _granted_roots(self, requester_id: str, requester_type: str, action: str) -> frozenset:
        """The origin roots of every resource this requester may `action`, memoized.

        ⚑ **Measured 2026-08-28 on 71/home, and this is why it exists.** `_roots_include` walked
        `_root_of(resource)` for every granted resource on EVERY call. That principal holds 7,268 of
        the store's 7,388 grants, and the SSE narrowing path asks the question once per principal
        whose index it might open — 51 principals in that index. One recall therefore ran **103,111
        SQL statements to return ZERO hits**, of which `_roots_include` was **11.51s of a 13.93s**
        `cProfile`, via 61,698 `_root_of` and 95,761 `get_origin_parent` calls.

        `_grants` was already memoized; the walk over its results was not. Hoisting it is the whole
        change: the roots are a fact about the REQUESTER's grants, so the 51 questions share one
        answer instead of each paying for it.

        📄 The same fix was made to the recall path's origin-chain walk on 2026-08-25 and did not
        reach this second walker — see `status/RETRIEVAL-PATH-2026-08-25.md`, which also records
        that its first attempt was written up as effective and never hit once. Hence
        `tests/test_granted_roots_is_memoized.py`, which counts the walks rather than trusting this
        paragraph.

        ⛔ **THE STALENESS WINDOW IS NOT NEW, AND MUST NOT BECOME NEW.** This reuses `_grants`'s own
        cache, key shape, lock, clock and `_memo_ttl` — the TTL capped by the requester's earliest
        grant expiry — so how long a REVOKED grant keeps working is exactly what it was before this
        method existed. `invalidate` clears it for free, because it is the same dict and this key
        carries `requester_id` first, which is what `invalidate(requester_id)` filters on. A cache
        of its own, with a lifetime of its own, would silently change post-revocation validity, and
        that is a policy change wearing a refactor's clothes.

        `ttl_s <= 0` still means NO memo: the walk runs per call, exactly as it did before.
        """
        from mantle.entities.grant import mask_of

        # ── the memo is OPTIONAL, and `ttl_s <= 0` already says so ──────────────────────────────
        # `_roots_include` used to need only `_grants` and `_root_of`, and tests build a verifier
        # with `LightConeGrantVerifier.__new__` and stub exactly those two seams — see
        # `test_a_virgin_store_takes_its_first_write.py`. Such an object has no `_cache`, `_clock`,
        # `_lock` or `_ttl_s` at all, and reading one unguarded turned a seam-level test into an
        # AttributeError. Absent infrastructure is read as `ttl_s = 0`, which this class ALREADY
        # defines as "no memo at all — nothing is stored and every call recomputes". So the fallback
        # is the documented configuration rather than a new branch invented to keep a test quiet.
        ttl_s = getattr(self, "_ttl_s", 0.0)
        ck = (requester_id, requester_type, "\x00roots:" + str(action))
        now = self._clock() if ttl_s > 0 else 0.0
        if ttl_s > 0:
            with self._lock:
                hit = self._cache.get(ck)
                if hit is not None and hit[0] > now:
                    return hit[1]

        allow, deny = [], set()
        for g in self._grants(requester_id, requester_type):
            rid = getattr(g, "resource_id", None)
            if not rid:
                continue
            m = mask_of(g)
            if m.is_deny and m.carries(action):
                deny.add(str(rid))
            elif m.allows(action):
                allow.append(str(rid))

        roots = set()
        for resource in allow:
            if resource in deny:
                continue
            try:
                roots.add(str(self._root_of(resource)))
            except Exception:  # noqa: BLE001 — an unreadable chain under-reaches, fail closed
                continue
        result = frozenset(roots)

        # The TTL is computed the way `_grants` computes it, from the same resolver, so the two
        # entries expire together rather than drifting apart.
        if ttl_s > 0:
            from .lightcone import LightConeResolver
            resolver = self._resolver or LightConeResolver(self._db)
            ttl = self._memo_ttl(resolver, requester_id, requester_type)
            if ttl > 0:
                with self._lock:
                    self._cache[ck] = (now + ttl, result)
        return result

    def _root_of(self, artifact_id: str) -> str:
        """The top of an artifact's origin chain — its cell principal. A seam, for the same reason
        `_chain` is one: a test describes a containment shape without reaching into the store."""
        from mantle.db.backend import get_origin_root

        return str(get_origin_root(self._db, artifact_id))

    def _reaches(self, requester_id: str, requester_type: str,
                 collection_id: str, action: str) -> bool:
        """Does a grant this requester holds sit on `collection_id`, or above it and still conduct?

        Deny is checked at every level before allow, matching `check_access`: a deny nearer the
        artifact refuses, and nothing further up re-allows it.

        The grant set comes from `_grants` below, which is memoized on the same key and TTL the
        pair cache used — so revocation is bounded exactly as it was, and `invalidate` still clears
        it.
        """
        from mantle.entities.grant import mask_of

        allow, deny = set(), set()
        for g in self._grants(requester_id, requester_type):
            rid = getattr(g, "resource_id", None)
            if not rid:
                continue
            m = mask_of(g)
            if m.is_deny and m.carries(action):
                deny.add(str(rid))
            elif m.allows(action):
                allow.add(str(rid))
        if not allow:
            return False

        try:
            for resource in self._chain(str(collection_id), action):
                if resource in deny:
                    return False
                if resource in allow:
                    return True
        except Exception:  # noqa: BLE001 — a malformed or unreadable chain authorizes nothing,
            return False   # which is the fail-closed direction
        return False

    def _chain(self, collection_id: str, action: str):
        """Where a grant could sit to reach `collection_id` under `action`, nearest first.

        One method for one external dependency, so a test can describe a containment shape without
        reaching past this object into the store. Production reads
        `lattice_api.origin_chain`, the same walk `check_access` performs for an artifact.
        """
        from mantle.db.backend import origin_chain

        return origin_chain(self._db, collection_id, action)

    def _grants(self, requester_id: str, requester_type: str):
        """The requester's grants, memoized on `(requester, type)` for the same TTL the resolved
        pairs used.

        The memo exists because the enumerating path re-walked the whole light cone once per cell.
        A walk does not need it for speed — it is a handful of edges — but it is kept because the
        TTL is what BOUNDS POST-REVOCATION VALIDITY, and dropping it would silently change when a
        revoked grant stops working. That is a policy change, not a refactor, so the memo stays and
        `invalidate` still clears it.
        """
        ck = (requester_id, requester_type, "\x00grants")
        now = self._clock()
        if self._ttl_s > 0:
            with self._lock:
                hit = self._cache.get(ck)
                if hit is not None and hit[0] > now:
                    return hit[1]

        from .lightcone import LightConeResolver

        resolver = self._resolver or LightConeResolver(self._db)
        grants = tuple(resolver._grants_for(requester_id, requester_type) or ())
        ttl = self._memo_ttl(resolver, requester_id, requester_type)
        if ttl > 0:
            with self._lock:
                self._cache[ck] = (now + ttl, grants)
        return grants

    def invalidate(self, requester_id: Optional[str] = None) -> None:
        """Drop memoized authorization decisions (grant change / revocation).

        Clears THIS verifier's memo, which is this process's. A deployment with more
        than one worker holds one memo per worker and this call reaches exactly one of
        them; ``wiring._verifier_ttl_s`` is where that fact is accounted for.
        """
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

    Every key-issuing method takes a required :class:`KeyRequest` and verifies it
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

        #
        # This is the load-bearing line: binding `requester_id` to the acting principal means a
        # caller can only ever ask as itself; naming someone else is a denial. Comparing
        # `requester_id == principal_id` on its own would compare two caller-supplied values — an
        # identity dressed as a check — so the comparison only means something once `requester_id`
        # is bound to an authenticated identity here. `require_acting_principal` raises when nothing
        # is in scope, so an unauthenticated path gets no key instead of an unchecked one.
        actor = require_acting_principal()
        if request.requester_id != actor.principal_id:
            raise GrantDenied(
                f"requester_id {request.requester_id!r} is not the authenticated "
                f"acting principal {actor.principal_id!r}; a caller may only request "
                f"keys as itself"
            )

        # SELF narrows; it does not skip. Falls through to the grant check
        # below rather than returning, so this arm is strictly stronger than a plain grant check.
        if request.purpose is KeyPurpose.SELF and request.requester_id != principal_id:
            raise GrantDenied(
                f"self-issuance requires requester == principal, but "
                f"{request.requester_id!r} != {principal_id!r}"
            )

        if self._verifier is None:
            # Fail closed. An oracle with no way to check grants cannot establish
            # that the requester has one, and "we could not check, so we allowed
            # it" is the fail-open shape this guards against.
            raise GrantDenied(
                f"no grant verifier is wired into this oracle; refusing to issue a "
                f"key for principal {principal_id!r} to {request.requester_id!r}"
            )

        # The create base case — the creator of an artifact holds that artifact's key at the
        # moment it creates it. Three conditions, and every one is load-bearing:
        #
        #   · `may_create`         — minting only. A read never takes this branch.
        #   · requester == creator — `requester_id` was bound to the authenticated acting principal
        #                            above ("a caller may only request keys as itself"), and
        #                            `creator_id` comes off the stored doc, never off the request
        #                            body. So this compares two server-side facts.
        #   · principal == collection — confines it to a self-rooted artifact keying to itself. A
        #                            collection's key has principal != collection and cannot be
        #                            obtained this way.
        #
        # It widens nothing that a grant would not: one statement later the creator receives an
        # owner grant on this same artifact; this authorizes the instant between, which is the only
        # instant in which no grant can exist yet.
        if (
            request.may_create
            and request.creator_id
            and request.requester_id == request.creator_id
            and collection_id
            and collection_id == principal_id
        ):
            return

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
        """Run the authorization check without deriving or returning key material.

        For callers that hold a plaintext cache and must decide whether to serve it.
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

        ``request`` is required and is verified before any key material is touched;
        see :meth:`_authorize`. ``collection_id`` narrows the grant check to one
        context when the caller has one (cell keys do; the per-principal SSE key
        does not).

        Raises :class:`GrantDenied` when the requester holds no grant reaching the
        context, and :class:`MasterKeyMissing` when no key exists and this request is
        not entitled to create one. Thread-safe: concurrent first-access calls won't
        generate duplicate keys for the same principal.

        Minting is refused on a read for two reasons:

        1. *Mint-ahead.* Allowing any caller to name a principal that does not exist
           yet, cause a key to be generated and persisted, and keep the bytes would
           make every artifact later written under that principal readable by
           whoever pre-seeded it. Authenticating the requester (see
           :meth:`_authorize`) already makes this hard, since no grant can reach a
           principal that does not exist; refusing to create on a read closes it
           outright rather than relying on that.
        2. *A read that mints would be a silent data-loss detector.* If a
           principal's key is missing because the store lost it, minting a fresh one
           on read would return a valid key that decrypts nothing — surfacing as "no
           results" rather than "the key is gone". That is the same failure
           :class:`MasterKeyUnavailable` exists to prevent, arriving by a different
           door.
        """
        if not principal_id:
            raise ValueError("principal_id is required")

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

            # No key exists. Only a write — or a caller that explicitly says it is entitled to
            # create one — may bring it into being. `may_create` exists because the encrypt path
            # demands `read` for sound authorization reasons yet must still mint on first write;
            # see the field's own note on KeyRequest.
            if request.action not in _WRITE_ACTIONS and not request.may_create:
                raise MasterKeyMissing(
                    f"no master key exists for principal {principal_id!r} and "
                    f"action {request.action!r} does not create one; refusing to mint "
                    f"a key on a read (a fresh key would decrypt nothing and report "
                    f"'no results' instead of surfacing a missing key)"
                )

            # First write by this principal — generate.
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

        ``request`` is required; the grant check runs against the specific
        ``(principal_id, collection_id)`` context and raises :class:`GrantDenied`
        when the requester cannot reach it.

        ``cluster_id`` is the routing anchor of the cell (canonical plan §5.1:
        the AnchorSet is the partition; one cell per ``(principal, collection,
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
    # SSE key derivation (MANTLE-SSE encrypted lexical)
    # ------------------------------------------------------------------

    def derive_sse_key(self, principal_id: str, request: KeyRequest) -> bytes:
        """HKDF(master_key, info='sse') → 256-bit principal SSE key.

        ``request`` is required. The SSE key is per-principal, so the grant check
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
