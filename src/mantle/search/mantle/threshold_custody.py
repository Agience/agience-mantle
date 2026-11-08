"""Threshold custody of master keys: each key is 2-of-3 Shamir shares, and no node holds two.

Selected by ``MANTLE_MASTER_KEY_CUSTODY=threshold`` (default ``lattice``, the KEK-wrapped key in
:class:`.oracle.LatticeMasterKeyStore`). The same `get`/`put` contract as that store, over three
shares of each 256-bit master key (`prism.trust.shamir`, x = 1, 2, 3):

    share 1   this node's lattice, in the ``master-key:<principal>`` row, wrapped by the KEK
              provider and framed with the principal (``MS1:``), as the whole key was
    share 2   Origin, over ``/internal/key-shares`` (`clients/origin_key_share_client.py`), in
              Origin's own encrypted table
    share 3   the escrow: sealed to the operator's age recipient (``MANTLE_ESCROW_RECIPIENT_FILE``,
              a public key) and written to the durable object store (``CONTENT_DURABLE_BUCKET``,
              ``escrow/<sha256(principal)>.<kcv>.<escrow_id>.age``), never to the lattice, so
              losing this node's store loses one share. Each split writes a new object, only if
              absent, and the row names it (its ``kcv`` and ``escrow_id``); no escrow object is
              ever overwritten, so an earlier split's share 3 survives a later one. The identity
              is kept offline by the operator; this node cannot open it. ``mantle.system.split_master_keys --export-escrow``
              copies every sealed escrow share into one file for the operator's offline copy.

Threshold custody is refused at startup when no durable object store is configured: there would
be nowhere for share 3 that is not beside share 1.

`get` combines shares 1 and 2. Shares 1 and 3 are combined only in recovery mode
(``MANTLE_MASTER_KEY_RECOVERY=escrow`` with ``MANTLE_ESCROW_IDENTITY_FILE``, which an operator sets
for a recovery and removes after), never as an automatic fallback. Each combine is checked against
the key-check value stored with share 1 (`key_check_value`, an HMAC of the key and the principal),
because a Shamir combine of mismatched shares returns wrong bytes rather than an error.

The row is the same document id as a KEK-wrapped key, with its own content type
(:data:`.oracle.SHARE_CONTENT_TYPE`), so a principal has one key row in either form, and each store
refuses the other's form rather than reading it as absent: the lattice store would otherwise mint a
replacement over a split key, and this store would otherwise mint over an unsplit one.

Refusal-to-overwrite holds as in the lattice store. `get` answers None only when nothing anywhere
holds a share: no row here, and Origin answering explicitly that it holds none. A row that exists
but cannot be opened, a row with no Origin share, and Origin unreachable all raise
:class:`MasterKeyUnavailable`.

Every write of a key row happens under the custody lock (:class:`CustodyLock`, a file beside the
lattice that the server and ``split_master_keys`` both take), and each transition is conditional on
the stored form read under that lock. `put` writes the row ``pending``, then the escrow share, then
Origin's share, and marks the row ``active`` only if it is still the pending row it wrote. A pending
row holds a key that was never returned to anyone (the oracle returns a key only after `put`
returns), so `get` reads it as absent and the next `put` replaces it, naming its key-check value to
Origin as superseded.

Origin releases share 2 only on the acting caller's own credential, which it verifies itself, and
only when that credential's subject reaches this key, by Origin's own walk over its replica of the
authority ledger (the principal itself, or a collection or artifact keyed under it that the subject
reaches for `read`). Each release names the collection or artifact the key is wanted for
(`get_scoped`), so reach to one place cannot open a key for another. `share_credential` picks the
credential:

  * a request: the bearer the caller presented, when it is a person's Origin-issued access token or
    a delegation Origin issued (`acting_principal.forwardable_credential`, set at the request
    boundary)
  * background work, and any work acting as the platform system principal: a delegation Origin
    mints for the one principal being opened and the block's named job
    (`OriginShareCustodian.background_delegation`), minutes long, and only under a live job grant
    for both that the person, or the collection's Admin, made. Without one, that work refuses and
    says so. Nothing long-lived is kept on this node.
  * a grant key (``agk_…``) is Mantle's own credential, which Origin cannot verify: refused
    (:class:`GrantKeyCannotOpenShare`), on a cached key too

What each node yields alone. Mantle's store (a disk, a backup) holds share 1 under the KEK: one
share, which reveals nothing of the key. The durable store holds share 3 sealed to an offline key.
Origin holds share 2 and releases it only for a credential that reaches the key. A running Mantle
with its signing key opens a key only with such a credential: a person's, for what that person
reaches and the life of their token, or, while the operator's background authority is in force, a
delegation Origin mints for one principal. A combined key stays in the oracle's process cache for at
most `oracle.MASTER_KEY_CACHE_TTL_MAX` seconds, after which Origin is asked again.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Callable, Optional, Protocol

from .custody import MasterKeyUnavailable
from .oracle import SHARE_CONTENT_TYPE, LatticeMasterKeyStore

logger = logging.getLogger(__name__)

#: The two thresholds custody is built for. Named once; the share x-coordinates follow from them.
K, N = 2, 3
MANTLE_X, ORIGIN_X, ESCROW_X = 1, 2, 3

CUSTODY_TAG = "threshold-2of3"
_SHARE_MAGIC = b"MS1:"
_KCV_INFO = b"agience-master-key-check-v1\x00"
ESCROW_PREFIX = "escrow/"


class ShareCustodian(Protocol):
    def get_share(self, principal_id: str, *, credential: str,
                  collection_id: Optional[str] = None,
                  artifact_id: Optional[str] = None) -> Optional[dict]: ...

    def background_delegation(self, principal_id: str, job: str) -> str: ...

    def put_share(self, principal_id: str, share: dict, *,
                  supersedes: Optional[str] = None,
                  credential: Optional[str] = None) -> str: ...


class EscrowStore(Protocol):
    def put_if_absent(self, name: str, sealed: bytes) -> None: ...

    def get(self, name: str) -> Optional[bytes]: ...


# -- the acting caller's credential --------------------------------------------------------------

class ShareRefused(MasterKeyUnavailable):
    """Origin answered and refused to release its share, or there is no credential to show it.
    A refusal is an answer about who may open the key: it never falls through to escrow
    recovery, which is only for an Origin that cannot answer."""


class GrantKeyCannotOpenShare(ShareRefused):
    """A grant key acts here, and Origin cannot verify a grant key: no share is released for it."""


def _acting_kind():
    """`(acting, is_system)` for the acting principal, or MasterKeyUnavailable naming why none
    can be shown to Origin."""
    from mantle.services.acting_principal import current_acting_principal

    acting = current_acting_principal()
    if acting is None:
        raise MasterKeyUnavailable("no acting principal is in scope; Origin releases a key share "
                                   "only for the acting caller's own credential")
    if acting.principal_type == "grant_key":
        raise GrantKeyCannotOpenShare(
            "a grant key is acting, and under threshold custody Origin releases a key share only "
            "on a credential it issued and can verify itself; a grant key is Mantle's own. Sign in "
            "as the person, or use a delegation Origin issued.")
    try:
        from mantle.services.peer_signing import get_system_principal_id
        system_id = get_system_principal_id()
    except Exception:  # noqa: BLE001 — no namespace: nothing acts as the system principal
        system_id = ""
    # Work run as a person who presented their own credential is theirs, even in a background
    # block (`acting_as` keeps it). A seed pass shows the operator's seed credential. Otherwise a
    # system-sourced block, or the system principal, is background work.
    if acting.source == "seed" and acting.credential is not None:
        return acting, False
    is_system = (bool(system_id and acting.principal_id == system_id)
                 or (acting.source == "system" and acting.credential is None))
    if not is_system and acting.credential is None:
        raise MasterKeyUnavailable(
            "this request carries no credential Origin issued (a person's access token or a "
            "delegation); under threshold custody Origin releases a key share only on one")
    return acting, is_system


def is_persons_own_credential(acting) -> bool:
    """Does `acting` carry a person's own credential: their access token, or a person delegation
    with no scope, not narrowed and not a persona's (review 26, L2)? Only such a credential splits
    a whole key on a call. A persona's server delegation, Ember's narrowed delegation, background
    work and a seed never do."""
    from prism.trust.use_verdict import SERVER_PREFIX

    if acting is None or acting.credential is None:
        return False
    if getattr(acting, "source", "") == "seed" or acting.principal_type != "user":
        return False
    if getattr(acting, "narrowed", None) or (getattr(acting, "scope", None) or "").strip():
        return False
    return not str(getattr(acting, "actor", None) or "").startswith(SERVER_PREFIX)


def is_root_here(db, principal_id: str) -> bool:
    """Is `principal_id` a root this store holds (a top-level collection or artifact, its own
    origin root), whose own master key keys its legacy content (review 27)?"""
    if db is None:
        return False
    try:
        doc = db.artifacts.get_artifact(str(principal_id))
    except Exception:  # noqa: BLE001 — unreadable: not known to be a root
        return False
    if not doc or str(doc.get("id", "")).startswith("master-key:"):
        return False
    return str(doc.get("origin_root") or principal_id) == str(principal_id)


def splits_on_this_call(principal_id: str, db=None) -> bool:
    """May this call split `principal_id`'s whole key (reviews 25 to 27)? On a person's own
    credential only, and then: a master key that is their own; a collection key
    (`ck:<collection>`); or the master key of a root this store holds (`is_root_here`), since no
    one acts as a collection or an artifact. Origin's release rule and first-share rule then
    decide whether that person reaches it. The platform user, the system principal and a
    persona's principal hold no root here and are split only by the pass, under a job grant."""
    from prism.trust.ledger import COLLECTION_KEY_PREFIX

    acting, is_system = _acting_kind()
    if is_system or not is_persons_own_credential(acting):
        return False
    if str(principal_id).startswith(COLLECTION_KEY_PREFIX):
        return True
    if str(acting.principal_id) == str(principal_id):
        return True
    return is_root_here(db, principal_id)


def background_job(acting) -> str:
    """The named background job the acting block runs as: the scope it declared
    (`system_acting_context(scope=...)`, `acting_as(..., scope=...)`). Empty when it named none,
    and Origin then mints nothing for it."""
    return str(getattr(acting, "scope", "") or "")


def acting_credential(background_principal: str, custodian) -> str:
    """The credential Origin is shown for the acting principal: the person's own, or, for
    background work, a delegation Origin mints for `background_principal` and the block's named
    job, under a job grant the person or a collection's Admin made (John, 2026-11-02)."""
    acting, is_system = _acting_kind()
    if not is_system:
        return acting.credential.token
    if custodian is None or not hasattr(custodian, "background_delegation"):
        raise MasterKeyUnavailable("background key use under threshold custody needs Origin to "
                                   "mint a delegation, and no Origin is configured")
    job = background_job(acting)
    if not job:
        raise MasterKeyUnavailable(
            "background work under threshold custody names its job (the acting block's scope), "
            "and this one names none: Origin mints nothing for it")
    try:
        return custodian.background_delegation(background_principal, job)
    except Exception as exc:
        raise MasterKeyUnavailable(
            f"background work {job!r} under threshold custody needs a delegation Origin mints for "
            f"{background_principal!r} under a live job grant for that job, and Origin minted "
            f"none: {exc}") from exc


def share_credential(principal_id: str, custodian=None) -> str:
    """The credential Origin is shown to release `principal_id`'s share."""
    return acting_credential(principal_id, custodian)


# -- age (pyrage) ---------------------------------------------------------------------------------

def check_recipient(recipient: str) -> str:
    import pyrage

    try:
        pyrage.x25519.Recipient.from_str(recipient.strip())
    except Exception as exc:
        raise ValueError(f"not an age X25519 recipient: {exc}") from None
    return recipient.strip()


def seal(data: bytes, recipient: str) -> bytes:
    """An age file encrypting `data` to `recipient`; `age -d -i <identity>` opens it."""
    import pyrage

    return pyrage.encrypt(data, [pyrage.x25519.Recipient.from_str(recipient)])


def open_sealed(sealed: bytes, identity: str) -> bytes:
    import pyrage

    return pyrage.decrypt(sealed, [pyrage.x25519.Identity.from_str(identity.strip())])


def recipient_from_file(path: str) -> str:
    """The first `age1…` recipient in a file: a bare recipient, or `age-keygen`'s
    `# public key: age1…` line."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            for token in line.replace(":", " ").split():
                if token.startswith("age1"):
                    return check_recipient(token)
    raise ValueError(f"{path} names no age recipient")


def identity_from_file(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip().startswith("AGE-SECRET-KEY-1"):
                return line.strip()
    raise ValueError(f"{path} holds no age identity")


# -- the escrow store -----------------------------------------------------------------------------

#: The longest a custody-lock waiter waits. Every call made while holding the lock is bounded
#: well inside it (`ESCROW_S3_TIMEOUTS`, `origin_key_share_client.CALL_TIMEOUT`), so a hung peer
#: refuses the put rather than holding the lock past it.
LOCK_TIMEOUT = 30.0

#: Connect and read timeouts (seconds) and attempts for the durable store, under the lock.
ESCROW_S3_TIMEOUTS = {"connect_timeout": 3, "read_timeout": 8, "total_max_attempts": 1}


def escrow_object_key(principal_id: str, kcv: str, escrow_id: str) -> str:
    pid = hashlib.sha256(principal_id.encode("utf-8")).hexdigest()
    return f"{ESCROW_PREFIX}{pid}.{kcv}.{escrow_id}.age"


def escrow_name(row: dict) -> str:
    """The escrow object a key row's split wrote: named by its principal, kcv and escrow id."""
    principal_id = str(row.get("id", ""))[len("master-key:"):]
    return escrow_object_key(principal_id, row.get("kcv") or "", row.get("escrow_id") or "")


def _missing(exc: Exception) -> bool:
    code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
    return code in ("NoSuchKey", "404", "NotFound")


class DurableEscrowStore:
    """Sealed escrow shares in the durable object store, one object per split, never replaced."""

    def __init__(self, client, bucket: str) -> None:
        if client is None or not bucket:
            raise ValueError("the escrow store needs the durable object store")
        self._client = client
        self._bucket = bucket

    def put_if_absent(self, name: str, sealed: bytes) -> None:
        """Write `name` only if no object has it. Called under the custody lock, which every
        writer of escrow objects takes, so the check and the write are one step."""
        try:
            self._client.head_object(Bucket=self._bucket, Key=name)
        except Exception as exc:
            if not _missing(exc):
                raise
        else:
            raise MasterKeyUnavailable(f"escrow object {name} exists; it is never overwritten")
        self._client.put_object(Bucket=self._bucket, Key=name, Body=sealed,
                                ContentType="application/octet-stream")

    def get(self, name: str) -> Optional[bytes]:
        try:
            obj = self._client.get_object(Bucket=self._bucket, Key=name)
        except Exception as exc:
            if _missing(exc):
                return None
            raise
        return obj["Body"].read()


def durable_escrow_store() -> DurableEscrowStore:
    """The escrow store over the configured durable bucket, with its own client whose timeouts
    fit inside the custody lock. Refuses when no durable store is configured."""
    from mantle.services import content_service as cs

    if not cs._durable_store_enabled():
        raise ValueError("MANTLE_MASTER_KEY_CUSTODY=threshold needs the durable object store "
                         "(CONTENT_DURABLE_BUCKET): it holds the escrow shares, apart from the "
                         "lattice")
    client = cs._make_s3_client(
        access_key_id=cs._DURABLE_ACCESS_KEY_ID, secret_access_key=cs._DURABLE_SECRET_ACCESS_KEY,
        region_name=cs._DURABLE_REGION, endpoint_url=cs._DURABLE_ENDPOINT_URL,
        **ESCROW_S3_TIMEOUTS)
    return DurableEscrowStore(client, cs._DURABLE_BUCKET)


# -- the custody lock -----------------------------------------------------------------------------

def default_lock_path() -> str:
    from mantle.db import lattice_api

    lattice = os.path.abspath(os.path.expanduser(
        os.getenv("MANTLE_LATTICE_PATH", str(lattice_api.DEFAULT_LATTICE_PATH))))
    return lattice + ".custody.lock"


class CustodyLock:
    """An exclusive lock on a file beside the lattice, taken around every write of a key row.

    The server and `mantle.system.split_master_keys` both take it, so a migration and a live `put`
    cannot interleave. An OS file lock (`fcntl.flock`, `msvcrt.locking`) is released by the OS when
    the holder dies, so a crash leaves nothing to clean up. Not reentrant."""

    def __init__(self, path: str, *, timeout: float = LOCK_TIMEOUT) -> None:
        self.path = path
        self._timeout = timeout
        self._local = threading.Lock()

    @staticmethod
    def _try(fh) -> bool:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    @staticmethod
    def _release(fh) -> None:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def __call__(self):
        deadline = time.monotonic() + self._timeout
        if not self._local.acquire(timeout=self._timeout):
            raise MasterKeyUnavailable("the master-key custody lock is held; try again")
        try:
            with open(self.path, "a+b") as fh:
                while not self._try(fh):
                    if time.monotonic() > deadline:
                        raise MasterKeyUnavailable(
                            f"the master-key custody lock ({self.path}) is held by another "
                            f"process; try again")
                    time.sleep(0.05)
                try:
                    yield
                finally:
                    self._release(fh)
        finally:
            self._local.release()


# -- shares ---------------------------------------------------------------------------------------

def key_check_value(principal_id: str, master_key: bytes) -> str:
    """A MAC of the key and its principal (128 bits, hex): equal only for the same key, and
    reveals nothing of a 256-bit random key. Checked after every combine."""
    mac = hmac.new(master_key, _KCV_INFO + principal_id.encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()[:32]


def stored_form(doc: Optional[dict]) -> Optional[tuple]:
    """The fields of a key row that say which key it holds and in what state. Two reads with the
    same form hold the same thing; store bookkeeping (`_seq`, `_origin`) is not part of it."""
    if not doc:
        return None
    return tuple(doc.get(k) for k in ("content_type", "state", "token", "share", "kcv",
                                      "escrow_id"))


def _frame_share(principal_id: str, x: int, y: bytes) -> bytes:
    pid = principal_id.encode("utf-8")
    if len(pid) > 0xFFFF:
        raise ValueError("principal id too long to frame")
    return _SHARE_MAGIC + len(pid).to_bytes(2, "big") + pid + bytes([x]) + y


def _unframe_share(principal_id: str, raw: bytes) -> tuple[int, bytes]:
    if not raw.startswith(_SHARE_MAGIC):
        raise MasterKeyUnavailable(f"the share stored for {principal_id!r} is not a framed share")
    n = int.from_bytes(raw[4:6], "big")
    bound_to = raw[6:6 + n].decode("utf-8", "replace")
    if bound_to != principal_id:
        raise MasterKeyUnavailable(
            f"the share stored for {principal_id!r} is bound to {bound_to!r}; refusing it")
    return raw[6 + n], raw[7 + n:]


def escrow_entry(principal_id: str, x: int, y: bytes, kcv: str) -> dict:
    return {"principal_id": principal_id, "x": x, "y": base64.b64encode(y).decode(), "kcv": kcv}


def split_key(principal_id: str, master_key: bytes, key_provider, escrow_recipient: str, *,
              state: str = "active",
              origin_y: Optional[bytes] = None) -> tuple[dict, dict, dict, bytes]:
    """Split one key. Returns `(row, origin_share, escrow_entry, sealed_escrow)`: the row holds
    share 1 wrapped, `origin_share` is share 2 as Origin stores it, `escrow_entry` is share 3 in
    the clear (for the migration's sealed export) and `sealed_escrow` is share 3 sealed to the
    escrow recipient. Checks every pair recombines before returning."""
    from prism.trust import shamir

    if origin_y is None:
        shares = dict(shamir.split(master_key, k=K, n=N))
    else:
        # The split Origin's share already belongs to (review 27, M2): with k=2 the key and that
        # share fix the line, so this node's share and the escrow share are its other points.
        shares = {ORIGIN_X: bytes(origin_y), **dict(shamir.shares_through(
            master_key, (ORIGIN_X, bytes(origin_y)), [MANTLE_X, ESCROW_X]))}
    for a, b in ((MANTLE_X, ORIGIN_X), (MANTLE_X, ESCROW_X), (ORIGIN_X, ESCROW_X)):
        if shamir.combine([(a, shares[a]), (b, shares[b])]) != master_key:
            raise RuntimeError("a split did not recombine; nothing written")
    kcv = key_check_value(principal_id, master_key)
    escrow = escrow_entry(principal_id, ESCROW_X, shares[ESCROW_X], kcv)
    sealed = seal(json.dumps(escrow, sort_keys=True).encode(), escrow_recipient)
    row = {
        "id": LatticeMasterKeyStore._id(principal_id),
        "content_type": SHARE_CONTENT_TYPE,
        "custody": CUSTODY_TAG,
        "state": state,
        "share": key_provider.wrap(_frame_share(principal_id, MANTLE_X, shares[MANTLE_X])),
        "kcv": kcv,
        "escrow_id": os.urandom(8).hex(),
    }
    origin_share = {"x": ORIGIN_X, "y": base64.b64encode(shares[ORIGIN_X]).decode(), "kcv": kcv}
    return row, origin_share, escrow, sealed


class ThresholdMasterKeyStore:
    """The master key store under threshold custody. See the module note."""

    def __init__(self, key_provider, db_factory: Callable[[], object], custodian: ShareCustodian,
                 escrow_recipient: str, escrow_store: EscrowStore, lock: CustodyLock, *,
                 recovery_identity: Optional[str] = None) -> None:
        self._recipient = check_recipient(escrow_recipient)    # a mistyped one fails at boot
        self._kek = key_provider
        self._db_factory = db_factory
        self._custodian = custodian
        self._escrow = escrow_store
        self._lock = lock
        self._recovery_identity = recovery_identity

    # -- reading ---------------------------------------------------------------------------

    def _row(self, principal_id: str) -> Optional[dict]:
        try:
            return self._db_factory().artifacts.get_artifact(LatticeMasterKeyStore._id(principal_id))
        except Exception as exc:
            raise MasterKeyUnavailable(
                f"could not read the key share for {principal_id!r}; refusing to continue, "
                f"because generating a replacement would overwrite it: {exc}") from exc

    #: A requester's verdict on a key, from Origin, is reused at most this long (architecture
    #: §51: the verdict memo bounds post-revocation validity).
    VERDICT_TTL_S = 30.0

    def verdict(self, principal_id: str, *, collection_id: Optional[str] = None,
                artifact_id: Optional[str] = None) -> None:
        """Before a cached key is used for this requester: Origin's decision for them, memoized
        per requester for at most `VERDICT_TTL_S`. A refusal raises `ShareRefused`."""
        import time as _time

        scope = {"collection_id": collection_id or None,
                 "artifact_id": self._root_of(artifact_id)}
        held = self.__dict__.setdefault("_verdicts", {}).get(self._verdict_key(principal_id,
                                                                               scope))
        if held is not None and held > _time.monotonic():
            return
        self._origin_share(principal_id, scope)        # Origin decides; a refusal raises

    def _verdict_key(self, principal_id: str, scope: dict) -> tuple:
        from mantle.services.acting_principal import current_acting_principal

        import hashlib

        actor = current_acting_principal()
        who = (actor.principal_id, actor.principal_type) if actor else ("", "")
        # Per credential, not per person (review 12, M7): a narrowed delegation, or a new token
        # after a sign-out, never reuses a verdict another credential earned.
        cred = getattr(actor, "credential", None) if actor else None
        token = getattr(cred, "token", None) or ""
        tag = hashlib.sha256(token.encode()).hexdigest()[:16] if token else ""
        return (who, tag, principal_id, (scope or {}).get("collection_id"),
                (scope or {}).get("artifact_id"))

    def forget_verdicts_of(self, person: Optional[str]) -> None:
        """Forget the key-release verdicts held for `person` (None: everyone's)."""
        held = self.__dict__.get("_verdicts") or {}
        for key in [k for k in held if person is None or k[0][0] == person]:
            held.pop(key, None)

    def _remember_verdict(self, principal_id: str, scope: dict) -> None:
        import time as _time

        self.__dict__.setdefault("_verdicts", {})[self._verdict_key(principal_id, scope)] = (
            _time.monotonic() + self.VERDICT_TTL_S)

    def check_caller(self) -> None:
        """Raise unless the acting caller has a credential Origin would be shown. The oracle calls
        this before it serves a cached key, so a caller that could not open a key cold (a grant
        key, a request with no Origin credential) is not served one warm."""
        _acting_kind()

    def _root_of(self, artifact_id: Optional[str]) -> Optional[str]:
        """The root id Origin's replica knows an artifact by (a version id names its root)."""
        if not artifact_id:
            return None
        try:
            doc = self._db_factory().artifacts.get_artifact(artifact_id)
        except Exception:  # noqa: BLE001 — unreadable: name it as given
            doc = None
        return str((doc or {}).get("root_id") or artifact_id)

    def _origin_share(self, principal_id: str, scope: Optional[dict] = None) -> Optional[dict]:
        from mantle.clients.origin_key_share_client import AuthorityRefused

        scope = scope or {}
        try:
            credential = share_credential(principal_id, self._custodian)
        except ShareRefused:
            raise
        except MasterKeyUnavailable as exc:
            raise ShareRefused(str(exc)) from exc
        try:
            share = self._custodian.get_share(principal_id, credential=credential, **scope)
            self._remember_verdict(principal_id, scope)      # Origin decided for this requester
            return share
        except AuthorityRefused as exc:
            raise ShareRefused(
                f"Origin refused to release the key share of {principal_id!r}: {exc}") from exc
        except Exception as exc:
            raise MasterKeyUnavailable(
                f"Origin could not answer for the key share of {principal_id!r}: {exc}") from exc

    def _combine(self, principal_id: str, mine: tuple[int, bytes], other: tuple[int, bytes],
                 kcv: str) -> Optional[bytes]:
        from prism.trust import shamir

        try:
            key = shamir.combine([mine, other])
        except ValueError:
            return None
        return key if hmac.compare_digest(key_check_value(principal_id, key), kcv) else None

    def get(self, principal_id: str) -> bytes | None:
        return self.get_scoped(principal_id)

    def get_scoped(self, principal_id: str, *, collection_id: Optional[str] = None,
                   artifact_id: Optional[str] = None) -> bytes | None:
        """`get`, naming the collection (and artifact) the key is wanted for, which Origin's
        reach check is asked about. The oracle calls this."""
        self.check_caller()           # a caller Origin would refuse is refused here, by name
        scope = {"collection_id": collection_id or None,
                 "artifact_id": self._root_of(artifact_id)}
        doc = self._row(principal_id)
        ct = (doc or {}).get("content_type")
        if ct == LatticeMasterKeyStore.CONTENT_TYPE and (doc or {}).get("token"):
            # A key the cutover's split pass did not reach is split on the first call that may
            # split it (`splits_on_this_call`): a master key on its owner's own credential, a
            # collection key on any person's own credential that Origin admits for it (reviews 25
            # and 26, H1). Anyone else is refused.
            if splits_on_this_call(principal_id, self._db_factory()):
                return self._owner_called(principal_id,
                                          self._split_whole(principal_id, doc, scope))
            raise MasterKeyUnavailable(
                f"the key for {principal_id!r} is stored whole under the KEK, not split; it is "
                f"split on the next call of a person's own credential that may open it, or by "
                f"`python -m mantle.system.split_master_keys`. Refusing to mint over it.")
        if ct != SHARE_CONTENT_TYPE:
            if self._origin_share(principal_id, scope) is not None:
                raise MasterKeyUnavailable(
                    f"Origin holds a key share for {principal_id!r} and this node holds none; "
                    f"refusing to mint a replacement over a key that exists")
            return None                               # nothing anywhere: first use
        if doc.get("state") == "pending":
            return None                               # a put that never completed; never served
        kcv = doc.get("kcv") or ""
        try:
            mine = _unframe_share(principal_id, self._kek.unwrap(doc["share"]))
        except MasterKeyUnavailable:
            raise
        except Exception as exc:
            raise MasterKeyUnavailable(
                f"a key share EXISTS for {principal_id!r} but could not be unwrapped "
                f"(wrong KEK / rotated provider?); refusing to overwrite it: {exc}") from exc

        # Offline recovery combines this node's share with the escrow share and asks Origin
        # nothing. John's decision on review 33's F1: recovery exists for when Origin is down,
        # refuses, has lost its shares, or no longer releases a key (an activated charged root),
        # and asking Origin first made it run in none of those. The escrow identity is loaded only
        # offline (`threshold_store_from_env(offline=True)`, which a running node refuses), so a
        # taken-over Mantle never holds it; on a running node a refusal from Origin still never
        # falls through to escrow.
        if self._recovery_identity:
            key = self._from_escrow(principal_id, doc, mine, kcv)
            logger.warning("master key for %s recovered offline from the escrow share",
                           principal_id)
            return key

        why = ""
        try:
            theirs = self._origin_share(principal_id, scope)
        except ShareRefused:
            raise                                     # an answer, not an outage: no recovery
        except MasterKeyUnavailable as exc:
            theirs, why = None, str(exc)
        if theirs is not None:
            key = self._combine(principal_id, mine,
                                (int(theirs["x"]), base64.b64decode(theirs["y"])), kcv)
            if key is not None:
                return self._owner_called(principal_id, key)
            why = "Origin's share does not recombine with this node's to the stored key check"
        elif not why:
            why = "Origin holds no share for it"

        raise MasterKeyUnavailable(
            f"the master key for {principal_id!r} cannot be combined: {why}. Refusing to mint a "
            f"replacement. Recovery combines this node's share with the escrow share "
            f"(MANTLE_MASTER_KEY_RECOVERY=escrow, MANTLE_ESCROW_IDENTITY_FILE).")

    def _owner_called(self, principal_id: str, key: bytes) -> bytes:
        """A person's own credential opened their own master key: re-key their content still
        under it, in the background (`services/owner_rekey.py`, review 26, open item a). A
        collection key, a non-person principal and anyone else's credential start nothing.
        Never fails the call."""
        from prism.trust.ledger import COLLECTION_KEY_PREFIX

        try:
            if (not str(principal_id).startswith(COLLECTION_KEY_PREFIX)
                    and splits_on_this_call(principal_id, self._db_factory())):
                from mantle.services import owner_rekey
                acting, _ = _acting_kind()
                owner_rekey.schedule(self._db_factory, principal_id, acting.credential.token)
        except Exception:  # noqa: BLE001
            logger.warning("owner re-key not scheduled for %s", principal_id, exc_info=True)
        return key

    def _ask_origin(self, what: str, principal_id: str, call):
        """One call to Origin during a split, its refusal a `ShareRefused` and any other failure a
        `MasterKeyUnavailable` (review 26, L2): never a raw client error."""
        from mantle.clients.origin_key_share_client import AuthorityRefused

        try:
            return call()
        except (ShareRefused, AuthorityRefused) as exc:
            raise ShareRefused(f"Origin refused {what} for {principal_id!r}; the key stays "
                               f"whole: {exc}") from exc
        except MasterKeyUnavailable:
            raise
        except Exception as exc:
            raise MasterKeyUnavailable(f"Origin could not answer {what} for {principal_id!r}; "
                                       f"the key stays whole: {exc}") from exc

    def _split_whole(self, principal_id: str, doc: dict, scope: Optional[dict] = None) -> bytes:
        """Split a whole key in place on the acting person's own credential (reviews 25 and 26).

        In order, each step only once the one before it has answered:
          1. Origin's release rule is asked for this credential and scope (a refusal writes
             nothing);
          2. under the custody lock, the row is re-read and split, and Origin's share stored on
             that credential (Origin's first-share rule admits the key's principal, a person who
             may open it, or one holding Admin on what it keys);
          3. only then the escrow share written;
          4. Origin's share read back and recombined with this node's to the key (L1);
          5. only then the whole row replaced by this node's share, Origin told the key is
             active, and the store file settled so the replaced row leaves no bytes behind (H2).
        """
        import hmac as _hmac

        from prism.trust import shamir

        scope = scope or {}
        acting, _ = _acting_kind()
        credential = acting.credential.token
        held = self._ask_origin("the release", principal_id, lambda: self._custodian.get_share(
            principal_id, credential=credential, **scope))
        with self._lock():
            current = self._row(principal_id)
            if stored_form(current) != stored_form(doc):
                raise MasterKeyUnavailable(
                    f"the key row for {principal_id!r} changed while it was being split")
            key = LatticeMasterKeyStore(self._kek, self._db_factory).get(principal_id)
            if key is None:
                raise MasterKeyUnavailable(f"the whole key for {principal_id!r} did not open")
            origin_y = None
            if held is not None:
                if held.get("kcv") != key_check_value(principal_id, key) \
                        or int(held.get("x", 0)) != ORIGIN_X:
                    raise MasterKeyUnavailable(
                        f"Origin holds a share of another key for {principal_id!r}; the key "
                        f"stays whole")
                # An activated share (a key written back whole by `--combine`, its Origin share
                # kept): complete that share's line, since no one may store another over it. A
                # share still pending is replaced by a fresh split (review 28, L4), so a line
                # planted by an earlier call does not survive: its storer, the key's principal or
                # an Admin may replace it, and anyone else is refused until one of them calls.
                if not held.get("pending"):
                    origin_y = base64.b64decode(held["y"])
            row, origin_share, escrow, sealed = split_key(principal_id, key, self._kek,
                                                          self._recipient, origin_y=origin_y)
            x, y = _unframe_share(principal_id, self._kek.unwrap(row["share"]))
            if x != MANTLE_X or not _hmac.compare_digest(shamir.combine(
                    [(x, y), (escrow["x"], base64.b64decode(escrow["y"]))]), key):
                raise MasterKeyUnavailable(f"the split of {principal_id!r} does not recombine")
            self._ask_origin("to store its share", principal_id, lambda: self._custodian.put_share(
                principal_id, origin_share, credential=credential))
            try:
                self._escrow.put_if_absent(escrow_name(row), sealed)
            except Exception as exc:
                raise MasterKeyUnavailable(
                    f"the escrow share for {principal_id!r} could not be written; the key stays "
                    f"whole: {exc}") from exc
            theirs = self._ask_origin("its share back", principal_id, lambda: (
                self._custodian.get_share(principal_id, credential=credential, **scope)))
            if theirs is None or self._combine(
                    principal_id, (x, y), (int(theirs["x"]), base64.b64decode(theirs["y"])),
                    row["kcv"]) != key:
                raise MasterKeyUnavailable(
                    f"Origin's stored share of {principal_id!r} does not recombine with this "
                    f"node's to the key; the key stays whole")
            db = self._db_factory()
            db.artifacts.put_artifact(row)
        activate = getattr(self._custodian, "activate_share", None)

        def landed() -> None:
            # Only once the row replacing the whole key has committed (review 27, L3): a
            # caller's transaction that rolls back leaves the key whole, and Origin's share
            # replaceable, as it found them.
            if activate is not None:
                try:
                    activate(principal_id, credential=credential)
                except Exception:  # noqa: BLE001 — it stays replaceable until told; logged
                    logger.warning("Origin was not told %s is active", principal_id,
                                   exc_info=True)
            from mantle.db import scrub
            scrub.settle_replaced(db)
            logger.warning("key %s split on the acting person's credential", principal_id)

        lattice = getattr(getattr(db, "artifacts", None), "db", None)
        if lattice is not None and hasattr(lattice, "after_commit"):
            lattice.after_commit(landed)
        else:
            landed()
        return key

    def _from_escrow(self, principal_id: str, doc: dict, mine: tuple[int, bytes],
                     kcv: str) -> bytes:
        try:
            sealed = self._escrow.get(escrow_name(doc))
            if sealed is None:
                raise MasterKeyUnavailable(f"no escrow share is stored for {principal_id!r}")
            entry = json.loads(open_sealed(sealed, self._recovery_identity))
        except MasterKeyUnavailable:
            raise
        except Exception as exc:
            raise MasterKeyUnavailable(
                f"the escrow share for {principal_id!r} could not be opened: {exc}") from exc
        if entry.get("principal_id") != principal_id:
            raise MasterKeyUnavailable(f"the escrow share stored for {principal_id!r} names "
                                       f"{entry.get('principal_id')!r}")
        key = self._combine(principal_id, mine, (int(entry["x"]), base64.b64decode(entry["y"])),
                            kcv)
        if key is None:
            raise MasterKeyUnavailable(
                f"the escrow share for {principal_id!r} does not recombine to the stored key check")
        return key

    # -- writing ---------------------------------------------------------------------------

    def put(self, principal_id: str, master_key: bytes) -> None:
        with self._lock():
            self._put_locked(principal_id, master_key)

    def _put_locked(self, principal_id: str, master_key: bytes) -> None:
        # Origin stores a first share only on the credential of the key's principal or of whoever
        # brought it into being, and replaces one only on a credential that reaches the key: none
        # here refuses before anything is written.
        credential = share_credential(principal_id, self._custodian)
        doc = self._row(principal_id)
        ct = (doc or {}).get("content_type")
        kcv = key_check_value(principal_id, master_key)
        supersedes = None
        if ct == SHARE_CONTENT_TYPE:
            if doc.get("state") == "pending":
                supersedes = doc.get("kcv")
            elif doc.get("kcv") != kcv:
                raise MasterKeyUnavailable(
                    f"a split master key is stored for {principal_id!r}; refusing to overwrite "
                    f"it with a different key")
        elif ct == LatticeMasterKeyStore.CONTENT_TYPE and (doc or {}).get("token"):
            raise MasterKeyUnavailable(
                f"the master key for {principal_id!r} is stored whole; split it with "
                f"`mantle.system.split_master_keys` rather than writing over it")

        row, origin_share, _entry, sealed = split_key(principal_id, master_key, self._kek,
                                                      self._recipient, state="pending")
        artifacts = self._db_factory().artifacts
        artifacts.put_artifact(row)
        try:
            self._escrow.put_if_absent(escrow_name(row), sealed)
        except Exception as exc:
            raise MasterKeyUnavailable(
                f"the escrow share for {principal_id!r} could not be written to the durable "
                f"store; refusing the key: {exc}") from exc
        self._custodian.put_share(principal_id, origin_share, supersedes=supersedes,
                                  credential=credential)
        if stored_form(self._row(principal_id)) != stored_form(row):
            raise MasterKeyUnavailable(
                f"the key row for {principal_id!r} changed while it was being written; "
                f"not activating it")
        artifacts.put_artifact({**row, "state": "active"})
        # Origin's share stops being replaceable once this key is active.
        activate = getattr(self._custodian, "activate_share", None)
        if activate is not None:
            try:
                activate(principal_id, credential=credential)
            except Exception:  # noqa: BLE001 — it stays replaceable until told; logged
                logger.warning("Origin was not told %s is active", principal_id, exc_info=True)


def build_threshold_store(key_provider, db_factory: Callable[[], object], *,
                          offline: bool = False) -> ThresholdMasterKeyStore:
    """The store `MANTLE_MASTER_KEY_CUSTODY=threshold` selects, from its settings.

    ``MANTLE_ESCROW_RECIPIENT_FILE`` names the age public key (required), and the durable object
    store (``CONTENT_DURABLE_BUCKET``) must be configured, since it holds the escrow shares.
    Recovery mode needs both ``MANTLE_MASTER_KEY_RECOVERY=escrow`` and
    ``MANTLE_ESCROW_IDENTITY_FILE``; either alone is refused, so an identity file left configured
    does not quietly turn recovery on.
    """
    from mantle.clients.origin_key_share_client import OriginShareCustodian

    recipient_file = (os.getenv("MANTLE_ESCROW_RECIPIENT_FILE") or "").strip()
    if not recipient_file:
        raise ValueError("MANTLE_MASTER_KEY_CUSTODY=threshold needs MANTLE_ESCROW_RECIPIENT_FILE")
    escrow_store = durable_escrow_store()
    recovery = (os.getenv("MANTLE_MASTER_KEY_RECOVERY") or "").strip().lower()
    identity_file = (os.getenv("MANTLE_ESCROW_IDENTITY_FILE") or "").strip()
    # The escrow identity opens every escrow share, so it never sits on a running node: the
    # server refuses to start with it configured, recovery or not (review 7). Recovery runs
    # offline, as `split_master_keys --combine` (`offline=True`).
    if identity_file and not offline:
        raise ValueError("MANTLE_ESCROW_IDENTITY_FILE is set on a running node under threshold "
                         "custody; the escrow identity stays offline. Unset it, and run recovery "
                         "offline: `python -m mantle.system.split_master_keys --combine` with "
                         "MANTLE_MASTER_KEY_RECOVERY=escrow")
    if recovery not in ("", "escrow"):
        raise ValueError(f"Unknown MANTLE_MASTER_KEY_RECOVERY: {recovery!r} (expected escrow)")
    if bool(recovery) != bool(identity_file):
        raise ValueError("recovery needs both MANTLE_MASTER_KEY_RECOVERY=escrow and "
                         "MANTLE_ESCROW_IDENTITY_FILE")
    identity = identity_from_file(identity_file) if recovery else None
    if identity:
        logger.warning("master key custody is in RECOVERY mode: keys this node cannot combine "
                       "with Origin's share are combined with the escrow share")
    return ThresholdMasterKeyStore(
        key_provider, db_factory, OriginShareCustodian(), recipient_from_file(recipient_file),
        escrow_store, CustodyLock(default_lock_path()), recovery_identity=identity)


def custody_kind() -> str:
    kind = (os.getenv("MANTLE_MASTER_KEY_CUSTODY", "lattice") or "lattice").strip().lower()
    if kind not in ("lattice", "threshold"):
        raise ValueError(f"Unknown MANTLE_MASTER_KEY_CUSTODY: {kind!r} (expected lattice | threshold)")
    return kind
