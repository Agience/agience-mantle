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
              ever overwritten, so an earlier split's share 3 survives a later one. The identity is kept offline, in ``ikailo_infra`` and on the
              NAS; this node cannot open it. ``mantle.system.split_master_keys --export-escrow``
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

What each node yields alone. Mantle's store (a disk, a backup) holds share 1 under the KEK: one
share, which reveals nothing of the key. The durable store holds share 3 sealed to an offline key.
Origin holds share 2. The running Mantle process can still obtain share 2, because Origin releases
it to Mantle's service identity, so a compromise of Mantle together with its signing key
(``mantle.private.pem``) and a route to Origin reaches every key; each token it signs for Origin
opens one share, once, within 60 seconds. Origin checks no grant before it releases a share; that
is the next step (Origin running the grant check against a signed ledger).
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
    def get_share(self, principal_id: str) -> Optional[dict]: ...

    def put_share(self, principal_id: str, share: dict, *,
                  supersedes: Optional[str] = None) -> str: ...


class EscrowStore(Protocol):
    def put_if_absent(self, name: str, sealed: bytes) -> None: ...

    def get(self, name: str) -> Optional[bytes]: ...


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
              state: str = "active") -> tuple[dict, dict, dict, bytes]:
    """Split one key. Returns `(row, origin_share, escrow_entry, sealed_escrow)`: the row holds
    share 1 wrapped, `origin_share` is share 2 as Origin stores it, `escrow_entry` is share 3 in
    the clear (for the migration's sealed export) and `sealed_escrow` is share 3 sealed to the
    escrow recipient. Checks every pair recombines before returning."""
    from prism.trust import shamir

    shares = dict(shamir.split(master_key, k=K, n=N))
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

    def _origin_share(self, principal_id: str) -> Optional[dict]:
        try:
            return self._custodian.get_share(principal_id)
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
        doc = self._row(principal_id)
        ct = (doc or {}).get("content_type")
        if ct == LatticeMasterKeyStore.CONTENT_TYPE and (doc or {}).get("token"):
            raise MasterKeyUnavailable(
                f"the master key for {principal_id!r} is stored whole under the KEK, not split; "
                f"run `python -m mantle.system.split_master_keys`, or select "
                f"MANTLE_MASTER_KEY_CUSTODY=lattice. Refusing to mint over it.")
        if ct != SHARE_CONTENT_TYPE:
            if self._origin_share(principal_id) is not None:
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

        why = ""
        try:
            theirs = self._origin_share(principal_id)
        except MasterKeyUnavailable as exc:
            theirs, why = None, str(exc)
        if theirs is not None:
            key = self._combine(principal_id, mine,
                                (int(theirs["x"]), base64.b64decode(theirs["y"])), kcv)
            if key is not None:
                return key
            why = "Origin's share does not recombine with this node's to the stored key check"
        elif not why:
            why = "Origin holds no share for it"

        if self._recovery_identity:
            key = self._from_escrow(principal_id, doc, mine, kcv)
            logger.warning("master key for %s recovered from the escrow share (%s)",
                           principal_id, why)
            return key
        raise MasterKeyUnavailable(
            f"the master key for {principal_id!r} cannot be combined: {why}. Refusing to mint a "
            f"replacement. Recovery combines this node's share with the escrow share "
            f"(MANTLE_MASTER_KEY_RECOVERY=escrow, MANTLE_ESCROW_IDENTITY_FILE).")

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
        self._custodian.put_share(principal_id, origin_share, supersedes=supersedes)
        if stored_form(self._row(principal_id)) != stored_form(row):
            raise MasterKeyUnavailable(
                f"the key row for {principal_id!r} changed while it was being written; "
                f"not activating it")
        artifacts.put_artifact({**row, "state": "active"})


def build_threshold_store(key_provider, db_factory: Callable[[], object]) -> ThresholdMasterKeyStore:
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
