"""S3-backed :class:`PostingStore` adapter.

Production wiring of the in-memory store from :mod:`posting`. Encrypted blobs persist in S3
(or any S3-compatible store like MinIO) under::

    {prefix}/{principal_id}/sse/posting/{blind_token}.enc
    {prefix}/{principal_id}/sse/manifests/{artifact_id}.enc

A third object used to sit beside them — ``{prefix}/{principal_id}/sse/stats.enc``, the
per-owner BM25 corpus aggregates — written and read by a stats store this module also carried.
Recall computes no corpus statistic, so nothing writes or reads it and the adapter is gone.
The objects an existing bucket already holds are inert: nothing opens them, and nothing here
deletes them either, because deleting an operator's data is an operator's decision.

The adapter is deliberately independent of :mod:`services.content_service`
so the MANTLE-SSE package can be wired against any boto3-compatible
client. Mirror-image of :class:`S3CellStore` for the same reasons.

Wire format on disk: the raw bytes returned by :func:`pack_posting` /
:func:`pack_manifest` (`nonce ‖ ciphertext ‖ tag`).
GCM authentication happens inside the SSE reader; this
adapter is a thin dictionary over S3.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _is_not_found(exc: Exception) -> bool:
    """Detect S3 NoSuchKey / 404 across boto3 + minio variants."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code")
        if code in {"NoSuchKey", "404", "NotFound"}:
            return True
    return type(exc).__name__ in {"NoSuchKey", "404"}


def _join_key(*parts: str) -> str:
    return "/".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# S3PostingStore
# ---------------------------------------------------------------------------


class S3PostingStore:
    """:class:`PostingStore` Protocol implementation backed by an S3 bucket.

    Layout::

        {prefix}/{principal_id}/sse/posting/{blind_token}.enc
        {prefix}/{principal_id}/sse/manifests/{artifact_id}.enc

    Args:
        s3_client: A boto3 S3 client (or compatible). Caller manages
            credentials and endpoint URL.
        bucket: Bucket name. Must already exist; this class does not
            create it.
        prefix: Optional key prefix. Defaults to ``"mantle-sse"``. Empty
            string disables the prefix (blobs live at bucket root).
    """

    def __init__(
        self,
        s3_client: object,
        bucket: str,
        prefix: str = "mantle-sse",
    ) -> None:
        if not bucket:
            raise ValueError("S3PostingStore: bucket name is required")
        self._s3 = s3_client
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _posting_key(self, principal_id: str, blind_token: str) -> str:
        return _join_key(
            self._prefix, principal_id, "sse", "posting", f"{blind_token}.enc",
        )

    def _manifest_key(self, principal_id: str, artifact_id: str) -> str:
        return _join_key(
            self._prefix, principal_id, "sse", "manifests", f"{artifact_id}.enc",
        )

    def _entries_key(self, principal_id: str, blind_token: str) -> str:
        """The entry envelope's own object key.

        A different suffix from the legacy blob. Both describe the same slot, and sharing
        `{blind_token}.enc` would have a conversion overwrite the legacy blob with a JSON envelope,
        which `get_posting` then hands to `narrowing` as ciphertext to decrypt. Separate keys let a
        slot hold both during the conversion and let the legacy read stay exactly
        as truthful as it was.
        """
        return _join_key(
            self._prefix, principal_id, "sse", "posting", f"{blind_token}.entries",
        )

    def _posting_owner_prefix(self, principal_id: str) -> str:
        return _join_key(self._prefix, principal_id, "sse", "posting") + "/"

    # ------------------------------------------------------------------
    # PostingStore Protocol — entries
    # ------------------------------------------------------------------
    #
    # One object per slot rather than per entry, which is why these operations are entry-level in
    # the protocol rather than "one object per entry".
    #
    # Per-entry rows are what `SqlitePostingStore` wants: an add is one upsert against a primary key.
    # Per-entry OBJECTS would be a disaster here. One object per (owner × term × artifact) is the
    # object explosion the local store was just rescued from, and worse, every probe would become a
    # LIST instead of a GET — a paginated round trip per term per owner, against a read path already
    # measured at 4,520 probes for a ten-term query over 194 owners.
    #
    # So this adapter keeps one object per token holding a JSON envelope of sealed entries, and does
    # the read-modify-write itself. That is the SAME O(entries) cost the old caller-side shape had —
    # honestly so — but here it is one round trip either way and the network dominates it, whereas on
    # local disk it was the whole cost. The protocol states the operation; the store states the
    # layout.
    #
    # The envelope is a JSON object of ``{entry_key: base64(sealed_entry)}`` rather than a
    # `pack_posting` blob, because these bytes are already sealed per entry and cannot be re-sealed
    # under a key this adapter does not have. It is a container rather than a cipher.

    def _slot_envelope(self, principal_id: str, blind_token: str) -> dict:
        raw = self._get(self._entries_key(principal_id, blind_token))
        if raw is None:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Not an entry envelope. Almost certainly a LEGACY `pack_posting` blob, which is
            # ciphertext and therefore not UTF-8 JSON — `get_posting` is what reads those, and
            # `narrowing`/`indexer` handle the conversion where the keys are. Returning empty here
            # keeps the entry path from mistaking one for a corrupt envelope and deleting it.
            return {}
        return payload if isinstance(payload, dict) else {}

    def _put_slot_envelope(self, principal_id: str, blind_token: str, payload: dict) -> None:
        key = self._entries_key(principal_id, blind_token)
        if not payload:
            self._delete(key)
            return
        self._put(key, json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    @staticmethod
    def _entry_key(artifact_id: str, collection_id: str) -> str:
        # Length-prefixed so no two (artifact, collection) pairs can produce one key — the same rule
        # `posting.entry_aad` states, and it has to hold here too or two entries would share a slot.
        return "%d:%s:%s" % (len(artifact_id.encode("utf-8")), artifact_id, collection_id or "")

    def add_entry(self, principal_id: str, blind_token: str, artifact_id: str,
                  collection_id: str, blob: bytes) -> None:
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("S3PostingStore.add_entry expects bytes")
        payload = self._slot_envelope(principal_id, blind_token)
        payload[self._entry_key(artifact_id, collection_id)] = base64.b64encode(
            bytes(blob)).decode("ascii")
        self._put_slot_envelope(principal_id, blind_token, payload)

    def get_entries(self, principal_id: str, blind_token: str) -> List[Tuple[str, str, bytes]]:
        """Every entry with its identity, recovered from the envelope key.

        The key is ``len(artifact):artifact:collection``, so the split is on the FIRST colon for
        the length and then that many BYTES for the artifact — never ``split(":")``, which an id
        containing a colon would break.
        """
        out: List[Tuple[str, str, bytes]] = []
        for key, encoded in self._slot_envelope(principal_id, blind_token).items():
            try:
                head, rest = key.split(":", 1)
                width = int(head)
                raw = rest.encode("utf-8")
                artifact_id = raw[:width].decode("utf-8")
                collection_id = raw[width + 1:].decode("utf-8")
                out.append((artifact_id, collection_id, base64.b64decode(encoded)))
            except Exception:                      # noqa: BLE001
                # One unreadable entry must not cost the slot. The reader gives an unopenable
                # entry the same treatment, so dropping it here keeps the two agreeing.
                logger.warning("S3PostingStore: undecodable entry %r in %s/%s",
                               key, principal_id, blind_token[:8])
        return out

    def delete_entries_for_artifact(self, principal_id: str, blind_token: str,
                                    artifact_id: str) -> int:
        payload = self._slot_envelope(principal_id, blind_token)
        prefix = "%d:%s:" % (len(artifact_id.encode("utf-8")), artifact_id)
        doomed = [k for k in payload if k.startswith(prefix)]
        for k in doomed:
            del payload[k]
        if doomed:
            self._put_slot_envelope(principal_id, blind_token, payload)
        return len(doomed)

    def delete_entry(self, principal_id: str, blind_token: str, artifact_id: str,
                     collection_id: str) -> bool:
        payload = self._slot_envelope(principal_id, blind_token)
        if payload.pop(self._entry_key(artifact_id, collection_id), None) is None:
            return False
        self._put_slot_envelope(principal_id, blind_token, payload)
        return True

    # ------------------------------------------------------------------
    # PostingStore Protocol — the legacy whole-slot blob, read-only
    # ------------------------------------------------------------------

    def get_posting(self, principal_id: str, blind_token: str) -> Optional[bytes]:
        return self._get(self._posting_key(principal_id, blind_token))

    def put_posting(self, principal_id: str, blind_token: str, blob: bytes) -> None:
        self._put(self._posting_key(principal_id, blind_token), blob)

    def delete_posting(self, principal_id: str, blind_token: str) -> None:
        self._delete(self._posting_key(principal_id, blind_token))

    def list_tokens_for_owner(self, principal_id: str) -> List[str]:
        """Both layouts — see the Protocol. A lister that saw only one suffix would make a re-key
        pass silently skip every slot in the other."""
        prefix = self._posting_owner_prefix(principal_id)
        out: set = set()
        for key in self._list_under(prefix):
            base = key[len(prefix):]
            for suffix in (".enc", ".entries"):
                if base.endswith(suffix):
                    out.add(base[: -len(suffix)])
                    break
        return sorted(out)

    def list_owners(self) -> List[str]:
        """Every principal with an index tree under this prefix.

        Read by listing keys and cutting at the ``/sse/`` segment every layout in this store puts
        between the principal and its blobs, rather than by asking S3 for common prefixes with a
        delimiter: `_list_under` is the one listing path here, it already paginates, and a fake
        client in a test only has to support the call it already supports.

        Principal ids appear RAW in these keys — `_posting_key` does not escape them, unlike the
        file store's directory names — so there is nothing to decode on the way back out.
        """
        base = (self._prefix + "/") if self._prefix else ""
        out: set[str] = set()
        for key in self._list_under(base):
            rest = key[len(base):]
            cut = rest.find("/sse/")
            if cut > 0:
                out.add(rest[:cut])
        return sorted(out)

    # ------------------------------------------------------------------
    # PostingStore Protocol — manifests
    # ------------------------------------------------------------------

    def get_manifest(self, principal_id: str, artifact_id: str) -> Optional[bytes]:
        return self._get(self._manifest_key(principal_id, artifact_id))

    def put_manifest(self, principal_id: str, artifact_id: str, blob: bytes) -> None:
        self._put(self._manifest_key(principal_id, artifact_id), blob)

    def delete_manifest(self, principal_id: str, artifact_id: str) -> None:
        self._delete(self._manifest_key(principal_id, artifact_id))

    # ------------------------------------------------------------------
    # Shared S3 helpers
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Which analysis wrote this index
    # ------------------------------------------------------------------
    #
    # One small object beside the index rather than a per-principal one: the generation is a fact
    # about the software that wrote the bucket, not about any owner's content, and it is cleartext
    # for the same reason — a version number reveals nothing about what is stored.

    def _analyzer_key(self) -> str:
        return _join_key(self._prefix, "index", "analyzer")

    def analyzer_generation(self):
        """The stamp, or ``None`` for a bucket written before stamping existed — and for any
        error reaching it, because a diagnostic must not be what takes an index offline."""
        raw = self._get(self._analyzer_key())
        if raw is None:
            return None
        try:
            return int(raw.decode("utf-8").strip())
        except (UnicodeDecodeError, ValueError):
            return None

    def record_analyzer_generation(self, generation: int) -> None:
        """Stamp the writing generation. Idempotent; last writer wins."""
        self._put(self._analyzer_key(), str(int(generation)).encode("utf-8"))

    def _get(self, key: str) -> Optional[bytes]:
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            logger.warning("S3PostingStore get failed for %s: %s", key, exc)
            raise
        body = resp.get("Body")
        if body is None:
            return None
        try:
            return body.read()
        finally:
            try:
                body.close()
            except Exception:
                pass

    def _put(self, key: str, blob: bytes) -> None:
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("S3PostingStore.put expects bytes")
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=bytes(blob),
            ContentType="application/octet-stream",
        )

    def _delete(self, key: str) -> None:
        try:
            self._s3.delete_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            if _is_not_found(exc):
                return
            logger.warning("S3PostingStore delete failed for %s: %s", key, exc)
            raise

    def _list_under(self, prefix: str) -> List[str]:
        keys: List[str] = []
        paginator = getattr(self._s3, "get_paginator", None)
        if paginator is not None:
            pages = list(paginator("list_objects_v2").paginate(
                Bucket=self._bucket, Prefix=prefix,
            ))
        else:
            pages = [self._s3.list_objects_v2(
                Bucket=self._bucket, Prefix=prefix,
            )]
        for page in pages:
            for entry in page.get("Contents", []) or []:
                key = entry.get("Key", "")
                if key:
                    keys.append(key)
        return keys


__all__ = ["S3PostingStore"]
