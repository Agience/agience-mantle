"""HTTP client for Origin's part in key custody: its share of each master key
(`/internal/key-shares`), background delegations (`/internal/custody-delegations`) and the
authority ledger (`/internal/authority/entries`).

Under threshold custody (`MANTLE_MASTER_KEY_CUSTODY=threshold`) a principal's master key is split
2-of-3, and Origin holds share 2 (`search/mantle/threshold_custody.py`). Origin releases it only to
a credential whose subject reaches the key, by its own walk over its replica of the authority
ledger, which Mantle feeds through `submit_entry` (`services/authority_ledger.py`). This client is
separate from `origin_client.OriginClient` (the operator-id lookup) because the two differ in kind:
that one is optional and degrades to a local answer, and this one is required, so every failure
here is raised rather than read as an absent share. "Origin did not answer" and "Origin holds no
share" are different answers, and only the second may let a key be minted.

Each call carries Mantle's own service JWT (`peer_signing.sign_service_jwt(audience="origin")`),
which Origin verifies against the authority manifest's anchor for `mantle`, bound to that one call:
it lives `TOKEN_TTL` seconds, carries a fresh `jti` that Origin accepts once, and names the method
(`htm`), the route path (`htu`) and what the call is about: the principal and the scope of a
release (`principal_id`, `collection_id`, `artifact_id`), or the SHA-256 of a ledger entry
(`body_sha256`).

A release and a ledger submission also carry the acting caller's own credential in
`X-Acting-Credential`, which Origin verifies itself. It is never logged here.

Only a 404 whose body says `"absent": true` is read as "no share". A 404 from anything else (a
proxy, an Origin without the route) is a failure like any other.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from typing import Optional
from urllib.parse import quote

import httpx

from mantle.services import peer_signing

logger = logging.getLogger(__name__)

#: Every call is bounded well inside the custody lock's 30 s (`threshold_custody.LOCK_TIMEOUT`),
#: since `put` calls Origin while holding it: a hung Origin refuses the put rather than holding it.
CALL_TIMEOUT = httpx.Timeout(8.0, connect=3.0)
_PATH = "/internal/key-shares/"
_DELEGATION_PATH = "/internal/custody-delegations/"
_ENTRIES_PATH = "/internal/authority/entries"
ACTING_CREDENTIAL_HEADER = "X-Acting-Credential"

#: The lifetime of a custody token; Origin refuses one that lives longer than 60 seconds.
TOKEN_TTL = 30

#: A minted background delegation is used until this many seconds before it expires.
DELEGATION_MARGIN = 60


class CustodianUnavailable(RuntimeError):
    """Origin could not answer for a share: unreachable, refused, or answered unexpectedly."""


class ShareConflict(CustodianUnavailable):
    """Origin holds a share of another key for this principal and kept it."""


class AuthorityRefused(CustodianUnavailable):
    """Origin answered, and refused: `status` and Origin's reason."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"Origin refused ({status}): {detail}")
        self.status, self.detail = status, detail


def canonical(obj) -> str:
    """The canonical JSON Origin hashes (`authority_ledger.canonical`)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def entry_digest(entry: dict) -> str:
    return hashlib.sha256(canonical(entry).encode("utf-8")).hexdigest()


def _detail(resp) -> str:
    try:
        return str(resp.json().get("detail") or "")[:300]
    except (ValueError, AttributeError):
        return ""


#: The most any one answer from Origin may carry (review 19): a ledger page of 500 entries is a few
#: hundred kilobytes, so this bounds a misbehaving or impersonated Origin, never a real answer.
MAX_ANSWER_BYTES = 4 * 1024 * 1024


class _Bounded:
    """An `httpx.Client` whose every answer is read, streamed, to at most `MAX_ANSWER_BYTES`; a
    larger one raises `CustodianUnavailable` before it is parsed (review 19)."""

    def __init__(self, inner: httpx.Client) -> None:
        self._inner = inner

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        with self._inner.stream(method, url, **kwargs) as resp:
            declared = resp.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > MAX_ANSWER_BYTES:
                raise CustodianUnavailable(f"Origin's answer is too large ({declared} bytes)")
            body = bytearray()
            for chunk in resp.iter_bytes():       # decoded: the bound is on what is parsed
                body.extend(chunk)
                if len(body) > MAX_ANSWER_BYTES:
                    raise CustodianUnavailable("Origin's answer is too large")
            # The body is already decoded, so the answer carries no encoding of its own: kept, a
            # gzip `Content-Encoding` had it decoded twice (2026-11-09: every answer the edge
            # compressed failed as `DecodingError`, the ledger seed included; loopback, uncompressed,
            # never showed it).
            headers = [(k, v) for k, v in resp.headers.multi_items()
                       if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")]
            return httpx.Response(resp.status_code, headers=headers, content=bytes(body),
                                  request=resp.request)

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)


class OriginShareCustodian:
    """Store and fetch Origin's share of a principal's master key; mint background delegations;
    submit ledger entries."""

    def __init__(self, base_uri: Optional[str] = None, client: Optional[httpx.Client] = None):
        if base_uri is None:
            from mantle import config
            base_uri = getattr(config, "ORIGIN_URI", "")
        self._base = (base_uri or "").strip().rstrip("/")
        if not self._base:
            raise ValueError("threshold custody needs ORIGIN_URI: Origin holds a share of each key")
        self._client = _Bounded(client or httpx.Client(timeout=CALL_TIMEOUT))
        self._minted: dict[str, tuple[float, str]] = {}
        self._minted_lock = threading.Lock()

    def _url(self, principal_id: str) -> str:
        return f"{self._base}{_PATH}{quote(principal_id, safe='')}"

    def _token(self, method: str, path: str, claims: dict) -> str:
        return peer_signing.sign_service_jwt(
            audience="origin", ttl_seconds=TOKEN_TTL,
            additional_claims={"jti": uuid.uuid4().hex, "htm": method, "htu": path, **claims})

    def _headers(self, method: str, principal_id: str, **scope) -> dict:
        claims = {"principal_id": principal_id, **{k: v for k, v in scope.items() if v}}
        return {"Authorization": "Bearer " + self._token(method, _PATH + principal_id, claims)}

    def get_share(self, principal_id: str, *, credential: str,
                  collection_id: Optional[str] = None,
                  artifact_id: Optional[str] = None) -> Optional[dict]:
        """`{"x", "y", "kcv", "pending"}`, or None when Origin answers that it holds none.
        `pending` is True unless Origin says the share is activated (review 29, L1): a release
        that says nothing of it fails towards a fresh split. `credential` is
        the acting caller's own, which Origin verifies, and whose reach to this key (for this
        collection or artifact) it decides, before it releases anything."""
        if not credential:
            raise CustodianUnavailable("no acting credential to show Origin")
        scope = {"collection_id": collection_id, "artifact_id": artifact_id}
        headers = {**self._headers("GET", principal_id, **scope),
                   ACTING_CREDENTIAL_HEADER: credential}
        params = {k: v for k, v in scope.items() if v}
        try:
            resp = self._client.get(self._url(principal_id), headers=headers, params=params)
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code == 404:
            try:
                absent = resp.json().get("absent") is True
            except (ValueError, AttributeError):
                absent = False
            if absent:
                return None
            raise CustodianUnavailable("Origin answered 404 without saying it holds no share")
        if resp.status_code in (401, 403):
            raise AuthorityRefused(resp.status_code,
                                   f"Origin refused to release the key share: {_detail(resp)}")
        if resp.status_code != 200:
            raise CustodianUnavailable(f"Origin answered {resp.status_code} for a key share")
        body = resp.json()
        return {"x": body["x"], "y": body["y"], "kcv": body["kcv"],
                "pending": body.get("pending") is not False}

    def put_share(self, principal_id: str, share: dict, *,
                  supersedes: Optional[str] = None, credential: Optional[str] = None) -> str:
        """Store a share; returns Origin's outcome (`stored`, `unchanged`, `replaced`).
        Replacing a share Origin holds needs `credential`, which must reach the key."""
        body = dict(share)
        if supersedes:
            body["supersedes"] = supersedes
        headers = self._headers("PUT", principal_id)
        if credential:
            headers[ACTING_CREDENTIAL_HEADER] = credential
        try:
            resp = self._client.put(self._url(principal_id), json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code == 409:
            raise ShareConflict(f"Origin holds a share of another key for {principal_id!r}")
        if resp.status_code in (401, 403):
            raise AuthorityRefused(resp.status_code, _detail(resp))
        if resp.status_code != 200:
            raise CustodianUnavailable(f"Origin answered {resp.status_code} storing a key share")
        return resp.json().get("outcome", "stored")

    # -- background delegations ----------------------------------------------------------------

    def background_delegation(self, principal_id: str, job: str) -> str:
        """A delegation Origin minted for `principal_id` and the named background `job`, reused
        until a minute before it expires. Origin mints one only under a live job grant naming
        that job and covering that principal (John, 2026-11-02); otherwise this raises
        `AuthorityRefused` naming why. The job is bound in the service token."""
        if not job:
            raise AuthorityRefused(403, "background work names no job, and Origin mints for "
                                        "none without one")
        now = time.monotonic()
        key = (principal_id, job)
        with self._minted_lock:
            hit = self._minted.get(key)
            if hit and hit[0] > now:
                return hit[1]
        path = _DELEGATION_PATH + principal_id
        headers = {"Authorization": "Bearer " + self._token("POST", path,
                                                            {"principal_id": principal_id,
                                                             "job": job})}
        try:
            resp = self._client.post(f"{self._base}{_DELEGATION_PATH}"
                                     f"{quote(principal_id, safe='')}", headers=headers,
                                     json={"job": job})
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code in (401, 403):
            raise AuthorityRefused(resp.status_code, _detail(resp))
        if resp.status_code != 200:
            raise CustodianUnavailable(f"Origin answered {resp.status_code} minting a delegation")
        token = resp.json()["token"]
        lifetime = _lifetime(token)
        with self._minted_lock:
            self._minted[key] = (now + max(lifetime - DELEGATION_MARGIN, 0), token)
        return token

    def forget_delegations(self) -> None:
        with self._minted_lock:
            self._minted.clear()

    # -- the authority ledger ------------------------------------------------------------------

    def activate_share(self, principal_id: str, *, credential: str) -> None:
        """This node activated the key: Origin's share of it is no longer replaceable. On the
        acting credential of the key's principal or an Admin on what it keys (review 25)."""
        path = f"/internal/key-shares/{principal_id}/activate"
        headers = {"Authorization": "Bearer " + self._token("POST", path,
                                                            {"principal_id": principal_id}),
                   ACTING_CREDENTIAL_HEADER: credential}
        try:
            resp = self._client.post(f"{self._base}{path}", headers=headers)
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code != 200:
            raise AuthorityRefused(resp.status_code, _detail(resp))

    def reach_verdict(self, resource_id: str, action: str, *, credential: str) -> bool:
        """Origin's own walk: does this credential's subject reach `resource_id` for `action`?"""
        if not credential:
            raise CustodianUnavailable("no acting credential to show Origin")
        path = "/internal/authority/reach"
        headers = {"Authorization": "Bearer " + self._token("GET", path, {
            "resource_id": resource_id, "action": action}), ACTING_CREDENTIAL_HEADER: credential}
        try:
            resp = self._client.get(f"{self._base}{path}", headers=headers,
                                    params={"resource_id": resource_id, "action": action})
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code != 200:
            raise AuthorityRefused(resp.status_code, _detail(resp))
        return resp.json().get("allowed") is True

    def use_verdict(self, credential_id: str, card_id: str, nonce: str, *,
                    credential: str) -> str:
        """Origin's signed verdict that the persona whose delegation `credential` is may use
        `credential_id`'s secret for `card_id`, for the persona's `nonce` (reviews 21 and 22).
        Raises `AuthorityRefused` when Origin refuses, `CustodianUnavailable` when it does not
        answer."""
        if not credential:
            raise CustodianUnavailable("no persona delegation to show Origin")
        path = "/internal/authority/use-verdict"
        body = {"credential_id": str(credential_id), "card_id": str(card_id),
                "nonce": str(nonce)}
        headers = {"Authorization": "Bearer " + self._token("POST", path, {
            "body_sha256": entry_digest(body)}), ACTING_CREDENTIAL_HEADER: credential}
        try:
            resp = self._client.post(f"{self._base}{path}", headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code != 200:
            raise AuthorityRefused(resp.status_code, _detail(resp))
        verdict = (resp.json() or {}).get("verdict")
        if not isinstance(verdict, str) or not verdict:
            raise CustodianUnavailable("Origin answered with no verdict")
        return verdict

    def record_credential_seals(self, credential_id: str, sealed_sha256: dict,
                                endpoints_sha256: str, *, credential: str) -> None:
        """Record at Origin, on the writer's own `credential`, the digests of a credential's
        sealed copies and endpoint list (review 22, F2)."""
        if not credential:
            raise CustodianUnavailable("no writer's credential to show Origin")
        path = "/internal/authority/credential-seals"
        body = {"credential_id": str(credential_id), "sealed_sha256": dict(sealed_sha256),
                "endpoints_sha256": str(endpoints_sha256)}
        headers = {"Authorization": "Bearer " + self._token("POST", path, {
            "body_sha256": entry_digest(body)}), ACTING_CREDENTIAL_HEADER: credential}
        try:
            resp = self._client.post(f"{self._base}{path}", headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code != 200:
            raise AuthorityRefused(resp.status_code, _detail(resp))

    def verify_reach(self, digest: dict) -> dict:
        """Compare this store's digest of reach with Origin's replica: `{"match", "replica"}`."""
        path = "/internal/authority/verify"
        body = {"digest": digest["digest"], "grants": int(digest["grants"]),
                "edges": int(digest["edges"])}
        headers = {"Authorization": "Bearer " + self._token("POST", path, {
            "body_sha256": entry_digest(body)})}
        try:
            resp = self._client.post(f"{self._base}{path}", headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code != 200:
            raise AuthorityRefused(resp.status_code, _detail(resp))
        return resp.json()

    def charged_roots(self, ids: list) -> list:
        """The roots among `ids` a part of which was removed (review 30): their master keys are
        rotated at step 21 and open for their own principal alone."""
        path = "/internal/authority/principal-kinds"
        body = {"ids": [str(i) for i in ids]}
        headers = {"Authorization": "Bearer " + self._token("POST", path, {
            "body_sha256": entry_digest(body)})}
        try:
            resp = self._client.post(f"{self._base}{path}", headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code != 200:
            raise AuthorityRefused(resp.status_code, _detail(resp))
        charged = (resp.json() or {}).get("charged")
        if not isinstance(charged, list):
            raise CustodianUnavailable("Origin's answer names no charged roots")
        return [str(c) for c in charged]

    def principal_kinds(self, ids: list) -> dict:
        """What each key principal is, by Origin's replica (`{id: kind}`; review 27): `person`,
        `collection_key`, `platform`, `root` or `unknown`."""
        path = "/internal/authority/principal-kinds"
        body = {"ids": [str(i) for i in ids]}
        headers = {"Authorization": "Bearer " + self._token("POST", path, {
            "body_sha256": entry_digest(body)})}
        try:
            resp = self._client.post(f"{self._base}{path}", headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code != 200:
            raise AuthorityRefused(resp.status_code, _detail(resp))
        kinds = (resp.json() or {}).get("kinds")
        if not isinstance(kinds, dict):
            raise CustodianUnavailable("Origin's answer names no kinds")
        return {str(k): str(v) for k, v in kinds.items()}

    def platform_users(self, *, credential: str, limit: int, offset: int) -> dict:
        """One page of Origin's person records (`{"users", "total"}`), for the platform admin whose
        own credential this is; Origin decides that from its own replica."""
        if not credential:
            raise CustodianUnavailable("no acting credential to show Origin")
        path = "/internal/authority/platform-users"
        headers = {"Authorization": "Bearer " + self._token("GET", path, {
            "limit": int(limit), "offset": int(offset)}), ACTING_CREDENTIAL_HEADER: credential}
        try:
            resp = self._client.get(f"{self._base}{path}", headers=headers,
                                    params={"limit": int(limit), "offset": int(offset)})
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code != 200:
            raise AuthorityRefused(resp.status_code, _detail(resp))
        return resp.json()

    def ledger_feed(self, after: int) -> dict:
        """Ledger entries after `after` (`/internal/authority/feed`)."""
        path = "/internal/authority/feed"
        headers = {"Authorization": "Bearer " + self._token("GET", path, {})}
        try:
            resp = self._client.get(f"{self._base}{path}", headers=headers,
                                    params={"after": int(after)})
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code != 200:
            raise AuthorityRefused(resp.status_code, _detail(resp))
        return resp.json()

    def confirm_entries(self, seqs: list) -> None:
        """The writes for these ledger entries committed; Origin no longer lets them be
        withdrawn."""
        path = "/internal/authority/confirm"
        body = {"seqs": [int(s) for s in seqs]}
        headers = {"Authorization": "Bearer " + self._token("POST", path, {
            "body_sha256": entry_digest(body)})}             # bound to exactly these seqs
        try:
            resp = self._client.post(f"{self._base}{path}", headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if resp.status_code != 200:
            raise AuthorityRefused(resp.status_code, _detail(resp))

    def seed_done(self, credential: str) -> None:
        """The seed pass is over: Origin revokes its seed credential, so it serves one pass."""
        _done(self, "/internal/authority/seed-done", credential)

    def bootstrap_done(self, credential: str) -> None:
        """The bootstrap batch is over: Origin closes its credential, good for this batch only."""
        _done(self, "/internal/authority/bootstrap-done", credential)

    def submit_entry(self, entry: dict, *, credential: str, bootstrap: bool = False,
                     proof: Optional[dict] = None, compensates: Optional[int] = None,
                     batch: Optional[str] = None) -> dict:
        """Submit one ledger entry on the acting person's credential (or, with `bootstrap`, the
        operator's bootstrap credential); Origin's receipt (`outcome`, `seq`, `entry_hash`,
        `signature`, `kid`), or `AuthorityRefused`. `proof` (an invite claim's token) is checked
        by Origin and not stored; `compensates` names an entry this one withdraws."""
        if not credential:
            raise CustodianUnavailable("no acting credential to show Origin")
        claims = {"body_sha256": entry_digest(entry)}
        if bootstrap:
            claims["bootstrap"] = True
        if compensates is not None:
            claims["compensates"] = int(compensates)
        if proof:
            claims["proof_sha256"] = entry_digest(proof)
        if batch:
            claims["batch"] = batch
        headers = {"Authorization": "Bearer " + self._token("POST", _ENTRIES_PATH, claims),
                   ACTING_CREDENTIAL_HEADER: credential}
        body = {"entry": entry, "bootstrap": bootstrap}
        if proof:
            body["proof"] = proof
        if compensates is not None:
            body["compensates"] = int(compensates)
        if batch:
            body["batch"] = batch
        try:
            resp = self._client.post(f"{self._base}{_ENTRIES_PATH}", headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
        if 400 <= resp.status_code < 500:
            raise AuthorityRefused(resp.status_code, _detail(resp))
        if resp.status_code != 200:
            raise CustodianUnavailable(f"Origin answered {resp.status_code} for a ledger entry")
        return resp.json()


def _done(custodian, path: str, credential: str) -> None:
    headers = {"Authorization": "Bearer " + custodian._token("POST", path, {}),
               ACTING_CREDENTIAL_HEADER: credential}
    try:
        resp = custodian._client.post(f"{custodian._base}{path}", headers=headers)
    except httpx.HTTPError as exc:
        raise CustodianUnavailable(f"Origin is unreachable ({type(exc).__name__})") from exc
    if resp.status_code != 200:
        raise AuthorityRefused(resp.status_code, _detail(resp))


def _lifetime(token: str) -> float:
    """Seconds the (Origin-signed) token has left by its own claims; 0 when unreadable."""
    import base64

    try:
        seg = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))
        return float(claims["exp"]) - time.time()
    except (IndexError, KeyError, ValueError, TypeError):
        return 0.0
