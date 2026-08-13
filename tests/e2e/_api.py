"""Thin HTTP client for the blackbox suite.

One `Api` instance = one bearer identity (a user token, an api key, or nothing).
Methods return the raw `httpx.Response` so tests assert on status AND body; a few
high-level helpers wrap the well-known auth flows.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx
from jose import jwt as _jwt

from _config import HTTP_TIMEOUT, MANTLE_URL, ORIGIN_URL


def sub_of(token: str) -> str:
    """The `sub` claim (the person/user id) from a token, unverified. Login
    responses don't echo the id, but it IS the token subject."""
    try:
        return _jwt.get_unverified_claims(token).get("sub", "")
    except Exception:
        return ""


class Api:
    def __init__(self, token: Optional[str] = None, *, origin: str = ORIGIN_URL,
                 mantle: str = MANTLE_URL) -> None:
        self.token = token
        self.origin = origin.rstrip("/")
        self.mantle = mantle.rstrip("/")
        self._c = httpx.Client(timeout=HTTP_TIMEOUT)

    # -- low-level --------------------------------------------------------
    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if extra:
            h.update(extra)
        return h

    def _url(self, base: str, path: str) -> str:
        return path if path.startswith("http") else f"{base}{path}"

    def get(self, path: str, *, on: str = "mantle", params: Optional[dict] = None,
            headers: Optional[dict] = None) -> httpx.Response:
        base = self.mantle if on == "mantle" else self.origin
        return self._c.get(self._url(base, path), headers=self._headers(headers), params=params)

    def post(self, path: str, json: Any = None, *, on: str = "mantle",
             headers: Optional[dict] = None) -> httpx.Response:
        base = self.mantle if on == "mantle" else self.origin
        return self._c.post(self._url(base, path), headers=self._headers(headers), json=json)

    def patch(self, path: str, json: Any = None, *, on: str = "mantle",
              headers: Optional[dict] = None) -> httpx.Response:
        base = self.mantle if on == "mantle" else self.origin
        return self._c.patch(self._url(base, path), headers=self._headers(headers), json=json)

    def delete(self, path: str, *, on: str = "mantle",
               headers: Optional[dict] = None, json: Any = None) -> httpx.Response:
        base = self.mantle if on == "mantle" else self.origin
        return self._c.request("DELETE", self._url(base, path),
                               headers=self._headers(headers), json=json)

    def with_token(self, token: str) -> "Api":
        return Api(token, origin=self.origin, mantle=self.mantle)

    def close(self) -> None:
        self._c.close()

    # -- artifact conveniences (raise on unexpected status) ---------------
    COLLECTION_TYPE = "application/vnd.agience.collection+json"

    def create_collection(self, name: str, *, content_type: str | None = None) -> dict:
        """Top-level container → committed. Returns the created doc."""
        r = self.post("/artifacts", json={
            "name": name, "content_type": content_type or self.COLLECTION_TYPE,
        })
        r.raise_for_status()
        return r.json()

    def create_child(self, container_id: str, *, name: str = "", content: str = "",
                     content_type: str = "text/plain", index: str | None = None) -> dict:
        """Child in a container → draft. `index` ∈ {eager, lazy, None}."""
        body: dict = {"container_id": container_id, "name": name,
                      "content": content, "content_type": content_type}
        if index is not None:
            body["index"] = index
        r = self.post("/artifacts", json=body)
        r.raise_for_status()
        return r.json()

    def get_artifact(self, artifact_id: str) -> httpx.Response:
        return self.get(f"/artifacts/{artifact_id}")

    def commit(self, artifact_id: str) -> httpx.Response:
        return self.patch(f"/artifacts/{artifact_id}", json={"state": "committed"})

    def visible(self, *, content_type: str | None = None, action: str = "read") -> list:
        params = {"action": action}
        if content_type:
            params["content_type"] = content_type
        r = self.get("/artifacts/visible", params=params)
        r.raise_for_status()
        return r.json()

    def search(self, query_text: str, *, state: str = "committed",
               scope: list[str] | None = None, size: int = 20) -> httpx.Response:
        body: dict = {"query_text": query_text, "state": state, "size": size}
        if scope is not None:
            body["scope"] = scope
        return self.post("/artifacts/recall", json=body)


# --- well-known Origin auth flows (shapes confirmed against auth_router) -----

def bootstrap_claim(token: str, *, email: str, name: str, password: str) -> dict:
    """Claim the single-use bootstrap token -> the first operator (platform admin).
    Returns the JSON payload ({access_token, refresh_token, person_id})."""
    r = Api().post("/auth/bootstrap/claim", on="origin",
                   json={"token": token, "email": email, "name": name, "password": password})
    r.raise_for_status()
    return r.json()


def register(username: str, password: str, *, name: str = "", email: str = "") -> httpx.Response:
    return Api().post("/auth/password/register", on="origin",
                      json={"username": username, "password": password, "name": name, "email": email})


def login(identifier: str, password: str) -> dict:
    r = Api().post("/auth/password/login", on="origin",
                   json={"identifier": identifier, "password": password})
    r.raise_for_status()
    return r.json()
