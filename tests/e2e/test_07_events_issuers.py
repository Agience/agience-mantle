"""The events websocket (delivery + ACL) and issuer admin-gating."""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest
import websockets

import _config as cfg


# --- issuers (admin-gated) --------------------------------------------------

def test_issuers_require_admin(user):
    """A non-admin user cannot list or register issuers."""
    r = user["api"].get("/system/issuers")
    assert r.status_code == 403, r.text
    r2 = user["api"].post("/system/issuers", json={
        "issuer": "https://rogue.test/", "jwks": {"keys": []}, "role": "external",
    })
    assert r2.status_code == 403, r2.text


@pytest.mark.external_issuer
def test_registered_issuer_is_listed(operator_api, second_issuer):
    r = operator_api.get("/system/issuers")
    assert r.status_code == 200, r.text
    issuers = r.json().get("issuers", [])
    assert any(i.get("issuer") == second_issuer.issuer for i in issuers), issuers


# --- events websocket -------------------------------------------------------

def _ws_url(token: str) -> str:
    base = cfg.MANTLE_URL
    if base.startswith("https://"):
        ws = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        ws = "ws://" + base[len("http://"):]
    else:
        ws = base
    return f"{ws}/events?access_token={token}"


def _event_artifact_id(evt: dict) -> str | None:
    """The changed artifact's id. The change-feed nests it as
    payload.artifact.{id}; fall back to payload.id for other shapes."""
    payload = evt.get("payload") or {}
    art = payload.get("artifact") if isinstance(payload.get("artifact"), dict) else payload
    return (art or {}).get("id")


async def _subscribe_and_collect(token: str, trigger, *, timeout: float = 6.0) -> list[dict]:
    """Open /events, subscribe to artifact.* , run `trigger()` once acked, and
    collect events until the timeout. Returns the list of received event msgs."""
    received: list[dict] = []
    async with websockets.connect(_ws_url(token)) as ws:
        await ws.send(json.dumps({
            "op": "subscribe", "id": "e2e",
            "filter": {"event_names": ["artifact.created", "artifact.updated"]},
        }))
        # Wait for ack.
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
                if msg.get("ack") == "e2e":
                    break
        except asyncio.TimeoutError:
            return received

        # Fire the trigger now that we're subscribed.
        trigger()

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline - loop.time())
            except asyncio.TimeoutError:
                break
            msg = json.loads(raw)
            if "event" in msg:
                received.append(msg)
    return received


@pytest.mark.events
def test_event_delivered_to_owner(user):
    created_id: dict = {}

    def trigger():
        coll = user["api"].create_collection(f"evt-{uuid.uuid4().hex[:6]}")
        created_id["id"] = coll["id"]

    events = asyncio.run(_subscribe_and_collect(user["token"], trigger))
    ids = {_event_artifact_id(e) for e in events}
    assert created_id.get("id") in ids, f"owner did not receive their own create event: {events}"


@pytest.mark.events
def test_event_acl_hides_other_users_writes(user_factory):
    """A stranger subscribed to the feed must NOT receive events for an artifact
    they cannot read."""
    owner = user_factory("owner")
    stranger = user_factory("stranger")
    created_id: dict = {}

    def trigger():
        coll = owner["api"].create_collection(f"evt-priv-{uuid.uuid4().hex[:6]}")
        created_id["id"] = coll["id"]

    events = asyncio.run(_subscribe_and_collect(stranger["token"], trigger))
    ids = {_event_artifact_id(e) for e in events}
    assert created_id.get("id") not in ids, "ACL leak: stranger received owner's create event"
