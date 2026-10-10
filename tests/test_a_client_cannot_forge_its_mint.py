"""A client writes its own context keys and nothing else: the store writes how an artifact was made.

`services/mint_context` separates the two authors of a stored context. The store writes the
server-owned facets (`SERVER_OWNED_KEYS`: the mint, the rung, the content address, the upload
bookkeeping); a client writes the rest. Pinned here at the doors a client uses:

* **A create is minted by the server, from the authenticated principal.** Whatever the client
  sends — `minted_by`, a `minted` block naming someone else, a whole context copied off another
  artifact — has its server-owned keys removed, and the mint the router made reaches the service
  as a `Minted`, which no request body can construct.
* **An update keeps the stored facets.** A PATCH, an identity upsert and an identity update
  replace the client's keys and leave the mint, the rung and the content address as stored.
* **The client's keys are stored once, at the top level.** `caller` is never written: it is a
  shape the store reads (`client_view`) and a shape an update may be sent in (`update_keys`).
* **The rung is what the principal earned.** `prism.mass.authorize_and_stamp` derives it from
  the principal and the caller's `provenance` claim.
* **The content address sits where the content routes read it**, on every create path.
"""
from __future__ import annotations

import hashlib
import json

import pytest
from starlette.requests import Request
from starlette.responses import Response

from mantle.db import lattice_api as store
from mantle.routers import artifacts_router as artifacts
from mantle.routers.artifacts_router import _mint_context
from mantle.services import mint_context as mc
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext, get_auth, get_store_db

_ROUTE_STORE_DB = get_store_db
VICTIM, ATTACKER = "u-victim", "k-attacker"


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = store.LatticeDatabase(str(tmp_path / "mint.db"), origin="mint-test")
    from mantle.services import content_crypto
    real_encrypt, real_decrypt = content_crypto.encrypt_content, content_crypto.decrypt_content

    def _provider(pid: str) -> bytes:
        return hashlib.sha256(("m:" + pid).encode()).digest()

    monkeypatch.setattr(content_crypto, "encrypt_content",
                        lambda pid, pt, **kw: real_encrypt(pid, pt, master_key_provider=_provider, **kw))
    monkeypatch.setattr(content_crypto, "decrypt_content",
                        lambda pid, b, **kw: real_decrypt(pid, b, master_key_provider=_provider, **kw))
    from mantle.search.ingest import pipeline_unified as pu
    monkeypatch.setattr(pu, "index_queue", None)
    monkeypatch.setattr(pu, "_sse_index_artifact", lambda *a, **k: pu.ARM_WRITTEN)
    monkeypatch.setattr(pu, "_mantle_index_artifact", lambda *a, **k: pu.ARM_WRITTEN)
    monkeypatch.setattr(pu, "move_artifact_segments", lambda *a, **k: None)
    monkeypatch.setattr(pu, "_mark_indexed", lambda artifact_id: None)
    # The body is addressed in a CAS this test has none of (conftest's `_addressed`); hydrating
    # it back is not the question — where the address is recorded is.
    from mantle.db import doc_boundary
    monkeypatch.setattr(doc_boundary, "decrypt_artifact_content", lambda d, **k: d)
    monkeypatch.setattr(store, "_decrypt_artifact_content", lambda d, **k: d)
    monkeypatch.setattr(artifacts, "check_access", lambda *a, **k: None)
    return database


def _auth(user: str, ptype: str) -> AuthContext:
    return AuthContext(principal_id=user, principal_type=ptype, user_id=user)


def _request() -> Request:
    return Request({"type": "http", "method": "POST", "headers": [], "path": "/artifacts"})


async def _create(db, user, ptype, **body):
    with acting_as(user, principal_type=ptype):
        return await artifacts.create_artifact(
            request=_request(), response=Response(),
            body=artifacts.CreateArtifactRequest(**body), auth=_auth(user, ptype), store_db=db)


async def _patch(db, user, ptype, artifact_id, **body):
    with acting_as(user, principal_type=ptype):
        return await artifacts.update_artifact(
            artifact_id=artifact_id, body=artifacts.UpdateArtifactRequest(**body),
            auth=_auth(user, ptype), store_db=db)


def _ctx(db, artifact_id) -> dict:
    return json.loads(db.artifacts.get_artifact(artifact_id)["context"])


#: The facets a forged update would target, named here rather than read from
#: `SERVER_OWNED_KEYS`, so the check does not move if that set does.
FACETS = ("minted_by", "minted", "provenance", "provenance_history", "addressing", "placement",
          "screen", "content_key", "content_cas_ref")


def _server_facets(ctx: dict) -> dict:
    return {k: ctx.get(k) for k in FACETS}


def _client_keys(ctx: dict) -> dict:
    """The client's keys of a stored context, asserted to be stored at the top level only."""
    assert "caller" not in ctx, "a stored context carries a `caller`"
    return {k: v for k, v in ctx.items() if k not in mc.SERVER_OWNED_KEYS}


# ── replay ────────────────────────────────────────────────────────────────────────────────────

async def test_replaying_a_stored_top_level_context_is_minted_again(db):
    victim = await _create(db, VICTIM, "user", context={"provenance": "human_validated"},
                           content="the victim's note")
    seen = _ctx(db, victim["id"])
    assert (seen["minted"]["principal_id"], seen["provenance"]) == (VICTIM, "human_validated"), \
        "the positive control: a user's claim is earned"

    forged = await _create(db, ATTACKER, "api_key", context=seen, content="evil")
    got = _ctx(db, forged["id"])
    assert got["minted"]["principal_id"] == ATTACKER
    assert got["minted"]["principal_type"] == "api_key"
    assert got["provenance"] == "assertion", "a client key was granted a person's rung"
    assert got["content_key"] != seen["content_key"], "the victim's content address was reused"
    assert _client_keys(got) == {}


async def test_replaying_a_member_context_into_a_collection_is_minted_again(db):
    with acting_as(VICTIM, principal_type="user"):
        inbox = ws.create_container(db, VICTIM, name="Inbox").id
        # The attacker may write into the collection; what it may not do is write as the victim.
        store.upsert_user_collection_grant(db, user_id=ATTACKER, collection_id=inbox,
                                           granted_by=VICTIM, can_read=True, can_create=True)
    member = await _create(db, VICTIM, "user", container_id=inbox,
                           context={"provenance": "human_validated"}, content="v")
    seen = _ctx(db, member["id"])
    assert seen["minted"]["principal_id"] == VICTIM
    replay = {k: v for k, v in seen.items() if k not in ("content_key", "content_cas_ref")}

    forged = await _create(db, ATTACKER, "api_key", container_id=inbox, context=replay,
                           content="evil")
    got = _ctx(db, forged["id"])
    assert got["minted"]["principal_id"] == ATTACKER
    assert got["provenance"] == "assertion"


# ── updates keep the stored facets ────────────────────────────────────────────────────────────

FORGED = {"minted_by": "mantle.mint_context", "provenance": "human_validated",
          "minted": {"principal_id": VICTIM, "principal_type": "user", "at": "x"},
          "content_key": "u-victim/theirs.content", "note": "mine"}


@pytest.mark.parametrize("where", ["member", "top_level"])
async def test_a_patch_cannot_restate_how_the_artifact_was_made(db, where):
    body = {"context": {"note": "first"}, "content": "body"}
    if where == "member":
        with acting_as(ATTACKER, principal_type="api_key"):
            body["container_id"] = ws.create_container(db, ATTACKER, name="w").id
    created = await _create(db, ATTACKER, "api_key", **body)
    before = _server_facets(_ctx(db, created["id"]))

    await _patch(db, ATTACKER, "api_key", created["id"], context=FORGED)

    got = _ctx(db, created["id"])
    assert _server_facets(got) == before, "a PATCH rewrote a server-owned facet"
    assert got["note"] == "mine", "the client's own key was not applied"
    # A top-level PATCH also carries the `collections` list `update_workspace_context` adds.
    assert {k: v for k, v in _client_keys(got).items() if k != "collections"} == {"note": "mine"}


_REDIRECT = {"access": "public", "uri": "https://attacker.example/login",
             "content_source": "external-url", "note": "mine"}


async def test_a_client_cannot_make_its_artifact_a_download_redirect(db):
    """`access: "public"` with a `uri` makes the download-URL answer hand out that uri as the
    artifact's own download. A client that could write them could send every reader of a shared
    collection to an address of its choosing, so neither a create nor a PATCH stores them."""
    with acting_as(ATTACKER, principal_type="api_key"):
        inbox = ws.create_container(db, ATTACKER, name="w").id
    created = await _create(db, ATTACKER, "api_key", container_id=inbox,
                            context=dict(_REDIRECT), content="body")
    got = _ctx(db, created["id"])
    assert not {"access", "uri", "content_source"} & set(got), got
    assert got["note"] == "mine"

    await _patch(db, ATTACKER, "api_key", created["id"], context=dict(_REDIRECT, note="again"))
    got = _ctx(db, created["id"])
    assert not {"access", "uri", "content_source"} & set(got), got
    assert got["note"] == "again"


async def test_an_identity_upsert_cannot_restate_how_the_artifact_was_made(db):
    with acting_as(ATTACKER, principal_type="api_key"):
        inbox = ws.create_container(db, ATTACKER, name="w").id
    first = await _create(db, ATTACKER, "api_key", container_id=inbox, identity="file-1",
                          context={"note": "first"}, content="one")
    before = _server_facets(_ctx(db, first["id"]))

    again = await _create(db, ATTACKER, "api_key", container_id=inbox, identity="file-1",
                          context=FORGED, content="one")
    assert again["id"] == first["id"]
    got = _ctx(db, first["id"])
    assert _server_facets(got) == before
    assert got["note"] == "mine"
    # A top-level update also carries the `collections` list `update_workspace_context` adds.
    assert {k: v for k, v in _client_keys(got).items() if k != "collections"} == {"note": "mine"}


async def test_a_top_level_identity_update_cannot_restate_it_either(db):
    first = await _create(db, ATTACKER, "api_key", identity="doc-1", context={"note": "first"},
                          content="one")
    before = _server_facets(_ctx(db, first["id"]))
    await _create(db, ATTACKER, "api_key", identity="doc-1", context=FORGED, content="one")
    got = _ctx(db, first["id"])
    assert _server_facets(got) == before
    assert got["note"] == "mine"
    # A top-level update also carries the `collections` list `update_workspace_context` adds.
    assert {k: v for k, v in _client_keys(got).items() if k != "collections"} == {"note": "mine"}


# ── a legitimate create ───────────────────────────────────────────────────────────────────────

async def test_a_legitimate_create_records_its_principal_and_the_rung_it_earned(db):
    created = await _create(db, VICTIM, "user", context={"provenance": "human_validated",
                                                         "note": "n"}, content="c")
    got = _ctx(db, created["id"])
    assert got["minted_by"] == mc.ARM_THIN
    assert (got["minted"]["principal_id"], got["minted"]["principal_type"]) == (VICTIM, "user")
    assert got["provenance"] == "human_validated"
    assert _client_keys(got) == {"note": "n"}


async def test_an_over_claim_is_recorded_in_the_history_not_kept_as_a_key(db):
    created = await _create(db, ATTACKER, "api_key", context={"provenance": "human_validated",
                                                              "note": "n"}, content="c")
    got = _ctx(db, created["id"])
    assert got["provenance"] == "assertion"
    assert got["provenance_history"] == ["human_validated"], "the over-claim is not auditable"
    assert _client_keys(got) == {"note": "n"}


def test_a_mint_handed_over_is_stored_as_is_with_the_content_address_on_top(db):
    """The out-of-band handoff: the service stores the router's `Minted`, and records the
    content address onto it at the top level."""
    with acting_as(VICTIM, principal_type="user"):
        made = mc.mint(artifact_id="x", content="body", caller={"note": "n"})
        entity = ws.create_container(db, VICTIM, context=json.dumps({"ignored": True}),
                                     content="body", minted=mc.Minted(made))
    got = json.loads(entity.context)
    assert {k: v for k, v in got.items() if k not in ("content_key", "content_cas_ref")} == made
    assert got["content_key"] and got["content_cas_ref"]


# ── the content address is where the content routes read it ─────────────────────────────────

async def test_a_top_level_create_with_a_body_can_be_fetched(db, client, monkeypatch):
    from mantle.main import app
    import mantle.services.content_service as cs

    created = await _create(db, VICTIM, "user", context={"note": "n"}, content="the body",
                            content_type="text/plain")
    ctx = _ctx(db, created["id"])
    assert ctx.get("content_key") and ctx.get("content_cas_ref"), ctx
    assert _client_keys(ctx) == {"note": "n", "content_type": "text/plain"}

    served = {}

    def _bytes(key, owner, *, cas_ref=None, collection_id=None, **_):
        served["args"] = (key, cas_ref)
        return b"the body"
    monkeypatch.setattr(cs, "get_bytes_decrypted", _bytes)

    def _store_db():
        yield db
    app.dependency_overrides[_ROUTE_STORE_DB] = _store_db
    app.dependency_overrides[get_auth] = lambda: _auth(VICTIM, "user")
    try:
        with acting_as(VICTIM, principal_type="user"):
            url = await client.get(f"/artifacts/{created['id']}/content-url")
            body = await client.get(f"/artifacts/{created['id']}/content")
    finally:
        app.dependency_overrides.pop(_ROUTE_STORE_DB, None)
        app.dependency_overrides.pop(get_auth, None)
    assert url.status_code == 200, url.text
    assert url.json()["url"] == f"/artifacts/{created['id']}/content"
    assert body.status_code == 200, body.text
    assert body.content == b"the body"
    assert served["args"] == (ctx["content_key"], ctx["content_cas_ref"])


# ── the rung ─────────────────────────────────────────────────────────────────────────────────

def _mint_as(principal_type, context, *, principal_id="p-1", actor=None):
    with acting_as(principal_id, principal_type=principal_type, actor=actor):
        caller_json, minted = _mint_context(json.dumps(context), db=None, artifact_id="a-1",
                                            content_type="text/plain", content="x",
                                            container_id=None)
    return caller_json, minted.context()


def test_the_router_hands_the_mint_over_as_an_object_and_the_client_keys_as_json():
    caller_json, got = _mint_as("api_key", FORGED)
    assert got["minted"]["principal_id"] == "p-1"
    assert json.loads(caller_json) == {"note": "mine"}, "a server-owned key survived the door"


@pytest.mark.parametrize("principal_type,claim,actor,earned", [
    ("user", "human_validated", None, "human_validated"),
    ("user", None, None, "unknown"),
    ("user", "human_validated", "bot-1", "assertion"),        # delegated: acting for the user
    ("server", "observed", None, "observed"),
    ("server", "human_validated", None, "assertion"),
    ("service", "observed", None, "observed"),
    ("service", None, None, "unknown"),
    ("api_key", "human_validated", None, "assertion"),
    ("api_key", "hypothesis", None, "hypothesis"),
])
def test_the_rung_is_what_the_principal_earned(principal_type, claim, actor, earned):
    ctx = {"note": "n"} if claim is None else {"provenance": claim}
    _caller, got = _mint_as(principal_type, ctx, actor=actor)
    assert got["provenance"] == earned, got
    if claim and claim != earned:
        assert got.get("provenance_history") == [claim], "the over-claim is not auditable"


# ── the caller's statement is the latest one ──────────────────────────────────────────────────

def _flat(db, artifact_id) -> dict:
    from mantle.clients.artifact_helpers import parse_artifact_context
    return parse_artifact_context({"context": db.artifacts.get_artifact(artifact_id)["context"]})


async def test_after_an_identity_upsert_the_caller_is_the_update(db):
    with acting_as(VICTIM, principal_type="user"):
        inbox = ws.create_container(db, VICTIM, name="w").id
    await _create(db, VICTIM, "user", container_id=inbox, identity="file-1",
                  context={"title": "first capture"}, content="one")
    m = await _create(db, VICTIM, "user", container_id=inbox, identity="file-1",
                      context={"title": "second capture"}, content="two")
    got = _ctx(db, m["id"])
    assert _client_keys(got) == {"title": "second capture"}
    assert _flat(db, m["id"])["title"] == "second capture"


async def test_a_key_a_patch_removed_does_not_return_through_the_flattening_reader(db):
    with acting_as(VICTIM, principal_type="user"):
        inbox = ws.create_container(db, VICTIM, name="w").id
    m = await _create(db, VICTIM, "user", container_id=inbox,
                      context={"title": "old", "tag": "secret-label"}, content="x")
    assert _flat(db, m["id"])["tag"] == "secret-label", "the positive control"
    await _patch(db, VICTIM, "user", m["id"], context={"title": "new"})
    flat = _flat(db, m["id"])
    assert flat["title"] == "new"
    assert "tag" not in flat, "a key the PATCH removed came back"
    assert _client_keys(_ctx(db, m["id"])) == {"title": "new"}


def test_the_flattening_reader_never_lifts_a_store_owned_key_out_of_caller():
    """`caller.provenance` is the claim a create made. With no top-level rung — a store with no
    `prism.mass`, or an unminted row — lifting it would present the claim as the rung."""
    from mantle.clients.artifact_helpers import parse_artifact_context
    flat = parse_artifact_context({"context": json.dumps(
        {"caller": {"provenance": "human_validated", "content_key": "x", "note": "n"}})})
    assert flat.get("note") == "n"
    assert "provenance" not in flat and "content_key" not in flat


# ── the mint's addressing follows the bytes ──────────────────────────────────────────────────

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("how", ["identity_member", "identity_top", "patch_member", "patch_top"])
async def test_a_content_overwrite_readdresses_the_mint(db, how):
    with acting_as(VICTIM, principal_type="user"):
        inbox = ws.create_container(db, VICTIM, name="w").id
    if how == "identity_member":
        first = await _create(db, VICTIM, "user", container_id=inbox, identity="f", content="one")
        await _create(db, VICTIM, "user", container_id=inbox, identity="f", content="two")
    elif how == "identity_top":
        first = await _create(db, VICTIM, "user", identity="t", content="one")
        await _create(db, VICTIM, "user", identity="t", content="two")
    elif how == "patch_member":
        first = await _create(db, VICTIM, "user", container_id=inbox, content="one")
        await _patch(db, VICTIM, "user", first["id"], content="two")
    else:
        first = await _create(db, VICTIM, "user", content="one")
        await _patch(db, VICTIM, "user", first["id"], content="two")
    ctx = _ctx(db, first["id"])
    assert ctx["content_cas_ref"] == "cas/" + _sha("two")
    assert (ctx["addressing"]["sha256"], ctx["addressing"]["bytes"]) == (_sha("two"), 3), \
        "the context names two contents"


def test_readdressing_drops_the_enrichment_of_the_old_bytes():
    stored = {"addressing": {"sha256": _sha("one"), "bytes": 3, "content_type": "text/plain"},
              "enriched": {"summary": "about one"}, "minted_by": mc.ARM_THIN}
    got = json.loads(mc.readdressed(json.dumps(stored), "four"))
    assert got["addressing"] == {"sha256": _sha("four"), "bytes": 4, "content_type": "text/plain"}
    assert "enriched" not in got
    assert json.loads(mc.readdressed(json.dumps({"note": "n"}), "x")) == {"note": "n"}


# ── upload-initiate: a client context, minted ────────────────────────────────────────────────

FORGED_UPLOAD = {"minted_by": "mantle.mint_context", "provenance": "human_validated",
                 "minted": {"principal_id": VICTIM, "principal_type": "user", "at": "x"},
                 "placement": {"collection_id": "somewhere-else"},
                 "upload": {"status": "complete"}, "content_cas_ref": "cas/" + "a" * 64,
                 "note": "mine"}


async def test_upload_initiate_accepts_a_context_and_mints_the_upload(db):
    with acting_as(ATTACKER, principal_type="api_key"):
        inbox = ws.create_container(db, ATTACKER, name="w").id
        out = await artifacts.upload_initiate(
            artifact_id=inbox,
            body=artifacts.UploadInitiateRequest(filename="f.bin", content_type="application/pdf",
                                                 size=10, context=FORGED_UPLOAD),
            auth=_auth(ATTACKER, "api_key"), store_db=db)
    got = _ctx(db, out["upload_id"])
    assert got["note"] == "mine", "the client's own key was dropped"
    assert (got["minted"]["principal_id"], got["minted"]["principal_type"]) \
        == (ATTACKER, "api_key")
    assert got["provenance"] != "human_validated"
    assert got["placement"]["collection_id"] == inbox
    assert got["upload"]["status"] == "initiated", "a client set the upload's state"
    assert "content_cas_ref" not in got
    assert got["addressing"]["bytes"] == 0 and got["provenance"]


async def test_an_in_process_member_create_is_minted_for_the_acting_principal(db):
    with acting_as(ATTACKER, principal_type="api_key"):
        inbox = ws.create_container(db, ATTACKER, name="w").id
        m = ws.create_workspace_artifact(db, ATTACKER, inbox, context=json.dumps(FORGED_UPLOAD),
                                         content="")
    got = json.loads(m.context)
    assert (got["minted"]["principal_id"], got["minted_by"]) == (ATTACKER, mc.ARM_THIN)
    assert got["provenance"] != "human_validated"
    assert got["placement"]["collection_id"] == inbox


def test_context_patch_naming_a_store_owned_key_is_refused():
    from pydantic import ValidationError
    with pytest.raises(ValidationError) as exc:
        artifacts.UploadStatusRequest(
            context_patch={"content_cas_ref": "cas/" + "b" * 64, "note": "n"})
    assert "content_cas_ref" in str(exc.value)
    assert artifacts.UploadStatusRequest(context_patch={"note": "n"}).context_patch == {"note": "n"}


# ── authorization carried in a context ───────────────────────────────────────────────────────

def test_a_webhook_body_cannot_restate_its_source(db, monkeypatch):
    with acting_as(VICTIM, principal_type="user"):
        inbox = ws.create_container(db, VICTIM, name="w")
    monkeypatch.setattr(ws, "resolve_card_key", lambda *a, **k: ({"id": "card-1"}, inbox, VICTIM))
    with acting_as(VICTIM, principal_type="user"):
        msg_id, _ = ws.receive_card_inbound_message(
            db, "card-1", "tok", "hello", channel="sms",
            context={"source_artifact_id": "some-other-card",
                     "inbound": {"channel": "forged", "via": "forged"}, "note": "n"})
    got = _ctx(db, msg_id)
    assert got["source_artifact_id"] == "card-1"
    assert got["inbound"] == {"channel": "sms", "via": "webhook"}
    assert got["note"] == "n"


async def test_read_on_a_collection_does_not_let_a_binding_link_notes_into_it(db):
    """A binding is the caller's own statement in its workspace context. Linking a note into the
    bound collection takes `add` there, as adding a member does; read is not enough."""
    with acting_as(VICTIM, principal_type="user"):
        theirs = ws.create_container(db, VICTIM, name="theirs").id
        store.upsert_user_collection_grant(db, user_id=ATTACKER, collection_id=theirs,
                                           granted_by=VICTIM, can_read=True)
    mine = (await _create(db, ATTACKER, "user", name="mine"))["id"]
    await _patch(db, ATTACKER, "user", mine, context={
        "bindings": {"target_collections": {"artifact_ids": [theirs]}}})
    note = await _create(db, ATTACKER, "user", container_id=mine, content="spam")

    members = {r.get("root_id") or r.get("id") for r in store.list_collection_artifacts(db, theirs)}
    assert note["root_id"] not in members, "a read-only grantee linked a note into the collection"


async def test_add_on_a_collection_lets_a_binding_link_notes_into_it(db):
    """The positive control: with `add`, the binding links."""
    with acting_as(VICTIM, principal_type="user"):
        theirs = ws.create_container(db, VICTIM, name="theirs").id
        store.upsert_user_collection_grant(db, user_id=ATTACKER, collection_id=theirs,
                                           granted_by=VICTIM, can_read=True, can_add=True)
    mine = (await _create(db, ATTACKER, "user", name="mine"))["id"]
    await _patch(db, ATTACKER, "user", mine, context={
        "bindings": {"target_collections": {"artifact_ids": [theirs]}}})
    note = await _create(db, ATTACKER, "user", container_id=mine, content="welcome")

    members = {r.get("root_id") or r.get("id") for r in store.list_collection_artifacts(db, theirs)}
    assert note["root_id"] in members


# ── a read-modify-write keeps the client's keys ──────────────────────────────────────────────
#
# A client that reads a context and sends it back — Astra's layout toggle and card editor, the
# MCP tools that patch one field — sends the client keys where it read them, at the top level,
# and the top level of an update is the whole statement (`mint_context.update_keys`): a key the
# client did not change is kept, and one it removed is removed.

RMW_CONTEXT = {"title": "Quarterly", "tags": ["a"], "layout": {"mode": "grid"}}


async def _rmw_subject(db, origin):
    with acting_as(VICTIM, principal_type="user"):
        inbox = ws.create_container(db, VICTIM, name="w").id
        if origin == "in_process":
            return ws.create_workspace_artifact(db, VICTIM, inbox,
                                                context=json.dumps(RMW_CONTEXT), content="one").id
    return (await _create(db, VICTIM, "user", container_id=inbox, context=RMW_CONTEXT,
                          content="one"))["id"]


@pytest.mark.parametrize("origin", ["router", "in_process"])
@pytest.mark.parametrize("shape", ["raw_plus_layout", "unchanged_raw_with_a_content_edit"])
async def test_a_read_modify_write_keeps_every_key_it_did_not_change(db, origin, shape):
    aid = await _rmw_subject(db, origin)
    raw = _ctx(db, aid)
    want = dict(RMW_CONTEXT)
    if shape == "raw_plus_layout":
        await _patch(db, VICTIM, "user", aid, context={**raw, "layout": {"mode": "list"}})
        want["layout"] = {"mode": "list"}
    else:
        await _patch(db, VICTIM, "user", aid, context=raw, content="two")
    assert _client_keys(_ctx(db, aid)) == want


# ── a `caller` in an update never outvotes its top level ─────────────────────────────────────
#
# A stored context can hold client keys under `caller` — in both places, or some under `caller`
# alone. A client reads it and sends it back whole, `caller` included. The top level of that
# update decides every key the stored top level holds; the update's `caller` supplies only a key
# neither top level holds (`mint_context.update_keys`).

def _store_raw(db, artifact_id, ctx: dict) -> None:
    """Store ``ctx`` as given: a shape the store must read, whatever wrote it."""
    entity = store.get_artifact(db, artifact_id)
    entity.context = json.dumps(ctx)
    store.update_artifact(db, entity)


async def _member_stored_as(db, client_shape: dict) -> str:
    """A minted member whose client keys are stored exactly as ``client_shape`` lays them out."""
    with acting_as(VICTIM, principal_type="user"):
        inbox = ws.create_container(db, VICTIM, name="w").id
    aid = (await _create(db, VICTIM, "user", container_id=inbox, content="one"))["id"]
    facets = {k: v for k, v in _ctx(db, aid).items() if k in mc.SERVER_OWNED_KEYS}
    _store_raw(db, aid, {**facets, **client_shape})
    return aid


IN_BOTH = {"title": "One", "tags": ["a"], "caller": {"title": "One", "tags": ["a"]}}


async def test_a_key_deleted_at_the_top_level_stays_deleted(db):
    aid = await _member_stored_as(db, IN_BOTH)
    raw = _ctx(db, aid)
    raw.pop("tags")                       # the edit: `caller` is sent back as it was read
    await _patch(db, VICTIM, "user", aid, context=raw)
    assert _client_keys(_ctx(db, aid)) == {"title": "One"}


async def test_a_second_save_from_a_cached_copy_keeps_the_first(db):
    aid = await _member_stored_as(db, IN_BOTH)
    cached = _ctx(db, aid)
    await _patch(db, VICTIM, "user", aid, context={**cached, "title": "Two"})
    await _patch(db, VICTIM, "user", aid,
                 context={**cached, "title": "Two", "layout": {"mode": "list"}})
    assert _client_keys(_ctx(db, aid)) == {"title": "Two", "tags": ["a"],
                                           "layout": {"mode": "list"}}


async def test_a_top_level_value_is_not_reverted_to_the_one_under_caller(db):
    aid = await _member_stored_as(db, {"title": "B", "caller": {"title": "A"}})
    assert mc.client_view(_ctx(db, aid)) == {"title": "B"}, "the read before any write"
    raw = _ctx(db, aid)
    await _patch(db, VICTIM, "user", aid, context={**raw, "layout": {"mode": "list"}})
    assert _client_keys(_ctx(db, aid)) == {"title": "B", "layout": {"mode": "list"}}


async def test_a_save_racing_another_to_the_same_value_keeps_that_value(db):
    """X reads, Y saves `Two`, and X saves `Two` too from its read. X stated `Two`."""
    aid = await _member_stored_as(db, IN_BOTH)
    x_read = _ctx(db, aid)
    await _patch(db, VICTIM, "user", aid, context={**_ctx(db, aid), "title": "Two"})     # Y
    await _patch(db, VICTIM, "user", aid, context={**x_read, "title": "Two"})            # X
    assert _client_keys(_ctx(db, aid))["title"] == "Two"


@pytest.mark.parametrize("with_facets", [False, True])
async def test_an_update_stating_its_keys_under_caller_alone_still_applies(db, with_facets):
    """The update carries no client key at the top level — a context read back from a row whose
    keys sit under `caller` alone, facets and all, or a bare `{"caller": {...}}`."""
    aid = await _member_stored_as(db, {"title": "One", "tags": ["a"]})
    update = {"caller": {"title": "Two"}}
    if with_facets:
        update.update({k: v for k, v in _ctx(db, aid).items() if k in mc.SERVER_OWNED_KEYS})
    await _patch(db, VICTIM, "user", aid, context=update)
    assert _client_keys(_ctx(db, aid)) == {"title": "Two"}


def test_update_keys_reads_caller_only_for_a_key_neither_top_level_holds():
    both = {"a": 1, "b": 2, "caller": {"a": 1, "b": 2}}
    assert mc.update_keys({"a": 1, "caller": {"a": 9, "b": 2}}, both) == {"a": 1}
    assert mc.update_keys({"a": 1, "caller": {"a": 9, "b": 2}}, {"caller": {"b": 2}}) \
        == {"a": 1, "b": 2}
    assert mc.update_keys({"a": 1, "caller": {"a": 9, "b": 2}}) == {"a": 1, "b": 2}
    assert mc.update_keys({"minted_by": "x", "caller": {"a": 9, "b": 2}}, both) \
        == {"a": 9, "b": 2}
    assert mc.update_keys({"a": 1, "caller": "a string"}) == {"a": 1}
    assert mc.update_keys({"caller": "a string"}) == {}
    assert mc.update_keys("") == {} and mc.update_keys({}) == {}


LEGACY_SPLIT = {"content_type": "text/plain", "caller": {"title": "Quarterly", "source": "/x.md"}}


async def test_a_raw_round_trip_of_a_split_legacy_row_keeps_the_keys_under_caller(db):
    """`content_type` at the top, the create's keys under `caller` alone: a client edits one
    top-level key of the raw context and sends all of it back."""
    aid = await _member_stored_as(db, LEGACY_SPLIT)
    raw = _ctx(db, aid)
    await _patch(db, VICTIM, "user", aid, context={**raw, "layout": {"mode": "list"}})
    assert _client_keys(_ctx(db, aid)) == {"content_type": "text/plain", "title": "Quarterly",
                                           "source": "/x.md", "layout": {"mode": "list"}}


async def test_a_legacy_key_the_client_deleted_from_its_caller_copy_is_deleted(db):
    aid = await _member_stored_as(db, LEGACY_SPLIT)
    raw = _ctx(db, aid)
    raw["caller"] = {"title": "Quarterly"}
    await _patch(db, VICTIM, "user", aid, context={**raw, "layout": {"mode": "list"}})
    assert _client_keys(_ctx(db, aid)) == {"content_type": "text/plain", "title": "Quarterly",
                                           "layout": {"mode": "list"}}


async def test_a_legacy_router_minted_row_reads_correctly_before_its_first_write(db):
    """Client keys under `caller` alone, the store's facets at the top: every reader finds them."""
    from mantle.clients.artifact_helpers import parse_artifact_context
    with acting_as(VICTIM, principal_type="user"):
        target = ws.create_container(db, VICTIM, name="target").id
    mine = (await _create(db, VICTIM, "user", name="mine"))["id"]
    facets = {k: v for k, v in _ctx(db, mine).items() if k in mc.SERVER_OWNED_KEYS}
    legacy = {**facets, "caller": {
        "title": "Legacy", "bindings": {"target_collections": {"artifact_ids": [target]}}}}
    _store_raw(db, mine, legacy)

    assert mc.client_view(_ctx(db, mine))["title"] == "Legacy"
    assert parse_artifact_context({"context": json.dumps(legacy)})["title"] == "Legacy"
    with acting_as(VICTIM, principal_type="user"):
        assert ws.get_workspace_context(db, VICTIM, mine)["title"] == "Legacy"
    assert "caller" in _ctx(db, mine), "the reads wrote nothing"
    note = await _create(db, VICTIM, "user", container_id=mine, content="one")
    members = {r.get("root_id") or r.get("id") for r in store.list_collection_artifacts(db, target)}
    assert note["root_id"] in members, "a binding held under `caller` did nothing"


# ── a stored context never carries `caller` after a write ────────────────────────────────────

LEGACY_MEMBER = {"title": "T", "caller": {"title": "T", "source": "/x.md"}}


async def _written(db, monkeypatch, how) -> dict:
    """The stored context after one write of kind ``how``, from a stored `caller` where the
    write updates a row, and with a client `caller` key sent where it creates one."""
    sent = {"note": "n", "caller": {"note": "forged"}}
    with acting_as(VICTIM, principal_type="user"):
        inbox = ws.create_container(db, VICTIM, name="w").id
    if how == "create_member":
        return _ctx(db, (await _create(db, VICTIM, "user", container_id=inbox, context=sent,
                                       content="c"))["id"])
    if how == "create_top":
        return _ctx(db, (await _create(db, VICTIM, "user", context=sent, content="c"))["id"])
    if how == "create_in_process":
        with acting_as(VICTIM, principal_type="user"):
            return json.loads(ws.create_workspace_artifact(
                db, VICTIM, inbox, context=json.dumps(sent), content="c").context)
    if how == "upload_initiate":
        with acting_as(VICTIM, principal_type="user"):
            out = await artifacts.upload_initiate(
                artifact_id=inbox,
                body=artifacts.UploadInitiateRequest(filename="f.txt", content_type="text/plain",
                                                     size=1, context=sent),
                auth=_auth(VICTIM, "user"), store_db=db)
        return _ctx(db, out["upload_id"])
    if how in ("identity_member", "identity_member_content_only"):
        m = await _create(db, VICTIM, "user", container_id=inbox, identity="f", content="one")
        _store_raw(db, m["id"], {**_ctx(db, m["id"]), **LEGACY_MEMBER})
        extra = {} if how.endswith("content_only") else {"context": {"title": "U"}}
        await _create(db, VICTIM, "user", container_id=inbox, identity="f", content="two",
                      **extra)
        return _ctx(db, m["id"])
    if how in ("identity_top", "identity_top_content_only"):
        m = await _create(db, VICTIM, "user", identity="t", content="one")
        _store_raw(db, m["id"], {**_ctx(db, m["id"]), **LEGACY_MEMBER})
        extra = {} if how.endswith("content_only") else {"context": {"title": "U"}}
        await _create(db, VICTIM, "user", identity="t", content="two", **extra)
        return _ctx(db, m["id"])
    if how in ("patch_member", "patch_member_content_only", "patch_top",
               "patch_top_content_only"):
        where = {"container_id": inbox} if "member" in how else {}
        m = await _create(db, VICTIM, "user", content="one", **where)
        _store_raw(db, m["id"], {**_ctx(db, m["id"]), **LEGACY_MEMBER})
        extra = {"content": "two"} if how.endswith("content_only") else {"context": {"title": "U"}}
        await _patch(db, VICTIM, "user", m["id"], **extra)
        return _ctx(db, m["id"])
    if how == "context_patch":
        with acting_as(VICTIM, principal_type="user"):
            out = await artifacts.upload_initiate(
                artifact_id=inbox,
                body=artifacts.UploadInitiateRequest(filename="f.txt", content_type="text/plain",
                                                     size=1),
                auth=_auth(VICTIM, "user"), store_db=db)
            _store_raw(db, out["upload_id"], {**_ctx(db, out["upload_id"]), **LEGACY_MEMBER})
            await artifacts.upload_status(
                artifact_id=out["upload_id"],
                body=artifacts.UploadStatusRequest(context_patch={"note": "n"}),
                auth=_auth(VICTIM, "user"), store_db=db)
        return _ctx(db, out["upload_id"])
    if how == "put_content":
        _bytes_stored_locally(monkeypatch)
        m = await _create(db, VICTIM, "user", container_id=inbox, content="one")
        _store_raw(db, m["id"], {**_ctx(db, m["id"]), **LEGACY_MEMBER})
        with acting_as(VICTIM, principal_type="user"):
            await artifacts.put_artifact_content(artifact_id=m["id"], request=_BodyRequest(b"x"),
                                                 auth=_auth(VICTIM, "user"), store_db=db)
        return _ctx(db, m["id"])
    if how == "set_binding":
        mine = (await _create(db, VICTIM, "user", name="mine"))["id"]
        _store_raw(db, mine, {**_ctx(db, mine), **LEGACY_MEMBER})
        with acting_as(VICTIM, principal_type="user"):
            ws.set_binding(db, VICTIM, mine, "memory", artifact_id="x-1")
        return _ctx(db, mine)
    raise AssertionError(how)


@pytest.mark.parametrize("how", [
    "create_member", "create_top", "create_in_process", "upload_initiate",
    "identity_member", "identity_member_content_only", "identity_top",
    "identity_top_content_only", "patch_member", "patch_member_content_only", "patch_top",
    "patch_top_content_only", "context_patch", "put_content", "set_binding"])
async def test_no_write_leaves_a_caller_in_the_stored_context(db, monkeypatch, how):
    got = await _written(db, monkeypatch, how)
    assert "caller" not in got, got
    keys = _client_keys(got)
    if how.startswith("create") or how == "upload_initiate":
        assert keys.get("note") == "n", "the client's own key was dropped"
    elif how.endswith("content_only") or how in ("context_patch", "put_content", "set_binding"):
        assert keys.get("source") == "/x.md", "a key held under `caller` was lost"


def test_a_replica_holding_keys_under_caller_is_stored_with_them_at_the_top_level():
    """The mesh apply changes the context and nothing else: the row stays the peer's."""
    from mantle.mesh import sync

    written: list = []

    class _Artifacts:
        def put_many(self, docs, batch=None, stamp_rev=True, **kw):
            assert stamp_rev is False
            written.extend(docs)
            return len(docs)

    class _Store:
        artifacts = _Artifacts()

    sent = {"id": "p-1", "content_type": "text/plain", "_origin": "peer-7", "_seq": 41,
            "_rev": "r-41", "created_by": "u-peer", "modified_by": "u-peer", "state": "live",
            "context": json.dumps({"minted_by": mc.ARM_THIN, "title": "T",
                                   "caller": {"title": "old", "source": "/x.md"}})}
    plain = {"id": "p-2", "content_type": "text/plain", "_origin": "peer-7", "_seq": 42,
             "state": "live", "context": '{"title":"flat"}'}
    as_sent = json.dumps([sent, plain], sort_keys=True)
    assert sync._apply_artifacts(_Store(), [sent, plain]) == 2

    got, same = written
    assert {k: v for k, v in got.items() if k != "context"} \
        == {k: v for k, v in sent.items() if k != "context"}, "a field besides the context moved"
    assert json.loads(got["context"]) == {"minted_by": mc.ARM_THIN, "title": "T",
                                          "source": "/x.md"}
    assert same["context"] == '{"title":"flat"}', "a context with no `caller` was rewritten"
    assert json.dumps([sent, plain], sort_keys=True) == as_sent, "the doc as sent was changed"


def test_a_seed_holding_keys_under_caller_is_stored_with_them_at_the_top_level(monkeypatch):
    from pathlib import Path
    from mantle.services.seed_provisioning import loader

    created: list = []
    monkeypatch.setattr(loader, "db_get_artifact", lambda db, aid: None)
    monkeypatch.setattr(loader, "db_create_artifact", lambda db, e: created.append(e))
    for slug, context in (("a", {"title": "T", "caller": {"title": "old", "source": "/x.md"}}),
                          ("b", {"title": "flat", "k": "é"})):
        body = {"namespace": "test", "slug": slug, "content_type": "text/plain",
                "context": context}
        loader._apply_artifact_card(None, loader._RawArtifact(Path("s.json"), body, "artifact"),
                                    "seed-" + slug, loader.SeedReport())
    lifted, flat = created
    assert json.loads(lifted.context) == {"title": "T", "source": "/x.md"}
    assert flat.context == '{"title":"flat","k":"é"}', "a seed with no `caller` was rewritten"


def test_the_seed_writers_store_the_client_keys_at_the_top_level(monkeypatch):
    """The subscription save, the lead conversion and the platform-email reconcile write a
    context outside the router; each keeps the stored facets and writes no `caller`."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from mantle.entities.subscription import Subscription
    from mantle.services.seed_provisioning import platform_email as pe
    from mantle.services.seed_provisioning import user_provisioning as up

    stored = json.dumps({"minted_by": mc.ARM_THIN, "provenance": "unknown",
                         "caller": {"type": "lead", "email": "j@x.com"}})

    existing = SimpleNamespace(context=stored)
    Subscription(name="w", owner_id="u-1")._overlay_onto(existing)
    got = json.loads(existing.context)
    assert "caller" not in got and got["minted_by"] == mc.ARM_THIN and "filter" in got

    lead = SimpleNamespace(context=stored, content="{}")
    monkeypatch.setattr(up, "get_id_optional", lambda slug: "leads")
    monkeypatch.setattr(up, "db_list_collection_artifacts", lambda db, c: [{"root_id": "l-1"}])
    monkeypatch.setattr(up, "db_get_artifact", lambda db, rid: lead)
    monkeypatch.setattr(up, "db_update_artifact", lambda db, e: None)
    assert up._convert_leads_for_person(MagicMock(), "u-1", "j@x.com") == 1, \
        "a lead held under `caller` was not found"
    got = json.loads(lead.context)
    assert "caller" not in got and got["minted_by"] == mc.ARM_THIN
    assert (got["type"], got["person_id"], got["status"]) == ("lead", "u-1", "converted")

    box = SimpleNamespace(context=stored)
    monkeypatch.setattr(pe, "db_get_artifact", lambda db, aid: box)
    monkeypatch.setattr(pe, "db_update_artifact", lambda db, e: None)
    pe._ensure_artifact(None, "a-1", content_type="t", name="n", description="d",
                        context={"kind": "k"}, content="", owner="o")
    got = json.loads(box.context)
    assert got == {"minted_by": mc.ARM_THIN, "provenance": "unknown", "kind": "k"}


# ── an identity recapture, with and without a context ───────────────────────────────────────

@pytest.mark.parametrize("where", ["member", "top_level"])
async def test_an_identity_recapture_with_no_context_keeps_the_context(db, where):
    """No `context` in the request keeps the stored one, `content_type` or not: the content
    type is the artifact's own field."""
    body = {"identity": "file-1", "content_type": "text/markdown"}
    if where == "member":
        with acting_as(VICTIM, principal_type="user"):
            body["container_id"] = ws.create_container(db, VICTIM, name="w").id
    m = await _create(db, VICTIM, "user", context={"title": "Notes", "source": "/x.md"},
                      content="one", **body)
    before = {k: v for k, v in _client_keys(_ctx(db, m["id"])).items() if k != "collections"}
    assert before == {"title": "Notes", "source": "/x.md", "content_type": "text/markdown"}
    await _create(db, VICTIM, "user", content="two", **body)
    await _create(db, VICTIM, "user", content="three", identity="file-1",
                  **({"container_id": body["container_id"]} if where == "member" else {}))
    after = {k: v for k, v in _client_keys(_ctx(db, m["id"])).items() if k != "collections"}
    assert after == before


@pytest.mark.parametrize("where", ["member", "top_level"])
async def test_an_empty_context_means_the_same_on_a_recapture_and_a_patch(db, where):
    """`context: ""` states no client keys, so it removes them — on PATCH and on an identity
    recapture alike. Omitting `context` is how a request keeps it."""
    where_body = {}
    if where == "member":
        with acting_as(VICTIM, principal_type="user"):
            where_body["container_id"] = ws.create_container(db, VICTIM, name="w").id
    recaptured = await _create(db, VICTIM, "user", identity="r", content="one",
                               context={"title": "T", "source": "/x.md"}, **where_body)
    await _create(db, VICTIM, "user", identity="r", content="one", context="", **where_body)
    patched = await _create(db, VICTIM, "user", content="one",
                            context={"title": "T", "source": "/x.md"}, **where_body)
    await _patch(db, VICTIM, "user", patched["id"], context="")
    by_recapture = _client_keys(_ctx(db, recaptured["id"]))
    assert by_recapture == _client_keys(_ctx(db, patched["id"]))
    assert "title" not in by_recapture and "source" not in by_recapture


# ── the context's `content_type` key follows the field ───────────────────────────────────────

@pytest.mark.parametrize("how", ["recapture_member", "recapture_top_level", "patch_member"])
async def test_a_new_content_type_reaches_the_context_key_the_bytes_are_typed_by(
        db, monkeypatch, how):
    typed: list = []
    stored = ws._store_content_in_s3

    def _typing(artifact_id, content, context_str, **kw):
        typed.append(json.loads(context_str or "{}").get("content_type"))
        return stored(artifact_id, content, context_str, **kw)
    monkeypatch.setattr(ws, "_store_content_in_s3", _typing)

    where = {}
    if how != "recapture_top_level":
        with acting_as(VICTIM, principal_type="user"):
            where["container_id"] = ws.create_container(db, VICTIM, name="w").id
    ident = {} if how == "patch_member" else {"identity": "f"}
    m = await _create(db, VICTIM, "user", context={"title": "N"}, content="one",
                      content_type="text/markdown", **ident, **where)
    assert _client_keys(_ctx(db, m["id"]))["content_type"] == "text/markdown"
    if how == "patch_member":
        await _patch(db, VICTIM, "user", m["id"], content="two", content_type="text/plain")
    else:
        await _create(db, VICTIM, "user", content="two", content_type="text/plain", **ident,
                      **where)
    got = _client_keys(_ctx(db, m["id"]))
    assert {k: v for k, v in got.items() if k != "collections"} \
        == {"title": "N", "content_type": "text/plain"}
    assert db.artifacts.get_artifact(m["id"])["content_type"] == "text/plain"
    assert typed[-1] == "text/plain", "the new bytes were typed by the old content type"


async def test_a_context_with_no_content_type_key_is_not_given_one(db):
    with acting_as(VICTIM, principal_type="user"):
        inbox = ws.create_container(db, VICTIM, name="w").id
    m = await _create(db, VICTIM, "user", container_id=inbox, identity="f",
                      context={"title": "N"}, content="one")
    await _create(db, VICTIM, "user", container_id=inbox, identity="f", content="two",
                  content_type="text/plain")
    assert _client_keys(_ctx(db, m["id"])) == {"title": "N"}


# ── every other writer stores the client keys at the top level ──────────────────────────────

async def test_context_patch_stores_the_client_keys_at_the_top_level(db):
    with acting_as(VICTIM, principal_type="user"):
        inbox = ws.create_container(db, VICTIM, name="w").id
        out = await artifacts.upload_initiate(
            artifact_id=inbox,
            body=artifacts.UploadInitiateRequest(filename="f.bin", content_type="text/plain",
                                                 size=1),
            auth=_auth(VICTIM, "user"), store_db=db)
    await _patch(db, VICTIM, "user", out["upload_id"], context={"title": "first"})
    with acting_as(VICTIM, principal_type="user"):
        await artifacts.upload_status(
            artifact_id=out["upload_id"],
            body=artifacts.UploadStatusRequest(context_patch={"title": "second"}),
            auth=_auth(VICTIM, "user"), store_db=db)
    assert _client_keys(_ctx(db, out["upload_id"]))["title"] == "second"


async def test_set_binding_stores_the_client_keys_at_the_top_level(db):
    mine = (await _create(db, VICTIM, "user", name="mine", context={"title": "W"}))["id"]
    with acting_as(VICTIM, principal_type="user"):
        ws.set_binding(db, VICTIM, mine, "memory", artifact_id="x-1")
    got = _client_keys(_ctx(db, mine))
    assert got["title"] == "W"
    assert got["bindings"]["memory"]["artifact_id"] == "x-1"
    with acting_as(VICTIM, principal_type="user"):
        ws.clear_binding(db, VICTIM, mine, "memory")
    assert "memory" not in _client_keys(_ctx(db, mine)).get("bindings", {})


async def test_a_binding_sent_on_the_create_is_honoured_at_once(db):
    with acting_as(ATTACKER, principal_type="user"):
        theirs = ws.create_container(db, ATTACKER, name="theirs").id
        store.upsert_user_collection_grant(db, user_id=VICTIM, collection_id=theirs,
                                           granted_by=ATTACKER, can_read=True, can_add=True)
    mine = (await _create(db, VICTIM, "user", name="mine", context={
        "bindings": {"target_collections": {"artifact_ids": [theirs]}}}))["id"]
    note = await _create(db, VICTIM, "user", container_id=mine, content="one")
    members = {r.get("root_id") or r.get("id") for r in store.list_collection_artifacts(db, theirs)}
    assert note["root_id"] in members, "a binding sent on the create did nothing"


# ── the byte route and upload completion readdress the mint ──────────────────────────────────

class _BodyRequest:
    def __init__(self, body: bytes, ctype: str = "text/plain"):
        self._body, self.headers = body, {"content-type": ctype}

    async def body(self):
        return self._body


def _bytes_stored_locally(monkeypatch):
    from mantle.services import content_service
    monkeypatch.setattr(content_service, "put_bytes_encrypted",
                        lambda key, body, ct, owner, **kw: "cas/" + hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(content_service, "local_content_has", lambda ref: True)


async def test_put_content_readdresses_the_mint(db, monkeypatch):
    _bytes_stored_locally(monkeypatch)
    with acting_as(VICTIM, principal_type="user"):
        inbox = ws.create_container(db, VICTIM, name="w").id
    m = await _create(db, VICTIM, "user", container_id=inbox, content="one",
                      context={"title": "t"}, content_type="text/plain")
    with acting_as(VICTIM, principal_type="user"):
        await artifacts.put_artifact_content(artifact_id=m["id"],
                                             request=_BodyRequest(b"two-new-bytes"),
                                             auth=_auth(VICTIM, "user"), store_db=db)
    ctx = _ctx(db, m["id"])
    digest = hashlib.sha256(b"two-new-bytes").hexdigest()
    assert ctx["content_sha256"] == digest
    assert (ctx["addressing"]["sha256"], ctx["addressing"]["bytes"]) == (digest, 13)
    assert _client_keys(ctx)["title"] == "t"


async def test_a_completed_upload_is_addressed_by_the_bytes_that_landed(db, monkeypatch):
    import mantle.services.content_service as cs
    _bytes_stored_locally(monkeypatch)
    monkeypatch.setattr(cs, "head_object", lambda key: {"ContentLength": 10,
                                                        "ContentType": "text/plain"})
    monkeypatch.setattr(cs, "persist_object_to_durable", lambda key: True)
    with acting_as(VICTIM, principal_type="user"):
        inbox = ws.create_container(db, VICTIM, name="w").id
        out = await artifacts.upload_initiate(
            artifact_id=inbox,
            body=artifacts.UploadInitiateRequest(filename="f.txt", content_type="text/plain",
                                                 size=10),
            auth=_auth(VICTIM, "user"), store_db=db)
        assert _ctx(db, out["upload_id"])["addressing"]["bytes"] == 0, "the initiated body"
        await artifacts.put_artifact_content(artifact_id=out["upload_id"],
                                             request=_BodyRequest(b"0123456789"),
                                             auth=_auth(VICTIM, "user"), store_db=db)
        await artifacts.upload_status(artifact_id=out["upload_id"],
                                      body=artifacts.UploadStatusRequest(status="complete"),
                                      auth=_auth(VICTIM, "user"), store_db=db)
    ctx = _ctx(db, out["upload_id"])
    assert "upload" not in ctx
    assert ctx["addressing"]["bytes"] == 10
    assert ctx["addressing"]["sha256"] == ctx["content_sha256"] \
        == hashlib.sha256(b"0123456789").hexdigest()


async def test_a_client_size_never_addresses_the_bytes(db, monkeypatch):
    """With no mirror to ask, the length is the one the content route measured — never the
    client's `size`, which `context_patch` can set."""
    import mantle.services.content_service as cs
    _bytes_stored_locally(monkeypatch)
    monkeypatch.setattr(cs, "head_object", lambda key: None)
    monkeypatch.setattr(cs, "persist_object_to_durable", lambda key: False)
    with acting_as(VICTIM, principal_type="user"):
        inbox = ws.create_container(db, VICTIM, name="w").id
        out = await artifacts.upload_initiate(
            artifact_id=inbox,
            body=artifacts.UploadInitiateRequest(filename="f.txt", content_type="text/plain",
                                                 size=999),
            auth=_auth(VICTIM, "user"), store_db=db)
        await artifacts.put_artifact_content(artifact_id=out["upload_id"],
                                             request=_BodyRequest(b"0123456789"),
                                             auth=_auth(VICTIM, "user"), store_db=db)
        await artifacts.upload_status(
            artifact_id=out["upload_id"],
            body=artifacts.UploadStatusRequest(context_patch={"size": 424242}),
            auth=_auth(VICTIM, "user"), store_db=db)
        await artifacts.upload_status(artifact_id=out["upload_id"],
                                      body=artifacts.UploadStatusRequest(status="complete"),
                                      auth=_auth(VICTIM, "user"), store_db=db)
    ctx = _ctx(db, out["upload_id"])
    assert "upload" not in ctx
    assert ctx["addressing"]["bytes"] == 10, ctx["addressing"]


# ── the refusal's shape on the wire ──────────────────────────────────────────────────────────

async def test_context_patch_refusal_is_the_declared_validation_error(client):
    from mantle.main import app
    declared = app.openapi()["paths"]["/artifacts/{artifact_id}/upload-status"]["patch"]
    assert "422" in declared["responses"]
    r = await client.patch("/artifacts/a-1/upload-status",
                           json={"context_patch": {"minted_by": "x", "note": "n"}})
    assert r.status_code == 422, r.text
    (err,) = r.json()["detail"]
    assert err["loc"] == ["body", "context_patch"] and "minted_by" in err["msg"]
