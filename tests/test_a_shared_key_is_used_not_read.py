"""A shared secret is released only sealed to a proven persona, for a card that names it, to someone
granted `invoke` — never readable by them, and never in clear outside that persona.

2026-10-19 (John): a key and an authorizer's secret are one thing; a platform resource is an
artifact with a grant or a public grant; information is crypto-secured, visible but not readable,
and a secret is in clear only at the point of use, with proper authentication and delegation.

Runs on a real store with real grants, real content encryption and the real access check.
"""
from __future__ import annotations

import json
import uuid

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.responses import Response
from starlette.requests import Request

from mantle.attenuation import ACTIONS
from mantle.entities.grant import Grant as GrantEntity
from mantle.routers import artifacts_router as artifacts
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext

from .test_there_is_no_promotion_state import USER, real_node

COLLECTION = "application/vnd.agience.collection+json"
CREDENTIAL = "application/vnd.agience.credential+json"
CONNECTION = "application/vnd.agience.llm-connection+json"
TEAMMATE = "u-teammate"
STRANGER = "u-stranger"
LUMEN_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwk(key, kid):
    from prism.trust import sealing
    n = key.public_key().public_numbers()
    def b(i):
        return sealing._b64(i.to_bytes((i.bit_length() + 7) // 8, "big"))
    return {"kty": "RSA", "kid": kid, "n": b(n.n), "e": b(n.e)}


class _Manifest:
    def get_jwks(self, name):
        if name == "lumen":
            return {"keys": [_jwk(LUMEN_KEY, "lumen-1")]}
        raise KeyError(name)


@pytest.fixture
def db(tmp_path, monkeypatch):
    import prism.trust.authority_trust as at
    monkeypatch.setattr(at, "get_authority_manifest", lambda: _Manifest())
    return real_node(tmp_path, monkeypatch)


def _grant(db, resource_id, grantee, actions):
    from mantle.db import lattice_api as la
    la.create_grant(db, GrantEntity(
        id=str(uuid.uuid4()), resource_id=resource_id, grantee_type="user", grantee_id=grantee,
        granted_by=USER, effect="allow", state=GrantEntity.STATE_ACTIVE,
        **{f"can_{a}": (a in actions) for a in ACTIONS}))


def _as(user, *, actor=None, signer=None, principal_type="user", authority="origin"):
    """`authority`: who signed the delegation; "origin" means Mantle's configured issuer."""
    from mantle import config
    return AuthContext(principal_id=user, principal_type=principal_type, user_id=user, actor=actor,
                       actor_signer=signer,
                       authority=config.AUTHORITY_ISSUER if authority == "origin" else authority)


LUMEN = dict(actor="agience-server-lumen", signer="lumen")


async def _create(db, **body):
    req = Request({"type": "http", "method": "POST", "headers": [], "path": "/artifacts", "query_string": b""})
    with acting_as(USER, principal_type="user"):
        return await artifacts.create_artifact(request=req, response=Response(), auth=_as(USER), store_db=db,
                                               body=artifacts.CreateArtifactRequest(**body))


async def _shared(db, persona_read=True):
    """A shelf holding a key, a connection naming it, another connection and a plain note; the
    teammate holds `invoke` on the shelf and nothing else. Shared with Lumen, the key carries Lumen's
    own Read, which the shelf's Admin grants (`persona_read`)."""
    with acting_as(USER, principal_type="user"):
        shelf = ws.create_container(db, USER, content_type=COLLECTION, name="Model connections").id
    key = await _create(db, container_id=shelf, content_type=CREDENTIAL,
                        context=json.dumps({"title": "Model key"}), content="sk-team-secret")
    conn = await _create(db, container_id=shelf, content_type=CONNECTION, context=json.dumps(
        {"title": "Team model", "provider": "anthropic", "credentials_ref": {"secret_id": key["id"]}}))
    other = await _create(db, container_id=shelf, content_type=CONNECTION, context=json.dumps(
        {"title": "Other model", "credentials_ref": {"secret_id": "some-other-key"}}))
    note = await _create(db, container_id=shelf, context=json.dumps({"title": "notes"}), content="x")
    _grant(db, shelf, TEAMMATE, {"invoke"})
    if persona_read:
        _grant(db, key["id"], "agience-server-lumen", {"read"})
    return key["id"], conn["id"], other["id"], note["id"]


class _Origin:
    """Origin's part, which `agience-origin`'s own tests decide (review 21): it answers every use
    Mantle lets through with a verdict, so these tests read Mantle's own checks."""

    def use_verdict(self, credential_id, card_id, nonce, *, credential):
        return f"verdict:{credential_id}:{card_id}"


@pytest.fixture(autouse=True)
def _origin_answers(monkeypatch):
    monkeypatch.setattr(artifacts, "_use_verdict_custodian", lambda: _Origin())


async def _use(db, auth, key, card):
    from mantle.services.acting_principal import ForwardedCredential
    with acting_as(auth.principal_id, principal_type="user",
                   credential=ForwardedCredential("the-persona-s-delegation")):
        return await artifacts.use_credential(artifact_id=key, body=artifacts.CredentialUseRequest(**{"for": card, "nonce": "n" * 32}),
                                              auth=auth, store_db=db)


def _open_raw(out, key, card):
    from prism.trust import sealing
    return sealing.open_sealed(out["sealed"], LUMEN_KEY, kid="lumen-1", context=f"{key}").decode()


def _open(out, key, card):
    """The secret in the sealed copy (sealed with the endpoints it may go to)."""
    return json.loads(_open_raw(out, key, card))["material"]


async def test_the_proven_persona_gets_the_secret_sealed_to_its_key_for_that_card(db):
    key, conn, _, _ = await _shared(db)
    out = await _use(db, _as(TEAMMATE, **LUMEN), key, conn)
    assert "sk-team-secret" not in json.dumps(out), "the answer carries ciphertext only"
    assert _open(out, key, conn) == "sk-team-secret"
    from prism.trust import sealing
    with pytest.raises(sealing.SealError):
        sealing.open_sealed(out["sealed"], LUMEN_KEY, kid="lumen-1", context="another-credential")


async def test_the_person_granted_invoke_cannot_read_the_secret(db):
    key, _, _, _ = await _shared(db)
    with pytest.raises(HTTPException) as e:
        with acting_as(TEAMMATE, principal_type="user"):
            await artifacts.read_artifact(artifact_id=key, auth=_as(TEAMMATE), store_db=db)
    assert e.value.status_code in (403, 404)


@pytest.mark.parametrize("auth", [
    _as(TEAMMATE),                                                          # their own sign-in
    _as(TEAMMATE, actor="agience-server-lumen", signer="chorus"),           # asserted, not proven
    _as(TEAMMATE, actor="agience-server-lumen"),                            # no signer at all
    _as(TEAMMATE, actor="agience-server-lumen", signer="iris"),             # signer is someone else
    _as(TEAMMATE, actor="some-client", principal_type="mcp_client"),
    # Review 19, H2: a delegation Mantle (or any service but Origin) signed is no one's.
    _as(TEAMMATE, **LUMEN, authority="mantle"),
    _as(TEAMMATE, **LUMEN, authority=None),
])
async def test_only_a_proven_persona_is_handed_a_secret(db, auth):
    key, conn, _, _ = await _shared(db)
    with pytest.raises(HTTPException) as e:
        await _use(db, auth, key, conn)
    assert e.value.status_code == 403


async def test_a_secret_is_released_only_for_its_persona_s_kind_of_card_and_only_to_it(db):
    """Mantle reads the card as stored, for its type only: its context is sealed, and Mantle opens
    no key on this path. That the card names this credential is the persona's reading of the card;
    the copy it gets is bound to the credential's id."""
    key, conn, other, note = await _shared(db)
    with pytest.raises(HTTPException) as e:
        await _use(db, _as(TEAMMATE, **LUMEN), key, note)
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        await _use(db, _as(TEAMMATE, actor="agience-server-iris", signer="iris"), key, conn)
    assert e.value.status_code == 403, "an LLM connection's key is Lumen's to use, not Iris's"


async def test_no_grant_no_secret(db):
    key, conn, _, _ = await _shared(db)
    with pytest.raises(HTTPException) as e:
        await _use(db, _as(STRANGER, **LUMEN), key, conn)
    assert e.value.status_code in (403, 404)


# ── the key demand ────────────────────────────────────────────────────────────────────────────
# `real_node` hands every requester every key, so the tests above cannot see which right the
# content key was demanded under. These read it off the oracle's request directly.

def test_a_content_key_is_demanded_as_read_except_where_use_widens_it(monkeypatch):
    from mantle.search.mantle import wiring
    from mantle.services import content_crypto

    seen = []

    class _Oracle:
        def get_or_create_master_key(self, principal_id, request, collection_id=None):
            seen.append(request.action)
            return b"k" * 32
    monkeypatch.setattr(wiring, "_build_oracle", lambda: _Oracle())
    with acting_as(TEAMMATE, principal_type="user"):
        content_crypto._default_master_key("owner", "shelf")
        with content_crypto.content_key_action("invoke"):
            content_crypto._default_master_key("owner", "shelf")
        content_crypto._default_master_key("owner", "shelf")
    assert seen == ["read", "invoke", "read"]


async def test_an_invoke_caller_gets_the_result_and_never_the_key(db, monkeypatch):
    """John, 2026-11-01: a shared credential is used server-side under Invoke. It was sealed to
    the persona when written; the use route hands the persona that sealed copy without opening
    the credential, and Mantle's oracle releases nothing to the caller (architecture §51)."""
    from mantle.search.mantle import wiring

    key, conn, _, _ = await _shared(db)
    oracle = wiring._build_oracle()
    asked = []
    for name in ("get_or_create_master_key", "get_or_create_collection_key"):
        real = getattr(oracle, name)

        def spy(*a, _real=real, **k):
            req = next((x for x in list(a) + list(k.values()) if hasattr(x, "requester_id")),
                       None)
            asked.append(getattr(req, "requester_id", None))
            return _real(*a, **k)
        monkeypatch.setattr(oracle, name, spy)
    from mantle.services import content_crypto
    from mantle.services.acting_principal import current_acting_principal
    opened_as = []
    real_decrypt = content_crypto.decrypt_content

    def watching(*a, **k):
        actor = current_acting_principal()
        opened_as.append(actor.principal_id if actor else None)
        return real_decrypt(*a, **k)
    monkeypatch.setattr(content_crypto, "decrypt_content", watching)
    out = await _use(db, _as(TEAMMATE, **LUMEN), key, conn)
    assert TEAMMATE not in opened_as                           # nothing opened as the caller
    assert _open(out, key, conn) == "sk-team-secret"           # the persona's result
    assert "sk-team-secret" not in json.dumps(out)             # the caller holds ciphertext
    assert TEAMMATE not in asked                               # no key was asked as the caller
    raw = db.artifacts.get_artifact(key)
    assert raw.get("persona_sealed", {}).get("lumen")          # sealed to the persona, at write


# ── John, 2026-11-02: use is Invoke, and two ordinary grants ──────────────────────────────────
# The caller holds Invoke on the credential (it releases no key); the persona's own principal
# holds Read on its credential, granted by the collection's Admin when it is shared with the
# persona. Mantle checks both by grant alone: it acts as no persona, mints no delegation for one,
# and opens no key, the card's included.

def _shared_with_lumen(db, key):
    _grant(db, key, "agience-server-lumen", {"read"})


async def test_without_the_persona_s_own_read_nothing_is_released(db):
    key, conn, _, _ = await _shared(db, persona_read=False)
    with pytest.raises(HTTPException) as e:
        await _use(db, _as(TEAMMATE, **LUMEN), key, conn)
    assert e.value.status_code == 403 and "Read" in e.value.detail
    _shared_with_lumen(db, key)
    out = await _use(db, _as(TEAMMATE, **LUMEN), key, conn)
    assert _open(out, key, conn) == "sk-team-secret"


async def test_use_acts_as_no_persona_mints_nothing_and_opens_no_key(db, monkeypatch):
    from mantle.search.mantle import threshold_custody, wiring
    from mantle.services import acting_principal, content_crypto

    key, conn, _, _ = await _shared(db)
    oracle = wiring._build_oracle()
    calls = []

    def refuse(name):
        def _f(*a, **k):
            calls.append(name)
            raise AssertionError(f"{name} was called on the use path")
        return _f
    for name in ("get_or_create_master_key", "get_or_create_collection_key"):
        monkeypatch.setattr(oracle, name, refuse(name))
    monkeypatch.setattr(content_crypto, "decrypt_content", refuse("decrypt_content"))
    monkeypatch.setattr(threshold_custody, "acting_credential", refuse("acting_credential"))
    real_as = acting_principal.acting_as

    def watch_as(principal_id, *a, **k):
        if str(principal_id).startswith("agience-server-"):
            calls.append(f"acting_as {principal_id}")
        return real_as(principal_id, *a, **k)
    monkeypatch.setattr(acting_principal, "acting_as", watch_as)
    out = await _use(db, _as(TEAMMATE, **LUMEN), key, conn)
    assert calls == [] and out["sealed"]


async def test_the_invoke_check_stays_on_the_caller(db):
    key, conn, _, _ = await _shared(db)
    with pytest.raises(HTTPException) as e:
        await _use(db, _as(STRANGER, **LUMEN), key, conn)       # Lumen's Read does not suffice
    assert e.value.status_code in (403, 404)


# -- review 10, R1: a credential names where it may be sent --------------------------------------

API = "https://api.anthropic.com"


async def test_a_credential_seals_the_endpoints_it_names_and_use_returns_them(db):
    """The endpoints a credential may be sent to are its own, fixed by its writer (who holds Admin
    on it by creation), sealed to the persona with the secret, and returned beside the copy."""
    with acting_as(USER, principal_type="user"):
        shelf = ws.create_container(db, USER, content_type=COLLECTION, name="Model connections").id
    key = await _create(db, container_id=shelf, content_type=CREDENTIAL, content="sk-team-secret",
                        context=json.dumps({"title": "Model key", "allowed_endpoints": [API]}))
    conn = await _create(db, container_id=shelf, content_type=CONNECTION, context=json.dumps(
        {"title": "Team model", "provider": "anthropic", "credentials_ref": {"secret_id": key["id"]}}))
    _grant(db, shelf, TEAMMATE, {"invoke"})
    _grant(db, key["id"], "agience-server-lumen", {"read"})
    out = await _use(db, _as(TEAMMATE, **LUMEN), key["id"], conn["id"])
    assert out["endpoints"] == [API]
    sealed = json.loads(_open_raw(out, key["id"], conn["id"]))
    assert sealed == {"material": "sk-team-secret", "endpoints": [API]}


@pytest.mark.parametrize("bad", [["http://api.anthropic.com"], ["ftp://x"], ["not a url"],
                                 [API + "?x=1"], "https://one.example"])
async def test_a_credential_names_only_https_endpoints_as_a_list(db, bad):
    with acting_as(USER, principal_type="user"):
        shelf = ws.create_container(db, USER, content_type=COLLECTION, name="Keys").id
    with pytest.raises(HTTPException) as refused:
        await _create(db, container_id=shelf, content_type=CREDENTIAL, content="sk",
                      context=json.dumps({"title": "k", "allowed_endpoints": bad}))
    assert refused.value.status_code == 400


async def test_a_credential_that_names_no_endpoint_is_sealed_with_none(db):
    key, conn, _, _ = await _shared(db)
    out = await _use(db, _as(TEAMMATE, **LUMEN), key, conn)
    assert out["endpoints"] == []
    assert json.loads(_open_raw(out, key, conn)) == {"material": "sk-team-secret", "endpoints": []}


async def test_editing_a_credential_s_context_never_moves_its_endpoints(db):
    """The binding is fixed at writing: a later edit of the credential's context changes neither
    the sealed list nor the one `/use` returns."""
    from mantle.db import lattice_api as la

    with acting_as(USER, principal_type="user"):
        shelf = ws.create_container(db, USER, content_type=COLLECTION, name="Keys").id
    key = await _create(db, container_id=shelf, content_type=CREDENTIAL, content="sk",
                        context=json.dumps({"title": "k", "allowed_endpoints": [API]}))
    doc = la.get_artifact(db, key["id"])
    assert doc.credential_endpoints == [API]
    with acting_as(USER, principal_type="user"):
        doc.context = json.dumps({"title": "k", "allowed_endpoints": ["https://attacker.example"]})
        la.update_artifact(db, doc)
    assert la.get_artifact(db, key["id"]).credential_endpoints == [API]
