"""Review 11: narrowed delegations are recognised from Origin's claims and fail closed, and Ember
makes no grant at all (John, 2026-11-02: sharing happens only in the person's own session).

A narrowed delegation is one Origin's token exchange issued: a delegation whose `aud` equals its
`act.sub` and names a narrow audience (`MANTLE_NARROW_AUDIENCES`, default Ember and the personas),
or one carrying a scope that is not a platform purpose. It reaches only its scope's routes; an
audience or scope Mantle does not know reaches nothing.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException

from mantle.services import dependencies as D
from mantle.services.artifact_identity import derive_artifact_id


def _req(method, path):
    return NS(method=method, url=NS(path=path))


def _auth(aud, scope, *, act=None, user="p"):
    return D.AuthContext(principal_id=user, principal_type="user", user_id=user,
                         actor=act if act is not None else aud, delegation_aud=aud,
                         scope=scope or None)


def _reach(auth, method, path):
    try:
        D.enforce_narrowed_bounds(auth, _req(method, path))
        return True
    except HTTPException:
        return False


def test_ember_reaches_its_memory_create_and_verdict_only():
    a = _auth("ember", "ember:memory ember:read")
    assert _reach(a, "POST", "/artifacts")
    assert _reach(a, "GET", "/grants/my-access")
    for m, p in [("GET", "/artifacts/abc"), ("GET", "/artifacts/visible"),
                 ("POST", "/artifacts/abc/use"), ("POST", "/grants"), ("DELETE", "/artifacts/x")]:
        assert not _reach(a, m, p), (m, p)


def test_ember_share_is_gone():
    assert "ember:share" not in D.NARROWED_SCOPE_ROUTES
    assert not _reach(_auth("ember", "ember:share"), "POST", "/grants")


def test_an_audience_mantle_does_not_know_reaches_nothing():
    a = _auth("crm-bridge", "crm:read")
    assert D.is_narrowed(a)
    assert not _reach(a, "GET", "/artifacts/x") and not _reach(a, "DELETE", "/artifacts/x")


def test_a_persona_narrowed_delegation_reaches_only_the_operators_verdict():
    """It reaches no content and no write. It does reach the platform operators' verdict, which
    Crystal's operator gate asks with it before every persona tool call: refused, every persona
    tool call answered 503 on production (2026-11-10)."""
    for name in ("iris", "lumen", "sage", "crystal"):
        a = _auth(name, f"persona:{name}")
        assert D.is_narrowed(a)
        assert _reach(a, "GET", "/artifacts/platform-operators"), name
        for m, p in [("GET", "/artifacts/x"), ("GET", "/artifacts/visible"), ("POST", "/artifacts"),
                     ("POST", "/artifacts/platform-operators"), ("GET", "/grants/my-access"),
                     ("POST", "/grants")]:
            assert not _reach(a, m, p), (name, m, p)
    # A scope counts only for its own audience: Lumen's scope on a delegation to Sage reaches nothing.
    assert not _reach(_auth("sage", "persona:lumen"), "GET", "/artifacts/platform-operators")


def test_recognised_from_claims_not_from_the_actor_name():
    # A narrowed claim set with an actor that looks like a persona server is still narrowed...
    assert D.is_narrowed(_auth("ember", "ember:memory", act="ember"))
    # ...and a persona server's own delegation (aud = its client id, no scope) is not.
    server = _auth("agience-server-lumen", None)
    assert not D.is_narrowed(server) and _reach(server, "GET", "/artifacts/x")
    # An actor named "ember" on an ordinary delegation addressed to a server is not narrowed by
    # its name alone.
    assert not D.is_narrowed(_auth("agience-server-lumen", None, act="ember"))


def test_a_malformed_narrow_audience_setting_fails_closed(monkeypatch):
    monkeypatch.setenv("MANTLE_NARROW_AUDIENCES", "{not json")
    server = _auth("agience-server-lumen", None)
    assert D.is_narrowed(server) and not _reach(server, "GET", "/artifacts/x")


def test_ember_creates_only_plain_memories_in_its_own_collection():
    person = str(uuid.uuid4())
    mine = derive_artifact_id(person, "ember:memory")
    a = _auth("ember", "ember:memory", user=person)
    D.ember_create_allowed(a, {"container_id": mine, "content": "x"})
    D.ember_create_allowed(a, {"container_id": mine, "content": "x", "content_type": "text/markdown"})
    for body in ({"container_id": mine, "content_type": "application/vnd.agience.collection+json"},
                 {"container_id": mine, "content_type": "application/vnd.agience.credential+json",
                  "context": {"allowed_endpoints": ["https://evil.test/"]}, "content": "x"},
                 {"identity": "ember:memory"}, {"container_id": str(uuid.uuid4())}):
        with pytest.raises(HTTPException):
            D.ember_create_allowed(a, body)


def test_grants_route_refuses_any_ember_delegation():
    import inspect

    from mantle.routers import grants_router
    assert not hasattr(grants_router, "_ember_share_bounds")
    assert "EMBER_SHARE_MAX_DAYS" not in inspect.getsource(grants_router)


# -- M2: Origin's sign-in bounds ------------------------------------------------------------------

def test_a_token_from_before_its_person_s_sign_in_bound_is_refused(monkeypatch):
    from prism.trust import not_before

    nb = not_before.NotBefore("http://origin.test")
    nb.apply({"entries": [{"sub": "alice", "not_before": 1000}]})
    monkeypatch.setattr(not_before, "_shared", nb)
    monkeypatch.setattr(D, "verify_token", lambda t: {"sub": "alice", "iat": 999,
                                                      "aud": D.config.AUTHORITY_ISSUER,
                                                      "iss": D.config.AUTHORITY_ISSUER})
    with pytest.raises(HTTPException) as refused:
        D.resolve_auth(token="a.b.c", store_db=None)
    assert refused.value.status_code == 401
    # The person's token as the web app sends it, addressed to Mantle (review 23).
    monkeypatch.setattr(D, "verify_token", lambda t: {"sub": "alice", "iat": 1001,
                                                      "aud": "mantle",
                                                      "iss": D.config.AUTHORITY_ISSUER})
    assert D.resolve_auth(token="a.b.c", store_db=None).user_id == "alice"


def test_a_moved_bound_forgets_that_person_s_origin_verdicts(monkeypatch):
    from prism.trust import not_before

    from mantle.services import authority_ledger as L

    L._verdicts[(("alice", "user"), "c", "r", "read")] = 10**12
    L._verdicts[(("bob", "user"), "c", "r", "read")] = 10**12
    nb = not_before.NotBefore("http://origin.test")
    monkeypatch.setattr(not_before, "_shared", nb)
    L.listen_for_sign_in_bounds()
    nb.apply({"entries": [{"sub": "alice", "not_before": 5}]})
    assert (("alice", "user"), "c", "r", "read") not in L._verdicts
    assert (("bob", "user"), "c", "r", "read") in L._verdicts
    L.forget_verdicts()


# -- the memory collection, for every person ------------------------------------------------------

def test_the_memory_collection_is_made_for_a_person_who_already_holds_grants(monkeypatch):
    from mantle.services.seed_provisioning import user_provisioning as up

    made = []
    monkeypatch.setattr(up, "_ensure_ember_memory", lambda db, uid: made.append(uid) or "m")
    person = D.AuthContext(principal_id="alice", principal_type="user", user_id="alice")
    mine = derive_artifact_id("alice", "ember:memory")
    assert up.ensure_ember_memory_for(None, person, {"some-grant-target"}) == "m"
    assert made == ["alice"]
    assert up.ensure_ember_memory_for(None, person, {mine}) is None          # already there
    ember = _auth("ember", "ember:memory", user="alice")
    assert up.ensure_ember_memory_for(None, ember, set()) is None            # never on Ember's
    assert made == ["alice"]


def test_the_memory_collection_is_never_made_on_a_narrowed_delegation(monkeypatch):
    from mantle.services.acting_principal import acting_as
    from mantle.services.seed_provisioning import user_provisioning as up

    from mantle.services import workspace_service

    made = []
    monkeypatch.setattr(up, "db_get_artifact", lambda db, i: None)
    monkeypatch.setattr(workspace_service, "create_container",
                        lambda *a, **k: made.append(k.get("artifact_id")))
    with acting_as("alice", principal_type="user", narrowed="ember"):
        assert up._ensure_ember_memory(object(), "alice") is None
    assert made == []
    with acting_as("alice", principal_type="user"):
        up._ensure_ember_memory(object(), "alice")
    assert made == [derive_artifact_id("alice", "ember:memory")]
