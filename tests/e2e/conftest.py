"""Shared fixtures for the blackbox E2E suite.

Design:
- The suite runs against a LIVE stack. If Origin/Mantle are unreachable, every
  test is skipped (not failed) with a clear reason.
- `operator` is idempotent across reruns: it claims the single-use bootstrap
  token on a fresh stack, and falls back to logging in with the SAME deterministic
  credentials if the bootstrap was already consumed by an earlier run.
- `user_factory` registers unique users on demand; `second_issuer` stands up a
  self-signed IdP and registers it as an admin so external-tenant tokens verify.
"""
from __future__ import annotations

import uuid

import httpx
import pytest

import _config as cfg
from _api import Api, bootstrap_claim, login, register, sub_of
from _issuer import SelfSignedIssuer

# Deterministic operator identity so a claimed-then-rerun stack still logs in.
_OPERATOR_EMAIL = "e2e-operator@agience.test"
_OPERATOR_PASSWORD = "e2e-operator-pw-123456"
_OPERATOR_NAME = "E2E Operator"


def _reachable(url: str) -> bool:
    try:
        r = httpx.get(f"{url}/openapi.json", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def require_stack():
    """Skip the whole suite unless both services answer."""
    missing = [name for name, url in (("origin", cfg.ORIGIN_URL), ("mantle", cfg.MANTLE_URL))
               if not _reachable(url)]
    if missing:
        pytest.skip(f"stack not reachable: {', '.join(missing)} "
                    f"(origin={cfg.ORIGIN_URL} mantle={cfg.MANTLE_URL})")


@pytest.fixture(scope="session")
def operator() -> dict:
    """The platform operator (admin). Returns {token, person_id, email, password}."""
    tok = cfg.bootstrap_token()
    if tok:
        try:
            claimed = bootstrap_claim(tok, email=_OPERATOR_EMAIL, name=_OPERATOR_NAME,
                                      password=_OPERATOR_PASSWORD)
            return {"token": claimed["access_token"], "person_id": claimed["person_id"],
                    "email": _OPERATOR_EMAIL, "password": _OPERATOR_PASSWORD}
        except httpx.HTTPStatusError as e:
            # 410 = already claimed; fall through to login with the same creds.
            if e.response.status_code != 410:
                raise
    # Bootstrap unavailable/consumed — log in with the deterministic operator.
    try:
        data = login(_OPERATOR_EMAIL, _OPERATOR_PASSWORD)
    except httpx.HTTPStatusError:
        pytest.skip("no bootstrap token and operator login failed — start from a "
                    "fresh .data-local or set E2E_BOOTSTRAP_TOKEN")
    return {"token": data["access_token"], "person_id": sub_of(data["access_token"]),
            "email": _OPERATOR_EMAIL, "password": _OPERATOR_PASSWORD}


@pytest.fixture(scope="session")
def operator_api(operator) -> Api:
    return Api(operator["token"])


@pytest.fixture
def user_factory():
    """Register + log in fresh users on demand.

    Returns a callable -> {token, person_id, username, password, api}.
    """
    created: list[Api] = []

    def make(prefix: str = "user") -> dict:
        uname = f"{prefix}-{uuid.uuid4().hex[:10]}"
        pw = "user-pw-" + uuid.uuid4().hex[:12]
        r = register(uname, pw, name=uname, email=f"{uname}@agience.test")
        r.raise_for_status()
        data = login(uname, pw)
        api = Api(data["access_token"])
        created.append(api)
        return {"token": data["access_token"], "person_id": sub_of(data["access_token"]),
                "username": uname, "password": pw, "api": api}

    yield make
    for a in created:
        a.close()


@pytest.fixture
def user(user_factory) -> dict:
    return user_factory("user")


@pytest.fixture(scope="session")
def second_issuer(operator_api) -> SelfSignedIssuer:
    """Stand up a self-signed external IdP and register it (admin) so Mantle
    trusts its tokens. Namespaced -> a distinct tenant from Origin users.

    The exact registration endpoint/body is asserted by test_02; this fixture
    performs the registration and skips dependent tests if it is unavailable.
    """
    iss = SelfSignedIssuer.create(
        issuer="https://e2e-idp.agience.test/",
        audience="e2e-external-client",
        namespace="e2e-tenant-b",
    )
    body = {
        "issuer": iss.issuer,
        "audience": iss.audience,
        "jwks": iss.jwks(),
        "namespace": iss.namespace,
        "role": "external",
    }
    r = operator_api.post("/system/issuers", on="mantle", json=body)
    if r.status_code not in (200, 201):
        pytest.skip(f"could not register second issuer (POST /system/issuers -> {r.status_code}: {r.text[:200]})")
    return iss
