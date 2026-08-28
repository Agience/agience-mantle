"""A light cone too large to materialise must not be read as "this principal has nothing".

THE DEFECT. `list_visible` resolves the caller's read cone and, on `EdgesTruncated`, sets
`read_authorized = set()` so the scanning pager can answer instead. The very next branch is

    if auth.user_id and not read_authorized:      # -> provision_user(...)

which is the FIRST-LOGIN path. `truncated_cone` is set on the line above and is not consulted until
three checks later, so a principal whose cone is merely too dense falls into new-user provisioning:
a multi-write transaction, on a GET, on **every** request. It then re-resolves inside the same
`try`, raises `EdgesTruncated` a second time, and logs *"First-login provisioning failed … "* about
a user who has been provisioned for months.

The two states are opposites. "Too many edges to materialise" and "no grants at all" produce the
same empty set, and only `truncated_cone` distinguishes them. The router's own comment says the two
principals in this state on node 71 "are the two that hold the corpus" — so the cost falls exactly
on the accounts the fallback exists to serve.
"""
from __future__ import annotations

import pytest

from mantle.db.edge import EdgesTruncated


@pytest.fixture
def truncating_cone(monkeypatch):
    """Force the resolver to truncate, and record whether provisioning was attempted."""
    import mantle.search.mantle.lightcone as lc
    import mantle.services.seed_provisioning as sp

    calls = {"provision": 0}

    class _Truncating:
        def __init__(self, *a, **kw):
            pass

        def resolve(self, *a, **kw):
            raise EdgesTruncated("cone too large to materialise")

    def _provision(*a, **kw):
        calls["provision"] += 1
        return None

    monkeypatch.setattr(lc, "LightConeResolver", _Truncating)
    monkeypatch.setattr(lc, "authorized_page", lambda *a, **kw: [])
    monkeypatch.setattr(sp, "provision_user", _provision)
    return calls


@pytest.mark.asyncio
async def test_a_truncated_cone_does_not_trigger_first_login_provisioning(client, truncating_cone):
    r = await client.get("/artifacts/visible")
    assert r.status_code == 200, r.text
    assert truncating_cone["provision"] == 0, (
        "a truncated light cone was treated as a first login and ran %d provisioning "
        "transaction(s) on a GET" % truncating_cone["provision"])


@pytest.mark.asyncio
async def test_the_truncated_path_still_answers(client, truncating_cone):
    """Guard against 'fixing' this by refusing the request: the fallback must still serve."""
    r = await client.get("/artifacts/visible")
    assert r.status_code == 200, r.text


@pytest.fixture
def empty_cone(monkeypatch):
    """A genuinely empty cone — the real first login. Nothing truncates."""
    import mantle.search.mantle.lightcone as lc
    import mantle.services.seed_provisioning as sp

    calls = {"provision": 0}

    class _Empty:
        def __init__(self, *a, **kw):
            pass

        def resolve(self, *a, **kw):
            return set()

    def _provision(*a, **kw):
        calls["provision"] += 1
        return None

    monkeypatch.setattr(lc, "LightConeResolver", _Empty)
    monkeypatch.setattr(lc, "authorized_page", lambda *a, **kw: [])
    monkeypatch.setattr(sp, "provision_user", _provision)
    return calls


@pytest.mark.asyncio
async def test_a_genuinely_empty_cone_still_provisions(client, empty_cone):
    """The guard must be narrow: it excludes truncation, not every empty result.

    Without this, 'fix' the defect by deleting the branch and first login breaks in silence —
    the same class of change this audit keeps finding.
    """
    r = await client.get("/artifacts/visible")
    assert r.status_code == 200, r.text
    assert empty_cone["provision"] == 1, (
        "a real first login (empty cone, no truncation) must still provision; got %d call(s)"
        % empty_cone["provision"])
