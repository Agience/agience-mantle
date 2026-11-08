"""Round 13 (H1): an invite's raw claim token never reaches Mantle. Origin makes, mails, describes
and claims invites; Mantle has no route that takes a token to read an invite's context or details."""
from __future__ import annotations


def _paths():
    from fastapi.routing import APIRoute

    from mantle.main import app
    return {(m, r.path) for r in app.routes if isinstance(r, APIRoute) for m in r.methods}


def test_mantle_has_no_route_that_reads_an_invite_by_its_token():
    paths = _paths()
    assert not [p for p in paths if p[1] in ("/grants/invite-context", "/grants/invite-details")]


def test_no_mantle_route_takes_a_raw_invite_token_as_a_query():
    """Every route's declared parameters: none is an invite token in the query string."""
    from fastapi.routing import APIRoute

    from mantle.main import app
    bad = []
    for r in app.routes:
        if not isinstance(r, APIRoute):
            continue
        for p in r.dependant.query_params:
            if p.name == "token" and r.path.startswith("/grants"):
                bad.append(r.path)
    assert bad == []
