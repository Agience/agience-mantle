import pytest
from fastapi.routing import APIRoute

from mantle.main import app
from mantle.services.dependencies import get_auth


def _iter_api_routes(container, _seen=None):
    """Yield every mounted APIRoute, descending included routers.

    FastAPI >= 0.139 includes routers lazily: ``app.routes`` holds ``_IncludedRouter``
    marker objects, not the flattened routes, so a plain ``isinstance(r, APIRoute)``
    scan of ``app.routes`` misses everything mounted via ``include_router``. We descend
    the markers (their source router's routes already carry full, prefixed paths).
    """
    _seen = _seen if _seen is not None else set()
    for route in getattr(container, "routes", []):
        if isinstance(route, APIRoute):
            yield route
        else:
            inner = getattr(route, "original_router", None) or getattr(route, "router", None)
            if inner is not None and id(inner) not in _seen:
                _seen.add(id(inner))
                yield from _iter_api_routes(inner, _seen)


def _get_route(path: str, method: str) -> APIRoute:
    method = method.upper()
    for route in _iter_api_routes(app):
        if route.path == path and method in route.methods:
            return route
    raise AssertionError(f"Route not found for {method} {path}")


def _route_dependency_calls(route: APIRoute):
    return {dep.call for dep in route.dependant.dependencies if dep.call is not None}


# =============================================================================
# The policy is INVERTED: auth is the default, and every exception is named here.
#
# An enumerated list of protected routes is uncoverable by construction — a route added
# tomorrow is simply absent from it, and absence is indistinguishable from "not critical
# yet". That is exactly how all 16 `/grants` operations came to be unasserted: the list
# said they belonged to another repo's tests, and `agience-origin/src` has no grants
# surface at all, so the check ran nowhere.
#
# Here, a new route is covered the day it is mounted. Making one PUBLIC requires adding it
# below with a reason — a visible, reviewable change, which is precisely the change that
# should be hard to make by accident.
# =============================================================================

PUBLIC = {
    ("GET", "/"):                                   "service root",
    ("GET", "/status"):                             "liveness; no store access",
    ("GET", "/version"):                            "build identity; no store access",
    ("GET", "/.well-known/agience-source"):         "source disclosure document",
    ("GET", "/.well-known/oauth-protected-resource"):
        "RFC 9728 resource metadata — MUST be reachable unauthenticated, it is how a "
        "client discovers WHICH authority to get a token from",
    ("GET", "/auth/callback"):                      "the OAuth redirect target: pre-auth by definition",
    ("GET", "/mcp"):                                "405 with `Allow: POST`; serves no data",
    ("GET", "/mcp/"):                               "as above, trailing-slash form",
    ("GET", "/grants/invite-context"):
        "an invite link is followed BEFORE the recipient has an account. Safe because it "
        "discloses no identity: it returns `has_target=bool(...)`, a boolean over the "
        "target and never the target, while `resource_id`/`granted_by`/`name` are returned "
        "only by `/grants/invite-details`, which is authenticated",
}


def _all_operations():
    """Every (method, path) mounted on the app, HEAD/OPTIONS excluded."""
    out = []
    for route in _iter_api_routes(app):
        for method in sorted((route.methods or set()) - {"HEAD", "OPTIONS"}):
            out.append((method, route.path))
    return sorted(set(out))


@pytest.mark.parametrize("method,path", _all_operations())
def test_every_route_requires_auth_unless_declared_public(method, path):
    """Auth is the default. A route is exempt only by appearing in `PUBLIC` with a reason."""
    if (method, path) in PUBLIC:
        pytest.skip("declared public: %s" % PUBLIC[(method, path)])
    route = _get_route(path, method)
    assert get_auth in _route_dependency_calls(route), (
        "%s %s does not depend on get_auth. If it is meant to be reachable without "
        "authentication, add it to PUBLIC in this file WITH A REASON — do not delete this "
        "assertion." % (method, path)
    )


@pytest.mark.parametrize("method,path", sorted(PUBLIC))
def test_the_public_allowlist_is_not_stale(method, path):
    """Every exemption must still be real, and must still be an exemption.

    Two ways an allowlist rots, both silent:

      * the route is DELETED or renamed — the entry then documents a policy for something
        that does not exist, and the next reader trusts it;
      * the route GAINS `get_auth` — the entry now grants an exemption nobody is using,
        and the day someone removes the dependency again, nothing complains.

    Asserting both keeps the list honest in the direction that matters: it can only shrink
    by someone noticing.
    """
    route = _get_route(path, method)          # raises if the route is gone
    assert get_auth not in _route_dependency_calls(route), (
        "%s %s is listed in PUBLIC but now DEPENDS on get_auth. Remove it from PUBLIC — "
        "the route is protected, and leaving the entry preserves a stale exemption." % (method, path)
    )
