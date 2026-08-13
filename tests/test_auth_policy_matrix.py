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


@pytest.mark.parametrize(
    "method,path",
    [
        # Unified artifact endpoints (Mantle). Grant endpoints live in Origin;
        # their auth-dependency check belongs in `origin/tests/`.
        ("POST", "/artifacts"),
        ("GET", "/artifacts/{artifact_id}"),
        ("PATCH", "/artifacts/{artifact_id}"),
        ("DELETE", "/artifacts/{artifact_id}"),
        # POST /artifacts/{id}/op/{op_name} (operation dispatch) lives in the gateway
        # (crystal) — its auth check lives in the gateway's tests.
        ("POST", "/artifacts/recall"),
    ],
)
def test_all_critical_routes_use_get_auth(method, path):
    """All critical routes use the unified get_auth() dependency."""
    route = _get_route(path, method)
    calls = _route_dependency_calls(route)
    assert get_auth in calls, f"Expected get_auth on {method} {path}"
