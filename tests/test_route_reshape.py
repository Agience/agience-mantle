"""The reshaped route table — and the old paths that must NOT answer.

The reshape was one breaking window with no compatibility shims: every consumer is
in-house, so an old path 404s rather than redirecting. A redirect would be worse than a
404 here — it would keep a stale client working while its author believes they migrated,
and the migration would then be discovered by whoever removed the redirect months later.

Both halves are asserted, because only one of them is self-evident. That `/system/users`
exists is visible in the router; that `/platform/users` is *gone* is only visible if
something checks, and a stray re-mount is exactly the mistake this file catches.

The admin surface additionally has to keep its gate. Merging two routers into one
namespace is a refactor whose failure mode is an endpoint that quietly loses
``require_platform_admin`` on the way across, so every `/system` route is checked for it
by name rather than by spot-check.
"""
from __future__ import annotations

import inspect

import pytest
from fastapi.routing import APIRoute

from mantle.main import app
from mantle.services.dependencies import require_platform_admin


def _routes():
    seen: set = set()

    def walk(container):
        for route in getattr(container, "routes", []):
            if isinstance(route, APIRoute):
                yield route
            else:
                inner = getattr(route, "original_router", None) or getattr(route, "router", None)
                if inner is not None and id(inner) not in seen:
                    seen.add(id(inner))
                    yield from walk(inner)

    return list(walk(app))


_PATHS = {(m, r.path) for r in _routes() for m in r.methods}


# ---------------------------------------------------------------------------
# What the surface is now
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,path", [
    ("POST", "/artifacts/recall"),
    ("GET", "/system/issuers"),
    ("POST", "/system/issuers"),
    ("DELETE", "/system/issuers/{artifact_id}"),
    ("GET", "/system/users"),
    ("POST", "/system/seed"),
    ("POST", "/system/users/{user_id}/grant-admin"),
    ("DELETE", "/system/users/{user_id}/revoke-admin"),
    ("POST", "/system/erasure/{person_id}"),
])
def test_the_reshaped_route_is_served(method, path):
    assert (method, path) in _PATHS, f"{method} {path} is not mounted"


# ---------------------------------------------------------------------------
# What must not answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix", ["/artifacts/search", "/issuers", "/platform", "/servers"])
def test_the_old_path_is_gone_entirely(prefix):
    """404, not 301. Every consumer is in-house; a redirect would let a stale client keep
    working while its author believes it was migrated."""
    stale = sorted(p for _, p in _PATHS if p == prefix or p.startswith(prefix + "/"))
    assert not stale, f"{prefix} still answers at {stale}"


def test_the_retired_routers_are_off_disk():
    """A router module that mounts nothing is a claim about the API that is not true.
    `test_every_router_is_mounted` catches the unmounted case; this catches the leftover
    file."""
    import pathlib

    routers = pathlib.Path(__file__).resolve().parents[1] / "src" / "mantle" / "routers"
    for stem in ("issuers_router", "platform_router", "servers_router"):
        assert not (routers / f"{stem}.py").exists(), f"{stem}.py survived the merge"


# ---------------------------------------------------------------------------
# The gate came across intact
# ---------------------------------------------------------------------------

def test_every_system_route_requires_platform_admin():
    """One namespace, one predicate. An endpoint that lost its gate in the merge would be
    an admin operation open to any authenticated caller."""
    system_routes = [r for r in _routes() if r.path.startswith("/system")]
    assert system_routes, "no /system routes found — the walker is wrong"
    for route in system_routes:
        src = inspect.getsource(route.endpoint)
        assert "require_platform_admin" in src, (
            f"{sorted(route.methods)} {route.path} does not call require_platform_admin"
        )


def test_the_gate_is_the_shared_one():
    """Imported from `services.dependencies`, not redefined — a second definition could
    disagree with the one `GET /system/users` reports admin status with."""
    from mantle.routers import system_router

    assert system_router.require_platform_admin is require_platform_admin


# ---------------------------------------------------------------------------
# Pagination (Phase 9's router-level items)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,path", [
    ("GET", "/artifacts/visible"),
    ("GET", "/artifacts/{artifact_id}/children"),
    ("GET", "/system/users"),
    ("GET", "/artifacts/{artifact_id}/access-log"),
])
def test_the_unbounded_listings_now_take_a_page(method, path):
    """An unbounded listing is work proportional to how much exists, not to what was
    asked for. `/access-log` already had the shape; the other three copy it."""
    route = next(r for r in _routes() if r.path == path and method in r.methods)
    params = {p.name: p for p in route.dependant.query_params}
    assert "limit" in params and "offset" in params, f"{path} takes no page"
    assert params["limit"].default == 100
    assert params["offset"].default == 0


def test_visible_hydrates_the_page_and_not_the_whole_light_cone():
    """The N+1 this replaces loaded one artifact per authorized id, so the cost scaled with
    the caller's entire grant reach. The batch loader must see only the page."""
    src = inspect.getsource(
        __import__("mantle.routers.artifacts_router", fromlist=["x"]).list_visible
    )
    assert "_hydrate_batch, store_db, page" in src, "the batch loader no longer sees the page"
    assert "offload_sync(_hydrate_batch" in src, (
        "the page load is back on the event loop — it is the largest block of blocking work "
        "in this router"
    )
    assert "_find_artifact" not in src, "the per-id loader is back in the visible listing"
