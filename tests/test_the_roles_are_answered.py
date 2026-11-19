"""Mantle answers its role presets, so no client keeps a copy of them."""
from mantle.attenuation import ACTIONS
from mantle.entities.grant import Grant
from mantle.routers import grants_router


def test_every_preset_is_answered_with_its_actions_in_order():
    got = grants_router.roles()
    assert [r["name"] for r in got] == list(Grant.ROLE_PRESETS)
    for r in got:
        preset = Grant.ROLE_PRESETS[r["name"]]
        assert r["actions"] == [a for a in ACTIONS if preset.get(f"can_{a}")]
    assert next(r for r in got if r["name"] == "viewer")["actions"] == ["read"]


def test_roles_is_declared_before_the_grant_id_route():
    paths = [getattr(r, "path", "") for r in grants_router.router.routes]
    assert paths.index("/grants/roles") < paths.index("/grants/{grant_id}")
