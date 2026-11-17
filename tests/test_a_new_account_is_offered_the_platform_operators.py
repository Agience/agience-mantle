"""A new account's workflow builder offers every operator it may invoke, the platform's included.

BUG-091 (71/test 202611.15.1, Tester runs 27 and 28): a new workflow in a fresh workspace offered
no operators and no recipes. The builder lists `/artifacts/visible` for operators the caller may
invoke; its first call is the account's first sign-in.
"""
from __future__ import annotations

from mantle.services.acting_principal import acting_as
from mantle.services.seed_provisioning import platform_operators as P

from .test_platform_operators_are_granted_artifacts import _auth, db  # noqa: F401
from .test_there_is_no_promotion_state import USER

OPERATOR_CT = P.OPERATOR_CT


async def _offered(db, user):
    from mantle.routers import artifacts_router as artifacts
    with acting_as(user, principal_type="user"):
        page = await artifacts.list_visible(content_type=OPERATOR_CT, action="invoke", limit=500,
                                            offset=0, auth=_auth(user), store_db=db)
    return {i.get("root_id") or i.get("id") for i in page["items"]}


def _public():
    return {P.ids()[o["slug"]] for o in P.OPERATORS if o.get("public")}


async def test_a_new_account_is_offered_every_platform_operator_on_its_first_call(db):  # noqa: F811
    P.ensure_platform_operators(db)
    assert await _offered(db, USER) >= _public()


async def test_an_account_provisioned_before_the_operators_is_offered_them_after(db):  # noqa: F811
    await _offered(db, USER)
    P.ensure_platform_operators(db)
    P.grant_to_every_account(db)
    assert await _offered(db, USER) >= _public()
