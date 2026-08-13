"""A grant key is its own principal — it does not borrow its issuer's identity.

`ActingPrincipal.principal_id` is the principal the work is done on behalf of: what the
grant verifier resolves a light cone for, and what `oracle._authorize` compares
`requester_id` against.

A grant key holds CRUDEASIO bits directly, and there is no owner to fall back to — it
is not a scope ceiling over someone else's grants, it is the permission. Resolving it
to its issuer would hand a detached credential that person's entire light cone, which
is precisely the widening the key exists to prevent.

Two consequences this file pins:

  1. The subject is the KEY (its root grant's id), never the issuer.
  2. Because the subject is the key, the light-cone lookup cannot go through
     `ledger_grantee_type` — a `grant_key` grant is stored under the token HASH, not
     under the grant's own id, so that lookup finds nothing. `LightConeResolver`
     resolves the bundle instead. Getting this wrong fails closed (a key that can
     search nothing), which is why it needs a test rather than a bug report.

`test_authz_ceilings.py` holds the permission half; this file holds the identity half.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from mantle.entities.grant import Grant as GrantEntity
from mantle.services.acting_principal import acting_from_auth
from mantle.services.dependencies import AuthContext
from mantle.search.mantle.lightcone import LightConeResolver, ledger_grantee_type

KEY_GRANT_ID = "e779c6f0-f1a9-4333-afd7-1303bea6b1f9"
USER_ID = "e83134ad-27a8-4e87-9efd-62c87192b8a4"


def _key_auth():
    return AuthContext(
        principal_id=KEY_GRANT_ID,
        principal_type="grant_key",
        user_id=None,
        grant_key_id=KEY_GRANT_ID,
    )


def test_a_grant_key_acts_as_itself():
    """The subject is the key's root grant, not whoever minted it."""
    actor = acting_from_auth(_key_auth())

    assert actor.principal_id == KEY_GRANT_ID
    assert actor.principal_type == "grant_key"


def test_a_grant_key_never_resolves_to_its_issuer():
    """Even when an issuer is known, the key must not be promoted to that identity.

    This is the widening the whole design exists to prevent: a leaked key would get
    the issuer's full light cone rather than the slice it was granted.
    """
    auth = AuthContext(
        principal_id=KEY_GRANT_ID,
        principal_type="grant_key",
        user_id=USER_ID,          # an issuer identity present on the context
        grant_key_id=KEY_GRANT_ID,
    )

    actor = acting_from_auth(auth)

    assert actor.principal_id == KEY_GRANT_ID, "the key was promoted to its issuer"
    assert actor.principal_type == "grant_key"


def test_the_light_cone_resolves_a_key_through_its_bundle():
    """The lookup must walk the bundle, not `get_active_grants_for_grantee`.

    A `grant_key` grant's `grantee_id` is the token hash, so looking the acting
    principal's id up under grantee_type="grant_key" returns nothing and the key can
    reach no resource at all.
    """
    root = GrantEntity(
        id=KEY_GRANT_ID, resource_id="", grantee_type="grant_key",
        grantee_id="sha256-of-the-token", granted_by=USER_ID, can_read=True,
    )
    member = GrantEntity(
        id="m-1", resource_id="col-1", grantee_type="grant",
        grantee_id=KEY_GRANT_ID, granted_by=USER_ID, can_read=True,
    )

    def _members(_db, grantee_id, grantee_type="user"):
        return [member] if grantee_type == "grant" and grantee_id == KEY_GRANT_ID else []

    with (
        patch("mantle.db.backend.get_grant_by_id", return_value=root),
        patch("mantle.db.backend.get_active_grants_for_grantee", side_effect=_members),
        patch("mantle.db.backend.list_origin_descendants", return_value=set()),
    ):
        reached = LightConeResolver(db=MagicMock()).resolve(
            KEY_GRANT_ID, "read", principal_type="grant_key")

    assert reached == {"col-1"}, (
        "a grant key reached nothing — the resolver looked its principal id up as a "
        "grantee instead of resolving the bundle hanging off it"
    )


def test_a_revoked_key_reaches_nothing():
    """Positive control for the branch above: it must still respect key state."""
    revoked = GrantEntity(
        id=KEY_GRANT_ID, resource_id="col-1", grantee_type="grant_key",
        grantee_id="hash", granted_by=USER_ID, can_read=True,
        state=GrantEntity.STATE_REVOKED,
    )

    with patch("mantle.db.backend.get_grant_by_id", return_value=revoked):
        reached = LightConeResolver(db=MagicMock()).resolve(
            KEY_GRANT_ID, "read", principal_type="grant_key")

    assert reached == set()


def test_ledger_grantee_type_still_maps_ordinary_principals_to_user():
    """Control: the mapping itself is unchanged for everything that is not a key."""
    assert ledger_grantee_type("user") == "user"
    assert ledger_grantee_type("service") == "user"
    assert ledger_grantee_type("mcp_client") == "user"
    assert ledger_grantee_type("grant_key") == "grant_key", (
        "key-shaped principals stay their own kind — what changed is how a key's "
        "grants are FOUND, not what kind they are"
    )


def test_every_other_principal_kind_is_untouched():
    """The control: a user, a service, or a delegation resolves exactly as before."""
    user = acting_from_auth(
        AuthContext(principal_id=USER_ID, principal_type="user", user_id=USER_ID))
    assert (user.principal_id, user.principal_type, user.actor) == (USER_ID, "user", None)

    svc = acting_from_auth(AuthContext(principal_id="chorus", principal_type="service"))
    assert (svc.principal_id, svc.principal_type) == ("chorus", "service")

    # A delegation already resolves to principal_type="user" with the server in `actor`;
    # removing the api_key arm must not disturb the actor a delegation chain carries.
    dele = acting_from_auth(
        AuthContext(principal_id=USER_ID, principal_type="user", user_id=USER_ID,
                    actor="agience-server-astra"))
    assert dele.actor == "agience-server-astra"
