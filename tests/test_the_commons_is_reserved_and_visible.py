"""What the mesh publishes, the API must admit — and nobody may authenticate as the commons.

`db/access.py` describes publication: *"Public is either 'born public' ... OR 'made public' by
granting THIS reserved principal Read on the resource — no copy, no re-key: the same grant
mechanism, with `PUBLIC_PRINCIPAL` as grantee, is what turns a private thing public."* Two things
were wrong with how that stood.

**The publication was invisible.** `mesh/sync._withheld_lattice` stops withholding a resource the
moment `PUBLIC_PRINCIPAL` holds a Read grant reaching it — that is how made-public replicates to
every peer. `services.dependencies.check_access` never consulted publicity, so the resource could
be meshing out while every API read of it returned 404. Publication that is real and unobservable
from the only surface an operator watches is exfiltration whether or not anyone meant it.

**The reservation was not enforced.** "THIS reserved principal" is the literal string `public`,
and `user_id` on the default JWT path is `str(payload.get("sub"))` — straight from the token. A
token with `sub: "public"` therefore produced a caller whose grant lookups returned the COMMONS'
grants as its own, because `_check_grants` asks the ledger by `grantee_id=auth.user_id` and every
publication is a grant to exactly that id.

Two decisions are recorded here because both were live options and the code cannot show which was
chosen:

* The API was widened to agree with the mesh, rather than the mesh narrowed to agree with the API.
  Publication is a documented feature, so making it visible was the fix, not removing it.
* It honours **made public only**, never `is_public`'s "born public" half. See
  `test_an_ungated_artifact_is_still_private` for why that distinction is the whole safety of it.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from mantle.db import access, lattice_api
from mantle.entities.artifact import Artifact
from mantle.entities.grant import Grant
from mantle.services.dependencies import AuthContext, check_access


OWNER = "u-owner"
STRANGER = "u-stranger"
COLLECTION = "col-1"
ARTIFACT = "art-1"


@pytest.fixture
def store(tmp_path):
    db = lattice_api.LatticeDatabase(str(tmp_path / "commons.db"), origin="node-a")
    lattice_api.create_artifact(db, Artifact(
        id=COLLECTION, root_id=COLLECTION, collection_id="", name="collection", content="",
        created_by=OWNER, modified_by=OWNER))
    lattice_api.create_artifact(db, Artifact(
        id=ARTIFACT, root_id=ARTIFACT, collection_id=COLLECTION, name="memo", content="",
        created_by=OWNER, modified_by=OWNER))
    lattice_api.add_artifact_to_collection(db, COLLECTION, ARTIFACT)
    return db


def _auth(user_id=STRANGER):
    return AuthContext(principal_id=user_id, principal_type="user", user_id=user_id)


def _grant(store, gid, resource_id, *, grantee, effect="allow", **flags):
    bits = {"can_read": True}
    bits.update(flags)
    return lattice_api.create_grant(store, Grant(
        id=gid, resource_id=resource_id, grantee_type="user", grantee_id=grantee,
        granted_by=OWNER, effect=effect, state="active", **bits))


def _owner_grant(store):
    """Somebody administers this collection, which is what makes it gated rather than public top."""
    _grant(store, "g-owner", COLLECTION, grantee=OWNER, can_admin=True)


def _refused(store, auth, artifact_id=ARTIFACT, action="read"):
    try:
        check_access(auth, artifact_id, action, store)
        return False
    except HTTPException as exc:
        assert exc.status_code == 404, f"expected a 404, got {exc.status_code}"
        return True


# ── the API admits what the mesh publishes ───────────────────────────────────────────────────


def test_a_stranger_is_refused_a_private_artifact(store):
    """The control. Without it, every test below could pass over a function that allows everything."""
    _owner_grant(store)
    assert _refused(store, _auth())


def test_a_grant_to_the_commons_makes_it_readable_through_the_api(store):
    """The fix. A Read grant to `PUBLIC_PRINCIPAL` on the collection publishes its members, and the
    API now answers the same way the mesh already did."""
    _owner_grant(store)
    assert _refused(store, _auth())

    _grant(store, "g-public", COLLECTION, grantee=access.PUBLIC_PRINCIPAL)
    grant = check_access(_auth(), ARTIFACT, "read", store)
    assert grant is not None
    assert grant.grantee_id == access.PUBLIC_PRINCIPAL, (
        "the returned grant should name the commons, so an audit line says WHY this was allowed"
    )


def test_the_api_and_the_mesh_read_the_same_function(store):
    """Not "agree by inspection" — the same call.

    The two deciders disagreed because each had its own reading of publicity. `is_made_public` is
    the one `check_access` calls, and `reachable_collections(store, PUBLIC_PRINCIPAL)` inside it is
    exactly what `mesh/sync._withheld_lattice` computes to decide what stops being withheld.
    """
    _owner_grant(store)
    doc = lattice_api.get_artifact(store, ARTIFACT).to_dict()
    assert access.is_made_public(store, doc) is False

    _grant(store, "g-public", COLLECTION, grantee=access.PUBLIC_PRINCIPAL)
    assert access.is_made_public(store, doc) is True
    assert COLLECTION in access.reachable_collections(store, access.PUBLIC_PRINCIPAL), (
        "the mesh's publicity input and the API's must be the same set"
    )


def test_a_grant_on_the_artifact_alone_publishes_just_it(store):
    """Publicity follows the same reach as any other grant — a one-artifact share is one artifact."""
    lattice_api.create_artifact(store, Artifact(
        id="art-2", root_id="art-2", collection_id=COLLECTION, name="other", content="",
        created_by=OWNER, modified_by=OWNER))
    lattice_api.add_artifact_to_collection(store, COLLECTION, "art-2")
    _owner_grant(store)

    _grant(store, "g-public", ARTIFACT, grantee=access.PUBLIC_PRINCIPAL)
    assert check_access(_auth(), ARTIFACT, "read", store) is not None
    assert _refused(store, _auth(), "art-2"), (
        "publishing one artifact published its sibling"
    )


# ── the limits of it ─────────────────────────────────────────────────────────────────────────


def test_publicity_grants_read_and_nothing_else(store):
    """A grant to the commons is a READ grant. `check_access` consults publicity only for `read`,
    and the grant it synthesizes carries exactly one bit — so no caller can read more authority out
    of the return value than the commons was actually given."""
    _owner_grant(store)
    _grant(store, "g-public", COLLECTION, grantee=access.PUBLIC_PRINCIPAL)

    grant = check_access(_auth(), ARTIFACT, "read", store)
    assert grant.can_read is True
    for flag in ("can_create", "can_update", "can_delete", "can_admin", "can_share", "can_add"):
        assert getattr(grant, flag) is False, f"the commons grant carried {flag}"

    for action in ("update", "delete", "admin", "share"):
        assert _refused(store, _auth(), ARTIFACT, action), (
            f"a public artifact accepted {action} from a stranger"
        )


def test_a_deny_naming_the_caller_still_wins_over_publicity(store):
    """The placement is the design of the check.

    Publicity is consulted last, after the direct grant, the root grant and the light-cone walk —
    every one of which raises on a deny naming this principal. So a public resource remains
    revokable for a named principal. Consulting publicity first would have made publication
    override deny, undoing exactly what C1 closed.
    """
    _owner_grant(store)
    _grant(store, "g-public", COLLECTION, grantee=access.PUBLIC_PRINCIPAL)
    assert check_access(_auth(), ARTIFACT, "read", store) is not None

    _grant(store, "g-deny", ARTIFACT, grantee=STRANGER, effect="deny")
    assert _refused(store, _auth()), (
        "a deny naming this principal was overridden by the resource being public"
    )


def test_an_ungated_artifact_is_still_private(store):
    """`is_public` has a second half that the API does not use.

    `is_public` also answers True for **born public**: ungrounded, or grounded in a collection no
    grant gates. That is right for the ember/shard model this module was written for, where an
    ungated collection IS the un-keyed top. It is catastrophic here: "no grant yet" is the ordinary
    state of a private thing, and `lattice_api._stamp_origin_root` deliberately leaves the field
    unset rather than guess a principal, so an unstampable artifact would have become world-readable
    too.

    This collection has NO grants at all, so `is_public` says True and the API must still say no.
    """
    doc = lattice_api.get_artifact(store, ARTIFACT).to_dict()
    assert access.is_public(store, doc) is True, (
        "fixture assumption: an ungated collection reads as born-public"
    )
    assert access.is_made_public(store, doc) is False
    assert _refused(store, _auth()), (
        "the API adopted born-public and now serves every artifact whose collection has no grants"
    )


def test_a_nonexistent_artifact_is_still_a_404(store):
    """Publicity must not become an existence oracle. An id that does not exist answers exactly the
    way a refused one does — the property `check_access` holds throughout."""
    _owner_grant(store)
    _grant(store, "g-public", COLLECTION, grantee=access.PUBLIC_PRINCIPAL)
    assert _refused(store, _auth(), "no-such-artifact")


# ── the reservation ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("field", ["principal_id", "user_id"])
def test_nobody_authenticates_as_the_commons(field):
    """The commons is a grantEE, never a caller.

    Enforced in `AuthContext.__post_init__` rather than at the five construction sites in
    `get_auth`, so it covers all five and the sixth that gets added later. 401, not 403: the token
    does not name a principal that may act at all.
    """
    kwargs = {"principal_id": "u-1", "principal_type": "user", "user_id": "u-1"}
    kwargs[field] = access.PUBLIC_PRINCIPAL
    with pytest.raises(HTTPException) as caught:
        AuthContext(**kwargs)
    assert caught.value.status_code == 401
    assert "commons" in str(caught.value.detail)


def test_an_ordinary_principal_still_authenticates():
    """The guard must be exact. `public` is reserved; nothing that merely resembles it is."""
    for uid in ("u-1", "publicity", "public-relations", "PUBLIC", "the public"):
        auth = AuthContext(principal_id=uid, principal_type="user", user_id=uid)
        assert auth.user_id == uid


def test_granting_to_the_commons_is_still_allowed(store):
    """The reservation is on ACTING, not on being granted to. Refusing the grantee would have
    removed the publication mechanism `db/access.py` documents, which was the option not taken."""
    _owner_grant(store)
    grant = _grant(store, "g-public", COLLECTION, grantee=access.PUBLIC_PRINCIPAL)
    assert grant is not None
    assert COLLECTION in access.reachable_collections(store, access.PUBLIC_PRINCIPAL)


def test_the_two_enforcement_points_name_the_same_id():
    """Two places enforce the reservation — the auth guard and the publicity check — and a
    reservation whose two halves named different strings would mean nothing."""
    from mantle.services import dependencies

    assert dependencies._PUBLIC_PRINCIPAL is access.PUBLIC_PRINCIPAL
