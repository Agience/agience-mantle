"""A deny grant denies — at every enforcement point, not just the ones that remembered to ask.

Audit finding **S1** was one expression, `getattr(grant, flag_attr, False)`, answering only half
the authorization question. A grant carries two things: nine CRUDEASIO bits saying *which*
actions it speaks about, and an `effect` saying whether it speaks in favour. A **deny** grant
sets the bits for the actions it denies — that is how `check_access` recognises one — so reading
the bit alone reports every deny grant as an allow.

`mantle.attenuation` made the joined question available as one call (`mask_of(g).allows(action)`;
`propagates(column, action)` for the edge form). This file is the other half of that work: one
test per site that used to ask only half, each one seeding the exact shape — an ACTIVE, UNEXPIRED
grant whose effect is `deny` and whose bit for the action IS set — and asserting the site now
denies.

Every test here fails on the pre-fix code. That is the property that makes them regression tests
rather than descriptions: `tests/test_attenuation_is_single_sourced.py` proves the *expression* is
gone, and this file proves the *behaviour* changed with it. Neither alone is enough — a site could
be re-pointed at the operator and still be asked the wrong question.

Sibling files: `tests/test_attenuation_algebra.py` (the operator's laws and the codec's fidelity
to the stored column) and `tests/test_attenuation_is_single_sourced.py` (the AST sweep).
"""
from __future__ import annotations

import uuid

import pytest

from mantle.attenuation import ACTIONS
from mantle.entities.grant import Grant as GrantEntity, mask_of


# ═════════════════════════════════════════════════════════════════════════════════════════
# 0 · Fixtures — a real lattice, because these are lattice-level reads
# ═════════════════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _test_master_key(monkeypatch):
    """A deterministic content master key. The key oracle is not up in a unit test, but the
    envelope boundary itself runs for real on every store write — the same stub
    `src/mantle/db/test_lattice_api.py` installs, with the same full signature, so a new
    argument on the real path cannot be absorbed silently here."""
    from mantle.services import content_crypto

    monkeypatch.setattr(
        content_crypto, "_default_master_key",
        lambda principal_id, collection_id=None, *, may_create=False, creator_id=None: b"\x01" * 32,
    )


@pytest.fixture()
def store(tmp_path):
    """A fresh on-disk lattice. The sites under test read raw grant DOCUMENTS out of the store,
    so a dict double would not exercise the thing that was broken: the defect lived precisely in
    the doc-shaped read (`d.get("can_invoke")`), which never sees an entity."""
    from mantle.db import lattice_api as la

    return la.open_database(str(tmp_path / "lat.db"), origin="test-deny")


def _grant(store, *, resource_id, grantee, effect, actions, grantee_type="user", **kw):
    """Mint a grant carrying exactly *actions*, with *effect*, and store it.

    Written through the ordinary entity + `create_grant` path rather than by hand-crafting a doc,
    so what these tests read back is what the product actually stores — including the `effect`
    normalisation the entity constructor applies.
    """
    from mantle.db import lattice_api as la

    flags = {f"can_{a}": (a in actions) for a in ACTIONS}
    g = GrantEntity(
        id=str(uuid.uuid4()),
        resource_id=resource_id,
        grantee_type=grantee_type,
        grantee_id=grantee,
        granted_by="owner",
        effect=effect,
        state=GrantEntity.STATE_ACTIVE,
        **{**flags, **kw},
    )
    la.create_grant(store, g)
    return g


def test_the_seed_shape_is_the_dangerous_one() -> None:
    """The premise every test below rests on: a deny grant really does carry its bits.

    If a deny grant stored `can_invoke=False`, the bare `.get` would already have been safe and
    none of this would be a defect. It does not — so the bit is True and the effect is what has
    to be read.
    """
    g = GrantEntity(resource_id="r", grantee_type="user", grantee_id="u", granted_by="o",
                    effect="deny", can_invoke=True, can_read=False)
    assert g.can_invoke is True, "a deny grant's bits name what it denies; that is the hazard"
    assert mask_of(g).carries("invoke") is True
    assert mask_of(g).allows("invoke") is False, "the operator answers the whole question"


# ═════════════════════════════════════════════════════════════════════════════════════════
# 1 · `db.access.invokable_resources` — the sharpest site: it failed OPEN
# ═════════════════════════════════════════════════════════════════════════════════════════
#
# Discharge actuates the world. `may_invoke` and `DischargeAuthority.may_discharge` answer from
# this set, so a deny grant reading as an allow here is not an information leak, it is a firing.

def test_a_deny_invoke_grant_does_not_authorise_discharge(store) -> None:
    from mantle.db import access

    _grant(store, resource_id="crystal-1", grantee="mallory", effect="deny", actions={"invoke"})

    assert access.invokable_resources(store, "mallory") == set(), (
        "a deny-effect grant carrying can_invoke seeded the discharge light cone — S1, and it "
        "fails OPEN: this is the set `may_invoke` answers from")
    assert access.may_invoke(store, "crystal-1", "mallory") is False
    assert access.DischargeAuthority(store).may_discharge("mallory", "crystal-1") is False


def test_an_allow_invoke_grant_still_authorises_discharge(store) -> None:
    """The control. A test that only ever asserts False passes on a function that returns the
    empty set, which would be a different bug wearing this one's clothes."""
    from mantle.db import access

    _grant(store, resource_id="crystal-1", grantee="alice", effect="allow", actions={"invoke"})

    assert access.may_invoke(store, "crystal-1", "alice") is True
    assert access.DischargeAuthority(store).may_discharge("alice", "crystal-1") is True


def test_a_deny_beats_an_allow_on_the_same_resource(store) -> None:
    """Deny-first, matching `services.dependencies.check_access`, which tests deny before allow
    at every level of its walk. Without the subtraction the answer would depend on the order the
    store happened to return two grants in — a coin flip deciding whether something fires."""
    from mantle.db import access

    _grant(store, resource_id="crystal-1", grantee="bob", effect="allow", actions={"invoke"})
    _grant(store, resource_id="crystal-1", grantee="bob", effect="deny", actions={"invoke"})

    assert access.may_invoke(store, "crystal-1", "bob") is False


def test_an_expired_deny_does_not_wrongly_subtract(store) -> None:
    """Symmetry check on the subtraction: expiry is applied before the effect is read, so a
    lapsed deny neither authorises nor keeps denying."""
    from mantle.db import access

    _grant(store, resource_id="crystal-1", grantee="carol", effect="allow", actions={"invoke"})
    _grant(store, resource_id="crystal-1", grantee="carol", effect="deny", actions={"invoke"},
           expires_at="2000-01-01T00:00:00+00:00")

    assert access.may_invoke(store, "crystal-1", "carol") is True


# ═════════════════════════════════════════════════════════════════════════════════════════
# 2 · `db.access.gated_collections` — deny-blind ON PURPOSE, and pinned as such
# ═════════════════════════════════════════════════════════════════════════════════════════

def test_a_deny_grant_still_gates_a_collection(store) -> None:
    """The one site that must NOT start honouring the effect.

    Gating asks "is this collection administered by somebody" — a collection nobody has granted
    anything on is the public commons. A deny grant is administration. Reading it as `allows`
    would drop the collection out of the gated set, `is_public` would answer True for its
    members, and a grant that says "no" would have made the collection world-readable. That is
    the fail-OPEN direction, which is why the fix here is `Mask.carries` (the bit, effect-blind)
    and not `Mask.allows`.
    """
    from mantle.db import access

    _grant(store, resource_id="col-private", grantee="dana", effect="deny", actions={"read"})

    assert "col-private" in access.gated_collections(store), (
        "a deny grant stopped gating its collection — reading the effect here would publish it")
    assert access.is_public(store, {"id": "m1", "collection_id": "col-private"}) is False


def test_a_collection_nobody_granted_anything_on_is_public(store) -> None:
    """The other half of the same claim: gating comes from grants, not from a flag."""
    from mantle.db import access

    assert access.gated_collections(store) == set()
    assert access.is_public(store, {"id": "m1", "collection_id": "col-open"}) is True


def test_a_deny_grantee_is_not_reported_as_owner_when_a_real_holder_exists(store) -> None:
    """`gated_owner_map`'s keys must stay exactly `gated_collections`', so a collection gated
    only by a deny grant still gets a name. But whenever a real holder exists, the allow pass
    runs first and it is the one `setdefault` keeps."""
    from mantle.db import access

    _grant(store, resource_id="col-x", grantee="mallory", effect="deny", actions={"read"})
    _grant(store, resource_id="col-x", grantee="owner-real", effect="allow", actions={"read"})

    assert access.gated_owner_map(store)["col-x"] == "owner-real"
    assert set(access.gated_owner_map(store)) <= access.gated_collections(store), (
        "the owner map named a collection the gated set does not contain")


# ═════════════════════════════════════════════════════════════════════════════════════════
# 3 · `db.lattice_api.get_active_collection_ids_for_user` — the read cone, and it failed OPEN
# ═════════════════════════════════════════════════════════════════════════════════════════

def test_a_deny_read_grant_does_not_seed_the_read_light_cone(store) -> None:
    from mantle.db import lattice_api as la
    from mantle.db import access

    _grant(store, resource_id="col-secret", grantee="mallory", effect="deny", actions={"read"})

    assert la.get_active_collection_ids_for_user(store, "mallory") == [], (
        "a deny-effect grant carrying can_read seeded the reader's own light cone")
    assert access.reachable_collections(store, "mallory") == set()


def test_an_allow_read_grant_still_seeds_the_read_light_cone(store) -> None:
    from mantle.db import lattice_api as la

    _grant(store, resource_id="col-mine", grantee="alice", effect="allow", actions={"read"})
    assert la.get_active_collection_ids_for_user(store, "alice") == ["col-mine"]


def test_a_deny_read_beats_an_allow_read_on_the_same_resource(store) -> None:
    from mantle.db import lattice_api as la

    _grant(store, resource_id="col-mine", grantee="bob", effect="allow", actions={"read"})
    _grant(store, resource_id="col-mine", grantee="bob", effect="deny", actions={"read"})
    assert la.get_active_collection_ids_for_user(store, "bob") == []


# ═════════════════════════════════════════════════════════════════════════════════════════
# 4 · `services.grant_service` — share / admin, and the invite ceiling above them
# ═════════════════════════════════════════════════════════════════════════════════════════
#
# These take a `Database`, so the grant lookup is stubbed rather than stored: the defect was in
# how the returned grants were READ, not in how they were found, and stubbing the lookup is what
# makes the seeded shape unambiguous.

@pytest.fixture()
def grants_returning(monkeypatch):
    """Point `grant_service`'s only grant lookup at a fixed list."""
    def _install(*grants):
        from mantle.services import grant_service
        monkeypatch.setattr(
            grant_service, "get_active_grants_for_principal_resource",
            lambda db, grantee_id, resource_id: list(grants))
    return _install


def _entity(effect, **flags):
    return GrantEntity(resource_id="res", grantee_type="user", grantee_id="u",
                       granted_by="o", effect=effect, state=GrantEntity.STATE_ACTIVE,
                       **{"can_read": False, **flags})


def test_a_deny_grant_does_not_confer_share_or_admin(grants_returning) -> None:
    from mantle.services import grant_service

    grants_returning(_entity("deny", can_share=True, can_admin=True))

    assert grant_service.can_share(None, "u", "res") is False, (
        "a deny grant carrying can_share conferred the right to mint invites")
    assert grant_service.can_admin(None, "u", "res") is False, (
        "a deny grant carrying can_admin conferred grant administration")


def test_an_allow_grant_still_confers_share_and_admin(grants_returning) -> None:
    from mantle.services import grant_service

    grants_returning(_entity("allow", can_share=True, can_admin=True))
    assert grant_service.can_share(None, "u", "res") is True
    assert grant_service.can_admin(None, "u", "res") is True


def test_a_deny_beats_an_allow_for_share(grants_returning) -> None:
    from mantle.services import grant_service

    grants_returning(_entity("allow", can_share=True), _entity("deny", can_share=True))
    assert grant_service.can_share(None, "u", "res") is False


def test_the_column_named_shim_agrees_with_the_action_named_question(grants_returning) -> None:
    """`user_has_any_flag` is a back-compat spelling, not a second implementation. An unknown
    flag name must contribute nothing rather than open a hole."""
    from mantle.services import grant_service

    grants_returning(_entity("allow", can_share=True))
    assert grant_service.user_has_any_flag(None, "u", "res", "can_share") is True
    assert grant_service.user_has_any_flag(None, "u", "res", "can_publish") is False
    assert grant_service.user_has_any_flag(None, "u", "res") is False

    grants_returning(_entity("deny", can_share=True))
    assert grant_service.user_has_any_flag(None, "u", "res", "can_share") is False


def test_a_deny_grant_is_not_a_ceiling_an_invite_can_be_minted_against(grants_returning) -> None:
    """`effective_flags` is the ceiling `create_invite` clamps a role preset against, so a flag
    wrongly reported held here becomes a real grant on somebody's account one claim later: mint
    an `admin` invite, claim it, and a denial has been converted into an allow."""
    from mantle.services import grant_service

    grants_returning(_entity("deny", can_admin=True, can_share=True))
    held = grant_service.effective_flags(None, "u", "res")
    assert held["can_admin"] is False and held["can_share"] is False

    grants_returning(_entity("allow", can_share=True), _entity("deny", can_admin=True))
    held = grant_service.effective_flags(None, "u", "res")
    assert held["can_share"] is True, "the allow half must survive"
    assert held["can_admin"] is False, "the deny must remove what it names, whatever the order"


# ═════════════════════════════════════════════════════════════════════════════════════════
# 5 · `routers.artifacts_router.list_visible` — the bearer grant it adds by hand
# ═════════════════════════════════════════════════════════════════════════════════════════

def _list_visible(bearer_grant, action="read"):
    """Drive the handler with the light cone empty, so the ONLY thing that can put an id in the
    result is the bearer-grant line under test."""
    import asyncio
    from unittest.mock import patch

    from mantle.routers import artifacts_router as ar
    from mantle.services.dependencies import AuthContext

    auth = AuthContext(principal_id="k", principal_type="grant_key", user_id=None,
                       bearer_grant=bearer_grant)

    class _Resolver:
        def __init__(self, *a, **kw): pass
        def resolve(self, *a, **kw): return set()

    with patch("mantle.search.mantle.lightcone.LightConeResolver", _Resolver), \
            patch.object(ar, "_hydrate_batch",
                         side_effect=lambda db, ids: {i: {"id": i} for i in ids}):
        body = asyncio.run(ar.list_visible(
            content_type=None, action=action, limit=100, offset=0, auth=auth, store_db=None))
    #: `/visible` returns `{items, total, has_more}` since 2026-08-25 (P-6/P-7). Asserted here so
    #: every test below keeps comparing a plain list while the envelope itself stays covered.
    assert set(body) == {"items", "total", "has_more"}, "the list envelope changed: %r" % (body,)
    return body["items"]


def test_a_deny_bearer_grant_does_not_appear_in_the_visible_list() -> None:
    g = GrantEntity(resource_id="art-secret", grantee_type="grant_key", grantee_id="h",
                    granted_by="o", effect="deny", can_read=True)
    assert _list_visible(g) == [], (
        "a deny-effect bearer grant added its own resource to the caller's visible set — S1 in "
        "the listing path")


def test_an_allow_bearer_grant_still_appears_in_the_visible_list() -> None:
    g = GrantEntity(resource_id="art-mine", grantee_type="grant_key", grantee_id="h",
                    granted_by="o", effect="allow", can_read=True)
    assert [d["id"] for d in _list_visible(g)] == ["art-mine"]


def test_a_bearer_grant_without_the_bit_does_not_appear_for_that_action() -> None:
    """The half of the question that was never broken, kept so a fix that answered only the
    effect would be caught too."""
    g = GrantEntity(resource_id="art-mine", grantee_type="grant_key", grantee_id="h",
                    granted_by="o", effect="allow", can_read=True)
    assert _list_visible(g, action="delete") == []


def test_an_unknown_action_is_still_a_400() -> None:
    """`action not in ACTIONS` replaced an `_ACTION_FLAG_MAP` lookup. The two vocabularies are
    the same one (`ACTION_FLAGS` is built from `ACTIONS`), and this pins that they stay so."""
    from fastapi import HTTPException

    g = GrantEntity(resource_id="a", grantee_type="grant_key", grantee_id="h",
                    granted_by="o", effect="allow", can_read=True)
    with pytest.raises(HTTPException) as exc:
        _list_visible(g, action="publish")
    assert exc.value.status_code == 400


# ═════════════════════════════════════════════════════════════════════════════════════════
# 6 · The two edge prunes — `propagates()` in place of the inline membership test
# ═════════════════════════════════════════════════════════════════════════════════════════
#
# These two sites were not deny-blind; they were the SAME RULE written twice, in two modules.
# The risk is drift, so what is pinned here is agreement — measured against the verbatim
# expression they used to hold, over every column shape the store is known to contain.

def _frozen_decode(v):
    """Verbatim `lattice_api._prop_mask`, minus the edge-attribute lookup. Frozen: never
    re-point this at `attenuation`, or the comparison below proves a function equals itself."""
    import json
    if isinstance(v, str) and v.startswith("["):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _inline_prune(mask, action) -> bool:
    """The expression as `list_origin_descendants` and `check_access` each used to spell it.
    Also frozen."""
    return not (mask is not None and action not in mask)


@pytest.mark.parametrize("column", [
    None, "[]", '["read"]', '["read", "invoke"]', "r", "crudeasio", "read,update",
    [], ["read"], ("read", "invoke"), {"read": True},
])
def test_the_two_prunes_still_answer_what_the_inline_expression_answered(column) -> None:
    from mantle.attenuation import propagates

    decoded = _frozen_decode(column)
    for action in ACTIONS:
        assert propagates(column, action) is _inline_prune(decoded, action), (
            f"the operator disagrees with the live column semantics on {column!r}/{action}")


def test_the_origin_bfs_prunes_exactly_where_it_used_to(store) -> None:
    """End to end through the real BFS: an edge that propagates only `read` must not carry
    `update` into the subtree behind it."""
    from mantle.db import lattice_api as la

    la.add_artifact_to_collection(store, "ws", "child", origin=True, propagate=["read"])
    la.add_artifact_to_collection(store, "child", "grandchild", origin=True, propagate=None)

    assert la.list_origin_descendants(store, ["ws"], "read") == {"child", "grandchild"}
    assert la.list_origin_descendants(store, ["ws"], "update") == set()


def test_an_unrestricted_edge_does_not_carry_a_verb_that_is_not_a_verb(store) -> None:
    """The one deliberate difference from the inline expression, and it is the closed direction:
    a NULL mask used to pass ANY string, because `is not None` short-circuited before the
    membership test ever ran. `Mask.allows` answers False for a name outside CRUDEASIO."""
    from mantle.db import lattice_api as la

    la.add_artifact_to_collection(store, "ws", "child", origin=True, propagate=None)

    assert la.list_origin_descendants(store, ["ws"], "read") == {"child"}
    assert la.list_origin_descendants(store, ["ws"], "publish") == set()
    assert _inline_prune(None, "publish") is True, (
        "the frozen expression must still show the OLD, open answer, or this test is not "
        "recording a change")


# ═════════════════════════════════════════════════════════════════════════════════════════
# 7 · The doc-shaped mask adapters — the piece both lattice-side fixes rest on
# ═════════════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("helper", ["access", "lattice_api"])
def test_a_grant_doc_reads_the_same_as_the_entity_it_came_from(helper) -> None:
    """`_grant_mask` / `_doc_mask` exist because the lattice never loads an entity on these
    paths. If they disagreed with `mask_of` the fix would be cosmetic."""
    from mantle.db.access import _grant_mask
    from mantle.db.lattice_api import _doc_mask

    under_test = _grant_mask if helper == "access" else _doc_mask

    for effect in ("allow", "deny", "ALLOW", "weird", ""):
        entity = GrantEntity(resource_id="r", grantee_type="user", grantee_id="u",
                             granted_by="o", effect=effect, can_read=True, can_invoke=True)
        assert under_test(entity.to_dict()) == mask_of(entity), (
            f"the doc adapter and the entity adapter disagree on effect={effect!r}")


@pytest.mark.parametrize("helper", ["access", "lattice_api"])
def test_a_doc_missing_a_column_confers_nothing_for_it(helper) -> None:
    """`Grant.from_dict` would have defaulted `can_read` to True. A decoder feeding an
    authorization decision must default the other way."""
    from mantle.db.access import _grant_mask
    from mantle.db.lattice_api import _doc_mask

    under_test = _grant_mask if helper == "access" else _doc_mask
    m = under_test({"effect": "allow", "resource_id": "r"})
    assert m.carries("read") is False, "a missing column widened into an allow"
    assert m.actions == frozenset()


@pytest.mark.parametrize("helper", ["access", "lattice_api"])
def test_a_doc_with_no_effect_at_all_confers_nothing(helper) -> None:
    """Positive matching: an absent or unrecognised effect is neither allow nor deny, and must
    authorize nothing."""
    from mantle.db.access import _grant_mask
    from mantle.db.lattice_api import _doc_mask

    under_test = _grant_mask if helper == "access" else _doc_mask
    for doc in ({"can_read": True}, {"can_read": True, "effect": "maybe"}):
        m = under_test(doc)
        assert m.allows("read") is False
        assert m.carries("read") is True, "the bit is still readable; only the effect is refused"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ── the resolver that decides KEY CUSTODY ─────────────────────────────────────────────────
#
# This section exists because the file above did not cover the one path that matters most.
# Four enforcement points were pinned — `db/access`, `lattice_api.get_active_collection_ids_
# for_user`, `grant_service`, `list_visible` — and `LightConeResolver.resolve` was not among
# them. It is the resolver that feeds `oracle.LightConeGrantVerifier`, so it does not merely
# decide what a listing shows: it decides which content keys get issued.
#
# Without the subtraction, with allow-read and deny-read both on `col-1`:
#
#     resolve('u1', 'read') -> {'col-1', 'art-a'}      check_access('col-1') -> 404
#
# `recall` returned the denied artifact and hydration decrypted its content inline. And because
# keys are derived per (principal, collection) while `contexts` is deliberately un-narrowed, the
# denied principal was issued a key for the WHOLE COLLECTION rather than a stray id in a list.
#
# The resolver's own module docstring states the contract these tests enforce: "this resolver
# must not exceed [check_access]" — which tests deny first, at every level of its walk.

def _resolver_grant(resource_id: str, effect: str):
    from mantle.entities.grant import Grant

    return Grant(resource_id=resource_id, grantee_type="user", grantee_id="u1",
                 granted_by="u1", effect=effect, can_read=True, state="active")


def _resolve_with(grants, descendants):
    """`resolve` over a stubbed grant set and containment graph."""
    from unittest.mock import MagicMock, patch

    from mantle.search.mantle.lightcone import LightConeResolver

    resolver = LightConeResolver(MagicMock())
    with patch.object(LightConeResolver, "_grants_for", return_value=grants), \
         patch("mantle.db.backend.list_origin_descendants",
               side_effect=lambda db, ids, a: {d for i in ids for d in descendants.get(i, ())}), \
         patch("mantle.services.context_service.reach") as reach:
        reach.return_value = MagicMock(ids=set())
        return resolver.resolve("u1", "read")


def test_the_resolver_returns_what_an_allow_confers_positive_control() -> None:
    """The control. Everything below asserts absence; without this one they would all pass on a
    resolver that returned the empty set for every input."""
    got = _resolve_with([_resolver_grant("col-1", "allow")], {"col-1": {"art-a"}})
    assert got == {"col-1", "art-a"}


def test_a_deny_beside_an_allow_removes_the_resource_from_the_resolver() -> None:
    """The reported defect. `allows()` already stopped a deny from SEEDING the cone, so a deny
    standing alone was handled — but nothing removed what a co-located allow had already seeded.
    Absorbing a deny is not the same as subtracting one."""
    got = _resolve_with(
        [_resolver_grant("col-1", "allow"), _resolver_grant("col-1", "deny")],
        {"col-1": {"art-a"}},
    )
    assert got == set(), (
        "a denied resource reached the resolver, which issues content keys — %r" % (got,))


def test_a_deny_on_a_member_removes_only_that_member() -> None:
    """Deny is subtracted at artifact granularity, not collection granularity. Losing the whole
    collection here would be the fail-closed direction, but it would also make a single deny
    revoke everything filed beside it."""
    got = _resolve_with(
        [_resolver_grant("col-1", "allow"), _resolver_grant("art-a", "deny")],
        {"col-1": {"art-a", "art-b"}},
    )
    assert got == {"col-1", "art-b"}


def test_a_deny_on_a_container_reaches_the_artifacts_inside_it() -> None:
    """Deny expands through containment exactly as the allow did. Without this, denying a
    container would leave every artifact inside it reachable — the same bug one level down, and
    the reason `db/access.invokable_resources` expands both piles."""
    got = _resolve_with(
        [_resolver_grant("col-1", "allow"), _resolver_grant("col-2", "allow"),
         _resolver_grant("col-2", "deny")],
        {"col-1": {"art-a"}, "col-2": {"art-b", "art-c"}},
    )
    assert got == {"col-1", "art-a"}


def test_a_deny_carrying_a_different_action_does_not_subtract() -> None:
    """The bit test is per-action on the deny side too. A deny that speaks only about `update`
    must not remove a `read` the principal holds — over-subtracting is fail-closed, but it would
    make deny grants unusable for narrowing a single verb.

    `can_read=False` is load-bearing in this fixture. `Grant.__init__` defaults `can_read=True`
    (`entities/grant.py:176`), so a deny grant constructed for `update` alone carries the `read`
    bit as well — and a deny grant's bits name
    the actions it DENIES. Built without this argument, the fixture denies read, the subtraction
    fires correctly, and the test fails while the code is right.

    That default is a real hazard in its own direction: on the ALLOW side the same line means a
    partial grant doc materialises as allow-read. It is recorded in the audit; this fixture only
    has to state which behaviour it is pinning.
    """
    from mantle.entities.grant import Grant

    deny_update_only = Grant(resource_id="col-1", grantee_type="user", grantee_id="u1",
                             granted_by="u1", effect="deny",
                             can_read=False, can_update=True, state="active")
    got = _resolve_with([_resolver_grant("col-1", "allow"), deny_update_only],
                        {"col-1": {"art-a"}})
    assert got == {"col-1", "art-a"}
