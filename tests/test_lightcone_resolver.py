"""Unit tests for `search.mantle.LightConeResolver`.

CRUDEASIO lives in Mantle (the lattice grants collection). The resolver reads
grants from `db_store.get_active_grants_for_grantee` — no Origin HTTP
calls. Tests cover:

- empty grant set → empty result
- grant with action mismatch (e.g. read-only grant for "create") → not included
- direct grant only (no descendants) → just that ID
- direct grant + descendants → union
- propagate mask blocks descendants → only direct included
- unknown action → empty
- deny-effect grant → not authorizing
- the context lattice is seeded from what containment reached, held under a stated action
  ceiling, CONFINED to the grant-derived set, and bounded (D16)

The context lattice's own properties — nesting, multi-parent, cycles, that authority strictly
attenuates along a context chain, and the swept invariant that a context edge never widens
what grants authorize — are in `tests/test_context_lattice.py`. What is tested here is only
what the RESOLVER decides: what it seeds the walk with, what two ceilings it holds it under,
and that it does not run the walk at all for a principal with no grants.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mantle.attenuation import Mask
from mantle.db import backend as db_store
from mantle.entities.grant import Grant as GrantEntity
from mantle.search.mantle import LightConeResolver
from mantle.services import context_service

def _grant(resource_id: str, **flags) -> GrantEntity:
    """Build a GrantEntity with CRUDEASIO flags defaulting to False."""
    defaults = {
        "can_read": False, "can_create": False, "can_update": False,
        "can_delete": False, "can_evict": False, "can_invoke": False,
        "can_add": False, "can_share": False, "can_admin": False,
    }
    defaults.update(flags)
    effect = defaults.pop("effect", "allow")
    return GrantEntity(
        resource_id=resource_id,
        grantee_type="user",
        grantee_id="user-1",
        granted_by="admin",
        effect=effect,
        **defaults,
    )


def test_empty_grants_returns_empty_set():
    with patch.object(db_store, "get_active_grants_for_grantee", return_value=[]):
        resolver = LightConeResolver(db=MagicMock())
        assert resolver.resolve("user-1") == set()


def test_grant_lacking_action_flag_is_excluded():
    # Grant has can_read=False; resolving for "read" must skip it.
    with patch.object(db_store, "get_active_grants_for_grantee",
                      return_value=[_grant("col-1", can_read=False, can_admin=True)]):
        resolver = LightConeResolver(db=MagicMock())
        assert resolver.resolve("user-1", "read") == set()


# ---------------------------------------------------------------------------
# effect — deny is the absorbing element
# ---------------------------------------------------------------------------
#
# The resolver used to filter on the CRUDEASIO flag alone (`getattr(g, flag_attr, False)`),
# which is only half the authorization question: a `deny`-effect grant carrying `can_read`
# was read as AUTHORIZING. It feeds key issuance (`oracle.LightConeGrantVerifier`) and
# search-result decryption (`sse/router_accessor`), so the failure would have handed out a
# content key on the strength of a grant that says the opposite of what it means.
#
# Structurally this is now impossible — `mask_of(g).allows(action)` is the meet of the bit
# and the effect, and deny is that operator's absorbing zero (`tests/
# test_attenuation_algebra.py` proves it over the whole domain). These tests keep the
# guarantee visible at the level it matters, because "unrepresentable" is a claim about the
# operator and this is a claim about the resolver.


def test_a_deny_grant_does_not_seed_the_light_cone():
    """The regression. A deny grant with `can_read=True` reaches nothing."""
    deny = _grant("col-1", can_read=True, effect="deny")
    with (
        patch.object(db_store, "get_active_grants_for_grantee", return_value=[deny]),
        patch.object(db_store, "list_origin_descendants", return_value={"art-a"}),
    ):
        reached = LightConeResolver(db=MagicMock()).resolve("user-1", "read")

    assert reached == set(), (
        "a deny-effect grant seeded the light cone — it authorized decryption and key "
        "issuance while saying the opposite"
    )


def test_a_deny_grant_does_not_drag_its_descendants_in_either():
    """The BFS must not even be asked: pruning starts at the seed, so a deny grant must
    contribute no root for the walk to expand."""
    called = []

    def fake_descendants(_db, root_ids, action):
        called.append(list(root_ids))
        return {"art-a"}

    with (
        patch.object(db_store, "get_active_grants_for_grantee",
                     return_value=[_grant("col-1", can_read=True, effect="deny")]),
        patch.object(db_store, "list_origin_descendants", side_effect=fake_descendants),
    ):
        reached = LightConeResolver(db=MagicMock()).resolve("user-1", "read")

    assert reached == set()
    assert called == [], f"the BFS was seeded from a deny grant: {called}"


def test_an_unrecognized_effect_confers_nothing_here_too():
    """`grant_is_allow` is positive matching, not `not grant_is_deny(...)` — an effect
    nobody can read as an allow must fail closed rather than fall through."""
    with (
        patch.object(db_store, "get_active_grants_for_grantee",
                     return_value=[_grant("col-1", can_read=True, effect="permit")]),
        patch.object(db_store, "list_origin_descendants", return_value=set()),
    ):
        assert LightConeResolver(db=MagicMock()).resolve("user-1", "read") == set()


def test_an_allow_grant_alongside_a_deny_one_still_reaches_its_own_resource():
    """The positive control. Without it every assertion above would pass against a resolver
    that reached nothing at all.

    It also states the residual, deliberately: deny here is *absorbing*, not *subtractive* —
    a deny on `col-1` does not revoke a separate allow on `col-2`. Per-resource deny
    PRECEDENCE (a deny beating a co-located allow) lives in `services.dependencies.
    check_access`, which is the gate in front of every artifact read.
    """
    grants = [
        _grant("col-1", can_read=True, effect="deny"),
        _grant("col-2", can_read=True),
    ]
    with (
        patch.object(db_store, "get_active_grants_for_grantee", return_value=grants),
        patch.object(db_store, "list_origin_descendants", return_value=set()),
    ):
        assert LightConeResolver(db=MagicMock()).resolve("user-1", "read") == {"col-2"}


def test_the_deny_check_is_not_a_blanket_and_still_honours_the_action():
    """Both halves of the meet must still be live: a deny grant is filtered on its effect,
    an allow grant lacking the bit on its bit. Collapsing either into the other would pass
    the regression above while breaking the other axis."""
    with (
        patch.object(db_store, "get_active_grants_for_grantee", return_value=[
            _grant("col-1", can_read=True),                      # allow + bit  -> in
            _grant("col-2", can_update=True),                    # allow, no bit -> out
            _grant("col-3", can_read=True, effect="deny"),       # bit, no allow -> out
        ]),
        patch.object(db_store, "list_origin_descendants", return_value=set()),
    ):
        assert LightConeResolver(db=MagicMock()).resolve("user-1", "read") == {"col-1"}


def test_unknown_action_returns_empty():
    with patch.object(db_store, "get_active_grants_for_grantee",
                      return_value=[_grant("col-1", can_read=True)]):
        resolver = LightConeResolver(db=MagicMock())
        assert resolver.resolve("user-1", "no-such-action") == set()


def test_direct_grant_with_no_descendants():
    with (
        patch.object(db_store, "get_active_grants_for_grantee",
                     return_value=[_grant("col-1", can_read=True)]),
        patch.object(db_store, "list_origin_descendants", return_value=set()),
    ):
        resolver = LightConeResolver(db=MagicMock())
        assert resolver.resolve("user-1", "read") == {"col-1"}


def test_direct_grant_unions_with_descendants():
    with (
        patch.object(db_store, "get_active_grants_for_grantee",
                     return_value=[_grant("col-1", can_read=True)]),
        patch.object(db_store, "list_origin_descendants",
                     return_value={"art-a", "art-b", "sub-col"}),
    ):
        resolver = LightConeResolver(db=MagicMock())
        result = resolver.resolve("user-1", "read")
    assert result == {"col-1", "art-a", "art-b", "sub-col"}


def test_two_grants_descendants_unioned():
    captured = {}

    def fake_descendants(_db, root_ids, action):
        captured["root_ids"] = list(root_ids)
        captured["action"] = action
        return {"x", "y"}

    with (
        patch.object(db_store, "get_active_grants_for_grantee", return_value=[
            _grant("col-1", can_read=True),
            _grant("col-2", can_read=True),
        ]),
        patch.object(db_store, "list_origin_descendants", side_effect=fake_descendants),
    ):
        resolver = LightConeResolver(db=MagicMock())
        result = resolver.resolve("user-1", "read")

    assert result == {"col-1", "col-2", "x", "y"}
    # The resolver passes both granted IDs to the BFS in one call.
    assert set(captured["root_ids"]) == {"col-1", "col-2"}
    assert captured["action"] == "read"


def test_action_passes_through_to_descendant_lookup():
    captured = {}

    def fake_descendants(_db, root_ids, action):
        captured["action"] = action
        return set()

    with (
        patch.object(db_store, "get_active_grants_for_grantee",
                     return_value=[_grant("col-1", can_create=True)]),
        patch.object(db_store, "list_origin_descendants", side_effect=fake_descendants),
    ):
        resolver = LightConeResolver(db=MagicMock())
        resolver.resolve("user-1", "create")

    assert captured["action"] == "create"


# ---------------------------------------------------------------------------
# the context lattice (D16) — one walk, not a second pass
# ---------------------------------------------------------------------------
#
# Context is an artifact with edges to its own context, so authorization and selection are the
# same traversal rather than "compute the authorized set, then filter". These pin the three
# decisions the RESOLVER makes about that walk; what the walk itself does is
# `tests/test_context_lattice.py`.


def test_the_context_walk_is_seeded_from_everything_containment_reached():
    """Not from the grants alone. A context edge hanging off a descendant is as real as one
    hanging off the granted resource, and seeding only the grants would make reach depend on
    where in a collection someone happened to attach it."""
    seen = {}

    def fake_reach(_db, seeds, action, **kw):
        seen["seeds"] = set(seeds)
        seen["action"] = action
        seen["authority"] = kw.get("authority")
        seen["within"] = kw.get("within")
        seen["bound"] = (kw.get("max_depth"), kw.get("max_nodes"))
        seen["expander"] = kw.get("expand_containment")
        return context_service.ContextReach(frozenset({"ctx-child"}), 1, False, None)

    with (
        patch.object(db_store, "get_active_grants_for_grantee",
                     return_value=[_grant("col-1", can_read=True)]),
        patch.object(db_store, "list_origin_descendants", return_value={"art-a"}),
        patch.object(context_service, "reach", side_effect=fake_reach),
    ):
        reached = LightConeResolver(db=MagicMock()).resolve("user-1", "read")
        # Asserted inside the patch: the expander the resolver hands over must be the LIVE
        # containment BFS, so the two lattices alternate inside one walk rather than being
        # unioned afterwards.
        assert seen["expander"] is db_store.list_origin_descendants

    assert seen["seeds"] == {"col-1", "art-a"}
    assert seen["action"] == "read"
    # `ctx-child` is outside what grants authorize, so it does not survive — even though the
    # (here deliberately lying) walk returned it. The resolver intersects rather than trusts.
    assert reached == {"col-1", "art-a"}


def test_the_context_walk_may_not_leave_what_grants_authorize():
    """The confinement, at the point the resolver states it.

    `within` must be exactly the grant-derived set: the granted ids plus the origin
    containment cone below them. A resolver that passed `UNCONFINED` and relied on filtering
    the result afterwards would be one refactor away from the bug this replaced — a grant on
    `org` returning `{"org", "project", "doc-1"}`, ids `check_access` would refuse to read
    and `oracle.LightConeGrantVerifier` would issue a content key for.
    """
    seen = {}

    with (
        patch.object(db_store, "get_active_grants_for_grantee",
                     return_value=[_grant("col-1", can_read=True)]),
        patch.object(db_store, "list_origin_descendants", return_value={"art-a", "art-b"}),
        patch.object(context_service, "reach",
                     side_effect=lambda *a, **kw: seen.update(kw) or context_service.EMPTY_REACH),
    ):
        reached = LightConeResolver(db=MagicMock()).resolve("user-1", "read")

    assert seen["within"] == {"col-1", "art-a", "art-b"}
    assert seen["within"] is not context_service.UNCONFINED
    assert reached == {"col-1", "art-a", "art-b"}


def test_the_context_walk_is_held_under_a_ceiling_that_states_the_action():
    """Every seed already passed `allows(action)`, and they came from grants with different
    bits. `Mask.of((action,))` is exactly what is true of all of them — and because the first
    hop meets with it, it is a real ceiling rather than a comment."""
    seen = {}

    with (
        patch.object(db_store, "get_active_grants_for_grantee",
                     return_value=[_grant("col-1", can_update=True)]),
        patch.object(db_store, "list_origin_descendants", return_value=set()),
        patch.object(context_service, "reach",
                     side_effect=lambda *a, **kw: seen.update(kw) or context_service.EMPTY_REACH),
    ):
        LightConeResolver(db=MagicMock()).resolve("user-1", "update")

    assert seen["authority"] == Mask.of(("update",))
    assert seen["authority"].allows("update") and not seen["authority"].allows("read")


def test_the_resolver_bounds_the_context_walk_and_the_bound_is_per_node():
    """`resolve()` runs the containment BFS to exhaustion, which is tolerable on a shallow
    engine-driven structure. The context lattice is deeper by construction, so its walk is
    bounded — and by a value fixed at construction, because a per-request bound would be a
    way to make any single query as expensive as the caller liked."""
    import inspect

    seen = {}
    with (
        patch.object(db_store, "get_active_grants_for_grantee",
                     return_value=[_grant("col-1", can_read=True)]),
        patch.object(db_store, "list_origin_descendants", return_value=set()),
        patch.object(context_service, "reach",
                     side_effect=lambda *a, **kw: seen.update(kw) or context_service.EMPTY_REACH),
    ):
        LightConeResolver(db=MagicMock()).resolve("user-1", "read")
        assert seen["max_depth"] == context_service.DEFAULT_MAX_DEPTH
        assert seen["max_nodes"] == context_service.DEFAULT_MAX_NODES

        LightConeResolver(db=MagicMock(), max_depth=3, max_nodes=7).resolve("user-1", "read")
        assert (seen["max_depth"], seen["max_nodes"]) == (3, 7)

    resolve_params = inspect.signature(LightConeResolver.resolve).parameters
    assert "max_depth" not in resolve_params and "max_nodes" not in resolve_params


def test_a_principal_with_no_grants_does_not_enter_the_context_lattice():
    """No seed, no walk. A context edge must not be an entry point for a principal the
    containment cone already refused."""
    calls = []
    with (
        patch.object(db_store, "get_active_grants_for_grantee", return_value=[]),
        patch.object(context_service, "reach",
                     side_effect=lambda *a, **kw: calls.append(a) or context_service.EMPTY_REACH),
    ):
        assert LightConeResolver(db=MagicMock()).resolve("user-1", "read") == set()
    assert calls == []


def test_a_deny_grant_does_not_enter_the_context_lattice_either():
    """S1 again, one layer out: the new edges must not become a second way for a deny grant to
    seed a cone."""
    calls = []
    with (
        patch.object(db_store, "get_active_grants_for_grantee",
                     return_value=[_grant("col-1", can_read=True, effect="deny")]),
        patch.object(db_store, "list_origin_descendants", return_value=set()),
        patch.object(context_service, "reach",
                     side_effect=lambda *a, **kw: calls.append(a) or context_service.EMPTY_REACH),
    ):
        assert LightConeResolver(db=MagicMock()).resolve("user-1", "read") == set()
    assert calls == []


# ---------------------------------------------------------------------------
# principal_type -> ledger grantee_type
# ---------------------------------------------------------------------------
#
# The platform system principal acts with principal_type "service", but
# `seed_provisioning/platform_email.py` issues its grants via
# `upsert_user_collection_grant`, which stores grantee_type "user". The resolver
# must map "service" to "user" when querying the ledger rather than passing it
# through verbatim, or system-principal grants would not be found.
#
# Both directions are asserted: the mapping must reach the ledger (it is not dropped) and must not
# be the identity function (it does not pass "service" on).

def test_service_principal_resolves_against_the_ledgers_principal_grant_kind():
    captured = {}

    def fake_grants(_db, *, grantee_id, grantee_type):
        captured["grantee_type"] = grantee_type
        return [_grant("col-1", can_update=True)]

    with (
        patch.object(db_store, "get_active_grants_for_grantee", side_effect=fake_grants),
        patch.object(db_store, "list_origin_descendants", return_value=set()),
    ):
        resolver = LightConeResolver(db=MagicMock())
        result = resolver.resolve("sys-1", "update", principal_type="service")

    # "service" is not a grantee_type this ledger ever writes. If this assertion ever reads
    # "service", the system principal's real grants become invisible.
    assert captured["grantee_type"] != "service"
    assert captured["grantee_type"] == "user"
    assert result == {"col-1"}


def test_non_key_principals_resolve_as_a_principal_id():
    """Entity kinds with no credential of their own hold `user`-type grants."""
    seen = []

    def fake_grants(_db, *, grantee_id, grantee_type):
        seen.append(grantee_type)
        return []

    with patch.object(db_store, "get_active_grants_for_grantee", side_effect=fake_grants):
        resolver = LightConeResolver(db=MagicMock())
        resolver.resolve("s-1", "read", principal_type="server")
        resolver.resolve("d-1", "read", principal_type="delegation")
        resolver.resolve("u-1", "read", principal_type="user")

    assert seen == ["user", "user", "user"]


def test_a_grant_key_is_resolved_through_its_bundle_not_by_grantee_lookup():
    """A key must NOT be looked up as a grantee of its own principal id.

    Its `grantee_id` is the token hash, so that lookup returns nothing and the key
    silently reaches zero resources. The resolver loads the root grant and expands the
    bundle instead. `test_acting_principal_grant_key_subject.py` covers the resulting
    reach; this pins that the grantee-lookup path is not taken at all.
    """
    root = GrantEntity(
        id="k-1", resource_id="col-root", grantee_type="grant_key",
        grantee_id="token-hash", granted_by="admin", can_read=True,
    )
    grantee_lookups = []

    def fake_grants(_db, *, grantee_id, grantee_type):
        grantee_lookups.append((grantee_id, grantee_type))
        return []

    with (
        patch.object(db_store, "get_active_grants_for_grantee", side_effect=fake_grants),
        patch.object(db_store, "get_grant_by_id", return_value=root),
        patch.object(db_store, "list_origin_descendants", return_value=set()),
    ):
        reached = LightConeResolver(db=MagicMock()).resolve(
            "k-1", "read", principal_type="grant_key")

    assert reached == {"col-root"}
    assert ("k-1", "grant_key") not in grantee_lookups, (
        "the resolver looked the key up as a grantee of its own id — that matches "
        "nothing, so the key would reach no resource at all"
    )


def test_verifier_does_not_drop_requester_type_before_the_lookup():
    """The parameter must reach the resolver rather than being dropped along the way."""
    import mantle.search.mantle.lightcone as lc
    from mantle.search.mantle.oracle import LightConeGrantVerifier

    captured = {}

    class _Recorder:
        def resolve(self, principal_id, action="read", *, principal_type="user"):
            captured["principal_type"] = principal_type
            return set()

        def _grants_for(self, principal_id, principal_type):
            # A collection-scoped request is answered by walking up from the collection and
            # testing each resource against the requester's grants, so this is where the type
            # has to arrive. A grant key's grants are a resolved bundle and a user's are a
            # ledger lookup; defaulting to "user" reads the wrong set for a key.
            captured["principal_type"] = principal_type
            return []

    with patch.object(
        lc, "_raw_artifact", return_value={"id": "col-1", "collection_id": "col-1"},
    ):
        v = LightConeGrantVerifier(MagicMock(), resolver=_Recorder())
        v.authorized(
            requester_id="sys-1", requester_type="service",
            principal_id="p-1", collection_id="col-1", action="update",
        )

    # principal_type must be threaded through to the resolver rather than falling
    # back to the default "user" value regardless of who asked.
    assert captured["principal_type"] == "service"


class TestAPostingKeyedByRootStillResolves:
    """`_raw_artifacts` must answer for a ROOT id, not only for a vertex id.

    `ingest/pipeline_unified._sse_index_artifact` indexes under `artifact.root_id or artifact.id`,
    so every posting for a versioned artifact names the ROOT — and `workspace_service
    .upsert_identity_member` deliberately never materialises that root ("the root is the identity
    and the id is the version"). Without the second hop the narrowing names an id, the id lookup
    misses, and `resolve_authorized_scope` drops it at `if not doc: continue`.

    Measured 2026-08-25 on 71/home: the `Claude Code` capture collection held 97 containment edges
    with zero existing targets and 183 captures with zero materialised roots. Every session
    transcript and file mirror the hooks wrote for a month was narrowed to and then dropped.
    """

    @staticmethod
    def _store(vertices, lineages):
        """A store double: `id` lookups miss, `versions_of_many` answers by root."""
        class _Artifacts:
            def __init__(self):
                self.calls = []

            @property
            def db(self):
                raise RuntimeError("no SQL handle")     # force the per-id fallback path

            def versions_of_many(self, root_ids):
                self.calls.append(list(root_ids))
                return {r: lineages[r] for r in root_ids if r in lineages}

        class _Store:
            def __init__(self):
                self.artifacts = _Artifacts()

        return _Store()

    def test_a_root_id_resolves_to_its_newest_committed_version(self, monkeypatch):
        from mantle.search.mantle import lightcone

        store = self._store({}, {
            "root-1": [
                {"id": "v1", "root_id": "root-1", "state": "committed", "name": "old"},
                {"id": "v2", "root_id": "root-1", "state": "committed", "name": "new"},
            ],
        })
        monkeypatch.setattr(lightcone, "_raw_artifact", lambda *_a, **_k: None)
        docs = lightcone._raw_artifacts(store, ["root-1"])
        assert docs["root-1"]["name"] == "new", (
            "the lineage is oldest-first, so the LAST committed doc is the current one"
        )

    def test_a_draft_is_not_substituted_for_a_missing_commit(self, monkeypatch):
        """`recall` reads the committed segment; answering with a draft publishes what nobody did."""
        from mantle.search.mantle import lightcone

        store = self._store({}, {
            "root-2": [{"id": "d1", "root_id": "root-2", "state": "draft", "name": "wip"}],
        })
        monkeypatch.setattr(lightcone, "_raw_artifact", lambda *_a, **_k: None)
        assert lightcone._raw_artifacts(store, ["root-2"]) == {}

    def test_the_second_hop_runs_only_on_the_misses(self, monkeypatch):
        """An ordinary recall, where every id is a real vertex, must not pay a lineage read."""
        from mantle.search.mantle import lightcone

        store = self._store({}, {})
        monkeypatch.setattr(
            lightcone, "_raw_artifact",
            lambda _s, aid: {"id": aid, "state": "committed"},
        )
        docs = lightcone._raw_artifacts(store, ["a", "b"])
        assert set(docs) == {"a", "b"}
        assert store.artifacts.calls == [], "no miss, so no lineage read"

    def test_a_store_without_versions_of_many_behaves_exactly_as_before(self, monkeypatch):
        """A test double or non-lattice backend must not start raising because of this."""
        from mantle.search.mantle import lightcone

        class _Bare:
            class artifacts:                     # noqa: D106 - no versions_of_many at all
                pass

        monkeypatch.setattr(lightcone, "_raw_artifact", lambda *_a, **_k: None)
        assert lightcone._raw_artifacts(_Bare(), ["x"]) == {}


class TestAConeTooLargeToHoldCanStillBePaged:
    """`authorized_page` answers "what may I see" without materialising the answer.

    Measured 2026-08-25 on 71/home for `d36f0429`, the index-writer principal that holds
    `stage.0.lexicon` (1.85M members) alongside grammar, world, ontology and the whole pharos tree:

        LightConeResolver.resolve(...)      EdgesTruncated after 53.7s
        authorized_page(offset=0,  limit=20)   20 ids in 0.09s
        authorized_page(offset=100, limit=20)  20 ids in 0.49s

    The raise happens inside `resolve_authorized_scope`, so before this the principal lost the
    listing endpoint for EVERY collection it held, not only the oversized one.
    """

    @staticmethod
    def _store(ids):
        class _Conn:
            def execute(self, sql, params=()):
                assert "ORDER BY id" in sql, "offset indexes into a defined order or into nothing"
                return [(i, None) for i in sorted(ids)]

        class _Artifacts:
            class db:
                @staticmethod
                def read():
                    return _Conn()

        class _Store:
            artifacts = _Artifacts()

        return _Store()

    def _page(self, monkeypatch, ids, reachable, **kw):
        from mantle.search.mantle import lightcone

        monkeypatch.setattr(lightcone, "_grant_sets", lambda *_a, **_k: ({"g"}, set()))
        monkeypatch.setattr(
            lightcone, "_reaches",
            lambda _db, aid, *_a, **_k: aid in reachable,
        )
        return lightcone.authorized_page(self._store(ids), object(), "p", **kw)

    def test_it_returns_only_what_the_walk_authorizes(self, monkeypatch):
        page = self._page(monkeypatch, ["a", "b", "c"], reachable={"a", "c"}, limit=10)
        assert page == ["a", "c"]

    def test_offset_indexes_into_the_authorized_sequence_not_the_scan(self, monkeypatch):
        """Skipping must count authorized ids, or a page would drop rows the caller may see."""
        page = self._page(monkeypatch, ["a", "b", "c", "d"], reachable={"a", "c", "d"},
                          offset=1, limit=10)
        assert page == ["c", "d"]

    def test_limit_stops_the_scan(self, monkeypatch):
        page = self._page(monkeypatch, ["a", "b", "c", "d"], reachable={"a", "b", "c", "d"},
                          limit=2)
        assert page == ["a", "b"]

    def test_a_principal_with_no_grants_gets_nothing_without_scanning(self, monkeypatch):
        from mantle.search.mantle import lightcone

        monkeypatch.setattr(lightcone, "_grant_sets", lambda *_a, **_k: (set(), set()))
        called = []
        monkeypatch.setattr(lightcone, "_reaches",
                            lambda *_a, **_k: called.append(1) or True)
        assert lightcone.authorized_page(self._store(["a"]), object(), "p") == []
        assert called == [], "no grants is answerable without touching the store"

    def test_no_sql_handle_returns_empty_rather_than_raising(self, monkeypatch):
        from mantle.search.mantle import lightcone

        class _Bare:
            class artifacts:
                class db:
                    @staticmethod
                    def read():
                        raise RuntimeError("no handle")

        monkeypatch.setattr(lightcone, "_grant_sets", lambda *_a, **_k: ({"g"}, set()))
        assert lightcone.authorized_page(_Bare(), object(), "p") == []
