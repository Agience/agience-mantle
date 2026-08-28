"""An `mcp_client` acts FOR a user — the user is the subject, the client is the actor.

Origin mints a third-party MCP token with `sub` = the end user and `aud` = the OAuth client
(`auth_router`), and records `scopes` as "a record of the request and not a per-client
entitlement". So the only identity in that token that anything holds grants under is `sub`.

Reading `aud` as the principal split this one caller in two, and the halves disagreed inside a
single request:

  * `check_access` looks grants up by `user_id` — the USER. Worked.
  * `list_visible` and `recall_artifacts` resolve the light cone from `auth.user_id`, as a
    plain `"user"` — the USER. Worked.
  * `acting_principal.acting_from_auth` reads `principal_id` — the CLIENT ID. Found no grants
    under a string no grant is keyed by, so the key oracle refused every content key.

A client could therefore LIST artifacts and never READ one, and `create_artifact` could not
encrypt what it was asked to store. Resolving `sub` is what makes the three agree.

**It widens nothing**, and that is the claim this file has to carry rather than assert. The
reachable set was already the user's: two of those three call sites had been resolving it from
`sub` all along, on every request, and they are the ones that decide what may be reached.
`principal_id` decides only which principal's keys may be held — the half that was failing
closed. `TestItWidensNothing` pins the reachable set on both sides of that line, and
`TestEveryOtherPrincipalTypeIsUnchanged` pins the blast radius to this one branch.

`test_acting_principal_grant_key_subject.py` holds the same identity question for a grant key,
where the answer is the opposite one — a key acts as ITSELF, because it acts for nobody. The
two files are the same rule read in both directions: the subject is whoever the credential
does the work on behalf of, and `actor` names the machine either way.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mantle.config import AUTHORITY_ISSUER
from mantle.entities.grant import Grant as GrantEntity
from mantle.services.acting_principal import acting_from_auth
from mantle.services.dependencies import AuthContext, resolve_auth
from mantle.search.mantle.lightcone import LightConeResolver, ledger_grantee_type

USER = "9d1b7c5e-0000-4f00-8000-0000000000aa"
CLIENT = "mcp-client-2f6c"


def _mcp_payload(**overrides) -> dict:
    """The claims Origin's `auth_router` puts in a scoped third-party token."""
    base = {
        "sub": USER,
        "aud": CLIENT,
        "iss": AUTHORITY_ISSUER,
        "principal_type": "mcp_client",
        "scopes": ["read"],
    }
    base.update(overrides)
    return base


def _resolve(payload: dict) -> AuthContext:
    with patch("mantle.services.dependencies.verify_token", return_value=payload):
        return resolve_auth(token="mcp-jwt", store_db=MagicMock())


# ── the identity ──────────────────────────────────────────────────────────────────────────────
class TestTheSubjectIsTheUser:
    def test_the_principal_is_the_user_the_client_acts_for(self):
        ctx = _resolve(_mcp_payload())

        assert ctx.principal_id == USER, "grants are keyed by the user, so the principal is too"
        assert ctx.user_id == USER
        assert ctx.principal_id != CLIENT

    def test_the_client_id_is_not_read_as_an_identity(self):
        """`aud` is an audience. Nothing mints grants under a client id, so resolving one as
        the principal names a holder of nothing — which is why this failed closed rather than
        open, and why it took a content read to notice."""
        ctx = _resolve(_mcp_payload())
        assert CLIENT not in (ctx.principal_id, ctx.user_id)

    def test_the_kind_survives_as_mcp_client(self):
        """Not collapsed to `"user"` the way `delegation` is. `ledger_grantee_type` already
        maps it onto the ledger's `"user"` grantee vocabulary, so the grant lookup is identical
        either way — and keeping the name is what lets an audit reader, or any
        `principal_type == "user"` gate, still tell a machine from a person."""
        ctx = _resolve(_mcp_payload())

        assert ctx.principal_type == "mcp_client"
        assert ledger_grantee_type("mcp_client") == "user"

    def test_a_token_with_no_sub_is_refused_outright(self):
        """`resolve_auth`'s JWT branch requires `sub` before it dispatches at all, so an absent
        subject is a 401 and never a principal named by the client id."""
        from fastapi import HTTPException

        payload = _mcp_payload()
        payload.pop("sub")
        with pytest.raises(HTTPException) as exc:
            _resolve(payload)
        assert exc.value.status_code == 401

    def test_a_null_sub_names_nobody_rather_than_the_client(self):
        """The other shape: `sub` present but empty passes the `"sub" in payload` guard and
        reaches this branch. It must resolve to an EMPTY principal — which every consumer
        refuses — rather than falling back to the client id, which is the one string here that
        would look like an identity while holding no grants."""
        ctx = _resolve(_mcp_payload(sub=None))

        assert ctx.principal_id == "" and ctx.user_id is None
        assert ctx.principal_id != CLIENT
        # And it is genuinely unusable: no acting principal can be built from it.
        from mantle.services.acting_principal import NoActingPrincipal

        with pytest.raises(NoActingPrincipal):
            acting_from_auth(ctx)


class TestTheClientIsRetainedAsTheActor:
    """Which client acted for this user is worth keeping — it is just not the authorization key."""

    def test_aud_is_kept_as_the_actor(self):
        assert _resolve(_mcp_payload()).actor == CLIENT

    def test_an_array_aud_is_recorded_as_a_name_and_not_a_python_repr(self):
        """RFC 7519 allows `aud` to be an array. A bare `str` would write `"['a', 'b']"`
        into the access log — a name no client id search would ever match. It stays
        provenance either way; this is about the log being readable, not about a decision."""
        assert _resolve(_mcp_payload(aud=[CLIENT])).actor == CLIENT
        assert _resolve(_mcp_payload(aud=[CLIENT, "other"])).actor == f"{CLIENT},other"

    def test_the_subject_is_untouched_by_the_shape_of_aud(self):
        """Whatever `aud` turns out to be, it never reaches the principal."""
        ctx = _resolve(_mcp_payload(aud=[CLIENT, "other"]))
        assert ctx.principal_id == USER and ctx.user_id == USER

    def test_the_acting_principal_carries_both_halves(self):
        """`acting_from_auth` is the one reader of `principal_id`, and the reason the split
        was observable. It now sees the subject, and the client rides along as provenance."""
        actor = acting_from_auth(_resolve(_mcp_payload()))

        assert actor.principal_id == USER
        assert actor.principal_type == "mcp_client"
        assert actor.actor == CLIENT

    def test_the_actor_is_never_an_authorization_input(self):
        """Stated as a property of the tree, not of a code path: if nothing reads `actor` to
        decide access, then recording it here cannot have granted anything.

        Asserted as the property, not as a count. This required `len(reads) == 1` — the audit
        context being the only reader — which is a PROXY for "actor decides nothing", and the two
        come apart in both directions. It fired on 2026-08-26 for a second reader that was a
        `logger.warning` naming the actor of a refused platform-scoped token: provenance, in a log
        line, deciding nothing. Its own message asked the right question — *"is one a decision?"* —
        and the count could not answer it.

        Relaxing the number to 2 would have been the wrong repair: it buys silence until the
        third reader, and the third might be a branch. What matters is WHERE the value is read, so
        that is what this asks — `auth.actor` may be passed, stored or logged, and may never appear
        in a test, a comparison, or a boolean. A future reader that wants to consult it for access
        still has to come here and say so.
        """
        import ast
        import inspect

        from mantle.services import dependencies

        tree = ast.parse(inspect.getsource(dependencies))

        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        reads = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "actor"
            and isinstance(node.value, ast.Name) and node.value.id == "auth"
        ]
        assert reads, "auth.actor is no longer read at all — the audit record has lost the client"

        deciding = []
        for r in reads:
            node, hops = r, 0
            while node in parents and hops < 12:
                parent = parents[node]
                # The value steers control flow: `if auth.actor`, `x if auth.actor else y`,
                # `auth.actor ==`, `auth.actor and...`, `not auth.actor`.
                if isinstance(parent, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
                    deciding.append(ast.unparse(parent))
                    break
                if isinstance(parent, (ast.If, ast.IfExp, ast.While)) and node is parent.test:
                    deciding.append(ast.unparse(parent.test))
                    break
                # A call argument or a dict value is provenance — stop climbing; it is recorded,
                # not consulted.
                if isinstance(parent, (ast.Call, ast.Dict, ast.Assign, ast.Return)):
                    break
                node, hops = parent, hops + 1

        assert not deciding, (
            "auth.actor is consulted to decide something, which makes the machine's identity an "
            "authorization input: %s" % deciding)

    def test_the_audit_record_names_the_client(self):
        """A person who was not present must not be the whole story in the access log."""
        recorded: list = []
        auth = _resolve(_mcp_payload())
        store = MagicMock()
        store.get_active_grants_for_principal_resource = MagicMock(return_value=[])

        from mantle.services.dependencies import check_access
        from fastapi import HTTPException

        with patch("mantle.db.backend.get_raw_artifact", return_value={"_key": "art-1"}), \
             patch("mantle.db.backend.get_origin_parent", return_value=None), \
             patch("mantle.db.backend.get_active_grants_for_principal_resource",
                   return_value=[]), \
             patch("mantle.services.audit_service.record_access",
                   side_effect=lambda **kw: recorded.append(kw)):
            with pytest.raises(HTTPException):
                check_access(auth, "art-1", "read", store)

        assert recorded, "the decision was not witnessed at all"
        ctx = recorded[-1]["context"]
        assert ctx["actor"] == CLIENT
        assert ctx["principal_id"] == USER
        assert ctx["via"] == "mcp_client"


# ── the blast radius ──────────────────────────────────────────────────────────────────────────
class TestEveryOtherPrincipalTypeIsUnchanged:
    """One branch changed. This is the table that says so — a case per principal type, each
    asserting the whole (principal_id, principal_type, user_id, actor) tuple, so a future edit
    to any other branch cannot pass by only being locally plausible."""

    def test_a_user_resolves_to_itself(self):
        ctx = _resolve({"sub": "user-123", "aud": AUTHORITY_ISSUER})

        assert (ctx.principal_id, ctx.principal_type, ctx.user_id, ctx.actor) == (
            "user-123", "user", "user-123", None)

    def test_a_service_holds_no_user_and_no_actor(self):
        ctx = _resolve({"sub": "origin", "iss": "origin", "aud": "mantle",
                        "principal_type": "service"})

        assert (ctx.principal_id, ctx.principal_type, ctx.user_id, ctx.actor) == (
            "origin", "service", None, None)

    def test_a_delegation_still_normalizes_to_the_user_with_its_server_as_actor(self):
        """The precedent this change followed, unchanged by having followed it."""
        ctx = _resolve({"sub": "user-42", "aud": "agience-server-astra",
                        "iss": AUTHORITY_ISSUER, "act": {"sub": "agience-server-astra"},
                        "principal_type": "delegation", "host_id": "host-abc"})

        assert (ctx.principal_id, ctx.principal_type, ctx.user_id, ctx.actor) == (
            "user-42", "user", "user-42", "agience-server-astra")

    def test_a_grant_key_is_still_its_own_principal_and_still_has_no_user(self):
        """`user_id=None` on purpose: a key is not a person, and nothing downstream may
        mistake it for the issuing user and hand it that user's light cone."""
        root = GrantEntity(
            id="grant-root", resource_id="art-1", grantee_type="grant_key",
            grantee_id="hash", granted_by="user-123", can_read=True, state="active",
        )
        with patch("mantle.services.grant_key_service.authenticate", return_value=root), \
             patch("mantle.services.grant_key_service.resolve", return_value=[root]), \
             patch("mantle.services.grant_key_service.touch"):
            ctx = resolve_auth(token="agk_key", store_db=MagicMock())

        assert (ctx.principal_id, ctx.principal_type, ctx.user_id, ctx.actor) == (
            "grant-root", "grant_key", None, None)

    def test_a_server_is_still_refused_by_name(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            _resolve({"sub": "server/abc", "aud": "x", "principal_type": "server"})
        assert exc.value.status_code == 401

    def test_only_the_mcp_client_branch_names_aud_as_the_actor(self):
        """The four above carry `actor` from `act.sub` or not at all. If `aud` had leaked into
        another branch, one of these tuples would already have failed — this states the
        converse directly, so the intent is on the record rather than inferred."""
        assert _resolve(_mcp_payload()).actor == CLIENT
        assert _resolve({"sub": "user-123", "aud": AUTHORITY_ISSUER}).actor is None


# ── it widens nothing ─────────────────────────────────────────────────────────────────────────
class TestItWidensNothing:
    """The reachable set is the user's, and WAS the user's before this change.

    `check_access` keys on `user_id`, and the listing and recall paths resolve the light cone
    from `auth.user_id`. All three saw the user under the old resolution too. So the set of
    artifacts an `mcp_client` may reach is identical on both sides of this change; what moved
    is which principal's keys it may hold, and that moved from "a client id holding nothing"
    to "the same user those three call sites had already resolved".
    """

    def test_the_cone_is_the_users_and_the_client_id_holds_nothing(self):
        users_grant = GrantEntity(
            id="g1", resource_id="art-user", grantee_type="user", grantee_id=USER,
            granted_by=USER, can_read=True, state="active",
        )
        db = MagicMock()

        def _by_grantee(_db, grantee_id, grantee_type):
            assert grantee_type == "user", "the ledger's only principal grantee vocabulary"
            return [users_grant] if grantee_id == USER else []

        with patch("mantle.db.backend.get_active_grants_for_grantee", side_effect=_by_grantee), \
             patch("mantle.db.backend.list_origin_descendants", return_value=[]), \
             patch("mantle.services.context_service.reach") as reach:
            reach.return_value = MagicMock(ids=frozenset())
            resolver = LightConeResolver(db)

            as_subject = resolver.resolve(USER, "read", principal_type="mcp_client")
            as_client = resolver.resolve(CLIENT, "read", principal_type="mcp_client")

        assert as_subject == {"art-user"}
        # The old resolution asked THIS question, and this is the answer it got — an empty
        # cone, which is the refusal the tree shipped rather than any narrower authority.
        assert as_client == set()

    def test_the_reach_equals_a_plain_users_reach_and_does_not_exceed_it(self):
        """The exact statement of "no widening": for the same subject, an `mcp_client` cone and
        a `user` cone are the same set. `mcp_client` gets the user's grants — no more, and no
        scope narrowing either, because Origin says `scopes` is not an entitlement."""
        grant = GrantEntity(
            id="g1", resource_id="art-user", grantee_type="user", grantee_id=USER,
            granted_by=USER, can_read=True, state="active",
        )
        db = MagicMock()
        with patch("mantle.db.backend.get_active_grants_for_grantee",
                   side_effect=lambda _db, grantee_id, grantee_type: (
                       [grant] if grantee_id == USER else [])), \
             patch("mantle.db.backend.list_origin_descendants", return_value=[]), \
             patch("mantle.services.context_service.reach") as reach:
            reach.return_value = MagicMock(ids=frozenset())
            resolver = LightConeResolver(db)

            assert (resolver.resolve(USER, "read", principal_type="mcp_client")
                    == resolver.resolve(USER, "read", principal_type="user"))

    def test_check_access_was_already_keying_on_the_user(self):
        """The load-bearing historical fact. If this were ever false, the change WOULD widen —
        so it is asserted rather than recalled: the grant lookup is made with `user_id`, which
        for an `mcp_client` is `sub`, and was `sub` before this change too."""
        seen: list = []
        auth = _resolve(_mcp_payload())
        from mantle.services.dependencies import check_access
        from fastapi import HTTPException

        def _lookup(_db, grantee_id, resource_id):
            seen.append(grantee_id)
            return []

        with patch("mantle.db.backend.get_raw_artifact", return_value={"_key": "art-1"}), \
             patch("mantle.db.backend.get_origin_parent", return_value=None), \
             patch("mantle.db.backend.get_active_grants_for_principal_resource",
                   side_effect=_lookup), \
             patch("mantle.services.audit_service.record_access"):
            with pytest.raises(HTTPException):
                check_access(auth, "art-1", "read", MagicMock())

        assert seen and set(seen) == {USER}, "the access decision is made against the user"
        assert CLIENT not in seen
