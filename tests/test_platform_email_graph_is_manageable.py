"""The platform email graph must be manageable by the operator who owns it.

⛔ WHAT WAS WRONG. `platform_email` granted `can_read` (+ `can_invoke` on the operator artifact)
and nothing else, to both grantees. `grant_service` has no creator fast-path — "created_by is
provenance and grants no access, so the creator holds exactly what their explicit grant gives,
minted at creation" — so NO principal held `can_admin` on any of the four artifacts.

Measured 2026-09-11 against the live node: the operator, on artifacts he created, got 403 from
both `POST /grants` and `POST /grants/keys`, with the message "Only the resource creator or an
admin can manage grants" — naming a creator path that does not exist. The graph was frozen: a
send could not be delegated and an existing grant could not be revoked without deleting and
re-provisioning the credentials.

The provisioner's own comment shows delegation was intended: the grants are "Issued to BOTH the
operator (direct human sends) AND the platform system principal (webhook/background sends act AS
this principal)".

⚠ The operator gets `can_admin`; the system principal does not. It performs sends, which needs
read and invoke — not the ability to widen access to a credential.
"""
from __future__ import annotations

import ast
import pathlib

SRC = (pathlib.Path(__file__).resolve().parents[1]
       / "src" / "mantle" / "services" / "seed_provisioning" / "platform_email.py")


def _grant_call() -> ast.Call:
    """The single `db_upsert_user_collection_grant(...)` call, from the source.

    Parsed rather than executed: running the provisioner needs a store, an operator and live
    credentials, none of which a unit test should require to answer "what does it grant".
    """
    tree = ast.parse(SRC.read_text(encoding="utf-8-sig"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "db_upsert_user_collection_grant"]
    assert len(calls) == 1, f"expected exactly one grant call, found {len(calls)}"
    return calls[0]


def _kwargs(call: ast.Call) -> dict:
    return {k.arg: k.value for k in call.keywords if k.arg}


def test_the_parse_is_not_vacuous():
    """A rename would leave every assertion below passing against nothing."""
    kw = _kwargs(_grant_call())
    assert "can_read" in kw and "granted_by" in kw


def test_can_admin_is_granted():
    """⛔ THE REGRESSION. Without it the graph is immutable for everyone, including its owner."""
    kw = _kwargs(_grant_call())
    assert "can_admin" in kw, (
        "platform_email grants no can_admin, so no principal can ever add, change or revoke a "
        "grant on the email credentials — POST /grants and /grants/keys both answer 403"
    )


def test_can_admin_is_conditional_on_being_the_operator():
    """It must not be `True` for every grantee: the system principal performs sends and has no
    business widening access to the credential it uses."""
    kw = _kwargs(_grant_call())
    node = kw["can_admin"]
    assert not isinstance(node, ast.Constant), (
        "can_admin is a constant; it must be conditional so only the operator receives it"
    )
    rendered = ast.unparse(node)
    assert "operator" in rendered, f"can_admin does not key off the operator: {rendered}"


def test_the_system_principal_still_gets_read_and_invoke():
    """Background sends act as the system principal, so removing its read would break sending
    while looking like a tightening."""
    kw = _kwargs(_grant_call())
    assert ast.unparse(kw["can_read"]) == "True"
    assert "invoke" in ast.unparse(kw["can_invoke"])
