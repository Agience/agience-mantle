"""The platform email provisioner must be CALLED, not merely defined.

⛔ WHAT WAS WRONG. `seed_provisioning/platform_email.py` is complete, documented and correct. Its
docstring says "Idempotent, non-fatal, runs every startup". `ensure_platform_email_sender` was
defined once and **called from nowhere** — not from `main`, not re-exported by the package, not
referenced by any test. Every deployment had no operator→authorizer→credential graph, so
`iris:send_email` answered `{"error": "No authorizer configured"}` while `GMAIL_OAUTH_*` sat
correctly configured in the environment.

A provisioner that is never invoked fails in the one way nothing surfaces: it logs nothing,
because it does not run. Its own INFO lines — "not fully configured", "operator not yet
resolvable", "provisioned email operator" — are all inside the function, so their absence looked
identical to a healthy no-op.

So the assertion here is not about what the function does. It is that the startup path reaches it.
"""
from __future__ import annotations

import ast
import pathlib

MAIN = pathlib.Path(__file__).resolve().parents[1] / "src" / "mantle" / "main.py"


def _called_names() -> set[str]:
    """Every function name called anywhere in `main.py`, read from the source.

    Parsed rather than imported: importing `mantle.main` runs module-level work and needs the
    whole service environment, which a unit test should not require to answer "is this called".
    """
    tree = ast.parse(MAIN.read_text(encoding="utf-8-sig"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                out.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                out.add(fn.attr)
    return out


def test_the_parse_is_not_vacuous():
    """If this returned nothing, every assertion below would pass trivially."""
    names = _called_names()
    assert len(names) > 50, f"only {len(names)} calls parsed — the reader is broken"
    # A neighbour that has always been called, so the reader is known to see this region.
    assert "seed_platform_issuer_artifacts" in names


def test_the_email_provisioner_is_called_at_startup():
    """⛔ THE REGRESSION. Defined, documented, and invoked by nothing."""
    assert "ensure_platform_email_sender" in _called_names(), (
        "mantle.main never calls ensure_platform_email_sender, so the platform email "
        "operator/authorizer/credential graph is never created and iris:send_email has no "
        "authorizer to resolve"
    )


def test_it_runs_under_a_system_acting_context():
    """It writes artifacts through the ordinary path, which needs an identity to get a content
    key. Startup has no request context, so the call must be inside `system_acting_context` —
    the same requirement the issuer seed beside it has."""
    src = MAIN.read_text(encoding="utf-8-sig")
    i = src.index("ensure_platform_email_sender(")
    window = src[max(0, i - 600):i]
    assert "system_acting_context" in window, (
        "the call is not inside a system_acting_context block — it will fail to obtain a "
        "content key and the graph will not be written"
    )


def test_the_function_still_exists_where_main_imports_it():
    """Guard on the guard: a rename would leave the string in `main.py` matching a function that
    is gone, and every assertion above would still pass."""
    from mantle.services.seed_provisioning import platform_email

    assert callable(platform_email.ensure_platform_email_sender)
