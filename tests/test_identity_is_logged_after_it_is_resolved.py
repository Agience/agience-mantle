"""The "Identity resolved" line must come after the thing that resolves identity.

`config.AUTHORITY_ISSUER` binds at import. `load_env` only mutates `os.environ` and does not touch
it; the rebind is `config.load_settings_from_db()`. Logging before that rebind would print
whatever the constant was at import — `http://localhost:8080` on a node configured through `.env`
— while the node goes on to run against the real issuer, a log describing a resolution 46 lines
before it happens.

That is worse than a missing log, and why it earns a test rather than a comment. That line is
the one place an operator looks to confirm which identity a node came up on. A wrong value there does
not merely fail to help: it sends someone hunting a `.env` that was in fact loaded correctly, and it
would survive any amount of testing that did not read a startup log with an external IdP configured.

The property is asserted on the source order, not by booting the app. Booting needs a store, a
keyset and a settings provider; the defect is a question of which line comes first, and that is
exactly what a future edit would get wrong.
"""
from __future__ import annotations

import pathlib
import re

import pytest

_MAIN = pathlib.Path(__file__).resolve().parents[1] / "src" / "mantle" / "main.py"

#: The identity constants. Each binds at import and is rebound by `load_settings_from_db`.
_CONSTANTS = ("AUTHORITY_ISSUER", "AUTHORITY_DOMAIN", "ORIGIN_URI", "MANTLE_URI")


def _lines():
    assert _MAIN.is_file(), "main.py is missing: %s" % _MAIN
    return _MAIN.read_text(encoding="utf-8", errors="replace").splitlines()


def _line_of(pattern: str, *, code_only: bool = True) -> int:
    """1-indexed line of the first match, ignoring comments."""
    for i, ln in enumerate(_lines(), 1):
        if code_only and ln.strip().startswith("#"):
            continue
        if re.search(pattern, ln):
            return i
    return -1


def test_the_pieces_are_all_still_there() -> None:
    """The precondition. If any of these is renamed the ordering test below would pass vacuously."""
    assert _line_of(r"config\.load_env\(") > 0, "main.py no longer calls config.load_env"
    assert _line_of(r"config\.load_settings_from_db\(") > 0, (
        "main.py no longer calls load_settings_from_db — that is what rebinds the config module "
        "from the environment, and without it the import-time defaults are what the node runs on")
    assert _line_of(r'"Identity resolved') > 0, "the Identity resolved log is gone"


def test_identity_is_logged_after_the_rebind() -> None:
    """The order is the whole property. Logging before the rebind prints the import-time default."""
    rebind = _line_of(r"config\.load_settings_from_db\(")
    log = _line_of(r'"Identity resolved')
    assert log > rebind, (
        "the 'Identity resolved' log is at line %d, BEFORE load_settings_from_db at line %d.\n"
        "  config.AUTHORITY_ISSUER binds at import; load_env only sets os.environ; the rebind is\n"
        "  what makes the constant true. Logged earlier, it reports http://localhost:8080 on a node\n"
        "  configured through .env — and that line is where an operator checks." % (log, rebind))


def test_nothing_reads_the_identity_constants_before_the_rebind() -> None:
    """The bigger version of the same risk, and the reason moving one log was enough.

    If anything between `load_env` and the rebind read `config.AUTHORITY_ISSUER`, it would be acting
    on the import-time default — and unlike a log, that would change behaviour rather than merely
    report it.
    """
    load_env = _line_of(r"config\.load_env\(")
    rebind = _line_of(r"config\.load_settings_from_db\(")
    offenders = []
    for i, ln in enumerate(_lines(), 1):
        if not (load_env < i < rebind):
            continue
        stripped = ln.strip()
        if stripped.startswith("#"):
            continue
        for const in _CONSTANTS:
            if re.search(r"config\.%s\b" % const, stripped):
                offenders.append("%d: %s" % (i, stripped[:90]))
    assert not offenders, (
        "these read an identity constant before load_settings_from_db rebinds it, so they see the "
        "import-time default:\n  %s" % "\n  ".join(offenders))


@pytest.mark.parametrize("const", _CONSTANTS)
def test_the_log_reports_resolved_values_not_raw_environment(const: str) -> None:
    """`os.getenv` would be a different answer, and the wrong one.

    The environment is one input to resolution; stored settings are another, and
    `load_settings_from_db` combines them. What an operator needs is the value the verifier will
    use — `iss` and `aud` are checked against `config.*`, not against the environment. A log reading
    `os.getenv` would be right in the common case and wrong in exactly the case worth logging.
    """
    text = "\n".join(_lines())
    m = re.search(r'logger\.info\(\s*\n?\s*"Identity resolved.*?\)\n', text, re.S)
    assert m, "could not locate the Identity resolved call"
    call = m.group(0)
    if const == "AUTHORITY_DOMAIN":
        return          # not reported by this line; listed above only for the read-before-rebind check
    assert re.search(r"config\.%s\b|getattr\(config, \"%s\"" % (const, const), call), (
        "the Identity resolved log no longer reports %s from `config` — if it now reads os.getenv, "
        "it is reporting an input rather than the resolved value the verifier uses" % const)
