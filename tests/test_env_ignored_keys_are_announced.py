"""A `.env` naming a variable this module bound at import is announced, not silently ignored.

`config.py` reads ten environment variables into module-level constants at import. `load_env()`
is called from `main.py` after that, so a `.env` naming any of them reaches `os.environ` too late:
the constant already took its value from the shell. The line is read, stored, and never consulted
— an operator sees a file that looks like configuration and is not.

Nothing said so. `load_dotenv(override=False)` does not report what it skipped, and afterwards a
variable that came from the shell is indistinguishable from one that was ignored.

The behaviour is unchanged — a `.env` still cannot set these, which is correct 12-factor
precedence. What changed is that failing to know is no longer free.
"""
from __future__ import annotations

import ast
import io
import logging
import os

import mantle.config as config


def test_the_bound_set_matches_what_the_module_actually_reads_at_import():
    """The tuple is a claim about this file; this re-derives it from the file.

    It also caught a stale count: an earlier note recorded six such variables and there are ten.
    A hand-maintained list of "things bound at import" is exactly the kind that drifts, and a new
    constant quietly rejoining the silent group is the failure this prevents."""
    src = io.open(config.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    derived = set()
    for node in tree.body:                      # MODULE level only — not inside functions
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name == "getenv" and sub.args and isinstance(sub.args[0], ast.Constant):
                derived.add(sub.args[0].value)

    declared = set(config._BOUND_AT_IMPORT)
    assert declared == derived, (
        "`_BOUND_AT_IMPORT` says %r; the module actually reads %r at import. Missing entries are "
        "variables a `.env` will ignore in silence."
        % (sorted(declared - derived), sorted(derived - declared)))


def test_an_ignored_key_is_named_in_the_warning(tmp_path, caplog):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "AUTHORITY_ISSUER=https://example.invalid\n"
        "export MANTLE_URI=https://also.invalid\n"
        "SOMETHING_ELSE=fine\n",
        encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        ignored = config._announce_ignored_env(env)

    # The warning's own words are "those lines have no effect" -- measured 2026-08-26 by running
    # it, `AUTHORITY_ISSUER` in a `.env` does take effect: `main.py` calls
    # `config.load_settings_from_db()` at :269, whose last line re-derives it from the environment.
    # Asserting the rule, not a literal list: a key a `.env` genuinely cannot set is named, and a
    # key it can set is not.
    assert ignored == ["MANTLE_URI"], ignored
    # `getMessage()` is what formats a LogRecord; `record.message % record.args` double-applies
    # the formatting and raises TypeError instead.
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "MANTLE_URI" in msg, msg
    assert "AUTHORITY_ISSUER" not in msg, (
        "a variable the rebind RECOVERS was reported as having no effect: %s" % msg)
    assert "SOMETHING_ELSE" not in msg, "a variable that is NOT bound at import was reported"


def test_the_warning_never_calls_a_working_line_dead(tmp_path, caplog):
    """The property the old assertion inverted, pinned on its own so it cannot drift back.

    Three of the ten names bound at import are recovered by `load_settings_from_db` --
    `AUTHORITY_ISSUER`, `ORIGIN_URI`, `FACET_URI` -- and a `.env` setting any of them works. Naming
    them in a warning that says *"those lines have no effect"* is not an over-count, it is a false
    statement about the most consequential identity value the file can carry.

    And the cost is the seven that are true: `AGIENCE_TRUSTED_ISSUERS` really is ignored, and an
    empty trusted set rejects every external token silently, presenting as a client fault rather
    than as configuration. **Three false names in a true warning is how a true warning gets
    ignored** -- so this is about the credibility of the other seven, not about tidiness.
    """
    recovered = sorted(config._rebound_from_env() & set(config._BOUND_AT_IMPORT))
    assert recovered, "nothing is recovered any more -- the derivation has stopped working"

    env = tmp_path / ".env"
    env.write_text("".join("%s=https://set.invalid" % k + chr(10) for k in recovered),
                   encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        ignored = config._announce_ignored_env(env)

    assert ignored == [], (
        "a `.env` naming only variables the rebind recovers was warned about: %s" % ignored)
    assert not caplog.records, (
        "silence is the correct answer here: %s" % [r.getMessage() for r in caplog.records])


def test_the_dangerous_keys_are_still_announced(tmp_path, caplog):
    """The other direction, and the one that costs data: narrowing the set must not go so far
    that a genuinely dead line goes unreported.

    `AGIENCE_TRUSTED_ISSUERS` is the case that matters: bound at import, never re-read, and an
    empty trusted set means every passthrough token from an external IdP is refused as an unknown
    issuer -- silently, and looking like the client's fault. `srv/foresight` is configured entirely
    around two external issuers, so this is a live node's failure mode, not a hypothetical."""
    env = tmp_path / ".env"
    env.write_text(chr(10).join(["AGIENCE_TRUSTED_ISSUERS=[]", "KEYS_DIR=/tmp/k",
                                "AGIENCE_OPERATOR_ID=op", ""]), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        ignored = config._announce_ignored_env(env)

    assert ignored == ["AGIENCE_OPERATOR_ID", "AGIENCE_TRUSTED_ISSUERS", "KEYS_DIR"], ignored
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "no effect" in msg, msg


def test_a_env_with_nothing_ignored_says_nothing(tmp_path, caplog):
    """The inverted guard. A warning on every `.env` would be noise, and noise gets filtered —
    at which point the real case is invisible again."""
    env = tmp_path / ".env"
    env.write_text("MANTLE_LATTICE_PATH=/tmp/x.db\nAGIENCE_NO_DOTENV=1\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert config._announce_ignored_env(env) == []
    assert not caplog.records, [r.message for r in caplog.records]


def test_an_unreadable_env_does_not_raise(tmp_path):
    """Diagnostics must not break boot. This runs inside `load_env` on the startup path."""
    assert config._announce_ignored_env(tmp_path / "does-not-exist") == []


def test_load_env_still_returns_the_file_it_loaded(tmp_path, monkeypatch):
    """The announcement is additive — `load_env`'s contract is unchanged."""
    monkeypatch.delenv("AGIENCE_NO_DOTENV", raising=False)
    (tmp_path / ".env").write_text("AUTHORITY_ISSUER=https://example.invalid\n", encoding="utf-8")
    try:
        found = config.load_env(tmp_path)
    finally:
        os.environ.pop("AUTHORITY_ISSUER", None)
    assert found is not None and found.name == ".env"
