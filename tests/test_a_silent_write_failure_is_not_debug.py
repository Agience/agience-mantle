"""A failure that leaves durable wrong state must not be logged below the default level.

TWO WRITE-PATH FAILURES IN `workspace_service.create_workspace_artifact` WERE `logger.debug`:

    owner grant on create failed        -> the artifact exists and NO principal can reach it
    index enqueue failed                -> the artifact is stored and will never be searchable

Neither is a best-effort notification. The codebase uses `logger.debug(... failed)` widely and
correctly — an event that did not emit, a websocket send, an existence probe — where the caller has
already succeeded and the failure costs nothing durable. These two are the opposite: the write
COMPLETED, and what failed is the part that makes the result usable.

MEASURED 2026-08-26 why the first one matters. `host.71` was exactly this shape, from a different
writer: one row, genuinely probed content, `created_by` NULL, and `get_artifact` answering 404 on
every surface — for weeks. Authorization walks GRANTS (`check_access::_check_grants`); `created_by`
is never consulted. So an owner grant that fails silently produces a row that exists and cannot be
read, updated, or used as a container.

And the 404 cannot tell anyone which it is. `check_access` returns the same code for "absent" and
for "present but you hold no grant" — deliberately, so it is not an existence oracle. **The log line
is the only place that difference would ever be visible.**

The second was recorded independently in `_scratch/CURRENT/PUBLISHING.md` §4: all three enqueue
sites log, but one does so *below the default level* — "it logs without being seen", which is the
same as not logging at all.

This file deliberately does not forbid `logger.debug(... failed)` generally. There are ten or
more such sites and most are right. A check that flagged them all would be answered by lowering the
bar, not by raising these two.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_SVC = pathlib.Path(__file__).resolve().parents[1] / "src" / "mantle" / "services" / "workspace_service.py"

#: The failures whose consequence is a durable, unusable artifact. Named individually, with the
#: consequence, because that is what distinguishes them from the house pattern.
_MUST_BE_VISIBLE = [
    ("owner grant", "the artifact exists and no principal can reach it"),
    ("index enqueue", "the artifact is stored and will never be searchable"),
]


def _log_calls():
    """Every `logger.<level>(...)` in the file, as (level, first-argument-text, line)."""
    tree = ast.parse(_SVC.read_text(encoding="utf-8", errors="replace"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and getattr(f.value, "id", "") == "logger"):
            continue
        first = node.args[0] if node.args else None
        text = first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else ""
        out.append((f.attr, text, node.lineno))
    return out


def test_the_file_still_has_logging_to_check() -> None:
    """A derived set that quietly became empty would make the assertions below vacuous."""
    calls = _log_calls()
    assert len(calls) >= 5, "only %d logger calls found in %s" % (len(calls), _SVC.name)


@pytest.mark.parametrize("needle,consequence", _MUST_BE_VISIBLE, ids=[n for n, _ in _MUST_BE_VISIBLE])
def test_the_failure_is_logged_where_it_will_be_seen(needle: str, consequence: str) -> None:
    """`debug` is below the default level. A failure logged there is a failure nobody learns about
    until they go looking for something else."""
    matches = [(lvl, txt, ln) for lvl, txt, ln in _log_calls()
               if needle in txt.lower() and "fail" in txt.lower()]
    assert matches, (
        "no log call mentions a %r failure any more — either it was renamed (update this test) or "
        "the failure is now unlogged, which is worse than logging it quietly" % needle)
    for lvl, txt, ln in matches:
        assert lvl in ("warning", "error", "critical"), (
            "%s:%d logs a %r failure at `%s`, below the default level — so %s, and nothing says so."
            % (_SVC.name, ln, needle, lvl, consequence))


def test_the_message_names_the_consequence_not_just_the_event() -> None:
    """"index enqueue failed" tells a reader what did not happen, not what it costs them. A log
    line at warning level competes for attention; the one that earns it says what is now untrue
    about the system."""
    for lvl, txt, ln in _log_calls():
        low = txt.lower()
        if not any(n in low for n, _ in _MUST_BE_VISIBLE) or "fail" not in low:
            continue
        assert len(txt) > 40 and "—" in txt or " will " in low or " cannot " in low, (
            "%s:%d says %r — it names the event but not the consequence; a reader cannot tell "
            "whether to act" % (_SVC.name, ln, txt))
