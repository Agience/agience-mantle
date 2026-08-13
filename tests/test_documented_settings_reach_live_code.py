"""Every setting `.env.example` documents must reach code that reads it.

`platform_settings_service.DEFAULTS` already names this defect on its own side: "a dead default is
an operator control that visibly accepts a value and does nothing". `.env.example` is the same kind
of control with a wider audience — it is the file an operator copies to `.env` — and it carries the
same failure in a form that is harder to see, because a variable nothing reads produces no error, no
warning and no difference. It reads as configured.

`MANTLE_SIGNIN_URI` was that. Its only reader was `main._explorer_html`, a landing page
`ui/browse_page.py` superseded and which the same commit left behind with zero callers, so the
setting was documented as the switch that opens the sign-in door and reached nothing at all. The
check below is what would have caught it the moment the caller went away.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_ENV_EXAMPLE = _REPO / ".env.example"
_SRC = _REPO / "src" / "mantle"

#: `NAME=` or `#NAME=` at the start of a line — a documented setting, commented out or not.
_DOCUMENTED = re.compile(r"^#?([A-Z][A-Z0-9_]{2,})=", re.M)

#: A name in quotes, which is how one reaches `os.getenv`, `os.environ` or a `_SETTING_MAP` entry.
#: Matching the bare word instead would count a name that survives only in a comment explaining why
#: it was removed — exactly the state this test exists to reject.
_QUOTED = re.compile(r"""['"]([A-Z][A-Z0-9_]{2,})['"]""")


def _sources() -> str:
    files = sorted(_SRC.rglob("*.py"))
    assert len(files) > 100, (
        "only %d modules under %s — the glob is not reaching the package, and an absence-assertion "
        "over an empty corpus passes silently" % (len(files), _SRC))
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)


def test_every_documented_setting_is_named_somewhere_in_the_package():
    assert _ENV_EXAMPLE.is_file(), "%s is missing — the roster below would be empty" % _ENV_EXAMPLE
    documented = sorted(set(_DOCUMENTED.findall(_ENV_EXAMPLE.read_text(encoding="utf-8"))))
    assert len(documented) > 40, (
        "only %d settings parsed out of .env.example — the pattern stopped matching" % len(documented))

    quoted = set(_QUOTED.findall(_sources()))
    dead = [name for name in documented if name not in quoted]
    assert not dead, (
        "documented in .env.example and read by nothing: %r. A setting no module names is a control "
        "that accepts a value and does nothing, which is indistinguishable from working until "
        "someone depends on it. Either wire it up or delete the entry." % dead)


def test_the_signin_setting_is_gone_rather_than_documented_and_dead():
    """The specific instance, pinned so it cannot come back as a doc entry without a reader.

    Not a redundant restatement of the check above: this asserts the `.env.example` ENTRY is
    absent, where that one would be satisfied by re-adding the entry alongside any module that
    merely mentions the string.
    """
    env = _ENV_EXAMPLE.read_text(encoding="utf-8")
    assert not re.search(r"^#?MANTLE_SIGNIN_URI=", env, re.M), \
        "MANTLE_SIGNIN_URI is documented again — it needs a reader, not an entry"

    import mantle.config as cfg
    assert not hasattr(cfg, "SIGNIN_URI"), \
        "config.SIGNIN_URI is back with no consumer; browse_page runs the flow, it does not link out"

    import mantle.main as main
    for gone in ("_explorer_html", "_access_line"):
        assert not hasattr(main, gone), (
            "%s is back. `read_root` and `auth_callback` both render `ui/browse_page.render()`; a "
            "second HTML surface nothing serves is how a page rots unnoticed." % gone)
