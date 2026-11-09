"""Make text another person typed inert before it reaches someone's inbox.

An invite mail carries text its sender chose: their personal message, their display name and the
name of what they share.
Mail clients turn `example.com`-shaped text into a link even in a plain-text message, so escaping
markup is not enough; the text itself has to stop looking like a link.

`defang(text)` rewrites every URL-like thing as text a person can still read:

  schemes        `https://` -> `hxxps://`, `http://` -> `hxxp://`, `ftp://` -> `fxp://`; any other
                 scheme that makes a link has one letter bracketed: `javascript:` -> `jav[a]script:`
  hosts          every dot of a host is bracketed: `www.evil.example` -> `www[.]evil[.]example`,
                 including the dots IDNA treats as dots (`。` `．` `｡`), Unicode hosts and
                 `xn--` punycode labels; IPv4 addresses the same way
  markdown       `[text](url)` -> `text (url)`, with the url defanged
  HTML anchors   `<a href="url">text</a>` -> `text (url)`, with the url defanged

The rules are the ones the website's contact form applies to lead mail (`www.agience.ai`,
`bff/defang.py`) and Origin applies to its invites (`origin.services.defang`), kept the same here
so a link is inert by one definition wherever Agience mails what a person typed.
"""
from __future__ import annotations

import re

#: Schemes that make a link without `//` after the colon. Any scheme followed by `//` is matched
#: whether listed or not.
_LINK_SCHEMES = (
    "https", "http", "ftps", "ftp", "sftp", "file", "javascript", "vbscript", "data", "mailto",
    "tel", "sms", "callto", "magnet", "wss", "ws", "ssh", "telnet", "gopher", "ircs", "irc",
    "ldap", "smb", "nntp", "webcal", "view-source", "blob", "intent", "jar", "ms-settings",
)
_BEFORE_SCHEME = r"(?<![\w+.\-\[\]])"
_SCHEME_RE = re.compile(
    # Any scheme followed by `//`, a line break between them included. Already-defanged ones are left.
    _BEFORE_SCHEME + r"(?!(?:hxxps?|fxps?)\b)([A-Za-z][A-Za-z0-9+.\-]{1,31})(?=:\s*//)"
    # A listed scheme followed directly by its target: `javascript:alert(1)`, `mailto:a@b`.
    + r"|" + _BEFORE_SCHEME + r"(" + "|".join(re.escape(s) for s in _LINK_SCHEMES) + r")(?=:\S)",
    re.IGNORECASE,
)

#: The full stop and the three characters IDNA maps to it.
_DOT = r"[.。．｡]"
_LABEL = r"[^\W_][\w\-]*"
#: A top-level label: letters in any script (so a lookalike TLD counts) or punycode.
_TLD = r"(?:[^\W\d_]{2,63}|[Xx][Nn]--[\w\-]+)"
_HOST_RE = re.compile(r"(?<![\w\-])" + _LABEL + r"(?:" + _DOT + _LABEL + r")*" + _DOT + _TLD
                      + r"(?![\w\-])")
_IPV4_RE = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?![\w.]*\d)")
_DOT_ONLY_RE = re.compile(_DOT)

_MARKDOWN_LINK_RE = re.compile(r"(!?)\[([^\[\]]*)\]\(\s*<?([^()\s<>]*)>?(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)")
_HREF_RE = re.compile(r"""\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE)
_ANCHOR_RE = re.compile(r"<a\b([^>]*)>(.*?)</a\s*>", re.IGNORECASE | re.DOTALL)
_ANCHOR_OPEN_RE = re.compile(r"<a\b([^>]*)>", re.IGNORECASE)
_ANCHOR_CLOSE_RE = re.compile(r"</a\s*>", re.IGNORECASE)


def _defang_scheme(word: str) -> str:
    low = word.lower()
    if low in ("http", "https", "ftp", "ftps"):
        # Replace the `t`s after the first letter, keeping case: HTTPS -> HXXPS, ftp -> fxp.
        head = "".join(("X" if c.isupper() else "x") if c in "tT" else c for c in word[1:3])
        return word[0] + head + word[3:]
    i = 3 if len(word) >= 4 else len(word) - 1
    return f"{word[:i]}[{word[i]}]{word[i + 1:]}"


def _href(attrs: str) -> str:
    m = _HREF_RE.search(attrs)
    return next((g for g in m.groups() if g is not None), "") if m else ""


def _unlink(text: str) -> str:
    """Strip the link form of markdown links and HTML anchors, keeping their text and target."""
    text = _MARKDOWN_LINK_RE.sub(lambda m: f"{m.group(1)}{m.group(2)} ({m.group(3)})", text)

    def anchor(m: re.Match) -> str:
        href = _href(m.group(1))
        return f"{m.group(2)} ({href})" if href else m.group(2)

    text = _ANCHOR_RE.sub(anchor, text)
    text = _ANCHOR_OPEN_RE.sub(lambda m: f"({_href(m.group(1))})", text)
    return _ANCHOR_CLOSE_RE.sub("", text)


def defang(text: str) -> str:
    """`text` with every URL-like thing rewritten as inert text. Text with none is returned as is."""
    text = _unlink(text)
    text = _SCHEME_RE.sub(lambda m: _defang_scheme(m.group(0)), text)
    text = _HOST_RE.sub(lambda m: _DOT_ONLY_RE.sub("[.]", m.group(0)), text)
    return _IPV4_RE.sub(lambda m: m.group(0).replace(".", "[.]"), text)
