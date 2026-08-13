"""Shared plumbing for the mantle-integration Claude Code hooks in this directory.

Every hook here is best-effort: mantle being down, slow, or misconfigured must never block a
prompt, a file edit, or a session ending. Every network call in this module is wrapped so a
failure returns None/empty instead of raising, and callers are expected to no-op on that.

Which node these hooks talk to
------------------------------
⭐ ONE SOURCE OF TRUTH, BECAUSE THERE ARE TWO CONSUMERS. Claude Code reaches mantle through the
`mcpServers.mantle` entry in `~/.claude.json` (a static bearer), and these hooks reach it through
this module (an OAuth refresh grant). Those are separate paths to the same node, and nothing
connects them — so pointing one at `home` while the other still answers from `dev` gives a session
whose recalled context comes from one store while its writes land in another, silently, with no
error on either side.

`~/.claude/mantle-target.json` is that single source of truth, and `mantle_target.py` is what
rewrites BOTH from it. Shape:

    {"active": "dev",
     "targets": {"dev":  {"mcp_url": ..., "token_url": ..., "client_id": ...,
                          "refresh_token_file": ..., "mcp_bearer": ...},
                 "home": {...}}}

Resolution order per value, most specific first:

    1. an explicit environment variable   — a deliberate one-off override, always wins
    2. the active target in that file     — the ordinary case
    3. the built-in default               — this machine's `71/dev` node

⚠ AN ABSENT OR BROKEN FILE IS NOT A FAILURE. It resolves to the defaults, which is exactly what
this module did before the file existed. A capture layer must not stop capturing because a config
file was mistyped.

⚠ SWITCHING TARGETS SWITCHES IDENTITY. Each node is its own authority with its own principals, so
the same person is a different `sub` on each — artifacts do not follow you across a switch. That is
the whole reason a promote path exists, and it is a property of the design rather than a gap in it.

The refresh token lives in a file OUTSIDE this repo (the user's home `.claude` dir) precisely so
it is never something `git add .` can pick up. Access tokens are short-lived (about 4 hours) and
are never cached to disk here -- each call mints a fresh one from the refresh token, which
Origin does not rotate (see agience-origin's `_grant_refresh_token`), so the same file keeps
working indefinitely without any hook here needing to write back to it. A token file is PER TARGET:
dev's refresh token is meaningless to home's Origin, so the two must never share a path.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

#: The switch file. Read once at import — a hook is a short-lived process, so re-reading per call
#: would buy nothing but a race with `mantle_target.py` rewriting it mid-run.
MANTLE_TARGET_FILE = Path(os.environ.get(
    "MANTLE_TARGET_FILE", str(Path.home() / ".claude" / "mantle-target.json")))


def _active_target() -> Dict[str, Any]:
    """The active target's settings, or {} when there is no usable file.

    Silent on every failure, by the same rule as the rest of this module: a missing, unreadable or
    malformed switch file must degrade to the built-in defaults rather than take the hooks down.
    Returning {} makes every lookup below fall through to its default with no branch of its own.
    """
    try:
        data = json.loads(MANTLE_TARGET_FILE.read_text(encoding="utf-8"))
        target = (data.get("targets") or {}).get(data.get("active") or "")
        return target if isinstance(target, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return {}


_TARGET = _active_target()


def _setting(env_var: str, key: str, default: str) -> str:
    """One config value, resolved env → active target → default (see the module docstring)."""
    from_env = os.environ.get(env_var)
    if from_env:
        return from_env
    value = _TARGET.get(key)
    return value if isinstance(value, str) and value else default


MANTLE_MCP_URL = _setting("MANTLE_MCP_URL", "mcp_url", "http://localhost:8182/mcp")
MANTLE_TOKEN_URL = _setting("MANTLE_TOKEN_URL", "token_url", "http://localhost:8180/auth/token")
MANTLE_CLIENT_ID = _setting("MANTLE_CLIENT_ID", "client_id", "dcr_gnnC5SZ17BvrW1UpEeBvwjM8jg7nMbru")
MANTLE_REFRESH_TOKEN_FILE = os.path.expanduser(_setting(
    "MANTLE_REFRESH_TOKEN_FILE", "refresh_token_file",
    str(Path.home() / ".claude" / "mantle-refresh-token")))

#: Which target these values came from — recorded on every `log_event` line so the hook log says
#: WHICH STORE it was talking to. Without it, two runs against different nodes are indistinguishable
#: in the log, and "the recall came back empty" cannot be told from "it came back empty over there".
MANTLE_TARGET_NAME = _TARGET.get("name") or os.environ.get("MANTLE_TARGET") or "default"

#: Reads. Short on purpose: `recall_context` runs before every prompt, so this is latency the
#: user waits through.
_HTTP_TIMEOUT_SECONDS = 5

#: Writes. A create/update carries the whole document and the server does content encryption, SSE
#: indexing and the density pass before it answers -- seconds for anything substantial. At the
#: read timeout a 15KB write COMPLETED SERVER-SIDE and then timed out waiting for the reply, so
#: the hook recorded a failure, never cached the returned id, and would have created a second
#: copy on the next write of the same file. A write that succeeds must not be reported as failed:
#: that is how duplicates get made by the very code meant to prevent them.
#:
#: ⚠ THE SAME BUG CAME BACK AT A LARGER SIZE, which is why this is 150 and not 60. A rendered
#: session transcript capped at the old 800_000 chars measured 64.2s to write -- over the 60s
#: budget -- so EVERY large transcript archived server-side and reported failure client-side, and
#: `archive_transcript` had never once recorded an id in its whole history. Note what does NOT
#: reproduce it: `'x '*400000` at the same byte count writes fine, because cost here is SSE
#: indexing over distinct terms, not bytes. Probe this path with real prose or it reads as healthy.
#:
#: 150 buys headroom, but the durable fix is that callers must treat a timeout as UNKNOWN rather
#: than FAILED and reconcile against the store -- see `_find_by_title` and its use in
#: `archive_transcript.py`. A timeout can always be provoked by a big enough document; only the
#: reconcile makes it non-damaging.
_HTTP_WRITE_TIMEOUT_SECONDS = 150


def _post_json(url: str, payload: Dict[str, Any] = None, *, form: Dict[str, str] = None,
                headers: Optional[Dict[str, str]] = None,
                timeout: int = _HTTP_TIMEOUT_SECONDS) -> Optional[Dict[str, Any]]:
    """POST JSON or form-encoded data; return the parsed JSON response, or None on any failure."""
    try:
        if form is not None:
            data = urllib.parse.urlencode(form).encode("ascii")
            req_headers = {"Content-Type": "application/x-www-form-urlencoded"}
        else:
            data = json.dumps(payload or {}).encode("utf-8")
            req_headers = {"Content-Type": "application/json"}
        req_headers.update(headers or {})
        req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError, ValueError):
        return None


def get_access_token() -> Optional[str]:
    """A bearer for the active target: a static one if it has one, else minted from the refresh token.

    ⭐ TWO WAYS IN, BECAUSE THE TWO NODES ARE AUTHENTICATED DIFFERENTLY. `dev` sits behind an Origin
    with dynamic client registration, so it has a `dcr_` client and a refresh token, and this mints
    a short-lived access token per call. `home` is its own authority whose mantle verifies a token
    signed by the keyset in its own `KEYS_DIR` — measured 2026-08-13: a 30-day token minted by
    `dev_mint_token.py --keys-dir <home>/keys` is accepted by `mantle.home.agience.ai` and resolves
    to the subject the keyset derives. So home needs no OAuth round trip, no registered client and
    no password, and the SAME bearer serves both this and Claude Code's `mcpServers` header.

    A static bearer is preferred when present rather than used as a fallback: reaching for a refresh
    grant against an Origin that has no client registered for it would fail slowly, once per hook
    run, to arrive at a token the target already holds.

    ⚠ IT EXPIRES. 30 days is long enough to forget and short enough to strand a node — re-mint with
    the same command and rewrite the target file. That is the cost of not running an OAuth client
    here, and it is the honest trade: the alternative is a refresh token that never expires at all.
    """
    static = _TARGET.get("mcp_bearer")
    if isinstance(static, str) and static.strip():
        return static.strip()
    try:
        refresh_token = Path(MANTLE_REFRESH_TOKEN_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not refresh_token:
        return None

    body = _post_json(MANTLE_TOKEN_URL, form={
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": MANTLE_CLIENT_ID,
    })
    if not body:
        return None
    return body.get("access_token")


#: Tools that carry a document and index it. Everything else is a read.
_WRITE_TOOLS = {"create_artifact", "update_artifact", "delete_artifact"}


def mcp_call(tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Call a mantle MCP tool; return its `structuredContent`, or None on any failure.

    A tool-level error (`isError: true` -- a 4xx/5xx the REST handler raised) also returns
    None: every caller here treats "mantle refused" the same as "mantle unreachable" -- nothing
    a best-effort hook can act on differently.
    """
    token = get_access_token()
    if not token:
        return None
    resp = _post_json(
        MANTLE_MCP_URL,
        {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json, text/event-stream",
        },
        timeout=(_HTTP_WRITE_TIMEOUT_SECONDS if tool_name in _WRITE_TOOLS
                 else _HTTP_TIMEOUT_SECONDS),
    )
    if not resp or "result" not in resp:
        return None
    result = resp["result"]
    if result.get("isError"):
        return None
    return result.get("structuredContent")


def recall(query_text: str, *, size: int = 5) -> List[Dict[str, Any]]:
    """The `recall` tool's hits, or [] on any failure or empty result."""
    data = mcp_call("recall", {"query_text": query_text, "size": size})
    if not data:
        return []
    return data.get("hits") or []


def find_by_title(title: str) -> Optional[str]:
    """The id of an artifact whose title is EXACTLY `title`, or None.

    ⭐ THIS IS THE TIMEOUT RECONCILER, and it exists because a write that times out is UNKNOWN,
    not failed. The server finishes encrypting and indexing regardless of whether the client is
    still listening, so `create_artifact` returning None covers two opposite outcomes: nothing was
    stored, or something was stored and we do not know its id. Treating both as failure is what
    turns one slow write into an unbounded pile of duplicates -- the caller re-creates next time
    because its index still shows nothing tracked.

    Matching is on the EXACT title rather than on relevance: `recall` scores, so its top hit for a
    session title is whatever scored best, which on a store holding several transcripts is not
    reliably the one just written. The caller's titles are deterministic (`store_file` uses the
    relative path, `archive_transcript` a session-derived string), so exact equality is available
    and is the only comparison that cannot adopt the wrong artifact.
    """
    for hit in recall(f'title:"{title}"', size=10):
        if isinstance(hit, dict) and hit.get("title") == title:
            artifact_id = hit.get("id")
            if isinstance(artifact_id, str) and artifact_id:
                return artifact_id
    return None


def _artifact_id_of(data: Optional[Dict[str, Any]]) -> Optional[str]:
    """The stored artifact's id out of a create/update response, or None.

    The write tools answer with the artifact object itself; `result` is checked too so a
    response that ever grows an envelope does not silently start returning None here.
    """
    if not data:
        return None
    if isinstance(data.get("id"), str):
        return data["id"]
    inner = data.get("result")
    if isinstance(inner, dict) and isinstance(inner.get("id"), str):
        return inner["id"]
    return None


def store_artifact(*, identity: str, content: str, name: str, content_type: str,
                   description: str = "",
                   context: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Store a thing under a stable NAME, creating or updating as needed. Returns its id.

    ⭐ THE IDEMPOTENT WRITE, and the reason these hooks no longer keep an id index.

    `identity` names the THING -- `file:c:/repo/README.md`, `session:7c7bcb7b` -- and Mantle
    derives the artifact id from it (`services/artifact_identity`). Calling this twice with one
    identity therefore leaves ONE artifact holding the newer content, on every path: whether the
    first call's reply arrived, whether this process ever saw it, whether two hook processes
    raced each other.

    That property is worth stating against what it replaces. The id used to be minted per write,
    so the only way to update rather than duplicate was to remember it locally -- and a write
    whose reply is lost still SUCCEEDS on the server, leaving the client with nothing recorded
    and the next write creating a second root that nothing would ever reconcile. Measured before
    this landed: one README as two artifacts three minutes apart, one session as five. None of
    those failure modes has a target here, because there is no remembered id to lose and no
    create-or-update decision for a race to get wrong.

    A `None` return is now genuinely just "the call did not come back". Retrying is safe and
    lands on the same artifact, which is what makes the timeout question uninteresting.
    """
    args = {
        "identity": identity, "content": content, "name": name,
        "content_type": content_type, "description": description,
    }
    if context is not None:
        args["context"] = json.dumps(context)
    return _artifact_id_of(mcp_call("create_artifact", args))


def create_artifact(*, content: str, name: str, content_type: str,
                    description: str = "", context: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Store a NEW artifact under a fresh id; return that id, or None on any failure.

    ⚠ Prefer :func:`store_artifact` for anything that might be written more than once. This one
    mints a new artifact per call by construction, so two calls about one thing leave two copies
    and `recall` answers with whichever scored best -- which may be either of them.
    """
    args = {
        "content": content, "name": name, "content_type": content_type,
        "description": description,
    }
    if context is not None:
        args["context"] = json.dumps(context)
    return _artifact_id_of(mcp_call("create_artifact", args))


def update_artifact(artifact_id: str, *, content: Optional[str] = None,
                    name: Optional[str] = None, content_type: Optional[str] = None,
                    description: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Rewrite an EXISTING artifact in place; return its id, or None on any failure.

    None on failure covers the case that matters to a caller holding a remembered id: the
    artifact was deleted server-side, so the id is stale. Treat it as "not there" and create
    instead -- that is what keeps a local id cache self-healing rather than permanently broken
    against a store someone cleaned up.
    """
    args: Dict[str, Any] = {"artifact_id": artifact_id}
    if content is not None:
        args["content"] = content
    if name is not None:
        args["name"] = name
    if content_type is not None:
        args["content_type"] = content_type
    if description is not None:
        args["description"] = description
    if context is not None:
        args["context"] = json.dumps(context)
    return _artifact_id_of(mcp_call("update_artifact", args))


#: Append-only record of what each hook did, one JSON object per line.
#:
#: These hooks are best-effort by design: every failure path returns quietly so a dead service
#: can never block a prompt or an edit. That is right, and it also means a hook that has stopped
#: working looks EXACTLY like a hook with nothing to say. Two separate faults hid behind that
#: this way -- a hook reading the wrong payload field, and a hook whose file did not exist -- and
#: both were invisible until someone went looking. This is the difference between "silently did
#: nothing" and "reported that it did nothing", and it is the only durable evidence that the
#: pipeline is running on every turn rather than merely configured to.
MANTLE_HOOK_LOG = Path(
    os.environ.get("MANTLE_HOOK_LOG", str(Path.home() / ".claude" / "mantle-hook.log"))
)

_LOG_MAX_BYTES = 2_000_000


def log_event(event: str, **fields: Any) -> None:
    """Record one hook firing. Never raises -- observability must not become a failure mode."""
    try:
        from datetime import datetime, timezone
        line = json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            # WHICH STORE this line is about. Two runs against different nodes are otherwise
            # identical in the log, and "recall came back empty" cannot be distinguished from
            # "it came back empty on the other node".
            "target": MANTLE_TARGET_NAME,
            **fields,
        })
        MANTLE_HOOK_LOG.parent.mkdir(parents=True, exist_ok=True)
        # Cheap bound: when it gets large, keep the recent half rather than growing forever.
        try:
            if MANTLE_HOOK_LOG.stat().st_size > _LOG_MAX_BYTES:
                kept = MANTLE_HOOK_LOG.read_text(encoding="utf-8").splitlines()[-1000:]
                MANTLE_HOOK_LOG.write_text("\n".join(kept) + "\n", encoding="utf-8")
        except OSError:
            pass
        with MANTLE_HOOK_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def read_stdin_json() -> Dict[str, Any]:
    """The hook's input payload from stdin. {} if it is missing or not valid JSON -- a hook
    that cannot parse its own input has nothing safe to do but no-op."""
    import sys
    try:
        return json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        return {}
