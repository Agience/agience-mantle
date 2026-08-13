#!/usr/bin/env python3
"""Stop + SessionEnd hook: archive the session's conversation into mantle as one artifact.

REGISTERED ON BOTH EVENTS, WRITING ONE ARTIFACT. `Stop` fires after each assistant turn and keeps
the artifact current WHILE the session runs; `SessionEnd` fires at the close and captures the
final state. Both are the same operation against the same `session:` index key, so the second
never duplicates the first.

Stop is what makes the store grow with the work instead of in one lump at the end, and it is also
the insurance: SessionEnd is not guaranteed to fire -- closing a window or losing the process
skips it -- so a capture layer that runs only there loses exactly the long sessions worth keeping.
Stop pays for that with a per-turn write, which is why `_MIN_GROWTH_CHARS` exists.

The third leg of the capture pipeline. `store_file.py` captures what was WRITTEN and
`recall_context.py` reads context back IN; without this, the reasoning that produced those files
-- what was asked, what was tried, what was rejected and why -- exists only in a local JSONL file
that nothing ever reads again. That reasoning is usually the part worth recalling in a later
session: the code is on disk already, the argument for why it looks that way is not.

Stores the RENDERED conversation, not the raw transcript. A Claude Code transcript is mostly tool
plumbing -- every file read, every command's full stdout -- which is bulk that buries the actual
exchange and would make `recall` answer with tool noise. What is kept is what a person said and
what the assistant said back, with tool activity reduced to the names of the tools used per turn.

ONE ARTIFACT PER SESSION, updated in place, on the same reasoning as `store_file.py`: SessionEnd
can fire more than once for a session (a clear, then a logout), and each firing describes the
same conversation rather than a new one.

Best-effort, like every hook here: any failure is silent and exits 0. A session ending is the
worst possible moment to raise -- there is no longer anyone to tell.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from mantle_common import log_event, read_stdin_json, store_artifact

#: Rendered-transcript ceiling.
#:
#: ⚠ 120_000, DOWN FROM 800_000, AND THE OLD VALUE WAS NOT MERELY GENEROUS -- IT WAS UNWRITABLE.
#: The reasoning it replaced ("a long artifact costs its storage, not its length") is true of the
#: READ path, where recall previews are density-cut, and false of the WRITE path, which encrypts
#: and SSE-indexes the whole body before answering. A capped-at-800_000 transcript measured 64.2s
#: to store, past every timeout in the chain, so the artifact landed server-side and every client
#: reported failure. The ceiling has to sit inside the write budget, not inside the disk budget.
#:
#: This also has to hold for the STOP path below, which rewrites the whole artifact after a turn:
#: whatever this number is, a session pays it repeatedly, not once.
_MAX_CHARS = 120_000

#: Stop fires after every assistant turn, and most turns add little. Re-encrypting and re-indexing
#: the entire transcript to append a sentence is the kind of cost that gets a capture layer turned
#: off, so a Stop-triggered write has to earn itself: skip unless the rendering grew by at least
#: this much since the last one. SessionEnd ignores this entirely -- the final state of a session
#: is worth one write no matter how small the last turn was.
#:
#: ⚠ 10_000 IS SET FROM A MEASUREMENT, NOT A GUESS, AND THE MEASUREMENT IS SURPRISING: an update
#: costs ~32s at 30K, 60K AND 120K chars alike -- flat in size, ~3x the cost of the equivalent
#: create. The price of a Stop write is therefore paid per WRITE, not per byte, so the only lever
#: that reduces it is writing less often. That is this constant, and it is why raising `_MAX_CHARS`
#: is nearly free while lowering this is not.
#:
#: A Stop hook BLOCKS the turn from completing, so every write that clears this threshold is a
#: visible stall. 10_000 is roughly a few substantive turns' worth of rendered prose -- frequent
#: enough that the store tracks the work, rare enough that the stall is not per-turn.
_MIN_GROWTH_CHARS = 10_000

#: ⛔ A TRANSCRIPT IS `text/markdown`, AND MINTING A TYPE FOR IT WAS A MISTAKE — reverted.
#:
#: `text/vnd.agience.transcript` was introduced here so the automatic recall could exclude
#: transcripts, and it bought that at the price of a category error: a content type says what
#: the BYTES ARE, and these bytes are markdown. What actually distinguished a transcript was
#: never its format but its PROVENANCE — what made it, and through which connection. Encoding
#: provenance in the format field means every new source of prose wants its own MIME type, and
#: the type stops describing content at all.
#:
#: Provenance goes in the context, where the rest of it already lives, as an OPERATOR. That is
#: the system's own third leg: an artifact is content + context + operator, and
#: `ARCHITECTURE-OF-RECORD` notes that all 117k WordNet synsets sharing ONE operator is the gap
#: rather than the design — "if training is the goal, widen the operators". Naming the operator
#: on every artifact a hook writes is exactly that widening, and it costs nothing, because
#: something already knew it at write time and was throwing it away.
TRANSCRIPT_CONTENT_TYPE = "text/markdown"

#: What made this artifact. `op.<domain>.<name>`, the convention already in use for
#: `op.content.mirror` and `op.source.wordnet`.
OPERATOR = "op.claude-code.transcript"

#: `session:<id>` -> the body size last stored, so the debounce below knows how much has been
#: added since. A CACHE, and it is worth being explicit about the difference from the id index
#: this replaced: losing this file costs one redundant write, because the identity of the
#: artifact does not live here. Losing the id index used to cost a permanent duplicate, because
#: the identity did.
#:
#: Deliberately a different file from the retired `mantle-file-artifacts.json`, so that one can
#: be deleted without taking the debounce with it.
_SIZE_FILE = Path(
    os.environ.get("MANTLE_SESSION_SIZES", str(Path.home() / ".claude" / "mantle-session-sizes.json"))
)


def _load_sizes() -> Dict[str, Any]:
    try:
        data = json.loads(_SIZE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_sizes(sizes: Dict[str, Any]) -> None:
    try:
        _SIZE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _SIZE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(sizes, indent=1), encoding="utf-8")
        os.replace(tmp, _SIZE_FILE)
    except OSError:
        pass


def _text_of(content: Any, tool_names: List[str]) -> str:
    """The human-readable text of one message's content, recording tool names as a side effect.

    Content is either a plain string or a list of blocks. Only `text` blocks carry the exchange;
    `thinking` is deliberately dropped (it is not what was said), and a tool block contributes
    its name only -- enough to see that a search or an edit happened at that point without
    pasting the file it returned.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif btype == "tool_use":
            name = block.get("name")
            if isinstance(name, str):
                tool_names.append(name)
    return "\n".join(p for p in parts if p.strip())


def _render(transcript_path: Path) -> str:
    """The transcript as markdown: who said what, in order, tool activity summarised."""
    lines: List[str] = []
    try:
        raw = transcript_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue

        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role") or entry.get("type")
        if role not in ("user", "assistant"):
            continue

        tool_names: List[str] = []
        text = _text_of(message.get("content"), tool_names).strip()
        if not text and not tool_names:
            continue

        # `## turn — role`, not a bare `## role`: assistant messages routinely CONTAIN markdown
        # headings, so a plain `## user` marker is indistinguishable from a heading someone
        # wrote, and anything reading this back cannot tell a turn boundary from a section
        # title. The `turn — ` prefix is not something prose produces.
        lines.append(f"## turn — {role}")
        if text:
            lines.append(text)
        if tool_names:
            seen = sorted(set(tool_names))
            lines.append(f"_(tools used: {', '.join(seen)})_")
        lines.append("")

    return "\n".join(lines)


def _cap(text: str) -> str:
    """Bound the stored size, keeping BOTH ends when it has to cut.

    The opening of a session says what was being attempted and the close says how it landed;
    a plain head-truncation keeps the question and throws away the answer, which is the half a
    later session is more likely to be looking for.
    """
    if len(text) <= _MAX_CHARS:
        return text
    head = _MAX_CHARS * 2 // 5
    tail = _MAX_CHARS - head
    return (
        text[:head]
        + f"\n\n---\n_[{len(text) - _MAX_CHARS} characters of the middle omitted]_\n\n---\n\n"
        + text[-tail:]
    )


def main() -> int:
    payload = read_stdin_json()

    # `Stop` fires after every assistant turn; `SessionEnd` once at the close. Both describe the
    # same conversation and write the same artifact -- the difference is only that Stop is
    # incremental and must not pay full price on a turn that added nothing, so it debounces below
    # while SessionEnd never does.
    event = payload.get("hook_event_name") or "SessionEnd"
    incremental = event == "Stop"

    # ⚠ THESE THREE EXITS USED TO BE SILENT, and that is why this hook looked like it had never
    # run at all: no artifact, no log line, nothing to distinguish "declined to store" from "never
    # invoked". A best-effort hook still has to say what it decided.
    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        log_event("archive_transcript", outcome="skipped", hook_event=event, reason="no-transcript-path")
        return 0
    path = Path(transcript_path)
    if not path.exists():
        log_event("archive_transcript", outcome="skipped", hook_event=event,
                  reason="transcript-missing", path=str(path))
        return 0

    body = _cap(_render(path))
    if not body.strip():
        log_event("archive_transcript", outcome="skipped", hook_event=event, reason="empty-render")
        return 0

    session_id = str(payload.get("session_id") or path.stem)
    cwd = payload.get("cwd") or os.getcwd()
    project = Path(cwd).name
    title = f"Session transcript — {project} — {session_id[:8]}"

    context = {
        "title": title,
        "tags": ["claude-code", "transcript", "session"],
        # ⭐ WHAT MADE THIS, beside what it says. `operator` is the semantic home; the matching
        # tag is what makes it FILTERABLE, because `recall` narrows on ten fields and
        # `operator` is not one of them while `tags` is. Two spellings of one fact is a cost
        # worth naming: the day `operator:` joins the filterable roster, the tag is redundant.
        "operator": OPERATOR,
        "project": cwd,
        "session_id": session_id,
        "end_reason": payload.get("reason") or "",
    }
    # ⚠ THE TAG IS HYPHENATED, NOT THE DOTTED OPERATOR NAME. `search/ingest/tags.normalize_tags`
    # keeps only `[a-z0-9-_ ]`, so `op.claude-code.transcript` INDEXES as
    # `opclaude-codetranscript` — and a colon inside a filter value is worse, since the query
    # parser splits on the first one. A spelling that survives normalisation unchanged and
    # carries no colon is the only one where what is written, what is indexed and what a filter
    # matches are the same three strings.
    context["tags"] = context["tags"] + [OPERATOR.replace(".", "-")]
    description = f"Claude Code session in {cwd} (reason: {payload.get('reason') or 'unknown'})"

    # ⭐ THE SESSION ID IS THE ARTIFACT'S IDENTITY. Mantle derives the id from this string, so
    # every `Stop` and the final `SessionEnd` write the same artifact — one session, one
    # artifact, with no id remembered anywhere and nothing for a concurrent hook process or a
    # lost reply to get wrong.
    #
    # What this replaces is worth recording, because it was measured rather than theorised. The
    # id used to live in a local index with no lock around read → create → write, so two `Stop`
    # firings close together both read "not tracked", both created, and one overwrote the
    # other's entry. On 2026-08-13 that produced SIX copies of one session in a 24-artifact
    # store, growing about one per turn, until five of recall's top hits were the same
    # transcript and real content could not place. Two successive patches — adopt-by-title
    # before creating, then reconcile-by-title after a timeout — narrowed the window without
    # closing it, because both were still reasoning about an id the client had to hold. A
    # derived id removes the question instead of answering it faster.
    identity = f"session:{session_id}"
    size_key = f"session-size:{session_id}"
    sizes = _load_sizes()

    # Debounce: a Stop that would rewrite the whole artifact to add a couple of sentences is not
    # worth the encrypt-and-index pass. Compared against what was last STORED, not what was last
    # seen, so a run that failed to write does not suppress the retry after it.
    if incremental:
        try:
            last_size = int(sizes.get(size_key, 0))
        except (TypeError, ValueError):
            last_size = 0
        if last_size and len(body) - last_size < _MIN_GROWTH_CHARS:
            log_event("archive_transcript", outcome="skipped", hook_event=event, reason="below-growth-threshold",
                      session=session_id, chars=len(body), since_last=len(body) - last_size)
            return 0

    stored = store_artifact(
        identity=identity, content=body, name=title, content_type=TRANSCRIPT_CONTENT_TYPE,
        description=description, context=context,
    )
    if stored:
        sizes[size_key] = len(body)
        _save_sizes(sizes)
        log_event("archive_transcript", outcome="stored", hook_event=event, session=session_id,
                  chars=len(body), id=stored)
        return 0

    # Unknown, not failed — the server finishes encrypting and indexing whether or not the
    # client is still listening. It needs no reconciling: the next Stop carries the same
    # identity and lands on the same artifact. The size cache is deliberately NOT advanced, so
    # that next write happens promptly rather than being debounced away.
    log_event("archive_transcript", outcome="unconfirmed", hook_event=event, session=session_id,
              chars=len(body))
    return 0


if __name__ == "__main__":
    sys.exit(main())
