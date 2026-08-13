#!/usr/bin/env python3
"""PostToolUse hook (matcher: Write|Edit): keep mantle's copy of a file current.

ONE ARTIFACT PER FILE -- not one per edit, and not one per lost reply. The file's resolved path
IS the artifact's identity: it goes over the wire as `identity`, Mantle derives the id from it,
and every write of that path lands on that one artifact. A `recall` therefore answers with the
file's current content, never with a copy of some earlier draft that happened to score better.

⭐ THERE IS NO LOCAL ID INDEX, and its absence is the fix rather than a simplification.

This hook used to map path -> artifact id in `~/.claude/mantle-file-artifacts.json` and consult
it to decide create-vs-update. That map was a second source of truth about identity, and it
failed the way second sources of truth fail: a write whose reply is lost still SUCCEEDS
server-side, so the entry was never recorded, and the next write of the same path created a
second root that nothing would ever reconcile. It also had no lock around read -> create ->
write, so two hook processes for one file could both read "not tracked" and both create.
Measured 2026-08-13, before the fix: this repo's README.md stored as two artifacts three
minutes apart, the older and larger of which was the stale one.

A recall-by-path-tag adoption step was added to paper over that, and it is gone too -- with a
derived id there is nothing to adopt, no extra round trip on the create path, and no race
window. `MANTLE_FILE_INDEX` is no longer read; the file can be deleted.

Reads the file back off disk after the tool ran, rather than trusting `tool_input` -- Write's
input carries the full content but Edit's carries only the diff (old_string/new_string), so disk
is the one place both tools' input shapes agree on "the current content." Best-effort: any
failure (mantle down, file gone, too large, not text) is silent and exits 0 -- this must never
turn a successful edit into a visible error.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from mantle_common import log_event, read_stdin_json, store_artifact

_MAX_BYTES = 512_000  # ~500KB; larger writes are probably generated/binary, skip rather than truncate

_CONTENT_TYPE_BY_SUFFIX = {
    ".py": "text/x-python", ".md": "text/markdown", ".json": "application/json",
    ".js": "text/javascript", ".jsx": "text/javascript", ".ts": "text/typescript",
    ".tsx": "text/typescript", ".yaml": "application/yaml", ".yml": "application/yaml",
    ".toml": "application/toml", ".html": "text/html", ".css": "text/css",
    ".sh": "text/x-shellscript", ".sql": "application/sql",
}


#: Suffixes worth storing. PROSE ONLY, and that is the whole point.
#:
#: Source code is already in git, on disk, and re-readable at any time -- storing it again buys
#: nothing and costs the corpus its signal. What it costs is not hypothetical: a 200KB source
#: file matches a query on many more distinct stems than a short note does, and `recall`'s
#: `coverage` ordering counts stems, so the big file wins on surface area every time. A store
#: that is 90% source ends up answering questions about reasoning with compilers.
#:
#: What is irreplaceable is the reasoning -- notes, decisions, docs, transcripts. None of that
#: is recoverable from a checkout, and all of it is what a later session actually needs.
#:
#: Set MANTLE_CAPTURE_ALL=1 to restore the old capture-everything behaviour.
_PROSE_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".adoc", ".org"}

#: Path fragments whose contents are ephemeral by construction. An OS temp file is written to be
#: thrown away; keeping it forever inverts the one thing its location was telling us.
#:
#: `_scratch` is deliberately NOT here. It reads like a temp directory and is not one -- it is a
#: working directory holding notes and write-ups meant to be kept, which is exactly the prose this
#: hook exists to capture. Machine-generated caches (`.git`, `node_modules`, `__pycache__`) stay
#: excluded because nothing in them was written by a person.
_EPHEMERAL_MARKERS = ("\\temp\\", "/temp/", "\\tmp\\", "/tmp/", "\\scratchpad\\", "/scratchpad/",
                      "\\.git\\", "/.git/",
                      "\\node_modules\\", "/node_modules/", "\\__pycache__\\", "/__pycache__/")


def _should_capture(path: Path) -> bool:
    """Is this file worth keeping in the lattice at all?"""
    if os.environ.get("MANTLE_CAPTURE_ALL") == "1":
        return True
    lowered = os.path.normcase(str(path))
    if any(marker in lowered for marker in _EPHEMERAL_MARKERS):
        return False
    return path.suffix.lower() in _PROSE_SUFFIXES


def _content_type_for(path: Path) -> str:
    return _CONTENT_TYPE_BY_SUFFIX.get(path.suffix.lower(), "text/plain")


def _identity(path: Path) -> str:
    """The name this file is stored under: `file:` + its resolved absolute path, case-folded.

    Resolved and absolute because the SAME file reached from a different working directory --
    or through a symlink, or with different capitalisation on Windows -- has to produce one
    name, or the store ends up with two artifacts for one file, which is the whole failure.

    ⭐ THIS TRAVELS TO THE SERVER. It used to be a local cache key; it is now the `identity`
    Mantle derives the artifact id from, so this string is what makes the write idempotent.
    Changing how it is computed re-points every file at a NEW artifact and orphans the old one
    -- the same break as renaming a primary key, and worth the same caution.

    The `file:` prefix keeps this caller's namespace tidy: identities are per-principal, so a
    file and a session cannot collide with each other by accident.
    """
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    return "file:" + os.path.normcase(str(resolved))


def main() -> int:
    payload = read_stdin_json()
    if payload.get("tool_name") not in ("Write", "Edit"):
        return 0

    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not file_path:
        return 0

    cwd = payload.get("cwd") or os.getcwd()
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(cwd) / path

    if not _should_capture(path):
        log_event("store_file", outcome="skipped", path=str(path), reason="not-prose-or-ephemeral")
        return 0

    try:
        if path.stat().st_size > _MAX_BYTES:
            return 0
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0

    try:
        rel = str(path.relative_to(cwd))
    except ValueError:
        rel = str(path)

    identity = _identity(path)
    content_type = _content_type_for(path)
    # ⭐ A TAG THAT NAMES THE FILE, because the title cannot. `title` is the RELATIVE path, so
    # "README.md" is the title of a README in every repo on this machine — recovering identity from
    # it would let one project's write adopt another project's artifact. `file_path` in the context
    # is unambiguous but `recall` does not return context, so it cannot be searched on. Tags are
    # both filterable AND returned in hits, so the identity goes there, hashed: the absolute path
    # of a source tree is not something to publish into a shared store as a searchable term.
    path_tag = "path-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    # `title` stays the readable relative label; `file_path` carries the absolute one, so an
    # artifact says both what to call it and which file on disk it mirrors.
    context = {"title": rel, "tags": ["claude-code", "file", path_tag], "project": cwd,
               "file_path": str(path)}
    description = f"File written by Claude Code in {cwd}"

    # One call, whether this file has been stored before or not. The server resolves `identity`
    # to an id and creates or updates accordingly, so there is no decision to make here, no
    # index to consult, and nothing a lost reply or a concurrent hook process can corrupt.
    stored = store_artifact(
        identity=identity, content=text, name=rel, content_type=content_type,
        description=description, context=context,
    )
    if stored:
        log_event("store_file", outcome="stored", path=rel, chars=len(text), id=stored)
    else:
        # Unknown rather than failed -- the write may well have landed. It needs no reconciling
        # either way: the next write of this path carries the same identity and lands on the
        # same artifact, so a missed reply costs one stale artifact until the next edit, never a
        # duplicate.
        log_event("store_file", outcome="unconfirmed", path=rel, chars=len(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
