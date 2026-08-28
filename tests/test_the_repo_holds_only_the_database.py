"""Mantle is the standalone database layer. This repository holds that, and not its clients.

The boundary is easy to cross. `git add -f` puts a file into this repository past the `.gitignore`
rule and past the commit that decided the question, and nothing objects afterwards. The ignore rule
was
advisory, the earlier decision was prose, and neither is a mechanism.

A cleanup would not prevent the next one. `git add -f` is one flag, the reasoning that justifies
it is always locally plausible ("it's just a reference copy"), and the cost lands months later on
whoever cannot tell which files this repo is answerable for. So the rule is asserted here, where
crossing it fails a run.

What the rule is. This repo is the DATABASE: the store, its retrieval, its authorization, its
wire surfaces, and the operator scripts those need. It is not the home of anything that merely
CALLS the database. A consumer — an editor's hooks, a capture script, a persona, a reduction
worker — has no more claim on this tree than any other client of the seven MCP tools, and being
written by the same person on the same day is not a claim.

The test is on TRACKED FILES rather than on files present, because that is the actual boundary: a
checkout may contain anything (`.data/`, a virtualenv, a scratch note), and what this repository
is answerable for is what it carries. It is also exactly what `-f` subverts, which is what makes
it the right thing to measure.

`test_attenuation_is_single_sourced.py` is the model — a sweep that fails on a second
implementation rather than a comment asking for one implementation.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Path prefixes that are a CLIENT's, never the database's. Each is a directory some tool wants
#: to keep its own configuration in; none describes the store.
#:
#: `.claude/` is the measured case. The others are here because the same argument admits them and
#: it is cheaper to say so now than to have the conversation again per editor.
_CLIENT_ONLY_PREFIXES = (
    ".claude/",        # Claude Code hooks, settings, agents — machine-specific by `2ec2a1f`
    ".cursor/",
    ".vscode/",
    ".idea/",
    ".aider/",
)


def _tracked_files() -> list[str]:
    """Every path this repository carries, from git itself.

    Skipped rather than failed when git is unavailable or this is not a checkout: the rule is
    about what the repository tracks, and where there is no repository there is nothing to
    assert. A test that failed there would be reporting on the environment, not on the tree.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:      # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")
    if out.returncode != 0:                                    # pragma: no cover
        pytest.skip("not a git checkout")
    return [ln.strip().replace("\\", "/") for ln in out.stdout.splitlines() if ln.strip()]


def test_no_client_tooling_is_tracked_here():
    """An editor's or an agent's own wiring does not live in the database's repository.

    The failure message names the files and the remedy, because the person who hits this will be
    mid-change and the useful thing is the next command, not a principle.
    """
    tracked = _tracked_files()
    offenders = sorted(
        path for path in tracked
        if any(path.startswith(prefix) for prefix in _CLIENT_ONLY_PREFIXES)
    )
    assert not offenders, (
        "Client tooling is tracked in the Mantle repository:\n  "
        + "\n  ".join(offenders)
        + "\n\nMantle is the standalone database layer. A consumer of its MCP surface — hooks, "
          "capture scripts, personas, reduction workers — belongs with the client, not here.\n"
          "Untrack (the working copies are unaffected):\n"
          "    git rm -r --cached " + " ".join(sorted({p.split("/")[0] for p in offenders}))
    )


def test_the_ignore_rules_are_not_being_bypassed():
    """Nothing tracked here is a file `.gitignore` says should not be.

    The general form, catching the mechanism rather than one instance. `git add -f` is how a
    tracked file comes to contradict the ignore rules, and it leaves no trace: the file exists,
    looking as legitimate as its neighbours, while the rule that should keep it out reads as if it
    is working. `git check-ignore` asks the
    rules directly, so a bypass shows up here whatever directory it happened in — including ones
    nobody thought to list above.
    """
    tracked = _tracked_files()
    if not tracked:                                            # pragma: no cover
        pytest.skip("nothing tracked")

    # `--no-index` is the whole point: without it git reports "not ignored" for anything already
    # tracked, which is precisely the state under test and would make this pass unconditionally.
    #
    # `-v` is not for readability — it is load-bearing. Its output names the RULE that matched,
    # and a rule beginning with `!` is a NEGATION, meaning the path is deliberately un-ignored.
    # Without it this test reports `.env.example` as a bypass: `.gitignore` ignores `.env.*` and
    # then re-admits `!.env*.example`, which is correct and intentional. A guard that flags correct
    # configuration gets switched off, so the negations are read rather than guessed at.
    #
    # Bytes rather than text. With `text=True` Python translates `\n` to `\r\n` writing to the pipe
    # on Windows, so git receives `.env.example\r` — a path that does not match the negation
    # `!.env*.example`, because the name does not end in `.example`. Git then reports the earlier
    # rule (`.env.*`) as the match and the file reads as a bypass, so the line ending changes the
    # answer rather than its rendering.
    proc = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "--no-index", "-v", "--stdin"],
        input="\n".join(tracked).encode("utf-8"), capture_output=True, timeout=120,
    )
    proc = subprocess.CompletedProcess(
        proc.args, proc.returncode,
        proc.stdout.decode("utf-8", "replace"), proc.stderr.decode("utf-8", "replace"),
    )
    # exit 0 = at least one path matched a rule, 1 = none matched, other = a real error.
    if proc.returncode not in (0, 1):                          # pragma: no cover
        pytest.skip(f"git check-ignore unavailable: {proc.stderr.strip()[:120]}")

    offenders = []
    for line in proc.stdout.splitlines():
        if "\t" not in line:
            continue
        rule, path = line.rsplit("\t", 1)
        pattern = rule.rsplit(":", 1)[-1]
        if pattern.startswith("!"):
            continue                                   # re-admitted on purpose
        # check-ignore echoes the path as given: the stdin newline leaves a CR on Windows, and
        # git quotes any path it had to escape. Neither is part of the path.
        offenders.append(path.strip().strip('"').strip())
    offenders = sorted(set(offenders))
    assert not offenders, (
        "These files are TRACKED but `.gitignore` says they should not be — someone used "
        "`git add -f`:\n  " + "\n  ".join(offenders)
        + "\n\nEither untrack them (`git rm --cached <path>`), or, if they genuinely belong to "
          "the database, change `.gitignore` so the rule and the tree agree. A tracked file that "
          "contradicts the ignore rules leaves no record of the decision that put it there."
    )
