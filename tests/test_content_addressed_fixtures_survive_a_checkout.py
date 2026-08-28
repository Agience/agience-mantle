"""A file whose NAME is the hash of its BYTES must be exempt from git's line-ending conversion.

The bug this exists to catch is invisible on the machine that commits it [2026-08-23].

`.gitattributes` opens with `* text=auto`, which asks git to GUESS whether a file is text. For an
OCI blob under `blobs/sha256/<hex>` it guesses wrong in the one way that matters: a manifest or
config blob is JSON, contains no NUL byte, and is therefore classified as text and line-ending
converted **on checkout**. Measured: 476 bytes become 491 — fifteen LF turned into CRLF — and the
content stops hashing to the name it is stored under. `mantle.oci.layout` then refuses it, which is
correct behaviour: verifying content against its ref is the entire contract of a content-addressed
store, and that refusal is what stops a bad file here becoming an unreadable blob later.

The blobs in the git OBJECTS are byte-correct. The damage happens at checkout. So the working tree
that captured the fixture keeps passing forever, and every FRESH CLONE fails — which is every CI
runner and every contributor. `test_oci_layout_real.py` was green here and red in the sovereign
plane's `_ci-work` checkout **at the same commit**, and that is how it was found: the first
cross-repo cycle run after 25 days, not by anyone reading the file.

So the assertion is about the attribute, not about the bytes. A test that only re-hashed the
files on disk would pass on the developer's machine for exactly the reason the bug is invisible
there. `git check-attr` reports what git WOULD DO to this path on a checkout, which is the thing
that is actually wrong. Both are checked below — the bytes because they are the property that
matters, the attribute because it is the property that travels.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Every tree in this repo whose filenames are hashes of the file's own bytes. One entry today; the
#: list is what a second captured layout gets added to, and the reason this is a list rather than a
#: constant is that the guard has to keep meaning something when that happens.
BLOB_TREES = [REPO / "tests" / "fixtures" / "oci-layout-real" / "blobs" / "sha256"]


def _blobs():
    out = []
    for tree in BLOB_TREES:
        if tree.is_dir():
            out.extend(sorted(p for p in tree.iterdir() if p.is_file()))
    return out


BLOBS = _blobs()


def test_there_are_blobs_to_check():
    """The negative control, and it is not decoration.

    Every assertion below is parametrised over a directory listing. If the fixture is renamed or
    moved, the parametrisation collects zero cases and this file reports green having checked
    nothing — the same vacuous pass `ci_runner` prints its denominator to catch. `test_oci_layout_real`
    skips itself when the fixture is missing, deliberately and with a stated reason; this one fails,
    because a fixture that is gone should be noticed by somebody.
    """
    assert BLOBS, (
        "no content-addressed blobs found under %s. If the fixture moved, move this guard with it; "
        "if it was deleted, delete this file rather than leaving a suite that measures nothing."
        % [str(t) for t in BLOB_TREES])


@pytest.mark.parametrize("blob", BLOBS, ids=lambda p: p.name[:12])
def test_a_blob_hashes_to_its_own_name(blob: Path):
    """The property itself. True here even when the checkout is wrong, which is why it is not enough
    on its own — but a fixture that is corrupt at rest is a different and worse fault, and nothing
    else in the suite would say so."""
    got = hashlib.sha256(blob.read_bytes()).hexdigest()
    assert got == blob.name, (
        "%s does not hash to its own name (got %s).\n"
        "  If this fails in a FRESH CLONE only, the cause is line-ending conversion and the fix is "
        "the `-text` rule in .gitattributes, not the fixture." % (blob.name, got))


@pytest.mark.parametrize("blob", BLOBS, ids=lambda p: p.name[:12])
def test_git_will_not_convert_a_blob_on_checkout(blob: Path):
    """The property that travels. `text` must be explicitly UNSET (`-text`), not merely absent.

    Absent is not safe: with no attribute the `* text=auto` at the top of `.gitattributes` applies,
    which is the whole defect. `unset` is what `-text` produces and it is the only answer that means
    "never convert this, in either direction".
    """
    rel = blob.relative_to(REPO).as_posix()
    proc = subprocess.run(["git", "check-attr", "text", "--", rel],
                          cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip("git is not available to answer check-attr here: %s" % proc.stderr.strip())

    # `path: text: <value>` — the value is what git would do to this file.
    value = proc.stdout.rsplit(":", 1)[-1].strip()
    assert value == "unset", (
        "git reports `text: %s` for %s, so a checkout may rewrite its line endings and it will stop "
        "hashing to its own name. This passes on the machine that committed the file and fails on "
        "every fresh clone.\n"
        "  Fix: a `-text -diff` rule covering its directory in .gitattributes." % (value, rel))
