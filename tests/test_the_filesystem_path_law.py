"""A caller-supplied id must never escape the index root, and must never collide with another.

`sse/file_stores.py` used to hold `FilePostingStore`, which was replaced by
`SqlitePostingStore` — but the module survived, because the part of it that was never about
postings is still load-bearing. `search/mantle/file_cell_store.py` — the VECTOR arm's local store —
is a directory tree, and it imports `encode_component`, `decode_component`, `_shard` and
`_atomic_write` from there rather than restating them. Two trees on one disk disagreeing about how
to escape an id is a bug that shows up as one owner's data in another owner's directory.

The path law is tested here, under its own name. Held alongside posting-store tests it travels
with a class that can be retired, and it is the only thing standing between an artifact id and an
arbitrary-file-write. The
tests below are that half, recovered and re-pointed at the consumer that remains.

Three properties, each stated as what breaks without it:

* **Reversible** — two distinct ids that encoded to one name would share one blob.
* **Single safe segment** — an id containing `/` or `..` would write outside the root.
* **Case-distinct** — on Windows and default macOS, `Owner-A` and `owner-A` would be one file.
"""
from __future__ import annotations

import os

import pytest

from mantle.search.mantle.file_cell_store import FileCellStore
from mantle.search.mantle.sse.file_stores import decode_component, encode_component


def _all_files(root: str) -> list:
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        out.extend(os.path.abspath(os.path.join(dirpath, f)) for f in filenames)
    return out


# ── the escaping itself ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", [
    "", "..", ".", "a/b", "..\\..\\x", "nul", "con", "aux", "prn", "com1", "lpt9",
    "~", "~~", "~7e", "Owner-A", "owner-a", "UPPER", "a b", "a\tb", "a\nb",
    "e\u0301clair", "\u4f60\u597d", "art-optics-1", "a" * 64,
    "user@example.com", "7b13537b-187e-4389-8d63-ed6547b2dfc6",
])
def test_escape_is_reversible(raw):
    """The escape must be lossless: two distinct ids that encoded to one filename would share one
    blob — one owner's data would silently overwrite another's — and any listing that decodes names
    back would return ids that were never written."""
    assert decode_component(encode_component(raw)) == raw


@pytest.mark.parametrize("raw", [
    "", "..", ".", "a/b", "..\\..\\x", "/etc/passwd", "C:\\Windows\\system32",
    "nul", "con", "aux", "prn", "com1", "com9", "lpt1", "lpt9", "~",
])
def test_an_escaped_name_is_one_safe_segment(raw):
    """An id containing a separator or `..` must not interpolate into a path that writes outside the
    root — that is an arbitrary-file-write primitive reachable from an artifact id.

    Windows device names are excluded too: `nul.enc` IS `NUL`, so a component that escaped to one
    would write to the null device and read back as absent.
    """
    encoded = encode_component(raw)
    assert encoded, "an escaped component must never be empty"
    assert "/" not in encoded and "\\" not in encoded and os.sep not in encoded
    assert "." not in encoded, "no dot means `..` is unconstructible"
    assert encoded.upper() not in {
        "CON", "PRN", "AUX", "NUL",
        *("COM%d" % i for i in range(1, 10)),
        *("LPT%d" % i for i in range(1, 10)),
    }


def test_an_escaped_name_is_lowercase_only():
    """Lowercase-only is what makes the case property below achievable at all: the escape cannot
    rely on case to distinguish two names, because the filesystem may not."""
    encoded = encode_component("Owner-A")
    assert encoded == encoded.lower()
    assert decode_component(encoded) == "Owner-A", "and it must still be reversible"


@pytest.mark.parametrize("bad", ["~z", "~zz", "~g0", "abc~", "~7", "~", "a~f"])
def test_a_malformed_escape_is_refused_rather_than_guessed(bad):
    """A truncated or non-hex escape raises. A decoder that returned a best-effort string would hand
    a caller an id that names a different thing — and both callers of it are enumerating a store to
    decide what to migrate or re-key.

    `~7` is in this list because it used not to be. `int` on the two-character slice accepts a
    one-character tail, so `~7` decoded to `""` — a name `encode_component` would have written
    as `~07`, and therefore never wrote. Tightening it to exactly two hex digits is safe by
    construction (the encoder emits `%02x`), and it turns "a directory this module did not write"
    back into the `ValueError` both callers already skip on.

    `"a~f"` covers the trailing-escape case with a one-digit tail after real content, and bare `"~"`
    is the empty-string sentinel — which is why it is tested through the encoder below rather than
    asserted to raise here.
    """
    if bad == "~":
        assert decode_component(bad) == "", "bare ~ is the empty-string sentinel, not an error"
        return
    with pytest.raises(ValueError):
        decode_component(bad)


def test_the_decoder_is_lenient_about_what_it_would_never_have_written():
    """Records the decoder's leniency rather than endorsing it. `decode_component`'s docstring says
    it "raises ValueError on a malformed name", which holds for malformed escapes alone. A name
    carrying characters the encoder never emits — uppercase, a dot, a separator — decodes through
    rather than raising.

    That matters because both callers of it are enumerating a store to decide what to migrate or
    re-key (`manage_sse_index._owner_dirs`, `FileCellStore.list_cells`), so a stray directory that
    nothing here wrote is reported as a principal id rather than skipped. Both callers wrap the call
    and skip on `ValueError`, which is the right shape — it just catches less than its name suggests.

    Pinned so the leniency is a decision someone can revisit rather than a surprise, and so
    tightening it shows up as this test failing rather than as a migration quietly covering less.
    """
    assert decode_component("UPPER") == "UPPER"
    assert decode_component("a.b") == "a.b"
    assert decode_component("a/b") == "a/b"
    assert decode_component(encode_component("")) == ""      # `~` alone is the empty sentinel


# ── and the path actually written ────────────────────────────────────────────────────────────


def test_a_traversing_id_stays_inside_the_root(tmp_path):
    """Proves the path actually written is under the root, not merely that the encoder would have
    been safe if it had been used. The store is the thing that has to use it."""
    root = str(tmp_path / "cells")
    store = FileCellStore(root)
    store.put("../../escapee", "../../also-escapee", b"x", cluster_id="../../cluster")

    written = _all_files(root)
    assert written, "nothing was written — the assertion below would pass vacuously"
    for path in written:
        assert os.path.commonpath([os.path.abspath(root), path]) == os.path.abspath(root), (
            f"{path} escaped the index root"
        )


def test_ids_differing_only_in_case_do_not_share_a_blob(tmp_path):
    """On a case-insensitive filesystem (Windows, default macOS), `Owner-A` and `owner-A` must not
    resolve to one file — that would let one principal read and overwrite another's encrypted data,
    a cross-principal leak created by the filesystem rather than by any code path that looks wrong.

    Only the OWNER's case varies; the rest is held identical. Varying more would make this pass for
    the wrong reason — the fan-out shard is sha256 of the escaped name, so two case-different names
    land in different shard directories and cannot collide even where the escaping is case-blind.
    The owner directory is the real hazard, because it is neither hashed nor sharded.
    """
    store = FileCellStore(str(tmp_path / "cells"))
    store.put("Owner-A", "col-1", b"upper", cluster_id="cluster-1")
    store.put("owner-a", "col-1", b"lower", cluster_id="cluster-1")
    assert store.get("Owner-A", "col-1", "cluster-1") == b"upper"
    assert store.get("owner-a", "col-1", "cluster-1") == b"lower"


def test_the_two_trees_escape_an_id_the_same_way(tmp_path):
    """The reason `file_stores.py` still exists: one path law, imported, not restated.

    `manage_sse_index` decodes owner directory names written by the retired posting store, and
    `FileCellStore` writes owner directory names now. If the two ever disagreed, a migration would
    read one owner's directory as another owner's id.
    """
    from mantle.search.mantle import file_cell_store
    from mantle.search.mantle.sse import file_stores

    assert file_cell_store.encode_component is file_stores.encode_component
    assert file_cell_store.decode_component is file_stores.decode_component
