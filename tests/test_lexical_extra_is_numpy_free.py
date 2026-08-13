"""`mantle[lexical]` must genuinely not require numpy — verified by blocking numpy, not by grepping.

The install contract in `pyproject.toml` says the encrypted lexical arm (SSE blind tokens,
tokenizer, encrypted posting lists, blind-token narrowing) runs on stdlib + `cryptography`,
and that numpy belongs to `mantle[semantic]`. A dependency claim checked by reading import statements is worth very little:
it misses re-exports, lazy `__getattr__` paths that are lazy in name only, and anything a
transitive module drags in. So the claim is checked by removing numpy from the import system.

Failure modes:

  1. **Something on the lexical path starts importing numpy** (directly, or by importing a module
     that does). `test_lexical_surface_imports_and_works_without_numpy` fails, and its verdict
     names the module whose import raised.
  2. **The blocker stops blocking** — e.g. someone ports it to the legacy `find_module` API, which
     Python 3.12 ignores entirely. Every assertion below would then pass vacuously, having tested
     nothing. `test_blocker_actually_fires` is the guard: it asserts `import numpy` raises under
     the blocker, and it runs as its own test so the proof is visible in the suite rather than
     buried as a precondition.
  3. **The lexical surface imports but cannot do anything** — a split that moves the weight by
     breaking the feature. So the subprocess does real work under the blocker: tokenize, derive
     blind tokens, encrypt and decrypt a posting list, index an artifact through the real
     `SseIndexer`, and narrow a query back to it through the real `TokenNarrower` — asserting
     both the artifact it reached and the coverage count it carried.
  4. **The semantic surface silently stops needing numpy**, which would mean the split had drifted
     and `[semantic]` was over-declaring. `test_semantic_surface_still_requires_numpy` fails.

Every check runs in a subprocess: `sys.meta_path` finders are consulted only for modules not
already in `sys.modules`, and this pytest process imported numpy long ago (mantle's own suite, the
BLAS pin tests). Blocking numpy in-process would therefore block nothing and prove nothing — the
same vacuity as failure mode 2, arrived at from the other direction.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

# ── the blocker, as source, so the subprocess and the docstring cannot drift apart ──────────────
#
# `find_spec` on a `MetaPathFinder`. The legacy `find_module` hook was removed in Python 3.12 —
# a finder that only defines it is silently ignored, and a test built on one is vacuous.
_BLOCKER = '''
import sys, json

assert "numpy" not in sys.modules, (
    "numpy was already imported before the blocker went up; a meta_path finder is only consulted "
    "for modules NOT in sys.modules, so this run would prove nothing"
)

MARK = "numpy is BLOCKED by the negative control"


class NumpyBlocker:
    """Blocks numpy at the import system. `find_spec`, because 3.12 ignores `find_module`."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ImportError(f"{MARK}: {fullname}")
        return None          # everything else resolves normally


sys.meta_path.insert(0, NumpyBlocker())
v = {}

# ── 1. prove the blocker fires, before concluding anything from its silence ─────────────────────
try:
    import numpy
    v["blocker_fires"] = False
    v["blocker_detail"] = "import numpy succeeded under the blocker"
except ImportError as exc:
    v["blocker_fires"] = MARK in str(exc)
    v["blocker_detail"] = str(exc)
'''

_LEXICAL = '''
# ── 2. the lexical surface imports ──────────────────────────────────────────────────────────────
try:
    from mantle.search.mantle import sse
    from mantle.search import query_parser, types          # noqa: F401
    from mantle import db                                  # noqa: F401  the base store
    v["lexical_imports"] = True
except BaseException as exc:
    v["lexical_imports"] = False
    v["lexical_error"] = f"{type(exc).__name__}: {exc}"

# ── 3. and it works: index -> blind tokens -> encrypted postings -> narrow -> coverage ──────────
if v.get("lexical_imports"):
    try:
        key = b"k" * 32

        stems = sse.tokenize("The Observers were observing observable signals")
        assert stems, "tokenizer returned nothing"

        tok = sse.blind_token(key, sse.FIELD_CONTENT, stems[0])
        assert isinstance(tok, str) and len(tok) == 64, f"blind token shape: {tok!r}"
        assert sse.blind_token(key, sse.FIELD_CONTENT, stems[0]) == tok, "not deterministic"
        assert sse.blind_token(b"j" * 32, sse.FIELD_CONTENT, stems[0]) != tok, "key does not bind"

        # encrypted posting round-trip -- this is the `cryptography` dependency doing real work
        entries = [{"artifact_id": "a1", "collection_id": "c1", "field": "content"}]
        pk = sse.derive_posting_key(key, tok)
        blob = sse.pack_posting(entries, pk)
        assert isinstance(blob, (bytes, bytearray)) and blob != b""
        assert sse.unpack_posting(blob, pk) == entries, "posting round-trip lost data"

        # A full index-then-narrow round trip through the production classes, which is what
        # the arm actually does: write encrypted posting lists, then look terms up in them and
        # count how many of the query each artifact matched.
        class _Keys:
            def derive_sse_key(self, principal_id, request):
                return key

        store = sse.InMemoryPostingStore()
        sse.SseIndexer(_Keys(), store).index_artifact(
            "owner-A", "col-1", "a1",
            {"content": "observers observing observable signals"}, None,
        )
        found = sse.TokenNarrower(_Keys(), store).ids_for_stems(
            sse.tokenize("observers signals"), [("owner-A", "col-1")], None,
        )
        v["coverage"] = {a: list(c) for a, c in found.items()}
        assert set(found) == {"a1"}, f"the narrowing did not reach the indexed artifact: {found}"
        assert found["a1"].stems == 2, "both query stems are in the document"

        v["lexical_works"] = True
    except BaseException as exc:
        v["lexical_works"] = False
        v["lexical_work_error"] = f"{type(exc).__name__}: {exc}"

# numpy must still be absent after all of that
v["numpy_in_sys_modules"] = "numpy" in sys.modules

# ── 4. the semantic surface must fail, or the split is over-declared ────────────────────────────
try:
    import mantle.search.beacon.engine                     # noqa: F401
    v["semantic_fails"] = False
except ImportError as exc:
    v["semantic_fails"] = MARK in str(exc)
    v["semantic_detail"] = str(exc)
except BaseException as exc:
    v["semantic_fails"] = False
    v["semantic_detail"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(v))
'''


def _verdict() -> dict:
    """Run the blocker + probes in a clean interpreter and return the JSON verdict.

    `sys.path` is handed over in-band rather than through PYTHONPATH, so this works from a bare
    `pytest` with nothing exported."""
    prelude = f"import sys\nsys.path[:0] = {json.dumps(sys.path)}\n"
    proc = subprocess.run(
        [sys.executable, "-c", prelude + _BLOCKER + _LEXICAL],
        capture_output=True, text=True, env=dict(os.environ), timeout=300,
    )
    assert proc.returncode == 0, (
        f"probe process failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_blocker_actually_fires() -> None:
    """The precondition, promoted to a test: `import numpy` must raise under the blocker.

    Everything else in this file concludes from numpy's absence. If the blocker silently did
    nothing — the outcome for anyone who writes it against the legacy `find_module` hook, which
    Python 3.12 ignores — those conclusions would be vacuous and the suite would still be green.
    A guard that cannot be shown to fire is indistinguishable from no guard."""
    v = _verdict()
    assert v["blocker_fires"], (
        f"the numpy blocker did not fire, so every other assertion in this file would be "
        f"vacuous: {v.get('blocker_detail')}"
    )


def test_lexical_surface_imports_and_works_without_numpy() -> None:
    """The SSE lexical arm imports and does real work with numpy removed from the import
    system: tokenize, blind tokens, an encrypted posting round-trip, an index write, and a
    narrowing read back off it.

    Fails if anything on that path acquires a numpy edge — directly or transitively."""
    v = _verdict()
    assert v["blocker_fires"], "blocker did not fire; this result would mean nothing"
    assert v["lexical_imports"], f"lexical surface failed to import: {v.get('lexical_error')}"
    assert v["lexical_works"], f"lexical surface imported but broke: {v.get('lexical_work_error')}"
    assert not v["numpy_in_sys_modules"], "numpy reached sys.modules despite the blocker"
    assert v["coverage"] == {"a1": [2, 0]}, (
        "the index-then-narrow round trip produced no coverage: %r" % (v.get("coverage"),))


def test_semantic_surface_still_requires_numpy() -> None:
    """The other half of the split: `mantle.search.beacon.engine` must fail without numpy.

    If this ever passes, `[semantic]` is declaring a dependency the code no longer has — the split
    would have drifted, and the extras would be describing an arrangement that is no longer true."""
    v = _verdict()
    assert v["blocker_fires"], "blocker did not fire; this result would mean nothing"
    assert v["semantic_fails"], (
        f"the semantic surface imported without numpy: {v.get('semantic_detail')}. Either numpy "
        f"is not required there (so `[semantic]` over-declares) or the blocker missed it."
    )
