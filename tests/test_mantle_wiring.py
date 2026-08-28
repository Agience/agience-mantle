"""Tests for `search.mantle.wiring` — production accessor + indexer builders.

These exercise the graceful-degradation contract: when any prerequisite
(encryption key, S3 client, the lattice handle) is missing, the builder
returns ``None`` so callers can fall back rather than silently using
ephemeral in-memory stores.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet

from mantle.search.mantle import wiring
from mantle.search.mantle.file_cell_store import FileCellStore
from mantle.search.mantle.indexer import MantleIndexer
from mantle.search.mantle.s3_cell_store import S3CellStore
from mantle.search.mantle.sse import S3PostingStore, SqlitePostingStore


class _FakeStoreDB:
    pass


class _FakeS3Client:
    def get_object(self, **_):
        raise RuntimeError("not used in builder smoke tests")

    def put_object(self, **_):
        return {}

    def delete_object(self, **_):
        return {}

    def list_objects_v2(self, **_):
        return {"Contents": []}


def _patch_oracle_ok():
    """Supply the platform key `_build_oracle()` really resolves, at the module that really owns it.

    `wiring` has no `get_encryption_key` of its own and never did: it reaches the key through
    `key_provider.build_key_provider()` → `prism.trust.key_manager.get_encryption_key`. This helper
    used to `patch("…wiring.get_encryption_key", create=True, …)`, and `create=True` is what made
    that survive — it manufactures the attribute instead of raising the AttributeError that would
    have said the target does not exist. A mock production can never reach is worse than no mock:
    it looks like the prerequisite was satisfied.

    The singleton is cleared alongside it, because a cached oracle short-circuits the build and the
    patch would again make no difference.
    """
    return [
        patch("prism.trust.key_manager.get_encryption_key", return_value=Fernet.generate_key()),
        patch.object(wiring, "_oracle_singleton", None),
    ]


def _patch_content_service_ok():
    return [
        patch.object(wiring, "_build_cell_store", lambda *a, **k: S3CellStore(_FakeS3Client(), bucket="b")),
    ]


# ---------------------------------------------------------------------------
# _build_oracle — the prerequisite every builder below stubs out
# ---------------------------------------------------------------------------


class TestBuildOracle:
    """Every other test in this file replaces `_build_oracle` with a fake, so nothing here would
    notice if the real one stopped resolving a key. This pins the seam those fakes stand in for.
    """

    def test_the_platform_key_is_what_decides_whether_an_oracle_exists(self, monkeypatch):
        """Two halves, and the first is the control: with the platform key removed the build must
        return None. Without that half, the second assertion would pass on the suite-wide key that
        `conftest._init_test_encryption_key` installs, and would keep passing while pointed at a
        patch target production never reads — which is exactly what `create=True` used to hide.
        """
        import prism.trust.key_manager as km

        monkeypatch.delenv("MANTLE_KEK_PROVIDER", raising=False)
        monkeypatch.setattr(km, "_encryption_key", None, raising=False)

        with patch.object(wiring, "_oracle_singleton", None):
            assert wiring._build_oracle() is None, (
                "an oracle was built with no platform key — the control half of this test is not "
                "controlling anything, so the assertion below would prove nothing")

        with ExitStack() as stack:
            for p in _patch_oracle_ok():
                stack.enter_context(p)
            assert wiring._build_oracle() is not None, (
                "the injected key did not reach `_build_oracle` — `_patch_oracle_ok` is aimed at a "
                "name production does not read")

    def test_no_oracle_when_kek_custody_is_unavailable(self):
        """A KEK provider that cannot be constructed is "no search" (503), never a plaintext path."""
        with ExitStack() as stack:
            stack.enter_context(patch.object(wiring, "_oracle_singleton", None))
            stack.enter_context(patch(
                "mantle.search.mantle.key_provider.build_key_provider",
                side_effect=RuntimeError("no KEK custody on this box")))
            assert wiring._build_oracle() is None


# ---------------------------------------------------------------------------
# build_indexer
# ---------------------------------------------------------------------------

class TestBuildIndexer:
    def test_returns_indexer_when_prereqs_satisfied(self):
        with patch.object(wiring, "_build_oracle", lambda: _FakeOracle()), \
             patch.object(wiring, "_build_cell_store", lambda *a, **k: S3CellStore(_FakeS3Client(), "b")):
            indexer = wiring.build_indexer(_FakeStoreDB())
        assert isinstance(indexer, MantleIndexer)

    def test_returns_none_when_any_prereq_missing(self):
        with patch.object(wiring, "_build_oracle", lambda: None):
            assert wiring.build_indexer(_FakeStoreDB()) is None


# ---------------------------------------------------------------------------
# build_sse_indexer
# ---------------------------------------------------------------------------


class TestBuildSseIndexer:
    def test_returns_none_when_oracle_unavailable(self):
        with patch.object(wiring, "_build_oracle", lambda: None), \
             patch.object(wiring, "_build_sse_store", lambda *a, **k: _s3_posting_store()):
            assert wiring.build_sse_indexer(_FakeStoreDB()) is None

    def test_returns_none_when_the_posting_store_is_unavailable(self):
        with patch.object(wiring, "_build_oracle", lambda: _FakeOracle()), \
             patch.object(wiring, "_build_sse_store", lambda *a, **k: None):
            assert wiring.build_sse_indexer(_FakeStoreDB()) is None

    def test_returns_indexer_when_prereqs_satisfied(self):
        from mantle.search.mantle.sse import SseIndexer

        with patch.object(wiring, "_build_oracle", lambda: _FakeOracle()), \
             patch.object(wiring, "_build_sse_store", lambda *a, **k: _s3_posting_store()):
            indexer = wiring.build_sse_indexer(_FakeStoreDB())
        assert isinstance(indexer, SseIndexer)

    def test_the_indexer_takes_one_store_where_it_took_two(self):
        """The second was the BM25 corpus-statistics store. Nothing computes a corpus
        statistic, so there is nothing for it to hold and no backend built for it — which is
        why `_build_sse_store` is singular."""
        with patch.object(wiring, "_build_oracle", lambda: _FakeOracle()), \
             patch.object(wiring, "_build_sse_store", lambda *a, **k: _s3_posting_store()):
            indexer = wiring.build_sse_indexer(_FakeStoreDB())
        assert isinstance(indexer._postings, S3PostingStore)
        assert not hasattr(indexer, "_stats")


# ---------------------------------------------------------------------------
# build_sse_search_accessor
# ---------------------------------------------------------------------------


def _s3_posting_store():
    return S3PostingStore(_FakeS3Client(), "b")


def _all_three_present(**overrides):
    """Patches for a node with an oracle, a posting store and a cell store.

    ``overrides`` replaces one of them with whatever a test wants — ``None``, to take one
    prerequisite away and read what the builder does about it.
    """
    built = {
        "_build_oracle": lambda: _FakeOracle(),
        "_build_sse_store": lambda *a, **k: _s3_posting_store(),
        "_build_cell_store": lambda *a, **k: S3CellStore(_FakeS3Client(), "b"),
    }
    for name, value in overrides.items():
        built[name] = (lambda v: (lambda *a, **k: v))(value)
    return [patch.object(wiring, name, fn) for name, fn in built.items()]


#: The three names, so the parametrized sweep below cannot drift from the check it measures.
PREREQUISITES = ("_build_oracle", "_build_sse_store", "_build_cell_store")


class TestEveryQueryPrerequisiteIsHard:
    """Take any one of the three away and the query builder refuses. No degraded mode.

    A best-effort cell store reads backwards: a node with no SSE store answers 503 while a node
    with no cell store answers 200 with an empty list. Recall narrows on the postings and ranks on
    the cells, so a missing cell store is the state in which nothing can rank — and a missing
    posting store is worse still, because the only two things a recall can do without a narrowing
    are widen to the caller's whole light cone or silently match nothing.
    """

    @pytest.mark.parametrize("missing", PREREQUISITES)
    def test_the_router_accessor_refuses(self, missing):
        """Which is what the router turns into a 503 — see
        `tests/test_router_artifacts.py::test_503_when_sse_prereqs_missing`."""
        with ExitStack() as stack:
            for p in _all_three_present(**{missing: None}):
                stack.enter_context(p)
            assert wiring.build_sse_search_accessor(_FakeStoreDB()) is None

    def test_all_three_present_builds_one(self):
        """The control. Without it every refusal above could be a builder that never works."""
        from mantle.search.mantle.sse import MantleSseSearchAccessor

        with ExitStack() as stack:
            for p in _all_three_present():
                stack.enter_context(p)
            accessor = wiring.build_sse_search_accessor(_FakeStoreDB())
        assert isinstance(accessor, MantleSseSearchAccessor)


class TestBuildSseSearchAccessor:
    def test_the_accessor_is_built_with_both_a_narrower_and_a_ranker(self):
        """The other half of the loud failure: `search()` refuses to answer without a
        narrower, so the builder has to supply one on every node it builds an accessor for.
        A `None` here would move the failure from wiring time to query time."""
        from mantle.search.mantle.engine import MantleQueryEngine
        from mantle.search.mantle.sse import TokenNarrower

        with ExitStack() as stack:
            for p in _all_three_present():
                stack.enter_context(p)
            acc = wiring.build_sse_search_accessor(_FakeStoreDB())
        assert isinstance(acc._narrower, TokenNarrower)
        assert isinstance(acc._ranker, MantleQueryEngine)

    def test_the_accessor_holds_no_third_collaborator(self):
        """There were three. The fused accessor served `candidates()` alone, and `candidates()`
        now runs the same narrow-then-order path `search()` does — so the retrieval story is
        the narrower and the ranker, and a node builds exactly those two."""
        with ExitStack() as stack:
            for p in _all_three_present():
                stack.enter_context(p)
            acc = wiring.build_sse_search_accessor(_FakeStoreDB())
        assert not hasattr(acc, "_unified")
        assert not hasattr(wiring, "build_unified_accessor")


# ---------------------------------------------------------------------------
# SSE store backend selection — S3 vs the local file-backed index
# ---------------------------------------------------------------------------


class TestSseStoreSelection:
    """`MANTLE_SSE_STORE` picks the backend; both sides implement the same two Protocols.

    The load-bearing distinction under test is configured vs reachable. An operator who set up
    edge object storage must keep getting object storage — including its failure mode — or a
    transient outage would silently start a second, local, divergent index that nothing ever
    reconciles.
    """

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MANTLE_SSE_STORE", raising=False)
        monkeypatch.setenv("MANTLE_SSE_DIR", str(tmp_path / "sse"))

    @staticmethod
    def _s3(configured: bool, reachable: bool = True):
        return [
            patch.object(wiring, "edge_object_storage_is_configured", lambda: configured),
            patch.object(
                wiring, "edge_s3_if_reachable",
                lambda _what: (_FakeS3Client(), "bucket") if reachable else (None, None),
            ),
        ]

    def _build(self, *patches, segment="committed"):
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return wiring._build_sse_store(segment)

    def test_standalone_install_gets_the_local_index(self):
        """With no object storage configured, the builder falls back to the local index rather
        than returning None — otherwise `POST /artifacts/recall {candidates}` would answer 503 on every
        standalone install, with no setting that could fix it."""
        assert isinstance(self._build(*self._s3(configured=False)), SqlitePostingStore)

    def test_s3_configured_install_is_unaffected(self):
        """Configured and reachable resolves to the S3 adapter, matching every existing
        deployment's expectation."""
        assert isinstance(self._build(*self._s3(configured=True)), S3PostingStore)

    def test_configured_but_unreachable_refuses_rather_than_going_local(self):
        """Configured but unreachable returns None rather than falling through to the local
        store: a MinIO that is merely down at boot must not leave the process writing an index
        nobody reads and reading an index nobody writes. Returning None here, so the caller
        answers 503, is what avoids that split-brain."""
        assert self._build(*self._s3(configured=True, reachable=False)) is None

    def test_file_is_selectable_even_where_s3_is_configured(self, monkeypatch):
        """`MANTLE_SSE_STORE=file` selects the local index even when S3 is configured, so an
        operator does not have to unset content credentials to get one."""
        monkeypatch.setenv("MANTLE_SSE_STORE", "file")
        assert isinstance(self._build(*self._s3(configured=True)), SqlitePostingStore)

    def test_s3_is_selectable_and_never_falls_back(self, monkeypatch):
        """`MANTLE_SSE_STORE=s3` never falls back to the local index: an operator who pinned
        object storage is stating that a local index is not acceptable, so an unreachable bucket
        must 503."""
        monkeypatch.setenv("MANTLE_SSE_STORE", "s3")
        assert isinstance(self._build(*self._s3(configured=False)), S3PostingStore)
        assert self._build(*self._s3(configured=False, reachable=False)) is None

    def test_unknown_value_refuses_instead_of_guessing(self, monkeypatch):
        """An unrecognised `MANTLE_SSE_STORE` value returns None rather than guessing a backend:
        `MANTLE_SSE_STORE=s2` silently picking the local store would index into a directory the
        operator never intended and never looks at."""
        monkeypatch.setenv("MANTLE_SSE_STORE", "s2")
        assert self._build(*self._s3(configured=True)) is None

    def test_segments_get_separate_local_trees(self):
        """Per-state index segments use separate directories, so a draft's postings cannot
        answer a committed-only query."""
        committed = self._build(*self._s3(configured=False), segment="committed")
        draft = self._build(*self._s3(configured=False), segment="draft")
        committed.put_posting("owner-A", "a" * 64, b"committed")
        draft.put_posting("owner-A", "a" * 64, b"draft")
        assert committed.get_posting("owner-A", "a" * 64) == b"committed"
        assert draft.get_posting("owner-A", "a" * 64) == b"draft"

    def test_local_root_follows_MANTLE_SSE_DIR(self, monkeypatch, tmp_path):
        """The local index root follows `MANTLE_SSE_DIR`, so it lands on the mounted volume the
        operator pointed it to rather than a container's ephemeral layer."""
        monkeypatch.setenv("MANTLE_SSE_DIR", str(tmp_path / "elsewhere"))
        assert wiring.local_sse_root() == str(tmp_path / "elsewhere")

    def test_local_root_default_is_absolute(self, monkeypatch):
        """The default local index root is absolute, not cwd-relative: starting mantle from a
        different directory must not open a different, empty index — the same property
        `MANTLE_LATTICE_PATH` documents next door."""
        monkeypatch.delenv("MANTLE_SSE_DIR", raising=False)
        assert os.path.isabs(wiring.local_sse_root())

    def test_availability_probe_tracks_the_selected_backend(self):
        """The reindex gate asks about the selected backend rather than always S3, so an install
        with a local index gets a full pass instead of leaving search permanently empty."""
        with ExitStack() as stack:
            for p in self._s3(configured=False):
                stack.enter_context(p)
            assert wiring.sse_index_storage_available() is True
        with ExitStack() as stack:
            for p in self._s3(configured=True, reachable=False):
                stack.enter_context(p)
            assert wiring.sse_index_storage_available() is False


# ---------------------------------------------------------------------------
# Cell store backend selection — S3 vs the local file-backed cell tree
# ---------------------------------------------------------------------------


class TestCellStoreSelection:
    """`MANTLE_CELL_STORE` picks the vector arm's backend, on the SSE arm's vocabulary.

    Same three answers, same configured-vs-reachable distinction, and for the same reason:
    an unconfigured S3 has to be a complete configuration for BOTH arms, or the air-gap
    claim holds for lexical recall and quietly does not hold for semantic recall.
    """

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MANTLE_CELL_STORE", raising=False)
        monkeypatch.setenv("MANTLE_CELL_DIR", str(tmp_path / "cells"))

    @staticmethod
    def _s3(configured: bool, reachable: bool = True):
        return [
            patch.object(wiring, "edge_object_storage_is_configured", lambda: configured),
            patch.object(
                wiring, "edge_s3_if_reachable",
                lambda _what: (_FakeS3Client(), "bucket") if reachable else (None, None),
            ),
        ]

    def _build(self, *patches, segment="committed"):
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return wiring._build_cell_store(segment)

    def test_standalone_install_gets_the_local_cell_tree(self):
        """With no object storage configured the vector arm gets a store instead of None —
        which is the whole difference between a semantic arm that exists on a standalone
        install and one that is unreachable in every default configuration."""
        assert isinstance(self._build(*self._s3(configured=False)), FileCellStore)

    def test_s3_configured_install_is_unaffected(self):
        assert isinstance(self._build(*self._s3(configured=True)), S3CellStore)

    def test_configured_but_unreachable_refuses_rather_than_going_local(self):
        """Same split-brain refusal as the SSE arm: a bucket that is merely down at boot must
        not leave the process writing cells nobody reads."""
        assert self._build(*self._s3(configured=True, reachable=False)) is None

    def test_file_is_selectable_even_where_s3_is_configured(self, monkeypatch):
        monkeypatch.setenv("MANTLE_CELL_STORE", "file")
        assert isinstance(self._build(*self._s3(configured=True)), FileCellStore)

    def test_s3_is_selectable_and_never_falls_back(self, monkeypatch):
        monkeypatch.setenv("MANTLE_CELL_STORE", "s3")
        assert isinstance(self._build(*self._s3(configured=False)), S3CellStore)
        assert self._build(*self._s3(configured=False, reachable=False)) is None

    def test_unknown_value_refuses_instead_of_guessing(self, monkeypatch):
        monkeypatch.setenv("MANTLE_CELL_STORE", "s2")
        assert self._build(*self._s3(configured=True)) is None

    def test_segments_get_separate_local_trees(self):
        """A draft's cells cannot answer a committed-only query, on disk as in S3."""
        committed = self._build(*self._s3(configured=False), segment="committed")
        draft = self._build(*self._s3(configured=False), segment="draft")
        committed.put("owner-A", "col-1", b"committed", "anchor-0")
        draft.put("owner-A", "col-1", b"draft", "anchor-0")
        assert committed.get("owner-A", "col-1", "anchor-0") == b"committed"
        assert draft.get("owner-A", "col-1", "anchor-0") == b"draft"

    def test_local_root_follows_MANTLE_CELL_DIR(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MANTLE_CELL_DIR", str(tmp_path / "elsewhere"))
        assert wiring.local_cell_root() == str(tmp_path / "elsewhere")

    def test_local_root_default_is_absolute(self, monkeypatch):
        monkeypatch.delenv("MANTLE_CELL_DIR", raising=False)
        assert os.path.isabs(wiring.local_cell_root())

    def test_availability_probe_tracks_the_selected_backend(self):
        with ExitStack() as stack:
            for p in self._s3(configured=False):
                stack.enter_context(p)
            assert wiring.cell_storage_available() is True
        with ExitStack() as stack:
            for p in self._s3(configured=True, reachable=False):
                stack.enter_context(p)
            assert wiring.cell_storage_available() is False

    def test_the_vector_arm_is_wired_on_a_node_with_no_object_storage(self):
        """The point of all of the above, stated end to end: an accessor built on a node with
        nothing configured comes back holding a RANKER, not only a narrower."""
        from mantle.search.mantle.engine import MantleQueryEngine

        with ExitStack() as stack:
            for p in self._s3(configured=False):
                stack.enter_context(p)
            stack.enter_context(patch.object(wiring, "_build_oracle", lambda: _FakeOracle()))
            stack.enter_context(patch.dict(
                os.environ, {"MANTLE_SSE_DIR": wiring.local_cell_root() + "-sse"},
            ))
            acc = wiring.build_sse_search_accessor(_FakeStoreDB())
        assert acc is not None
        assert isinstance(acc._ranker, MantleQueryEngine), (
            "the vector arm is still dark with no S3")


class TestEdgeObjectStorageIsConfigured:
    """The predicate that keeps configured separate from reachable."""

    @pytest.mark.parametrize(
        "attr", ["_EDGE_ACCESS_KEY_ID", "_EDGE_SECRET_ACCESS_KEY", "_EDGE_ENDPOINT_URL_INTERNAL"],
    )
    def test_any_edge_credential_or_endpoint_counts_as_configured(self, monkeypatch, attr):
        """Configuration is read from explicit credentials and endpoint, not the boto3 client's
        existence: that client is built unconditionally at import with no credentials at all, so
        testing for it alone would answer "yes" on every standalone install."""
        from mantle.services import content_service
        for name in ("_EDGE_ACCESS_KEY_ID", "_EDGE_SECRET_ACCESS_KEY",
                     "_EDGE_ENDPOINT_URL_INTERNAL"):
            monkeypatch.setattr(content_service, name, None, raising=False)
        assert wiring.edge_object_storage_is_configured() is False
        monkeypatch.setattr(content_service, attr, "set-by-the-operator", raising=False)
        assert wiring.edge_object_storage_is_configured() is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeOracle:
    """Minimal stand-in for OracleService — wiring only checks for non-None."""
